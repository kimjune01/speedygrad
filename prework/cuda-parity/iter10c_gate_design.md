# iter 10c: gate design for `_apply_map_to_tensors` JIT-replay fast path

**Status:** draft for adversarial review (gemini). No code changes yet.

## Goal

Skip the per-decode-token O(|all_tensors|) `topovisit` walk inside `tinygrad.tensor._apply_map_to_tensors` (`tensor.py:23`) on the JIT-input realize hot path, without breaking tensor-identity semantics in the general case.

Target impact: ~1.5–2 ms / decode token (Llama 3.2 1B fp16, ~150+ live tensors). Brings speedygrad fp16 1B decode from ~8.04 ms to ~6 ms = ~1.35x of llama.cpp (currently 1.80x).

## Trace of the hot path

Per-decode-token call sequence in the `bench/speedygrad_llama32_1b.py` loop:

```python
next_tok = model(Tensor([[last_tok]], device="CUDA"), start_pos, ...).item()
```

1. `Tensor([[last_tok]], device="CUDA")` — fresh construction:
   - `__init__` (`tensor.py:94`) routes to `_frompy(data, dtypes.default_int, "CUDA")`
   - `_frompy` (`tensor.py:53`) builds `UOp.empty(shape, dtype, "PYTHON")`, `.allocate(memoryview(...))` — fake-realized BUFFER on PYTHON device with a fresh `Ops.UNIQUE` source
   - `__init__` line 142: `self.uop = data.copy_to_device("CUDA")` — wraps in `Ops.COPY`
   - `all_tensors[weakref.ref(self)] = None`
   - **Result:** `tok_tensor.uop` shape is `COPY(BUFFER(UNIQUE, DEVICE), CUDA_DEVICE)` (or thin RESHAPE chain). Hashconsing: BUFFER's `Ops.UNIQUE` source is freshly allocated, so the entire chain is `is`-unique. **No other Tensor in `all_tensors` has any UOp in this DAG as a source.**

2. `model.__call__` is `TinyJit.__call__` (`jit.py:256`) → `_prepare_jit_inputs(args, kwargs)` (`jit.py:215`)

3. Inside `_prepare_jit_inputs`:
   - `tensors = [tok_tensor]` (start_pos and floats filtered out)
   - `tok_tensor.uop.is_realized` is False (`COPY` is not `BUFFER`)
   - `Tensor.realize(tok_tensor)` (line 222)

4. `Tensor.realize` (`tensor.py:242`) → `Tensor.linear_with_vars(tok_tensor)` (`tensor.py:229`):
   - `big_sink, becomes_map = transform_to_call(UOp.sink(tok_tensor.uop))`
   - `becomes_map` keys are exactly the per-iteration UOps from `tok_tensor.uop` that need replacement (the `COPY` and the source `BUFFER`); values are the realized BUFFER UOps on CUDA
   - `_apply_map_to_tensors(becomes_map, name="buffers")` ← **the hot call**

5. `_apply_map_to_tensors` (`tensor.py:23`):
   ```python
   in_scope: dict[UOp, bool] = {}
   def visitor(node): return True if node in applied_map else any(in_scope.get(s, False) for s in node.src)
   scope_tensors = [t for tref in list(all_tensors)
                    if (t:=tref()) is not None and t.uop.topovisit(visitor, in_scope)]
   sink = UOp.sink(*[t.uop for t in scope_tensors])
   new_sink = sink.substitute(applied_map, name=f"substitute {name}", walk=walk)
   for t,s,ns in zip(scope_tensors, sink.src, new_sink.src):
     if s is ns: continue
     t.uop = ns
   ```
   - `~150` live tensors, ~168 `topovisit` calls/decode (cProfile)
   - Of those ~150, only `tok_tensor` has any UOp DAG node in `applied_map`
   - The walk's purpose is to *find* `tok_tensor` and update its `.uop`
   - All other ~149 tensors return False from `visitor`. Their topovisits are wasted work.

## Structural claim

In the JIT-replay decode hot path, **every key in `applied_map` appears in exactly one live Tensor's UOp DAG: the freshly-constructed input Tensor passed to `Tensor.realize` from `_prepare_jit_inputs`.**

Why this holds:
- The keys of `applied_map` (= `becomes_map` from `transform_to_call`) are subsets of `(self,)+lst`'s `.uop` DAGs (`tensor.py:231`).
- `(self,)+lst` for the JIT-input realize call is `unrealized_tensors` from `_prepare_jit_inputs:222`.
- Each `unrealized_tensor` was constructed in this Python frame via `Tensor([[tok]], device=...)`.
- Construction allocates a fresh `Ops.UNIQUE`-rooted BUFFER. UOp hashconsing makes the entire DAG `is`-unique.
- No other Tensor's `.uop` DAG, including model weights and KV-cache slots constructed before this iteration, contains these UOps.

Therefore the slow walk's general-case behavior reduces to: "update `unrealized_tensor.uop` for each input tensor; do nothing for everyone else."

## Counter-examples — when the structural claim breaks

The fast path **must not apply** when:

1. **The user passes a non-leaf Tensor as a JIT input.** Example: `model(some_persistent_tensor + offset_tensor, ...)`. The expression-result Tensor's UOp DAG contains references to `some_persistent_tensor.uop`, which is in `all_tensors`. After realize, `some_persistent_tensor` may also need its `.uop` updated if `transform_to_call` materialized it. Skipping the walk here would silently break tensor identity for `some_persistent_tensor`.
   - Note: in current speedygrad benches `tok_tensor` is constructed fresh from a Python list, so this case doesn't trigger. But the gate has to be defensive — a user calling a TinyJit-wrapped function with a precomputed Tensor input would hit this.

2. **`_apply_map_to_tensors` called from `callify` (`tensor.py:226`) or `assign` Embed View Assign (`tensor.py:279`).** Different call paths, applied_map keys are not from a fresh single-tensor `Tensor.realize` call. Fast path must not apply.

3. **`walk=True` callers.** The `walk` flag changes substitution semantics. Fast path is for `walk=False` only.

4. **`Tensor.realize(*lst)` called outside `_prepare_jit_inputs`** with multiple tensors that share UOp ancestors with each other or with held tensors. Same as case 1.

## Gate condition (proposal)

Two layers, both must hold:

**Layer A — caller-asserted context flag.** Add a contextvar:
```python
_jit_input_caller_tensors: ContextVar[tuple[Tensor, ...] | None] = ContextVar("_jit_input_caller_tensors", default=None)
```
Set it in a rebound `_prepare_jit_inputs` (or the realize wrapper) around the `Tensor.realize(*unrealized_tensors)` call. Default None ⇒ all other call sites of `_apply_map_to_tensors` use the original walk.

**Layer B — UOp shape check on each caller tensor.** Even when the contextvar is set, validate that each caller tensor's UOp is a "fresh leaf" the structural argument applies to:

```python
def _is_fresh_jit_input_uop(uop: UOp) -> bool:
  # Walk the COPY/RESHAPE/CAST chain down to a BUFFER on PYTHON device with an UNIQUE source.
  # This is the shape produced by Tensor([[...]], device=DEVICE) → _frompy → copy_to_device.
  while uop.op in {Ops.COPY, Ops.RESHAPE, Ops.CAST, Ops.EXPAND}:
    uop = uop.src[0]
  if uop.op is not Ops.BUFFER: return False
  if not (len(uop.src) > 0 and uop.src[0].op is Ops.UNIQUE): return False
  if not uop.is_realized: return False  # _frompy's fake-realize must have populated the buffer
  return True
```

The fast path applies iff Layer A's contextvar is set AND every caller tensor in it passes Layer B AND `walk=False`.

## Fast-path implementation sketch

```python
import tinygrad.tensor as _tensor_mod
from tinygrad.uop.ops import UOp, Ops, TracingKey
from tinygrad.helpers import cpu_profile
from contextvars import ContextVar

_jit_input_caller_tensors: ContextVar[tuple | None] = ContextVar("_jit_input_caller_tensors", default=None)

_orig_apply_map_to_tensors = _tensor_mod._apply_map_to_tensors

def _is_fresh_jit_input_uop(u):
  while u.op in {Ops.COPY, Ops.RESHAPE, Ops.CAST, Ops.EXPAND}:
    u = u.src[0]
  return (u.op is Ops.BUFFER and len(u.src) > 0 and u.src[0].op is Ops.UNIQUE
          and u.is_realized)

def _apply_map_to_tensors_fast(applied_map, name, walk=False):
  callers = _jit_input_caller_tensors.get()
  if walk or callers is None or not all(_is_fresh_jit_input_uop(t.uop) for t in callers):
    return _orig_apply_map_to_tensors(applied_map, name, walk)
  with cpu_profile(TracingKey(name + " (fast)"), "TINY"):
    sink = UOp.sink(*[t.uop for t in callers])
    new_sink = sink.substitute(applied_map, name=f"substitute {name}", walk=False)
    for t, s, ns in zip(callers, sink.src, new_sink.src):
      if s is not ns: t.uop = ns

_tensor_mod._apply_map_to_tensors = _apply_map_to_tensors_fast

# Wrap _prepare_jit_inputs to set the contextvar around its realize call.
import tinygrad.engine.jit as _jit_mod
_orig_prepare = _jit_mod._prepare_jit_inputs

def _prepare_jit_inputs_fast(args, kwargs):
  # Reconstruct what _prepare_jit_inputs does up to the realize call so we can
  # set the contextvar around that one call. The cleaner alternative is to
  # rebind Tensor.realize and detect the caller via the call stack, but that's
  # more fragile.
  ...
```

(The `_prepare_jit_inputs` rebind needs care — it does extraction, var collection, and input_buf_uop computation around the realize call. Cleanest is probably to rebind a thinner inner function or to add a sentinel parameter to `Tensor.realize` that the JIT path sets. Will design the wrapper after gemini reviews the gate.)

## Questions for gemini

1. **Tensor identity.** Is the structural claim — "every key in `applied_map` appears in exactly one live Tensor's UOp DAG, the freshly-constructed JIT input" — actually robust? Specifically: are there any code paths in the JIT-replay decode loop where a side-effect of `Tensor([[tok]], device=...)` construction or `_frompy`'s `.allocate` could leak the input UOp into another live Tensor's DAG before `_apply_map_to_tensors` runs?

2. **Layer B coverage.** Does the `_is_fresh_jit_input_uop` shape check correctly identify the construction pattern from `Tensor([[tok]], device="CUDA")` (`_frompy → copy_to_device`)? Are there other Tensor constructors a user might use to create JIT inputs whose UOp chain doesn't match the COPY/RESHAPE/CAST/EXPAND-over-BUFFER-with-UNIQUE shape but still satisfies the isolation property?

3. **Layer A correctness.** A contextvar set only in a rebound `_prepare_jit_inputs` and read inside `_apply_map_to_tensors` skips the walk for the JIT-input realize call only. Is there any path inside `_prepare_jit_inputs → Tensor.realize → linear_with_vars → _apply_map_to_tensors` that re-enters `_apply_map_to_tensors` with a different `applied_map` (one that should NOT take the fast path)? E.g., does `transform_to_call` itself ever call `_apply_map_to_tensors`?

4. **False negatives.** Cases where the fast path *could* safely apply but doesn't (e.g., model() called with a Tensor that's not a fresh leaf but is still uniquely referenced) are fine — they degrade gracefully to the slow path. Are there any false-positive cases (gate says fast-path-safe when it isn't) you can construct?

5. **Iter 10 methodology guardrail.** The 1.5–2 ms / decode token impact estimate comes from cProfile cumtime divided by ~3x for instrumentation overhead. If the actual raw cost is, say, 0.5 ms / token (deflation factor was wrong), the fix is still correct but the headline impact shrinks. Independent cross-check: deflate via wall-clock comparison? Iter 9 had wall-clock 8.04 ms / token, identified subtotal 652 us, residue 1880 us ⇒ impact range 0.65–1.88 ms is plausible.
