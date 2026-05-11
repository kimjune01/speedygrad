"""Import and apply Cython-compiled rewrites and runtime fast path."""
import os
# default-on single-kernel graph capture: cuGraphLaunch beats cuCtxSetCurrent+cuLaunchKernel
# by ~17us on Windows. Set before tinygrad imports — getenv() is functools.cached.
os.environ.setdefault("GRAPH_ONE_KERNEL", "1")

from tinygrad.uop.ops import RewriteContext, PatternMatcher, UOp
try:
    from cy_rewrite import cy_unified_rewrite, cy_rewrite, cy_toposort, cy_dfs_match
    RewriteContext.unified_rewrite = cy_unified_rewrite
    PatternMatcher.rewrite = cy_rewrite
    UOp.toposort = cy_toposort
    UOp.dfs_match = cy_dfs_match
except ImportError:
    pass

# Runtime fast path: rebind run_linear at every import site so call sites that did
# `from tinygrad.engine.realize import run_linear` pick up the Cython version.
try:
    from cy_runtime import cy_run_linear
    import tinygrad.engine.realize as _realize_mod
    import tinygrad.engine.jit as _jit_mod
    import tinygrad.tensor as _tensor_mod
    _realize_mod.run_linear = cy_run_linear
    _jit_mod.run_linear = cy_run_linear
    _tensor_mod.run_linear = cy_run_linear
except ImportError:
    pass

# CUDAGraph.__call__ fast path: skip cu_time_execution lambda wrapper for wait=False.
# Saves 2 Python frames (lambda + cu_time_execution) per kernel call on the JIT-replay
# hot path. Same observable behavior; cu_time_execution(cb, enable=False) just calls cb().
try:
    import ctypes as _ct
    import tinygrad.runtime.autogen.cuda as _cuda
    from tinygrad.runtime.ops_cuda import check as _check, cu_time_execution as _cu_time
    from tinygrad.runtime.graph.cuda import CUDAGraph as _CUDAGraph
    from tinygrad.device import MultiBuffer as _MB

    def _cuda_graph_call_fast(self, input_uops, var_vals, wait=False):
        for j in self.updatable:
            (_, params, c_args, is_copy), dev_idx = self.nodes[j], self.calls[j][0]
            for pos, iidx in self.uop_replace[j]:
                buf = b.bufs[dev_idx] if isinstance(b:=input_uops[iidx].buffer, _MB) else b
                if not is_copy: setattr(c_args, f'f{pos}', buf._buf)
                else: setattr(params, 'srcDevice' if pos == 1 else 'dstDevice', buf._buf)
        for j, i, v in self.updated_vars(var_vals): setattr(self.nodes[j][2], f'v{i}', v)
        for j, global_dims, local_dims in self.updated_launch_dims(var_vals):
            node = self.nodes[j][1]
            node.blockDimX, node.blockDimY, node.blockDimZ, node.gridDimX, node.gridDimY, node.gridDimZ = *local_dims, *global_dims
        for j in self.updatable:
            node, c_node_params, c_args, is_copy = self.nodes[j]
            if not is_copy: _check(_cuda.cuGraphExecKernelNodeSetParams(self.instance, node, _ct.byref(c_node_params)))
            else: _check(_cuda.cuGraphExecMemcpyNodeSetParams(self.instance, node, _ct.byref(c_node_params), c_args))
        if wait:
            return _cu_time(lambda: _check(_cuda.cuGraphLaunch(self.instance, None)), enable=True)
        _check(_cuda.cuGraphLaunch(self.instance, None))
        return None

    _CUDAGraph.__call__ = _cuda_graph_call_fast
except ImportError:
    pass

# iter 10c-cont v3: memoize-walk for _apply_map_to_tensors.
# Replace the per-tensor `topovisit` walk with a cached UOp-DAG-id-set lookup.
# UOps are hashconsed (via UOpMetaClass.ucache weak refs) so when no Tensor
# references a UOp, its __del__ fires and frees memory.
#
# Cache design (iter 10c-cont v4: leak-free):
#   key:   id(uop)  (integer, doesn't keep uop alive)
#   value: frozenset of id(u) for u in uop.toposort()  (integers, doesn't
#          keep DAG members alive — they're held by the live tensor graph)
# When the keying UOp dies, weakref.finalize pops the cache entry. Cache
# size stays bounded by the live UOp footprint, not the historical one.
#
# Per-call: convert applied_map.keys() to {id(k) for k in applied_keys},
# then `applied_id_set.isdisjoint(cached_id_set)` is the gate.
try:
    import weakref as _weakref_mw
    import tinygrad.tensor as _tensor_mod_mw
    from tinygrad.uop.ops import UOp as _UOp_mw, TracingKey as _TracingKey_mw
    from tinygrad.helpers import cpu_profile as _cpu_profile_mw

    _orig_apply_mw = _tensor_mod_mw._apply_map_to_tensors
    _uop_dag_id_cache_mw: dict = {}  # id(uop) -> frozenset[int]

    def _uop_dag_id_set_mw(u):
        uid = id(u)
        cached = _uop_dag_id_cache_mw.get(uid)
        if cached is not None: return cached
        seen_ids: set = set()
        stack = [u]
        while stack:
            n = stack.pop()
            nid = id(n)
            if nid in seen_ids: continue
            seen_ids.add(nid)
            stack.extend(n.src)
        result = frozenset(seen_ids)
        _uop_dag_id_cache_mw[uid] = result
        # When u dies, evict its cache entry. The finalizer captures uid
        # by closure (not u itself, so doesn't keep u alive).
        _weakref_mw.finalize(u, _uop_dag_id_cache_mw.pop, uid, None)
        return result

    def _apply_map_memoized_mw(applied_map, name, walk=False):
        if walk:
            return _orig_apply_mw(applied_map, name, walk)
        with _cpu_profile_mw(_TracingKey_mw(name + " (memoized)"), "TINY"):
            applied_id_set = {id(k) for k in applied_map}
            scope_tensors = []
            for tref in list(_tensor_mod_mw.all_tensors):
                t = tref()
                if t is None: continue
                if not applied_id_set.isdisjoint(_uop_dag_id_set_mw(t.uop)):
                    scope_tensors.append(t)
            sink = _UOp_mw.sink(*[t.uop for t in scope_tensors])
            new_sink = sink.substitute(applied_map, name=f"substitute {name}", walk=walk)
            for t, s, ns in zip(scope_tensors, sink.src, new_sink.src):
                if s is ns: continue
                t.uop = ns

    _tensor_mod_mw._apply_map_to_tensors = _apply_map_memoized_mw
except ImportError:
    pass
