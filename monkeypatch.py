"""Import and apply Cython-compiled rewrites and runtime fast path."""
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
