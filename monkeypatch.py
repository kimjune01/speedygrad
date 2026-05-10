"""Import and apply Cython-compiled rewrites."""
from tinygrad.uop.ops import RewriteContext, PatternMatcher, UOp
try:
    from cy_rewrite import cy_unified_rewrite, cy_rewrite, cy_toposort, cy_dfs_match
    RewriteContext.unified_rewrite = cy_unified_rewrite
    PatternMatcher.rewrite = cy_rewrite
    UOp.toposort = cy_toposort
    UOp.dfs_match = cy_dfs_match
except ImportError:
    pass
