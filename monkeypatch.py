"""Import and apply Cython-compiled rewrites."""
from tinygrad.uop.ops import RewriteContext, PatternMatcher
try:
    from cy_rewrite import cy_unified_rewrite, cy_rewrite
    RewriteContext.unified_rewrite = cy_unified_rewrite
    PatternMatcher.rewrite = cy_rewrite
except ImportError:
    pass
