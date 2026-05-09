"""Import and apply the Cython-compiled unified_rewrite."""
from tinygrad.uop.ops import RewriteContext
try:
    from cy_rewrite import cy_unified_rewrite
    RewriteContext.unified_rewrite = cy_unified_rewrite
except ImportError:
    pass
