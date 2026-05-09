"""Regression test for tinygrad issue #13409: ScatterND infinite loop in graph_rewrite.

The original ScatterND implementation used __setitem__ with tensor indices in a loop,
creating one-hot masks that exploded the rewrite graph. This test verifies the fix
uses scatter_reduce instead, keeping the graph manageable.
"""
import unittest
import numpy as np
from tinygrad import Tensor
from tinygrad.nn.onnx import get_onnx_ops

np.random.seed(13409)
_onnx_ops = get_onnx_ops()
ScatterND = _onnx_ops["ScatterND"]


def numpy_scatter_nd(x, indices, updates, reduction="none"):
  """Reference implementation: iterate and write element-by-element."""
  out = x.copy()
  for idx in np.ndindex(indices.shape[:-1]):
    target = tuple(indices[idx])
    if reduction == "none": out[target] = updates[idx]
    elif reduction == "add": out[target] += updates[idx]
    elif reduction == "mul": out[target] *= updates[idx]
    elif reduction == "max": out[target] = max(out[target], updates[idx])
    elif reduction == "min": out[target] = min(out[target], updates[idx])
  return out


class TestScatterND(unittest.TestCase):
  def _compare(self, x_np, indices_np, updates_np, reduction="none", atol=1e-5):
    ref = numpy_scatter_nd(x_np, indices_np, updates_np, reduction)
    result = ScatterND(Tensor(x_np.copy()), Tensor(indices_np), Tensor(updates_np), reduction=reduction).numpy()
    np.testing.assert_allclose(result, ref, atol=atol, err_msg=f"reduction={reduction}")

  # --- basic correctness (small, fast) ---

  def test_full_index_no_duplicates(self):
    """Each index targets a unique element."""
    x = np.zeros((4, 5, 6), dtype=np.float32)
    indices = np.array([[0, 1, 2], [3, 4, 5], [1, 0, 3]], dtype=np.int64).reshape(3, 1, 3)
    updates = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)
    self._compare(x, indices, updates)

  def test_full_index_with_duplicates(self):
    """Two updates target the same element; last write wins."""
    x = np.zeros((3, 4), dtype=np.float32)
    indices = np.array([[1, 2], [1, 2], [0, 0]], dtype=np.int64).reshape(3, 1, 2)
    updates = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)
    self._compare(x, indices, updates)

  def test_partial_index(self):
    """K < ndim: each index specifies a row, update is a slice."""
    x = np.ones((4, 5), dtype=np.float32)
    indices = np.array([[1], [3]], dtype=np.int64)   # shape (2, 1)
    updates = np.array([[10, 20, 30, 40, 50], [60, 70, 80, 90, 100]], dtype=np.float32)  # shape (2, 5)
    self._compare(x, indices, updates)

  def test_reduction_add(self):
    x = np.ones((3, 3), dtype=np.float32)
    indices = np.array([[0, 0], [0, 0], [1, 1]], dtype=np.int64).reshape(3, 1, 2)
    updates = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)
    self._compare(x, indices, updates, reduction="add")

  def test_reduction_mul(self):
    x = np.full((3, 3), 2.0, dtype=np.float32)
    indices = np.array([[0, 0], [0, 0], [1, 1]], dtype=np.int64).reshape(3, 1, 2)
    updates = np.array([[3.0], [4.0], [5.0]], dtype=np.float32)
    self._compare(x, indices, updates, reduction="mul")

  def test_reduction_max(self):
    x = np.zeros((3, 3), dtype=np.float32)
    indices = np.array([[0, 0], [0, 0], [1, 1]], dtype=np.int64).reshape(3, 1, 2)
    updates = np.array([[10.0], [20.0], [5.0]], dtype=np.float32)
    self._compare(x, indices, updates, reduction="max")

  def test_reduction_min(self):
    x = np.full((3, 3), 100.0, dtype=np.float32)
    indices = np.array([[0, 0], [0, 0], [1, 1]], dtype=np.int64).reshape(3, 1, 2)
    updates = np.array([[10.0], [20.0], [5.0]], dtype=np.float32)
    self._compare(x, indices, updates, reduction="min")

  # --- the actual regression: large tensor indices that triggered #13409 ---

  def test_large_scatternd_does_not_blow_up(self):
    """Issue #13409 repro dimensions. Must complete in <30s, not hit REWRITE_STACK_LIMIT."""
    B, T, S, C, N = 1, 151, 15, 64, 32
    x = np.zeros((B, T, S, C), dtype=np.float32)
    b = np.zeros((B, T, S, N), dtype=np.int64)
    t = np.broadcast_to(np.arange(T, dtype=np.int64)[None, :, None, None], (B, T, S, N))
    s = np.broadcast_to(np.arange(S, dtype=np.int64)[None, None, :, None], (B, T, S, N))
    c = np.random.randint(0, C, size=(B, T, S, N), dtype=np.int64)
    indices = np.stack([b, t, s, c], axis=-1)
    updates = np.random.randn(B, T, S, N).astype(np.float32)
    # should not raise RuntimeError("infinite loop in graph_rewrite")
    result = ScatterND(Tensor(x), Tensor(indices), Tensor(updates)).realize()
    self.assertEqual(result.shape, (B, T, S, C))

  def test_medium_scatternd_correctness(self):
    """Medium-sized case with duplicates: verify correctness against numpy."""
    B, T, S, C, N = 1, 10, 5, 8, 4
    x = np.random.randn(B, T, S, C).astype(np.float32)
    b = np.zeros((B, T, S, N), dtype=np.int64)
    t = np.broadcast_to(np.arange(T, dtype=np.int64)[None, :, None, None], (B, T, S, N))
    s = np.broadcast_to(np.arange(S, dtype=np.int64)[None, None, :, None], (B, T, S, N))
    c = np.random.randint(0, C, size=(B, T, S, N), dtype=np.int64)
    indices = np.stack([b, t, s, c], axis=-1)
    updates = np.random.randn(B, T, S, N).astype(np.float32)
    self._compare(x, indices, updates)


if __name__ == "__main__":
  unittest.main()
