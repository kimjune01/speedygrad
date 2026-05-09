"""Regression tests for bf16 transcendental precision (issue #11756).

The bug: exp/log/cos decompose through constants like 1/log(2). When those
constants are stored in bf16, the 7-bit mantissa truncates them and the error
compounds. The fix upcasts to float32 for the constant multiply, then casts
back. These tests verify tinygrad bf16 results match torch bf16 reference
values (which compute in higher precision internally).
"""
import unittest, math, struct

from tinygrad import Tensor

def nearest_bf16(val: float) -> float:
  """Round a float64 value to the nearest bfloat16 representable value."""
  f32_bytes = struct.pack('f', val)
  f32_int = struct.unpack('I', f32_bytes)[0]
  bf16_trunc = f32_int >> 16
  bf16_round = bf16_trunc + ((f32_int >> 15) & 1)
  return struct.unpack('f', struct.pack('I', bf16_round << 16))[0]

class TestBF16Transcendental(unittest.TestCase):
  """bf16 exp/log/cos must match the nearest bf16 to the true f64 result,
  i.e. the same value torch produces."""

  # ---- exp ----
  def test_exp_12(self):
    """Issue #11756 headline: exp(12) in bf16."""
    result = Tensor([12.0], dtype="bfloat16").exp().tolist()[0]
    # torch bf16 gives 162816.0; exact is 162754.79...
    self.assertEqual(result, 162816.0)

  def test_exp_bf16_matches_f32_cast(self):
    """exp(x) in bf16 should equal cast(exp(cast(x, f32)), bf16)."""
    xs = [0.5, 1.0, 2.0, 5.0, 12.0, -1.0, -5.0]
    for x in xs:
      result = Tensor([x], dtype="bfloat16").exp().tolist()[0]
      ref = nearest_bf16(math.exp(x))
      self.assertEqual(result, ref, msg=f"exp({x}): got {result}, expected {ref}")

  # ---- log ----
  def test_log_12(self):
    """Issue #11756: log(12) in bf16."""
    result = Tensor([12.0], dtype="bfloat16").log().tolist()[0]
    # torch bf16 gives 2.484375; tinygrad gave 2.46875 before fix
    self.assertEqual(result, 2.484375)

  def test_log_bf16_matches_f32_cast(self):
    """log(x) in bf16 should equal cast(log(cast(x, f32)), bf16)."""
    xs = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 12.0, 100.0, 1000.0]
    for x in xs:
      result = Tensor([x], dtype="bfloat16").log().tolist()[0]
      ref = nearest_bf16(math.log(x))
      self.assertEqual(result, ref, msg=f"log({x}): got {result}, expected {ref}")

  # ---- cos ----
  def test_cos_12(self):
    """Issue #11756: cos(12) in bf16."""
    result = Tensor([12.0], dtype="bfloat16").cos().tolist()[0]
    # torch bf16 gives 0.84375
    self.assertEqual(result, 0.84375)

  def test_cos_bf16_matches_f32_cast(self):
    """cos(x) in bf16 should equal cast(cos(cast(x, f32)), bf16)."""
    xs = [0.0, 0.5, 1.0, math.pi/4, math.pi, 12.0]
    for x in xs:
      result = Tensor([x], dtype="bfloat16").cos().tolist()[0]
      ref = nearest_bf16(math.cos(x))
      self.assertEqual(result, ref, msg=f"cos({x}): got {result}, expected {ref}")

  # ---- f32 regression ----
  def test_f32_not_regressed(self):
    """Ensure float32 results are unchanged."""
    self.assertAlmostEqual(Tensor([12.0], dtype="float32").exp().tolist()[0], 162754.71875, places=1)
    self.assertAlmostEqual(Tensor([12.0], dtype="float32").log().tolist()[0], 2.4849066, places=4)
    self.assertAlmostEqual(Tensor([12.0], dtype="float32").cos().tolist()[0], 0.8438540, places=4)

  # ---- log10 ----
  def test_log10_bf16(self):
    """log10 has the same constant-precision pattern as log."""
    xs = [1.0, 10.0, 100.0, 1000.0]
    for x in xs:
      result = Tensor([x], dtype="bfloat16").log10().tolist()[0]
      ref = nearest_bf16(math.log10(x))
      self.assertAlmostEqual(result, ref, places=2, msg=f"log10({x}): got {result}, expected {ref}")

if __name__ == "__main__":
  unittest.main()
