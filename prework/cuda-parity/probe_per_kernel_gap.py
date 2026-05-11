"""Final v5 analysis: kernel-by-kernel attribution of the remaining 1500us GPU gap.

Maps each speedygrad kernel to its semantic role (matmul, attention, RoPE,
RMSNorm, softmax, embedding) and to its closest llama.cpp equivalent.
Computes:
  - per-forward time for each role (median × calls/forward)
  - speedygrad - llama.cpp gap per role
  - which roles to attack and what each is worth

This is a STATIC analysis using existing CSV data, no new bench runs.
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).parent

def read_csv(path):
    """Read a UTF-16 LE CSV (PowerShell out-file format)."""
    rows = []
    with open(path, encoding='utf-16') as f:
        # find header line (starts with "Time")
        for line in f:
            if line.lstrip().startswith('Time'):
                header = [c.strip() for c in line.split(',')]
                break
        for line in f:
            line = line.strip()
            if not line: continue
            cols = [c.strip() for c in line.split(',', maxsplit=8)]  # name has commas inside
            if len(cols) < 9: continue
            rows.append(dict(zip(header, cols)))
    return rows

# Load post-fix speedygrad kernel summary (UTF-8, no BOM)
sg_rows = []
with open(ROOT / 'sg3_kern_node.csv', encoding='utf-8') as f:
    for line in f:
        if line.lstrip().startswith('Time'):
            header = [c.strip() for c in line.split(',')]
            break
    for line in f:
        line = line.strip()
        if not line: continue
        cols = [c.strip() for c in line.split(',', maxsplit=8)]
        if len(cols) < 9: continue
        sg_rows.append(dict(zip(header, cols)))

# Load llama.cpp kernel summary (UTF-16, from earlier nsys export)
lc_rows = read_csv(ROOT / 'lc_kern_node.csv')

print(f"speedygrad kernels: {len(sg_rows)}, llama.cpp kernels: {len(lc_rows)}")

# Per-decode-token estimates use median × (calls / total_forwards). For the
# speedygrad trace, total_forwards = 91 (36 prefill + 5 burn + 50 decode).
# For llama.cpp, total_forwards inferred from rms_norm count = 6633 / 33 = ~201.
SG_FWDS = 91
LC_FWDS = 201

# Map speedygrad kernel names to their semantic role
def categorize_sg(name):
    if name.startswith('r_512_16_512_512_4_4'): return ('matmul:FFN_w1', 16)
    if name.startswith('r_1024_16_2_512'): return ('matmul:FFN_w2', 16)
    if name.startswith('r_32064_16_4_128'): return ('matmul:output_proj', 1)
    if name.startswith('r_8_32_2_16_128'): return ('matmul:Q_proj_fused_rmsnorm', 16)
    if name.startswith('r_4_32_2_16_2_2_128'): return ('attn:KV_RoPE_cache_write', 16)
    if name.startswith('r_32_2_16_32_32_4'): return ('matmul:WO_proj', 16)
    if name.startswith('r_256_16_8_32_4'): return ('matmul:post_attn_proj', 16)
    if name.startswith('r_64_8_32_'): return ('attn:A_x_V', 16)
    if name.startswith('r_4_28start_pos') or name.startswith('r_4_(start_pos'): return ('attn:Q_x_K^T', 16)
    if name.startswith('r_16_2_28start_pos') or name.startswith('r_16_2_(start_pos'): return ('attn:softmax_pass', 32)  # max + sum
    if name.startswith('E_28start_pos') or name.startswith('E_(start_pos'): return ('attn:softmax_normalize', 16)
    if name.startswith('r_256_8'): return ('rmsnorm', 49)  # 3 per layer + 1 final
    if name.startswith('E_128_4_16_16_4'): return ('embedding_lookup', 1)
    if name.startswith('r_2048_8_32'): return ('embedding_reduce', 1)
    if name.startswith('E_512_4'): return ('elementwise_misc', 16)
    if name.startswith('r_128_2_501'): return ('argmax_pass1', 1)
    if name.startswith('r_16_4_4'): return ('argmax_pass2', 1)
    if name.startswith('r_256_501'): return ('argmax_idx_pass1', 1)
    if name.startswith('r_64_4'): return ('argmax_idx_pass2', 1)
    if name.startswith('E_16384_32_8_2'): return ('startup_only', 0)
    if name.startswith('E_512_8_32_2_4'): return ('startup_only', 0)
    return ('other', 1)

def categorize_lc(name):
    # llama.cpp kernel names contain template params; match by prefix
    if 'mul_mat_vec_f<__half, float, (int)1, (int)256, (bool)1' in name: return ('matmul:proj_with_bias', 16)
    if 'mul_mat_vec_f<__half, __half, (int)1, (int)256, (bool)0' in name: return ('matmul:plain_half_no_bias', 50)
    if 'mul_mat_vec_f<__half, __half, (int)1, (int)256, (bool)1' in name: return ('matmul:plain_half_with_bias', 31)
    if 'mul_mat_vec_f<__half, float, (int)1, (int)32' in name: return ('matmul:small32', 16)
    if 'mul_mat_vec_f<__half, __half, (int)1, (int)128' in name: return ('matmul:small128', 16)
    if 'rms_norm_f32' in name: return ('rmsnorm', 33)
    if 'soft_max_f32' in name: return ('attn:softmax_fused', 16)
    if 'k_set_rows' in name: return ('attn:KV_cache_write', 16)
    if 'rope_norm<(bool)1, (bool)1, float, __half>' in name: return ('attn:RoPE_kv', 16)
    if 'rope_norm<(bool)1, (bool)1, float, float>' in name: return ('attn:RoPE_q', 16)
    if 'k_get_rows_float' in name: return ('embedding_lookup', 2)
    if 'k_bin_bcast' in name: return ('elementwise_misc', 1)
    return ('other', 1)

# Compute per-forward time per role (median basis)
def aggregate_by_role(rows, total_fwds, categorize_fn):
    roles = {}
    for row in rows:
        name = row['Name']
        role, _ = categorize_fn(name)
        median_ns = float(row['Med (ns)'])
        instances = int(row['Instances'])
        if instances == 0: continue
        calls_per_fwd = instances / total_fwds
        per_fwd_us = median_ns / 1000.0 * calls_per_fwd
        roles[role] = roles.get(role, 0) + per_fwd_us
    return roles

sg_per_fwd = aggregate_by_role(sg_rows, SG_FWDS, categorize_sg)
lc_per_fwd = aggregate_by_role(lc_rows, LC_FWDS, categorize_lc)

print("\n=== Per-forward GPU time by role (median × calls/forward) ===")
print(f"  {'role':<35} {'speedygrad us':>15} {'llama.cpp us':>15}")

# Combine roles into broad categories for comparison
def broad(role):
    if role.startswith('matmul:'): return 'matmul (all)'
    if role.startswith('attn:'): return 'attention (all)'
    if role.startswith('embedding'): return 'embedding'
    if role.startswith('argmax'): return 'argmax'
    return role

sg_broad = {}
for r, t in sg_per_fwd.items(): sg_broad[broad(r)] = sg_broad.get(broad(r), 0) + t
lc_broad = {}
for r, t in lc_per_fwd.items(): lc_broad[broad(r)] = lc_broad.get(broad(r), 0) + t

print("\n=== Broad category comparison ===")
print(f"  {'category':<25} {'speedygrad us':>15} {'llama.cpp us':>15} {'gap us':>10}")
all_cats = sorted(set(sg_broad) | set(lc_broad))
total_sg = total_lc = 0
for cat in all_cats:
    sg_t = sg_broad.get(cat, 0)
    lc_t = lc_broad.get(cat, 0)
    total_sg += sg_t
    total_lc += lc_t
    gap = sg_t - lc_t
    print(f"  {cat:<25} {sg_t:>15.1f} {lc_t:>15.1f} {gap:>+10.1f}")
print(f"  {'TOTAL GPU':<25} {total_sg:>15.1f} {total_lc:>15.1f} {total_sg-total_lc:>+10.1f}")

print("\n=== Detailed speedygrad per-forward by role ===")
for r, t in sorted(sg_per_fwd.items(), key=lambda x: -x[1]):
    print(f"  {r:<35} {t:>10.1f} us")

print("\n=== Detailed llama.cpp per-forward by role ===")
for r, t in sorted(lc_per_fwd.items(), key=lambda x: -x[1]):
    print(f"  {r:<35} {t:>10.1f} us")

print("\n=== Per-fix savings analysis ===")
softmax_3pass_us = sg_per_fwd.get('attn:softmax_pass', 0) + sg_per_fwd.get('attn:softmax_normalize', 0)
print(f"  3-pass softmax (current):           {softmax_3pass_us:>8.1f} us / forward")
print(f"  online softmax (1-pass, ~3x):       ~{softmax_3pass_us/3:>7.1f} us / forward")
print(f"  online-softmax-only saving:         ~{softmax_3pass_us*2/3:>7.1f} us / forward")
attn_total_sg = sum(t for r, t in sg_per_fwd.items() if r.startswith('attn:'))
attn_total_lc = sum(t for r, t in lc_per_fwd.items() if r.startswith('attn:'))
print(f"  full attention chain (current sg):   {attn_total_sg:>8.1f} us / forward")
print(f"  llama.cpp attention total:           {attn_total_lc:>8.1f} us / forward")
print(f"  full FlashAttention fusion saving:  ~{attn_total_sg-attn_total_lc:>7.1f} us / forward (theoretical max)")
matmul_sg = sg_broad.get('matmul (all)', 0)
matmul_lc = lc_broad.get('matmul (all)', 0)
print(f"  matmul total speedygrad/llama.cpp:   {matmul_sg:.1f} us / {matmul_lc:.1f} us  gap +{matmul_sg-matmul_lc:.1f} us")
