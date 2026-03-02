"""Per-layer and per-block quantization error analysis.

Compares original weights vs dequantized MXFP4 weights to identify
where quantization error is concentrated.

Usage:
    qstream-analyze \
        --orig_dir <path to original model> \
        --quant_dir <path to quantized model>
"""
import argparse
import itertools
import json
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from safetensors import safe_open

from qstream.core import BLOCK_SIZE, dequant_mxfp4
from qstream.fp8 import dequant_fp8_block
from qstream.gamma import extract_layer_index

_PREFETCH = 10
_BATCH_SIZE = 32

_FP8_DTYPES = set()
for _name in ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2"):
    if hasattr(torch, _name):
        _FP8_DTYPES.add(getattr(torch, _name))


def load_tensor(model_dir: Path, weight_map: dict, key: str) -> torch.Tensor:
    shard = weight_map[key]
    with safe_open(str(model_dir / shard), framework="pt") as f:
        return f.get_tensor(key)


def _load_group_cpu(
    pk: str,
    quant_dir: Path, quant_map: dict,
    orig_dir: Path, orig_map: dict,
    scale_inv_map: dict,
    quark_fmt: bool = False,
) -> tuple:
    """Load all tensors for one key group into CPU RAM (runs in thread pool)."""
    if quark_fmt:
        base = pk[: -len(".weight")]
        sk = base + ".weight_scale"
        orig_k = pk  # same name, loaded from orig_dir
    else:
        sk = pk.replace(".weight_packed", ".weight_scale")
        orig_k = pk.replace(".weight_packed", ".weight")

    packed = load_tensor(quant_dir, quant_map, pk)
    scales = load_tensor(quant_dir, quant_map, sk)
    raw_orig = load_tensor(orig_dir, orig_map, orig_k)

    scale_inv = None
    if raw_orig.dtype in _FP8_DTYPES:
        scale_k = scale_inv_map.get(orig_k)
        if scale_k:
            scale_inv = load_tensor(orig_dir, orig_map, scale_k)

    return pk, orig_k, packed, scales, raw_orig, scale_inv


def _dequant_fp8_batch(
    weight_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """Batched FP8 dequant: [N, out, in] fp8, [N, out//bs, in//bs] scale → [N, out, in] bf16."""
    N, out_f, in_f = weight_fp8.shape
    n_bo = scale_inv.shape[1]
    n_bi = scale_inv.shape[2]
    w = weight_fp8.float()
    w = w.reshape(N, n_bo, block_size, n_bi, block_size)
    w = w * scale_inv.float().unsqueeze(2).unsqueeze(4)
    return w.reshape(N, out_f, in_f).to(torch.bfloat16)


def _compute_batch_metrics(
    orig_w: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
) -> tuple:
    """Dequant + error metrics for a batch of same-shape tensors.

    Args:
        orig_w:  [N, out, in] float32
        packed:  [N, out, in//2] uint8
        scales:  [N, out, in//BLOCK_SIZE] uint8

    Returns:
        E_norm, frob_rel, worst_snr, median_snr — each [N] GPU tensors, no sync.
    """
    N = orig_w.shape[0]
    recon = dequant_mxfp4(packed, scales, orig_w.shape).float()
    E = orig_w - recon

    E_flat = E.reshape(N, -1)
    W_flat = orig_w.reshape(N, -1)
    E_norm  = E_flat.norm(dim=1)
    W_norm  = W_flat.norm(dim=1)
    frob_rel = E_norm / W_norm

    E_blocks = E.reshape(N, -1, BLOCK_SIZE)
    W_blocks = orig_w.reshape(N, -1, BLOCK_SIZE)
    snr = 20 * torch.log10(W_blocks.norm(dim=2) / (E_blocks.norm(dim=2) + 1e-12))
    worst_snr  = snr.min(dim=1).values
    median_snr = snr.median(dim=1).values

    return E_norm, frob_rel, worst_snr, median_snr


def analyze(orig_dir: Path, quant_dir: Path, device: str = "cuda", batch_size: int = _BATCH_SIZE):
    with open(quant_dir / "model.safetensors.index.json") as f:
        quant_map = json.load(f)["weight_map"]
    with open(orig_dir / "model.safetensors.index.json") as f:
        orig_map = json.load(f)["weight_map"]

    # Build FP8 scale_inv lookup for original model (MiniMax / DeepSeek style)
    scale_inv_map: dict[str, str] = {
        k.replace(".weight_scale_inv", ".weight"): k
        for k in orig_map
        if k.endswith(".weight_scale_inv")
    }
    if scale_inv_map:
        print(f"Detected FP8 original model ({len(scale_inv_map)} scale_inv entries).\n")

    packed_keys = sorted(k for k in quant_map if k.endswith(".weight_packed"))
    quark_fmt = False
    if not packed_keys:
        # Quark format: .weight (uint8 packed) + .weight_scale pairs
        quark_candidates = sorted(
            k for k in quant_map
            if k.endswith(".weight") and k[: -len(".weight")] + ".weight_scale" in quant_map
        )
        if quark_candidates:
            packed_keys = quark_candidates
            quark_fmt = True
            print(f"Detected Quark format ({len(packed_keys)} quantized tensors).\n")
        else:
            available = sorted({k.rsplit(".", 1)[-1] for k in quant_map})
            print(f"ERROR: No '.weight_packed' keys (qstream) or '.weight'+'.weight_scale' pairs (Quark) found.")
            print(f"  Available suffixes: {available[:20]}")
            return
    layer_results: dict[int, dict] = defaultdict(lambda: {"frob": [], "rel": [], "block_snr": []})

    print(f"Analyzing {len(packed_keys)} quantized tensors (batch_size={batch_size})...\n")

    # gpu_results: list of (orig_k, E_norm_scalar, frob_rel_scalar, worst_snr_scalar, median_snr_scalar)
    # We store 1-element GPU tensors and batch-collect .item() at the very end.
    gpu_results: list = []

    # Accumulators grouped by tensor shape — flush to GPU when full
    # shape → list of (pk, orig_k, packed_cpu, scales_cpu, raw_orig_cpu, scale_inv_cpu)
    accum: dict[tuple, list] = defaultdict(list)

    def flush(shape: tuple):
        items = accum.pop(shape)
        N = len(items)

        packed_b = torch.stack([it[2] for it in items]).to(device)
        scales_b = torch.stack([it[3] for it in items]).to(device)
        raw_b    = torch.stack([it[4] for it in items]).to(device)

        if raw_b.dtype in _FP8_DTYPES:
            sinvs = [it[5] for it in items]
            if all(s is not None for s in sinvs):
                sinv_b = torch.stack(sinvs).to(device)
                orig_w = _dequant_fp8_batch(raw_b, sinv_b).float()
            else:
                missing = [items[i][1] for i, s in enumerate(sinvs) if s is None]
                print(f"  WARNING: missing scale_inv for {missing} — raw FP8 values used")
                orig_w = raw_b.float()
        else:
            orig_w = raw_b.float()

        E_norm_b, frob_rel_b, worst_b, median_b = _compute_batch_metrics(orig_w, packed_b, scales_b)

        for i, item in enumerate(items):
            gpu_results.append((item[1], E_norm_b[i], frob_rel_b[i], worst_b[i], median_b[i]))

    def submit(pool, pk):
        return pool.submit(
            _load_group_cpu, pk,
            quant_dir, quant_map, orig_dir, orig_map, scale_inv_map, quark_fmt,
        )

    def ingest(item):
        shape = item[4].shape
        accum[shape].append(item)
        if len(accum[shape]) >= batch_size:
            flush(shape)

    with ThreadPoolExecutor(max_workers=_PREFETCH) as pool:
        pending: deque = deque()
        key_iter = iter(packed_keys)

        for pk in itertools.islice(key_iter, _PREFETCH):
            pending.append(submit(pool, pk))

        for pk in key_iter:
            pending.append(submit(pool, pk))
            ingest(pending.popleft().result())

        while pending:
            ingest(pending.popleft().result())

    for shape in list(accum):
        flush(shape)

    # Single sync point — collect all GPU scalars
    all_frob = []
    for orig_k, E_norm_t, frob_rel_t, worst_snr_t, median_snr_t in gpu_results:
        frob           = E_norm_t.item()
        frob_rel       = frob_rel_t.item()
        worst_block_snr  = worst_snr_t.item()
        median_block_snr = median_snr_t.item()

        li   = extract_layer_index(orig_k)
        proj = orig_k.split(".")[-2] if orig_k.endswith(".weight") else orig_k.split(".")[-1]

        if li is not None:
            layer_results[li]["frob"].append(frob)
            layer_results[li]["rel"].append(frob_rel)
            layer_results[li]["block_snr"].append(worst_block_snr)
        all_frob.append((frob_rel, li, proj, worst_block_snr, median_block_snr))

    print("Per-layer error (sum of Frobenius norms across projections):")
    print(f"  {'Layer':>5}  {'Frob(E) sum':>12}  {'Rel error':>10}  {'Worst block SNR':>16}")
    print("  " + "-" * 50)

    layer_frob_total = {}
    for li in sorted(k for k in layer_results if k is not None):
        total_frob = sum(layer_results[li]["frob"])
        avg_rel    = sum(layer_results[li]["rel"]) / len(layer_results[li]["rel"])
        worst_snr  = min(layer_results[li]["block_snr"])
        layer_frob_total[li] = total_frob
        print(f"  {li:>5}  {total_frob:>12.4f}  {avg_rel:>10.6f}  {worst_snr:>16.2f} dB")

    print("\nTop 15 worst tensors by relative error:")
    print(f"  {'Rel error':>10}  {'Layer':>5}  {'Proj':>20}  {'Worst SNR':>10}")
    print("  " + "-" * 55)
    for rel, li, proj, worst_snr, med_snr in sorted(all_frob, reverse=True)[:15]:
        print(f"  {rel:>10.6f}  {li:>5}  {proj:>20}  {worst_snr:>10.2f} dB")

    all_rel = [x[0] for x in all_frob]
    if not all_rel:
        print("\nNo results collected — nothing to summarize.")
        return
    t = torch.tensor(all_rel)
    print(f"\nRelative error distribution across {len(all_rel)} tensors:")
    print(f"  min:    {t.min():.6f}")
    print(f"  p25:    {t.quantile(0.25):.6f}")
    print(f"  median: {t.median():.6f}")
    print(f"  p75:    {t.quantile(0.75):.6f}")
    print(f"  p95:    {t.quantile(0.95):.6f}")
    print(f"  max:    {t.max():.6f}")
    print(f"  std:    {t.std():.6f}")

    top10_pct   = sorted(all_rel, reverse=True)[:max(1, len(all_rel) // 10)]
    top10_share = sum(top10_pct) / sum(all_rel)
    print(f"\nTop 10% of tensors account for {top10_share*100:.1f}% of total relative error")
    if top10_share > 0.5:
        print("  -> Error is CONCENTRATED (selective dequant/LoRA promising)")
    else:
        print("  -> Error is DIFFUSE (selective approaches unlikely to help)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_dir", required=True)
    parser.add_argument("--quant_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=_BATCH_SIZE,
                        help="Tensors per GPU batch (reduce if OOM)")
    args = parser.parse_args()

    orig_dir = Path(args.orig_dir)
    quant_dir = Path(args.quant_dir)

    if (orig_dir / "snapshots").exists():
        snaps = list((orig_dir / "snapshots").iterdir())
        orig_dir = snaps[0]
        print(f"Using snapshot: {orig_dir}")

    analyze(orig_dir, quant_dir, device=args.device, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
