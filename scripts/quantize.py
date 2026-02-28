"""CLI: file-to-file MXFP4 quantization.

Usage (shard-parallel):
    python scripts/quantize.py \\
        --model_dir /path/to/model \\
        --output_dir /path/to/output \\
        --workers 8

Usage (AWQ calibrated):
    python scripts/quantize.py \\
        --model_dir /path/to/model \\
        --output_dir /path/to/output \\
        --calibrate --corpus calibration.txt \\
        --model_family minimax --n_tokens 2048

Flags:
    --no_activation_aware   Disable γ-weighted MSE (unweighted MSE only).
    --scale_percentile N    Anchor percentile for MSE candidate generation (default 99.5).
    --exclude_layers ...    Substring patterns for tensors to skip.
    --calibrate             AWQ Hessian calibration mode (fused calibration + quantization).
"""

import argparse
import json
import os
import resource
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Raise the open-file-descriptor limit to the hard cap.
# Default soft limit (1024) is too low for large models with many shards and workers.
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(_hard, 65536), _hard))

# Allow running as a script without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from quant4.gamma import load_layernorm_gammas
from quant4.shard import classify_shard, detect_input_format, process_shard


def main():
    parser = argparse.ArgumentParser(description="MXFP4 quantization (quant4)")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--exclude_layers",
        nargs="*",
        default=["*self_attn*", "*.mlp.gate.", "*lm_head*", "*embed_tokens*"],
        help="Substring patterns for tensors to exclude from quantization",
    )
    parser.add_argument("--fp8_block_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads_per_worker", type=int, default=0,
                        help="0 = auto (cpu_count / workers)")
    parser.add_argument("--scale_percentile", type=float, default=99.5,
                        help="Anchor percentile for MSE candidate generation")
    parser.add_argument("--no_activation_aware", action="store_true",
                        help="Disable γ-weighted MSE, use unweighted MSE only")
    parser.add_argument("--device", default="cpu",
                        help="Device for quantize_mxfp4 kernel (cpu or cuda)")
    parser.add_argument("--calibration_stats", default=None,
                        help="Path to calibration_stats.json from scripts/calibrate.py. "
                             "When provided, overrides input_layernorm.weight γ proxy.")
    parser.add_argument("--no_zero_copy", action="store_true",
                        help="Disable zero-copy passthrough: read/write all tensors "
                             "(original behavior, slower but no symlinks)")
    # AWQ calibration mode
    parser.add_argument("--calibrate", action="store_true",
                        help="AWQ-style Hessian calibration: run forward pass and "
                             "quantize inline. Requires --corpus and --model_family.")
    parser.add_argument("--corpus", default=None,
                        help="Plain text file for calibration (required with --calibrate)")
    parser.add_argument("--model_family", choices=["qwen3", "minimax"], default=None,
                        help="Model family for LayerRunner (required with --calibrate)")
    parser.add_argument("--n_tokens", type=int, default=512,
                        help="Number of calibration tokens (with --calibrate)")
    parser.add_argument("--expert_buffer", type=int, default=32,
                        help="Experts loaded at once (MiniMax only, with --calibrate). "
                             "Peak RAM ≈ expert_buffer × 54 MB.")
    parser.add_argument("--keep_fp8_layers", type=int, default=0,
                        help="Keep N most sensitive layers at original precision "
                             "(FP8/BF16) instead of MXFP4. Requires --calibrate. "
                             "Adds modules_to_not_convert to quantization_config.")
    args = parser.parse_args()

    if args.threads_per_worker == 0:
        args.threads_per_worker = max(1, os.cpu_count() // args.workers)

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)

    # ---------------------------------------------------------------
    # AWQ calibration mode: fused calibration + quantization
    # ---------------------------------------------------------------
    if args.calibrate:
        if not args.corpus:
            parser.error("--calibrate requires --corpus")
        if not args.model_family:
            parser.error("--calibrate requires --model_family")

        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("transformers required for --calibrate: pip install transformers")

        from quant4.awq import run_awq_pipeline
        from quant4.calibrate import MiniMaxLayerRunner, Qwen3LayerRunner

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        with open(args.corpus) as f:
            text = f.read(args.n_tokens * 6)  # rough char estimate
        token_ids = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=args.n_tokens
        ).input_ids

        RUNNERS = {"qwen3": Qwen3LayerRunner, "minimax": MiniMaxLayerRunner}
        runner_cls = RUNNERS[args.model_family]
        runner_kwargs = {"device": args.device}
        if args.model_family == "minimax":
            runner_kwargs["expert_buffer"] = args.expert_buffer
        processor = runner_cls(model_dir, **runner_kwargs)

        with open(model_dir / "config.json") as f:
            _cfg = json.load(f)
        _cfg = _cfg.get("text_config", _cfg)
        n_layers = _cfg.get("num_hidden_layers")

        print(f"AWQ calibration mode")
        print(f"  Tokens:          {token_ids.shape[1]}")
        print(f"  Layers:          {n_layers}")
        print(f"  Model family:    {args.model_family}")
        print(f"  Device:          {args.device}")
        print(f"  Scale percentile:{args.scale_percentile}")
        print(f"  Exclude:         {args.exclude_layers}")
        if args.keep_fp8_layers:
            print(f"  Keep FP8 layers: {args.keep_fp8_layers} most sensitive")

        run_awq_pipeline(
            processor, token_ids, n_layers, model_dir, output_dir,
            args.exclude_layers, args.scale_percentile,
            keep_fp8_layers=args.keep_fp8_layers,
        )
        return

    # ---------------------------------------------------------------
    # Shard-parallel mode (existing path)
    # ---------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing {index_path}")

    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    shard_files = sorted(set(weight_map.values()))
    existing = [s for s in shard_files if (model_dir / s).exists()]
    missing = [s for s in shard_files if not (model_dir / s).exists()]
    if missing:
        print(f"WARNING: {len(missing)} missing shards skipped: {missing[:3]}...")

    input_format = detect_input_format(str(model_dir / existing[0]))
    print(f"Input format:     {input_format.upper()}")
    print(f"Shards:           {len(existing)}")
    print(f"Workers:          {args.workers} × {args.threads_per_worker} threads")
    print(f"Scale percentile: {args.scale_percentile}")
    print(f"Exclude patterns: {args.exclude_layers}")
    print(f"MSE scale select: enabled (3 candidates per block)")

    calibration_stats_path = None
    gamma_by_layer = None
    if not args.no_activation_aware:
        if args.calibration_stats:
            # Validate the file and print summary, but don't load tensors here —
            # workers load it themselves to avoid cross-process fd sharing.
            import json as _json
            with open(args.calibration_stats) as f:
                _raw = _json.load(f)
            layer_keys = sorted(int(k) for k in _raw)
            print(f"Calibration stats: {len(layer_keys)} layers "
                  f"(layers {layer_keys[0]}-{layer_keys[-1]}), "
                  f"keys: {list(_raw[str(layer_keys[0])].keys())}")
            calibration_stats_path = args.calibration_stats
        else:
            print("Loading input_layernorm.weight tensors for γ-weighted MSE...")
            gamma_by_layer = load_layernorm_gammas(model_dir, existing)
            if gamma_by_layer:
                print(f"  γ found for {len(gamma_by_layer)} layers "
                      f"(layers {min(gamma_by_layer)}-{max(gamma_by_layer)})")
            else:
                print("  No input_layernorm.weight found — using unweighted MSE")
    else:
        print("γ-weighted MSE:   disabled")

    # Zero-copy passthrough: for BF16 models, symlink shards with unchanged
    # tensors instead of reading and rewriting them.
    input_has_symlinks = any((model_dir / s).is_symlink() for s in existing)
    use_zero_copy = (input_format == "fp16" and not args.no_zero_copy
                     and model_dir.resolve() != output_dir.resolve()
                     and not input_has_symlinks)
    if use_zero_copy:
        print("Zero-copy:        enabled (symlink passthrough tensors)")
    elif input_has_symlinks:
        print("Zero-copy:        disabled (input contains symlinks — likely a previous zero-copy output)")
    elif input_format == "fp8":
        print("Zero-copy:        disabled (FP8 models need dequant)")
    elif args.no_zero_copy:
        print("Zero-copy:        disabled (--no_zero_copy)")
    elif model_dir.resolve() == output_dir.resolve():
        print("Zero-copy:        disabled (input == output directory)")

    # Classify shards and build dispatch plan
    final_weight_map = {}
    shards_to_process = []  # (shard_name, output_path, quantize_only_keys or None)
    n_linked = 0
    n_mixed = 0
    n_full = 0
    bytes_linked = 0

    for s in existing:
        input_path = model_dir / s

        if use_zero_copy:
            keys_q, keys_p = classify_shard(str(input_path), args.exclude_layers)

            if not keys_q:
                # ALL_PASSTHROUGH — symlink the entire shard
                dst = output_dir / s
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                os.symlink(input_path.resolve(), dst)
                for k in keys_p:
                    final_weight_map[k] = s
                n_linked += 1
                bytes_linked += os.path.getsize(input_path)
                print(f"  Linked {s} ({len(keys_p)} tensors, all passthrough)")
                continue

            elif keys_p:
                # MIXED — symlink original for passthrough, new shard for quantized
                dst = output_dir / s
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                os.symlink(input_path.resolve(), dst)
                for k in keys_p:
                    final_weight_map[k] = s
                # Quantized tensors go to a separate shard
                q_shard_name = s.replace(".safetensors", ".quantized.safetensors")
                shards_to_process.append((s, str(output_dir / q_shard_name), keys_q, q_shard_name))
                n_mixed += 1
            else:
                # ALL_QUANTIZE — process normally
                shards_to_process.append((s, str(output_dir / s), None, s))
                n_full += 1
        else:
            # No zero-copy — process everything
            shards_to_process.append((s, str(output_dir / s), None, s))
            n_full += 1

    if use_zero_copy:
        print(f"Shard plan:       {n_linked} linked, {n_mixed} mixed, {n_full} full-process "
              f"({len(shards_to_process)} to quantize)")

    total_in_size = sum(os.path.getsize(model_dir / s) for s in existing)
    process_size = sum(os.path.getsize(model_dir / s) for s, _, _, _ in shards_to_process)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for src_name, out_path, q_keys, out_shard_name in shards_to_process:
            fut = pool.submit(
                process_shard,
                str(model_dir / src_name),
                out_path,
                args.exclude_layers,
                input_format,
                args.fp8_block_size,
                args.threads_per_worker,
                args.scale_percentile,
                gamma_by_layer,
                calibration_stats_path,
                args.device,
                q_keys,
            )
            futures[fut] = (src_name, out_shard_name)

        import time
        t0 = time.monotonic()
        done = 0
        bytes_done = 0
        for fut in as_completed(futures):
            src_name, out_shard_name = futures[fut]
            try:
                result_map = fut.result()
                # Update index: result_map has {tensor_name: output_shard_basename}
                final_weight_map.update(result_map)
                done += 1
                bytes_done += os.path.getsize(model_dir / src_name)
                elapsed = time.monotonic() - t0
                pct = bytes_done / process_size * 100 if process_size else 100
                eta = elapsed / bytes_done * (process_size - bytes_done) if bytes_done else 0
                eta_m, eta_s = divmod(int(eta), 60)
                print(f"[{done}/{len(shards_to_process)}] {src_name} done  "
                      f"({pct:.0f}% | elapsed {int(elapsed)}s | ETA {eta_m}m{eta_s:02d}s)")
            except Exception as e:
                print(f"ERROR in {src_name}: {e}")
                raise

    if not final_weight_map:
        raise RuntimeError("No tensors found in output — something went wrong")

    # Copy tokenizer, configs, etc.
    for src in model_dir.iterdir():
        if src.is_file() and not src.name.endswith(".safetensors"):
            shutil.copy2(src, output_dir / src.name)
            print(f"Copied {src.name}")

    # Write index from explicitly tracked mappings
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": index.get("metadata", {}), "weight_map": final_weight_map}, f, indent=2)
    print(f"Index: {len(final_weight_map)} tensors across {len(set(final_weight_map.values()))} shards")

    config_path = output_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    # Modules with plain .weight in the output were not quantized.
    ignore_modules = sorted(
        k[: -len(".weight")]
        for k in final_weight_map
        if k.endswith(".weight")
    )
    config["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "mxfp4-pack-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "strategy": "group",
                    "group_size": 32,
                    "symmetric": True,
                },
            }
        },
        "ignore": ignore_modules,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    total_in = sum(os.path.getsize(model_dir / s) for s in existing)
    # Count actual new data written (exclude symlinked shards)
    new_data = 0
    total_out = 0
    for sf in output_dir.glob("*.safetensors"):
        size = os.path.getsize(sf)
        total_out += size
        if not sf.is_symlink():
            new_data += size
    print(f"\nDone! {total_in/1e9:.1f}GB → {total_out/1e9:.1f}GB ({total_out/total_in*100:.1f}%)")
    if use_zero_copy and new_data < total_out:
        print(f"New data written: {new_data/1e9:.1f}GB (rest symlinked from source)")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
