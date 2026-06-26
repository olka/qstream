"""CLI: file-to-file MXFP4 quantization.

Usage:
    qstream-quantize \\
        --model_dir /path/to/model \\
        --output_dir /path/to/output \\
        --workers 8

    # For vLLM fork (fused expert format):
    qstream-quantize \\
        --model_dir /path/to/model \\
        --output_dir /path/to/output \\
        --format fused

Flags:
    --format {ct,fused}     Output format: ct (stock vLLM, default) or fused (vLLM fork).
    --no_activation_aware   Disable γ-weighted MSE (unweighted MSE only).
    --scale_percentile N    Anchor percentile for MSE candidate generation (default 99.5).
    --exclude_layers ...    Substring patterns for tensors to skip.
    --include_layers ...    Substring patterns for tensors to quantize (overrides --exclude_layers).
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import resource
import signal
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

from safetensors import safe_open

# Raise the open-file-descriptor limit to the hard cap.
# Default soft limit (1024) is too low for large models with many shards and workers.
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(_hard, 65536), _hard))

from qstream.gamma import load_layernorm_gammas
from qstream.output import build_index_from_shards, build_quantization_config
from qstream.shard import classify_shard, detect_input_format, process_shard


def main():
    parser = argparse.ArgumentParser(description="Quantization (qstream)")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--quant_format", choices=["mxfp4", "fp8"], default="mxfp4",
        help="Quantization format: mxfp4 (4-bit, default) or fp8 (8-bit per-channel)",
    )
    parser.add_argument(
        "--exclude_layers",
        nargs="*",
        default=[
            "*self_attn*",
            "*.mlp.gate.",
            "*shared_expert*",
            "*lm_head*",
            "*embed_tokens*",
            "*visual*",
            "*mtp*",
        ],
        help="Substring patterns for tensors to exclude from quantization",
    )
    parser.add_argument(
        "--include_layers",
        nargs="*",
        default=None,
        help="Substring patterns for tensors to include (overrides --exclude_layers). "
             "Only matching tensors are quantized; everything else is passed through.",
    )
    parser.add_argument("--fp8_block_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads_per_worker", type=int, default=0,
                        help="0 = auto (cpu_count / workers)")
    parser.add_argument("--shard_timeout", type=int, default=1800,
                        help="Seconds without any shard completing before assuming "
                             "deadlock and retrying. Bump for very large shards.")
    parser.add_argument("--scale_percentile", type=float, default=99.5,
                        help="Anchor percentile for MSE candidate generation")
    parser.add_argument("--no_activation_aware", action="store_true",
                        help="Disable γ-weighted MSE, use unweighted MSE only")
    parser.add_argument("--device", default="cpu",
                        help="Device for quantize_mxfp4 kernel (cpu or cuda)")
    parser.add_argument("--calibration_stats", default=None,
                        help="Path to calibration_stats.json from scripts/calibrate.py. "
                             "When provided, overrides input_layernorm.weight γ proxy.")
    parser.add_argument("--use_zero_copy", action="store_true",
                        help="Enable zero-copy passthrough: symlink shards with only "
                             "passthrough tensors (faster but leaves stale BF16 expert "
                             "tensors in symlinked shards, incompatible with some loaders)")
    parser.add_argument("--format", choices=["fused", "ct"], default="ct",
                        dest="output_format",
                        help="Output format: 'ct' (compressed-tensors per-expert, stock vLLM) or "
                             "'fused' (w13/w2 interleaved, needs vLLM fork). CT handles both "
                             "Qwen3.5-style `.experts.PROJ` and Step-3.7-style `.moe.PROJ.weight` "
                             "inputs, normalizing emitted keys to `.experts.{E}.PROJ.weight_packed`.")
    parser.add_argument("--expert_config", default=None,
                        help="Path to expert_errors.json from analyze_experts.py. "
                             "Enables selective per-expert quantization: low-error "
                             "experts → MXFP4, high-error experts → keep FP8.")
    parser.add_argument("--expert_budget_gb", type=float, default=None,
                        help="Target GB to save by MXFP4-quantizing experts. "
                             "Requires --expert_config. Selects the N lowest-error "
                             "experts needed to reach the target savings.")
    args = parser.parse_args()

    if args.threads_per_worker == 0:
        args.threads_per_worker = max(1, os.cpu_count() // args.workers)

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
    else:
        # Some checkpoints (e.g. MiniMax-M3) ship without an index — reconstruct it
        # from shard headers so the rest of the pipeline is unchanged.
        all_shards = sorted(p.name for p in model_dir.glob("*.safetensors"))
        if not all_shards:
            raise FileNotFoundError(f"No index and no .safetensors shards in {model_dir}")
        print(f"No index found — generating from {len(all_shards)} shard headers")
        index = build_index_from_shards(str(model_dir), all_shards)

    weight_map = index["weight_map"]
    shard_files = sorted(set(weight_map.values()))
    existing = [s for s in shard_files if (model_dir / s).exists()]
    missing = [s for s in shard_files if not (model_dir / s).exists()]
    if missing:
        print(f"WARNING: {len(missing)} missing shards skipped: {missing[:3]}...")

    # Scan shards until a quantized format is found: the first shard may hold only
    # dense/embedding BF16 tensors (e.g. Step-3.7), while the FP8/MXFP8 experts live
    # in later shards. Default to fp16 only if no scale tensors exist anywhere.
    input_format = "fp16"
    for s in existing:
        fmt = detect_input_format(str(model_dir / s))
        if fmt != "fp16":
            input_format = fmt
            break
    print(f"Input format:     {input_format.upper()}")
    print(f"Quant format:     {args.quant_format.upper()}")
    print(f"Output format:    {args.output_format}")
    print(f"Shards:           {len(existing)}")
    print(f"Workers:          {args.workers} × {args.threads_per_worker} threads")
    if args.quant_format == "mxfp4":
        print(f"Scale percentile: {args.scale_percentile}")
    if args.include_layers:
        print(f"Include patterns: {args.include_layers}")
        print(f"  (--exclude_layers ignored)")
    else:
        print(f"Exclude patterns: {args.exclude_layers}")
    if args.quant_format == "mxfp4":
        print(f"MSE scale select: enabled (3 candidates per block)")
    else:
        print(f"FP8 scale:        per-channel (amax / FP8_MAX)")

    calibration_stats_path = None
    gamma_by_layer = None
    if not args.no_activation_aware:
        if args.calibration_stats:
            # Validate the file and print summary, but don't load tensors here —
            # workers load it themselves to avoid cross-process fd sharing.
            with open(args.calibration_stats) as f:
                _raw = json.load(f)
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

    # Selective per-expert quantization
    expert_quantize_set = None
    if args.expert_config:
        with open(args.expert_config) as f:
            expert_data = json.load(f)
        all_experts = expert_data["experts_by_error"]  # sorted ascending by error
        savings_per_expert = expert_data["savings_per_expert_mb"]

        if args.expert_budget_gb:
            n_needed = min(
                int(args.expert_budget_gb * 1024 / savings_per_expert) + 1,
                len(all_experts),
            )
            selected = all_experts[:n_needed]
            actual_savings = n_needed * savings_per_expert / 1024
            worst_error = selected[-1]["rel_error"] if selected else 0
            print(f"Expert config:    {n_needed}/{len(all_experts)} experts → MXFP4 "
                  f"(saves {actual_savings:.1f} GB, worst error {worst_error:.6f})")
        else:
            selected = all_experts
            print(f"Expert config:    all {len(all_experts)} experts → MXFP4")

        expert_quantize_set = {(e["layer"], e["expert"]) for e in selected}
        n_kept = len(all_experts) - len(selected)
        if n_kept > 0:
            print(f"                  {n_kept} experts kept in FP8")

    # Zero-copy passthrough: for BF16 models, symlink shards with unchanged
    # tensors instead of reading and rewriting them.
    input_has_symlinks = any((model_dir / s).is_symlink() for s in existing)
    use_zero_copy = (input_format == "fp16" and args.use_zero_copy
                     and model_dir.resolve() != output_dir.resolve()
                     and not input_has_symlinks)
    if use_zero_copy:
        print("Zero-copy:        enabled (symlink passthrough tensors)")
    elif input_has_symlinks:
        print("Zero-copy:        disabled (input contains symlinks — likely a previous zero-copy output)")
    elif input_format == "fp8":
        print("Zero-copy:        disabled (FP8 models need dequant)")
    elif not args.use_zero_copy:
        print("Zero-copy:        disabled (default)")
    elif model_dir.resolve() == output_dir.resolve():
        print("Zero-copy:        disabled (input == output directory)")

    final_weight_map = {}
    shards_to_process = []  # (shard_name, output_path, quantize_only_keys or None)
    n_linked = 0
    n_mixed = 0
    n_full = 0
    bytes_linked = 0

    for s in existing:
        input_path = model_dir / s

        if use_zero_copy:
            keys_q, keys_p = classify_shard(str(input_path), args.exclude_layers, args.include_layers)

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

    process_size = sum(os.path.getsize(model_dir / s) for s, _, _, _ in shards_to_process)

    # Use 'spawn' when CUDA is requested to avoid fork-induced deadlocks.
    # Forked processes inherit the parent's CUDA context, causing contention.
    mp_ctx = mp.get_context("spawn") if args.device != "cpu" else None

    shard_timeout = args.shard_timeout

    remaining = list(shards_to_process)
    t0 = time.monotonic()
    done = 0
    bytes_done = 0

    while remaining:
        batch = list(remaining)
        remaining = []

        pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=mp_ctx)
        futures = {}
        for src_name, out_path, q_keys, out_shard_name in batch:
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
                args.output_format,
                args.include_layers,
                args.quant_format,
                expert_quantize_set,
            )
            futures[fut] = (src_name, out_shard_name)

        pending = set(futures.keys())
        last_completion = time.monotonic()
        timed_out = False

        while pending:
            done_set, pending = wait(pending, timeout=30,
                                     return_when=FIRST_COMPLETED)

            if done_set:
                last_completion = time.monotonic()
                for fut in done_set:
                    src_name, out_shard_name = futures[fut]
                    try:
                        result_map = fut.result()
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

            elif time.monotonic() - last_completion > shard_timeout:
                # No shard completed within timeout — assume deadlock.
                stuck = [futures[f][0] for f in pending]
                print(f"\nTIMEOUT: no shard completed in {shard_timeout}s — "
                      f"{len(stuck)} stuck: {stuck}")
                for fut in pending:
                    remaining.append(next(
                        s for s in batch if s[0] == futures[fut][0]
                    ))
                    print(f"  Will retry: {futures[fut][0]}")
                timed_out = True
                break

        # Kill worker processes and shut down pool without blocking.
        if timed_out:
            for pid, proc in pool._processes.items():
                if proc.is_alive():
                    os.kill(pid, signal.SIGKILL)
        pool.shutdown(wait=not timed_out, cancel_futures=timed_out)

    if not final_weight_map:
        raise RuntimeError("No tensors found in output — something went wrong")

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

    fp8_modules = sorted(
        k[: -len(".weight")]
        for k in final_weight_map
        if k.endswith(".weight")
        and k.replace(".weight", ".weight_scale") in final_weight_map
    )
    fp8_modules_set = set(fp8_modules)
    ignore_modules = sorted(
        k[: -len(".weight")]
        for k in final_weight_map
        if k.endswith(".weight") and k[: -len(".weight")] not in fp8_modules_set
    )

    used_mixed_builder = False
    if input_format in ("mxfp8", "fp8") and any(
        k.endswith(".weight_packed") for k in final_weight_map
    ):
        # Mixed precision: routed experts → MXFP4 (.weight_packed); every other FP8
        # layer kept at native 8-bit (.weight + .weight_scale = fp8_modules).
        # MXFP8 input keeps e8m0 group-32 FP8 (M3); plain FP8 input keeps block FP8
        # (Step-3.7/DeepSeek, 128×128 float scales).
        mxfp4_modules = sorted(
            k[: -len(".weight_packed")]
            for k in final_weight_map
            if k.endswith(".weight_packed")
        )
        # Unquantized linears (router gate, lm_head, embeddings, vision, projector):
        # .weight with no scale companion. Exclude norms (not Linear).
        kept_ignore = sorted(
            k[: -len(".weight")]
            for k in final_weight_map
            if k.endswith(".weight")
            and k[: -len(".weight")] not in fp8_modules_set
            and "norm" not in k
        )
        config["quantization_config"] = build_quantization_config(
            mxfp4_modules, fp8_modules, kept_ignore,
            fp8_kind="mxfp8" if input_format == "mxfp8" else "block",
            fp8_block_size=args.fp8_block_size,
        )
        used_mixed_builder = True
    elif args.quant_format == "fp8":
        config["quantization_config"] = {
            "quant_method": "compressed-tensors",
            "format": "float-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 8,
                        "type": "float",
                        "strategy": "channel",
                        "symmetric": True,
                        "dynamic": False,
                    },
                }
            },
            "ignore": ignore_modules,
        }
    elif any("w13_weight" in k for k in final_weight_map) and args.output_format == "fused":
        config["quantization_config"] = {"quant_method": "mxfp4"}
    else:
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

    if not used_mixed_builder and fp8_modules and "config_groups" in config["quantization_config"]:
        sample_scale_key = fp8_modules[0] + ".weight_scale"
        sample_shard = output_dir / final_weight_map[sample_scale_key]
        with safe_open(str(sample_shard), framework="pt") as f:
            per_tensor = f.get_tensor(sample_scale_key).numel() == 1
        has_input_scale = any(
            (m + ".input_scale") in final_weight_map for m in fp8_modules
        )

        fp8_targets: list[str] = []
        for pat in (args.exclude_layers or []):
            stripped = pat.strip("*")
            if not stripped:
                continue
            probe = re.compile(re.escape(stripped))
            if any(probe.search(m) for m in fp8_modules):
                fp8_targets.append(f"re:.*{re.escape(stripped)}.*")
        if not fp8_targets:
            fp8_targets = ["Linear"]

        fp8_group: dict = {
            "targets": fp8_targets,
            "format": "float-quantized",
            "weights": {
                "num_bits": 8,
                "type": "float",
                "strategy": "tensor" if per_tensor else "block",
                "symmetric": True,
                "dynamic": False,
            },
        }
        if not per_tensor:
            fp8_group["weights"]["block_structure"] = [
                args.fp8_block_size, args.fp8_block_size
            ]
        if has_input_scale:
            fp8_group["input_activations"] = {
                "num_bits": 8,
                "type": "float",
                "strategy": "tensor",
                "symmetric": True,
                "dynamic": False,
            }
        config["quantization_config"]["config_groups"]["group_1"] = fp8_group

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    total_in = sum(os.path.getsize(model_dir / s) for s in existing)
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
