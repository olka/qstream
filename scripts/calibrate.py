"""CLI: per-layer calibration for activation-aware quantization.

Produces a calibration_stats.json that quantize.py can use instead of the
γ proxy (input_layernorm.weight) for activation-weighted MSE scale selection.

Stats format: {layer_idx: {pre_attn, post_attn, pre_mlp, pre_down}}
Each value is mean(|x|, dim=tokens) over the calibration corpus.

Usage:
    quant4-calibrate \\
        --model_dir /path/to/model \\
        --corpus calibration.txt \\
        --output_path calibration_stats.json \\
        --n_tokens 512 \\
        --model_family qwen3
"""

import argparse
import json
from pathlib import Path

from quant4.calibrate import MiniMaxLayerRunner, ModelConfig, Qwen3LayerRunner, collect_activation_stats

RUNNERS = {
    "qwen3": Qwen3LayerRunner,
    "minimax": MiniMaxLayerRunner,
}


def main():
    parser = argparse.ArgumentParser(description="Per-layer calibration (quant4)")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--corpus", required=True, help="Plain text file for calibration")
    parser.add_argument("--output_path", default="calibration_stats.json")
    parser.add_argument("--n_tokens", type=int, default=512)
    parser.add_argument("--model_family", choices=list(RUNNERS), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expert_buffer", type=int, default=32,
                        help="Experts loaded at once (MiniMax only). "
                             "Peak RAM ≈ expert_buffer × 54 MB. Default 32 → ~1.7 GB.")
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("transformers required: pip install transformers")

    model_dir = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    with open(args.corpus) as f:
        text = f.read(args.n_tokens * 6)  # rough char estimate

    token_ids = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=args.n_tokens
    ).input_ids

    print(f"Calibrating with {token_ids.shape[1]} tokens on {args.device}")

    runner_cls = RUNNERS[args.model_family]
    runner_kwargs = {"device": args.device}
    if args.model_family == "minimax":
        runner_kwargs["expert_buffer"] = args.expert_buffer
    runner = runner_cls(model_dir, **runner_kwargs)

    with open(model_dir / "config.json") as f:
        model_cfg = ModelConfig.from_json(json.load(f))

    stats = collect_activation_stats(runner, token_ids, n_layers=model_cfg.n_layers)

    # Serialize: {str(layer_idx): {act_type: [float, ...]}}
    serializable = {
        str(layer_idx): {
            act_type: tensor.tolist()
            for act_type, tensor in layer_stats.items()
        }
        for layer_idx, layer_stats in stats.items()
    }
    with open(args.output_path, "w") as f:
        json.dump(serializable, f)

    print(f"Saved calibration stats for {len(stats)} layers → {args.output_path}")


if __name__ == "__main__":
    main()
