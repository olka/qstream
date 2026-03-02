"""quant4 — file-to-file MXFP4 quantization for large language models."""

from .calibrate import ModelConfig
from .core import BLOCK_SIZE, compute_block_hessian, quantize_mxfp4
from .fp8 import dequant_fp8_block
from .gamma import extract_expert_index, extract_layer_index, load_layernorm_gammas
from .shard import detect_input_format, process_shard, should_quantize
