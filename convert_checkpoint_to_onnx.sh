#!/bin/bash

# Simple wrapper to convert a FALCON checkpoint to ONNX.
# Usage:
#   ./convert_checkpoint_to_onnx.sh --checkpoint <ckpt.pt> --output <out.onnx> [--config <config.yaml>]

# Ensure we are in the FALCON root directory (where this script lives)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate conda environment (assumes 'conda' is available in the shell)
conda activate fcgym

# Forward all arguments to the Python converter
python sim2real/utils/convert_checkpoint_to_onnx.py "$@"


