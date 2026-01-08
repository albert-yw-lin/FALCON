#!/usr/bin/env python3
"""
Convert RL checkpoint to ONNX format for sim2sim deployment.

Usage:
    python convert_checkpoint_to_onnx.py --checkpoint <path_to_checkpoint.pt> [--output <output_path.onnx>]
    
Example:
    python convert_checkpoint_to_onnx.py --checkpoint logs/g1_29dof_falcon/.../model_9999.pt --output models/falcon/g1_29dof.onnx
"""

# CRITICAL: Fix import path FIRST, before any other imports
# Python automatically adds the script's directory to sys.path[0], which causes
# local files (like math.py) to shadow standard library modules.
import sys
import os

# Get script directory and remove it from sys.path immediately
# This must happen before ANY imports that might trigger math import
_script_file = __file__
_script_dir = os.path.dirname(os.path.abspath(_script_file))

# Remove script directory from sys.path (Python adds it automatically at position 0)
# Remove all occurrences to be safe
while _script_dir in sys.path:
    sys.path.remove(_script_dir)
# Also try with resolved absolute path
_script_dir_resolved = os.path.realpath(_script_dir)
while _script_dir_resolved in sys.path:
    sys.path.remove(_script_dir_resolved)

# Clear any cached modules that might have been imported from the wrong location
# This is important because Python caches modules in sys.modules
if 'math' in sys.modules and hasattr(sys.modules['math'], '__file__'):
    math_file = sys.modules['math'].__file__
    if math_file and _script_dir in math_file:
        # The wrong math module was cached, remove it
        del sys.modules['math']

# Now safe to import other modules
from pathlib import Path
import argparse
import isaacgym
# Add FALCON root to path for importing humanoidverse
falcon_root = Path(_script_file).parent.parent.parent
if str(falcon_root) not in sys.path:
    sys.path.insert(0, str(falcon_root))

# Change working directory to FALCON root
os.chdir(falcon_root)

import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from humanoidverse.utils.logging import HydraLoggerBridge
import logging
from humanoidverse.utils.config_utils import *  # noqa: E402, F403
from loguru import logger
from humanoidverse.agents.base_algo.base_algo import BaseAlgo
from humanoidverse.utils.helpers import pre_process_config
from humanoidverse.utils.inference_helpers import export_multi_agent_decouple_policy_as_onnx


def convert_checkpoint_to_onnx(checkpoint_path, output_path=None, config_path=None):
    """
    Convert a checkpoint to ONNX format.
    
    Args:
        checkpoint_path: Path to the checkpoint file (.pt)
        output_path: Optional output path for ONNX file. If None, saves next to checkpoint.
        config_path: Optional path to config.yaml. If None, tries to find it automatically.
    """
    checkpoint = Path(checkpoint_path)
    
    # Find config file
    if config_path is None:
        config_path = checkpoint.parent / "config.yaml"
        if not config_path.exists():
            config_path = checkpoint.parent.parent / "config.yaml"
    
    logger.info(f"Loading config from: {config_path}")
    with open(config_path) as file:
        config = OmegaConf.load(file)
    
    config.checkpoint = str(checkpoint)
    
    # Handle eval overrides first
    if config.eval_overrides is not None:
        config = OmegaConf.merge(config, config.eval_overrides)
    
    # Force headless mode and disable viewer for conversion (after merge to override any eval settings)
    config.headless = True
    config.env.config.headless = True
    config.env.config.save_rendering_dir = None
    # Also set num_envs to 1 for conversion (faster)
    config.num_envs = 1
    config.env.config.num_envs = 1
    
    # Import isaacgym first, then torch (required order)
    import torch
    
    pre_process_config(config)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Create environment and algorithm
    env = instantiate(config.env, device=device)
    algo: BaseAlgo = instantiate(config.algo, env=env, device=device, log_dir=None)
    algo.setup()
    algo.load(config.checkpoint)
    
    logger.info("Checkpoint loaded successfully")
    
    # Get example observations
    example_obs_dict = algo.get_example_obs()
    logger.info(f"Example obs keys: {example_obs_dict.keys()}")
    
    # Determine output path
    if output_path is None:
        output_dir = checkpoint.parent / "exported"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / checkpoint.name.replace('.pt', '.onnx')
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get body keys from config
    body_keys = config.robot.get('body_keys', ['lower_body', 'upper_body'])
    logger.info(f"Using body keys: {body_keys}")
    
    # Export to ONNX
    logger.info(f"Exporting to ONNX: {output_path}")
    export_multi_agent_decouple_policy_as_onnx(
        algo.inference_model,
        str(output_path.parent),
        output_path.name,
        example_obs_dict,
        body_keys
    )
    
    logger.info(f"✓ Successfully exported ONNX model to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert RL checkpoint to ONNX format for sim2sim deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (.pt)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for ONNX file. If not specified, saves to checkpoint_dir/exported/"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (auto-detected if not specified)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger.remove()
    logger.add(sys.stdout, level="INFO", colorize=True)
    
    output_path = convert_checkpoint_to_onnx(
        args.checkpoint,
        args.output,
        args.config
    )
    print(f"\n✓ Conversion complete!")
    print(f"  ONNX model saved to: {output_path}")
    print(f"\n  You can now use it with:")
    print(f"    --model_path={output_path}")


if __name__ == "__main__":
    main()

