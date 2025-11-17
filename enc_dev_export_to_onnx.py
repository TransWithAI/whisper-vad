#!/usr/bin/env python
"""Export trained encoder_only_decoder model to ONNX format."""

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from transformers import WhisperFeatureExtractor

from models.lightning_modules import EncoderOnlyVADModule


def export_to_onnx(
    checkpoint_path: str,
    config_path: str,
    output_path: str = None,
    opset_version: int = 17,
    batch_size: int = 1,
    verbose: bool = True,
):
    """Export encoder_only_decoder model to ONNX format.

    Args:
        checkpoint_path: Path to the trained model checkpoint
        config_path: Path to the config file used for training
        output_path: Path for the exported ONNX model (optional)
        opset_version: ONNX opset version (default: 17, recommended: 17-19)
        batch_size: Batch size for export (default: 1)
        verbose: Enable verbose export logging

    Note:
        Opset versions:
        - 17: Good compatibility, mature support (recommended)
        - 18: Better transformer support, widely compatible
        - 19: Latest stable with broad runtime support
        - 20+: Cutting edge, check runtime compatibility
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load the trained model
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = EncoderOnlyVADModule.load_from_checkpoint(
        checkpoint_path,
        model_type='lightweight_decoder',
        whisper_model_name=config['model']['whisper_model_name'],
        freeze_encoder=config['model'].get('freeze_encoder', False),
        decoder_layers=config['model'].get('decoder_layers', 2),
        decoder_heads=config['model'].get('decoder_heads', 8),
        dropout=0.0,  # Disable dropout for inference
    )

    # Set model to evaluation mode
    model.eval()
    model.to('cpu')  # Export on CPU for compatibility

    # Create dummy input for tracing
    # Whisper expects mel spectrogram features of shape (batch, 80, 3000) for 30s audio
    dummy_input = torch.randn(batch_size, 80, 3000)

    # Determine output path if not provided
    if output_path is None:
        checkpoint_dir = Path(checkpoint_path).parent
        model_name = Path(checkpoint_path).stem
        output_path = str(checkpoint_dir / f"{model_name}.onnx")

    # Export to ONNX
    print(f"Exporting model to ONNX: {output_path}")

    # We need to extract the actual model from Lightning module
    actual_model = model.model

    # Define dynamic axes for variable batch size
    dynamic_axes = {
        'input_features': {0: 'batch_size'},
        'frame_logits': {0: 'batch_size'},
    }

    # If the model also outputs clip logits, add them
    if hasattr(actual_model, 'clip_level') and actual_model.clip_level:
        dynamic_axes['clip_logits'] = {0: 'batch_size'}

    # Export to ONNX - using external_data=False to keep everything in one file
    # Note: Large models (>2GB) will always use external data format
    with torch.no_grad():
        # Check PyTorch version for external_data parameter support
        # Handle versions like '2.0.1+cu118' by splitting on '+' first
        version_str = torch.__version__.split('+')[0]
        torch_version = tuple(map(int, version_str.split('.')[:2]))

        export_kwargs = {
            'model': actual_model,
            'args': dummy_input,
            'f': output_path,
            'input_names': ['input_features'],
            'output_names': ['frame_logits', 'clip_logits'] if hasattr(actual_model, 'clip_level') and actual_model.clip_level else ['frame_logits'],
            'dynamic_axes': dynamic_axes,
            'opset_version': opset_version,
            'do_constant_folding': True,
            'export_params': True,
            'verbose': verbose,
        }

        # PyTorch 1.10+ supports external_data parameter
        if torch_version >= (1, 10):
            # Use external_data=False to force single file (if model < 2GB)
            # For models > 2GB, this will still create external data
            export_kwargs['external_data'] = False

        torch.onnx.export(**export_kwargs)

    print(f"✓ Model successfully exported to: {output_path}")

    # Save metadata alongside the ONNX model
    metadata = {
        'model_type': 'encoder_only_decoder',
        'whisper_model_name': config['model']['whisper_model_name'],
        'decoder_layers': config['model'].get('decoder_layers', 2),
        'decoder_heads': config['model'].get('decoder_heads', 8),
        'input_shape': [batch_size, 80, 3000],
        'output_shape': [batch_size, 1500],  # 1500 frames for 30s audio
        'frame_duration_ms': 20,  # Each frame represents 20ms
        'total_duration_ms': 30000,  # 30 seconds
        'opset_version': opset_version,
        'export_batch_size': batch_size,
        'config_path': config_path,
        'checkpoint_path': checkpoint_path,
    }

    metadata_path = output_path.replace('.onnx', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Metadata saved to: {metadata_path}")

    # Verify the exported model
    try:
        import onnxruntime as ort

        print("\nVerifying exported model...")
        session = ort.InferenceSession(output_path)

        # Get model info
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()

        print(f"  Input: {input_info.name}, shape: {input_info.shape}, dtype: {input_info.type}")
        for out in output_info:
            print(f"  Output: {out.name}, shape: {out.shape}, dtype: {out.type}")

        # Run inference test
        test_input = {input_info.name: dummy_input.numpy()}
        outputs = session.run(None, test_input)

        print(f"  Test inference successful!")
        print(f"  Output shape: {outputs[0].shape}")

    except ImportError:
        print("\n⚠ onnxruntime not installed. Skipping verification.")
        print("  Install with one of:")
        print("    uv pip install whisper-vad[onnx]      # For CPU inference")
        print("    uv pip install whisper-vad[onnx-gpu]  # For GPU inference")
        print("\n  Or directly:")
        print("    uv add onnxruntime      # For CPU")
        print("    uv add onnxruntime-gpu  # For GPU")
    except Exception as e:
        print(f"\n⚠ Failed to verify model: {e}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Export encoder_only_decoder model to ONNX')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to the trained model checkpoint (.ckpt file)'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the config file used for training'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for the ONNX model (default: same dir as checkpoint)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for export (default: 1)'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=17,
        help='ONNX opset version (default: 17, recommended: 17-19 for compatibility)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Disable verbose export logging'
    )

    args = parser.parse_args()

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return 1

    # Check if config exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return 1

    # Export model
    export_to_onnx(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_path=args.output,
        opset_version=args.opset,
        batch_size=args.batch_size,
        verbose=not args.quiet,
    )

    return 0


if __name__ == '__main__':
    exit(main())