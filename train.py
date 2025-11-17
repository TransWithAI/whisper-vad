"""
Main training script for Whisper VAD models.

Supports both Refinement I (Encoder-Only) and Refinement II (DETR-style) models.
"""

import os
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings("ignore")

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
    RichModelSummary
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from models import EncoderOnlyVADModule, DETRVADModule, WhisperSegLightning
from utils.dataset import create_dataloaders
from utils.whisperseg_data import WhisperSegDataModule
from utils.metrics import VADMetrics


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_callbacks(config: Dict[str, Any], run_name: str):
    """Create training callbacks."""
    callbacks = []

    # Model checkpoint
    monitor_metric = config.get('monitor_metric', 'val/loss')
    # Format the metric name for filename (replace / with _)
    metric_name = monitor_metric.replace('/', '_')

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{run_name}",
        filename=f"{{epoch:02d}}-{{{monitor_metric}:.4f}}",
        monitor=monitor_metric,
        mode=config.get('monitor_mode', 'min'),
        save_top_k=config.get('save_top_k', 3),
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    if config.get('early_stopping', {}).get('enabled', True):
        early_stopping = EarlyStopping(
            monitor=monitor_metric,  # Use the same metric as checkpoint
            patience=config['early_stopping'].get('patience', 10),
            mode=config.get('monitor_mode', 'min'),
            verbose=True
        )
        callbacks.append(early_stopping)

    # Learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval='step'))

    # Progress bar
    callbacks.append(RichProgressBar())

    # Model summary
    callbacks.append(RichModelSummary(max_depth=2))

    return callbacks


def create_logger(config: Dict[str, Any], run_name: str):
    """Create experiment logger."""
    logger_type = config.get('logger', 'tensorboard')

    if logger_type == 'wandb':
        logger = WandbLogger(
            project=config.get('project_name', 'whisper-vad'),
            name=run_name,
            save_dir='logs',
            log_model=True
        )
    else:
        logger = TensorBoardLogger(
            save_dir='logs',
            name=run_name,
            version=config.get('experiment_name', run_name)
        )

    return logger


def main(args):
    """Main training function."""
    # Load configuration
    config = load_config(args.config)

    # Set seed for reproducibility
    pl.seed_everything(config.get('seed', 42))

    data_config = config.get('data', {})
    training_config = config.get('training', {})
    model_config = config.get('model', {})
    loss_config = config.get('loss', {})
    validation_config = config.get('validation', {})

    train_loader = val_loader = None
    datamodule = None
    model: Optional[pl.LightningModule] = None
    model_name = args.model
    experiment_name = config.get('experiment_name')
    dataset_path = data_config.get('dataset_path')
    if dataset_path is None:
        raise ValueError('`data.dataset_path` must be set in the config file')

    if args.model == 'whisperseg':
        datamodule = WhisperSegDataModule(
            dataset_name_or_path=dataset_path,
            model_name_or_path=model_config.get('whisper_model_name', 'openai/whisper-base'),
            train_split=data_config.get('train_split', 'train'),
            val_split=data_config.get('val_split', 'validation'),
            test_split=data_config.get('test_split', 'test'),
            batch_size=training_config.get('batch_size', 8),
            num_workers=data_config.get('num_workers', 4),
            total_spec_columns=data_config.get('total_spec_columns', 1000),
            spec_time_step=data_config.get('spec_time_step', 0.01),
            sampling_rate=data_config.get('sampling_rate', 16000),
            spec_augment=data_config.get('spec_augment', True),
            time_mask_param=data_config.get('time_mask_param', 10),
            freq_mask_param=data_config.get('freq_mask_param', 27),
            species=data_config.get('species', 'human'),
            cluster_codebook=data_config.get('cluster_codebook'),
            ignore_cluster=data_config.get('ignore_cluster', False),
            cache_dir=data_config.get('cache_dir'),
            streaming=data_config.get('streaming', False),
            preprocessing_num_workers=data_config.get('preprocessing_num_workers', 4),
            train_samples=data_config.get('max_train_samples'),
            val_samples=data_config.get('max_val_samples'),
            test_samples=data_config.get('max_test_samples'),
        )

        inferred_training_steps = training_config.get('num_training_steps')
        if inferred_training_steps is None:
            try:
                datamodule.setup('fit')
                steps_per_epoch = len(datamodule.train_dataloader())
                inferred_training_steps = steps_per_epoch * training_config.get('max_epochs', 1)
            except TypeError:
                inferred_training_steps = None

        model = WhisperSegLightning(
            model_name_or_path=model_config.get('whisper_model_name', 'openai/whisper-base'),
            learning_rate=training_config.get('learning_rate', 3e-6),
            warmup_steps=training_config.get('warmup_steps', 500),
            weight_decay=training_config.get('weight_decay', 0.01),
            dropout=model_config.get('dropout', 0.1),
            total_spec_columns=data_config.get('total_spec_columns', 1000),
            freeze_encoder=model_config.get('freeze_encoder', False),
            gradient_checkpointing=model_config.get('gradient_checkpointing', False),
            label_smoothing=loss_config.get('label_smoothing', 0.0),
            scheduler_type=training_config.get('scheduler', 'linear'),
            num_training_steps=inferred_training_steps,
            ignore_cluster=data_config.get('ignore_cluster', False),
            cluster_codebook=data_config.get('cluster_codebook'),
            val_max_gen_length=validation_config.get('max_generation_length', 256),
            val_early_stopping=validation_config.get('early_stopping', True),
            val_num_beams=validation_config.get('num_beams', 1),
            val_batch_metrics=validation_config.get('parallel_metrics', True),
            val_use_cache=validation_config.get('use_cache', True),
        )
        model_name = 'whisperseg'
    else:
        train_loader, val_loader = create_dataloaders(
            dataset_path=dataset_path,
            batch_size=training_config.get('batch_size', 8),
            num_workers=data_config.get('num_workers', 4),
            whisper_model_name=model_config.get('whisper_model_name', 'openai/whisper-base'),
            max_train_samples=data_config.get('max_train_samples'),
            max_val_samples=data_config.get('max_val_samples'),
            streaming=data_config.get('streaming', False),
            train_dataset_length=data_config.get('train_dataset_length'),
            val_dataset_length=data_config.get('val_dataset_length')
        )

    # Initialize model
    if args.model == 'whisperseg':
        pass  # Model already initialized above

    elif args.model == 'encoder_only_linear':
        model = EncoderOnlyVADModule(
            model_type='linear',
            whisper_model_name=config['model']['whisper_model_name'],
            freeze_encoder=config['model'].get('freeze_encoder', False),
            learning_rate=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 1e-5),
            warmup_epochs=config['training'].get('warmup_epochs', 2),
            max_epochs=config['training'].get('max_epochs', 50),
            focal_alpha=config['loss'].get('focal_alpha', 0.25),
            focal_gamma=config['loss'].get('focal_gamma', 2.0),
            dropout=config['model'].get('dropout', 0.1)
        )
        model_name = 'encoder_only_linear'

    elif args.model == 'encoder_only_decoder':
        model = EncoderOnlyVADModule(
            model_type='lightweight_decoder',
            whisper_model_name=config['model']['whisper_model_name'],
            freeze_encoder=config['model'].get('freeze_encoder', False),
            learning_rate=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 1e-5),
            warmup_epochs=config['training'].get('warmup_epochs', 2),
            max_epochs=config['training'].get('max_epochs', 50),
            focal_alpha=config['loss'].get('focal_alpha', 0.25),
            focal_gamma=config['loss'].get('focal_gamma', 2.0),
            decoder_layers=config['model'].get('decoder_layers', 2),
            decoder_heads=config['model'].get('decoder_heads', 8),
            decoder_ff_dim=config['model'].get('decoder_ff_dim', 2048),
            dropout=config['model'].get('dropout', 0.1)
        )
        model_name = 'encoder_only_decoder'

    elif args.model == 'detr':
        model = DETRVADModule(
            whisper_model_name=model_config.get('whisper_model_name'),
            num_queries=model_config.get('num_queries', 20),
            decoder_layers=model_config.get('decoder_layers', 3),
            decoder_heads=model_config.get('decoder_heads', 8),
            freeze_encoder=model_config.get('freeze_encoder', False),
            learning_rate=training_config.get('learning_rate'),
            weight_decay=training_config.get('weight_decay', 1e-5),
            warmup_epochs=training_config.get('warmup_epochs', 5),
            max_epochs=training_config.get('max_epochs', 100),
            cost_class=loss_config.get('cost_class', 1.0),
            cost_span=loss_config.get('cost_span', 5.0),
            cost_giou=loss_config.get('cost_giou', 2.0),
            dropout=model_config.get('dropout', 0.1)
        )
        model_name = 'detr_vad'

    else:
        raise ValueError(f"Unknown model type: {args.model}")

    if model is None:
        raise RuntimeError("Model failed to initialize")

    experiment_name = experiment_name or model_name

    torch.set_float32_matmul_precision('medium')

    # Create callbacks
    callbacks = create_callbacks(config, experiment_name)

    # Create logger
    logger = create_logger(config, experiment_name)

    # Create trainer
    trainer = Trainer(
        max_epochs=config['training'].get('max_epochs', 50),
        accelerator=config.get('accelerator', 'auto'),
        devices=config.get('devices', 1),
        precision=config.get('precision', 16),
        gradient_clip_val=config['training'].get('gradient_clip_val', 1.0),
        accumulate_grad_batches=config['training'].get('accumulate_grad_batches', 1),
        val_check_interval=config['training'].get('val_check_interval', 1.0),
        log_every_n_steps=config['training'].get('log_every_n_steps', 10),
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=True,
        deterministic=True
    )

    # Train model
    if datamodule is not None:
        if args.resume_from:
            trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume_from)
        else:
            trainer.fit(model, datamodule=datamodule)

        if args.test:
            trainer.test(model, datamodule=datamodule)
    else:
        if args.resume_from:
            trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_from)
        else:
            trainer.fit(model, train_loader, val_loader)

        if args.test:
            trainer.test(model, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Whisper VAD models")

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['encoder_only_linear', 'encoder_only_decoder', 'detr', 'whisperseg'],
        help='Model architecture to train'
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )

    parser.add_argument(
        '--resume_from',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Run test after training'
    )

    args = parser.parse_args()
    main(args)