"""Training script with full PyTorch pipeline."""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import parse_cli_args, load_config, RunManager
from src.checkpoint import CheckpointManager
from src.data import create_dataloaders
from src.model import create_model, count_parameters
from src.train_utils import (
    evaluate,
    get_cosine_schedule_with_warmup,
    train_epoch,
)


def main():
    """Main training function with full pipeline."""
    # Parse CLI arguments
    args = parse_cli_args()

    # Load configuration
    config = load_config(args)

    # Create run manager
    run_manager = RunManager(config, tag=args.tag)

    # Log configuration
    run_manager.logger.info("=" * 60)
    run_manager.logger.info("Training Configuration:")
    run_manager.logger.info("=" * 60)
    run_manager.logger.info(str(config))

    # Set random seeds for reproducibility
    torch.manual_seed(config.seed)

    # Determine device
    if config.device == "mps" and torch.backends.mps.is_available():
        device = "mps"
        run_manager.logger.info("Using device: MPS (Apple Silicon)")
    else:
        device = "cpu"
        run_manager.logger.info("Using device: CPU")

    # Create dataloaders
    run_manager.logger.info("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_size=config.dataset.size,
        input_dim=10,
        num_classes=2,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        split=config.dataset.split,
    )
    run_manager.logger.info(f"Train batches: {len(train_loader)}")
    run_manager.logger.info(f"Val batches: {len(val_loader)}")
    run_manager.logger.info(f"Test batches: {len(test_loader)}")

    # Create model
    run_manager.logger.info("\nCreating model...")
    model = create_model(
        input_dim=10,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        num_classes=2,
        dropout=config.model.dropout,
        device=device,
    )
    num_params = count_parameters(model)
    run_manager.logger.info(f"Model parameters: {num_params:,}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Create scheduler
    num_training_steps = len(train_loader) * config.training.epochs
    num_warmup_steps = len(train_loader)  # 1 epoch warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Checkpoint manager
    checkpoint_manager = CheckpointManager(run_manager.get_checkpoint_dir())

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        run_manager.logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint_info = checkpoint_manager.load(
            Path(args.resume), model, optimizer, scheduler
        )
        start_epoch = checkpoint_info["epoch"] + 1
        run_manager.logger.info(f"Resumed from epoch {checkpoint_info['epoch']}")

    # Training loop
    run_manager.logger.info("\n" + "=" * 60)
    run_manager.logger.info("Starting training...")
    run_manager.logger.info("=" * 60)

    best_val_loss = float("inf")
    training_start_time = time.time()

    for epoch in range(start_epoch, config.training.epochs):
        # Train epoch
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            epoch=epoch,
            run_manager=run_manager,
        )

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Log epoch metrics
        run_manager.logger.info(
            f"Epoch {epoch + 1}/{config.training.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Time: {train_metrics['time_seconds']:.2f}s"
        )

        # Save metrics
        metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "train_time_seconds": train_metrics["time_seconds"],
        }
        run_manager.log_metrics(metrics, step=epoch)

        # Save checkpoint
        is_best = val_metrics["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["loss"]

        checkpoint_manager.save(
            epoch=epoch,
            step=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=metrics,
            is_best=is_best,
        )

        # Cleanup old checkpoints periodically to reduce filesystem overhead
        cleanup_interval = 5  # Cleanup every 5 epochs
        if (epoch + 1) % cleanup_interval == 0:
            checkpoint_manager.cleanup_old_checkpoints(keep_last_n=3)

    # Training finished
    total_time = time.time() - training_start_time
    run_manager.logger.info("\n" + "=" * 60)
    run_manager.logger.info("Training finished!")
    run_manager.logger.info(f"Total time: {total_time / 60:.2f} minutes")
    run_manager.logger.info("=" * 60)

    # Evaluate on test set
    run_manager.logger.info("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader, criterion, device)
    run_manager.logger.info(
        f"Test Loss: {test_metrics['loss']:.4f} | "
        f"Test Acc: {test_metrics['accuracy']:.4f}"
    )

    # Final metrics
    final_metrics = {
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "total_time_minutes": total_time / 60,
    }
    run_manager.log_metrics(final_metrics, step=config.training.epochs)

    run_manager.logger.info(f"Results saved to: {run_manager.run_dir}")


if __name__ == "__main__":
    main()
