"""Training script for delivery time prediction model."""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src import parse_cli_args, load_config, RunManager
from src.checkpoint import CheckpointManager
from src.data import create_delivery_dataloaders
from src.model import create_delivery_model, count_parameters
from src.train_utils import evaluate


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    criterion,
    device: str,
    epoch: int,
    run_manager,
) -> dict:
    """Train for one epoch.
    
    Args:
        model: The neural network model
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        epoch: Current epoch number
        run_manager: Run manager for logging
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    start_time = time.time()
    
    for batch_idx, (features, targets) in enumerate(dataloader):
        features = features.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    epoch_time = time.time() - start_time
    
    return {
        "loss": avg_loss,
        "time_seconds": epoch_time,
    }


def main():
    """Main training function for delivery time prediction."""
    # Parse CLI arguments
    args = parse_cli_args()

    # Load configuration
    config = load_config(args)

    # Create run manager
    run_manager = RunManager(config, tag=args.tag)

    # Log configuration
    run_manager.logger.info("=" * 60)
    run_manager.logger.info("Delivery Time Prediction - Training Configuration:")
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
    train_loader, val_loader, test_loader = create_delivery_dataloaders(
        csv_path=config.dataset.path,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        split=config.dataset.split,
    )
    run_manager.logger.info(f"Train batches: {len(train_loader)}")
    run_manager.logger.info(f"Val batches: {len(val_loader)}")
    run_manager.logger.info(f"Test batches: {len(test_loader)}")

    # Create model
    run_manager.logger.info("\nCreating model...")
    model = create_delivery_model(
        input_dim=config.model.input_dim,
        hidden_size_1=config.model.hidden_size_1,
        hidden_size_2=config.model.hidden_size_2,
        device=device,
    )
    num_params = count_parameters(model)
    run_manager.logger.info(f"Model parameters: {num_params:,}")
    run_manager.logger.info(f"Model architecture:\n{model}")

    # Create optimizer (using SGD to match notebook)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.training.learning_rate,
    )

    # Loss function (MSE for regression)
    criterion = nn.MSELoss()

    # Checkpoint manager
    checkpoint_manager = CheckpointManager(run_manager.get_checkpoint_dir())

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        run_manager.logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint_info = checkpoint_manager.load(
            Path(args.resume), model, optimizer, None
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
            criterion=criterion,
            device=device,
            epoch=epoch,
            run_manager=run_manager,
        )

        # Validate every 5000 epochs or on the last epoch
        if (
            (epoch + 1) % config.logging.save_interval == 0
            or epoch == config.training.epochs - 1
        ):
            val_metrics = evaluate(model, val_loader, criterion, device)

            # Log epoch metrics
            run_manager.logger.info(
                f"Epoch [{epoch + 1}/{config.training.epochs}] | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Time: {train_metrics['time_seconds']:.2f}s"
            )

            # Save metrics
            metrics = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_time_seconds": train_metrics["time_seconds"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            run_manager.log_metrics(metrics, step=epoch)

            # Save checkpoint if best model
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                checkpoint_manager.save(
                    model=model,
                    optimizer=optimizer,
                    scheduler=None,
                    epoch=epoch,
                    step=epoch,
                    metrics=metrics,
                    is_best=True,
                )
                run_manager.logger.info(
                    f"✓ New best model saved! Val Loss: {best_val_loss:.4f}"
                )

            # Save regular checkpoint
            checkpoint_manager.save(
                model=model,
                optimizer=optimizer,
                scheduler=None,
                epoch=epoch,
                step=epoch,
                metrics=metrics,
                is_best=False,
            )

    # Final evaluation on test set
    run_manager.logger.info("\n" + "=" * 60)
    run_manager.logger.info("Final Evaluation on Test Set")
    run_manager.logger.info("=" * 60)
    
    test_metrics = evaluate(model, test_loader, criterion, device)
    run_manager.logger.info(f"Test Loss (MSE): {test_metrics['loss']:.4f}")
    
    # Calculate RMSE and MAE for better interpretability
    with torch.no_grad():
        all_predictions = []
        all_targets = []
        for features, targets in test_loader:
            features = features.to(device)
            targets = targets.to(device)
            predictions = model(features)
            all_predictions.append(predictions)
            all_targets.append(targets)
        
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        
        mse = criterion(all_predictions, all_targets).item()
        rmse = mse ** 0.5
        mae = torch.mean(torch.abs(all_predictions - all_targets)).item()
        
    run_manager.logger.info(f"Test RMSE: {rmse:.4f} minutes")
    run_manager.logger.info(f"Test MAE: {mae:.4f} minutes")

    total_time = time.time() - training_start_time
    run_manager.logger.info(f"\nTotal training time: {total_time:.2f}s")
    run_manager.logger.info(f"Best validation loss: {best_val_loss:.4f}")
    
    # Log final test metrics
    final_metrics = {
        "epoch": config.training.epochs - 1,
        "test_loss": test_metrics["loss"],
        "test_rmse": rmse,
        "test_mae": mae,
        "total_training_time": total_time,
        "best_val_loss": best_val_loss,
    }
    run_manager.log_metrics(final_metrics, step=config.training.epochs)

    run_manager.logger.info("\n✓ Training complete!")
    run_manager.logger.info(f"Results saved to: {run_manager.run_dir}")


if __name__ == "__main__":
    main()
