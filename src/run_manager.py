"""Run management for training experiments."""

from datetime import datetime
from pathlib import Path

from src.config import Config
from src.logger import MetricsLogger, setup_logger
from src.system import get_system_info


class RunManager:
    """Manages run directories and logging."""

    def __init__(self, config: Config, tag: str = "exp"):
        """Initialize run manager.

        Args:
            config: Training configuration
            tag: Tag for the run (used in directory name)
        """
        self.config = config
        self.tag = tag
        self.run_dir = self._create_run_dir()
        self.logger = setup_logger(
            "train",
            self.run_dir,
            level=config.logging.level,
            format_type=config.logging.format,
        )
        self.metrics_logger = MetricsLogger(self.run_dir / "metrics.jsonl")
        self.system_info = get_system_info(config.device)

        # Save configuration and system info
        self._save_metadata()

    def _create_run_dir(self) -> Path:
        """Create run directory with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{timestamp}_{self.tag}"
        run_dir = Path(self.config.paths.runs_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _save_metadata(self) -> None:
        """Save configuration and system info to run directory."""
        # Save resolved configuration
        self.config.save(str(self.run_dir / "config_resolved.yaml"))

        # Save system information
        self.system_info.save(str(self.run_dir / "system.json"))

        # Log initial information
        self.logger.info(f"Run directory: {self.run_dir}")
        self.logger.info(f"Device: {self.config.device}")
        self.logger.info(f"Git commit: {self.system_info.git_commit}")
        self.logger.info(f"Git branch: {self.system_info.git_branch}")

    def log_metrics(self, metrics: dict, step: int) -> None:
        """Log metrics for current step.

        Args:
            metrics: Dictionary of metric values
            step: Current step/epoch number
        """
        self.metrics_logger.log(metrics, step=step)

        # Also log to main logger every save_interval steps
        if step % self.config.logging.save_interval == 0:
            metrics_str = ", ".join(
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in metrics.items()
            )
            self.logger.info(f"Step {step} - {metrics_str}")

    def get_checkpoint_dir(self) -> Path:
        """Get checkpoint directory for this run."""
        checkpoint_dir = self.run_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        return checkpoint_dir

    def __repr__(self) -> str:
        """String representation."""
        return f"RunManager(run_dir={self.run_dir})"
