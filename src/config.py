"""Configuration management for training."""

import argparse
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    name: str = "dummy"
    path: str = "./data"
    size: int = 1000
    split: list = field(default_factory=lambda: [0.8, 0.1, 0.1])


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    epochs: int = 10
    batch_size: int = 32
    num_workers: int = 0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    def __post_init__(self) -> None:
        """Convert string values to appropriate types and validate ranges."""
        # Normalize and validate learning_rate and weight_decay
        for field_name in ("learning_rate", "weight_decay"):
            value = getattr(self, field_name)

            # Convert from string if necessary
            if isinstance(value, str):
                try:
                    value = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid value for {field_name}: {value!r}. Expected a float."
                    ) from exc

            # Ensure the value is numeric
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"Invalid type for {field_name}: {type(value).__name__}. "
                    "Expected a float."
                )

            # Validate ranges
            if field_name == "learning_rate" and value <= 0.0:
                raise ValueError(
                    f"Invalid value for {field_name}: {value}. "
                    "Expected a positive float."
                )
            if field_name == "weight_decay" and value < 0.0:
                raise ValueError(
                    f"Invalid value for {field_name}: {value}. "
                    "Expected a non-negative float."
                )

            setattr(self, field_name, float(value))


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    input_dim: int = 4
    hidden_size_1: int = 64
    hidden_size_2: int = 32
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1


@dataclass
class PathsConfig:
    """Paths configuration."""

    runs_dir: str = "./runs"
    checkpoints_dir: str = "./checkpoints"
    data_dir: str = "./data"


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "json"  # json or text
    save_interval: int = 100


@dataclass
class Config:
    """Complete configuration."""

    seed: int = 42
    device: str = "mps"
    precision: str = "fp32"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data or {})

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        config_data = {
            "seed": data.get("seed", 42),
            "device": data.get("device", "mps"),
            "precision": data.get("precision", "fp32"),
            "dataset": DatasetConfig(**data.get("dataset", {})),
            "training": TrainingConfig(**data.get("training", {})),
            "model": ModelConfig(**data.get("model", {})),
            "paths": PathsConfig(**data.get("paths", {})),
            "logging": LoggingConfig(**data.get("logging", {})),
        }
        return cls(**config_data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def save(self, path: str) -> None:
        """Save configuration to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def __str__(self) -> str:
        """Pretty print configuration."""
        return yaml.dump(self.to_dict(), default_flow_style=False)


def parse_cli_args(overrides: Optional[list] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Delivery Time Prediction - Training script"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["mps", "cpu"],
        help="Device to use (mps, cpu)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "amp"],
        help="Precision (fp32, amp)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        dest="batch_size",
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        dest="learning_rate",
        help="Learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="exp",
        help="Tag for the run directory",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from",
    )

    args = parser.parse_args(overrides)
    return args


def load_config(cli_args: argparse.Namespace) -> Config:
    """Load config from YAML and apply CLI overrides."""
    # Load from YAML
    config = Config.from_yaml(cli_args.config)

    # Apply CLI overrides
    if cli_args.seed is not None:
        config.seed = cli_args.seed
    if cli_args.device is not None:
        config.device = cli_args.device
    if cli_args.precision is not None:
        config.precision = cli_args.precision
    if cli_args.batch_size is not None:
        config.training.batch_size = cli_args.batch_size
    if cli_args.learning_rate is not None:
        config.training.learning_rate = cli_args.learning_rate
    if cli_args.epochs is not None:
        config.training.epochs = cli_args.epochs

    return config
