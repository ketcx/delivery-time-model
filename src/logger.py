"""Structured logging for training."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from rich.logging import RichHandler


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def setup_logger(
    name: str,
    log_dir: Path,
    level: str = "INFO",
    format_type: str = "json",
) -> logging.Logger:
    """Set up a logger with both console and file handlers.

    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type (json or text)

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # File handler
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"

    if format_type == "json":
        file_formatter: logging.Formatter = JsonFormatter()
    else:
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, level.upper()))
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler (always text with Rich)
    console_handler = RichHandler(rich_tracebacks=True)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


class MetricsLogger:
    """Logger for metrics in JSONL format."""

    def __init__(self, log_file: Path):
        """Initialize metrics logger.

        Args:
            log_file: Path to metrics JSONL file
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to JSONL file.

        Args:
            metrics: Dictionary of metric values
            step: Optional step/epoch number
        """
        log_data = {
            "timestamp": datetime.now().isoformat(),
            **metrics,
        }
        if step is not None:
            log_data["step"] = step

        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_data) + "\n")

    def load(self) -> list[Dict[str, Any]]:
        """Load all logged metrics.

        Returns:
            List of metric dictionaries
        """
        if not self.log_file.exists():
            return []

        metrics = []
        with open(self.log_file) as f:
            for line in f:
                if line.strip():
                    metrics.append(json.loads(line))
        return metrics
