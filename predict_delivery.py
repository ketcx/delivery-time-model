"""Interactive script to predict delivery time from user inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import torch

from src.model import create_delivery_model


def _load_normalization_stats(csv_path: Path) -> Tuple[float, float, float, float]:
    """Load dataset and compute mean/std for distance and time_of_day.

    Args:
        csv_path: Path to the dataset CSV.

    Returns:
        Tuple of (distance_mean, distance_std, time_mean, time_std).
    """
    df = pd.read_csv(csv_path)
    distance_mean = float(df["distance_miles"].mean())
    distance_std = float(df["distance_miles"].std())
    time_mean = float(df["time_of_day_hours"].mean())
    time_std = float(df["time_of_day_hours"].std())
    return distance_mean, distance_std, time_mean, time_std


def _rush_hour_feature(time_of_day: float, is_weekend: int) -> float:
    """Compute rush hour feature.

    Rush hour is defined as:
    - Morning: 8:00-10:00 AM (8.0-10.0)
    - Evening: 4:00-7:00 PM (16.0-19.0)
    - Only on weekdays (is_weekend == 0)
    """
    is_morning_rush = 8.0 <= time_of_day < 10.0
    is_evening_rush = 16.0 <= time_of_day < 19.0
    is_weekday = is_weekend == 0
    return float(is_weekday and (is_morning_rush or is_evening_rush))


def _normalize(value: float, mean: float, std: float) -> float:
    """Normalize a value with mean and std."""
    if std == 0:
        return 0.0
    return (value - mean) / std


def _prompt_float(prompt: str) -> float:
    """Prompt user for a float value."""
    while True:
        raw = input(prompt).strip()
        try:
            return float(raw)
        except ValueError:
            print("Invalid number. Please try again.")


def _prompt_int(prompt: str, allowed: Tuple[int, ...]) -> int:
    """Prompt user for an int value from allowed options."""
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
        except ValueError:
            print("Invalid integer. Please try again.")
            continue
        if value not in allowed:
            print(f"Invalid option. Choose one of: {allowed}")
            continue
        return value


def main() -> None:
    """Run interactive prediction flow."""
    print("=== Delivery Time Prediction ===")

    csv_default = Path("data/data_with_features.csv")
    csv_input = input(f"Dataset CSV path [{csv_default}]: ").strip()
    csv_path = Path(csv_input) if csv_input else csv_default

    if not csv_path.exists():
        print(f"Dataset not found: {csv_path}")
        return

    checkpoint_default = Path("runs")
    ckpt_input = input(
        "Checkpoint path (e.g., runs/<run>/checkpoints/best.pt): "
    ).strip()
    ckpt_path = Path(ckpt_input) if ckpt_input else checkpoint_default

    if ckpt_path.is_dir():
        print("Please provide a specific checkpoint file, not a directory.")
        return
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return

    distance_miles = _prompt_float("Distance (miles, <= 20): ")
    time_of_day = _prompt_float("Time of day (hours, 8.0 - 20.0): ")
    is_weekend = _prompt_int("Weekend? (0=weekday, 1=weekend): ", (0, 1))

    (
        distance_mean,
        distance_std,
        time_mean,
        time_std,
    ) = _load_normalization_stats(csv_path)
    norm_distance = _normalize(distance_miles, distance_mean, distance_std)
    norm_time = _normalize(time_of_day, time_mean, time_std)
    rush_hour = _rush_hour_feature(time_of_day, is_weekend)

    features = torch.tensor(
        [[norm_distance, norm_time, float(is_weekend), rush_hour]],
        dtype=torch.float32,
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = create_delivery_model(device=device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    if "model" in checkpoint:
        model_state = checkpoint["model"]
    elif "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
    else:
        print(
            "Checkpoint does not contain model weights under key "
            "'model' or 'model_state_dict'."
        )
        return
    model.load_state_dict(model_state)
    model.eval()

    with torch.no_grad():
        prediction = model(features.to(device)).cpu().item()

    print("\n=== Prediction Result ===")
    print(f"Estimated delivery time: {prediction:.2f} minutes")


if __name__ == "__main__":
    main()
