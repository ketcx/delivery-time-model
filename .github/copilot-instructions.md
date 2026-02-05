# Delivery Time Prediction Model - GitHub Copilot Instructions

## Project Overview

**distance_model** is a production-ready PyTorch regression model for predicting delivery times based on distance, time of day, and day of week.

**Purpose**: Implement a neural network with feature engineering (rush hour detection) to predict delivery times in minutes, using enterprise-grade ML development standards with emphasis on reproducibility and code quality.

**Task Type**: Regression (predicting continuous values - delivery time in minutes)

**Model Architecture**: 
- Input: 4 features (normalized distance, normalized time_of_day, is_weekend, rush_hour)
- Hidden Layer 1: Linear(4 → 64) + ReLU
- Hidden Layer 2: Linear(64 → 32) + ReLU  
- Output: Linear(32 → 1) - predicted delivery time

**Loss Function**: MSE (Mean Squared Error) for regression  
**Optimizer**: SGD with learning rate 0.01  
**Training**: 30,000 epochs on 100 delivery samples

---

## Code Quality Standards

### Mandatory Quality Gates

ALL code must pass these checks before commit:

```bash
make lint      # ruff - Fast Python linter
make typecheck # mypy - Static type checking
make format    # black - Code formatting (line length: 100)
make test      # pytest - Unit tests
```

#### Linting (ruff)

- **No unused imports** - Remove all `import X` that aren't used
- **No undefined names** - All variables must be defined
- **Import ordering**:
  1. Standard library (e.g., `import os`, `import json`)
  2. Third-party (e.g., `import torch`, `import pandas`)
  3. Local imports (e.g., `from src.config import Config`)
- **No trailing whitespace** - Clean line endings

#### Type Checking (mypy)

- **All function signatures must have type hints**:
  ```python
  def train_epoch(
      model: nn.Module,
      dataloader: DataLoader,
      device: str,
  ) -> Dict[str, float]:
  ```
- **Dict/List type hints are required**:
  ```python
  # GOOD
  results: Dict[str, float] = {}
  items: List[int] = []
  
  # BAD - no type hints
  results = {}
  items = []
  ```

#### Code Formatting (black)

- **Line length**: 100 characters (NOT 88, that's default)
- **Quotes**: Use double quotes `"` (black standard)
- **Spacing**: Black auto-formats, just run `make format`

### Running Quality Checks

**Before every commit**:
```bash
make all       # Check everything
make format    # Auto-format code
```

---

## Project Structure

```
distance_model/
├── src/                    # Main codebase
│   ├── __init__.py        # Public API exports
│   ├── config.py          # Configuration management (YAML → CLI override)
│   ├── train.py           # Original training script (classification)
│   ├── train_utils.py     # Training loop utilities, evaluation
│   ├── model.py           # DeliveryTimeModel + SimpleMLP architectures
│   ├── data.py            # DeliveryDataset with feature engineering
│   ├── checkpoint.py      # Save/load checkpoints (model + optimizer + metadata)
│   ├── logger.py          # Structured logging (JSON + rich console)
│   ├── run_manager.py     # Run directory management (runs/YYYYMMDD_HHMMSS_tag/)
│   ├── system.py          # System info collection (Python, torch, git, device)
│   ├── device.py          # MPS/CPU device management with fallback
│   └── benchmark.py       # Performance benchmarking (MPS vs CPU)
├── train_delivery.py      # Main training script for delivery time prediction
├── data/
│   └── data_with_features.csv  # Delivery dataset (100 samples)
├── configs/
│   └── train.yaml         # Training config (30k epochs, SGD, model params)
├── scripts/
│   └── benchmark.py       # Executable: `python scripts/benchmark.py`
├── tests/                 # Unit tests (pytest)
├── Makefile               # Quality gates & task automation
├── requirements.txt       # Python dependencies
├── README_DELIVERY.md     # Delivery model documentation
└── .github/
    └── copilot-instructions.md  # This file
```

---

## Key Modules and APIs

### Delivery Dataset (`src/data.py`)

**Purpose**: Load CSV data with automatic feature engineering (rush hour detection).

```python
from src.data import DeliveryDataset, create_delivery_dataloaders

# Create dataset from CSV
dataset = DeliveryDataset(csv_path="data/data_with_features.csv")
# Automatically:
# 1. Loads CSV with pandas
# 2. Engineers rush_hour feature (weekday mornings 8-10am, evenings 4-7pm)
# 3. Normalizes distance and time_of_day
# 4. Converts to PyTorch tensors

# Or use dataloaders directly
train_loader, val_loader, test_loader = create_delivery_dataloaders(
    csv_path="data/data_with_features.csv",
    batch_size=32,
    split=(0.8, 0.1, 0.1),  # train/val/test
)
```

**Dataset Features**:
- `distance_miles`: Normalized distance (mean=0, std=1)
- `time_of_day_hours`: Normalized time (24-hour format, mean=0, std=1)
- `is_weekend`: Binary (0=weekday, 1=weekend)
- `rush_hour`: Engineered binary (1=weekday rush hour, 0=otherwise)

**Target Variable**:
- `delivery_time_minutes`: Continuous value (target for regression)

**Rush Hour Logic**:
```python
# Morning rush: 8:00-10:00 AM on weekdays
# Evening rush: 4:00-7:00 PM (16:00-19:00) on weekdays
# Weekends never have rush hour
is_morning_rush = (hours >= 8.0) & (hours < 10.0)
is_evening_rush = (hours >= 16.0) & (hours < 19.0)
is_weekday = (weekend_flag == 0)
rush_hour = is_weekday & (is_morning_rush | is_evening_rush)
```

**Data Normalization**:
```python
# Applied to continuous features only
distances_norm = (distances - mean) / std
time_norm = (time - mean) / std
# Binary features (is_weekend, rush_hour) are NOT normalized
```

### Delivery Model (`src/model.py`)

**Purpose**: Neural network for delivery time regression.

```python
from src.model import DeliveryTimeModel, create_delivery_model

# Create model
model = create_delivery_model(
    input_dim=4,          # 4 features
    hidden_size_1=64,     # First hidden layer
    hidden_size_2=32,     # Second hidden layer
    device="mps",         # or "cpu"
)

# Forward pass
features = torch.tensor([[norm_dist, norm_time, is_weekend, rush_hour]])
prediction = model(features)  # Returns: delivery time in minutes (shape: [1, 1])
```

**Architecture Details**:
```python
nn.Sequential(
    nn.Linear(4, 64),    # Input layer: 4 features → 64 neurons
    nn.ReLU(),           # Activation
    nn.Linear(64, 32),   # Hidden layer: 64 → 32 neurons
    nn.ReLU(),           # Activation
    nn.Linear(32, 1),    # Output layer: 32 → 1 prediction
)
```

**Key Differences from SimpleMLP**:
- **No dropout** (regression typically doesn't use dropout like classification)
- **Output dimension = 1** (single continuous value, not multiple classes)
- **Fixed architecture** (not configurable num_layers parameter)
- **No softmax** (regression outputs raw continuous values)

### Configuration (`src/config.py`)

**Purpose**: Load training config from YAML with CLI overrides.

```python
from src import parse_cli_args, load_config

args = parse_cli_args()  # --config, --lr, --epochs, etc.
config = load_config(args)
```

**Config structure for delivery model**:
```python
config.seed                      # int: 41 (matching notebook seed)
config.device                    # str: "mps" or "cpu"
config.precision                # str: "fp32" (recommended)
config.training.epochs          # int: 30000
config.training.batch_size      # int: 32
config.training.learning_rate   # float: 0.01 (SGD learning rate)
config.training.weight_decay    # float: 0.0 (no regularization)
config.model.input_dim          # int: 4
config.model.hidden_size_1      # int: 64
config.model.hidden_size_2      # int: 32
config.dataset.path             # str: "./data/data_with_features.csv"
config.dataset.split            # list: [0.8, 0.1, 0.1]
config.logging.save_interval    # int: 5000 (log every N epochs)
```

### Main Training Script (`train_delivery.py`)

**Purpose**: End-to-end training for delivery time prediction.

```bash
# Train with default config
python train_delivery.py --config configs/train.yaml

# Override parameters
python train_delivery.py --config configs/train.yaml --lr 0.005 --epochs 10000

# Use CPU instead of MPS
python train_delivery.py --config configs/train.yaml --device cpu

# Resume from checkpoint
python train_delivery.py --config configs/train.yaml --resume runs/<run_name>/checkpoints/best.pt
```

**Training Process**:
1. Loads CSV and prepares data with feature engineering
2. Creates train/val/test splits (80/10/10)
3. Initializes DeliveryTimeModel
4. Trains for 30,000 epochs with SGD optimizer
5. Validates every 5,000 epochs
6. Saves checkpoints (best model + regular checkpoints)
7. Final evaluation with MSE, RMSE, MAE metrics

**Training Output Example**:
```
Epoch [5000/30000] | Train Loss: 3.0901 | Val Loss: 3.2145 | Time: 2.34s
✓ New best model saved! Val Loss: 3.2145
Epoch [10000/30000] | Train Loss: 1.6064 | Val Loss: 1.7823 | Time: 2.41s
...
Test Loss (MSE): 1.5234
Test RMSE: 1.2343 minutes
Test MAE: 0.9876 minutes
✓ Training complete!
```

**Key Metrics**:
- **MSE** (Mean Squared Error): Loss function, penalizes large errors heavily
- **RMSE** (Root Mean Squared Error): √MSE, in same units as target (minutes)
- **MAE** (Mean Absolute Error): Average absolute error, more interpretable

### Training Utilities (`src/train_utils.py`)

**Purpose**: Evaluation function (used by train_delivery.py).

```python
from src import evaluate

# Validate model (no gradients)
val_metrics = evaluate(model, val_loader, nn.MSELoss(), device="mps")
# Returns: {'loss': float} - MSE loss
```

**Note**: `train_delivery.py` has its own training loop optimized for regression:
- No accuracy metric (regression doesn't have "accuracy")
- Custom epoch training with MSE loss
- Additional metrics (RMSE, MAE) calculated separately

### Checkpointing (`src/checkpoint.py`)

**Purpose**: Save/load full training state (model + optimizer + metadata).

```python
from src import CheckpointManager

ckpt_manager = CheckpointManager(checkpoint_dir=Path("runs/<run_name>/checkpoints"))

# Save checkpoint
ckpt_manager.save(
    epoch=10000,
    model=model,
    optimizer=optimizer,
    scheduler=None,  # No scheduler in delivery model
    metrics={"val_loss": 1.7823, "train_loss": 1.6064},
    is_best=True,  # Also saves to "best.pt"
)

# Load checkpoint
info = ckpt_manager.load(
    checkpoint_path=Path("runs/<run_name>/checkpoints/best.pt"),
    model=model,
    optimizer=optimizer,
    scheduler=None,
)
start_epoch = info["epoch"] + 1
```

### Run Manager (`src/run_manager.py`)

**Purpose**: Create timestamped run directories with auditable logs.

```python
from src import RunManager

run_manager = RunManager(config=config, tag="my-delivery-run")
# Creates: runs/20260205_143022_my-delivery-run/

# Log metrics
run_manager.log_metrics(
    metrics={"train_loss": 1.6064, "val_loss": 1.7823, "epoch": 10000}
)

# Get checkpoint directory
checkpoint_dir = run_manager.get_checkpoint_dir()
# Returns: runs/20260205_143022_my-delivery-run/checkpoints/

# Saved artifacts:
# - config_resolved.yaml: Final config used
# - system.json: Python version, torch version, git commit, device info
# - metrics.jsonl: One JSON object per line (metrics over time)
# - checkpoints/: Model checkpoints
```

### Device Management (`src/device.py`)

**Purpose**: MPS availability check with CPU fallback.

```python
from src import is_mps_available, get_device

# Smart device selection
device = get_device(preferred_device="mps", fallback_on_error=True)
# Returns "mps" if available on M1/M2/M3 Mac, else "cpu"

# Check explicitly
if is_mps_available():
    print("✓ MPS available - using Apple Silicon acceleration")
else:
    print("Using CPU")
```

---

## Development Workflow

### 1. **Setup Environment**

```bash
# Navigate to project
cd distance_model

# Activate virtual environment
source ../.venv/bin/activate  # or wherever your venv is

# Install dependencies (if needed)
pip install -r requirements.txt
```

### 2. **Train the Model**

```bash
# Default training (30k epochs)
python train_delivery.py --config configs/train.yaml

# Quick test run (fewer epochs)
python train_delivery.py --config configs/train.yaml --epochs 1000

# Custom learning rate
python train_delivery.py --config configs/train.yaml --lr 0.005
```

### 3. **Monitor Training**

```bash
# Watch metrics in real-time
tail -f runs/<latest_run>/metrics.jsonl

# View final results
cat runs/<latest_run>/metrics.jsonl | grep test
```

### 4. **Load and Use Trained Model**

```python
import torch
from src.model import create_delivery_model

# Load model
model = create_delivery_model(device='cpu')
checkpoint = torch.load('runs/<run_name>/checkpoints/best.pt', map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()

# Make prediction
# (You need normalization params from training data)
with torch.no_grad():
    features = torch.tensor([[norm_distance, norm_time, is_weekend, rush_hour]])
    predicted_time = model(features)
    print(f"Predicted delivery time: {predicted_time.item():.2f} minutes")
```

### 5. **Write New Code**

Follow these patterns:

```python
"""Module docstring - what this does."""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn

from src.config import Config


def process_delivery_data(
    csv_path: str | Path,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process delivery data from CSV.
    
    Args:
        csv_path: Path to CSV file
        normalize: Whether to normalize continuous features
        
    Returns:
        Tuple of (features, targets) tensors
    """
    df = pd.read_csv(csv_path)
    # ... implementation
    return features, targets
```

### 6. **Run Quality Checks**

```bash
# Format code
make format

# Check linting
make lint

# Check types
make typecheck

# Run tests
make test

# Or all at once
make all
```

### 7. **Commit Changes**

```bash
git add .
git commit -m "Add feature: improved rush hour detection

- Extended evening rush hour to 7 PM
- Added docstrings to helper functions
- Updated tests for new logic"
```

---

## Data Format and Business Rules

### CSV Structure

The `data/data_with_features.csv` file contains:

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| distance_miles | float | 0-20 | Delivery distance in miles |
| time_of_day_hours | float | 8.0-20.0 | Dispatch time (24-hour format) |
| is_weekend | int | 0 or 1 | 0=weekday, 1=weekend |
| delivery_time_minutes | float | >0 | Actual delivery time (target) |

### Business Rules

- **Deliveries only occur between 8:00 AM and 8:00 PM** (8.0 to 20.0)
- **Maximum delivery distance is 20 miles**
- **Rush hour only applies to weekdays**:
  - Morning: 8:00-10:00 AM (8.0-10.0)
  - Evening: 4:00-7:00 PM (16.0-19.0)
- **Dataset size**: 100 historical delivery records

### Feature Engineering Rules

When adding the `rush_hour` feature:
```python
# Must be weekday (is_weekend == 0)
# AND either morning rush (8-10) OR evening rush (16-19)
is_rush = (is_weekend == 0) & ((8 <= time < 10) | (16 <= time < 19))
```

---

## Mac M1/M2/M3 Best Practices

### Device Selection

```python
# In training code
device = "mps" if torch.backends.mps.is_available() else "cpu"

# For this small model (4→64→32→1), CPU and MPS perform similarly
# MPS becomes advantageous with larger batch sizes or models
```

### Data Types

```python
# Recommended for M1/M2/M3
precision = "fp32"  # Stable, safest choice

# Avoid for this model
precision = "fp16"  # Can cause instability in small models
```

### Memory Management

```python
# Use no_grad() for inference
with torch.no_grad():
    output = model(batch)

# Clear gradients before backward pass
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## Testing

### Run Tests

```bash
make test  # Run all tests with pytest
pytest tests/test_model.py  # Run specific test file
pytest tests/test_data.py::test_rush_hour -v  # Run specific test
```

### Write Tests for Delivery Model

```python
# tests/test_delivery_model.py
import pytest
import torch
from src.model import create_delivery_model
from src.data import DeliveryDataset

def test_delivery_model_forward():
    """Test model forward pass with 4 features."""
    model = create_delivery_model(device="cpu")
    batch = torch.randn(32, 4)  # 32 samples, 4 features
    
    output = model(batch)
    
    assert output.shape == (32, 1)  # batch_size, 1 output
    
def test_rush_hour_feature():
    """Test rush hour detection logic."""
    from src.data import DeliveryDataset
    
    # Weekday morning rush (should be 1)
    hours = torch.tensor([9.0])
    weekends = torch.tensor([0.0])
    dataset = DeliveryDataset("data/data_with_features.csv")
    rush = dataset._rush_hour_feature(hours, weekends)
    assert rush.item() == 1.0
    
    # Weekend (should be 0, even in rush hours)
    hours = torch.tensor([9.0])
    weekends = torch.tensor([1.0])
    rush = dataset._rush_hour_feature(hours, weekends)
    assert rush.item() == 0.0
```

---

## Common Issues & Solutions

### Issue: "Model not learning / Loss not decreasing"

**Solutions**:
1. Check learning rate (try 0.01, 0.005, 0.001)
2. Verify features are normalized correctly
3. Check for NaN values in data: `torch.isnan(features).any()`
4. Increase number of epochs

### Issue: "Loss is NaN"

**Solutions**:
1. Reduce learning rate (try 0.001 instead of 0.01)
2. Check for inf/NaN in input data
3. Gradient clipping (though not needed for this model usually)

### Issue: "MPS out of memory"

**Solutions**:
1. Reduce batch_size (try 16 instead of 32)
2. Use CPU instead: `--device cpu`
3. Clear Python cache: `del model; torch.mps.empty_cache()`

### Issue: "Predictions are all the same"

**Solutions**:
1. Check if model is training: `model.train()` before training loop
2. Verify gradients are flowing: `print(model.model[0].weight.grad)`
3. Check if optimizer is stepping: add debug print after `optimizer.step()`

### Issue: "Training too slow"

**Solutions**:
1. Use smaller number of epochs for testing: `--epochs 1000`
2. Increase batch_size if memory allows: `--batch-size 64`
3. Try MPS if on Mac: `--device mps`

---

## Model Interpretation

### Understanding Predictions

```python
# After training, predictions are in minutes
prediction = model(features)  # e.g., tensor([[42.5]])
print(f"Estimated delivery time: {prediction.item():.1f} minutes")

# RMSE tells you typical error
# If RMSE = 1.5 minutes, predictions are typically within ±1.5 minutes
```

### Feature Importance

The model learns these patterns:
- **Distance**: Strong linear relationship (more miles = more time)
- **Rush hour**: Adds delay during peak traffic times
- **Time of day**: Captures traffic patterns throughout the day
- **Weekend**: Generally less traffic, faster deliveries

---

## Useful Commands

```bash
# Training
python train_delivery.py --config configs/train.yaml

# Quick test (100 epochs)
python train_delivery.py --config configs/train.yaml --epochs 100

# Resume training
python train_delivery.py --config configs/train.yaml --resume runs/<run>/checkpoints/best.pt

# View results
ls -lh runs/*/checkpoints/
cat runs/*/metrics.jsonl | tail -5

# Check data
head -20 data/data_with_features.csv
wc -l data/data_with_features.csv  # Should be 101 (100 + header)

# Quality checks
make format && make lint && make typecheck

# Clean old runs (be careful!)
rm -rf runs/2026*
```

---

## References

- **PyTorch Docs**: https://pytorch.org/docs/stable/
- **Pandas Docs**: https://pandas.pydata.org/docs/
- **Original Notebook**: `../notebooks/distance_time.ipynb`
- **README**: `README_DELIVERY.md`
- **ruff**: https://docs.astral.sh/ruff/
- **black**: https://black.readthedocs.io/

---

## Project Comparison

This project is **adapted from pytorch-llm-phase0** with these key changes:

| Aspect | PyTorch-LLM-Phase0 | Distance Model |
|--------|-------------------|----------------|
| Task | Classification | **Regression** |
| Model | SimpleMLP (configurable) | **DeliveryTimeModel (fixed 4→64→32→1)** |
| Loss | CrossEntropyLoss | **MSELoss** |
| Optimizer | AdamW | **SGD** |
| Data | DummyDataset (random) | **DeliveryDataset (CSV + feature engineering)** |
| Metrics | Accuracy | **RMSE, MAE** |
| Training | 10 epochs, classification | **30k epochs, regression** |

The infrastructure (checkpointing, logging, run management) remains the same, but the model and training logic are customized for the delivery time prediction problem.
