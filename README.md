# Delivery Time Prediction Model

A PyTorch-based neural network for predicting delivery times based on distance, time of day, and day of week, with engineered rush hour features.

## Overview

This project implements a regression model that predicts delivery times in minutes using a multi-layer perceptron (MLP) architecture. The model is trained on real delivery data and includes feature engineering to capture rush hour patterns.

### Model Architecture

- **Input Layer**: 4 features
  - Normalized distance (miles)
  - Normalized time of day (hours, 24-hour format)
  - Weekend flag (binary: 0=weekday, 1=weekend)
  - Rush hour flag (binary, engineered feature)

- **Hidden Layers**: 
  - Layer 1: Linear(4 → 64) + ReLU
  - Layer 2: Linear(64 → 32) + ReLU

- **Output Layer**: Linear(32 → 1)
  - Predicts delivery time in minutes

### Feature Engineering

The model includes an engineered **rush hour feature** that identifies deliveries during peak traffic times:
- **Morning rush**: 8:00 AM - 10:00 AM (8.0 - 10.0)
- **Evening rush**: 4:00 PM - 7:00 PM (16.0 - 19.0)
- Only applies to **weekdays** (is_weekend == 0)

This feature helps the model better capture the impact of traffic patterns on delivery times.

## Project Structure

```
distance_model/
├── src/
│   ├── model.py           # DeliveryTimeModel definition
│   ├── data.py            # DeliveryDataset with feature engineering
│   ├── train.py           # Original training script (classification)
│   └── ...
├── configs/
│   └── train.yaml         # Training configuration
├── data/
│   └── data_with_features.csv  # Delivery dataset (100 samples)
├── train_delivery.py      # Custom training script for delivery model
├── requirements.txt       # Python dependencies
└── README_DELIVERY.md     # This file
```

## Installation

1. Ensure you have Python 3.8+ installed

2. Create and activate a virtual environment with uv:
```bash
uv venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

The main dependencies are:
- PyTorch 2.10.0+
- pandas 3.0.0+
- numpy 2.4.1+
- pyyaml 6.0.3+

## Dataset

The dataset (`data/data_with_features.csv`) contains 100 past delivery records with the following columns:

- `distance_miles`: Distance of delivery route (float)
- `time_of_day_hours`: Time order was dispatched (float, 24-hour format)
- `is_weekend`: Day type (0=weekday, 1=weekend)
- `delivery_time_minutes`: Actual delivery time in minutes (target variable)

**Business Rules**:
- Deliveries only occur between 8:00 AM and 8:00 PM
- Maximum delivery distance is 20 miles

## Training

### Quick Start

Train the model with default settings:

```bash
python train_delivery.py --config configs/train.yaml
```

### Configuration

The training configuration is defined in `configs/train.yaml`:

```yaml
seed: 41                    # Random seed
training:
  epochs: 30000            # Total training epochs
  batch_size: 32           # Batch size
  learning_rate: 0.01      # SGD learning rate
model:
  input_dim: 4             # Number of input features
  hidden_size_1: 64        # First hidden layer size
  hidden_size_2: 32        # Second hidden layer size
device: mps                # Use 'mps' for M1/M2/M3 Macs, 'cpu' otherwise
```

### Training Output

The training script will:
1. Load and preprocess the data with feature engineering
2. Create train/val/test splits (80%/10%/10%)
3. Train the model for 30,000 epochs
4. Log metrics every 5,000 epochs
5. Save checkpoints to `runs/<timestamp>/checkpoints/`
6. Evaluate on the test set and report MSE, RMSE, and MAE

### Custom Training Parameters

Override configuration via CLI:

```bash
# Change learning rate
python train_delivery.py --config configs/train.yaml --lr 0.005

# Use CPU instead of MPS
python train_delivery.py --config configs/train.yaml --device cpu

# Train for fewer epochs
python train_delivery.py --config configs/train.yaml --epochs 10000
```

### Resume Training

Resume from a checkpoint:

```bash
python train_delivery.py --config configs/train.yaml --resume runs/<run_name>/checkpoints/best.pt
```

## Model Performance

After training for 30,000 epochs, the model achieves:
- Low MSE loss on validation data
- Tight clustering around the perfect prediction line
- Accurate predictions across the full range of delivery times

The model captures:
- Linear relationship between distance and delivery time
- Rush hour delays (morning and evening)
- Weekend vs. weekday patterns

## Usage Example

After training, you can use the model for predictions:

```python
import torch
from src.model import create_delivery_model

# Load trained model
model = create_delivery_model(device='cpu')
checkpoint = torch.load('runs/<run_name>/checkpoints/best.pt')
model.load_state_dict(checkpoint['model'])
model.eval()

# Prepare input (must match training normalization)
# Example: 15 miles, 5:30 PM (17.5), weekday (0)
distance = 15.0
time_of_day = 17.5
is_weekend = 0

# You'll need the normalization parameters from training
# (stored in the dataset or checkpoint)
# Then predict:
with torch.no_grad():
    features = torch.tensor([[norm_distance, norm_time, is_weekend, rush_hour]])
    prediction = model(features)
    print(f"Predicted delivery time: {prediction.item():.2f} minutes")
```

  ### Predict from CLI

  You can also run the interactive predictor script:

  ```bash
  python predict_delivery.py
  ```

  The script will prompt for:
  - Dataset CSV path (defaults to data/data_with_features.csv)
  - Checkpoint path (e.g., runs/<run>/checkpoints/best.pt)
  - Distance, time of day, and weekend flag

### Run FastAPI Server

Start the API server locally:

```bash
# First activate the virtual environment
source .venv/bin/activate

# Set required environment variables
export MODEL_PATH="runs/<run_name>/checkpoints/best.pt"  # Required: Path to trained model
export DATA_PATH="data/data_with_features.csv"            # Optional: Path to training data for normalization stats

# Run the API server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**Required Environment Variables**:
- `MODEL_PATH`: Path to trained checkpoint file (e.g., `runs/<run_name>/checkpoints/best.pt`)
- `DATA_PATH`: Path to training data CSV for normalization statistics (defaults to `data/data_with_features.csv`)

**Note**: You must train a model first using `train_delivery.py` before running the API server. The checkpoint file is created in the `runs/` directory after training.

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

**Endpoints**:
- `GET /` - Root health check
- `GET /health` - Detailed health check (model loaded, stats loaded)
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions (up to 100 at once)

**Example request**:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"distance_miles": 5.5, "time_of_day_hours": 17.5, "is_weekend": 0}'
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

Format code with black:
```bash
black src/ train_delivery.py
```

Check types with mypy:
```bash
mypy src/
```

## License

MIT License (see LICENSE file)

## Acknowledgments

Based on the PyTorch LLM Phase 0 project structure, adapted for delivery time prediction.
