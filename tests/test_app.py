"""Tests for FastAPI application endpoints."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient

from app import app


@pytest.fixture
def mock_model():
    """Mock model that returns a fixed prediction."""
    model = MagicMock()
    model.return_value = torch.tensor([[30.5]])  # Mock prediction value
    model.eval = MagicMock()
    return model


@pytest.fixture
def mock_stats():
    """Mock normalization statistics."""
    return {
        "DISTANCE_MEAN": 10.0,
        "DISTANCE_STD": 5.0,
        "TIME_MEAN": 14.0,
        "TIME_STD": 4.0,
    }


@pytest.fixture
def client_with_mocks(mock_model, mock_stats):
    """Test client with mocked model and stats."""
    with patch("app.MODEL", mock_model):
        with patch("app.DISTANCE_MEAN", mock_stats["DISTANCE_MEAN"]):
            with patch("app.DISTANCE_STD", mock_stats["DISTANCE_STD"]):
                with patch("app.TIME_MEAN", mock_stats["TIME_MEAN"]):
                    with patch("app.TIME_STD", mock_stats["TIME_STD"]):
                        client = TestClient(app)
                        yield client


def test_root_endpoint():
    """Test root health check endpoint."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "message" in data
    assert "version" in data


def test_health_endpoint_model_not_loaded():
    """Test health endpoint when model is not loaded."""
    with patch("app.MODEL", None):
        with patch("app.DISTANCE_MEAN", None):
            client = TestClient(app)
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is False
            assert data["stats_loaded"] is False


def test_health_endpoint_model_loaded(client_with_mocks):
    """Test health endpoint when model and stats are loaded."""
    response = client_with_mocks.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["stats_loaded"] is True
    # Verify return types are booleans, not strings
    assert isinstance(data["model_loaded"], bool)
    assert isinstance(data["stats_loaded"], bool)


def test_predict_endpoint_success(client_with_mocks):
    """Test successful single prediction."""
    payload = {
        "distance_miles": 5.5,
        "time_of_day_hours": 17.5,
        "is_weekend": 0,
    }
    response = client_with_mocks.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_delivery_time_minutes" in data
    assert data["distance_miles"] == 5.5
    assert data["time_of_day_hours"] == 17.5
    assert data["is_weekend"] == 0
    assert "is_rush_hour" in data
    assert isinstance(data["is_rush_hour"], bool)
    assert isinstance(data["predicted_delivery_time_minutes"], float)


def test_predict_endpoint_model_not_loaded():
    """Test prediction when model is not loaded."""
    with patch("app.MODEL", None):
        client = TestClient(app)
        payload = {
            "distance_miles": 5.5,
            "time_of_day_hours": 17.5,
            "is_weekend": 0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]


def test_predict_validation_distance_too_high():
    """Test validation error for distance exceeding maximum."""
    client = TestClient(app)
    payload = {
        "distance_miles": 25.0,  # Max is 20
        "time_of_day_hours": 12.0,
        "is_weekend": 0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error


def test_predict_validation_distance_negative():
    """Test validation error for negative distance."""
    client = TestClient(app)
    payload = {
        "distance_miles": -5.0,
        "time_of_day_hours": 12.0,
        "is_weekend": 0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_validation_time_out_of_range():
    """Test validation error for time outside valid range."""
    client = TestClient(app)
    payload = {
        "distance_miles": 5.0,
        "time_of_day_hours": 22.0,  # Max is 20.0
        "is_weekend": 0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_validation_invalid_weekend_flag():
    """Test validation error for invalid weekend flag."""
    client = TestClient(app)
    payload = {
        "distance_miles": 5.0,
        "time_of_day_hours": 12.0,
        "is_weekend": 2,  # Must be 0 or 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rush_hour_morning_weekday(client_with_mocks):
    """Test rush hour detection for morning weekday."""
    payload = {
        "distance_miles": 5.0,
        "time_of_day_hours": 9.0,  # Morning rush (8-10)
        "is_weekend": 0,
    }
    response = client_with_mocks.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_rush_hour"] is True


def test_predict_rush_hour_evening_weekday(client_with_mocks):
    """Test rush hour detection for evening weekday."""
    payload = {
        "distance_miles": 5.0,
        "time_of_day_hours": 17.0,  # Evening rush (16-19)
        "is_weekend": 0,
    }
    response = client_with_mocks.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_rush_hour"] is True


def test_predict_no_rush_hour_weekend(client_with_mocks):
    """Test no rush hour on weekends."""
    payload = {
        "distance_miles": 5.0,
        "time_of_day_hours": 9.0,  # Would be rush hour on weekday
        "is_weekend": 1,
    }
    response = client_with_mocks.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_rush_hour"] is False


def test_predict_no_rush_hour_midday_weekday(client_with_mocks):
    """Test no rush hour during midday on weekday."""
    payload = {
        "distance_miles": 5.0,
        "time_of_day_hours": 13.0,  # Not rush hour
        "is_weekend": 0,
    }
    response = client_with_mocks.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_rush_hour"] is False


def test_predict_batch_success(client_with_mocks):
    """Test successful batch prediction."""
    payloads = [
        {
            "distance_miles": 5.5,
            "time_of_day_hours": 9.0,
            "is_weekend": 0,
        },
        {
            "distance_miles": 12.0,
            "time_of_day_hours": 14.5,
            "is_weekend": 1,
        },
    ]
    response = client_with_mocks.post("/predict/batch", json=payloads)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2
    for item in data:
        assert "predicted_delivery_time_minutes" in item
        assert "is_rush_hour" in item


def test_predict_batch_empty_list(client_with_mocks):
    """Test batch prediction with empty list."""
    response = client_with_mocks.post("/predict/batch", json=[])
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 0


def test_predict_batch_exceeds_maximum():
    """Test batch prediction exceeding maximum size."""
    with patch("app.MODEL", MagicMock()):
        client = TestClient(app)
        payloads = [
            {
                "distance_miles": 5.0,
                "time_of_day_hours": 12.0,
                "is_weekend": 0,
            }
        ] * 101  # 101 items exceeds max of 100
        response = client.post("/predict/batch", json=payloads)
        assert response.status_code == 400
        assert "Maximum 100" in response.json()["detail"]


def test_predict_batch_model_not_loaded():
    """Test batch prediction when model is not loaded."""
    with patch("app.MODEL", None):
        client = TestClient(app)
        payloads = [
            {
                "distance_miles": 5.0,
                "time_of_day_hours": 12.0,
                "is_weekend": 0,
            }
        ]
        response = client.post("/predict/batch", json=payloads)
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]


def test_predict_batch_validation_error():
    """Test batch prediction with invalid item."""
    client = TestClient(app)
    payloads = [
        {
            "distance_miles": 5.0,
            "time_of_day_hours": 12.0,
            "is_weekend": 0,
        },
        {
            "distance_miles": 25.0,  # Invalid: exceeds max
            "time_of_day_hours": 12.0,
            "is_weekend": 0,
        },
    ]
    response = client.post("/predict/batch", json=payloads)
    assert response.status_code == 422  # Validation error
