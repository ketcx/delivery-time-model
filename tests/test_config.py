"""Tests for src.config module - simplified."""

from src.config import Config


class TestConfigDataClass:
    """Test Config dataclass."""

    def test_config_creation(self) -> None:
        """Test that Config can be instantiated."""
        config = Config(
            seed=42,
            device="cpu",
            precision="fp32",
        )
        assert config.seed == 42
        assert config.device == "cpu"
        assert config.precision == "fp32"

    def test_config_defaults(self) -> None:
        """Test Config with default values."""
        config = Config()
        assert config.seed >= 0
        assert config.device in ["cpu", "mps"]
        assert config.precision in ["fp32", "fp16", "bf16"]

    def test_config_training_params(self) -> None:
        """Test config has training parameters."""
        config = Config()
        assert hasattr(config, "training")
        assert hasattr(config.training, "epochs")
        assert hasattr(config.training, "batch_size")
        assert hasattr(config.training, "learning_rate")
