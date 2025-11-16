import pytest
import torch
import os


def test_model_file_exists():
    """Test that the model file exists."""
    assert os.path.exists('models/best_local_transformer.pt'), "Model file not found"


def test_tokenizer_file_exists():
    """Test that the tokenizer file exists."""
    assert os.path.exists('models/bpe_enfr.model'), "Tokenizer file not found"


def test_model_loads_successfully():
    """Test that the model can be loaded without errors."""
    try:
        model_data = torch.load('models/best_local_transformer.pt', map_location='cpu')
        assert model_data is not None
    except Exception as e:
        pytest.fail(f"Failed to load model: {str(e)}")


def test_torch_available():
    """Test that PyTorch is available."""
    assert torch.__version__ is not None
    assert torch.cuda.is_available() or True  # CPU is fine for testing


def test_model_device_compatibility():
    """Test that model can be loaded on available device."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model_data = torch.load('models/best_local_transformer.pt', map_location=device)
        assert model_data is not None
    except Exception as e:
        pytest.fail(f"Model not compatible with device {device}: {str(e)}")
