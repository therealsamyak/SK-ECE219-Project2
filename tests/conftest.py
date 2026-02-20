import pytest
import numpy as np
import torch
import pandas as pd


# Register custom markers
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


@pytest.fixture
def sample_features():
    """Synthetic features for testing (50 samples, 50 dims)"""
    np.random.seed(42)
    return np.random.randn(50, 50).astype(np.float32)


@pytest.fixture
def sample_4096_features():
    """Synthetic 4096-dim features for testing (50 samples)"""
    np.random.seed(42)
    return np.random.randn(50, 4096).astype(np.float32)


@pytest.fixture
def sample_labels():
    """Synthetic labels for testing (50 samples, 5 classes)"""
    np.random.seed(42)
    return np.random.randint(0, 5, 50)


@pytest.fixture
def sample_class_names():
    """Sample class names for flower dataset"""
    return ["daisy", "dandelion", "roses", "sunflowers", "tulips"]


@pytest.fixture
def sample_image_batch():
    """Synthetic image batch (4 images, 3 channels, 224x224)"""
    torch.manual_seed(42)
    return torch.randn(4, 3, 224, 224)


@pytest.fixture
def outputs_dir(tmp_path):
    """Create a temporary outputs directory for testing"""
    outputs = tmp_path / "outputs"
    outputs.mkdir(exist_ok=True)
    return outputs


@pytest.fixture
def cache_dir(tmp_path):
    """Create a temporary cache directory for testing"""
    cache = tmp_path / ".cache"
    cache.mkdir(exist_ok=True)
    return cache


@pytest.fixture
def sample_pokedex():
    """Sample pokedex DataFrame (10 Pokemon)"""
    data = {
        "Name": [
            "Bulbasaur",
            "Charmander",
            "Squirtle",
            "Pikachu",
            "Eevee",
            "Jigglypuff",
            "Meowth",
            "Psyduck",
            "Geodude",
            "Gastly",
        ],
        "Type1": [
            "Grass",
            "Fire",
            "Water",
            "Electric",
            "Normal",
            "Normal",
            "Normal",
            "Water",
            "Rock",
            "Ghost",
        ],
        "Type2": ["Poison", "", "", "", "", "Fairy", "", "", "Ground", "Poison"],
        "image_path": [
            f"datasets/pokemon/images/{name}/0.jpg"
            for name in [
                "Bulbasaur",
                "Charmander",
                "Squirtle",
                "Pikachu",
                "Eevee",
                "Jigglypuff",
                "Meowth",
                "Psyduck",
                "Geodude",
                "Gastly",
            ]
        ],
    }
    df = pd.DataFrame(data)
    df["Type2"] = df["Type2"].str.strip()
    return df[["Name", "Type1", "Type2", "image_path"]]


@pytest.fixture
def sample_embeddings():
    """Normalized synthetic embeddings (50 samples, 512 dims)"""
    np.random.seed(42)
    embeddings = np.random.randn(50, 512).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)
    return embeddings
