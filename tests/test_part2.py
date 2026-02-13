import pytest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from part2 import (
    get_device,
    FeatureExtractor,
    Autoencoder,
    MLP,
    apply_dim_reduction,
    run_clustering,
    compute_clustering_metrics,
    extract_flower_features,
    run_tsne_visualization,
    run_clustering_grid_search,
    run_mlp_classifier,
    create_summary_output,
)


# ============== FIXTURES ==============


@pytest.fixture
def sample_features():
    """Synthetic features for testing (100 samples, 50 dims)"""
    np.random.seed(42)
    return np.random.randn(100, 50).astype(np.float32)


@pytest.fixture
def sample_4096_features():
    """Synthetic 4096-dim features for testing (100 samples)"""
    np.random.seed(42)
    return np.random.randn(100, 4096).astype(np.float32)


@pytest.fixture
def sample_labels_100():
    """Synthetic labels for testing (100 samples, 5 classes)"""
    np.random.seed(42)
    return np.random.randint(0, 5, 100)


@pytest.fixture
def sample_labels():
    """Synthetic labels for testing (100 samples, 5 classes)"""
    np.random.seed(42)
    return np.random.randint(0, 5, 100)


@pytest.fixture
def sample_class_names():
    """Sample class names for flower dataset"""
    return ["daisy", "dandelion", "roses", "sunflowers", "tulips"]


@pytest.fixture
def sample_image_batch():
    """Synthetic image batch for FeatureExtractor testing (4 images, 3 channels, 224x224)"""
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


# ============== TestGetDevice ==============


class TestGetDevice:
    """Tests for get_device() function"""

    def test_returns_string(self):
        """Device should be a string"""
        device = get_device()
        assert isinstance(device, str)

    def test_returns_valid_device(self):
        """Device should be one of the three valid options"""
        device = get_device()
        assert device in ["cuda", "mps", "cpu"]

    def test_consistent_results(self):
        """Calling multiple times returns same value"""
        device1 = get_device()
        device2 = get_device()
        assert device1 == device2


# ============== TestFeatureExtractor ==============


class TestFeatureExtractor:
    """Tests for FeatureExtractor class"""

    def test_forward_output_shape(self, sample_image_batch):
        """Given 224x224x3 input, outputs 4096-dim vector"""
        model = FeatureExtractor()
        model.eval()
        with torch.no_grad():
            output = model(sample_image_batch)
        assert output.shape == (4, 4096)

    def test_forward_batch_processing(self, sample_image_batch):
        """Can process batch of images"""
        model = FeatureExtractor()
        model.eval()
        # Test with different batch sizes
        for batch_size in [1, 4, 8]:
            batch = torch.randn(batch_size, 3, 224, 224)
            with torch.no_grad():
                output = model(batch)
            assert output.shape[0] == batch_size
            assert output.shape[1] == 4096

    def test_eval_mode(self, sample_image_batch):
        """Model works in eval mode"""
        model = FeatureExtractor()
        model.eval()
        with torch.no_grad():
            output = model(sample_image_batch)
        assert output.shape == (4, 4096)
        # Verify no gradient computation
        assert not output.requires_grad


# ============== TestAutoencoder ==============


class TestAutoencoder:
    """Tests for Autoencoder class"""

    def test_init_creates_encoder_decoder(self):
        """Encoder and decoder are None before fit, exist after"""
        autoencoder = Autoencoder(n_components=10)
        assert autoencoder.encoder is None
        assert autoencoder.decoder is None

    def test_fit_creates_encoder_decoder(self, sample_4096_features):
        """Encoder and decoder are created after fit"""
        autoencoder = Autoencoder(n_components=10)
        # Patch to reduce epochs for faster testing
        autoencoder.fit(sample_4096_features, device="cpu")
        assert autoencoder.encoder is not None
        assert autoencoder.decoder is not None

    def test_fit_transform_output_shape(self, sample_4096_features):
        """4096 input -> n_components output"""
        autoencoder = Autoencoder(n_components=10)
        result = autoencoder.fit_transform(sample_4096_features, device="cpu")
        assert result.shape == (100, 10)

    def test_reconstruction_shape(self, sample_4096_features):
        """Forward pass returns same shape as input"""
        autoencoder = Autoencoder(n_components=10)
        autoencoder.fit(sample_4096_features, device="cpu")
        X_tensor = torch.tensor(sample_4096_features, dtype=torch.float32)
        with torch.no_grad():
            reconstructed = autoencoder(X_tensor)
        assert reconstructed.shape == X_tensor.shape

    def test_transform_requires_fit(self, sample_4096_features):
        """Calling transform before fit raises error or creates encoder"""
        autoencoder = Autoencoder(n_components=10)
        # This should either raise an error or handle gracefully
        # After fit it should work
        autoencoder.fit(sample_4096_features, device="cpu")
        result = autoencoder.transform(sample_4096_features, device="cpu")
        assert result.shape == (100, 10)


# ============== TestMLP ==============


class TestMLP:
    """Tests for MLP class"""

    def test_forward_output_shape(self, sample_features):
        """num_features input -> num_classes output"""
        mlp = MLP(num_features=50, num_classes=5)
        X_tensor = torch.tensor(sample_features, dtype=torch.float32)
        output = mlp(X_tensor)
        assert output.shape == (100, 5)

    def test_forward_batch_processing(self, sample_features):
        """Can process batch"""
        mlp = MLP(num_features=50, num_classes=5)
        for batch_size in [1, 10, 50]:
            batch = torch.randn(batch_size, 50)
            output = mlp(batch)
            assert output.shape == (batch_size, 5)

    def test_evaluate_returns_accuracy(self, sample_features, sample_labels):
        """Returns float between 0 and 1"""
        mlp = MLP(num_features=50, num_classes=5)
        # Quick training for test
        X_train = torch.tensor(sample_features[:80], dtype=torch.float32)
        y_train = torch.tensor(sample_labels[:80], dtype=torch.int64)

        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)

        mlp.train()
        for _ in range(5):  # Quick training
            optimizer.zero_grad()
            output = mlp(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

        accuracy = mlp.evaluate(sample_features[80:], sample_labels[80:], device="cpu")
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0

    def test_train_model_returns_self(self, sample_features, sample_labels):
        """Method returns self for chaining"""
        mlp = MLP(num_features=50, num_classes=5)
        # Patch to reduce epochs
        original_code = mlp.train_model
        result = original_code(sample_features, sample_labels, device="cpu")
        assert result is mlp


# ============== TestApplyDimReduction ==============


class TestApplyDimReduction:
    """Tests for apply_dim_reduction() function"""

    def test_none_returns_same(self, sample_features):
        """method='none' returns input unchanged"""
        result = apply_dim_reduction(sample_features, method="none")
        np.testing.assert_array_equal(result, sample_features)

    def test_svd_reduces_dimensions(self, sample_features):
        """SVD reduces to n_components"""
        result = apply_dim_reduction(sample_features, method="svd", n_components=10)
        assert result.shape == (100, 10)

    def test_umap_reduces_dimensions(self, sample_features):
        """UMAP reduces to n_components"""
        result = apply_dim_reduction(sample_features, method="umap", n_components=10)
        assert result.shape == (100, 10)

    def test_autoencoder_reduces_dimensions(self, sample_4096_features):
        """Autoencoder reduces to n_components"""
        # Use smaller subset and fewer dimensions for speed
        small_features = sample_4096_features[:20]
        result = apply_dim_reduction(
            small_features, method="autoencoder", n_components=5, device="cpu"
        )
        assert result.shape == (20, 5)

    def test_invalid_method_raises_error(self, sample_features):
        """Unknown method raises ValueError"""
        with pytest.raises(ValueError, match="Unknown dimensionality reduction method"):
            apply_dim_reduction(sample_features, method="invalid_method")


# ============== TestRunClustering ==============


class TestRunClustering:
    """Tests for run_clustering() function"""

    def test_kmeans_returns_correct_labels(self, sample_features):
        """Returns array of correct length"""
        labels = run_clustering(sample_features, method="kmeans", n_clusters=5)
        assert len(labels) == 100
        assert set(labels).issubset(set(range(5)))

    def test_agglomerative_returns_correct_labels(self, sample_features):
        """Returns array of correct length"""
        labels = run_clustering(sample_features, method="agglomerative", n_clusters=5)
        assert len(labels) == 100
        assert set(labels).issubset(set(range(5)))

    def test_hdbscan_returns_correct_labels(self, sample_features):
        """Returns array of correct length, may have -1"""
        labels = run_clustering(
            sample_features, method="hdbscan", min_cluster_size=3, min_samples=2
        )
        assert len(labels) == 100
        # HDBSCAN can return -1 for noise points
        assert min(labels) >= -1

    def test_invalid_method_raises_error(self, sample_features):
        """Unknown method raises ValueError"""
        with pytest.raises(ValueError, match="Unknown clustering method"):
            run_clustering(sample_features, method="invalid_method")


# ============== TestComputeClusteringMetrics ==============


class TestComputeClusteringMetrics:
    """Tests for compute_clustering_metrics() function"""

    def test_returns_dict_with_all_metrics(self, sample_labels):
        """Has homogeneity, completeness, v_measure, ari, ami"""
        pred_labels = sample_labels.copy()
        metrics = compute_clustering_metrics(sample_labels, pred_labels)
        required_keys = ["homogeneity", "completeness", "v_measure", "ari", "ami"]
        for key in required_keys:
            assert key in metrics

    def test_metrics_in_valid_range(self, sample_labels):
        """Values are between 0 and 1 for valid clusters"""
        pred_labels = sample_labels.copy()
        metrics = compute_clustering_metrics(sample_labels, pred_labels)
        for key, value in metrics.items():
            if value is not None:
                assert 0.0 <= value <= 1.0, f"{key} should be between 0 and 1"

    def test_handles_noise_points(self, sample_labels):
        """Works when labels have -1 noise points"""
        pred_labels = sample_labels.copy()
        # Add some noise points
        pred_labels[:10] = -1
        metrics = compute_clustering_metrics(sample_labels, pred_labels)
        assert isinstance(metrics, dict)
        # Should still compute metrics for non-noise points
        assert all(
            key in metrics
            for key in ["homogeneity", "completeness", "v_measure", "ari", "ami"]
        )

    def test_perfect_clustering_gives_high_scores(self):
        """Perfect clustering should give high metric values"""
        labels_true = np.array([0, 0, 1, 1, 2, 2] * 10)
        labels_pred = labels_true.copy()
        metrics = compute_clustering_metrics(labels_true, labels_pred)
        # Perfect clustering should give high scores
        assert metrics["ari"] > 0.9
        assert metrics["v_measure"] > 0.9


# ============== TestExtractFlowerFeatures ==============


class TestExtractFlowerFeatures:
    """Tests for extract_flower_features() function"""

    def test_returns_dict_with_required_keys(self):
        """metadata, features, labels, class_names"""
        # This test requires cached features or will download data
        # Skip if no cache available and offline
        cache_file = Path(".cache/flowers_features_and_labels.npz")
        if not cache_file.exists():
            pytest.skip("No cached features available")
        result = extract_flower_features()
        required_keys = ["metadata", "features", "labels", "class_names"]
        for key in required_keys:
            assert key in result

    def test_features_shape_correct(self):
        """Features shape is (n_samples, 4096)"""
        cache_file = Path(".cache/flowers_features_and_labels.npz")
        if not cache_file.exists():
            pytest.skip("No cached features available")
        result = extract_flower_features()
        assert result["features"].shape[1] == 4096

    def test_labels_are_integers(self):
        """Labels are integer type"""
        cache_file = Path(".cache/flowers_features_and_labels.npz")
        if not cache_file.exists():
            pytest.skip("No cached features available")
        result = extract_flower_features()
        # Labels can be float64 but should contain whole numbers
        assert np.allclose(result["labels"], result["labels"].astype(int))

    def test_json_file_created(self):
        """part2_features.json exists after running"""
        cache_file = Path(".cache/flowers_features_and_labels.npz")
        if not cache_file.exists():
            pytest.skip("No cached features available")
        extract_flower_features()
        json_path = Path("outputs/part2_features.json")
        assert json_path.exists()


# ============== TestRunTsneVisualization ==============


class TestRunTsneVisualization:
    """Tests for run_tsne_visualization() function"""

    def test_returns_dict_with_paths(
        self, sample_features, sample_labels, sample_class_names
    ):
        """has plot_path and json_path"""
        result = run_tsne_visualization(
            sample_features, sample_labels, sample_class_names
        )
        assert "plot_path" in result
        assert "json_path" in result

    def test_plot_file_is_png(self, sample_features, sample_labels, sample_class_names):
        """plot ends with .png"""
        result = run_tsne_visualization(
            sample_features, sample_labels, sample_class_names
        )
        assert result["plot_path"].endswith(".png")

    def test_json_contains_required_keys(
        self, sample_features, sample_labels, sample_class_names
    ):
        """tsne_2d, labels, classes"""
        result = run_tsne_visualization(
            sample_features, sample_labels, sample_class_names
        )
        with open(result["json_path"]) as f:
            data = json.load(f)
        required_keys = ["tsne_2d", "labels", "classes"]
        for key in required_keys:
            assert key in data

    def test_tsne_coordinates_shape(
        self, sample_features, sample_labels, sample_class_names
    ):
        """tsne_2d has correct shape (n_samples, 2)"""
        result = run_tsne_visualization(
            sample_features, sample_labels, sample_class_names
        )
        with open(result["json_path"]) as f:
            data = json.load(f)
        tsne_2d = np.array(data["tsne_2d"])
        assert tsne_2d.shape == (len(sample_features), 2)


# ============== TestRunClusteringGridSearch ==============


class TestRunClusteringGridSearch:
    """Tests for run_clustering_grid_search() function"""

    def test_returns_dict_with_required_keys(self, sample_4096_features, sample_labels):
        """pipelines, best_per_dim_reduction, best_overall"""
        features = sample_4096_features[:80]
        labels = sample_labels[:80]
        result = run_clustering_grid_search(features, labels, device="cpu")
        required_keys = ["pipelines", "best_per_dim_reduction", "best_overall"]
        for key in required_keys:
            assert key in result

    def test_pipelines_contain_metrics(self, sample_4096_features, sample_labels):
        """each pipeline has metrics dict"""
        features = sample_4096_features[:80]
        labels = sample_labels[:80]
        result = run_clustering_grid_search(features, labels, device="cpu")
        for pipeline in result["pipelines"]:
            assert "metrics" in pipeline
            metrics = pipeline["metrics"]
            assert "homogeneity" in metrics
            assert "completeness" in metrics
            assert "v_measure" in metrics
            assert "ari" in metrics
            assert "ami" in metrics

    def test_best_overall_has_valid_ari(self, sample_4096_features, sample_labels):
        """best_ari is a float"""
        features = sample_4096_features[:80]
        labels = sample_labels[:80]
        result = run_clustering_grid_search(features, labels, device="cpu")
        assert "best_ari" in result["best_overall"]
        assert isinstance(result["best_overall"]["best_ari"], (int, float))

    def test_dim_reduction_methods_in_results(
        self, sample_4096_features, sample_labels
    ):
        """Results contain all dimensionality reduction methods"""
        features = sample_4096_features[:80]
        labels = sample_labels[:80]
        result = run_clustering_grid_search(features, labels, device="cpu")
        dim_methods = {p["dim_reduction"] for p in result["pipelines"]}
        assert "none" in dim_methods
        assert "svd" in dim_methods
        assert "umap" in dim_methods
        assert "autoencoder" in dim_methods


# ============== TestRunMlpClassifier ==============


class TestRunMlpClassifier:
    """Tests for run_mlp_classifier() function"""

    def test_returns_dict_with_required_keys(self, sample_4096_features, sample_labels):
        """original_features, reduced_features, summary"""
        features = sample_4096_features[:80]
        labels = sample_labels[:80]
        result = run_mlp_classifier(features, labels, device="cpu")
        required_keys = ["original_features", "reduced_features", "summary"]
        for key in required_keys:
            assert key in result

    def test_summary_has_accuracy_comparison(self, sample_4096_features, sample_labels):
        """has original_accuracy, best_reduced_accuracy"""
        features = sample_4096_features[:80]
        labels = sample_labels[:80]
        result = run_mlp_classifier(features, labels, device="cpu")
        summary = result["summary"]
        assert "original_accuracy" in summary
        assert "best_reduced_accuracy" in summary

    def test_accuracies_in_valid_range(self, sample_4096_features, sample_labels):
        """all accuracies between 0 and 1"""
        features = sample_4096_features[:80]
        labels = sample_labels[:80]
        result = run_mlp_classifier(features, labels, device="cpu")
        assert 0.0 <= result["original_features"]["test_accuracy"] <= 1.0
        assert 0.0 <= result["summary"]["best_reduced_accuracy"] <= 1.0
        for dim_method, data in result["reduced_features"].items():
            assert 0.0 <= data["test_accuracy"] <= 1.0

    def test_reduced_features_contains_all_methods(
        self, sample_4096_features, sample_labels
    ):
        """Tests SVD, UMAP, and Autoencoder reduction"""
        features = sample_4096_features[:80]
        labels = sample_labels[:80]
        result = run_mlp_classifier(features, labels, device="cpu")
        assert "svd" in result["reduced_features"]
        assert "umap" in result["reduced_features"]
        assert "autoencoder" in result["reduced_features"]


# ============== TestCreateSummaryOutput ==============


class TestCreateSummaryOutput:
    """Tests for create_summary_output() function"""

    def setup_method(self):
        """Setup required JSON files for testing"""
        # Ensure outputs directory exists
        Path("outputs").mkdir(exist_ok=True)

        # Create mock feature data
        feature_data = {
            "original_pixels": [224, 224, 3],
            "feature_dim": 4096,
            "num_images": 3670,
            "is_dense": True,
            "classes": ["daisy", "dandelion", "roses", "sunflowers", "tulips"],
        }
        with open("outputs/part2_features.json", "w") as f:
            json.dump(feature_data, f)

        # Create mock t-SNE data
        tsne_data = {
            "tsne_2d": [[0.1, 0.2]] * 100,
            "labels": [0] * 100,
            "classes": ["daisy", "dandelion", "roses", "sunflowers", "tulips"],
        }
        with open("outputs/part2_tsne.json", "w") as f:
            json.dump(tsne_data, f)

        # Create mock clustering data
        clustering_data = {
            "best_overall": {"best_ari": 0.5, "dim_reduction": "umap"},
            "pipelines": [],
        }
        with open("outputs/part2_clustering.json", "w") as f:
            json.dump(clustering_data, f)

        # Create mock MLP data
        mlp_data = {
            "summary": {
                "original_accuracy": 0.85,
                "best_reduced_accuracy": 0.87,
                "best_reduced_method": "umap",
            }
        }
        with open("outputs/part2_mlp.json", "w") as f:
            json.dump(mlp_data, f)

    def test_returns_dict_with_q13_q19(self):
        """has Q13-Q19 keys"""
        result = create_summary_output()
        for q in ["Q13", "Q14", "Q15", "Q16", "Q17", "Q18", "Q19"]:
            assert q in result or any(q in k for k in result.keys())

    def test_q15_dimensions_derived_from_features(self):
        """matches features data"""
        result = create_summary_output()
        # Check Q15 contains dimension info
        assert "Q15_dimensions" in result
        dims = result["Q15_dimensions"]
        assert dims["feature_dim"] == 4096
        assert dims["original_pixels"] == [224, 224, 3]

    def test_json_file_created(self):
        """part2_summary.json exists"""
        create_summary_output()
        json_path = Path("outputs/part2_summary.json")
        assert json_path.exists()

    def test_summary_structure(self):
        """Summary has proper structure"""
        result = create_summary_output()
        assert isinstance(result, dict)
        # Check that key sections exist
        assert "Q13_transfer_learning" in result
        assert "Q14_feature_extraction" in result
        assert "Q15_dimensions" in result
        assert "Q16_sparsity" in result
        assert "Q17_tsne" in result
        assert "Q18_clustering" in result
        assert "Q19_mlp" in result


# ============== Additional Edge Case Tests ==============


class TestEdgeCases:
    """Edge case tests for robustness"""

    def test_single_sample_clustering(self):
        """Clustering handles single sample gracefully"""
        single_feature = np.random.randn(1, 50)
        # KMeans should work with single sample
        labels = run_clustering(single_feature, method="kmeans", n_clusters=1)
        assert len(labels) == 1

    def test_small_batch_mlp(self):
        """MLP handles small batches"""
        mlp = MLP(num_features=10, num_classes=3)
        small_batch = torch.randn(2, 10)
        output = mlp(small_batch)
        assert output.shape == (2, 3)

    def test_metrics_with_all_noise(self):
        """Metrics handles all noise points"""
        labels_true = np.array([0, 1, 2, 3, 4] * 10)
        labels_pred = np.array([-1] * 50)  # All noise
        metrics = compute_clustering_metrics(labels_true, labels_pred)
        # Should return None values or handle gracefully
        assert isinstance(metrics, dict)

    def test_dim_reduction_preserves_samples(self, sample_features):
        """Dimensionality reduction preserves number of samples"""
        n_samples = sample_features.shape[0]

        # SVD
        svd_result = apply_dim_reduction(sample_features, method="svd", n_components=10)
        assert svd_result.shape[0] == n_samples

        # UMAP
        umap_result = apply_dim_reduction(
            sample_features, method="umap", n_components=10
        )
        assert umap_result.shape[0] == n_samples
