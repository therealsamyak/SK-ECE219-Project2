import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock
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
    run_tsne_visualization,
    run_clustering_grid_search,
    run_mlp_classifier,
)


class TestGetDevice:
    def test_returns_string(self):
        device = get_device()
        assert isinstance(device, str)

    def test_returns_valid_device(self):
        device = get_device()
        assert device in ["cuda", "mps", "cpu"]


class TestFeatureExtractor:
    @pytest.mark.slow
    def test_forward_output_shape(self, sample_image_batch):
        model = FeatureExtractor()
        model.eval()
        with torch.no_grad():
            output = model(sample_image_batch)
        assert output.shape == (4, 4096)


class TestAutoencoder:
    def test_init_creates_none_encoder_decoder(self):
        ae = Autoencoder(n_components=10)
        assert ae.encoder is None
        assert ae.decoder is None

    @patch("part2.tqdm")
    def test_fit_creates_encoder_decoder(self, mock_tqdm, sample_4096_features):
        ae = Autoencoder(n_components=10)
        mock_tqdm.return_value = iter(range(100))
        ae.fit(sample_4096_features[:20], device="cpu")
        assert ae.encoder is not None
        assert ae.decoder is not None

    @patch("part2.tqdm")
    def test_fit_transform_output_shape(self, mock_tqdm, sample_4096_features):
        small_data = sample_4096_features[:20]
        mock_tqdm.return_value = iter(range(100))
        ae = Autoencoder(n_components=10)
        result = ae.fit_transform(small_data, device="cpu")
        assert result.shape == (20, 10)


class TestMLP:
    def test_forward_output_shape(self, sample_features):
        mlp = MLP(num_features=50, num_classes=5)
        X_tensor = torch.tensor(sample_features, dtype=torch.float32)
        output = mlp(X_tensor)
        assert output.shape == (50, 5)

    def test_evaluate_returns_accuracy(self, sample_features, sample_labels):
        mlp = MLP(num_features=50, num_classes=5)
        accuracy = mlp.evaluate(sample_features, sample_labels, device="cpu")
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0


class TestApplyDimReduction:
    def test_none_returns_same(self, sample_features):
        result = apply_dim_reduction(sample_features, "none")
        np.testing.assert_array_equal(result, sample_features)

    def test_svd_reduces_dimensions(self, sample_features):
        result = apply_dim_reduction(sample_features, "svd", n_components=10)
        assert result.shape == (50, 10)

    def test_umap_reduces_dimensions(self, sample_features):
        result = apply_dim_reduction(sample_features, "umap", n_components=10)
        assert result.shape == (50, 10)

    @pytest.mark.slow
    @patch("part2.tqdm")
    def test_autoencoder_reduces_dimensions(self, mock_tqdm, sample_4096_features):
        small_data = sample_4096_features[:20]
        mock_tqdm.return_value = iter(range(100))
        result = apply_dim_reduction(
            small_data, "autoencoder", n_components=5, device="cpu"
        )
        assert result.shape == (20, 5)

    def test_invalid_method_raises_error(self, sample_features):
        with pytest.raises(ValueError, match="Unknown dimensionality reduction method"):
            apply_dim_reduction(sample_features, "invalid")


class TestRunClustering:
    def test_kmeans_returns_labels(self, sample_features):
        labels = run_clustering(sample_features, "kmeans", n_clusters=5)
        assert len(labels) == 50
        assert set(labels).issubset(set(range(5)))

    def test_agglomerative_returns_labels(self, sample_features):
        labels = run_clustering(sample_features, "agglomerative", n_clusters=5)
        assert len(labels) == 50

    def test_invalid_method_raises_error(self, sample_features):
        with pytest.raises(ValueError, match="Unknown clustering method"):
            run_clustering(sample_features, "invalid")


class TestComputeClusteringMetrics:
    def test_returns_dict_with_all_keys(self, sample_labels):
        pred = sample_labels.copy()
        metrics = compute_clustering_metrics(sample_labels, pred)
        required = ["homogeneity", "completeness", "v_measure", "ari", "ami"]
        assert all(k in metrics for k in required)

    def test_metrics_in_valid_range(self, sample_labels):
        pred = sample_labels.copy()
        metrics = compute_clustering_metrics(sample_labels, pred)
        for k, v in metrics.items():
            if v is not None:
                assert 0.0 <= v <= 1.0

    def test_perfect_clustering_gives_high_scores(self):
        labels_true = np.array([0, 0, 1, 1] * 12 + [0, 2])
        labels_pred = labels_true.copy()
        metrics = compute_clustering_metrics(labels_true, labels_pred)
        assert metrics["ari"] > 0.9
        assert metrics["v_measure"] > 0.9

    def test_all_noise_returns_none(self):
        labels_true = np.array([0, 1, 2, 3, 4] * 10)
        labels_pred = np.array([-1] * 50)
        metrics = compute_clustering_metrics(labels_true, labels_pred)
        assert all(v is None for v in metrics.values())


class TestRunTsneVisualization:
    @patch("part2.TSNE")
    @patch("part2.plt.savefig")
    def test_returns_dict_with_paths(
        self,
        mock_savefig,
        mock_tsne,
        sample_features,
        sample_labels,
        sample_class_names,
        outputs_dir,
    ):
        mock_tsne.return_value.fit_transform.return_value = np.random.randn(50, 2)
        with patch("part2.Path") as mock_path:
            mock_path.return_value = outputs_dir
            result = run_tsne_visualization(
                sample_features, sample_labels, sample_class_names
            )
        assert "plot_path" in result
        assert "json_path" in result

    @patch("part2.TSNE")
    @patch("part2.plt.savefig")
    def test_plot_ends_with_png(
        self,
        mock_savefig,
        mock_tsne,
        sample_features,
        sample_labels,
        sample_class_names,
        outputs_dir,
    ):
        mock_tsne.return_value.fit_transform.return_value = np.random.randn(50, 2)
        with patch("part2.Path") as mock_path:
            mock_path.return_value = outputs_dir
            result = run_tsne_visualization(
                sample_features, sample_labels, sample_class_names
            )
        assert result["plot_path"].endswith(".png")

    @patch("part2.TSNE")
    @patch("part2.plt.savefig")
    def test_json_has_required_keys(
        self,
        mock_savefig,
        mock_tsne,
        sample_features,
        sample_labels,
        sample_class_names,
        outputs_dir,
    ):
        mock_tsne.return_value.fit_transform.return_value = np.random.randn(50, 2)
        with patch("part2.Path") as mock_path:
            mock_path.return_value = outputs_dir
            result = run_tsne_visualization(
                sample_features, sample_labels, sample_class_names
            )
        with open(result["json_path"]) as f:
            data = json.load(f)
        required = ["tsne_2d", "labels", "classes"]
        assert all(k in data for k in required)


class TestRunClusteringGridSearch:
    @pytest.mark.slow
    @patch("part2.umap.UMAP")
    @patch("part2.tqdm")
    def test_returns_dict_with_keys(
        self, mock_tqdm, mock_umap, sample_4096_features, sample_labels, outputs_dir
    ):
        mock_tqdm.return_value = iter(range(100))
        mock_umap_instance = MagicMock()
        mock_umap_instance.fit_transform.return_value = np.random.randn(40, 50).astype(
            np.float32
        )
        mock_umap_instance.transform.return_value = np.random.randn(8, 50).astype(
            np.float32
        )
        mock_umap.return_value = mock_umap_instance
        features = sample_4096_features[:40]
        labels = sample_labels[:40]
        with patch("part2.Path") as mock_path:
            mock_path.return_value = outputs_dir
            result = run_clustering_grid_search(features, labels, device="cpu")
        required = ["pipelines", "best_per_dim_reduction", "best_overall"]
        assert all(k in result for k in required)

    @pytest.mark.slow
    @patch("part2.umap.UMAP")
    @patch("part2.tqdm")
    def test_pipelines_have_metrics(
        self, mock_tqdm, mock_umap, sample_4096_features, sample_labels, outputs_dir
    ):
        mock_tqdm.return_value = iter(range(100))
        mock_umap_instance = MagicMock()
        mock_umap_instance.fit_transform.return_value = np.random.randn(40, 50).astype(
            np.float32
        )
        mock_umap_instance.transform.return_value = np.random.randn(8, 50).astype(
            np.float32
        )
        mock_umap.return_value = mock_umap_instance
        features = sample_4096_features[:40]
        labels = sample_labels[:40]
        with patch("part2.Path") as mock_path:
            mock_path.return_value = outputs_dir
            result = run_clustering_grid_search(features, labels, device="cpu")
        for p in result["pipelines"]:
            assert "metrics" in p
            metrics = p["metrics"]
            assert "ari" in metrics
            assert "v_measure" in metrics

    @pytest.mark.slow
    @patch("part2.umap.UMAP")
    @patch("part2.tqdm")
    def test_dim_methods_in_results(
        self, mock_tqdm, mock_umap, sample_4096_features, sample_labels, outputs_dir
    ):
        mock_tqdm.return_value = iter(range(100))
        mock_umap_instance = MagicMock()
        mock_umap_instance.fit_transform.return_value = np.random.randn(40, 50).astype(
            np.float32
        )
        mock_umap_instance.transform.return_value = np.random.randn(8, 50).astype(
            np.float32
        )
        mock_umap.return_value = mock_umap_instance
        features = sample_4096_features[:40]
        labels = sample_labels[:40]
        with patch("part2.Path") as mock_path:
            mock_path.return_value = outputs_dir
            result = run_clustering_grid_search(features, labels, device="cpu")
        dim_methods = {p["dim_reduction"] for p in result["pipelines"]}
        assert "none" in dim_methods
        assert "svd" in dim_methods
        assert "umap" in dim_methods
        assert "autoencoder" in dim_methods


class TestRunMlpClassifier:
    @pytest.mark.slow
    @patch("part2.umap.UMAP")
    @patch("part2.tqdm")
    def test_returns_dict_with_keys(
        self, mock_tqdm, mock_umap, sample_4096_features, sample_labels, outputs_dir
    ):
        mock_tqdm.return_value = iter(range(100))
        mock_umap_instance = MagicMock()
        mock_umap_instance.fit_transform.return_value = np.random.randn(40, 50).astype(
            np.float32
        )
        mock_umap_instance.transform.return_value = np.random.randn(8, 50).astype(
            np.float32
        )
        mock_umap.return_value = mock_umap_instance
        features = sample_4096_features[:40]
        labels = sample_labels[:40]
        with patch("part2.Path") as mock_path:
            mock_path.return_value = outputs_dir
            result = run_mlp_classifier(features, labels, device="cpu")
        required = ["original_features", "reduced_features", "summary"]
        assert all(k in result for k in required)

    @pytest.mark.slow
    @patch("part2.umap.UMAP")
    @patch("part2.tqdm")
    def test_accuracies_in_valid_range(
        self, mock_tqdm, mock_umap, sample_4096_features, sample_labels, outputs_dir
    ):
        mock_tqdm.return_value = iter(range(100))
        mock_umap_instance = MagicMock()
        mock_umap_instance.fit_transform.return_value = np.random.randn(40, 50).astype(
            np.float32
        )
        mock_umap_instance.transform.return_value = np.random.randn(8, 50).astype(
            np.float32
        )
        mock_umap.return_value = mock_umap_instance
        features = sample_4096_features[:40]
        labels = sample_labels[:40]
        with patch("part2.Path") as mock_path:
            mock_path.return_value = outputs_dir
            result = run_mlp_classifier(features, labels, device="cpu")
        assert 0.0 <= result["original_features"]["test_accuracy"] <= 1.0
        assert 0.0 <= result["summary"]["best_reduced_accuracy"] <= 1.0

    @pytest.mark.slow
    @patch("part2.umap.UMAP")
    @patch("part2.tqdm")
    def test_reduced_features_methods(
        self, mock_tqdm, mock_umap, sample_4096_features, sample_labels, outputs_dir
    ):
        mock_tqdm.return_value = iter(range(100))
        mock_umap_instance = MagicMock()
        mock_umap_instance.fit_transform.return_value = np.random.randn(40, 50).astype(
            np.float32
        )
        mock_umap_instance.transform.return_value = np.random.randn(8, 50).astype(
            np.float32
        )
        mock_umap.return_value = mock_umap_instance
        features = sample_4096_features[:40]
        labels = sample_labels[:40]
        with patch("part2.Path") as mock_path:
            mock_path.return_value = outputs_dir
            result = run_mlp_classifier(features, labels, device="cpu")
        assert "svd" in result["reduced_features"]
        assert "umap" in result["reduced_features"]
        assert "autoencoder" in result["reduced_features"]
