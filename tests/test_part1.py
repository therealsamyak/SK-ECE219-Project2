import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from part1 import (
    get_device,
    apply_dimensionality_reduction,
    run_clustering_pipeline,
    compute_clustering_metrics,
    create_length_labels,
)


class TestGetDevice:
    def test_returns_valid_device(self):
        device = get_device()
        assert device in ["cuda", "mps", "cpu"]


class TestApplyDimensionalityReduction:
    def test_none_returns_array(self, sample_features):
        result = apply_dimensionality_reduction(sample_features, "none")
        assert result.shape == sample_features.shape

    def test_svd_reduces_dimensions(self, sample_features):
        result = apply_dimensionality_reduction(sample_features, "svd", n_components=10)
        assert result.shape == (50, 10)

    def test_umap_reduces_dimensions(self, sample_features):
        result = apply_dimensionality_reduction(
            sample_features, "umap", n_components=10
        )
        assert result.shape == (50, 10)


class TestRunClusteringPipeline:
    @patch("part1.KMeans")
    def test_kmeans_returns_labels(self, mock_kmeans, sample_features):
        mock_model = MagicMock()
        mock_model.fit_predict.return_value = np.array([0] * 25 + [1] * 25)
        mock_kmeans.return_value = mock_model

        labels, model = run_clustering_pipeline(sample_features, "kmeans", n_clusters=2)
        assert len(labels) == 50
        assert set(labels).issubset({0, 1})

    @patch("part1.AgglomerativeClustering")
    def test_agglomerative_returns_labels(self, mock_agg, sample_features):
        mock_model = MagicMock()
        mock_model.fit_predict.return_value = np.array([0] * 25 + [1] * 25)
        mock_agg.return_value = mock_model

        labels, model = run_clustering_pipeline(
            sample_features, "agglomerative", n_clusters=2
        )
        assert len(labels) == 50

    @patch("part1.HDBSCAN")
    def test_hdbscan_returns_labels(self, mock_hdbscan, sample_features):
        mock_model = MagicMock()
        mock_model.fit_predict.return_value = np.array([0] * 25 + [1] * 25)
        mock_hdbscan.return_value = mock_model

        labels, model = run_clustering_pipeline(sample_features, "hdbscan")
        assert len(labels) == 50


class TestComputeClusteringMetrics:
    def test_perfect_clustering(self):
        labels_true = np.array([0, 0, 1, 1] * 12 + [0, 2])
        labels_pred = labels_true.copy()
        metrics = compute_clustering_metrics(labels_true, labels_pred)
        assert metrics["ari"] > 0.9

    def test_metrics_in_valid_range(self, sample_labels):
        metrics = compute_clustering_metrics(sample_labels, sample_labels)
        for key, value in metrics.items():
            assert 0.0 <= value <= 1.0


class TestCreateLengthLabels:
    @patch("part1.pd.read_csv")
    @patch("part1.setup_logging")
    def test_returns_dataframe_with_length_label(
        self, mock_setup_logging, mock_read_csv
    ):
        mock_df = pd.DataFrame(
            {
                "review_text": ["short", "very long review text with many words here"]
                * 25,
                "user": [f"user{i}" for i in range(50)],
                "recommend": [True] * 50,
            }
        )
        mock_read_csv.return_value = mock_df
        mock_setup_logging.return_value = MagicMock()

        result = create_length_labels()
        assert "length_label" in result.columns
        assert set(result["length_label"].unique()).issubset({"Short", "Long"})

    @patch("part1.pd.read_csv")
    @patch("part1.setup_logging")
    def test_filters_by_word_count(self, mock_setup_logging, mock_read_csv):
        mock_df = pd.DataFrame(
            {
                "review_text": ["a"] * 20 + ["a b c d e f"] * 30,
                "user": [f"user{i}" for i in range(50)],
                "recommend": [True] * 50,
            }
        )
        mock_read_csv.return_value = mock_df
        mock_setup_logging.return_value = MagicMock()

        result = create_length_labels()
        assert len(result) > 0
        assert len(result) <= 50
