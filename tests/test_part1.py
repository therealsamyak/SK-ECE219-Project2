import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from part1 import (
    create_length_labels,
    run_task1_1,
    run_task1_2,
    plot_pca_visualizations,
)


class TestTask1_1:
    """Tests for create_length_labels() function"""

    @pytest.fixture
    def sample_data(self):
        """Create sample review data for testing"""
        data = {
            "user": [f"user{i}" for i in range(100)],
            "playtime": np.random.uniform(0, 1000, 100),
            "post_date": ["Nov 1, 2023"] * 100,
            "helpfulness": np.random.randint(0, 500, 100),
            "review_text": [
                "Very short",
                "This is a medium length review",
                "This is a very long review with many words that should definitely exceed the threshold",
            ]
            * 33
            + ["Another one"],
            "recommend": [True] * 100,
            "early_access_review": [""] * 100,
            "appid": [123456] * 100,
            "game_name": ["Test Game"] * 100,
            "release_date": ["Jan 1, 2023"] * 100,
            "genres": ["Action"] * 100,
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_csv(self, sample_data, tmp_path):
        """Create a temporary CSV file for testing"""
        csv_path = tmp_path / "test_reviews.csv"
        sample_data.to_csv(csv_path, index=False)
        return str(csv_path)

    def test_function_returns_dataframe(self):
        """Test that function returns a DataFrame"""
        result = create_length_labels()
        assert isinstance(result, pd.DataFrame)

    def test_dataframe_has_length_label_column(self):
        """Test that result DataFrame contains 'length_label' column"""
        result = create_length_labels()
        assert "length_label" in result.columns

    def test_length_label_values_are_short_or_long(self):
        """Test that length_label only contains 'Short' or 'Long' values"""
        result = create_length_labels()
        unique_labels = result["length_label"].unique()
        assert set(unique_labels).issubset({"Short", "Long"})
        assert len(unique_labels) <= 2

    def test_filtering_removes_middle_50_percent(self):
        """Test that middle 50% of reviews are filtered out"""
        result = create_length_labels()
        assert len(result) < 35000
        assert len(result) > 10000

    def test_logging_outputs_to_file(self):
        """Test that logs are written to outputs/part1.log"""
        log_file = Path("outputs/part1.log")
        create_length_labels()
        assert log_file.exists()
        content = log_file.read_text()
        assert len(content) > 0

    def test_logging_contains_thresholds(self):
        """Test that logs contain q25 and q75 thresholds"""
        log_file = Path("outputs/part1.log")
        create_length_labels()
        content = log_file.read_text()
        assert "q25" in content.lower() or "25th" in content.lower()
        assert "q75" in content.lower() or "75th" in content.lower()

    def test_logging_contains_statistics(self):
        """Test that logs contain filtering statistics"""
        log_file = Path("outputs/part1.log")
        create_length_labels()
        content = log_file.read_text()
        assert "total" in content.lower() or "reviews" in content.lower()

    def test_dataset_has_correct_columns(self, sample_csv):
        """Test that DataFrame preserves original columns"""
        result = create_length_labels()
        expected_columns = [
            "user",
            "playtime",
            "post_date",
            "helpfulness",
            "review_text",
            "recommend",
            "early_access_review",
            "appid",
            "game_name",
            "release_date",
            "genres",
            "length_label",
        ]
        for col in expected_columns:
            assert col in result.columns, f"Column {col} missing from result"

    def test_short_reviews_have_fewer_words_than_long_reviews(self):
        """Test that 'Short' labeled reviews have fewer words than 'Long' ones"""
        result = create_length_labels()
        if (
            len(result) > 0
            and "Short" in result["length_label"].values
            and "Long" in result["length_label"].values
        ):
            result["word_count"] = (
                result["review_text"].astype(str).str.split().str.len()
            )
            short_mean = result[result["length_label"] == "Short"]["word_count"].mean()
            long_mean = result[result["length_label"] == "Long"]["word_count"].mean()
            assert short_mean < long_mean, (
                "Short reviews should have fewer words on average"
            )

    def test_reproducibility_same_result_on_multiple_calls(self):
        """Test that function returns same result on multiple calls (reproducibility)"""
        result1 = create_length_labels()
        result2 = create_length_labels()
        assert len(result1) == len(result2)
        assert (
            result1["length_label"].value_counts().to_dict()
            == result2["length_label"].value_counts().to_dict()
        )


class TestTask1_2:
    """Tests for run_task1_1() function - Representations"""

    def test_function_returns_dict(self):
        """Test that function returns a dictionary with results"""
        result = run_task1_1()
        assert isinstance(result, dict)

    def test_result_contains_required_keys(self):
        """Test that result dictionary contains all required keys"""
        result = run_task1_1()
        required_keys = ["tfidf", "minilm", "dataset"]
        for key in required_keys:
            assert key in result, f"Key {key} missing from result"

    def test_tfidf_result_structure(self):
        """Test that TF-IDF result has correct structure"""
        result = run_task1_1()
        tfidf = result["tfidf"]
        assert "matrix_shape" in tfidf
        assert "vocabulary_size" in tfidf
        assert isinstance(tfidf["matrix_shape"], tuple)
        assert len(tfidf["matrix_shape"]) == 2

    def test_minilm_result_structure(self):
        """Test that MiniLM result has correct structure"""
        result = run_task1_1()
        minilm = result["minilm"]
        assert "matrix_shape" in minilm
        assert "embedding_dim" in minilm
        assert "normalized" in minilm
        assert isinstance(minilm["matrix_shape"], tuple)
        assert len(minilm["matrix_shape"]) == 2
        assert minilm["normalized"] is True

    def test_dataset_result_structure(self):
        """Test that dataset result contains correct info"""
        result = run_task1_1()
        dataset = result["dataset"]
        assert "size" in dataset
        assert "num_short" in dataset
        assert "num_long" in dataset
        assert dataset["size"] > 0
        assert dataset["num_short"] > 0
        assert dataset["num_long"] > 0

    def test_tfidf_matrix_shape_matches_dataset(self):
        """Test that TF-IDF matrix first dimension matches dataset size"""
        result = run_task1_1()
        assert result["tfidf"]["matrix_shape"][0] == result["dataset"]["size"]

    def test_minilm_matrix_shape_matches_dataset(self):
        """Test that MiniLM matrix first dimension matches dataset size"""
        result = run_task1_1()
        assert result["minilm"]["matrix_shape"][0] == result["dataset"]["size"]

    def test_minilm_embedding_dimension(self):
        """Test that MiniLM embeddings have correct dimension (384 for L6-v2)"""
        result = run_task1_1()
        assert result["minilm"]["embedding_dim"] == 384

    def test_json_files_created(self):
        """Test that all required JSON files are created in outputs/"""
        output_dir = Path("outputs")
        run_task1_1()
        required_files = [
            output_dir / "task1_1_results.json",
            output_dir / "task1_1_tfidf.json",
            output_dir / "task1_1_minilm.json",
        ]
        for file_path in required_files:
            assert file_path.exists(), f"JSON file {file_path} not created"

    def test_json_results_content(self):
        """Test that task1_1_results.json has correct structure"""
        run_task1_1()
        json_path = Path("outputs/task1_1_results.json")
        with open(json_path) as f:
            data = json.load(f)
        assert "dataset" in data
        assert "tfidf" in data
        assert "minilm" in data

    def test_json_tfidf_content(self):
        """Test that task1_1_tfidf.json has correct structure"""
        run_task1_1()
        json_path = Path("outputs/task1_1_tfidf.json")
        with open(json_path) as f:
            data = json.load(f)
        assert "matrix_shape" in data
        assert "vocabulary_size" in data
        assert "sparsity" in data

    def test_json_minilm_content(self):
        """Test that task1_1_minilm.json has correct structure"""
        run_task1_1()
        json_path = Path("outputs/task1_1_minilm.json")
        with open(json_path) as f:
            data = json.load(f)
        assert "matrix_shape" in data
        assert "embedding_dim" in data
        assert "normalized" in data

    def test_logging_contains_tfidf_dimensions(self):
        """Test that logs contain TF-IDF matrix dimensions"""
        log_file = Path("outputs/part1.log")
        run_task1_1()
        content = log_file.read_text()
        assert "tfidf" in content.lower() or "TF-IDF" in content

    def test_logging_contains_minilm_dimensions(self):
        """Test that logs contain MiniLM matrix dimensions"""
        log_file = Path("outputs/part1.log")
        run_task1_1()
        content = log_file.read_text()
        assert "minilm" in content.lower() or "MiniLM" in content

    def test_logging_contains_vocabulary_size(self):
        """Test that logs contain vocabulary size"""
        log_file = Path("outputs/part1.log")
        run_task1_1()
        content = log_file.read_text()
        assert "vocabular" in content.lower() or "vocabulary" in content.lower()

    def test_reproducibility_same_results_on_multiple_calls(self):
        """Test that function returns same results on multiple calls"""
        result1 = run_task1_1()
        result2 = run_task1_1()
        assert result1["dataset"]["size"] == result2["dataset"]["size"]
        assert result1["tfidf"]["matrix_shape"] == result2["tfidf"]["matrix_shape"]
        assert result1["minilm"]["matrix_shape"] == result2["minilm"]["matrix_shape"]


class TestTask1_3:
    """Tests for run_task1_2() function - Clustering Pipelines"""

    def test_function_returns_dict(self):
        """Test that function returns a dictionary with results"""
        result = run_task1_2()
        assert isinstance(result, dict)

    def test_result_contains_required_keys(self):
        """Test that result dictionary contains all required keys"""
        result = run_task1_2()
        required_keys = ["tfidf", "minilm", "summary"]
        for key in required_keys:
            assert key in result, f"Key {key} missing from result"

    def test_tfidf_results_structure(self):
        """Test that TF-IDF results have correct structure"""
        result = run_task1_2()
        tfidf_results = result["tfidf"]["results"]
        assert len(tfidf_results) == 2
        for pipeline in tfidf_results:
            assert "pipeline_id" in pipeline
            assert "dim_reduction" in pipeline
            assert "clustering" in pipeline
            assert "metrics" in pipeline
            assert "homogeneity" in pipeline["metrics"]
            assert "completeness" in pipeline["metrics"]
            assert "v_measure" in pipeline["metrics"]
            assert "ari" in pipeline["metrics"]
            assert "ami" in pipeline["metrics"]

    def test_minilm_results_structure(self):
        """Test that MiniLM results have correct structure"""
        result = run_task1_2()
        minilm_results = result["minilm"]["results"]
        assert len(minilm_results) == 7
        for pipeline in minilm_results:
            assert "pipeline_id" in pipeline
            assert "dim_reduction" in pipeline
            assert "clustering" in pipeline
            assert "metrics" in pipeline

    def test_tfidf_metrics_are_float(self):
        """Test that TF-IDF clustering metrics are float values"""
        result = run_task1_2()
        tfidf_results = result["tfidf"]["results"]
        for pipeline in tfidf_results:
            metrics = pipeline["metrics"]
            for metric_name in [
                "homogeneity",
                "completeness",
                "v_measure",
                "ari",
                "ami",
            ]:
                assert isinstance(metrics[metric_name], float), (
                    f"TF-IDF {metric_name} should be float"
                )
                assert 0 <= metrics[metric_name] <= 1, (
                    f"TF-IDF {metric_name} should be between 0 and 1"
                )

    def test_summary_structure(self):
        """Test that summary has correct structure"""
        result = run_task1_2()
        summary = result["summary"]
        assert "total_pipelines" in summary
        assert "tfidf_pipelines" in summary
        assert "minilm_pipelines" in summary
        assert "dataset_size" in summary
        assert "ground_truth_distribution" in summary
        assert summary["total_pipelines"] == 9
        assert summary["tfidf_pipelines"] == 2
        assert summary["minilm_pipelines"] == 7

    def test_json_files_created(self):
        """Test that all required JSON files are created in outputs/"""
        output_dir = Path("outputs")
        run_task1_2()
        required_files = [
            output_dir / "task1_2_clustering_results.json",
            output_dir / "task1_2_tfidf_clustering.json",
            output_dir / "task1_2_minilm_clustering.json",
        ]
        for file_path in required_files:
            assert file_path.exists(), f"JSON file {file_path} not created"

    def test_json_results_content(self):
        """Test that task1_2_clustering_results.json has correct structure"""
        run_task1_2()
        json_path = Path("outputs/task1_2_clustering_results.json")
        with open(json_path) as f:
            data = json.load(f)
        assert "tfidf" in data
        assert "minilm" in data
        assert "summary" in data

    def test_json_tfidf_content(self):
        """Test that task1_2_tfidf_clustering.json has correct structure"""
        run_task1_2()
        json_path = Path("outputs/task1_2_tfidf_clustering.json")
        with open(json_path) as f:
            data = json.load(f)
        assert "representation" in data
        assert "pipelines" in data
        assert len(data["pipelines"]) == 2

    def test_json_minilm_content(self):
        """Test that task1_2_minilm_clustering.json has correct structure"""
        run_task1_2()
        json_path = Path("outputs/task1_2_minilm_clustering.json")
        with open(json_path) as f:
            data = json.load(f)
        assert "representation" in data
        assert "pipelines" in data
        assert len(data["pipelines"]) == 7

    def test_reproducibility_same_results_on_multiple_calls(self):
        """Test that function returns same results on multiple calls"""
        result1 = run_task1_2()
        result2 = run_task1_2()
        assert result1["summary"]["dataset_size"] == result2["summary"]["dataset_size"]
        assert len(result1["tfidf"]["results"]) == len(result2["tfidf"]["results"])
        assert len(result1["minilm"]["results"]) == len(result2["minilm"]["results"])

    def test_dim_reduction_methods_in_configs(self):
        """Test that dimensionality reduction methods are correctly configured"""
        result = run_task1_2()

        tfidf_dims = {r["dim_reduction"] for r in result["tfidf"]["results"]}
        assert tfidf_dims == {"svd"}

        minilm_dims = {r["dim_reduction"] for r in result["minilm"]["results"]}
        assert minilm_dims == {"none", "svd", "umap"}

    def test_clustering_methods_in_configs(self):
        """Test that clustering methods are correctly configured"""
        result = run_task1_2()

        all_methods = set()
        for rep in ["tfidf", "minilm"]:
            all_methods.update({r["clustering"] for r in result[rep]["results"]})

        assert "kmeans" in all_methods
        assert "agglomerative" in all_methods
        assert "hdbscan" in all_methods


class TestTask1_4:
    """Tests for plot_pca_visualizations() function"""

    def test_function_returns_dict(self):
        """Test that function returns a dictionary with results"""
        result = plot_pca_visualizations()
        assert isinstance(result, dict)

    def test_function_with_param_returns_dict(self):
        """Test that function returns dict with clustering disabled parameter"""
        result = plot_pca_visualizations(use_clustering_results_if_available=False)
        assert isinstance(result, dict)

    def test_result_contains_required_keys(self):
        """Test that result dictionary contains all required keys"""
        result = plot_pca_visualizations()
        required_keys = ["plot_path", "best_pipelines", "tfidf_pca", "minilm_pca"]
        for key in required_keys:
            assert key in result, f"Key {key} missing from result"

    def test_result_contains_required_keys_without_clustering(self):
        """Test that result dictionary contains all required keys when clustering disabled"""
        result = plot_pca_visualizations(use_clustering_results_if_available=False)
        required_keys = ["plot_path", "best_pipelines", "tfidf_pca", "minilm_pca"]
        for key in required_keys:
            assert key in result, f"Key {key} missing from result"

    def test_plot_path_exists(self):
        """Test that plot file is created"""
        result = plot_pca_visualizations()
        plot_path = Path(result["plot_path"])
        assert plot_path.exists()

    def test_plot_path_is_png(self):
        """Test that plot is saved as PNG"""
        result = plot_pca_visualizations()
        plot_path = Path(result["plot_path"])
        assert plot_path.suffix == ".png"

    def test_best_pipelines_structure(self):
        """Test that best_pipelines has correct structure"""
        result = plot_pca_visualizations()
        best_pipelines = result["best_pipelines"]
        assert isinstance(best_pipelines, dict)
        for rep in ["tfidf", "minilm"]:
            assert rep in best_pipelines
            if best_pipelines[rep] is not None:
                assert "dim_reduction" in best_pipelines[rep]
                assert "clustering" in best_pipelines[rep]
                assert "v_measure" in best_pipelines[rep]
                assert "ari" in best_pipelines[rep]

    def test_best_pipelines_without_clustering_results(self):
        """Test that best_pipelines uses default SVD+K-Means when no results"""
        result = plot_pca_visualizations(use_clustering_results_if_available=False)
        best_pipelines = result["best_pipelines"]
        for rep in ["tfidf", "minilm"]:
            assert rep in best_pipelines
            assert best_pipelines[rep]["dim_reduction"] == "svd"
            assert best_pipelines[rep]["clustering"] == "kmeans"
            assert best_pipelines[rep]["v_measure"] is None

    def test_pca_variance_structure(self):
        """Test that PCA variance information has correct structure"""
        result = plot_pca_visualizations()
        for rep in ["tfidf_pca", "minilm_pca"]:
            pca_info = result[rep]
            assert "explained_variance_ratio" in pca_info
            assert "total_variance_explained" in pca_info
            assert isinstance(pca_info["explained_variance_ratio"], list)
            assert len(pca_info["explained_variance_ratio"]) == 2
            assert isinstance(pca_info["total_variance_explained"], float)
            assert 0 <= pca_info["total_variance_explained"] <= 1

    def test_variance_ratio_is_positive(self):
        """Test that explained variance ratios are positive"""
        result = plot_pca_visualizations()
        for rep in ["tfidf_pca", "minilm_pca"]:
            for ratio in result[rep]["explained_variance_ratio"]:
                assert ratio >= 0

    def test_v_measure_is_between_0_and_1(self):
        """Test that V-measure values are between 0 and 1"""
        result = plot_pca_visualizations()
        best_pipelines = result["best_pipelines"]
        for rep in ["tfidf", "minilm"]:
            if (
                best_pipelines[rep] is not None
                and best_pipelines[rep]["v_measure"] is not None
            ):
                v_measure = best_pipelines[rep]["v_measure"]
                assert 0 <= v_measure <= 1

    def test_ari_is_between_0_and_1(self):
        """Test that ARI values are between -1 and 1"""
        result = plot_pca_visualizations()
        best_pipelines = result["best_pipelines"]
        for rep in ["tfidf", "minilm"]:
            if (
                best_pipelines[rep] is not None
                and best_pipelines[rep]["ari"] is not None
            ):
                ari = best_pipelines[rep]["ari"]
                assert -1 <= ari <= 1

    def test_json_files_created(self):
        """Test that task1_4_results.json is created in outputs/"""
        output_dir = Path("outputs")
        plot_pca_visualizations()
        json_path = output_dir / "task1_4_results.json"
        assert json_path.exists()

    def test_json_results_content(self):
        """Test that task1_4_results.json has correct structure"""
        plot_pca_visualizations()
        json_path = Path("outputs/task1_4_results.json")
        with open(json_path) as f:
            data = json.load(f)
        assert "plot_path" in data
        assert "best_pipelines" in data
        assert "tfidf_pca" in data
        assert "minilm_pca" in data

    def test_reproducibility_same_results_on_multiple_calls(self):
        """Test that function returns same results on multiple calls"""
        result1 = plot_pca_visualizations(use_clustering_results_if_available=False)
        result2 = plot_pca_visualizations(use_clustering_results_if_available=False)
        assert result1["best_pipelines"] == result2["best_pipelines"]
        assert result1["tfidf_pca"] == result2["tfidf_pca"]
        assert result1["minilm_pca"] == result2["minilm_pca"]
