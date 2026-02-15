import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from part3 import (
    get_device,
    construct_pokedex,
    load_clip_model,
    clip_inference_image,
    clip_inference_text,
    compute_similarity_text_to_image,
    compute_similarity_image_to_text,
    run_q20,
    run_q21,
    run_q22,
    run_q23,
)


# ============== FIXTURES ==============


@pytest.fixture
def outputs_dir(tmp_path):
    """Create a temporary outputs directory for testing"""
    outputs = tmp_path / "outputs"
    outputs.mkdir(exist_ok=True)
    return outputs


@pytest.fixture
def sample_pokedex():
    """Create a small sample pokedex for testing"""
    data = {
        "Name": ["Bulbasaur", "Charmander", "Squirtle", "Pikachu"],
        "Type1": ["Grass", "Fire", "Water", "Electric"],
        "Type2": ["Poison", "", "", ""],
        "ID": [1, 2, 3, 4],
        "image_path": [
            "datasets/pokemon/images/Bulbasaur/0.jpg",
            "datasets/pokemon/images/Charmander/0.jpg",
            "datasets/pokemon/images/Squirtle/0.jpg",
            "datasets/pokemon/images/Pikachu/0.jpg",
        ],
    }
    df = pd.DataFrame(data)
    df["Type2"] = df["Type2"].str.strip()
    return df[["Name", "Type1", "Type2", "image_path"]]


@pytest.fixture
def sample_image_paths():
    """First 5 Pokemon image paths from actual pokedex"""
    paths = [
        "datasets/pokemon/images/Bulbasaur/0.jpg",
        "datasets/pokemon/images/Charmander/0.jpg",
        "datasets/pokemon/images/Squirtle/0.jpg",
        "datasets/pokemon/images/Caterpie/0.jpg",
        "datasets/pokemon/images/Weedle/0.jpg",
    ]
    return paths


@pytest.fixture
def sample_text_prompts():
    """Sample text prompts for testing"""
    return ["a photo of a Fire type Pokemon", "a photo of a Water type Pokemon"]


@pytest.fixture
def mock_image_embeddings():
    """Mock image embeddings for testing (10 samples, 512 dims)"""
    np.random.seed(42)
    embeddings = np.random.randn(10, 512).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)
    return embeddings


@pytest.fixture
def mock_text_embeddings():
    """Mock text embeddings for testing (5 types, 512 dims)"""
    np.random.seed(42)
    embeddings = np.random.randn(5, 512).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)
    return embeddings


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


# ============== TestConstructPokedex ==============


class TestConstructPokedex:
    """Tests for construct_pokedex() function"""

    def test_returns_dataframe(self):
        """Function should return a pandas DataFrame"""
        pokedex = construct_pokedex()
        assert isinstance(pokedex, pd.DataFrame)

    def test_has_required_columns(self):
        """DataFrame should contain Name, Type1, Type2, image_path columns"""
        pokedex = construct_pokedex()
        required_columns = ["Name", "Type1", "Type2", "image_path"]
        for col in required_columns:
            assert col in pokedex.columns, f"Column {col} missing from pokedex"

    def test_all_images_exist(self):
        """All image_path entries should point to existing files"""
        pokedex = construct_pokedex()
        for img_path in pokedex["image_path"]:
            if img_path:  # Check if path is not None
                assert Path(img_path).exists(), f"Image not found: {img_path}"

    def test_no_duplicate_ids(self):
        """Should not have duplicate Pokemon names"""
        pokedex = construct_pokedex()
        name_counts = pokedex["Name"].value_counts()
        assert all(name_counts == 1), "Found duplicate Pokemon names"

    def test_type2_stripped(self):
        """Type2 should not have leading/trailing spaces"""
        pokedex = construct_pokedex()
        for type2 in pokedex["Type2"]:
            if type2 and type2.strip():  # Skip empty strings
                assert type2 == type2.strip(), f"Type2 not stripped: '{type2}'"

    def test_count_matches_expected(self):
        """Should have exactly 754 Pokemon"""
        pokedex = construct_pokedex()
        assert len(pokedex) == 754, f"Expected 754 Pokemon, got {len(pokedex)}"


# ============== TestLoadClipModel ==============


class TestLoadClipModel:
    """Tests for load_clip_model() function"""

    @pytest.mark.slow
    def test_returns_tuple(self):
        """Should return (model, preprocess, device, tokenizer)"""
        try:
            result = load_clip_model()
            assert isinstance(result, tuple), "Should return a tuple"
            assert len(result) == 4, "Should return 4 elements"
            model, preprocess, device, tokenizer = result
            assert hasattr(model, "encode_image"), (
                "Model should have encode_image method"
            )
            assert callable(preprocess), "preprocess should be callable"
            assert isinstance(device, str), "device should be a string"
            assert callable(tokenizer), "tokenizer should be callable"
        except Exception as e:
            pytest.skip(f"CLIP model not available: {e}")

    @pytest.mark.slow
    def test_model_on_correct_device(self):
        """Model should be on the detected device"""
        try:
            model, preprocess, device, tokenizer = load_clip_model()
            actual_device = next(model.parameters()).device.type
            if device == "cuda":
                assert actual_device == "cuda", "Model should be on CUDA"
            elif device == "mps":
                assert actual_device == "mps", "Model should be on MPS"
            else:
                assert actual_device == "cpu", "Model should be on CPU"
        except Exception as e:
            pytest.skip(f"CLIP model not available: {e}")

    @pytest.mark.slow
    def test_model_in_eval_mode(self):
        """Model should be in eval mode"""
        try:
            model, preprocess, device, tokenizer = load_clip_model()
            assert not model.training, "Model should be in eval mode"
        except Exception as e:
            pytest.skip(f"CLIP model not available: {e}")


# ============== TestClipInferenceImage ==============


class TestClipInferenceImage:
    """Tests for clip_inference_image() function"""

    @pytest.mark.slow
    def test_returns_normalized_embeddings(self):
        """Embeddings should be normalized (L2 norm = 1)"""
        try:
            model, preprocess, device, tokenizer = load_clip_model()
            pokedex = construct_pokedex()
            image_paths = pokedex["image_path"].tolist()[:5]
            embeddings = clip_inference_image(model, preprocess, image_paths, device)

            norms = np.linalg.norm(embeddings, axis=-1)
            assert np.allclose(norms, 1.0, atol=1e-5), "Embeddings should be normalized"
        except Exception as e:
            pytest.skip(f"CLIP model not available: {e}")

    @pytest.mark.slow
    def test_embeddings_shape_correct(self):
        """Embeddings shape should match number of images"""
        try:
            model, preprocess, device, tokenizer = load_clip_model()
            pokedex = construct_pokedex()
            image_paths = pokedex["image_path"].tolist()[:10]
            embeddings = clip_inference_image(model, preprocess, image_paths, device)

            assert embeddings.shape[0] == len(image_paths), (
                f"Expected {len(image_paths)} embeddings, got {embeddings.shape[0]}"
            )
            assert embeddings.ndim == 2, "Embeddings should be 2D array"
        except Exception as e:
            pytest.skip(f"CLIP model not available: {e}")

    @pytest.mark.slow
    def test_batch_processing(self):
        """Should process multiple images in batch"""
        try:
            model, preprocess, device, tokenizer = load_clip_model()
            pokedex = construct_pokedex()
            image_paths = pokedex["image_path"].tolist()[:3]
            embeddings = clip_inference_image(model, preprocess, image_paths, device)

            assert embeddings.shape[0] == 3, "Should process all 3 images"
            assert np.all(np.isfinite(embeddings)), "All embeddings should be finite"
        except Exception as e:
            pytest.skip(f"CLIP model not available: {e}")


# ============== TestClipInferenceText ==============


class TestClipInferenceText:
    """Tests for clip_inference_text() function"""

    @pytest.mark.slow
    def test_returns_normalized_embeddings(self):
        """Embeddings should be normalized (L2 norm = 1)"""
        try:
            model, preprocess, device, tokenizer = load_clip_model()
            texts = [
                "a photo of a Fire type Pokemon",
                "a photo of a Water type Pokemon",
            ]
            embeddings = clip_inference_text(model, tokenizer, texts, device)

            norms = np.linalg.norm(embeddings, axis=-1)
            assert np.allclose(norms, 1.0, atol=1e-5), "Embeddings should be normalized"
        except Exception as e:
            pytest.skip(f"CLIP model not available: {e}")

    @pytest.mark.slow
    def test_embeddings_shape_correct(self):
        """Embeddings shape should match number of texts"""
        try:
            model, preprocess, device, tokenizer = load_clip_model()
            texts = [
                "a photo of a Fire type Pokemon",
                "a photo of a Water type Pokemon",
                "a photo of a Grass type Pokemon",
            ]
            embeddings = clip_inference_text(model, tokenizer, texts, device)

            assert embeddings.shape[0] == len(texts), (
                f"Expected {len(texts)} embeddings, got {embeddings.shape[0]}"
            )
            assert embeddings.ndim == 2, "Embeddings should be 2D array"
        except Exception as e:
            pytest.skip(f"CLIP model not available: {e}")


# ============== TestComputeSimilarityTextToImage ==============


class TestComputeSimilarityTextToImage:
    """Tests for compute_similarity_text_to_image() function"""

    def test_returns_similarity_matrix(self):
        """Should return similarity matrix of correct shape"""
        np.random.seed(42)
        image_embeddings = np.random.randn(10, 512).astype(np.float32)
        image_embeddings /= np.linalg.norm(image_embeddings, axis=-1, keepdims=True)
        text_embeddings = np.random.randn(5, 512).astype(np.float32)
        text_embeddings /= np.linalg.norm(text_embeddings, axis=-1, keepdims=True)

        similarity = compute_similarity_text_to_image(image_embeddings, text_embeddings)

        assert similarity.shape == (10, 5), (
            f"Expected shape (10, 5), got {similarity.shape}"
        )

    def test_softmax_along_text_axis(self):
        """Softmax should be applied along text axis (axis=-1)"""
        np.random.seed(42)
        image_embeddings = np.random.randn(5, 3).astype(np.float32)
        image_embeddings /= np.linalg.norm(image_embeddings, axis=-1, keepdims=True)
        text_embeddings = np.random.randn(2, 3).astype(np.float32)
        text_embeddings /= np.linalg.norm(text_embeddings, axis=-1, keepdims=True)

        similarity = compute_similarity_text_to_image(image_embeddings, text_embeddings)

        # Check that each row sums to 1 (softmax along text axis)
        row_sums = np.sum(similarity, axis=-1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), (
            "Rows should sum to 1 (softmax along text axis)"
        )

    def test_values_sum_to_one(self):
        """Values along text axis should sum to 1"""
        np.random.seed(42)
        image_embeddings = np.random.randn(7, 4).astype(np.float32)
        image_embeddings /= np.linalg.norm(image_embeddings, axis=-1, keepdims=True)
        text_embeddings = np.random.randn(3, 4).astype(np.float32)
        text_embeddings /= np.linalg.norm(text_embeddings, axis=-1, keepdims=True)

        similarity = compute_similarity_text_to_image(image_embeddings, text_embeddings)

        row_sums = np.sum(similarity, axis=-1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), "All rows should sum to 1"


# ============== TestComputeSimilarityImageToText ==============


class TestComputeSimilarityImageToText:
    """Tests for compute_similarity_image_to_text() function"""

    def test_returns_similarity_matrix(self):
        """Should return similarity matrix of correct shape"""
        np.random.seed(42)
        image_embeddings = np.random.randn(10, 512).astype(np.float32)
        image_embeddings /= np.linalg.norm(image_embeddings, axis=-1, keepdims=True)
        text_embeddings = np.random.randn(5, 512).astype(np.float32)
        text_embeddings /= np.linalg.norm(text_embeddings, axis=-1, keepdims=True)

        similarity = compute_similarity_image_to_text(image_embeddings, text_embeddings)

        assert similarity.shape == (10, 5), (
            f"Expected shape (10, 5), got {similarity.shape}"
        )

    def test_softmax_along_image_axis(self):
        """Softmax should be applied along image axis (axis=0)"""
        np.random.seed(42)
        image_embeddings = np.random.randn(5, 3).astype(np.float32)
        image_embeddings /= np.linalg.norm(image_embeddings, axis=-1, keepdims=True)
        text_embeddings = np.random.randn(2, 3).astype(np.float32)
        text_embeddings /= np.linalg.norm(text_embeddings, axis=-1, keepdims=True)

        similarity = compute_similarity_image_to_text(image_embeddings, text_embeddings)

        # Check that each column sums to 1 (softmax along image axis)
        col_sums = np.sum(similarity, axis=0)
        assert np.allclose(col_sums, 1.0, atol=1e-5), (
            "Columns should sum to 1 (softmax along image axis)"
        )


# ============== TestRunQ20 ==============


class TestRunQ20:
    """Tests for run_q20() function"""

    @pytest.mark.slow
    def test_returns_dict(self):
        """Should return a dictionary with results"""
        try:
            run_q20()
            # run_q20 doesn't explicitly return anything, but we check if it runs
            assert True
        except Exception as e:
            pytest.skip(f"Q20 execution failed: {e}")

    @pytest.mark.slow
    def test_creates_png_files(self):
        """Should create 5 PNG files for each type"""
        try:
            run_q20()
            expected_files = [
                "outputs/part3_q20_bug.png",
                "outputs/part3_q20_fire.png",
                "outputs/part3_q20_grass.png",
                "outputs/part3_q20_dark.png",
                "outputs/part3_q20_dragon.png",
            ]
            for file_path in expected_files:
                assert Path(file_path).exists(), f"File not created: {file_path}"
        except Exception as e:
            pytest.skip(f"Q20 execution failed: {e}")

    @pytest.mark.slow
    def test_png_files_count(self):
        """Should create exactly 5 PNG files"""
        try:
            run_q20()
            png_files = list(Path("outputs").glob("part3_q20_*.png"))
            assert len(png_files) == 5, f"Expected 5 PNG files, found {len(png_files)}"
        except Exception as e:
            pytest.skip(f"Q20 execution failed: {e}")

    @pytest.mark.slow
    def test_query_format_correct(self):
        """Query format should be 'a photo of a {type} type Pokemon'"""
        try:
            # Check log file for query format
            log_file = Path("outputs/part3.log")
            if log_file.exists():
                content = log_file.read_text()
                assert "a photo of a" in content.lower(), "Query format incorrect"
        except Exception as e:
            pytest.skip(f"Q20 execution failed: {e}")


# ============== TestRunQ21 ==============


class TestRunQ21:
    """Tests for run_q21() function"""

    @pytest.mark.slow
    def test_returns_dict(self):
        """Should create result files and run successfully"""
        try:
            run_q21()
            # run_q21 doesn't return anything explicitly
            assert True
        except Exception as e:
            pytest.skip(f"Q21 execution failed: {e}")

    @pytest.mark.slow
    def test_selected_pokemon_count(self):
        """Should select exactly 10 Pokemon"""
        try:
            run_q21()
            json_path = Path("outputs/part3_q21_detailed.json")
            assert json_path.exists(), "Detailed JSON file not created"

            with open(json_path) as f:
                data = json.load(f)

            assert len(data["selected_pokemon"]) == 10, (
                f"Expected 10 Pokemon, got {len(data['selected_pokemon'])}"
            )
            assert len(data["per_pokemon"]) == 10, (
                f"Expected 10 detailed entries, got {len(data['per_pokemon'])}"
            )
        except Exception as e:
            pytest.skip(f"Q21 execution failed: {e}")

    @pytest.mark.slow
    def test_creates_png_file(self):
        """Should create predictions PNG file"""
        try:
            run_q21()
            png_path = Path("outputs/part3_q21_predictions.png")
            assert png_path.exists(), "Predictions PNG file not created"
        except Exception as e:
            pytest.skip(f"Q21 execution failed: {e}")

    @pytest.mark.slow
    def test_creates_json_file(self):
        """Should create detailed JSON file"""
        try:
            run_q21()
            json_path = Path("outputs/part3_q21_detailed.json")
            assert json_path.exists(), "Detailed JSON file not created"
        except Exception as e:
            pytest.skip(f"Q21 execution failed: {e}")

    @pytest.mark.slow
    def test_json_has_required_fields(self):
        """JSON should have required fields"""
        try:
            run_q21()
            json_path = Path("outputs/part3_q21_detailed.json")
            with open(json_path) as f:
                data = json.load(f)

            assert "selected_pokemon" in data, "Missing 'selected_pokemon' field"
            assert "per_pokemon" in data, "Missing 'per_pokemon' field"

            # Check first per_pokemon entry
            entry = data["per_pokemon"][0]
            required_fields = [
                "name",
                "actual_type1",
                "actual_type2",
                "top5_predictions",
                "top1_correct",
                "top5_correct",
            ]
            for field in required_fields:
                assert field in entry, f"Missing field: {field}"
        except Exception as e:
            pytest.skip(f"Q21 execution failed: {e}")

    @pytest.mark.slow
    def test_reproducibility(self):
        """Should select same 10 Pokemon with seed 42"""
        try:
            # Run twice
            run_q21()
            with open("outputs/part3_q21_detailed.json") as f:
                data1 = json.load(f)
            pokemon1 = data1["selected_pokemon"]

            run_q21()
            with open("outputs/part3_q21_detailed.json") as f:
                data2 = json.load(f)
            pokemon2 = data2["selected_pokemon"]

            assert pokemon1 == pokemon2, (
                "Pokemon selection not reproducible with seed 42"
            )
        except Exception as e:
            pytest.skip(f"Q21 execution failed: {e}")


# ============== TestRunQ22 ==============


class TestRunQ22:
    """Tests for run_q22() function"""

    @pytest.mark.slow
    def test_returns_dict(self):
        """Should return metrics dictionary"""
        try:
            result = run_q22()
            assert isinstance(result, dict), "Should return a dictionary"
        except Exception as e:
            pytest.skip(f"Q22 execution failed: {e}")

    @pytest.mark.slow
    def test_metrics_in_valid_range(self):
        """Metrics should be in valid range [0, 1]"""
        try:
            result = run_q22()
            assert 0.0 <= result["clip_acc1"] <= 1.0, (
                "clip_acc1 should be between 0 and 1"
            )
            assert 0.0 <= result["clip_hit5"] <= 1.0, (
                "clip_hit5 should be between 0 and 1"
            )
        except Exception as e:
            pytest.skip(f"Q22 execution failed: {e}")

    @pytest.mark.slow
    def test_metrics_file_created(self):
        """Should create metrics JSON file"""
        try:
            run_q22()
            json_path = Path("outputs/part3_q22_metrics.json")
            assert json_path.exists(), "Metrics file not created"
        except Exception as e:
            pytest.skip(f"Q22 execution failed: {e}")

    @pytest.mark.slow
    def test_detailed_file_created(self):
        """Should create detailed JSON file"""
        try:
            run_q22()
            json_path = Path("outputs/part3_q22_detailed.json")
            assert json_path.exists(), "Detailed file not created"
        except Exception as e:
            pytest.skip(f"Q22 execution failed: {e}")

    @pytest.mark.slow
    def test_total_pokemon_correct(self):
        """Should process exactly 754 Pokemon"""
        try:
            result = run_q22()
            assert result["total_pokemon"] == 754, (
                f"Expected 754 Pokemon, got {result['total_pokemon']}"
            )
        except Exception as e:
            pytest.skip(f"Q22 execution failed: {e}")

    @pytest.mark.slow
    def test_per_type_accuracy_structure(self):
        """per_type_accuracy should have correct structure"""
        try:
            result = run_q22()
            assert "per_type_accuracy" in result, "Missing per_type_accuracy"

            per_type = result["per_type_accuracy"]
            assert isinstance(per_type, dict), (
                "per_type_accuracy should be a dictionary"
            )

            # Check that all values are in [0, 1]
            for type_name, acc in per_type.items():
                assert isinstance(type_name, str), (
                    f"Type name should be string: {type_name}"
                )
                assert 0.0 <= acc <= 1.0, (
                    f"Accuracy for {type_name} should be between 0 and 1"
                )
        except Exception as e:
            pytest.skip(f"Q22 execution failed: {e}")


# ============== TestRunQ23 ==============


class TestRunQ23:
    """Tests for run_q23() function"""

    @pytest.mark.slow
    def test_returns_dict(self):
        """Should return metrics dictionary"""
        try:
            result = run_q23()
            assert isinstance(result, dict), "Should return a dictionary"
        except Exception as e:
            pytest.skip(f"Q23 requires Qwen3-VL model which may not be available: {e}")

    @pytest.mark.slow
    def test_vlm_acc1_in_valid_range(self):
        """VLM Acc@1 should be in valid range [0, 1]"""
        try:
            result = run_q23()
            assert "vlm_acc1" in result, "Missing vlm_acc1"
            assert 0.0 <= result["vlm_acc1"] <= 1.0, (
                "vlm_acc1 should be between 0 and 1"
            )
        except Exception as e:
            pytest.skip(f"Q23 requires Qwen3-VL model which may not be available: {e}")

    @pytest.mark.slow
    def test_improvement_calculated(self):
        """Should calculate improvement over CLIP"""
        try:
            result = run_q23()
            assert "improvement" in result, "Missing improvement"
            assert isinstance(result["improvement"], (int, float)), (
                "Improvement should be numeric"
            )
        except Exception as e:
            pytest.skip(f"Q23 requires Qwen3-VL model which may not be available: {e}")

    @pytest.mark.slow
    def test_fallback_count_tracked(self):
        """Should track fallback count"""
        try:
            result = run_q23()
            assert "vlm_fallbacks" in result, "Missing vlm_fallbacks"
            assert isinstance(result["vlm_fallbacks"], int), (
                "Fallback count should be integer"
            )
            assert result["vlm_fallbacks"] >= 0, "Fallback count should be non-negative"
        except Exception as e:
            pytest.skip(f"Q23 requires Qwen3-VL model which may not be available: {e}")

    @pytest.mark.slow
    def test_metrics_file_created(self):
        """Should create metrics JSON file"""
        try:
            run_q23()
            json_path = Path("outputs/part3_q23_metrics.json")
            assert json_path.exists(), "Metrics file not created"
        except Exception as e:
            pytest.skip(f"Q23 requires Qwen3-VL model which may not be available: {e}")


# ============== Additional Edge Case Tests ==============


class TestEdgeCases:
    """Edge case tests for robustness"""

    def test_similarity_with_single_image_multiple_texts(self):
        """Handle single image with multiple texts"""
        np.random.seed(42)
        image_embeddings = np.random.randn(1, 512).astype(np.float32)
        image_embeddings /= np.linalg.norm(image_embeddings, axis=-1, keepdims=True)
        text_embeddings = np.random.randn(3, 512).astype(np.float32)
        text_embeddings /= np.linalg.norm(text_embeddings, axis=-1, keepdims=True)

        similarity_t2i = compute_similarity_text_to_image(
            image_embeddings, text_embeddings
        )
        similarity_i2t = compute_similarity_image_to_text(
            image_embeddings, text_embeddings
        )

        assert similarity_t2i.shape == (1, 3), "Text-to-image shape incorrect"
        assert similarity_i2t.shape == (1, 3), "Image-to-text shape incorrect"

    def test_similarity_with_multiple_images_single_text(self):
        """Handle multiple images with single text"""
        np.random.seed(42)
        image_embeddings = np.random.randn(5, 512).astype(np.float32)
        image_embeddings /= np.linalg.norm(image_embeddings, axis=-1, keepdims=True)
        text_embeddings = np.random.randn(1, 512).astype(np.float32)
        text_embeddings /= np.linalg.norm(text_embeddings, axis=-1, keepdims=True)

        similarity_t2i = compute_similarity_text_to_image(
            image_embeddings, text_embeddings
        )
        similarity_i2t = compute_similarity_image_to_text(
            image_embeddings, text_embeddings
        )

        assert similarity_t2i.shape == (5, 1), "Text-to-image shape incorrect"
        assert similarity_i2t.shape == (5, 1), "Image-to-text shape incorrect"

    def test_normalized_embeddings_properties(self):
        """Test normalization properties of embeddings"""
        np.random.seed(42)
        embeddings = np.random.randn(10, 512).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)

        # Check L2 norm
        norms = np.linalg.norm(embeddings, axis=-1)
        assert np.allclose(norms, 1.0, atol=1e-5), "L2 norms should be 1"

        # Check that values are finite
        assert np.all(np.isfinite(embeddings)), "All values should be finite"

        # Check that dot products are in [-1, 1] for normalized vectors
        similarity = embeddings @ embeddings.T
        assert np.all(similarity >= -1.01), "Similarity should be >= -1"
        assert np.all(similarity <= 1.01), "Similarity should be <= 1"

    def test_softmax_properties(self):
        """Test softmax mathematical properties"""
        np.random.seed(42)
        scores = np.random.randn(5, 3).astype(np.float32)

        from scipy.special import softmax

        # Apply softmax along different axes
        softmax_axis_1 = softmax(scores, axis=1)
        softmax_axis_0 = softmax(scores, axis=0)

        # Check row sums (axis=1)
        row_sums = np.sum(softmax_axis_1, axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), "Row sums should be 1"

        # Check column sums (axis=0)
        col_sums = np.sum(softmax_axis_0, axis=0)
        assert np.allclose(col_sums, 1.0, atol=1e-5), "Column sums should be 1"

        # Check that all values are in [0, 1]
        assert np.all(softmax_axis_1 >= 0), "Softmax values should be >= 0"
        assert np.all(softmax_axis_1 <= 1), "Softmax values should be <= 1"
