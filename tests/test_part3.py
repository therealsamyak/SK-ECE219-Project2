import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from part3 import (
    get_device,
    construct_pokedex,
    compute_similarity_text_to_image,
    compute_similarity_image_to_text,
)


class TestGetDevice:
    def test_returns_valid_device(self):
        device = get_device()
        assert device in ["cuda", "mps", "cpu"]


class TestConstructPokedex:
    @patch("part3.pd.read_csv")
    @patch("part3.glob")
    def test_returns_dataframe(self, mock_glob, mock_read_csv):
        mock_df = pd.DataFrame(
            {
                "Name": ["Bulbasaur", "Charmander"],
                "Type1": ["Grass", "Fire"],
                "Type2": ["Poison", ""],
                "ID": [1, 2],
            }
        )
        mock_read_csv.return_value = mock_df
        mock_glob.return_value = ["datasets/pokemon/images/Bulbasaur/0.jpg"]

        result = construct_pokedex()
        assert isinstance(result, pd.DataFrame)
        assert "Name" in result.columns
        assert "Type1" in result.columns

    @patch("part3.pd.read_csv")
    @patch("part3.glob")
    def test_filters_missing_images(self, mock_glob, mock_read_csv):
        mock_df = pd.DataFrame(
            {
                "Name": ["Bulbasaur", "MissingMon"],
                "Type1": ["Grass", "Fire"],
                "Type2": ["Poison", ""],
                "ID": [1, 2],
            }
        )
        mock_read_csv.return_value = mock_df
        mock_glob.side_effect = [["datasets/pokemon/images/Bulbasaur/0.jpg"], []]

        result = construct_pokedex()
        assert len(result) == 1
        assert result.iloc[0]["Name"] == "Bulbasaur"


class TestComputeSimilarity:
    def test_text_to_image_shape(self, sample_embeddings):
        image_emb = sample_embeddings[:10]
        text_emb = sample_embeddings[:5]
        similarity = compute_similarity_text_to_image(image_emb, text_emb)
        assert similarity.shape == (10, 5)

    def test_image_to_text_shape(self, sample_embeddings):
        image_emb = sample_embeddings[:10]
        text_emb = sample_embeddings[:5]
        similarity = compute_similarity_image_to_text(image_emb, text_emb)
        assert similarity.shape == (10, 5)

    def test_softmax_along_text_axis(self, sample_embeddings):
        image_emb = sample_embeddings[:10]
        text_emb = sample_embeddings[:5]
        similarity = compute_similarity_text_to_image(image_emb, text_emb)
        row_sums = np.sum(similarity, axis=-1)
        assert np.allclose(row_sums, 1.0, atol=1e-5)

    def test_softmax_along_image_axis(self, sample_embeddings):
        image_emb = sample_embeddings[:10]
        text_emb = sample_embeddings[:5]
        similarity = compute_similarity_image_to_text(image_emb, text_emb)
        col_sums = np.sum(similarity, axis=0)
        assert np.allclose(col_sums, 1.0, atol=1e-5)


@pytest.mark.slow
class TestLoadClipModel:
    @patch("part3.open_clip.create_model_and_transforms")
    @patch("part3.open_clip.get_tokenizer")
    def test_returns_tuple(self, mock_tokenizer, mock_create):
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_preprocess = MagicMock()
        mock_create.return_value = (mock_model, None, mock_preprocess)
        mock_tokenizer.return_value = MagicMock()

        from part3 import load_clip_model

        result = load_clip_model()
        assert isinstance(result, tuple)
        assert len(result) == 4


@pytest.mark.slow
class TestClipInferenceImage:
    @patch("part3.Image.open")
    def test_returns_normalized_embeddings(self, mock_open):
        from part3 import clip_inference_image

        mock_img = MagicMock()
        mock_open.return_value = mock_img

        mock_model = MagicMock()
        mock_tensor = np.random.randn(1, 512).astype(np.float32)
        mock_model.encode_image.return_value = MagicMock()

        with patch("torch.no_grad"):
            with patch.object(mock_model, "encode_image") as mock_encode:
                mock_encode.return_value = MagicMock()
                mock_encode.return_value.detach.return_value.cpu.return_value.numpy.return_value = mock_tensor

                mock_preprocess = MagicMock()
                mock_preprocess.return_value = MagicMock()

                embeddings = clip_inference_image(
                    mock_model, mock_preprocess, ["test.jpg"], "cpu"
                )
                assert embeddings.shape == (1, 512)


@pytest.mark.slow
class TestClipInferenceText:
    @patch("part3.open_clip.get_tokenizer")
    def test_returns_embeddings(self, mock_tokenizer):
        from part3 import clip_inference_text

        mock_tokens = MagicMock()
        mock_tokens.to.return_value = mock_tokens
        mock_tokenizer.return_value = mock_tokens

        mock_model = MagicMock()
        mock_tensor = np.random.randn(2, 512).astype(np.float32)
        mock_model.encode_text.return_value = MagicMock()

        with patch("torch.no_grad"):
            with patch.object(mock_model, "encode_text") as mock_encode:
                mock_encode.return_value = MagicMock()
                mock_encode.return_value.detach.return_value.cpu.return_value.numpy.return_value = mock_tensor

                embeddings = clip_inference_text(
                    mock_model, mock_tokens, ["text1", "text2"], "cpu"
                )
                assert embeddings.shape == (2, 512)


@pytest.mark.integration
class TestRunQ20:
    @patch("part3.load_clip_model")
    @patch("part3.construct_pokedex")
    @patch("part3.clip_inference_image")
    @patch("part3.clip_inference_text")
    @patch("part3.plt.savefig")
    @patch("part3.plt.subplots")
    @patch("part3.Image.open")
    def test_creates_png_files(
        self,
        mock_img_open,
        mock_subplots,
        mock_savefig,
        mock_text_inf,
        mock_img_inf,
        mock_pokedex,
        mock_clip,
        outputs_dir,
    ):
        from part3 import run_q20

        mock_clip.return_value = (MagicMock(), MagicMock(), "cpu", MagicMock())
        mock_pokedex.return_value = pd.DataFrame(
            {
                "Name": ["Charmander"] * 25,
                "Type1": ["Fire", "Grass", "Bug", "Dark", "Dragon"] * 5,
                "Type2": [""] * 25,
                "image_path": [f"test_{i}.jpg" for i in range(25)],
            }
        )
        mock_img_inf.return_value = np.random.randn(25, 512).astype(np.float32)
        mock_text_inf.return_value = np.random.randn(1, 512).astype(np.float32)
        mock_img_open.return_value = MagicMock()

        mock_fig = MagicMock()
        mock_axes = [MagicMock() for _ in range(5)]
        mock_subplots.return_value = (mock_fig, mock_axes)

        with patch("part3.Path") as mock_path:
            mock_path.return_value = outputs_dir
            run_q20()
        assert mock_savefig.call_count == 5


@pytest.mark.integration
@pytest.mark.skip("Requires complex matplotlib mocking - use manual testing")
class TestRunQ21:
    def test_runs_without_error(self):
        pytest.skip("Test skipped - requires complex matplotlib mocking")


@pytest.mark.integration
class TestRunQ22:
    @patch("part3.load_clip_model")
    @patch("part3.construct_pokedex")
    @patch("part3.clip_inference_image")
    @patch("part3.clip_inference_text")
    @patch("builtins.open", new_callable=MagicMock)
    def test_returns_metrics_dict(
        self,
        mock_file_open,
        mock_text_inf,
        mock_img_inf,
        mock_pokedex,
        mock_clip,
        outputs_dir,
    ):
        from part3 import run_q22

        mock_clip.return_value = (MagicMock(), MagicMock(), "cpu", MagicMock())
        sample_df = pd.DataFrame(
            {
                "Name": [f"Pokemon{i}" for i in range(20)],
                "Type1": ["Fire", "Water", "Grass", "Electric"] * 5,
                "Type2": [""] * 20,
                "image_path": [f"test_{i}.jpg" for i in range(20)],
            }
        )
        mock_pokedex.return_value = sample_df
        mock_img_inf.return_value = np.random.randn(1, 512).astype(np.float32)
        mock_text_inf.return_value = np.random.randn(4, 512).astype(np.float32)

        mock_json_file = MagicMock()
        mock_file_open.return_value.__enter__.return_value = mock_json_file

        with patch("part3.Path") as mock_path:
            mock_path.return_value = outputs_dir
            with patch("builtins.open", mock_file_open):
                result = run_q22()

        assert isinstance(result, dict)
        assert "clip_acc1" in result
        assert "clip_hit5" in result
        assert "total_pokemon" in result


@pytest.mark.integration
class TestRunQ23:
    @patch("part3.load_qwen3_vl")
    @patch("part3.construct_pokedex")
    @patch("part3.qwen_vl_infer_one")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("pathlib.Path.exists", return_value=True)
    def test_returns_metrics_dict(
        self,
        mock_exists,
        mock_file_open,
        mock_vl_infer,
        mock_pokedex,
        mock_qwen,
        outputs_dir,
    ):
        from part3 import run_q23

        mock_qwen.return_value = (MagicMock(), MagicMock())
        mock_pokedex.return_value = pd.DataFrame(
            {
                "Name": ["Pokemon1"],
                "Type1": ["Fire"],
                "Type2": [""],
                "image_path": ["test.jpg"],
            }
        )
        mock_vl_infer.return_value = '{"type1": "Fire"}'

        q22_data = {
            "per_pokemon": [
                {
                    "name": "Pokemon1",
                    "actual": "Fire",
                    "top1": "Fire",
                    "top5": ["Fire", "Water", "Grass", "Electric", "Normal"],
                }
            ]
        }

        q22_metrics = {
            "clip_acc1": 0.5,
            "clip_hit5": 0.8,
            "total_pokemon": 20,
            "per_type_accuracy": {"Fire": 0.5},
        }

        mock_file_data = MagicMock()
        mock_file_data.read.side_effect = [
            json.dumps(q22_data),
            json.dumps(q22_metrics),
        ]
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = mock_file_data
        mock_open = MagicMock(return_value=mock_cm)

        with patch("part3.Path") as mock_path:
            mock_path.return_value = outputs_dir
            mock_path.return_value.exists.return_value = True
            with patch("builtins.open", mock_open):
                result = run_q23()

        assert isinstance(result, dict)
        assert "vlm_acc1" in result
        assert "improvement" in result
