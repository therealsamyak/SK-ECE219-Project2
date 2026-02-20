"""Part 3: CLIP Multimodal Pokemon Analysis"""

import random
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
import logging
from PIL import Image
import torch
import open_clip
from tqdm import tqdm
from scipy.special import softmax
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


def load_qwen3_vl(model_id="Qwen/Qwen3-VL-2B-Instruct"):
    """
    Load Qwen3-VL model + processor.
    Works on GPU if available (recommended).
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading Qwen3-VL model: {model_id}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    ).eval()

    processor = AutoProcessor.from_pretrained(model_id)

    device = next(model.parameters()).device
    logger.info(f"Qwen3-VL loaded on device: {device}")

    return model, processor


@torch.no_grad()
def qwen_vl_infer_one(model, processor, image_path, prompt, max_new_tokens=128):
    """
    Run Qwen3-VL on a single image + prompt.
    Returns the generated text.
    """
    assert os.path.exists(image_path), f"Image not found: {image_path}"

    # You can pass a local file path directly (do NOT use file://)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    # Deterministic decoding
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )

    # Remove the prompt tokens from the output before decoding
    output_ids_trimmed = [
        out[len(inp) :] for inp, out in zip(inputs.input_ids, output_ids)
    ]

    text = processor.batch_decode(
        output_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return text.strip()


def setup_logging():
    """Configure logging to output to both console and file."""
    log_dir = Path("outputs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "part3.log"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def get_device():
    """
    Detect the best available device for model inference.

    Returns:
        str: Device name - 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_clip_model():
    """
    Load CLIP model and preprocess transforms using OpenCLIP.

    Returns:
        tuple: (model, preprocess, device, tokenizer)
    """
    device = get_device()
    logger = logging.getLogger(__name__)
    logger.info(f"Loading CLIP model on device: {device}")

    # Load ViT-L/14 model with OpenAI pretrained weights
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    model = model.to(device)
    model.eval()

    # Get tokenizer for this model
    tokenizer = open_clip.get_tokenizer("ViT-L-14")

    logger.info("CLIP model loaded successfully")
    return model, preprocess, device, tokenizer


def construct_pokedex(
    csv_path="datasets/pokemon/metadata.csv", image_dir="datasets/pokemon/images"
):
    """
    Load Pokemon metadata and construct pokedex with image paths.

    Args:
        csv_path: Path to Pokemon metadata CSV file
        image_dir: Path to Pokemon images directory

    Returns:
        pd.DataFrame: Pokedex with columns: Name, Type1, Type2, image_path
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading Pokemon metadata from {csv_path}")

    pokedex = pd.read_csv(csv_path)
    image_paths = []

    for pokemon_name in pokedex["Name"]:
        imgs = glob(f"{image_dir}/{pokemon_name}/0.jpg")
        if len(imgs) > 0:
            image_paths.append(imgs[0])
        else:
            image_paths.append(None)

    pokedex["image_path"] = image_paths
    pokedex = pokedex[pokedex["image_path"].notna()].reset_index(drop=True)

    # Only keep Pokemon with distinct ID
    ids, id_counts = np.unique(pokedex["ID"], return_counts=True)
    ids, id_counts = np.array(ids), np.array(id_counts)
    keep_ids = ids[id_counts == 1]

    pokedex = pokedex[pokedex["ID"].isin(keep_ids)].reset_index(drop=True)
    pokedex["Type2"] = pokedex["Type2"].str.strip()

    # Keep only required columns
    pokedex = pokedex[["Name", "Type1", "Type2", "image_path"]]

    logger.info(f"Loaded {len(pokedex)} Pokemon with images")
    return pokedex


def clip_inference_image(model, preprocess, image_paths, device):
    """
    Run CLIP inference on a list of image paths.

    Args:
        model: CLIP model
        preprocess: Image preprocessing function
        image_paths: List of image file paths
        device: Device to run inference on

    Returns:
        np.ndarray: Normalized image embeddings
    """
    logger = logging.getLogger(__name__)
    image_embeddings = []

    with torch.no_grad():
        for i, img_path in enumerate(tqdm(image_paths, desc="Encoding images")):
            if (i + 1) % 100 == 0:
                logger.debug(f"Processed {i + 1}/{len(image_paths)} images")

            img = Image.open(img_path)
            img_preprocessed = preprocess(img).unsqueeze(0).to(device)
            image_embedding = (
                model.encode_image(img_preprocessed).detach().cpu().numpy()
            )
            image_embeddings += [image_embedding]

    image_embeddings = np.concatenate(image_embeddings, axis=0)
    image_embeddings /= np.linalg.norm(image_embeddings, axis=-1, keepdims=True)

    logger.info(f"Generated image embeddings for {len(image_embeddings)} images")
    return image_embeddings


def clip_inference_text(model, tokenizer, texts, device):
    """
    Run CLIP inference on a list of texts.

    Args:
        model: CLIP model
        tokenizer: Text tokenizer
        texts: List of text strings
        device: Device to run inference on

    Returns:
        np.ndarray: Normalized text embeddings
    """
    logger = logging.getLogger(__name__)

    with torch.no_grad():
        text_tokens = tokenizer(texts)
        text_embeddings = (
            model.encode_text(text_tokens.to(device)).detach().cpu().numpy()
        )

    text_embeddings /= np.linalg.norm(text_embeddings, axis=-1, keepdims=True)

    logger.info(f"Generated text embeddings for {len(text_embeddings)} texts")
    return text_embeddings


def compute_similarity_text_to_image(image_embeddings, text_embeddings):
    """
    Compute similarity of texts to each image using softmax.

    Args:
        image_embeddings: Image embeddings (N, D)
        text_embeddings: Text embeddings (M, D)

    Returns:
        np.ndarray: Similarity matrix (N, M) with softmax applied along text axis
    """
    logger = logging.getLogger(__name__)

    # Compute raw scores
    raw_scores = 100.0 * image_embeddings @ text_embeddings.T
    logger.debug(f"Text-to-image raw scores shape: {raw_scores.shape}")
    logger.debug(f"Raw scores min/max: {raw_scores.min():.4f}/{raw_scores.max():.4f}")

    # Apply softmax along text axis (-1)
    similarity = softmax(raw_scores, axis=-1)

    logger.info("Computed text-to-image similarity matrix")
    return similarity


def compute_similarity_image_to_text(image_embeddings, text_embeddings):
    """
    Compute similarity of images to each text using softmax.

    Args:
        image_embeddings: Image embeddings (N, D)
        text_embeddings: Text embeddings (M, D)

    Returns:
        np.ndarray: Similarity matrix (N, M) with softmax applied along image axis
    """
    logger = logging.getLogger(__name__)

    # Compute raw scores
    raw_scores = 100.0 * image_embeddings @ text_embeddings.T
    logger.debug(f"Image-to-text raw scores shape: {raw_scores.shape}")
    logger.debug(f"Raw scores min/max: {raw_scores.min():.4f}/{raw_scores.max():.4f}")

    # Apply softmax along image axis (0)
    similarity = softmax(raw_scores, axis=0)

    logger.info("Computed image-to-text similarity matrix")
    return similarity


def run_q20():
    """
    Q20: Text-to-Image Retrieval Plots

    For each type (Bug, Fire, Grass, Dark, Dragon), query CLIP with
    "a photo of a {type} type Pokemon", retrieve top 5 Pokemon by similarity,
    and plot them horizontally.
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("Q20: Text-to-Image Retrieval Plots")
    logger.info("=" * 50)

    # Load CLIP model and pokedex
    model, preprocess, device, tokenizer = load_clip_model()
    pokedex = construct_pokedex()
    logger.info(f"Loaded {len(pokedex)} Pokemon")

    # Get all image paths and compute embeddings once
    image_paths = pokedex["image_path"].tolist()
    logger.info("Computing image embeddings for all Pokemon...")
    image_embeddings = clip_inference_image(model, preprocess, image_paths, device)

    # Types to query
    types = ["Bug", "Fire", "Grass", "Dark", "Dragon"]
    query_template = "a photo of a {type} type Pokemon"

    # Process each type
    for pokemon_type in types:
        logger.info(f"Processing type: {pokemon_type}")

        # Create text query
        query_text = query_template.format(type=pokemon_type)
        logger.info(f"Query: {query_text}")

        # Get text embedding
        text_embeddings = clip_inference_text(model, tokenizer, [query_text], device)

        # Compute similarity (text-to-image) using raw cosine similarity
        # For single text query, use raw scores instead of softmax
        raw_scores = 100.0 * image_embeddings @ text_embeddings.T
        similarity_scores = raw_scores[:, 0]

        # Get top 5 Pokemon by similarity score
        top_5_indices = np.argsort(similarity_scores)[::-1][:5]
        top_5_names = pokedex.iloc[top_5_indices]["Name"].tolist()

        logger.info(f"[Q20] Query: {query_text} -> Top 5: {top_5_names}")

        # Plot top 5 images horizontally
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))

        for i, (idx, ax) in enumerate(zip(top_5_indices, axes)):
            pokemon = pokedex.iloc[idx]
            img = Image.open(pokemon["image_path"])
            ax.imshow(img)

            # Create subtitle: "Name (Type1/Type2)" or "Name (Type1)"
            type2 = pokemon["Type2"]
            if type2 and type2.strip():
                subtitle = f"{pokemon['Name']}\n({pokemon['Type1']}/{type2})"
            else:
                subtitle = f"{pokemon['Name']}\n({pokemon['Type1']})"

            ax.set_title(subtitle, fontsize=10)
            ax.axis("off")

        # Set figure title as query text
        plt.suptitle(query_text, fontsize=12, y=1.05)

        # Save plot
        output_path = f"outputs/part3_q20_{pokemon_type.lower()}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved plot to {output_path}")

    logger.info("=" * 50)
    logger.info("Q20 completed successfully")
    logger.info("=" * 50)


def run_q21():
    """
    Q21: Random 10 Pokemon Type Predictions

    Randomly selects 10 Pokemon (seed 42), gets all unique Type1 values,
    and for each Pokemon predicts top 5 types using CLIP similarity.
    Plots predictions with actual types and saves detailed JSON.
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("Q21: Random 10 Pokemon Type Predictions")
    logger.info("=" * 50)

    model, preprocess, device, tokenizer = load_clip_model()
    pokedex = construct_pokedex()
    logger.info(f"Loaded {len(pokedex)} Pokemon")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    selected_indices = random.sample(range(len(pokedex)), 10)
    selected_pokemon = pokedex.iloc[selected_indices].reset_index(drop=True)
    selected_pokemon_names = selected_pokemon["Name"].tolist()

    logger.info(f"[Q21] Selected Pokemon: {selected_pokemon_names}")

    types = sorted(pokedex["Type1"].unique().tolist())
    logger.info(f"[Q21] Type list: {types}")

    type_texts = [f"a photo of a {type_} type Pokemon" for type_ in types]

    type_embeddings = clip_inference_text(model, tokenizer, type_texts, device)
    logger.info(f"[Q21] Type embeddings shape: {type_embeddings.shape}")

    import json

    results = []

    for idx, row in selected_pokemon.iterrows():
        pokemon_name = row["Name"]
        pokemon_type1 = row["Type1"]
        pokemon_type2 = row["Type2"] if row["Type2"] and row["Type2"].strip() else ""
        image_path = row["image_path"]

        image_embedding = clip_inference_image(model, preprocess, [image_path], device)[
            0
        ]

        raw_scores = 100.0 * image_embedding @ type_embeddings.T
        similarity = softmax(raw_scores, axis=-1)

        ranked_indices = np.argsort(similarity)[::-1][:5]
        top5_predictions = [(types[i], float(similarity[i])) for i in ranked_indices]

        top1_type = top5_predictions[0][0]
        top5_types = [pred[0] for pred in top5_predictions]

        top1_correct = top1_type == pokemon_type1
        top5_correct = pokemon_type1 in top5_types

        logger.info(
            f"[Q21] Pokemon: {pokemon_name} ({pokemon_type1}/{pokemon_type2 if pokemon_type2 else ''})"
        )
        logger.info(
            f"[Q21]   Top 5 predictions: {[(t, round(s, 4)) for t, s in top5_predictions]}"
        )
        logger.info(
            f"[Q21]   Correct in top-1? {top1_correct}, Correct in top-5? {top5_correct}"
        )

        results.append(
            {
                "name": pokemon_name,
                "actual_type1": pokemon_type1,
                "actual_type2": pokemon_type2,
                "top5_predictions": [[t, round(s, 4)] for t, s in top5_predictions],
                "top1_correct": top1_correct,
                "top5_correct": top5_correct,
                "image_path": image_path,
            }
        )

    fig, axes = plt.subplots(2, 5, figsize=(20, 10))

    for idx, (ax, result) in enumerate(zip(axes.flatten(), results)):
        img = Image.open(result["image_path"])
        ax.imshow(img)

        if result["actual_type2"]:
            title = (
                f"{result['name']}\n({result['actual_type1']}/{result['actual_type2']})"
            )
        else:
            title = f"{result['name']}\n({result['actual_type1']})"

        ax.set_title(title, fontsize=10, fontweight="bold")

        pred_text = "Top 5:\n"
        for i, (pred_type, score) in enumerate(result["top5_predictions"]):
            marker = "✓" if pred_type == result["actual_type1"] else ""
            pred_text += f"{i + 1}. {pred_type} ({score:.4f}) {marker}\n"

        ax.text(
            0.5,
            -0.35,
            pred_text,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            family="monospace",
        )

        ax.axis("off")

    plt.suptitle("Q21: Random 10 Pokemon Type Predictions", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = "outputs/part3_q21_predictions.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved plot to {output_path}")

    json_output = {
        "selected_pokemon": selected_pokemon_names,
        "per_pokemon": [
            {
                "name": r["name"],
                "actual_type1": r["actual_type1"],
                "actual_type2": r["actual_type2"],
                "top5_predictions": r["top5_predictions"],
                "top1_correct": r["top1_correct"],
                "top5_correct": r["top5_correct"],
            }
            for r in results
        ],
    }

    json_path = "outputs/part3_q21_detailed.json"
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)

    logger.info(f"Saved detailed results to {json_path}")

    top1_count = sum(1 for r in results if r["top1_correct"])
    top5_count = sum(1 for r in results if r["top5_correct"])
    acc1 = top1_count / len(results)
    hit5 = top5_count / len(results)

    logger.info(f"[Q21] Top-1 accuracy: {acc1:.4f} ({top1_count}/{len(results)})")
    logger.info(f"[Q21] Top-5 accuracy: {hit5:.4f} ({top5_count}/{len(results)})")

    logger.info("=" * 50)
    logger.info("Q21 completed successfully")
    logger.info("=" * 50)


def run_q22():
    """
    Q22: Full Dataset CLIP Evaluation (Acc@1 + Hit@5).

    Evaluates all Pokemon with CLIP type prediction, computing Acc@1 and Hit@5 metrics.
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("Q22: Full Dataset CLIP Evaluation")
    logger.info("=" * 50)

    model, preprocess, device, tokenizer = load_clip_model()

    pokedex = construct_pokedex()
    logger.info(f"Q22: Loaded {len(pokedex)} Pokemon for evaluation")

    types = sorted(pokedex["Type1"].unique().tolist())
    logger.info(f"[Q22] Type list: {types}")

    type_texts = [f"A photo of a {type_} type Pokemon" for type_ in types]
    type_embeddings = clip_inference_text(model, tokenizer, type_texts, device)
    logger.info(f"[Q22] Type embeddings shape: {type_embeddings.shape}")

    total = len(pokedex)
    top1_correct = 0
    top5_correct = 0
    per_type_stats = {type_: {"correct": 0, "total": 0} for type_ in types}

    detailed_results = []

    for idx, row in pokedex.iterrows():
        pokemon_name = row["Name"]
        pokemon_type1 = row["Type1"]
        image_path = row["image_path"]

        image_embedding = clip_inference_image(model, preprocess, [image_path], device)[
            0
        ]

        raw_scores = 100.0 * image_embedding.reshape(1, -1) @ type_embeddings.T
        similarity = softmax(raw_scores, axis=-1)[0]

        ranked_indices = np.argsort(similarity)[::-1]
        top1_type = types[ranked_indices[0]]
        top5_types = [types[i] for i in ranked_indices[:5]]

        acc1_correct = top1_type == pokemon_type1
        hit5_correct = pokemon_type1 in top5_types

        if acc1_correct:
            top1_correct += 1
        if hit5_correct:
            top5_correct += 1

        per_type_stats[pokemon_type1]["total"] += 1
        if acc1_correct:
            per_type_stats[pokemon_type1]["correct"] += 1

        detailed_results.append(
            {
                "name": pokemon_name,
                "actual": pokemon_type1,
                "top1": top1_type,
                "top5": top5_types,
                "acc1_correct": acc1_correct,
                "hit5_correct": hit5_correct,
            }
        )

        logger.info(
            f"[Q22] Pokemon: {pokemon_name} ({pokemon_type1}) -> "
            f"Top-1: {top1_type}, Top-5: {top5_types}"
        )

        if (idx + 1) % 50 == 0:
            logger.info(f"[Q22] Progress: {idx + 1}/{total} Pokemon processed...")

    acc1 = top1_correct / total
    hit5 = top5_correct / total
    per_type_accuracy = {
        type_: stats["correct"] / stats["total"]
        for type_, stats in per_type_stats.items()
    }

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    detailed_output = output_dir / "part3_q22_detailed.json"
    with open(detailed_output, "w") as f:
        import json

        json.dump({"per_pokemon": detailed_results}, f, indent=2)
    logger.info(f"Saved detailed results to {detailed_output}")

    metrics_output = output_dir / "part3_q22_metrics.json"
    metrics = {
        "clip_acc1": round(acc1, 4),
        "clip_hit5": round(hit5, 4),
        "total_pokemon": total,
        "per_type_accuracy": {
            type_: round(acc, 4) for type_, acc in per_type_accuracy.items()
        },
    }
    with open(metrics_output, "w") as f:
        import json

        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_output}")

    logger.info(f"[Q22] Acc@1: {acc1:.4f}, Hit@5: {hit5:.4f}")

    logger.info("=" * 50)
    logger.info("Q22 completed successfully")
    logger.info("=" * 50)

    return metrics


def run_q23():
    """
    Q23: VLM Reranking with Qwen3-VL-2B.

    Uses Qwen3-VL to rerank CLIP top-5 predictions and selects the most likely type.
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("Q23: VLM Reranking with Qwen3-VL-2B")
    logger.info("=" * 50)

    import json

    model, processor = load_qwen3_vl()

    q22_path = Path("outputs/part3_q22_detailed.json")
    if not q22_path.exists():
        logger.error(f"Q22 results not found at {q22_path}")
        raise FileNotFoundError(f"Run Q22 first to generate {q22_path}")

    with open(q22_path, "r") as f:
        q22_data = json.load(f)

    per_pokemon_q22 = q22_data["per_pokemon"]
    total = len(per_pokemon_q22)
    logger.info(f"Loaded {total} Pokemon from Q22 results")

    pokedex = construct_pokedex()

    vlm_results = []
    vlm_correct = 0
    vlm_fallbacks = 0

    for idx, pokemon_q22 in enumerate(per_pokemon_q22):
        name = pokemon_q22["name"]
        actual = pokemon_q22["actual"]
        clip_top1 = pokemon_q22["top1"]
        top5 = pokemon_q22["top5"]

        top5_types_str = ", ".join(top5)
        prompt = f'This is a Pokemon. From these types: {top5_types_str}, which is the most likely primary type? Return only the type name as JSON: {{"type1": "..."}}'

        pokemon_data = pokedex[pokedex["Name"] == name]
        if len(pokemon_data) == 0:
            logger.error(f"Pokemon {name} not found in pokedex")
            continue
        image_path = pokemon_data.iloc[0]["image_path"]

        try:
            vlm_response = qwen_vl_infer_one(model, processor, image_path, prompt)

            vlm_type = None
            try:
                parsed = json.loads(vlm_response)
                vlm_type = parsed.get("type1", None)
            except json.JSONDecodeError:
                for type_ in top5:
                    if type_.lower() in vlm_response.lower():
                        vlm_type = type_
                        break

            if vlm_type is None or vlm_type not in top5:
                vlm_type = clip_top1
                vlm_fallbacks += 1

            vlm_correct_flag = vlm_type == actual
            if vlm_correct_flag:
                vlm_correct += 1

            logger.info(
                f"[Q23] Pokemon: {name} ({actual}) -> "
                f"CLIP top-1: {clip_top1}, VLM selected: {vlm_type} "
                f"({'correct' if vlm_correct_flag else 'incorrect'})"
            )

            vlm_results.append(
                {
                    "name": name,
                    "actual": actual,
                    "clip_top1": clip_top1,
                    "vlm_selected": vlm_type,
                    "correct": vlm_correct_flag,
                }
            )

        except Exception as e:
            logger.error(f"[Q23] Error processing {name}: {e}")
            vlm_type = clip_top1
            vlm_fallbacks += 1
            vlm_results.append(
                {
                    "name": name,
                    "actual": actual,
                    "clip_top1": clip_top1,
                    "vlm_selected": vlm_type,
                    "correct": vlm_type == actual,
                }
            )

        if (idx + 1) % 50 == 0:
            logger.info(f"[Q23] Progress: {idx + 1}/{total} Pokemon processed...")

    vlm_acc1 = vlm_correct / total

    with open("outputs/part3_q22_metrics.json", "r") as f:
        q22_metrics = json.load(f)

    clip_acc1 = q22_metrics["clip_acc1"]
    clip_hit5 = q22_metrics["clip_hit5"]
    improvement = vlm_acc1 - clip_acc1

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    detailed_output = output_dir / "part3_q23_detailed.json"
    with open(detailed_output, "w") as f:
        json.dump({"per_pokemon": vlm_results}, f, indent=2)
    logger.info(f"Saved detailed results to {detailed_output}")

    metrics_output = output_dir / "part3_q23_metrics.json"
    metrics = {
        "clip_acc1": round(clip_acc1, 4),
        "clip_hit5": round(clip_hit5, 4),
        "vlm_acc1": round(vlm_acc1, 4),
        "improvement": round(improvement, 4),
        "total_pokemon": total,
        "vlm_fallbacks": vlm_fallbacks,
    }
    with open(metrics_output, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_output}")

    logger.info("[Q23] Comparison:")
    logger.info("| Method     | Acc@1 | Hit@5 |")
    logger.info("|------------|-------|-------|")
    logger.info(f"| CLIP       | {clip_acc1:.4f}| {clip_hit5:.4f}|")
    logger.info(f"| VLM Rerank | {vlm_acc1:.4f}| N/A   |")
    logger.info(f"[Q23] VLM improvement over CLIP: {improvement:+.4f}")
    logger.info(f"[Q23] VLM fallbacks: {vlm_fallbacks}/{total}")

    logger.info("=" * 50)
    logger.info("Q23 completed successfully")
    logger.info("=" * 50)

    return metrics


# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


if __name__ == "__main__":
    setup_logging()
    run_q20()
    run_q21()
    run_q22()
    run_q23()
