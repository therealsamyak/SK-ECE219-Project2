import pandas as pd
import numpy as np
import logging

from pathlib import Path
import json
from collections import Counter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from joblib import Memory
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN
from sklearn.metrics import (
    homogeneity_score,
    completeness_score,
    v_measure_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
)
from sklearn.neighbors import kneighbors_graph
import umap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import entropy


def setup_logging():
    """Configure logging to output to both console and file."""
    log_dir = Path("outputs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "part1.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def create_length_labels():
    """
    Create length labels for Steam reviews based on word count quantiles.

    Returns:
        pd.DataFrame: Filtered DataFrame with 'length_label' column.
                      Contains only 'Short' (≤ q25) and 'Long' (≥ q75) reviews.
    """
    logger = setup_logging()
    logger.info("Starting create_length_labels")

    dataset_path = Path("datasets/steam/main.csv")
    logger.info(f"Loading steam reviews from {dataset_path}")

    df = pd.read_csv(dataset_path)
    logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")

    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Total number of reviews: {len(df)}")

    df["word_count"] = df["review_text"].astype(str).str.split().str.len()

    q25, q75 = df["word_count"].quantile([0.25, 0.75])
    logger.info(f"25th percentile (q25): {q25:.1f} words")
    logger.info(f"75th percentile (q75): {q75:.1f} words")

    short_mask = df["word_count"] <= q25
    long_mask = df["word_count"] >= q75

    short_count = short_mask.sum()
    long_count = long_mask.sum()

    logger.info(f"Short reviews (≤{q25:.1f} words): {short_count}")
    logger.info(f"Long reviews (≥{q75:.1f} words): {long_count}")

    if short_count > 0:
        short_avg = df.loc[short_mask, "word_count"].mean()
        logger.info(f"Average word count for Short reviews: {short_avg:.2f}")

    if long_count > 0:
        long_avg = df.loc[long_mask, "word_count"].mean()
        logger.info(f"Average word count for Long reviews: {long_avg:.2f}")

    df["length_label"] = None
    df.loc[short_mask, "length_label"] = "Short"
    df.loc[long_mask, "length_label"] = "Long"

    filtered_df = df[df["length_label"].isin(["Short", "Long"])].copy()

    logger.info(
        f"Labeled dataset size: {len(filtered_df)} (after discarding middle 50%)"
    )
    logger.info(f"Filtered dataset size (samples): {len(filtered_df)}")

    logger.info("Finished create_length_labels")

    return filtered_df


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


def setup_qwen_model():
    """
    Load Qwen3-4B-Instruct model for LLM-based cluster labeling.

    Returns:
        tuple: (model, tokenizer) or (None, None) if loading fails
    """
    logger = logging.getLogger(__name__)
    model_name = "Qwen/Qwen3-4B-Instruct-2507"

    try:
        device = get_device()
        logger.info(f"Loading Qwen model on {device}...")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        logger.info("Qwen model loaded successfully")
        return model, tokenizer
    except Exception as e:
        logger.warning(f"Failed to load Qwen model: {e}")
        logger.warning("LLM labeling will be skipped")
        return None, None


def generate_llm_cluster_label(top_terms, exemplars, review_type, model, tokenizer):
    """
    Generate a cluster label using the Qwen LLM.

    Args:
        top_terms: List of dicts with 'term' and 'score' keys
        exemplars: List of dicts with 'review' key
        review_type: 'positive' or 'negative'
        model: The loaded Qwen model
        tokenizer: The loaded Qwen tokenizer

    Returns:
        str: Generated label or None if generation fails
    """
    logger = logging.getLogger(__name__)

    if model is None or tokenizer is None:
        return None

    # Format top terms
    terms_str = ", ".join([t["term"] for t in top_terms[:5]])

    # Format exemplars (truncate to avoid long prompts)
    exemplar_texts = []
    for ex in exemplars[:2]:
        text = ex.get("review", "")[:200]
        exemplar_texts.append(f'"{text}..."')
    exemplars_str = "\n".join(exemplar_texts)

    prompt = f"""I have a cluster of {review_type} reviews from a video game.
Top terms: {terms_str}
Example reviews:
{exemplars_str}

Generate a short 3-6 word label describing this cluster's theme.
Label:"""

    try:
        # Use Qwen's chat template format
        prompt_with_tokens = (
            f"<|im_start|>user\n{prompt.strip()}<|im_end|>\n<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt_with_tokens, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=20,
                temperature=None,
                top_p=None,
                top_k=None,
            )

        # Decode only newly generated tokens
        generated_ids = outputs[0][inputs.input_ids.shape[1] :]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        label = response.strip()

        logger.info(f"Generated LLM label for {review_type} cluster: {label}")
        return label
    except Exception as e:
        logger.warning(f"Failed to generate LLM label: {e}")
        return None


def run_task1_1():
    """
    Compute TF-IDF and MiniLM representations for filtered reviews.

    This function:
    1. Calls create_length_labels() to get Short/Long filtered dataset
    2. Computes TF-IDF representation using TfidfVectorizer
    3. Computes MiniLM embeddings using sentence-transformers
    4. L2 normalizes both representations
    5. Logs dimensions and statistics
    6. Saves results to 3 JSON files

    Returns:
        dict: Dictionary containing tfidf, minilm, and dataset results
    """
    logger = setup_logging()
    logger.info("Starting run_task1_1 - Representations")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    df = create_length_labels()
    logger.info(f"Loaded filtered dataset: {len(df)} reviews")

    short_count = (df["length_label"] == "Short").sum()
    long_count = (df["length_label"] == "Long").sum()
    logger.info(f"Short reviews: {short_count}, Long reviews: {long_count}")

    reviews_text = df["review_text"].astype(str).tolist()

    # TF-IDF representation
    logger.info("Computing TF-IDF representation...")
    tfidf_vectorizer = TfidfVectorizer(
        min_df=3,
        stop_words="english",
        ngram_range=(1, 1),
        norm="l2",
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(reviews_text)

    logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    logger.info(f"TF-IDF vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")

    sparsity = 1.0 - (
        tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])
    )
    logger.info(f"TF-IDF sparsity: {sparsity:.4f}")

    # MiniLM embeddings
    logger.info("Computing MiniLM embeddings...")
    device = get_device()
    logger.info(f"Using device: {device}")

    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)
    memory = Memory(location=str(cache_dir), verbose=0)

    @memory.cache
    def compute_minilm_cached(texts, device_name):
        model = SentenceTransformer("all-MiniLM-L6-v2", device=device_name)
        embeddings = model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings

    minilm_embeddings = compute_minilm_cached(reviews_text, device)

    logger.info(f"MiniLM embeddings shape: {minilm_embeddings.shape}")

    # Build results dictionary
    results = {
        "tfidf": {
            "matrix_shape": tuple(int(x) for x in tfidf_matrix.shape),
            "vocabulary_size": len(tfidf_vectorizer.vocabulary_),
        },
        "minilm": {
            "matrix_shape": tuple(int(x) for x in minilm_embeddings.shape),
            "embedding_dim": int(minilm_embeddings.shape[1]),
            "normalized": True,
        },
        "dataset": {
            "size": int(len(df)),
            "num_short": int(short_count),
            "num_long": int(long_count),
        },
    }

    # Save JSON files
    results_path = output_dir / "Q1_Q2_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")

    tfidf_path = output_dir / "Q2_tfidf.json"
    with open(tfidf_path, "w") as f:
        json.dump(
            {
                "matrix_shape": tuple(int(x) for x in tfidf_matrix.shape),
                "vocabulary_size": len(tfidf_vectorizer.vocabulary_),
                "sparsity": float(sparsity),
            },
            f,
            indent=2,
        )
    logger.info(f"Saved TF-IDF info to {tfidf_path}")

    minilm_path = output_dir / "Q2_minilm.json"
    with open(minilm_path, "w") as f:
        json.dump(
            {
                "matrix_shape": tuple(minilm_embeddings.shape),
                "embedding_dim": int(minilm_embeddings.shape[1]),
                "normalized": True,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved MiniLM info to {minilm_path}")

    logger.info("Finished run_task1_1 - Representations")

    return results


def apply_dimensionality_reduction(X, method, n_components=50, random_state=42):
    """
    Apply dimensionality reduction to the data.

    Args:
        X: Input data (sparse matrix or dense array)
        method: Reduction method - "none", "svd", or "umap"
        n_components: Number of components for reduction
        random_state: Random seed for reproducibility

    Returns:
        Reduced data (dense array)
    """
    if method == "none":
        if hasattr(X, "toarray"):
            return X.toarray()
        return np.array(X)
    elif method == "svd":
        svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        return svd.fit_transform(X)
    elif method == "umap":
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.array(X)
        if X.shape[1] > 200:
            svd_preprocess = TruncatedSVD(n_components=200, random_state=random_state)
            X = svd_preprocess.fit_transform(X)
        umap_model = umap.UMAP(
            n_components=n_components, random_state=random_state, n_jobs=1
        )
        return umap_model.fit_transform(X)
    elif method == "autoencoder":
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.array(X, dtype=np.float32)
        input_dim = X.shape[1]

        class Autoencoder(torch.nn.Module):
            def __init__(self, input_dim, latent_dim):
                super().__init__()
                self.encoder = torch.nn.Linear(input_dim, latent_dim)
                self.decoder = torch.nn.Linear(latent_dim, input_dim)

            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded

            def encode(self, x):
                return self.encoder(x)

        model = Autoencoder(input_dim, n_components)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()

        X_tensor = torch.tensor(X)
        dataset = torch.utils.data.TensorDataset(X_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        model.train()
        for epoch in range(50):
            for (batch,) in loader:
                optimizer.zero_grad()
                output = model(batch)
                loss = criterion(output, batch)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            encoded = model.encode(X_tensor).numpy()
        return encoded
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")


def run_clustering_pipeline(
    X, method, n_clusters=2, random_state=42, hdbscan_min_size=2
):
    """
    Run a clustering algorithm on data.

    Args:
        X: Input data (dense array)
        method: Clustering method - "kmeans", "agglomerative", or "hdbscan"
        n_clusters: Number of clusters (ignored for HDBSCAN)
        random_state: Random seed for reproducibility
        hdbscan_min_size: Minimum cluster size for HDBSCAN (default: 2)

    Returns:
        Tuple of (cluster_labels, model) where cluster_labels is numpy array and model is the fitted clustering model
    """
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        return model.fit_predict(X), model
    elif method == "agglomerative":
        k = min(50, X.shape[0] // 10) if X.shape[0] > 100 else 10
        conn = kneighbors_graph(
            X, n_neighbors=k, mode="connectivity", include_self=False
        )
        model = AgglomerativeClustering(
            n_clusters=n_clusters, linkage="ward", connectivity=conn
        )
        return model.fit_predict(X), model
    elif method == "hdbscan":
        # Use min_cluster_size=2 initially, can adjust if noise dominates
        model = HDBSCAN(min_cluster_size=hdbscan_min_size, min_samples=5, copy=False)
        return model.fit_predict(X), model
    else:
        raise ValueError(f"Unknown clustering method: {method}")


def compute_clustering_metrics(labels_true, labels_pred):
    """
    Compute clustering evaluation metrics against ground truth labels.

    Args:
        labels_true: Ground truth labels
        labels_pred: Predicted cluster labels

    Returns:
        Dictionary of metrics
    """
    return {
        "homogeneity": homogeneity_score(labels_true, labels_pred),
        "completeness": completeness_score(labels_true, labels_pred),
        "v_measure": v_measure_score(labels_true, labels_pred),
        "ari": adjusted_rand_score(labels_true, labels_pred),
        "ami": adjusted_mutual_info_score(
            labels_true, labels_pred, average_method="arithmetic"
        ),
    }


def run_task1_2(data=None):
    """
    Run clustering pipelines on TF-IDF and MiniLM representations.

    This function:
    1. Loads filtered reviews and length labels from create_length_labels()
    2. Computes or loads TF-IDF and MiniLM representations
    3. For each representation, runs clustering pipelines:
       - TF-IDF: SVD(50) + K-Means, SVD(50) + Agglomerative
       - MiniLM: None + K-Means, None + Agglomerative,
                SVD(50) + K-Means, SVD(50) + Agglomerative,
                UMAP(50) + K-Means, UMAP(50) + Agglomerative,
                HDBSCAN
    4. Computes evaluation metrics (homogeneity, completeness, v-measure, ARI, AMI)
    5. Logs metrics for each pipeline
    6. Saves comprehensive results to JSON files

    Returns:
        dict: Dictionary containing all clustering results
    """
    logger = setup_logging()
    logger.info("Starting run_task1_2 - Clustering Pipelines")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    logger.info("Loading filtered reviews and length labels...")
    df = create_length_labels()
    logger.info(f"Loaded {len(df)} reviews for clustering")

    true_labels = df["length_label"].map({"Short": 0, "Long": 1}).values
    logger.info(
        f"Ground truth labels: Short=0 ({(true_labels == 0).sum()}), Long=1 ({(true_labels == 1).sum()})"
    )

    reviews_text = df["review_text"].astype(str).tolist()

    logger.info("Computing TF-IDF representation...")
    tfidf_vectorizer = TfidfVectorizer(
        min_df=3, stop_words="english", ngram_range=(1, 1), norm="l2"
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(reviews_text)
    logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    logger.info("Computing MiniLM embeddings...")
    device = get_device()
    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)
    memory = Memory(location=str(cache_dir), verbose=0)

    @memory.cache
    def compute_minilm_cached(texts, device_name):
        model = SentenceTransformer("all-MiniLM-L6-v2", device=device_name)
        embeddings = model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings

    minilm_embeddings = compute_minilm_cached(reviews_text, device)
    logger.info(f"MiniLM embeddings shape: {minilm_embeddings.shape}")

    tfidf_configs = [
        {"dim_reduction": "svd", "clustering": "kmeans"},
        {"dim_reduction": "svd", "clustering": "agglomerative"},
    ]

    minilm_configs = [
        {"dim_reduction": "none", "clustering": "kmeans"},
        {"dim_reduction": "none", "clustering": "agglomerative"},
        {"dim_reduction": "svd", "clustering": "kmeans"},
        {"dim_reduction": "svd", "clustering": "agglomerative"},
        {"dim_reduction": "umap", "clustering": "kmeans"},
        {"dim_reduction": "umap", "clustering": "agglomerative"},
        {"dim_reduction": "none", "clustering": "hdbscan"},
        {"dim_reduction": "svd", "clustering": "hdbscan"},
        {"dim_reduction": "umap", "clustering": "hdbscan"},
    ]

    results = {
        "tfidf": {"results": []},
        "minilm": {"results": []},
    }

    logger.info("=" * 60)
    logger.info("Running TF-IDF clustering pipelines...")
    logger.info("=" * 60)

    for i, config in enumerate(tfidf_configs, 1):
        dim_red = config["dim_reduction"]
        cluster_method = config["clustering"]

        logger.info(
            f"\nTF-IDF Pipeline {i}/{len(tfidf_configs)}: {dim_red.upper()} + {cluster_method.upper()}"
        )

        logger.info(f"  Applying {dim_red.upper()} dimensionality reduction...")
        X_reduced = apply_dimensionality_reduction(tfidf_matrix, dim_red)
        logger.info(f"  Reduced shape: {X_reduced.shape}")

        logger.info(f"  Running {cluster_method.upper()} clustering...")
        cluster_labels, _ = run_clustering_pipeline(
            X_reduced, cluster_method, n_clusters=2
        )

        metrics = compute_clustering_metrics(true_labels, cluster_labels)

        logger.info("  Metrics:")
        logger.info(f"    Homogeneity: {metrics['homogeneity']:.4f}")
        logger.info(f"    Completeness: {metrics['completeness']:.4f}")
        logger.info(f"    V-Measure: {metrics['v_measure']:.4f}")
        logger.info(f"    ARI: {metrics['ari']:.4f}")
        logger.info(f"    AMI: {metrics['ami']:.4f}")

        result_entry = {
            "pipeline_id": i,
            "dim_reduction": dim_red,
            "clustering": cluster_method,
            "n_components": 50 if dim_red != "none" else None,
            "cluster_distribution": {
                "cluster_0": int((cluster_labels == 0).sum()),
                "cluster_1": int((cluster_labels == 1).sum()),
            },
            "metrics": metrics,
        }
        results["tfidf"]["results"].append(result_entry)

    logger.info("=" * 60)
    logger.info("Running MiniLM clustering pipelines...")
    logger.info("=" * 60)

    for i, config in enumerate(minilm_configs, 1):
        dim_red = config["dim_reduction"]
        cluster_method = config["clustering"]

        logger.info(
            f"\nMiniLM Pipeline {i}/{len(minilm_configs)}: {dim_red.upper()} + {cluster_method.upper()}"
        )

        logger.info(f"  Applying {dim_red.upper()} dimensionality reduction...")
        X_reduced = apply_dimensionality_reduction(minilm_embeddings, dim_red)
        logger.info(f"  Reduced shape: {X_reduced.shape}")

        n_clusters = 2 if cluster_method != "hdbscan" else None
        logger.info(f"  Running {cluster_method.upper()} clustering...")
        cluster_labels, _ = run_clustering_pipeline(
            X_reduced, cluster_method, n_clusters=n_clusters
        )

        unique_labels = np.unique(cluster_labels)
        n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)
        noise_count = int((cluster_labels == -1).sum()) if -1 in unique_labels else 0

        if n_clusters_found >= 2 and noise_count < len(cluster_labels):
            metrics = compute_clustering_metrics(true_labels, cluster_labels)
        else:
            logger.warning(
                f"  Warning: HDBSCAN found {n_clusters_found} clusters, skipping metrics"
            )
            metrics = {
                "homogeneity": None,
                "completeness": None,
                "v_measure": None,
                "ari": None,
                "ami": None,
            }

        logger.info("  Metrics:")
        logger.info(f"    Homogeneity: {metrics['homogeneity']:.4f}")
        logger.info(f"    Completeness: {metrics['completeness']:.4f}")
        logger.info(f"    V-Measure: {metrics['v_measure']:.4f}")
        logger.info(f"    ARI: {metrics['ari']:.4f}")
        logger.info(f"    AMI: {metrics['ami']:.4f}")

        result_entry = {
            "pipeline_id": i,
            "dim_reduction": dim_red,
            "clustering": cluster_method,
            "n_components": 50 if dim_red != "none" else None,
            "n_clusters_found": n_clusters_found,
            "noise_points": noise_count,
            "cluster_distribution": {
                f"cluster_{label}": int((cluster_labels == label).sum())
                for label in unique_labels
            },
            "metrics": metrics,
        }
        results["minilm"]["results"].append(result_entry)

    results["summary"] = {
        "total_pipelines": len(tfidf_configs) + len(minilm_configs),
        "tfidf_pipelines": len(tfidf_configs),
        "minilm_pipelines": len(minilm_configs),
        "dataset_size": int(len(df)),
        "ground_truth_distribution": {
            "Short": int((true_labels == 0).sum()),
            "Long": int((true_labels == 1).sum()),
        },
    }

    results_path = output_dir / "Q3_Q4_clustering_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved clustering results to {results_path}")

    tfidf_path = output_dir / "Q3_tfidf_clustering.json"
    with open(tfidf_path, "w") as f:
        json.dump(
            {
                "representation": "TF-IDF",
                "pipelines": results["tfidf"]["results"],
            },
            f,
            indent=2,
        )
    logger.info(f"Saved TF-IDF clustering results to {tfidf_path}")

    minilm_path = output_dir / "Q3_minilm_clustering.json"
    with open(minilm_path, "w") as f:
        json.dump(
            {
                "representation": "MiniLM",
                "pipelines": results["minilm"]["results"],
            },
            f,
            indent=2,
        )
    logger.info(f"Saved MiniLM clustering results to {minilm_path}")

    logger.info("\n" + "=" * 60)
    logger.info("Best Performing Pipelines Summary:")
    logger.info("=" * 60)

    for rep_name, rep_data in [
        ("TF-IDF", results["tfidf"]),
        ("MiniLM", results["minilm"]),
    ]:
        valid_results = [
            r for r in rep_data["results"] if r["metrics"]["v_measure"] is not None
        ]

        if valid_results:
            best_v_measure = max(valid_results, key=lambda x: x["metrics"]["v_measure"])
            best_ari = max(valid_results, key=lambda x: x["metrics"]["ari"])

            logger.info(f"\n{rep_name} - Best V-Measure:")
            logger.info(
                f"  {best_v_measure['dim_reduction'].upper()} + {best_v_measure['clustering'].upper()}: "
                f"V={best_v_measure['metrics']['v_measure']:.4f}"
            )

            logger.info(f"{rep_name} - Best ARI:")
            logger.info(
                f"  {best_ari['dim_reduction'].upper()} + {best_ari['clustering'].upper()}: "
                f"ARI={best_ari['metrics']['ari']:.4f}"
            )

    logger.info("\nFinished run_task1_2 - Clustering Pipelines")

    return results


def plot_pca_visualizations(use_clustering_results_if_available=True):
    """
    Create PCA visualization plots for TF-IDF and MiniLM representations.

    This function:
    1. Loads clustering results from Q3_Q4_clustering_results.json if available
    2. Computes TF-IDF and MiniLM representations from the filtered dataset
    3. Applies PCA to reduce dimensions to 2D
    4. Creates 2x2 visualization: ground truth and cluster colors for each representation
    5. Saves plots to outputs/Q5_pca_visualizations.png

    Args:
        use_clustering_results_if_available: If True, try to load and use clustering results.

    Returns:
        dict: Dictionary containing plot_path, best_pipelines, and PCA results
    """
    logger = setup_logging()
    logger.info("Starting plot_pca_visualizations")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Try to load clustering results if requested
    best_pipelines = {}

    if use_clustering_results_if_available:
        clustering_results_path = output_dir / "Q3_Q4_clustering_results.json"
        if clustering_results_path.exists():
            with open(clustering_results_path, "r") as f:
                clustering_results = json.load(f)
            logger.info("Loaded clustering results from Q3_Q4_clustering_results.json")

            # Find best pipeline for each representation
            for rep_name in ["tfidf", "minilm"]:
                if (
                    rep_name in clustering_results
                    and "results" in clustering_results[rep_name]
                ):
                    valid_results = [
                        r
                        for r in clustering_results[rep_name]["results"]
                        if r.get("metrics", {}).get("v_measure") is not None
                    ]
                    if valid_results:
                        best = max(
                            valid_results, key=lambda x: x["metrics"]["v_measure"]
                        )
                        best_pipelines[rep_name] = {
                            "dim_reduction": best["dim_reduction"],
                            "clustering": best["clustering"],
                            "v_measure": best["metrics"]["v_measure"],
                            "ari": best["metrics"]["ari"],
                        }
                        dim_red_upper = best["dim_reduction"].upper()
                        cluster_upper = best["clustering"].upper()
                        v_measure_val = best["metrics"]["v_measure"]
                        msg = rep_name.upper() + " Best Pipeline: "
                        msg += dim_red_upper + " + " + cluster_upper + " "
                        msg += "(V=" + "{:.4f}".format(v_measure_val) + ")"
                        logger.info(msg)
        else:
            logger.info(
                "Clustering results file not found, will compute default clustering"
            )

    # Load filtered reviews
    logger.info("Loading filtered reviews...")
    df = create_length_labels()
    logger.info("Loaded " + str(len(df)) + " reviews")

    # Get ground truth labels
    true_labels = df["length_label"].map({"Short": 0, "Long": 1}).values
    reviews_text = df["review_text"].astype(str).tolist()

    # Compute TF-IDF representation
    logger.info("Computing TF-IDF representation...")
    tfidf_vectorizer = TfidfVectorizer(
        min_df=3, stop_words="english", ngram_range=(1, 1), norm="l2"
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(reviews_text)
    logger.info("TF-IDF matrix shape: " + str(tfidf_matrix.shape))

    # Compute MiniLM embeddings
    logger.info("Computing MiniLM embeddings...")
    device = get_device()
    logger.info("Using device: " + device)

    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)
    memory = Memory(location=str(cache_dir), verbose=0)

    @memory.cache
    def compute_minilm_cached(texts, device_name):
        model = SentenceTransformer("all-MiniLM-L6-v2", device=device_name)
        embeddings = model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings

    minilm_embeddings = compute_minilm_cached(reviews_text, device)
    logger.info("MiniLM embeddings shape: " + str(minilm_embeddings.shape))

    # Determine best clustering methods
    if "tfidf" in best_pipelines:
        tfidf_dim_red = best_pipelines["tfidf"]["dim_reduction"]
        tfidf_cluster_method = best_pipelines["tfidf"]["clustering"]
    else:
        tfidf_dim_red = "svd"
        tfidf_cluster_method = "kmeans"
        best_pipelines["tfidf"] = {
            "dim_reduction": "svd",
            "clustering": "kmeans",
            "v_measure": None,
            "ari": None,
        }
        logger.info("Using default TF-IDF pipeline: SVD + K-Means")

    if "minilm" in best_pipelines:
        minilm_dim_red = best_pipelines["minilm"]["dim_reduction"]
        minilm_cluster_method = best_pipelines["minilm"]["clustering"]
    else:
        minilm_dim_red = "svd"
        minilm_cluster_method = "kmeans"
        best_pipelines["minilm"] = {
            "dim_reduction": "svd",
            "clustering": "kmeans",
            "v_measure": None,
            "ari": None,
        }
        logger.info("Using default MiniLM pipeline: SVD + K-Means")

    logger.info(
        "TF-IDF pipeline: "
        + tfidf_dim_red.upper()
        + " + "
        + tfidf_cluster_method.upper()
    )
    logger.info(
        "MiniLM pipeline: "
        + minilm_dim_red.upper()
        + " + "
        + minilm_cluster_method.upper()
    )

    # Compute cluster labels for each representation
    logger.info("Computing cluster labels for visualizations...")

    # TF-IDF clustering
    tfidf_reduced = apply_dimensionality_reduction(
        tfidf_matrix, tfidf_dim_red, n_components=50
    )
    tfidf_cluster_labels, _ = run_clustering_pipeline(
        tfidf_reduced, tfidf_cluster_method, n_clusters=2
    )
    unique_tfidf, counts_tfidf = np.unique(tfidf_cluster_labels, return_counts=True)
    logger.info(
        "TF-IDF cluster distribution: " + str(dict(zip(unique_tfidf, counts_tfidf)))
    )

    # MiniLM clustering
    minilm_reduced = apply_dimensionality_reduction(
        minilm_embeddings, minilm_dim_red, n_components=50
    )
    minilm_cluster_labels, _ = run_clustering_pipeline(
        minilm_reduced, minilm_cluster_method, n_clusters=2
    )
    unique_minilm, counts_minilm = np.unique(minilm_cluster_labels, return_counts=True)
    logger.info(
        "MiniLM cluster distribution: " + str(dict(zip(unique_minilm, counts_minilm)))
    )

    # Apply PCA to each representation (reduce to 2D for visualization)
    logger.info("Applying PCA for visualization...")
    pca_tfidf = PCA(n_components=2, random_state=42)
    tfidf_pca = pca_tfidf.fit_transform(tfidf_matrix.toarray())
    logger.info("TF-IDF PCA shape: " + str(tfidf_pca.shape))
    logger.info(
        "TF-IDF explained variance ratio: " + str(pca_tfidf.explained_variance_ratio_)
    )

    pca_minilm = PCA(n_components=2, random_state=42)
    minilm_pca = pca_minilm.fit_transform(minilm_embeddings)
    logger.info("MiniLM PCA shape: " + str(minilm_pca.shape))
    logger.info(
        "MiniLM explained variance ratio: " + str(pca_minilm.explained_variance_ratio_)
    )

    # Create visualization plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        "PCA Visualization: TF-IDF vs MiniLM Representations",
        fontsize=16,
        fontweight="bold",
    )

    # TF-IDF - Ground Truth
    ax = axes[0, 0]
    for label, color, label_name in [(0, "blue", "Short"), (1, "red", "Long")]:
        mask = true_labels == label
        if mask.sum() > 0:
            ax.scatter(
                tfidf_pca[mask, 0],
                tfidf_pca[mask, 1],
                c=color,
                label=label_name,
                alpha=0.6,
                s=10,
            )
    ax.set_title("TF-IDF: Ground Truth (Short vs Long)", fontsize=12, fontweight="bold")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # TF-IDF - Clusters
    ax = axes[0, 1]
    unique_labels = np.unique(tfidf_cluster_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    for i, label in enumerate(unique_labels):
        mask = tfidf_cluster_labels == label
        label_text = "Cluster " + str(label) if label != -1 else "Noise"
        if mask.sum() > 0:
            ax.scatter(
                tfidf_pca[mask, 0],
                tfidf_pca[mask, 1],
                c=[colors[i]],
                label=label_text,
                alpha=0.6,
                s=10,
            )
    if "tfidf" in best_pipelines:
        best = best_pipelines["tfidf"]
        dim_red_upper = best["dim_reduction"].upper()
        cluster_upper = best["clustering"].upper()
        v_measure_val = best["v_measure"]
        title = "TF-IDF: Best Clustering\n"
        title += dim_red_upper + " + " + cluster_upper + "\n"
        if v_measure_val is not None:
            title += "V-Measure: " + "{:.4f}".format(v_measure_val)
    else:
        title = "TF-IDF: Clusters (Default Pipeline)"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MiniLM - Ground Truth
    ax = axes[1, 0]
    for label, color, label_name in [(0, "blue", "Short"), (1, "red", "Long")]:
        mask = true_labels == label
        if mask.sum() > 0:
            ax.scatter(
                minilm_pca[mask, 0],
                minilm_pca[mask, 1],
                c=color,
                label=label_name,
                alpha=0.6,
                s=10,
            )
    ax.set_title("MiniLM: Ground Truth (Short vs Long)", fontsize=12, fontweight="bold")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MiniLM - Clusters
    ax = axes[1, 1]
    unique_labels = np.unique(minilm_cluster_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    for i, label in enumerate(unique_labels):
        mask = minilm_cluster_labels == label
        label_text = "Cluster " + str(label) if label != -1 else "Noise"
        if mask.sum() > 0:
            ax.scatter(
                minilm_pca[mask, 0],
                minilm_pca[mask, 1],
                c=[colors[i]],
                label=label_text,
                alpha=0.6,
                s=10,
            )
    if "minilm" in best_pipelines:
        best = best_pipelines["minilm"]
        dim_red_upper = best["dim_reduction"].upper()
        cluster_upper = best["clustering"].upper()
        v_measure_val = best["v_measure"]
        title = "MiniLM: Best Clustering\n"
        title += dim_red_upper + " + " + cluster_upper + "\n"
        if v_measure_val is not None:
            title += "V-Measure: " + "{:.4f}".format(v_measure_val)
    else:
        title = "MiniLM: Clusters (Default Pipeline)"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / "Q5_pca_visualizations.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    logger.info("Saved PCA visualizations to " + str(plot_path))

    # Close plot
    plt.close()

    # Build results dictionary
    results = {
        "plot_path": str(plot_path),
        "best_pipelines": best_pipelines,
        "tfidf_pca": {
            "explained_variance_ratio": pca_tfidf.explained_variance_ratio_.tolist(),
            "total_variance_explained": float(
                pca_tfidf.explained_variance_ratio_.sum()
            ),
        },
        "minilm_pca": {
            "explained_variance_ratio": pca_minilm.explained_variance_ratio_.tolist(),
            "total_variance_explained": float(
                pca_minilm.explained_variance_ratio_.sum()
            ),
        },
    }

    # Save results to JSON
    results_json_path = output_dir / "Q5_pca_results.json"
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved PCA results to " + str(results_json_path))

    logger.info("Finished plot_pca_visualizations")

    return results


# =============================================================================
# TASK 2: UNSUPERVISED GAME SIMILARITY & GENRE STRUCTURE (Q6-Q8)
# =============================================================================


def load_main_dataset():
    """Load the main Steam reviews dataset."""
    logger = logging.getLogger(__name__)
    dataset_path = Path("datasets/steam/main.csv")
    logger.info(f"Loading main dataset from {dataset_path}")
    df = pd.read_csv(dataset_path)
    logger.info(f"Loaded {len(df)} reviews for {df['game_name'].nunique()} games")
    return df


def run_task2_1():
    """
    Task 2.1: Construct game vectors from positive reviews only.
    Saves results to outputs/Q6_game_vectors.json
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("TASK 2.1: Game Vector Construction (Q6)")
    logger.info("=" * 60)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    df = load_main_dataset()
    df_positive = df[df["recommend"]].copy()
    logger.info(f"Positive reviews: {len(df_positive)} out of {len(df)}")

    game_groups = df_positive.groupby(["appid", "game_name"])
    logger.info(f"Number of games: {len(game_groups)}")

    game_info = []
    game_docs = []
    review_lists = []

    for (appid, game_name), group in game_groups:
        genres = group["genres"].iloc[0] if "genres" in group.columns else ""
        reviews = group["review_text"].astype(str).tolist()
        game_info.append(
            {
                "appid": appid,
                "game_name": game_name,
                "genres": genres,
                "num_reviews": len(reviews),
            }
        )
        game_docs.append(" ".join(reviews))
        review_lists.append(reviews)

    # TF-IDF game vectors
    logger.info("Computing TF-IDF game vectors...")
    tfidf_vectorizer = TfidfVectorizer(
        min_df=3, stop_words="english", ngram_range=(1, 1), norm="l2"
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(game_docs)
    logger.info(f"TF-IDF game matrix shape: {tfidf_matrix.shape}")

    # MiniLM game vectors
    logger.info("Computing MiniLM game vectors...")
    device = get_device()
    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)
    memory = Memory(location=str(cache_dir), verbose=0)

    @memory.cache
    def compute_minilm_game_cached(texts, device_name):
        model = SentenceTransformer("all-MiniLM-L6-v2", device=device_name)
        return model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )

    minilm_game_vectors = []
    for i, reviews in enumerate(review_lists):
        embeddings = compute_minilm_game_cached(reviews, device)
        minilm_game_vectors.append(embeddings.mean(axis=0))
        if (i + 1) % 50 == 0:
            logger.info(f"Processed {i + 1}/{len(review_lists)} games")

    minilm_matrix = np.array(minilm_game_vectors)
    logger.info(f"MiniLM game matrix shape: {minilm_matrix.shape}")

    game_df = pd.DataFrame(game_info)

    # Save results
    results = {
        "tfidf_shape": list(tfidf_matrix.shape),
        "minilm_shape": list(minilm_matrix.shape),
        "num_games": len(game_info),
        "games": game_info[:10],
    }
    with open(output_dir / "Q6_game_vectors.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    np.save(output_dir / "Q6_minilm_matrix.npy", minilm_matrix)
    game_df.to_csv(output_dir / "Q6_game_info.csv", index=False)

    logger.info(f"Saved results to {output_dir / 'Q6_game_vectors.json'}")

    return {
        "tfidf_matrix": tfidf_matrix,
        "minilm_matrix": minilm_matrix,
        "game_df": game_df,
        "tfidf_vectorizer": tfidf_vectorizer,
    }


def get_top_genres_for_cluster(game_df, labels, cluster_id):
    """Get top genres for a cluster."""
    cluster_games = game_df[labels == cluster_id]
    all_genres = []
    for genres_str in cluster_games["genres"]:
        if pd.notna(genres_str):
            all_genres.extend([g.strip() for g in str(genres_str).split(",")])

    genre_counts = Counter(all_genres)
    total = len(cluster_games)

    top_genres = []
    for genre, count in genre_counts.most_common(3):
        top_genres.append(
            {
                "genre": genre,
                "count": count,
                "percentage": (count / total) * 100 if total > 0 else 0,
            }
        )

    return top_genres, genre_counts


def compute_genre_purity(game_df, labels, cluster_id):
    """Compute genre purity: fraction of games with the most common genre."""
    cluster_games = game_df[labels == cluster_id]
    if len(cluster_games) == 0:
        return 0.0

    all_genres = []
    for genres_str in cluster_games["genres"]:
        if pd.notna(genres_str):
            all_genres.append(set([g.strip() for g in str(genres_str).split(",")]))

    if not all_genres:
        return 0.0

    genre_counter = Counter()
    for genre_set in all_genres:
        genre_counter.update(genre_set)

    if not genre_counter:
        return 0.0

    most_common_genre = genre_counter.most_common(1)[0][0]
    games_with_common = sum(
        1 for genre_set in all_genres if most_common_genre in genre_set
    )

    return games_with_common / len(all_genres)


def compute_genre_entropy(game_df, labels, cluster_id):
    """Compute genre entropy using Shannon entropy formula (natural log)."""
    cluster_games = game_df[labels == cluster_id]
    if len(cluster_games) == 0:
        return 0.0

    all_genres = []
    for genres_str in cluster_games["genres"]:
        if pd.notna(genres_str):
            all_genres.append(set([g.strip() for g in str(genres_str).split(",")]))

    if not all_genres:
        return 0.0

    genre_counter = Counter()
    for genre_set in all_genres:
        genre_counter.update(genre_set)

    if not genre_counter:
        return 0.0

    genre_counts = list(genre_counter.values())
    total = sum(genre_counts)
    probabilities = [count / total for count in genre_counts]

    return entropy(probabilities)


def run_task2_2(game_data=None):
    """
    Task 2.2: Cluster games with default pipelines.
    Saves results to outputs/Q7_Q8_game_clustering.json
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("TASK 2.2: Game Clustering (Q7-Q8)")
    logger.info("=" * 60)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    if game_data is None:
        game_data = run_task2_1()

    tfidf_matrix = game_data["tfidf_matrix"]
    minilm_matrix = game_data["minilm_matrix"]
    game_df = game_data["game_df"]

    pipelines = [
        {"rep": "minilm", "dim": "none", "cluster": "kmeans"},
        {"rep": "minilm", "dim": "none", "cluster": "agglomerative"},
        {"rep": "minilm", "dim": "svd", "cluster": "kmeans"},
        {"rep": "minilm", "dim": "svd", "cluster": "agglomerative"},
        {"rep": "minilm", "dim": "umap", "cluster": "kmeans"},
        {"rep": "minilm", "dim": "umap", "cluster": "agglomerative"},
        {"rep": "minilm", "dim": "none", "cluster": "hdbscan"},
        {"rep": "tfidf", "dim": "svd", "cluster": "kmeans"},
        {"rep": "tfidf", "dim": "svd", "cluster": "agglomerative"},
        {"rep": "minilm", "dim": "autoencoder", "cluster": "kmeans"},
        {"rep": "minilm", "dim": "autoencoder", "cluster": "agglomerative"},
        {"rep": "tfidf", "dim": "autoencoder", "cluster": "kmeans"},
        {"rep": "tfidf", "dim": "autoencoder", "cluster": "agglomerative"},
        {"rep": "minilm", "dim": "svd", "cluster": "hdbscan"},
        {"rep": "minilm", "dim": "umap", "cluster": "hdbscan"},
        {"rep": "minilm", "dim": "autoencoder", "cluster": "hdbscan"},
    ]

    results = []

    for i, pipeline in enumerate(pipelines, 1):
        logger.info(
            f"Pipeline {i}/{len(pipelines)}: {pipeline['rep']} - {pipeline['dim']} - {pipeline['cluster']}"
        )

        X = minilm_matrix if pipeline["rep"] == "minilm" else tfidf_matrix

        if pipeline["dim"] != "none":
            X_reduced = apply_dimensionality_reduction(
                X, pipeline["dim"], n_components=50
            )
        else:
            X_reduced = X if isinstance(X, np.ndarray) else X.toarray()

        n_clusters = 5 if pipeline["cluster"] != "hdbscan" else None
        hdbscan_min_size = 5 if pipeline["cluster"] == "hdbscan" else 2
        labels, model = run_clustering_pipeline(
            X_reduced,
            pipeline["cluster"],
            n_clusters=n_clusters,
            hdbscan_min_size=hdbscan_min_size,
        )

        unique_labels = np.unique(labels)
        n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)
        noise_count = int((labels == -1).sum()) if -1 in unique_labels else 0

        cluster_details = []
        for label in sorted(unique_labels):
            if label == -1:
                continue
            top_genres, _ = get_top_genres_for_cluster(game_df, labels, label)
            purity = compute_genre_purity(game_df, labels, label)
            entropy_value = compute_genre_entropy(game_df, labels, label)
            cluster_details.append(
                {
                    "cluster_id": int(label),
                    "size": int((labels == label).sum()),
                    "top_genres": top_genres,
                    "purity": float(purity),
                    "entropy": float(entropy_value),
                }
            )

        results.append(
            {
                "pipeline_id": i,
                "representation": pipeline["rep"],
                "dim_reduction": pipeline["dim"],
                "clustering": pipeline["cluster"],
                "n_clusters_found": n_clusters_found,
                "noise_count": noise_count,
                "noise_fraction": float(noise_count / len(labels))
                if len(labels) > 0
                else 0,
                "clusters": cluster_details,
            }
        )

    with open(output_dir / "Q7_Q8_game_clustering.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save best model for Task 3
    best_X = apply_dimensionality_reduction(minilm_matrix, "svd", n_components=50)
    best_labels, best_model = run_clustering_pipeline(best_X, "kmeans", n_clusters=5)
    np.save(output_dir / "Q7_Q8_best_labels.npy", best_labels)

    import pickle

    with open(output_dir / "Q7_Q8_best_model.pkl", "wb") as f:
        pickle.dump(
            {"model": best_model, "X_reduced": best_X, "labels": best_labels}, f
        )

    logger.info(
        f"Saved clustering results to {output_dir / 'Q7_Q8_game_clustering.json'}"
    )

    return results


# =============================================================================
# TASK 3: HELD-OUT GAME PROFILING AND THEME DISCOVERY (Q9-Q12)
# =============================================================================


def load_heldout_dataset():
    """Load the held-out game reviews dataset."""
    logger = logging.getLogger(__name__)
    dataset_path = Path("datasets/steam/heldout.csv")
    logger.info(f"Loading heldout dataset from {dataset_path}")
    df = pd.read_csv(dataset_path)
    logger.info(f"Loaded {len(df)} heldout reviews")
    return df


def run_task3_1(game_data=None, clustering_results=None):
    """
    Task 3.1: Genre estimation for held-out game.
    Saves results to outputs/Q9_genre_estimation.json
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("TASK 3.1: Held-out Game Genre Estimation (Q9)")
    logger.info("=" * 60)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    heldout_df = load_heldout_dataset()
    heldout_positive = heldout_df[heldout_df["recommend"]].copy()
    logger.info(f"Positive heldout reviews: {len(heldout_positive)}")

    if game_data is None:
        game_data = run_task2_1()

    game_df = game_data["game_df"]

    # Compute heldout game vector
    device = get_device()
    cache_dir = Path(".cache")
    memory = Memory(location=str(cache_dir), verbose=0)

    @memory.cache
    def compute_minilm_heldout_cached(texts, device_name):
        model = SentenceTransformer("all-MiniLM-L6-v2", device=device_name)
        return model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )

    heldout_reviews = heldout_positive["review_text"].astype(str).tolist()
    heldout_embeddings = compute_minilm_heldout_cached(heldout_reviews, device)
    heldout_game_vector = heldout_embeddings.mean(axis=0)

    # Load best model
    import pickle

    with open(output_dir / "Q7_Q8_best_model.pkl", "rb") as f:
        saved_data = pickle.load(f)

    best_model = saved_data["model"]
    labels = saved_data["labels"]

    # Reduce heldout vector
    minilm_matrix = np.load(output_dir / "Q6_minilm_matrix.npy")
    svd = TruncatedSVD(n_components=50, random_state=42)
    svd.fit(minilm_matrix)
    heldout_reduced = svd.transform(heldout_game_vector.reshape(1, -1))

    # Find nearest cluster
    if hasattr(best_model, "cluster_centers_"):
        distances = np.linalg.norm(
            best_model.cluster_centers_ - heldout_reduced, axis=1
        )
        nearest_cluster = int(np.argmin(distances))
    else:
        nearest_cluster = 0

    logger.info(f"Assigned cluster: {nearest_cluster}")

    top_genres, _ = get_top_genres_for_cluster(game_df, labels, nearest_cluster)
    cluster_games = game_df[labels == nearest_cluster]
    representative_games = cluster_games.head(3)[["game_name", "genres"]].to_dict(
        "records"
    )

    results = {
        "assigned_cluster": nearest_cluster,
        "top_genres": top_genres,
        "representative_games": representative_games,
        "heldout_positive_reviews": len(heldout_positive),
    }

    with open(output_dir / "Q9_genre_estimation.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Saved results to {output_dir / 'Q9_genre_estimation.json'}")

    return results


def get_top_tfidf_terms(reviews, n_terms=10):
    """Get top TF-IDF terms from a list of reviews."""
    vectorizer = TfidfVectorizer(
        min_df=1, stop_words="english", ngram_range=(1, 2), max_features=1000
    )
    tfidf_matrix = vectorizer.fit_transform(reviews)
    feature_names = vectorizer.get_feature_names_out()

    mean_tfidf = np.array(tfidf_matrix.mean(axis=0)).flatten()
    top_indices = mean_tfidf.argsort()[-n_terms:][::-1]

    return [
        {"term": feature_names[idx], "score": float(mean_tfidf[idx])}
        for idx in top_indices
    ]


def get_exemplar_reviews(reviews, embeddings, labels, cluster_id, n_exemplars=2):
    """Get exemplar reviews closest to cluster centroid."""
    cluster_indices = np.where(labels == cluster_id)[0]
    if len(cluster_indices) == 0:
        return []

    cluster_embeddings = embeddings[cluster_indices]
    centroid = cluster_embeddings.mean(axis=0)
    distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
    closest_indices = distances.argsort()[:n_exemplars]

    exemplars = []
    for idx in closest_indices:
        original_idx = cluster_indices[idx]
        review_text = reviews[original_idx]
        exemplars.append(
            {
                "review": review_text[:500] + "..."
                if len(review_text) > 500
                else review_text,
                "distance": float(distances[idx]),
            }
        )

    return exemplars


def run_task3_2():
    """
    Task 3.2: Theme clustering for held-out game reviews.
    Saves results to outputs/Q10_Q11_theme_clustering.json
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("TASK 3.2: Theme Discovery (Q10-Q11)")
    logger.info("=" * 60)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    heldout_df = load_heldout_dataset()
    positive_df = heldout_df[heldout_df["recommend"]].copy()
    negative_df = heldout_df[~heldout_df["recommend"]].copy()

    logger.info(
        f"Positive reviews: {len(positive_df)}, Negative reviews: {len(negative_df)}"
    )

    device = get_device()
    cache_dir = Path(".cache")
    memory = Memory(location=str(cache_dir), verbose=0)

    @memory.cache
    def compute_minilm_theme_cached(texts, device_name):
        model = SentenceTransformer("all-MiniLM-L6-v2", device=device_name)
        return model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )

    results = {}
    llm_examples = []

    qwen_model, qwen_tokenizer = setup_qwen_model()
    llm_prompt_template = """I have a cluster of {review_type} reviews from a video game.
Top terms: {top_terms}
Example reviews:
{exemplars}

Generate a short 3-6 word label describing this cluster's theme.
Label:"""

    for review_type, df in [("positive", positive_df), ("negative", negative_df)]:
        logger.info(f"\nProcessing {review_type.upper()} reviews")

        reviews = df["review_text"].astype(str).tolist()
        embeddings = compute_minilm_theme_cached(reviews, device)

        svd = TruncatedSVD(n_components=50, random_state=42)
        X_reduced = svd.fit_transform(embeddings)

        # K-Means
        kmeans_labels, _ = run_clustering_pipeline(X_reduced, "kmeans", n_clusters=5)
        # Agglomerative
        agg_labels, _ = run_clustering_pipeline(
            X_reduced, "agglomerative", n_clusters=5
        )
        # HDBSCAN
        hdb_labels, _ = run_clustering_pipeline(X_reduced, "hdbscan")
        noise_count = int((hdb_labels == -1).sum())

        # Use K-Means for analysis
        labels = kmeans_labels

        cluster_analysis = []
        for cluster_id in range(5):
            cluster_reviews = [
                reviews[i] for i in range(len(reviews)) if labels[i] == cluster_id
            ]
            if len(cluster_reviews) == 0:
                continue

            top_terms = get_top_tfidf_terms(cluster_reviews, n_terms=10)
            exemplars = get_exemplar_reviews(
                reviews, embeddings, labels, cluster_id, n_exemplars=2
            )

            llm_label = generate_llm_cluster_label(
                top_terms, exemplars, review_type, qwen_model, qwen_tokenizer
            )

            if llm_label and len(llm_examples) < 6:
                terms_str = ", ".join([t["term"] for t in top_terms[:5]])
                exemplar_texts = [ex.get("review", "")[:200] for ex in exemplars[:2]]
                llm_examples.append(
                    {
                        "review_type": review_type,
                        "cluster_id": cluster_id,
                        "prompt": llm_prompt_template.format(
                            review_type=review_type,
                            top_terms=terms_str,
                            exemplars="\n".join([f'"{e}..."' for e in exemplar_texts]),
                        ),
                        "response": llm_label,
                    }
                )

            cluster_info = {
                "cluster_id": cluster_id,
                "size": len(cluster_reviews),
                "top_terms": top_terms,
                "exemplars": exemplars,
            }
            if llm_label:
                cluster_info["llm_label"] = llm_label
                cluster_info["short_label"] = llm_label
            else:
                short_label = " ".join([t["term"] for t in top_terms[:3]])
                cluster_info["short_label"] = short_label
            cluster_analysis.append(cluster_info)

        results[review_type] = {
            "n_reviews": len(reviews),
            "clusters": cluster_analysis,
            "kmeans_labels": kmeans_labels.tolist(),
            "agg_labels": agg_labels.tolist(),
            "hdb_labels": hdb_labels.tolist(),
            "hdbscan_noise": noise_count,
        }

    results["llm_prompt_template"] = llm_prompt_template
    results["llm_examples"] = llm_examples

    with open(output_dir / "Q10_Q11_theme_clustering.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Saved results to {output_dir / 'Q10_Q11_theme_clustering.json'}")

    return results


# =============================================================================
# MAIN FUNCTION - RUNS ALL TASKS
# =============================================================================


def main():
    """Run all Part 1 tasks (Task 1, 2, 3) to generate outputs for Q1-Q12."""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("PART 1 - RUNNING ALL TASKS (Q1-Q12)")
    logger.info("=" * 60)

    # Task 1: Review Length Discovery (Q1-Q5)
    logger.info("\n" + "=" * 60)
    logger.info("TASK 1: Review Length Discovery")
    logger.info("=" * 60)

    task1_data = run_task1_1()
    task1_clustering = run_task1_2(task1_data)
    task1_viz = plot_pca_visualizations()

    # Task 2: Game Similarity & Genre Structure (Q6-Q8)
    logger.info("\n" + "=" * 60)
    logger.info("TASK 2: Game Similarity & Genre Structure")
    logger.info("=" * 60)

    task2_game_data = run_task2_1()
    task2_clustering = run_task2_2(task2_game_data)

    # Task 3: Held-out Game Profiling (Q9-Q12)
    logger.info("\n" + "=" * 60)
    logger.info("TASK 3: Held-out Game Profiling")
    logger.info("=" * 60)

    task3_genre = run_task3_1(task2_game_data, task2_clustering)
    task3_themes = run_task3_2()

    logger.info("\n" + "=" * 60)
    logger.info("ALL TASKS COMPLETE")
    logger.info("=" * 60)
    logger.info("\nOutput files generated in outputs/:")
    logger.info("  - Q1_Q2_results.json (Q1-Q2)")
    logger.info("  - Q3_Q4_clustering_results.json (Q3)")
    logger.info("  - Q5_pca_visualizations.png (Q5)")
    logger.info("  - Q6_game_vectors.json (Q6)")
    logger.info("  - Q7_Q8_game_clustering.json (Q7-Q8)")
    logger.info("  - Q9_genre_estimation.json (Q9)")
    logger.info("  - Q10_Q11_theme_clustering.json (Q10-Q11)")
    logger.info("\nSee Part1_Answers.md for complete answers to all 12 questions.")

    return {
        "task1": {"data": task1_data, "clustering": task1_clustering, "viz": task1_viz},
        "task2": {"game_data": task2_game_data, "clustering": task2_clustering},
        "task3": {"genre": task3_genre, "themes": task3_themes},
    }


if __name__ == "__main__":
    main()
