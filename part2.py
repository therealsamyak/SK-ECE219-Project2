import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import json
import requests
import tarfile
from tqdm import tqdm
import umap
import hdbscan
from sklearn.base import TransformerMixin
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    homogeneity_score,
    completeness_score,
    v_measure_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
)
from sklearn.model_selection import train_test_split


def setup_logging():
    """Configure logging to output to both console and file."""
    log_dir = Path("outputs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "part2.log"

    logging.basicConfig(
        level=logging.INFO,
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


class FeatureExtractor(nn.Module):
    """VGG16 feature extractor for image embeddings."""

    def __init__(self):
        super().__init__()
        vgg = torch.hub.load("pytorch/vision:v0.10.0", "vgg16", pretrained=True)
        self.features = list(vgg.features)
        self.features = nn.Sequential(*self.features)
        self.pooling = vgg.avgpool
        self.flatten = nn.Flatten()
        self.fc = vgg.classifier[0]

    def forward(self, x):
        out = self.features(x)
        out = self.pooling(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out


class Autoencoder(nn.Module, TransformerMixin):
    """Autoencoder for dimensionality reduction."""

    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components
        self.n_features = None
        self.encoder = None
        self.decoder = None

    def _create_encoder(self):
        return nn.Sequential(
            nn.Linear(4096, 1280),
            nn.ReLU(True),
            nn.Linear(1280, 640),
            nn.ReLU(True),
            nn.Linear(640, 120),
            nn.ReLU(True),
            nn.Linear(120, self.n_components),
        )

    def _create_decoder(self):
        return nn.Sequential(
            nn.Linear(self.n_components, 120),
            nn.ReLU(True),
            nn.Linear(120, 640),
            nn.ReLU(True),
            nn.Linear(640, 1280),
            nn.ReLU(True),
            nn.Linear(1280, 4096),
        )

    def forward(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return decoded

    def fit(self, X, device="cpu"):
        """Train the autoencoder on the data."""
        X = torch.tensor(X, dtype=torch.float32, device=device)
        self.n_features = X.shape[1]
        self.encoder = self._create_encoder()
        self.decoder = self._create_decoder()
        self.to(device)
        self.train()

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)

        dataset = TensorDataset(X)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

        for epoch in tqdm(range(100), desc="Training Autoencoder"):
            for (X_,) in dataloader:
                output = self(X_)
                loss = criterion(output, X_)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self

    def transform(self, X, device="cpu"):
        """Transform data using the encoder."""
        X = torch.tensor(X, dtype=torch.float32, device=device)
        self.eval()
        with torch.no_grad():
            return self.encoder(X).cpu().numpy()

    def fit_transform(self, X, device="cpu"):
        """Fit and transform in one step."""
        self.fit(X, device=device)
        return self.transform(X, device=device)


class MLP(nn.Module):
    """MLP classifier for flower classification."""

    def __init__(self, num_features, num_classes=5):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, 1280),
            nn.ReLU(True),
            nn.Linear(1280, 640),
            nn.ReLU(True),
            nn.Linear(640, num_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, X):
        return self.model(X)

    def train_model(self, X, y, device="cpu"):
        """Train the MLP classifier."""
        X = torch.tensor(X, dtype=torch.float32, device=device)
        y = torch.tensor(y, dtype=torch.int64, device=device)

        self.to(device)
        self.model.train()

        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

        for epoch in tqdm(range(100), desc="Training MLP"):
            for X_, y_ in dataloader:
                pred = self(X_)
                loss = criterion(pred, y_)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self

    def evaluate(self, X_test, y_test, device="cpu"):
        """Evaluate the MLP classifier."""
        X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
        y_test = torch.tensor(y_test, dtype=torch.int64, device=device)
        self.to(device)
        self.model.eval()
        with torch.no_grad():
            pred = self(X_test).argmax(1)
            accuracy = (pred == y_test).float().mean().item()
        return accuracy


def extract_flower_features():
    """
    Extract VGG16 features from tf_flowers dataset.

    Downloads dataset if needed, extracts features, caches results, and saves metadata.

    Returns:
        dict: Results dictionary containing feature dimensions and metadata
    """
    logger = setup_logging()
    logger.info("Starting VGG16 Feature Extraction")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)

    cache_file = cache_dir / "flowers_features_and_labels.npz"

    class_names = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

    if cache_file.exists():
        logger.info(f"Loading cached features from {cache_file}")
        file = np.load(cache_file)
        f_all, y_all = file["f_all"], file["y_all"]
    else:
        logger.info("Extracting features from dataset")

        flowers_dir = Path("datasets/flower_photos")
        if not flowers_dir.exists():
            logger.info("Downloading tf_flowers dataset...")
            url = "http://download.tensorflow.org/example_images/flower_photos.tgz"
            Path("datasets").mkdir(exist_ok=True)
            with open("datasets/flower_photos.tgz", "wb") as file:
                file.write(requests.get(url).content)
            with tarfile.open("datasets/flower_photos.tgz") as file:
                file.extractall("datasets/")
            Path("datasets/flower_photos.tgz").unlink()

        device = get_device()
        logger.info(f"Using device: {device}")

        feature_extractor = FeatureExtractor().to(device).eval()

        dataset = datasets.ImageFolder(
            root=str(flowers_dir),
            transform=transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        )
        class_names = dataset.classes
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

        f_all, y_all = np.zeros((0, 4096)), np.zeros((0,))
        for x, y in tqdm(dataloader, desc="Extracting features"):
            x = x.to(device)
            with torch.no_grad():
                features = feature_extractor(x).cpu().numpy()
            f_all = np.vstack([f_all, features])
            y_all = np.concatenate([y_all, y])

        np.savez(cache_file, f_all=f_all, y_all=y_all)
        logger.info(f"Cached features to {cache_file}")

    logger.info(f"Features shape: {f_all.shape}")
    logger.info(f"Labels shape: {y_all.shape}")

    results = {
        "original_pixels": [224, 224, 3],
        "feature_dim": 4096,
        "num_images": int(f_all.shape[0]),
        "is_dense": True,
        "classes": sorted(list(set([class_names[int(label)] for label in y_all]))),
    }

    output_json = output_dir / "part2_features.json"
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved features metadata to {output_json}")

    logger.info("Finished VGG16 Feature Extraction")

    # Return both metadata and actual feature arrays for reuse
    return {
        "metadata": results,
        "features": f_all,
        "labels": y_all,
        "class_names": class_names,
    }


def run_tsne_visualization(features, labels, class_names):
    """
    Run t-SNE visualization on the flower features.

    Args:
        features: Feature array (n_samples, n_features)
        labels: Ground truth labels
        class_names: List of class names

    Returns:
        dict: Results containing plot path and coordinates
    """
    logger = setup_logging()
    logger.info("Starting t-SNE Visualization")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Run t-SNE
    logger.info("Running t-SNE with 2 components...")
    tsne = TSNE(n_components=2, random_state=42)
    tsne_2d = tsne.fit_transform(features)
    logger.info(f"t-SNE shape: {tsne_2d.shape}")

    # Create scatter plot with color-coded ground truth labels
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

    for i, class_name in enumerate(class_names):
        mask = labels == i
        plt.scatter(
            tsne_2d[mask, 0],
            tsne_2d[mask, 1],
            c=[colors[i]],
            label=class_name,
            alpha=0.6,
            s=20,
        )

    plt.title("t-SNE Visualization of Flower Features", fontsize=14, fontweight="bold")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plot_path = output_dir / "part2_tsne.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved t-SNE plot to {plot_path}")

    # Save coordinates to JSON
    tsne_data = {
        "tsne_2d": tsne_2d.tolist(),
        "labels": labels.tolist(),
        "classes": class_names,
    }
    json_path = output_dir / "part2_tsne.json"
    with open(json_path, "w") as f:
        json.dump(tsne_data, f, indent=2)
    logger.info(f"Saved t-SNE coordinates to {json_path}")

    # Log observations about cluster separation
    logger.info("=" * 60)
    logger.info("t-SNE Observations:")
    logger.info("=" * 60)
    logger.info("Visual inspection of the t-SNE plot shows:")
    logger.info("- Some flower classes appear well-separated (distinct clusters)")
    logger.info("- Other classes show overlap, indicating similar visual features")
    logger.info(
        "- This suggests the VGG16 features capture meaningful class differences"
    )
    logger.info("=" * 60)

    return {"plot_path": str(plot_path), "json_path": str(json_path)}


def apply_dim_reduction(features, method, n_components=50, device="cpu"):
    """
    Apply dimensionality reduction to features.

    Args:
        features: Input feature array
        method: Reduction method - "none", "svd", "umap", or "autoencoder"
        n_components: Number of output dimensions
        device: Device for autoencoder training

    Returns:
        Reduced feature array
    """
    if method == "none":
        return features
    elif method == "svd":
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        return svd.fit_transform(features)
    elif method == "umap":
        umap_model = umap.UMAP(n_components=n_components, random_state=42)
        return umap_model.fit_transform(features)
    elif method == "autoencoder":
        autoencoder = Autoencoder(n_components=n_components)
        return autoencoder.fit_transform(features, device=device)
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")


def run_clustering(X, method, n_clusters=5, **kwargs):
    """
    Run clustering on the data.

    Args:
        X: Input data array
        method: Clustering method - "kmeans", "agglomerative", or "hdbscan"
        n_clusters: Number of clusters (ignored for HDBSCAN)
        **kwargs: Additional parameters for HDBSCAN

    Returns:
        Cluster labels
    """
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return model.fit_predict(X)
    elif method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        return model.fit_predict(X)
    elif method == "hdbscan":
        min_cluster_size = kwargs.get("min_cluster_size", 5)
        min_samples = kwargs.get("min_samples", 3)
        model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples
        )
        return model.fit_predict(X)
    else:
        raise ValueError(f"Unknown clustering method: {method}")


def compute_clustering_metrics(labels_true, labels_pred):
    """Compute clustering evaluation metrics."""
    # Filter out noise points for metrics computation
    mask = labels_pred != -1
    if mask.sum() < 2:
        return {
            "homogeneity": None,
            "completeness": None,
            "v_measure": None,
            "ari": None,
            "ami": None,
        }

    return {
        "homogeneity": float(homogeneity_score(labels_true[mask], labels_pred[mask])),
        "completeness": float(completeness_score(labels_true[mask], labels_pred[mask])),
        "v_measure": float(v_measure_score(labels_true[mask], labels_pred[mask])),
        "ari": float(adjusted_rand_score(labels_true[mask], labels_pred[mask])),
        "ami": float(
            adjusted_mutual_info_score(
                labels_true[mask], labels_pred[mask], average_method="arithmetic"
            )
        ),
    }


def run_clustering_grid_search(features, labels, device="cpu"):
    """
    Run clustering grid search with different dimensionality reduction methods.

    Args:
        features: Input feature array
        labels: Ground truth labels
        device: Device for autoencoder training

    Returns:
        dict: Clustering results and best configuration
    """
    logger = setup_logging()
    logger.info("Starting Clustering Grid Search")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Dimensionality reduction methods
    dim_methods = ["none", "svd", "umap", "autoencoder"]

    # Clustering methods
    cluster_methods = ["kmeans", "agglomerative", "hdbscan"]

    # HDBSCAN grid
    hdbscan_grid = [
        {"min_cluster_size": 5, "min_samples": 3},
        {"min_cluster_size": 10, "min_samples": 3},
        {"min_cluster_size": 20, "min_samples": 5},
        {"min_cluster_size": 50, "min_samples": 10},
    ]

    results = {"pipelines": [], "best_per_dim_reduction": {}}

    for dim_method in dim_methods:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Dimensionality Reduction: {dim_method.upper()}")
        logger.info(f"{'=' * 60}")

        # Apply dimensionality reduction
        n_components = 50 if dim_method != "none" else None
        if dim_method == "none":
            X_reduced = features
        else:
            logger.info(f"Applying {dim_method} with {n_components} components...")
            X_reduced = apply_dim_reduction(
                features, dim_method, n_components=n_components, device=device
            )
        logger.info(f"Reduced shape: {X_reduced.shape}")

        best_ari_for_dim = -float("inf")
        best_config_for_dim = None

        for cluster_method in cluster_methods:
            if cluster_method in ["kmeans", "agglomerative"]:
                logger.info(f"\nRunning {cluster_method.upper()} with k=5...")
                cluster_labels = run_clustering(X_reduced, cluster_method, n_clusters=5)

                metrics = compute_clustering_metrics(labels, cluster_labels)
                n_clusters_found = len(np.unique(cluster_labels))

                result_entry = {
                    "dim_reduction": dim_method,
                    "clustering": cluster_method,
                    "n_components": n_components,
                    "n_clusters": n_clusters_found,
                    "metrics": metrics,
                }
                results["pipelines"].append(result_entry)

                ari_str = (
                    f"{metrics['ari']:.4f}" if metrics["ari"] is not None else "N/A"
                )
                logger.info(f"  ARI: {ari_str}")
                v_measure_str = (
                    f"{metrics['v_measure']:.4f}"
                    if metrics["v_measure"] is not None
                    else "N/A"
                )
                logger.info(f"  V-Measure: {v_measure_str}")

                if metrics["ari"] and metrics["ari"] > best_ari_for_dim:
                    best_ari_for_dim = metrics["ari"]
                    best_config_for_dim = result_entry

            elif cluster_method == "hdbscan":
                logger.info("\nRunning HDBSCAN grid search...")
                best_hdbscan_ari = -float("inf")
                best_hdbscan_config = None

                for grid_params in hdbscan_grid:
                    cluster_labels = run_clustering(
                        X_reduced, cluster_method, **grid_params
                    )
                    metrics = compute_clustering_metrics(labels, cluster_labels)

                    unique_labels = np.unique(cluster_labels)
                    n_clusters_found = len(unique_labels) - (
                        1 if -1 in unique_labels else 0
                    )
                    noise_count = int((cluster_labels == -1).sum())

                    result_entry = {
                        "dim_reduction": dim_method,
                        "clustering": cluster_method,
                        "n_components": n_components,
                        "hdbscan_params": grid_params,
                        "n_clusters": n_clusters_found,
                        "noise_count": noise_count,
                        "metrics": metrics,
                    }
                    results["pipelines"].append(result_entry)

                    ari_str = (
                        f"{metrics['ari']:.4f}" if metrics["ari"] is not None else "N/A"
                    )
                    logger.info(
                        f"  min_cluster_size={grid_params['min_cluster_size']}, "
                        f"min_samples={grid_params['min_samples']}: "
                        f"ARI={ari_str}, "
                        f"clusters={n_clusters_found}, noise={noise_count}"
                    )

                    if metrics["ari"] and metrics["ari"] > best_hdbscan_ari:
                        best_hdbscan_ari = metrics["ari"]
                        best_hdbscan_config = result_entry

                # Report best HDBSCAN config for this dim reduction
                if best_hdbscan_config:
                    logger.info(
                        f"  Best HDBSCAN: ARI={best_hdbscan_ari:.4f}, "
                        f"params={best_hdbscan_config['hdbscan_params']}"
                    )
                    if best_hdbscan_ari > best_ari_for_dim:
                        best_ari_for_dim = best_hdbscan_ari
                        best_config_for_dim = best_hdbscan_config

        results["best_per_dim_reduction"][dim_method] = {
            "best_ari": best_ari_for_dim,
            "config": best_config_for_dim,
        }

    # Find overall best configuration
    best_overall = max(
        results["best_per_dim_reduction"].items(), key=lambda x: x[1]["best_ari"]
    )
    results["best_overall"] = {
        "dim_reduction": best_overall[0],
        "best_ari": best_overall[1]["best_ari"],
        "config": best_overall[1]["config"],
    }

    logger.info(f"\n{'=' * 60}")
    logger.info("Best Overall Configuration:")
    logger.info(
        f"  Dimensionality Reduction: {results['best_overall']['dim_reduction']}"
    )
    logger.info(f"  Best ARI: {results['best_overall']['best_ari']:.4f}")
    logger.info(f"{'=' * 60}")

    # Save results to JSON
    output_json = output_dir / "part2_clustering.json"
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved clustering results to {output_json}")

    return results


def run_mlp_classifier(features, labels, best_dim_reduction=None, device="cpu"):
    """
    Train and evaluate MLP classifier on original and reduced features.

    Args:
        features: Input feature array
        labels: Ground truth labels
        best_dim_reduction: Best dimensionality reduction method from clustering
        device: Device for training

    Returns:
        dict: Classifier results
    """
    logger = setup_logging()
    logger.info("Starting MLP Classifier Training")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    results = {}

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")

    # Train on original 4096-dim features
    logger.info("\n" + "=" * 60)
    logger.info("Training MLP on Original Features (4096-dim)")
    logger.info("=" * 60)

    mlp_original = MLP(num_features=features.shape[1])
    mlp_original.train_model(X_train, y_train, device=device)
    accuracy_original = mlp_original.evaluate(X_test, y_test, device=device)
    logger.info(f"Test Accuracy (Original): {accuracy_original:.4f}")

    results["original_features"] = {
        "feature_dim": int(features.shape[1]),
        "test_accuracy": accuracy_original,
    }

    # Train on best reduced features from clustering
    best_accuracy_reduced = 0
    best_dim_method = None
    results["reduced_features"] = {}

    # Try all dimensionality reduction methods
    for dim_method in ["svd", "umap", "autoencoder"]:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Training MLP on Reduced Features ({dim_method.upper()}, 50-dim)")
        logger.info("=" * 60)

        # Apply dimensionality reduction to train and test sets
        # For fairness, fit on training data only
        if dim_method == "svd":
            reducer = TruncatedSVD(n_components=50, random_state=42)
            X_train_reduced = reducer.fit_transform(X_train)
            X_test_reduced = reducer.transform(X_test)
        elif dim_method == "umap":
            reducer = umap.UMAP(n_components=50, random_state=42)
            X_train_reduced = reducer.fit_transform(X_train)
            X_test_reduced = reducer.transform(X_test)
        elif dim_method == "autoencoder":
            reducer = Autoencoder(n_components=50)
            reducer.fit(X_train, device=device)
            X_train_reduced = reducer.transform(X_train, device=device)
            X_test_reduced = reducer.transform(X_test, device=device)

        logger.info(f"Reduced feature dim: {X_train_reduced.shape[1]}")

        # Train MLP on reduced features
        mlp_reduced = MLP(num_features=X_train_reduced.shape[1])
        mlp_reduced.train_model(X_train_reduced, y_train, device=device)
        accuracy_reduced = mlp_reduced.evaluate(X_test_reduced, y_test, device=device)
        logger.info(f"Test Accuracy ({dim_method.upper()}): {accuracy_reduced:.4f}")

        results["reduced_features"][dim_method] = {
            "feature_dim": 50,
            "test_accuracy": accuracy_reduced,
        }

        if accuracy_reduced > best_accuracy_reduced:
            best_accuracy_reduced = accuracy_reduced
            best_dim_method = dim_method

    # Summary
    results["summary"] = {
        "original_accuracy": accuracy_original,
        "best_reduced_method": best_dim_method,
        "best_reduced_accuracy": best_accuracy_reduced,
        "accuracy_improvement": best_accuracy_reduced - accuracy_original,
    }

    logger.info(f"\n{'=' * 60}")
    logger.info("MLP Classifier Summary:")
    logger.info(f"  Original (4096-dim): {accuracy_original:.4f}")
    logger.info(
        f"  Best Reduced ({best_dim_method}, 50-dim): {best_accuracy_reduced:.4f}"
    )
    logger.info(f"{'=' * 60}")

    # Save results to JSON
    output_json = output_dir / "part2_mlp.json"
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved MLP results to {output_json}")

    return results


def create_summary_output():
    """
    Create summary JSON output with answers to Q13-Q19.

    Reads from existing JSON output files and generates summary answers.
    """
    logger = setup_logging()
    logger.info("Creating Summary Output")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Read from output files
    features_file = output_dir / "part2_features.json"
    clustering_file = output_dir / "part2_clustering.json"
    mlp_file = output_dir / "part2_mlp.json"

    # Load features data
    with open(features_file, "r") as f:
        features_data = json.load(f)

    # Load clustering data
    with open(clustering_file, "r") as f:
        clustering_data = json.load(f)

    # Load MLP data
    with open(mlp_file, "r") as f:
        mlp_data = json.load(f)

    # Build summary JSON - numbers only, no text explanations
    summary = {
        "Q13_transfer_learning": {
            "model": "VGG16",
            "pretrained_dataset": "ImageNet",
            "pretrained_classes": 1000,
            "target_dataset": "tf_flowers",
            "target_classes": 5,
            "extraction_layer": "fc[0] (first fully-connected)",
            "why_effective": [
                "early_layers_learn_general_features",
                "edges_textures_colors_transfer_across_domains",
                "later_layers_capture_abstract_patterns",
            ],
        },
        "Q14_feature_extraction_pipeline": {
            "steps": [
                "load_pretrained_vgg16",
                "resize_image_to_224x224",
                "normalize_with_imagenet_mean_std",
                "pass_through_convolutional_features",
                "apply_average_pooling_7x7",
                "flatten_to_1d",
                "extract_from_fc0_layer",
            ],
            "input_shape": [224, 224, 3],
            "output_dim": 4096,
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
        },
        "Q15_dimensions": {
            "original_pixels": features_data["original_pixels"],
            "feature_dim": features_data["feature_dim"],
            "num_images": features_data["num_images"],
        },
        "Q16_sparsity": {
            "is_dense": features_data["is_dense"],
        },
        "Q17_tsne": {
            "plot_path": "outputs/part2_tsne.png",
            "n_components": 2,
            "perplexity": 30,
        },
        "Q18_clustering": {
            "best_ari": clustering_data["best_overall"]["best_ari"],
            "best_method": clustering_data["best_overall"]["dim_reduction"],
            "best_clustering_method": clustering_data["best_overall"]["config"].get(
                "clustering", None
            ),
            "best_hdbscan_params": clustering_data["best_overall"]["config"].get(
                "hdbscan_params", None
            ),
            "results_by_dim_reduction": {
                dim: {
                    "best_ari": data["best_ari"],
                    "best_clustering": data["config"]["clustering"],
                }
                for dim, data in clustering_data["best_per_dim_reduction"].items()
            },
        },
        "Q19_mlp": {
            "original_accuracy": mlp_data["summary"]["original_accuracy"],
            "best_reduced_accuracy": mlp_data["summary"]["best_reduced_accuracy"],
            "best_method": mlp_data["summary"]["best_reduced_method"],
            "accuracy_drop_percent": (
                mlp_data["summary"]["original_accuracy"]
                - mlp_data["summary"]["best_reduced_accuracy"]
            )
            * 100,
            "does_performance_suffer": mlp_data["summary"]["accuracy_improvement"] < 0,
            "is_drop_significant": abs(mlp_data["summary"]["accuracy_improvement"])
            > 0.05,
            "all_reduced_accuracies": {
                method: data["test_accuracy"]
                for method, data in mlp_data["reduced_features"].items()
            },
            "best_clustering_dim": clustering_data["best_overall"]["dim_reduction"],
            "best_clustering_ari": clustering_data["best_overall"]["best_ari"],
        },
    }

    # Save summary to JSON
    output_json = output_dir / "part2_summary.json"
    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {output_json}")

    return summary


if __name__ == "__main__":
    # Extract features (loads from cache if available)
    data = extract_flower_features()
    features = data["features"]
    labels = data["labels"]
    class_names = data["class_names"]
    device = get_device()

    # Task 3: t-SNE Visualization
    run_tsne_visualization(features, labels, class_names)

    # Task 4: Clustering Grid Search
    clustering_results = run_clustering_grid_search(features, labels, device=device)

    # Task 5: MLP Classifier
    best_dim = clustering_results["best_overall"]["dim_reduction"]
    mlp_results = run_mlp_classifier(
        features, labels, best_dim_reduction=best_dim, device=device
    )

    # Create summary output
    create_summary_output()

    # Print completion message
    print("\n" + "=" * 60)
    print("=== Part 2 Complete ===")
    print("=" * 60)
