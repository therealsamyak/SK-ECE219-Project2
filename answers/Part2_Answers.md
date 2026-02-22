# Part 2 - Deep Learning and Clustering of Image Data

## Question 13

If the VGG network is trained on a dataset with totally different classes as targets, why would one expect its features to have discriminative power for a custom dataset?

**Answer:**

VGG16 learns visual patterns directly, and is not constrained to the specific classes it is trained on. Early layers learn very generic features (ex. edges, corners, color contrast, simple texture, etc.). Middle layers learn reusable patterns. The final layers are task-specific. Assuming the training data is very large and diverse, the network learns general visual primitives that transfer to new unseen tasks well.

---

## Question 14

Explain how the helper code performs feature extraction.

**Answer:**

1. Load VGG16 from torch.hub with pretrained ImageNet weights
2. Resize to 224×224, center crop, normalize with ImageNet mean/std
3. Run through 13 convolutional layers with pooling
4. Apply 7×7 adaptive average pooling
5. Flatten from 512×7×7 to 25,088 elements
6. Pass through first FC layer only → 4096-d embedding

The code discards fc[1] and fc[2], keeping only fc[0] output.

---

## Question 15

How many pixels in original images? How many features does VGG extract per image?

**Answer:**

- Original: 224 × 224 × 3 = 150,528 pixels
- VGG16 output: 4,096 features
- Dataset: 3,670 images

Compression: 150,528 → 4,096, which is ~36.8x reduction.

---

## Question 16

Are extracted features dense or sparse? Compare with TF-IDF.

**Answer:**

VGG16 features are dense.

TF-IDF vectors are mostly zeros, because most reviews do not contain every single word in the vocabulary. This causes ~99%+ sparsity.

VGG16 works differently. Each of the 4096 neurons receives weighted input from all 25,088 pooled features, so every dimension is non-zero. All dimensions capture information about the entire image.

---

## Question 17

Map features to 2D with t-SNE and plot. Color by ground-truth labels.

**Answer:**

File: [t-SNE plot](outputs/Q17_tsne.png)

_Settings: 2 components, perplexity=30, 3,670 points_

**Spatial Distribution by Quadrant:**

| Quadrant         | Dominant Class            | Count | Key Characteristics                       |
| ---------------- | ------------------------- | ----- | ----------------------------------------- |
| Upper-Right (Q1) | Dandelions (80.5%)        | 815   | Forms tightest, most cohesive cluster     |
| Upper-Left (Q2)  | Daisies (56.3%)           | 996   | Most dispersed, overlaps multiple classes |
| Lower-Left (Q3)  | Sunflowers (56.9%)        | 724   | Overlaps with daisies                     |
| Lower-Right (Q4) | Roses (49%), Tulips (44%) | 1,135 | Heavy overlap - 77% bounding box overlap  |

Roses and tulips cluster together due to similar petal arrangements, while dandelions form a distinct cluster due to their unique spherical shape.

---

## Question 18

Report best ARI result. For HDBSCAN, use parameter grid over min_cluster_size and min_samples.

**Answer:**

**Best:** ARI = 0.5635 (UMAP + HDBSCAN)

| Dim Reduction | Clustering | Parameters                          | ARI    | Clusters | Noise |
| ------------- | ---------- | ----------------------------------- | ------ | -------- | ----- |
| UMAP          | HDBSCAN    | min_cluster_size=50, min_samples=10 | 0.5635 | 10       | 934   |

Best ARI by method:

| Method             | Best Clustering | ARI    |
| ------------------ | --------------- | ------ |
| None (4096-d)      | Agglomerative   | 0.2184 |
| SVD (50-d)         | K-Means         | 0.1947 |
| UMAP (50-d)        | HDBSCAN         | 0.5635 |
| Autoencoder (50-d) | Agglomerative   | 0.2344 |

HDBSCAN grid (on UMAP features):

| min_cluster_size | min_samples | Clusters | Noise | ARI   |
| ---------------- | ----------- | -------- | ----- | ----- |
| 5                | 3           | 108      | 1,723 | 0.097 |
| 10               | 3           | 55       | 1,673 | 0.142 |
| 20               | 5           | 21       | 1,137 | 0.408 |
| 50               | 10          | 10       | 934   | 0.564 |

---

## Question 19

Report MLP test accuracy on original and reduced-dimension features. Does performance suffer? Does this align with clustering results?

**Answer:**

| Feature        | Dimension | Accuracy |
| -------------- | --------- | -------- |
| Original VGG16 | 4,096     | 91.01%   |
| SVD            | 50        | 91.14%   |
| UMAP           | 50        | 80.79%   |
| Autoencoder    | 50        | 88.56%   |

SVD beats the original features (91.14% vs 91.01%, 0.13% gain) while cutting dimensions by 98.8%. UMAP loses 10.2 points. Autoencoder sits in the middle at 88.56%.

Comparing with clustering:

| Method | Clustering ARI | Classification Accuracy |
| ------ | -------------- | ----------------------- |
| SVD    | 0.195 (3rd)    | 91.14% (1st)            |
| UMAP   | 0.564 (1st)    | 80.79% (4th)            |

UMAP excels at clustering but struggles with classification. SVD shows the opposite trend. This is because UMAP preserves local neighborhoods (good for density-based clustering) but distorts global structure (bad for linear classifiers). SVD preserves global variance (good for classification) but doesn't build a cluster-friendly manifold. This experiment shows we should use SVD for classification, and UMAP for clustering.
