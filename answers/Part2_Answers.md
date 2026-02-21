# Part 2 - Deep Learning and Clustering of Image Data

## Question 13

If the VGG network is trained on a dataset with totally different classes as targets, why would one expect its features to have discriminative power for a custom dataset?

**Answer:**

VGG16 learns visual patterns, not ImageNet-specific categories. Early layers detect edges, textures, and colors. Middle layers learn shapes. Deep layers capture abstract concepts. Because ImageNet has 1000 diverse classes, the network learns general visual primitives that transfer to new domains.

The 4096-d first FC layer encodes high-level content that distinguishes flower types even though no flowers appeared in training. VGG16 learns how to see, not what to see.

---

## Question 14

Explain how the helper code performs feature extraction.

**Answer:**

1. Load VGG16 from torch.hub with pretrained ImageNet weights
2. Preprocess: resize to 224×224, center crop, normalize with ImageNet mean/std
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

Compression: 150,528 → 4,096 (36.8x reduction)

---

## Question 16

Are extracted features dense or sparse? Compare with TF-IDF.

**Answer:**

VGG16 features are dense.

TF-IDF vectors are mostly zeros. A 50-word review might touch 30 terms out of 25,000, everything else is 0, giving 99%+ sparsity.

VGG16 works differently. Each of the 4096 neurons receives weighted input from all 25,088 pooled features, so every dimension is non-zero. All dimensions capture information about the entire image.

---

## Question 17

Map features to 2D with t-SNE and plot. Color by ground-truth labels.

**Answer:**

File: [t-SNE plot](outputs/Q17_tsne.png)

Settings: 2 components, perplexity=30, 3,670 points

Some flower classes form distinct blobs—tulips and sunflowers cluster pretty cleanly. Others overlap, like roses and dandelions, which makes sense since they have similar petal structures. A few classes split into multiple small clusters, probably because of variation within each class (different colors, angles, or developmental stages).

The five classes don't separate perfectly, which matches what we'll see later with clustering: the features capture class structure well enough, but aren't cleanly separable enough for any clustering method to nail it. Still, for a network trained on ImageNet that's never seen flowers, the class structure that emerges is pretty good.

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

What stands out:

- UMAP + HDBSCAN performs best, with 2.6x better ARI than anything else. UMAP's non-linear embedding works well with density-based clustering.
- Linear methods (SVD, no reduction) struggle, suggesting the flower feature manifold has non-linear structure.
- Conservative HDBSCAN settings help: larger min_cluster_size and min_samples prevent the algorithm from finding 108 tiny clusters and instead settle on 10 meaningful ones.
- The autoencoder performs similar to raw features, so it's not adding much beyond what's already there.

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

SVD actually beats the original features—91.14% vs 91.01%, a 0.13% gain while cutting dimensions by 98.8%. UMAP loses 10.2 points. Autoencoder sits in the middle at 88.56%.

Comparing with clustering:

| Method | Clustering ARI | Classification Accuracy |
| ------ | -------------- | ----------------------- |
| SVD    | 0.195 (3rd)    | 91.14% (1st)            |
| UMAP   | 0.564 (1st)    | 80.79% (4th)            |

UMAP excels at clustering but struggles with classification. SVD shows the opposite trend.

UMAP preserves local neighborhoods (good for density-based clustering) but distorts global structure (bad for linear classifiers). SVD preserves global variance (good for classification) but doesn't build a cluster-friendly manifold.

Use SVD for classification accuracy, UMAP for clustering.
