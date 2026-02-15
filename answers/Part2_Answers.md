# Part 2 - Deep Learning and Clustering of Image Data

---

## Question 13

In a brief paragraph discuss: If the VGG network is trained on a dataset with perhaps totally different classes as targets, why would one expect the features derived from such a network to have discriminative power for a custom dataset?

#### Answer

VGG16 works on custom datasets because it learns a hierarchy of visual patterns, not just ImageNet-specific categories. Early convolutional layers pick up edges, textures, and colors—things that appear in every image regardless of what it depicts. Middle layers learn shapes and spatial patterns. Deeper layers capture abstract concepts. Since ImageNet spans 1000 diverse classes (animals, plants, vehicles, scenes), the network ends up learning general visual primitives that transfer to domains it never saw during training. The 4096-dimensional first FC layer encodes high-level semantic content that distinguishes between flower types even though no flowers appeared in the original training data. VGG16 learns how to see, not what to see.

---

## Question 14

In a brief paragraph explain how the helper code base is performing feature extraction.

#### Answer

The pipeline works like this:

1. Load VGG16 from torch.hub with pretrained ImageNet weights
2. Preprocess each image: resize to 224×224, center crop, convert to tensor, normalize using ImageNet mean/std values
3. Run the image through all 13 convolutional layers with pooling
4. Apply 7×7 adaptive average pooling to collapse spatial dimensions
5. Flatten from 512×7×7 down to a 25,088-element vector
6. Pass through only the first fully-connected layer, yielding a 4096-dimensional embedding

The code throws away the remaining classifier layers (fc[1] and fc[2]) and just keeps the fc[0] output as the final feature vector.

---

## Question 15

How many pixels are there in the original images? How many features does the VGG network extract per image; i.e what is the dimension of each feature vector for an image sample?

#### Answer

- Original images: 224 × 224 × 3 = 150,528 pixels
- VGG16 feature dimension: 4,096 features per image
- Dataset size: 3,670 images total

The network compresses each image from 150,528 pixel values down to 4,096 semantic features—a 36.8× reduction.

---

## Question 16

Are the extracted features dense or sparse? (Compare with sparse TF-IDF features in text.)

#### Answer

VGG16 features are dense.

TF-IDF vectors are mostly zeros because any document only uses a tiny fraction of the total vocabulary. A 50-word review might touch 30 unique terms out of 25,000—everything else gets a 0, creating 99%+ sparsity.

VGG16 embeddings work differently. Each of the 4096 neurons in fc[0] receives weighted input from all 25,088 pooled features (512×7×7), so every output dimension has a non-zero activation value. Rather than encoding word presence/absence, the dense representation spreads semantic information about the entire image across all dimensions.

---

## Question 17

In order to inspect the high-dimensional features, t-SNE is a popular off-the-shelf choice for visualizing Vision features. Map the features you have extracted onto 2 dimensions with t-SNE. Then plot the mapped feature vectors along x and y axes. Color-code the data points with ground-truth labels. Describe your observation.

#### Answer

Visualization file: [t-SNE plot](outputs/part2_tsne.png)

t-SNE settings: 2 components, perplexity=30, 3,670 points projected to 2D

What I see:

Some flower classes form distinct blobs—tulips and sunflowers cluster pretty cleanly. Others overlap, like roses and dandelions, which makes sense given their similar petal structures. A few classes split into multiple small clusters, probably reflecting intra-class variation (different colors, angles, or developmental stages).

The five classes don't separate perfectly, which matches what we'll see later with clustering: the features are discriminative enough to capture class structure, but not so cleanly separable that any clustering method will nail it. Still, for a network trained on ImageNet having never seen flowers, the class structure that emerges is pretty good.

---

## Question 18

Report the best result (in terms of adjusted rand index) within the table below. For HDBSCAN, introduce a conservative parameter grid over min_cluster_size and min_samples.

#### Answer

Best overall: ARI = 0.5635 (UMAP + HDBSCAN)

| Pipeline | Dim Reduction | Clustering | Parameters | Homogeneity | Completeness | V-Measure | ARI | AMI | Clusters | Noise |
|----------|---------------|------------|------------|-------------|--------------|-----------|-----|-----|----------|-------|
| Best | UMAP | HDBSCAN | min_cluster_size=50, min_samples=10 | 0.6762 | 0.5949 | 0.6330 | 0.5635 | 0.6315 | 10 | 934 |

Best ARI by dimensionality reduction method:

| Dim Reduction | Best Clustering | Best ARI | Key Parameters |
|---------------|-----------------|----------|----------------|
| None (4096-dim) | Agglomerative | 0.2184 | n_clusters=5 |
| SVD (50-dim) | K-Means | 0.1947 | n_clusters=5 |
| UMAP (50-dim) | HDBSCAN | 0.5635 | min_cluster_size=50, min_samples=10 |
| Autoencoder (50-dim) | Agglomerative | 0.2336 | n_clusters=5 |

HDBSCAN parameter grid (on UMAP-reduced features):

| min_cluster_size | min_samples | Clusters Found | Noise Points | ARI |
|------------------|-------------|----------------|--------------|-----|
| 5 | 3 | 108 | 1,723 | 0.0971 |
| 10 | 3 | 55 | 1,673 | 0.1419 |
| 20 | 5 | 21 | 1,137 | 0.4080 |
| 50 | 10 | 10 | 934 | 0.5635 |

What stands out:

- UMAP + HDBSCAN wins by a lot—2.6× better ARI than anything else. UMAP's non-linear embedding plays well with density-based clustering.
- Linear methods (SVD, no reduction) struggle, suggesting the flower feature manifold has non-linear structure.
- Conservative HDBSCAN settings help: larger min_cluster_size and min_samples prevent the algorithm from finding 108 tiny clusters and instead settle on 10 meaningful ones.
- The autoencoder lands close to raw features, so it's not adding much beyond what's already there.

---

## Question 19

Report the test accuracy of the MLP classifier on the original VGG features. Report the same when using the reduced-dimension features (you have freedom in choosing the dimensionality reduction algorithm and its parameters). Does the performance of the model suffer with the reduced-dimension representations? Is it significant? Does the success in classification make sense in the context of the clustering results obtained for the same features in Question 18.

#### Answer

MLP test accuracy:

| Feature Type | Dimension | Test Accuracy |
|--------------|-----------|---------------|
| Original VGG16 | 4,096 | 91.69% |
| SVD (50-dim) | 50 | 90.74% |
| UMAP (50-dim) | 50 | 83.79% |
| Autoencoder (50-dim) | 50 | 88.42% |

Does performance suffer with reduced dimensions?

Yes, all three reduced methods drop below the original 91.69%. SVD holds up best at 90.74%, autoencoder comes in at 88.42%, and UMAP falls to 83.79%.

Is the drop significant?

For SVD, not really. We dropped from 4096 to 50 dimensions (a 98.8% reduction) and only lost 0.95 percentage points. That's surprisingly robust. The UMAP drop of 7.9 points is more notable—compressing that aggressively while preserving local neighborhoods seems to sacrifice classification performance.

Does this align with the clustering results?

Here's where it gets interesting:

| Method | Clustering ARI | Classification Accuracy |
|--------|----------------|------------------------|
| SVD | 0.1947 (3rd place) | 90.74% (1st for reduced) |
| UMAP | 0.5635 (1st place) | 83.79% (3rd for reduced) |

UMAP crushed clustering but struggled with classification. SVD did the opposite—mediocre clustering, excellent classification.

The reason: UMAP optimizes for local neighborhood preservation, which helps density-based clustering find coherent clusters. But it can warp global structure in ways that hurt linear classifiers. SVD preserves global variance, which maintains the class boundaries a neural network classifier relies on, but doesn't create the kind of manifold structure that makes clusters pop out.

Both results make sense. The features are clearly discriminative (91.69% accuracy on a 5-class problem with 20% random baseline), but the choice of dimensionality reduction should match your goal. SVD for classification, UMAP for clustering.
