# Large-Scale Data Mining: Models and Algorithms ECE 219 Winter 2026
Prof. Vwani Roychowdhury
UCLA, Department of ECE

## Project 2: Clustering & Unsupervised Learning and Intro to Multi-modal models
Due February 20, 2026 by 11:59 pm

## Introduction

Machine learning algorithms are applied to a wide variety of data, including text and images. Before applying these algorithms, one needs to convert the raw data into feature representations that are suitable for downstream algorithms. In project 1, we studied feature extraction from text data, and the downstream task of classification. We also learned that reducing the dimension of the extracted features often helps with a downstream task.

In this project, we explore the concepts of feature extraction and clustering together. In an ideal world, all we need are data points – encoded using certain features– and AI should be able to find what is important to learn, or more specifically, determine what are the underlying modes or categories in the dataset. This is the ultimate goal of General AI: the machine is able to bootstrap a knowledge base, acting as its own teacher and interacting with the outside world to explore to be able to operate autonomously in an environment.

We first explore this field of unsupervised learning using textual data, which is a continuation of concepts learned in Project 1. We ask if a combination of feature engineering and clustering techniques can automatically separate a document set into groups that match known labels.

Next we focus on a new type of data, i.e. images. Specifically, we first explore how to use "deep learning" or "deep neural networks (DNNs)" to obtain image features. Large neural networks have been trained on huge labeled image datasets to recognize objects of different types from images. For example, networks trained on the Imagenet dataset can classify more than one thousand different categories of objects. Such networks can be viewed as comprising two parts: the first part maps a given RGB image into a feature vector using convolutional filters, and the second part then classifies this feature vector into an appropriate category, using a fully-connected multi-layered neural network (we will study such NNs in a later lecture). Such pre-trained networks could be considered as experienced agents that have learned to discover features that are salient for image understanding. Can one use the experience of such pre-trained agents in understanding new images that the machine has never seen before? It is akin to asking a human expert on forensics to explore a new crime scene. One would expect such an expert to be able to transfer their domain knowledge into a new scenario. In a similar vein, can a pre-trained network for image understanding be used for transfer learning? One could use the output of the network in the last few layers as expert features. Then, given a multi-modal dataset –consisting of images from categories that the DNN was not trained for– one can use feature engineering (such as dimensionality reduction) and clustering algorithms to automatically extract unlabeled categories from such expert features.

For both the text and image data, one can use a common set of multiple evaluation metrics to compare the groups extracted by the unsupervised learning algorithms to the corresponding ground truth human labels.

## Clustering Methods

Clustering is the task of grouping a dataset in such a way that data points in the same group (called a cluster) are more similar (in some sense) to each other than to those in other groups (clusters). Thus, there is an inherent notion of a metric that is used to compute similarity among data points, and different clustering algorithms differ in the type of similarity measure they use, e.g., Euclidean vs Riemannian geometry. Clustering algorithms are considered "unsupervised learning", i.e. they do not require labels during training. In principle, if two categories of objects or concepts are distinct from some perspective (e.g. visual or functional), then data points from these two categories – when properly coded in a feature space and augmented with an associated distance metric – should form distinct clusters. Thus, if one can perform perfect clustering then one can discover and obtain computational characterizations of categories without any labeling. In practice, however, finding such optimal choices of features and metrics has proven to be a computationally intractable task, and any clustering result needs to be validated against tasks for which one can measure performance. Thus, we use labeled datasets in this project, which allows us to evaluate the learned clusters by comparing them with ground truth labels.

Below, we summarize several clustering algorithms:

**K-means:** K-means clustering is a simple and popular clustering algorithm. Given a set of data points $\{x_1, \ldots, x_N\}$ in multidimensional space, and a hyperparameter $K$ denoting the number of clusters, the algorithm finds the $K$ cluster centers such that each data point belongs to exactly one cluster. This cluster membership is found by minimizing the sum of the squares of the distances between each data point and the center of the cluster it belongs to. If we define $\mu_k$ to be the "center" of the $k$th cluster, and

$$r_{nk} = \begin{cases} 1, & \text{if } x_n \text{ is assigned to cluster } k \\ 0, & \text{otherwise} \end{cases}, \quad n = 1, \ldots, N \quad k = 1, \ldots, K$$

Then our goal is to find $r_{nk}$'s and $\mu_k$'s that minimize $J = \sum_{n=1}^{N} \sum_{k=1}^{K} r_{nk} \|x_n - \mu_k\|^2$. The approach of K-means algorithm is to repeatedly perform the following two steps until convergence:

1. (Re)assign each data point to the cluster whose center is nearest to the data point.
2. (Re)calculate the position of the centers of the clusters: setting the center of the cluster to the mean of the data points that are currently within the cluster.

The center positions may be initialized randomly.

**Hierarchical Clustering:** Hierarchical clustering is a general family of clustering algorithms that builds nested clusters by merging or splitting them successively. This hierarchy of clusters is represented as a tree (or dendrogram). A flat clustering result is obtained by cutting the dendrogram at a level that yields a desired number of clusters.

**DBSCAN:** DBSCAN or Density-Based Spatial Clustering of Applications with Noise finds core samples of high density and expands clusters from them. It is a density-based clustering non-parametric algorithm: Given a set of points, the algorithm groups together points that are closely packed together (points with many nearby neighbors), marking as outliers points that lie alone in low-density regions (whose nearest neighbors are too far away).

**HDBSCAN:** HDBSCAN extends DBSCAN by converting it into a hierarchical clustering algorithm, and then using an empirical technique to extract a flat clustering based on the stability of clusters (similar to the elbow method in k-Means). The resulting algorithm gets rid of the hyperparameter "epsilon", which is necessary in DBSCAN (see here for more on that).

## Common Clustering Evaluation Metrics

In order to evaluate a clustering pipeline, one can use the ground-truth class labels and compare them with the cluster labels. This analysis determines the quality of the clustering algorithm in recovering the ground-truth underlying labels. It also indicates if the adopted feature extraction and dimensionality reduction methods retain enough information about the ground-truth classes. Below we provide several evaluation metrics available in sklearn.metrics. Note that for the clustering sub-tasks, you do not need to separate your data to training and test sets.

**Homogeneity** is a measure of how "pure" the clusters are. If each cluster contains only data points from a single class, the homogeneity is satisfied.

**Completeness** indicates how much of the data points of a class are assigned to the same cluster.

**V-measure** is the harmonic average of homogeneity score and completeness score.

**Adjusted Rand Index** is similar to accuracy, which computes similarity between the clustering labels and ground truth labels. This method counts all pairs of points that both fall either in the same cluster and the same class or in different clusters and different classes.

**Adjusted mutual information score** measures the mutual information between the cluster label distribution and the ground truth label distributions.

## Dimensionality Reduction Methods

In project 1, we studied SVD/PCA and NMF as linear dimensionality reduction techniques. Here, we consider some additional non-linear methods.

**Uniform Manifold Approximation and Projection (UMAP):** The UMAP algorithm constructs a graph-based representation of the high-dimensional data manifold, and learns a low-dimensional representation space based on the relative inter-point distances. UMAP allows more choices of distance metrics besides Euclidean distance. In particular, we are interested in "cosine distance" for text data, because as we shall see it bypasses the magnitude of the vectors, meaning that the length of the documents does not affect the distance metric.

**Autoencoders:** An autoencoder¹ is a special type of neural network that is trained to copy its input to its output. For example, given an image of a handwritten digit, an autoencoder first encodes the image into a lower dimensional latent representation, then decodes the latent representation back to an image. An autoencoder learns to compress the data while minimizing the reconstruction error. Further details can be found in chapter 14 of [4].

¹ also known as "auto-associative networks" in older jargon

## Part 1 - Steam Reviews: Product Analytics with Representations and Clustering

Imagine you are a product engineer (or data scientist) on a game platform team. Your job is to help the company answer questions like:

- Are players actually enjoying this game? How does sentiment compare across games?
- What are players complaining about most? (crashes? balance? monetization? controls?)
- What makes a game feel like a particular genre? Can we infer genre from what players say?
- For a new or held-out game, can we profile it quickly? (positivity ratio, likely genre, top issues, top praises)

We will the following toolkit to solve these tasks:

- text representations (sparse vs dense),
- dimensionality reduction (to make patterns easier to discover),
- clustering (to discover themes without manual labeling),
- and evaluation using known signals (ratings, thumbs up/down) when available.

In this first part of the project, You will build a system that can:

1. discover clusters in the review lengths (Task 1),
2. infer game genres from positive player feedback (Task 2),
3. and generate a concise product report for a held-out game (Task 3).

### Dataset

You are provided with a curated subset of Steam reviews (Download here):

- **Main dataset (CSV):** reviews from the top 200 games ranked by total reviews (positive+negative) in the metadata. For each game, we keep:
  – 100 English Recommended reviews, and
  – 100 English Not Recommended reviews,
  selected by highest helpfulness (upvotes). This ensures the reviews are typically informative rather than one-word memes.

- **Held-out dataset (CSV):** A secret game's reviews are provided separately and will be used only in Task 3.

Each review row includes:

- **user:** User Name
- **playtime:** Number of hours this User plays this game.
- **post date:** When is the review posted.
- **helpfulness:** Number of upvotes this review received.
- **review text:** The review itself.
- **recommend:** Whether the user recommended the game or not. True or False
- **early access review:** Whether this review is for a game during early access stage, either empty or early access.
- **appid:** Unique id of the game.
- **game name:** Game name.
- **release date:** Release date of the game.
- **genres:** Genres of the game, either single a string separated by comma.

**Important note on genres:** Games often have multiple genres. In this project, genres should be treated as multi-label metadata.

### Methods and Modules

Your system will be built from modular choices:

- **Representations:** TF-IDF, MiniLM embeddings (sentence-transformers/all-MiniLM-L6-v2).
- **Dimensionality Reduction:** None, SVD, UMAP, Autoencoder.
- **Clustering:** K-Means, Agglomerative, HDBSCAN.

**Default setting policy:** To keep the project lightweight and focus on interpretation, you only need to run one default hyperparameter choice per method (unless a question explicitly requests a sweep).

| Module Alternatives | Default Hyperparameters |
|---------------------|-------------------------|
| **Dimensionality Reduction** | |
| None | N/A |
| SVD | r = 50 |
| UMAP | n_components = 50 |
| Autoencoder | latent_dim = 50 |
| **Clustering** | |
| K-Means | k = 2 (Task 1), k = 5 (Task 3 themes) |
| Agglomerative | n_clusters = 2 (Task 1), n_clusters = 5 (Task 3 themes) |
| HDBSCAN | min_cluster_size = 2/5 (can experiment here a bit if noise dominates) |

**Compute note (Colab):** Dense embeddings + UMAP/Autoencoder can be memory intensive.

**Note on TF-IDF:** For TF-IDF, due to its large and sparse representations, running certain methods can be really slow. Thus, you can skip the None, UMAP and Autoencoder for it.

**Note on HDBSCAN:** You cannot specify the designed number of cluster you want for HDBSCAN, instead, you will specify the minimum cluster size for each cluster it finds. For tasks that we want a specific number of cluster, consider finding the largest two clusters, getting it's centroids, and assign the rest points to them)

**Note on UMAP:** UMAP might not work efficiently on high-dimensional data. Consider first using SVD to reduce raw data to a dimension of 200, then use UMAP.

**Note on Agglomerative:** If Agglomerative method runs really slow, consider using the following setups:
```python
conn = kneighbors_graph(
    Z,
    n_neighbors=k,
    mode="connectivity",
    include_self=False
)
model = AgglomerativeClustering(
    n_clusters=2,
    linkage="ward",
    connectivity=conn
)
```