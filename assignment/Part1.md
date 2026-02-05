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

| Module Alternatives          | Default Hyperparameters                                               |
| ---------------------------- | --------------------------------------------------------------------- |
| **Dimensionality Reduction** |                                                                       |
| None                         | N/A                                                                   |
| SVD                          | r = 50                                                                |
| UMAP                         | n_components = 50                                                     |
| Autoencoder                  | latent_dim = 50                                                       |
| **Clustering**               |                                                                       |
| K-Means                      | k = 2 (Task 1), k = 5 (Task 3 themes)                                 |
| Agglomerative                | n_clusters = 2 (Task 1), n_clusters = 5 (Task 3 themes)               |
| HDBSCAN                      | min_cluster_size = 2/5 (can experiment here a bit if noise dominates) |

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

## Task 1 - Unsupervised Review Length Discovery

We begin with a warm-up unsupervised task designed to build intuition about representations, geometry, and clustering before tackling semantic concepts.

In this task, the goal is to answer the question:

"Can we discover review length structure from our textual representations"

Although review length is not a semantic label, it induces strong and interpretable structure in common text representations. This makes it an ideal first unsupervised learning problem.

### Task 1.1 Defining pseudo-labels (for evaluation only)

For each review, define its length as the number of tokens (or words) in the review text.

To simplify the task and create a clear separation:

- Reviews in the top 25% (≥ q75) by length are labeled as Long,
- Reviews in the bottom 25% (≤ q25) by length are labeled as Short,
- Reviews in the middle 50% are discarded for this task.

These labels are used only for evaluation. During clustering, review length labels must be treated as unknown.

**QUESTION 1:** Report the number of reviews retained after filtering, and the average length (in tokens) of Short and Long reviews.

### Task 1.2 Representations

1. **TF-IDF representation.** Construct TF-IDF features using min_df=3 and English stopwords. Use unigrams only.
2. **MiniLM embeddings.** Compute dense sentence embeddings using sentence-transformers/all-MiniLM-L6-v2.

**QUESTION 2:** Report the dimensions of the TF-IDF matrix and the MiniLM embedding matrix. Briefly explain why TF-IDF is sparse while MiniLM embeddings are dense.

### Task 1.3 Clustering pipelines

For each representation, run clustering pipelines with:

- Dimensionality reduction: None, SVD(50), UMAP(50),
- Clustering: K-Means (k = 2), Agglomerative (n_clusters = 2), optionally HDBSCAN

**Note:** You might notice when no dimensionality reduction method is used for TF-IDF embeddings, some of the clustering methods might crash; if it does, mention the reason in the report and skip them. Agglomerative requires dense input. If using TF-IDF, consider applying SVD first or convert to dense only if feasible. Applying UMAP directly on TF-IDF might be very slow, consider using SVD before.

**QUESTION 3:** For each pipeline, report the following clustering agreement metrics with respect to the ground-truth length labels: homogeneity, completeness, v-measure, ARI, AMI. Summarize results in a table and identify the best-performing pipeline.

### Task 1.4 Interpretation

**QUESTION 4:** Compare TF-IDF and MiniLM performance on this task. Which representation separates Short vs Long reviews more cleanly, and why?

**QUESTION 5:** Plots and Visualization: Select the best-performing configuration for TF–IDF and MiniLM based on clustering performance. For each representation:

- Reduce the embeddings to two dimensions using PCA (sklearn.decomposition.PCA).
- Create a split visualization with:
  – One plot colored by the ground-truth length label.
  – One plot colored by the cluster assignments obtained using your best clustering method.

The resulting plots should enable a direct visual comparison between the true labels and the discovered clusters for both TF–IDF and MiniLM representations, highlighting how well each representation supports unsupervised separation.

**Note.** You should observe that review length can be separated somehow well by certain settings of unsupervised clustering. This is intentional.

Review length induces strong, global structure in representation space, making it a good first sanity-check problem for unsupervised learning pipelines.

Another intuitive tasks, as you may think of, is the sentiment analysis (positive comments vs. negative comments). If you are interested, feel free to try on it and report what you find (it is definitely much more difficult!)

## Task 2 - Unsupervised Game Similarity & Genre Structure

In Task 2, we are now trying to answer a different question: "Can we group games into meaningful clusters based on what players praise?" In other words, instead of clustering reviews, we will cluster games.

### Task 2.1 Construct one representation per game (positive reviews only)

Use only positive reviews (recommend=True). For each game, construct a single vector representation:

- **TF-IDF game vector (baseline):** concatenate all positive reviews for the game into one document, then compute TF-IDF for all games.
- **MiniLM game vector:** compute MiniLM embeddings for each positive review, then average them to obtain one embedding per game.

**QUESTION 6:** Report the dimensions of the TF-IDF game matrix and the MiniLM game embedding matrix.

### Task 2.2 Cluster games with default pipelines

Run the same module table (one default hyperparameter each), but now clustering is performed on game vectors rather than independent review vectors:

- Dimensionality reduction: None, SVD(50), UMAP(50), Autoencoder(50).
- Clustering: K-Means (k = 5), Agglomerative (n_clusters = 5), HDBSCAN.

**Note on UMAP:** UMAP might not work well on high-dimensional inputs. Thus, consider first applying SVD to reduce dimension to 200, then use UMAP on it.

**QUESTION 7:** For each pipeline, report a summary table that includes:

- number of clusters found (for HDBSCAN, also report the fraction of games labeled as noise -1),
- cluster sizes,
- for each cluster: top 3 most common genres (by frequency across games in that cluster).

### Task 2.3 Multi-genre interpretation

Because games may have multiple genres, do not treat this as a single-label problem. Instead, you will evaluate clusters using genre-overlap analysis:

- For each cluster, compute the distribution over genre labels (multi-label frequency).
- Define **cluster genre purity** as: the fraction of games in the cluster that contain the cluster's most common genre labels.
- Define **cluster genre entropy** over the genre labels distribution (optional).

**QUESTION 8:** Pick one best pipeline (justify your choice), then report two cluster with high purity:

- top 3 genres with percentages,
- cluster genre purity (as defined above),
- Representative games in the cluster (game name + genres).

Provide a short interpretation: what type of games does these clusters represent?

## Task 3 - Held-Out Game Profiling and Theme Discovery (Clustering + LLM)

Finally, you will produce a product report for a held-out game. Your goal is to quickly answer:

- What genre does it most resemble (estimated)?
- What are the main complaint themes and praise themes?

### Task 3.1 Genre estimation via nearest game cluster

Use your best Task 2 pipeline to estimate the held-out game's genre profile:

1. Compute the held-out game's game vector using its positive reviews:
   • **TF-IDF:** concatenate positive reviews, transform using your fitted TF-IDF vectorizer.
   • **MiniLM:** average MiniLM embeddings of positive reviews.
2. Assign the held-out game to the closest cluster (e.g., nearest centroid for K-Means, nearest cluster medoid, or nearest cluster by average cosine distance).
3. Report the cluster's top genres as the held-out game's estimated genre distribution.

**QUESTION 9:** Report: (i) the assigned cluster ID, (ii) the top 3 genres of that cluster, and (iii) 3 representative games from that cluster. Briefly justify why this constitutes a genre estimate in a multi-genre world.

### Task 3.2 Theme clustering (positive and negative)

Now perform theme discovery separately on:

- negative reviews (complaints),
- positive reviews (praises).

For each subset, run the module table again with default hyperparameters, but now use:

- K-Means(k = 5) and Agglomerative (n_clusters = 5) for a fixed number of themes,
- HDBSCAN for a variable number of themes (explain handling of noise points).

For each discovered cluster, provide:

- top TF-IDF terms (cluster-level),
- 1–2 exemplar reviews (closest to cluster centroid / medoid),
- a short cluster label (3–6 words).

**QUESTION 10:** For negative reviews: report 3–5 clusters with (i) top terms and (ii) exemplar reviews, and assign a short label to each complaint cluster.

**QUESTION 11:** For positive reviews: repeat the same analysis and label 3–5 praise clusters.

### LLM labeling

You are now using an LLM to help assign cluster labels (For each cluster, you feed the LLM with examples and top terms, and let LLM think what is cluster of praise / complaints is about). You need to include:

- your prompt template,
- what evidence you provided (top terms + exemplars),

**QUESTION 12:** Include your prompting strategy and 3 example of LLM-generated labels for clusters. You can use the same helper code from Project 1 for Qwen model (Qwen/Qwen3-4B-Instruct-2507).
