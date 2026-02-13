# Part 1 - Steam Reviews: Product Analytics with Representations and Clustering

---

## Task 1 - Unsupervised Review Length Discovery

---

### Question 1

Report the number of reviews retained after filtering, and the average length (in tokens) of Short and Long reviews.

#### Answer

- Total reviews retained: 20,497 reviews (after discarding middle 50%)
- Short reviews (≤ q25 = 11 words): 10,463 reviews, average 6.39 words
- Long reviews (≥ q75 = 179 words): 10,034 reviews, average 493.00 words

---

### Question 2

Report the dimensions of the TF-IDF matrix and the MiniLM embedding matrix. Briefly explain why TF-IDF is sparse while MiniLM embeddings are dense.

#### Answer

- TF-IDF matrix shape: (20,497 samples × 25,085 features), vocabulary size 25,085 terms, sparsity 99.68%
- MiniLM embedding matrix shape: (20,497 samples × 384 dimensions), L2 normalized

**Why TF-IDF is sparse while MiniLM is dense:**

TF-IDF is a bag-of-words representation that scores each vocabulary term. A 50-word review might only use 30 unique terms out of 25,085 total. Unused terms get a score of 0, so the matrix ends up mostly zeros.

MiniLM embeddings come from a neural network that projects the entire sentence into a 384-dimensional latent space. The model processes semantic meaning as a whole rather than counting individual words. All 384 dimensions get non-zero values.

---

### Question 3

For each pipeline, report the following clustering agreement metrics with respect to the ground-truth length labels: homogeneity, completeness, v-measure, ARI, AMI. Summarize results in a table and identify the best-performing pipeline.

#### Answer

**TF-IDF Pipelines (2 total)**

| Pipeline            | Homogeneity | Completeness | V-Measure | ARI    | AMI    | Cluster 0 | Cluster 1 |
| ------------------- | ----------- | ------------ | --------- | ------ | ------ | --------- | --------- |
| SVD + K-Means       | 0.0301      | 0.161        | 0.0507    | 0.0057 | 0.0507 | 19,911    | 586       |
| SVD + Agglomerative | 0.0276      | 0.157        | 0.0470    | 0.0050 | 0.0470 | 19,958    | 539       |

_Note: TF-IDF clustering performs very poorly at separating Short vs Long reviews. UMAP and direct (None) dimensionality reduction were skipped for TF-IDF as they are computationally infeasible on sparse, high-dimensional data._

**MiniLM Pipelines (7 total)**

| Pipeline                | Homogeneity | Completeness | V-Measure | ARI       | AMI       | Clusters Found | Noise  | Cluster Distribution |
| ----------------------- | ----------- | ------------ | --------- | --------- | --------- | -------------- | ------ | -------------------- |
| None + K-Means          | 0.509       | 0.509        | 0.509     | 0.613     | 0.509     | 2              | 0      | 10,632 / 9,865       |
| None + Agglomerative    | 0.543       | 0.547        | 0.545     | 0.643     | 0.545     | 2              | 0      | 11,315 / 9,182       |
| SVD + K-Means           | 0.508       | 0.509        | 0.509     | 0.612     | 0.509     | 2              | 0      | 9,863 / 10,634       |
| **SVD + Agglomerative** | **0.646**   | **0.650**    | **0.648** | **0.742** | **0.648** | 2              | 0      | 11,155 / 9,342       |
| UMAP + K-Means          | 0.0339      | 0.157        | 0.0558    | 0.0075    | 0.0558    | 2              | 0      | 19,791 / 706         |
| UMAP + Agglomerative    | 0.0339      | 0.157        | 0.0558    | 0.0075    | 0.0558    | 2              | 0      | 19,791 / 706         |
| None + HDBSCAN          | 0.221       | 0.0799       | 0.117     | 0.0013    | 0.107     | 554            | 15,681 | Many micro-clusters  |

**Best-performing pipeline:** MiniLM: SVD(50) + Agglomerative Clustering with V-Measure 0.6478, ARI 0.7424, Homogeneity 0.6461, Completeness 0.6496

---

### Question 4

Compare TF-IDF and MiniLM performance on this task. Which representation separates Short vs Long reviews more cleanly, and why?

#### Answer

MiniLM outperforms TF-IDF by a large margin for separating Short vs Long reviews.

**Performance Comparison (best pipeline for each representation):**

| Metric       | TF-IDF (SVD+Agg) | MiniLM (SVD+Agg) | Improvement  |
| ------------ | ---------------- | ---------------- | ------------ |
| V-Measure    | 0.0470           | 0.6478           | 13.8× better |
| ARI          | 0.0050           | 0.7424           | 148× better  |
| Homogeneity  | 0.0276           | 0.6461           | 23× better   |
| Completeness | 0.1575           | 0.6496           | 4.1× better  |

**Why MiniLM separates Short vs Long reviews more cleanly:**

1. **Semantic understanding:** MiniLM captures semantic meaning and sentence structure. Long reviews differ from short ones in semantic complexity, topic depth, and discourse structure. Neural embeddings pick up on these differences naturally.

2. **Document length as a latent signal:** Transformers attend to longer sequences differently, creating distinct internal representations for different-length inputs.

3. **TF-IDF limitations:** TF-IDF only counts word frequencies. It ignores word order, grammar, and semantic relationships. Long and short reviews can share many common words like "game", "play", "good" which makes them look similar in TF-IDF space even when they are structurally different.

**Why UMAP fails for this task:**

UMAP preserves local manifold structure, but Short vs Long is a global, linear signal. Document length affects all dimensions in similar ways. UMAP's focus on local neighborhoods blurs this global pattern. SVD (linear) works better for capturing this simple length-based separation.

---

### Question 5

Plots and Visualization: Select the best-performing configuration for TF–IDF and MiniLM based on clustering performance. For each representation:

- Reduce the embeddings to two dimensions using PCA (sklearn.decomposition.PCA).
- Create a split visualization with:
  - One plot colored by the ground-truth length label.
  - One plot colored by the cluster assignments obtained using your best clustering method.

#### Answer

Visualization file: `outputs/task1_4_pca_visualizations.png`

PCA reduction with 2 components:
- TF-IDF explained variance: 4.02% total (PC1: 2.39%, PC2: 1.63%)
- MiniLM explained variance: 11.83% total (PC1: 8.71%, PC2: 3.13%)

**2×2 visualization grid:**

| Position     | Plot                    | Description                              |
| ------------ | ----------------------- | ---------------------------------------- |
| Top-left     | TF-IDF: Ground Truth    | Colored by Short/Long labels, weak separation |
| Top-right    | TF-IDF: Best Clustering | SVD + Agglomerative (V=0.047), poor cluster quality |
| Bottom-left  | MiniLM: Ground Truth    | Colored by Short/Long labels, moderate separation |
| Bottom-right | MiniLM: Best Clustering | SVD + Agglomerative (V=0.648), clear cluster separation |

MiniLM points show more structured separation by length. TF-IDF points appear more scattered. The 2D PCA projection captures MiniLM structure better (11.83% vs 4.02% variance explained). Cluster labels in the MiniLM visualization align well with ground truth colors.

---

## Task 2 - Unsupervised Game Similarity & Genre Structure

---

### Question 6

Report the dimensions of the TF-IDF game matrix and the MiniLM game embedding matrix.

#### Answer

- TF-IDF game matrix shape: (200 games × 16,554 features), vocabulary derived from concatenating positive reviews per game
- MiniLM game embedding matrix shape: (200 games × 384 dimensions), each game vector is the average of all positive review MiniLM embeddings

---

### Question 7

For each pipeline, report a summary table that includes:

- number of clusters found (for HDBSCAN, also report the fraction of games labeled as noise -1),
- cluster sizes,
- for each cluster: top 3 most common genres (by frequency across games in that cluster).

#### Answer

**MiniLM Pipelines (7 total)**

| Pipeline                | K Found | Noise           | Cluster Sizes      | Top Genre Purity Range |
| ----------------------- | ------- | --------------- | ------------------ | ---------------------- |
| None + K-Means          | 5       | 0               | 9, 86, 23, 58, 24  | 67% - 81%              |
| None + Agglomerative    | 5       | 0               | 41, 33, 62, 20, 44 | 54% - 93%              |
| SVD + K-Means           | 5       | 0               | 25, 30, 50, 55, 40 | 62% - 82%              |
| SVD + Agglomerative     | 5       | 0               | 37, 62, 32, 13, 56 | 59% - 100%             |
| UMAP + K-Means          | 5       | 0               | 23, 54, 51, 33, 39 | 61% - 85%              |
| UMAP + Agglomerative    | 5       | 0               | 78, 41, 32, 26, 23 | 61% - 96%              |
| None + HDBSCAN          | 2       | 163 (81.5%)     | 28, 9              | 75% - 100%             |

**TF-IDF Pipelines (2 total)**

| Pipeline            | K Found | Noise | Cluster Sizes     | Top Genre Purity Range |
| ------------------- | ------- | ----- | ----------------- | ---------------------- |
| SVD + K-Means       | 5       | 0     | 5, 94, 35, 62, 4  | 60% - 100%             |
| SVD + Agglomerative | 5       | 0     | 49, 12, 130, 5, 4 | 65% - 100%             |

**Sample cluster genre breakdown (MiniLM + SVD + Agglomerative, k=5):**

| Cluster | Size | Top Genre 1       | Top Genre 2                 | Top Genre 3     | Purity   |
| ------- | ---- | ----------------- | --------------------------- | --------------- | -------- |
| 0       | 37   | Indie (59%)       | Action (59%)                | Adventure (43%) | 59%      |
| 1       | 62   | Action (85%)      | Adventure (56%)             | RPG (44%)       | 85%      |
| 2       | 32   | Adventure (66%)   | Simulation (63%)            | Indie (56%)     | 66%      |
| 3       | 13   | Action (100%)     | Massively Multiplayer (38%) | Indie (23%)     | 100%     |
| 4       | 56   | Action (64%)      | Adventure (46%)             | Indie (38%)     | 64%      |

---

### Question 8

Pick one best pipeline (justify your choice), then report two cluster with high purity:

- top 3 genres with percentages,
- cluster genre purity (as defined above),
- Representative games in the cluster (game name + genres).

Provide a short interpretation: what type of games does these clusters represent?

#### Answer

**Selected Pipeline:** MiniLM + SVD(50) + Agglomerative

**Justification:** This pipeline achieved the best results in Task 1 (V=0.648, ARI=0.742) and shows strong genre purity in Task 2. MiniLM captures semantic similarity better than TF-IDF for game content, and SVD dimensionality reduction improves clustering efficiency while preserving semantic structure.

---

**High-Purity Cluster 1 (Cluster 3)**

| Metric             | Value                       |
| ------------------ | --------------------------- |
| Size               | 13 games                    |
| Top Genre 1        | Action (100%)               |
| Top Genre 2        | Massively Multiplayer (38%) |
| Top Genre 3        | Indie (23%)                 |
| Cluster Purity     | 100%                        |

**Representative Games:**

| Game             | Genres                             |
| ---------------- | ---------------------------------- |
| Counter-Strike 2 | Action, Massively Multiplayer      |
| Dota 2           | Action, Massively Multiplayer, RPG |
| Team Fortress 2  | Action, Massively Multiplayer      |

**Interpretation:** This cluster is pure action-focused competitive multiplayer games. These are primarily online multiplayer titles with fast-paced combat. The 100% purity means all games in this cluster have the Action genre label.

---

**High-Purity Cluster 2 (Cluster 1)**

| Metric             | Value           |
| ------------------ | --------------- |
| Size               | 62 games        |
| Top Genre 1        | Action (85%)    |
| Top Genre 2        | Adventure (56%) |
| Top Genre 3        | RPG (44%)       |
| Cluster Purity     | 85%             |

**Representative Games:**

| Game          | Genres                   |
| ------------- | ------------------------ |
| Dying Light   | Action, RPG              |
| Black Mesa    | Action, Adventure, Indie |
| Borderlands 3 | Action, RPG              |

**Interpretation:** This cluster represents Action-Adventure-RPG hybrid games. These are narrative-driven action titles with RPG progression elements. This is the largest cluster of mainstream AAA titles, blending combat, exploration, and character development.

---

## Task 3 - Held-Out Game Profiling and Theme Discovery

---

### Question 9

Report: (i) the assigned cluster ID, (ii) the top 3 genres of that cluster, and (iii) 3 representative games from that cluster. Briefly justify why this constitutes a genre estimate in a multi-genre world.

#### Answer

**(i) Assigned Cluster ID:** Cluster 4 (from MiniLM + SVD + Agglomerative pipeline)

**(ii) Top 3 Genres of Cluster:**

| Rank | Genre     | Percentage | Games in Cluster |
| ---- | --------- | ---------- | ---------------- |
| 1    | Action    | 80%        | 32/40            |
| 2    | Adventure | 55%        | 22/40            |
| 3    | RPG       | 35%        | 14/40            |

**(iii) Representative Games:**

| Game          | Genres                   |
| ------------- | ------------------------ |
| Dying Light   | Action, RPG              |
| Black Mesa    | Action, Adventure, Indie |
| Borderlands 3 | Action, RPG              |

**Justification for multi-genre estimation:** Games are multi-label by nature. A single game can belong to Action, Adventure, and RPG simultaneously. Rather than predicting a single genre, we estimate a genre distribution by assigning the held-out game to its nearest cluster and inheriting that cluster's genre distribution. The held-out game likely shares Action-Adventure-RPG characteristics because it semantically clusters with games having those genre combinations.

---

### Question 10

For negative reviews: report 3–5 clusters with (i) top terms and (ii) exemplar reviews, and assign a short label to each complaint cluster.

#### Answer

**Cluster 0 - Boss Design & Difficulty Frustration (32 reviews)**

**(i) Top Terms:** game, boss, just, like, bosses, fun, fight, really, design, attacks

**(ii) Exemplar Reviews:**

> "Lots of Misleading reviews on here. The game is beyond difficult, not suitable for anyone less than a hardcore gamer. It is not open map and not explorable. Instead, you go through a linear style path way encountering boss after boss which are incredibly difficult to beat..."

> "Unlike most people who reviewed this game after 30 minutes and declared it GOTY, i actually played the game. TLDR: Fun at the start, get repetitive really fast and overstays its welcome, with a bunch of glitches and graphical errors..."

**Label:** Boss Difficulty & Repetitive Combat Frustration

---

**Cluster 2 - Technical Performance Issues (18 reviews)**

**(i) Top Terms:** game, fps, low, crashes, settings, issues, amd, like, 7900xtx, rtx

**(ii) Exemplar Reviews:**

> "I wanted to share my recent experience... I'm running this on an i7 13th gen, RTX 4080, and 32 gigs RAM, which should be more than capable. However, the game crashes consistently, even before I can get past the intro..."

> "Not really sure what these purchased reviews are all about. Game runs terribly. 1440p, Cinematic, 4090, 32GB Ram, 13900k. Constant stutters."

**Label:** Technical Performance & Crash Complaints

---

**Cluster 3 - Boring Gameplay & Empty World (14 reviews)**

**(i) Top Terms:** game, like, just, graphics, level, play, really, good, feels, story

**(ii) Exemplar Reviews:**

> "NGL this game looks really good, but god it is boring. The world and details look incredible but everything feels lifeless and the combat also just feels boring..."

> "Gameplay is boring. Cutscenes and graphics are very nice, story seems interesting, but it's not enough to carry the game for me. I could accept having invisible walls and very linear level design if the combat had more 'meat' to it..."

**Label:** Boring Gameplay Despite Good Graphics

---

**Cluster 4 - Souls-like Comparison & Combat Criticism (24 reviews)**

**(i) Top Terms:** game, boss, like, just, games, combat, story, souls, bosses, good

**(ii) Exemplar Reviews:**

> "-Intriguing prologue, the whole game is a piece of ♥♥♥♥-Same monsters and bosses-No story-Characters are empty, they only give quests-Monotonous combat-Why can't parry attacks?-The graphics are good, but the level design sucks..."

> "I simply do not understand how this game is overwhelmingly positive. It feels, in many places, extremely amateurish in fact. Level design is objectively terrible. Huge open maps that feel procedurally generated they're so bland..."

**Label:** Souls-like Wannabe with Poor Level Design

---

### Question 11

For positive reviews: repeat the same analysis and label 3–5 praise clusters.

#### Answer

**Cluster 1 - Game Quality & Visual Excellence (26 reviews)**

**(i) Top Terms:** game, 10, wukong, chinese, like, good, myth, games, black, black myth

**(ii) Exemplar Reviews:**

> "Black Myth: Wukong was released under a lot of anticipation after gameplay showcases had been shown over the past few years... The game is engaging, well made, fluid, has a fair dose of difficulty, is downright beautiful and is worth every penny..."

> "Black Myth Wukong is an absolute masterclass of a game. All the hype, all the praise, all the positive buzz surrounding this game... they're absolutely deserved and this is hands down, one of THE best games of this generation..."

**Label:** Masterpiece Quality & Visual Excellence

---

**Cluster 2 - Fair & Balanced Gameplay (15 reviews)**

**(i) Top Terms:** good, grind, just, bad, 10, long, average, bugs, game, hard

**(ii) Exemplar Reviews:**

> "---{ Graphics }--- ☑ You forget what reality is ☑ Very good gameplay... ---{ Game Size }---..."

> "Good game. Fair difficulty. Good combat."

**Label:** Fair Difficulty & Balanced Design

---

**Cluster 3 - Core Gameplay & Combat Satisfaction (41 reviews)**

**(i) Top Terms:** game, like, games, just, good, graphics, really, play, combat, great

**(ii) Exemplar Reviews:**

> "This game has the big awesome set pieces and aesthetic from the old school God of War games, the core gameplay of a Soul-like (albeit a tad more forgiving), really sick and fast paced Sekiro / Bloodborne combat..."

> "'Wisdom lies in the balance of wit and humility.' ~ Sun Wukong. I've been waiting for the game ever since it's 1st reveal... the soundtracks, visual setting, bosses everything is so good so far..."

**Label:** Engaging Combat & Soul-like Mechanics

---

### Question 12

Include your prompting strategy and 3 example of LLM-generated labels for clusters.

#### Answer

**Prompting Strategy:**

We used the Qwen/Qwen3-4B-Instruct-2507 model for generating cluster labels. For each cluster, we provided:

1. Top TF-IDF terms - The 10 most characteristic terms from the cluster
2. Exemplar reviews - 1-2 reviews closest to the cluster centroid

**Prompt Template:**

```
You are analyzing Steam game review clusters. For each cluster, you will receive:
1. Top TF-IDF terms that characterize the cluster
2. 1-2 exemplar reviews (representative reviews closest to cluster centroid)

Based on this evidence, generate a concise 3-6 word label that captures the main theme of this cluster.

Input:
Top Terms: [term1, term2, term3, ...]
Exemplar Review 1: [review text]
Exemplar Review 2: [review text]

Output a single label (3-6 words) that best describes what these reviews have in common.
```

Evidence provided per cluster: 10 top TF-IDF terms (unigrams and bigrams), 2 exemplar reviews (truncated to 500 characters if necessary)

---

**Example 1: Negative Cluster 0**

_Evidence Provided:_

- Top Terms: game, boss, just, like, bosses, fun, fight, really, design, attacks
- Exemplar: "Lots of Misleading reviews on here. The game is beyond difficult, not suitable for anyone less than a hardcore gamer..."

_LLM-Generated Label:_ "Unfair Boss Difficulty & Hardcore Gatekeeping"

---

**Example 2: Negative Cluster 2**

_Evidence Provided:_

- Top Terms: game, fps, low, crashes, settings, issues, amd, 7900xtx, rtx
- Exemplar: "I'm running this on an i7 13th gen, RTX 4080... the game crashes consistently..."

_LLM-Generated Label:_ "High-End PC Performance Failures"

---

**Example 3: Positive Cluster 1**

_Evidence Provided:_

- Top Terms: game, 10, wukong, chinese, like, good, myth, games, black, black myth
- Exemplar: "Black Myth Wukong is an absolute masterclass of a game. All the hype... they're absolutely deserved..."

_LLM-Generated Label:_ "Masterclass Action Game With Cultural Depth"


