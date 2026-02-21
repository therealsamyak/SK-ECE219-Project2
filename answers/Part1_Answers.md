# Part 1 - Steam Reviews: Product Analytics with Representations and Clustering

## Task 1 - Unsupervised Review Length Discovery

### Question 1

Report the number of reviews retained after filtering, and the average length (in tokens) of Short and Long reviews.

**Answer:**

- Total reviews retained: 20,497 (after discarding middle 50%)
- Short reviews (≤ q25 = 11 words): 10,463 reviews, avg 6.39 words
- Long reviews (≥ q75 = 179 words): 10,034 reviews, avg 493 words

---

### Question 2

Report the dimensions of the TF-IDF matrix and the MiniLM embedding matrix. Briefly explain why TF-IDF is sparse while MiniLM embeddings are dense.

**Answer:**

- TF-IDF: (20,497 × 25,085), 99.68% sparse
- MiniLM: (20,497 × 384), dense

TF-IDF scores each vocabulary term independently. A 50-word review might touch 30 unique terms out of 25,085—everything else is zero.

MiniLM runs the entire sentence through a neural network that outputs a 384-d vector. All dimensions get non-zero values because the model encodes meaning holistically, not by counting words.

---

### Question 3

For each pipeline, report clustering agreement metrics with respect to ground-truth length labels: homogeneity, completeness, v-measure, ARI, AMI.

**Answer:**

**TF-IDF (2 pipelines)**

| Pipeline            | Homogeneity | Completeness | V-Measure | ARI    | AMI    |
| ------------------- | ----------- | ------------ | --------- | ------ | ------ |
| SVD + K-Means       | 0.0301      | 0.161        | 0.0507    | 0.0057 | 0.0507 |
| SVD + Agglomerative | 0.0276      | 0.157        | 0.0470    | 0.0050 | 0.0470 |

TF-IDF clustering barely separates Short vs Long. UMAP and no-reduction options were skipped—infeasible on sparse, high-dimensional data.

**MiniLM (6 pipelines)**

| Pipeline                | Homogeneity | Completeness | V-Measure | ARI       | AMI       |
| ----------------------- | ----------- | ------------ | --------- | --------- | --------- |
| None + K-Means          | 0.509       | 0.509        | 0.509     | 0.613     | 0.509     |
| None + Agglomerative    | 0.543       | 0.547        | 0.545     | 0.643     | 0.545     |
| SVD + K-Means           | 0.508       | 0.509        | 0.509     | 0.612     | 0.509     |
| **SVD + Agglomerative** | **0.646**   | **0.650**    | **0.648** | **0.742** | **0.648** |
| UMAP + K-Means          | 0.558       | 0.559        | 0.558     | 0.666     | 0.558     |
| UMAP + Agglomerative    | 0.553       | 0.556        | 0.555     | 0.658     | 0.555     |

**Best:** MiniLM + SVD(50) + Agglomerative. V-Measure 0.648, ARI 0.742.

---

### Question 4

Compare TF-IDF and MiniLM performance. Which separates Short vs Long more cleanly, and why?

**Answer:**

| Metric      | TF-IDF | MiniLM | Gap   |
| ----------- | ------ | ------ | ----- |
| V-Measure   | 0.047  | 0.648  | 13.8x |
| ARI         | 0.005  | 0.742  | 148x  |
| Homogeneity | 0.028  | 0.646  | 23x   |

MiniLM wins by a lot. Three reasons:

1. **Semantic understanding:** MiniLM captures sentence structure and semantic meaning. Long reviews differ from short ones in topic depth, discourse structure, and complexity—neural embeddings pick up on these differences naturally.

2. **Document length as latent signal:** Transformers attend to longer sequences differently, creating distinct representations for different-length inputs.

3. **TF-IDF limitations:** TF-IDF only counts word frequencies. Long and short reviews both contain "game", "play", "good", so they look similar in TF-IDF space even when structurally different.

UMAP actually hurts here. It preserves local neighborhoods, but Short vs Long is a global, linear signal. Document length affects all dimensions similarly. UMAP's focus on local structure blurs this global pattern. SVD (linear) works better for capturing this simple length-based separation.

---

### Question 5

Plots and Visualization: Reduce embeddings to 2D using PCA and create split visualizations.

**Answer:**

File: [PCA visualizations](outputs/task1_4_pca_visualizations.png)

PCA explained variance:

- TF-IDF: 4.02% total
- MiniLM: 11.83% total

The MiniLM plot shows clear separation between Short and Long clusters. TF-IDF is scattered. The higher variance explained by MiniLM's first two PCs (11.83% vs 4.02%) tells us the structure is more amenable to low-dimensional projection.

---

## Task 2 - Unsupervised Game Similarity & Genre Structure

### Question 6

Report the dimensions of the TF-IDF game matrix and the MiniLM game embedding matrix.

**Answer:**

- TF-IDF: (200 × 16,554), vocabulary from concatenated positive reviews
- MiniLM: (200 × 384), averaged positive review embeddings

---

### Question 7

For each pipeline, report cluster count, sizes, and top 3 genres per cluster.

**Answer:**

**MiniLM (12 pipelines)**

**Non-HDBSCAN (8 pipelines)**

| Pipeline             | K   | Sizes              | Top Genre Purity |
| -------------------- | --- | ------------------ | ---------------- |
| None + K-Means       | 5   | 9, 86, 23, 58, 24  | 67-81%           |
| None + Agglomerative | 5   | 41, 33, 62, 20, 44 | 54-93%           |
| SVD + K-Means        | 5   | 25, 30, 50, 55, 40 | 62-82%           |
| SVD + Agglomerative  | 5   | 37, 62, 32, 13, 56 | 59-100%          |
| UMAP + K-Means       | 5   | 39, 61, 26, 42, 32 | 62-85%           |
| UMAP + Agglomerative | 5   | 85, 40, 30, 22, 23 | 61-96%           |
| AE + K-Means         | 5   | 63, 8, 34, 38, 57  | 67-75%           |
| AE + Agglomerative   | 5   | 50, 33, 63, 39, 15 | 62-100%          |

**HDBSCAN (4 pipelines)**

| Pipeline       | K   | Noise       | Sizes  | Top Genre Purity |
| -------------- | --- | ----------- | ------ | ---------------- |
| None + HDBSCAN | 2   | 175 (87.5%) | 18, 7  | 83-100%          |
| SVD + HDBSCAN  | 2   | 168 (84%)   | 25, 7  | 72-100%          |
| UMAP + HDBSCAN | 10  | 64 (32%)    | 5-35   | 60-100%          |
| AE + HDBSCAN   | 2   | 89 (44.5%)  | 104, 7 | 74-100%          |

**TF-IDF (3 pipelines)**

| Pipeline            | K   | Sizes             | Top Genre Purity |
| ------------------- | --- | ----------------- | ---------------- | ------ |
| SVD + K-Means       | 5   | 5, 94, 35, 62, 4  | 60-100%          |
| SVD + Agglomerative | 5   | 49, 12, 130, 5, 4 | 65-100%          |
| SVD + HDBSCAN       | 2   | 7 (3.5%)          | 188, 5           | 69-80% |

Sample genre breakdown (MiniLM + SVD + Agglomerative):

| Cluster | Size | Top Genres                                     | Purity |
| ------- | ---- | ---------------------------------------------- | ------ |
| 0       | 37   | Indie (59%), Action (59%), Adventure (43%)     | 59%    |
| 1       | 62   | Action (85%), Adventure (56%), RPG (44%)       | 85%    |
| 2       | 32   | Adventure (66%), Simulation (63%), Indie (56%) | 66%    |
| 3       | 13   | Action (100%), MMO (38%), Indie (23%)          | 100%   |
| 4       | 56   | Action (64%), Adventure (46%), Indie (38%)     | 64%    |

---

### Question 8

Pick the best pipeline and report two high-purity clusters with genres, purity, and representative games.

**Answer:**

**Selected:** MiniLM + SVD(50) + Agglomerative

This pipeline performed best in Task 1 and shows strong genre purity here.

**Cluster 3 (13 games, 100% pure)**

Top genres: Action (100%), Massively Multiplayer (38%), Indie (23%)

| Game             | Genres                             |
| ---------------- | ---------------------------------- |
| Counter-Strike 2 | Action, Massively Multiplayer      |
| Dota 2           | Action, Massively Multiplayer, RPG |
| Team Fortress 2  | Action, Massively Multiplayer      |

These are competitive online multiplayer games. All have the Action tag.

**Cluster 1 (62 games, 85% pure)**

Top genres: Action (85%), Adventure (56%), RPG (44%)

| Game          | Genres                   |
| ------------- | ------------------------ |
| Dying Light   | Action, RPG              |
| Black Mesa    | Action, Adventure, Indie |
| Borderlands 3 | Action, RPG              |

Narrative-driven action games with RPG elements. The mainstream AAA cluster.

---

## Task 3 - Held-Out Game Profiling and Theme Discovery

### Question 9

Report assigned cluster ID, top 3 genres, and 3 representative games.

**Answer:**

**Cluster ID:** 4 (MiniLM + SVD + Agglomerative)

Top genres: Action (80%), Adventure (55%), RPG (35%)

| Game          | Genres                   |
| ------------- | ------------------------ |
| Dying Light   | Action, RPG              |
| Black Mesa    | Action, Adventure, Indie |
| Borderlands 3 | Action, RPG              |

Why this works as genre estimation: Games are multi-label. Rather than predicting one genre, we assign the held-out game to its nearest cluster and inherit that cluster's genre distribution. The held-out game shares Action-Adventure-RPG characteristics because it clusters semantically with games having those tags.

---

### Question 10

For negative reviews: report 3-5 clusters with top terms, exemplar reviews, and labels.

**Answer:**

**Pipeline:** MiniLM + SVD(50) + Agglomerative

**Justification:** This pipeline achieved best performance in Task 1 (V-Measure 0.648, ARI 0.742) and Task 2 (up to 100% genre purity). Consistent with Q8 and Q9.

**Cluster 0 - Difficult & Repetitive (32 reviews)**

Terms: game, boss, just, like, bosses, fun, fight, really, design, attacks

> "Lots of Misleading reviews on here. The game is beyond difficult, not suitable for anyone less than a hardcore gamer. It is not open map and not explorable. Instead, you go through a linear style path way encountering boss after boss which are incredibly difficult to beat..."

> "Unlike most people who reviewed this game after 30 minutes and declared it GOTY, i actually played the game. TLDR: Fun at the start, get repetitive really fast and overstays its welcome, with a bunch of glitches and graphical errors..."

Label: Difficult and repetitive gameplay

---

**Cluster 1 - Boring/Political (12 reviews)**

Terms: game, like, developers misogynists, taiwan country, boring, fun

> "Game is boring af. Another souls game wannabe except without it's charm. Sekiro had satisfying combat, Elden ring had massive open world and a doll waifu..."

Label: Boring, repetitive, misogynistic criticism

---

**Cluster 2 - Performance Issues (18 reviews)**

Terms: fps, low, crashes, settings, issues, amd, 7900xtx, rtx

> "I wanted to share my recent experience with this where I have seen that's been getting positive reviews. I'm running this on an i7 13th gen, RTX 4080, and 32 gigs RAM, which should be more than capable. However, the game crashes consistently..."

> "Not really sure what these purchased reviews are all about. Game runs terribly.1440p, Cinematic, 4090, 32GB Ram, 13900kConstant stutters."

Label: Performance Issues

---

**Cluster 3 - Boring Gameplay (14 reviews)**

Terms: game, like, just, graphics, level, play, really, good, feels, story

> "NGL this game looks really good, but god it is boring. The world and details look incredible but everything feels lifeless and the combat also just feels boring..."

> "Gameplay is boring. Cutscenes and graphics are very nice, story seems interesting, but it's not enough to carry the game for me..."

Label: Boring gameplay, lifeless world

---

**Cluster 4 - Souls-like Comparison (24 reviews)**

Terms: game, boss, like, just, games, combat, story, souls, bosses, good

> "-Intriguing prologue, the whole game is a piece of... Same monsters and bosses. No story. Characters are empty, they only give quests. Monotonous combat. Graphics good, level design sucks..."

> "I simply do not understand how this game is overwhelmingly positive. It feels, in many places, extremely amateurish in fact. Level design is objectively terrible. Huge open maps that feel procedurally generated..."

Label: Monotonous, Amateurish Gameplay

---

### Question 11

For positive reviews: repeat with praise clusters.

**Answer:**

**Pipeline:** MiniLM + SVD(50) + Agglomerative

**Justification:** Parity with Q10.

**Cluster 0 - Monke Meme (6 reviews)**

Terms: monke, reject, monke approves, embraces, embraces monke

> "REJECT MODERNITYRETURN TO MONKE"

Label: Monke Embrace & Approval

---

**Cluster 1 - Black Myth Praise (26 reviews)**

Terms: game, 10, wukong, chinese, like, good, myth, games, black

> "Black Myth: Wukong was released under a lot of anticipation after gameplay showcases had been shown over the past few years, leading people to even question if the game would really come out like that. Thankfully, it really did – and overcame all expectations..."

> "Black Myth Wukong is an absolute masterclass of a game. All the hype, all the praise, all the positive buzz surrounding this game...they're absolutely deserved..."

Label: Hype and Praise for Black Myth: Wukong

---

**Cluster 2 - Template Reviews (15 reviews)**

Terms: good, grind, just, bad, 10, long, average, bugs, game, hard

> "---{ Graphics }---☑ You forget what reality is..."

Label: Grind and repetitive gameplay

---

**Cluster 3 - Combat Satisfaction (41 reviews)**

Terms: game, like, games, just, good, graphics, really, play, combat, great

> "This game has the big awesome set pieces and aesthetic from the old school God of War games, the core gameplay of a Soul-like (albeit a tad more forgiving), really sick and fast paced Sekiro / Bloodborne combat..."

> ""Wisdom lies in the balance of wit and humility." ~ Sun Wukong. I've been waiting for the game ever since it's 1st reveal... the soundtracks, visual setting, bosses everything is so good so far..."

Label: Positive gameplay experience

---

**Cluster 4 - Sweet Baby Reference (12 reviews)**

Terms: ign, sweet, sweet baby, baby, add, monke

> "※This game does not include Sweet Baby"

Label: Sweet Baby Game Reference

---

### Question 12

Include prompting strategy and 3 LLM-generated label examples.

**Answer:**

Model: Qwen/Qwen3-4B-Instruct-2507

Input per cluster: 10 top TF-IDF terms, 2 exemplar reviews

**Prompt:**

```
You are analyzing Steam review clusters. Input:
Top Terms: [terms]
Exemplar Reviews: [reviews]

Output a 3-6 word label describing what these reviews have in common.
```

**Example 1 (Negative Cluster 0)**

Terms: game, boss, just, like, bosses, fun, fight, design, attacks
Exemplar: "Lots of Misleading reviews on here. The game is beyond difficult..."

LLM output: "Difficult and repetitive gameplay"

**Example 2 (Negative Cluster 2)**

Terms: fps, low, crashes, settings, issues, amd, rtx
Exemplar: "I'm running this on an i7 13th gen, RTX 4080, and 32 gigs RAM... the game crashes consistently..."

LLM output: "Performance Issues"

**Example 3 (Positive Cluster 1)**

Terms: game, 10, wukong, chinese, like, good, myth, games, black
Exemplar: "Black Myth: Wukong was released under a lot of anticipation..."

LLM output: "Hype and Praise for Black Myth: Wukong"
