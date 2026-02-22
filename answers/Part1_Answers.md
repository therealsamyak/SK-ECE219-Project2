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

TF-IDF scores each vocabulary term independently. However, MiniLM runs the entire sentence through a neural network that outputs a 384-d vector. All dimensions get non-zero values because the model captures meaning from the whole sentence, not by counting words.

---

### Question 3

For each pipeline, report clustering agreement metrics with respect to ground-truth length labels: homogeneity, completeness, v-measure, ARI, AMI.

**Answer:**

**TF-IDF (2 pipelines)**

| Pipeline            | Homogeneity | Completeness | V-Measure | ARI    | AMI    |
| ------------------- | ----------- | ------------ | --------- | ------ | ------ |
| SVD + K-Means       | 0.0301      | 0.161        | 0.0507    | 0.0057 | 0.0507 |
| SVD + Agglomerative | 0.0276      | 0.157        | 0.0470    | 0.0050 | 0.0470 |

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

MiniLM outperforms TF-IDF on all metrics. The neural network is able to capture sentence structure and semantic meaning, whereas TF-IDF only counts word frequencies. Transformers also treat longer sequences differently, creating distinct representations for different-length inputs.

SVD works better than UMAP here. Document length affects all dimensions similarly, so UMAP's focus on local neighborhoods doesn't differentiate the clusters effectively.

---

### Question 5

Plots and Visualization: Reduce embeddings to 2D using PCA and create split visualizations.

**Answer:**

File: [PCA visualizations](outputs/Q5_pca_visualizations.png)

PCA explained variance:

- TF-IDF: 4.02% total
- MiniLM: 11.83% total

MiniLM clearly separates Short and Long clusters. TF-IDF is scattered. MiniLM projects better in low dimensions, as its two PCs show 11.83% variance (vs 4.02% for TF-IDF).

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

### MiniLM Pipelines

**None + K-Means** (K=5)

| Cluster | Size | Top 3 Genres (purity)                          |
| ------- | ---- | ---------------------------------------------- |
| 0       | 9    | Simulation (67%), Sports (67%), Action (33%)   |
| 1       | 23   | Action (69%), Indie (55%), Adventure (50%)     |
| 2       | 24   | Action (74%), Adventure (52%), RPG (48%)       |
| 3       | 58   | Action (81%), Adventure (48%), RPG (33%)       |
| 4       | 86   | Adventure (71%), Simulation (54%), Indie (50%) |

---

**None + Agglomerative** (K=5)

| Cluster | Size | Top 3 Genres (purity)                          |
| ------- | ---- | ---------------------------------------------- |
| 0       | 20   | Action (63%), Adventure (49%), Indie (49%)     |
| 1       | 33   | Action (82%), Adventure (45%), RPG (27%)       |
| 2       | 41   | Action (55%), Indie (53%), Adventure (50%)     |
| 3       | 44   | Simulation (70%), Indie (60%), Adventure (55%) |
| 4       | 62   | Action (93%), Adventure (52%), RPG (48%)       |

---

**SVD + K-Means** (K=5)

| Cluster | Size | Top 3 Genres (purity)                          |
| ------- | ---- | ---------------------------------------------- |
| 0       | 25   | Action (76%), Adventure (52%), RPG (44%)       |
| 1       | 30   | Adventure (67%), Simulation (57%), Indie (53%) |
| 2       | 40   | Indie (62%), Action (58%), Adventure (54%)     |
| 3       | 50   | Action (82%), Indie (38%), Adventure (33%)     |
| 4       | 55   | Action (80%), Adventure (55%), RPG (35%)       |

---

**SVD + Agglomerative** (K=5)

| Cluster | Size | Top 3 Genres (purity)                                   |
| ------- | ---- | ------------------------------------------------------- |
| 0       | 13   | Indie (59%), Action (59%), Adventure (43%)              |
| 1       | 32   | Action (85%), Adventure (56%), RPG (44%)                |
| 2       | 37   | Adventure (66%), Simulation (62%), Indie (56%)          |
| 3       | 56   | Action (100%), Massively Multiplayer (38%), Indie (23%) |
| 4       | 62   | Action (64%), Adventure (46%), Indie (38%)              |

---

**UMAP + K-Means** (K=5)

| Cluster | Size | Top 3 Genres (purity)                          |
| ------- | ---- | ---------------------------------------------- |
| 0       | 26   | Action (85%), Adventure (49%), RPG (36%)       |
| 1       | 32   | Action (64%), Adventure (46%), Indie (39%)     |
| 2       | 39   | Simulation (62%), Adventure (54%), Indie (54%) |
| 3       | 42   | Indie (62%), Action (60%), Adventure (50%)     |
| 4       | 61   | Action (84%), Adventure (56%), RPG (34%)       |

---

**UMAP + Agglomerative** (K=5)

| Cluster | Size | Top 3 Genres (purity)                          |
| ------- | ---- | ---------------------------------------------- |
| 0       | 22   | Action (66%), Adventure (52%), Indie (41%)     |
| 1       | 23   | Action (85%), Adventure (45%), RPG (38%)       |
| 2       | 30   | Indie (63%), Action (53%), Adventure (47%)     |
| 3       | 40   | Simulation (64%), Adventure (55%), Indie (50%) |
| 4       | 85   | Action (87%), Adventure (52%), RPG (35%)       |

---

**AE + K-Means** (K=5)

| Cluster | Size | Top 3 Genres (purity)                          |
| ------- | ---- | ---------------------------------------------- |
| 0       | 29   | Indie (67%), Adventure (56%), Action (53%)     |
| 1       | 34   | Action (84%), Adventure (51%), RPG (30%)       |
| 2       | 36   | Action (74%), Adventure (59%), RPG (41%)       |
| 3       | 43   | Action (79%), Indie (40%), Adventure (38%)     |
| 4       | 58   | Indie (62%), Adventure (55%), Simulation (55%) |

---

**AE + Agglomerative** (K=5)

| Cluster | Size | Top 3 Genres (purity)                      |
| ------- | ---- | ------------------------------------------ |
| 0       | 27   | Indie (68%), Action (58%), Adventure (54%) |
| 1       | 40   | Action (85%), Adventure (50%), RPG (28%)   |
| 2       | 40   | Action (70%), Adventure (59%), RPG (37%)   |
| 3       | 43   | Action (77%), Indie (35%), Adventure (35%) |
| 4       | 50   | Indie (57%), Adventure (55%), Action (52%) |

---

**None + HDBSCAN** (K=2, 87.5% noise)

| Cluster | Size | Top 3 Genres (purity)                                          |
| ------- | ---- | -------------------------------------------------------------- |
| 0       | 7    | Action (83%), Adventure (56%), Indie (50%)                     |
| 1       | 18   | Action (100%), Massively Multiplayer (43%), Early Access (29%) |

---

**SVD + HDBSCAN** (K=2, 84.0% noise)

| Cluster | Size | Top 3 Genres (purity)                                          |
| ------- | ---- | -------------------------------------------------------------- |
| 0       | 7    | Action (72%), Adventure (64%), Indie (52%)                     |
| 1       | 25   | Action (100%), Massively Multiplayer (43%), Early Access (29%) |

---

**UMAP + HDBSCAN** (K=10, 32.0% noise)

| Cluster | Size | Top 3 Genres (purity)                                          |
| ------- | ---- | -------------------------------------------------------------- |
| 0       | 5    | Action (80%), Racing (80%), Adventure (60%)                    |
| 1       | 6    | Adventure (86%), Simulation (57%), Indie (43%)                 |
| 2       | 6    | Action (83%), RPG (50%), Adventure (50%)                       |
| 3       | 20   | Simulation (65%), Adventure (60%), Indie (55%)                 |
| 4       | 9    | Casual (80%), Indie (60%), Action (60%)                        |
| 5       | 35   | Action (91%), Adventure (60%), RPG (40%)                       |
| 6       | 14   | Action (100%), Free To Play (30%), Massively Multiplayer (20%) |
| 7       | 13   | Action (60%), Indie (53%), Adventure (47%)                     |
| 8       | 8    | Indie (85%), Adventure (60%), Action (60%)                     |
| 9       | 20   | Simulation (86%), Sports (86%), Action (14%)                   |

---

**AE + HDBSCAN** (K=2, 80.5% noise)

| Cluster | Size | Top 3 Genres (purity)                                          |
| ------- | ---- | -------------------------------------------------------------- |
| 0       | 6    | Action (76%), Indie (55%), Adventure (52%)                     |
| 1       | 33   | Action (100%), Early Access (33%), Massively Multiplayer (33%) |

---

### TF-IDF Pipelines

**SVD + K-Means** (K=5)

| Cluster | Size | Top 3 Genres (purity)                                     |
| ------- | ---- | --------------------------------------------------------- |
| 0       | 94   | Action (80%), Massively Multiplayer (60%), Strategy (40%) |
| 1       | 35   | Action (71%), Adventure (57%), RPG (46%)                  |
| 2       | 62   | Action (60%), Adventure (60%), Indie (43%)                |
| 3       | 5    | Action (71%), Indie (40%), Adventure (39%)                |
| 4       | 4    | Simulation (100%), Sports (100%)                          |

---

**SVD + Agglomerative** (K=5)

| Cluster | Size | Top 3 Genres (purity)                                     |
| ------- | ---- | --------------------------------------------------------- |
| 0       | 130  | Action (80%), Indie (45%), Adventure (43%)                |
| 1       | 12   | Action (67%), Adventure (58%), Indie (58%)                |
| 2       | 49   | Action (65%), Adventure (55%), RPG (41%)                  |
| 3       | 5    | Action (80%), Massively Multiplayer (60%), Strategy (40%) |
| 4       | 4    | Simulation (100%), Sports (100%)                          |

---

**SVD + HDBSCAN** (K=2, 3.5% noise)

| Cluster | Size | Top 3 Genres (purity)                                     |
| ------- | ---- | --------------------------------------------------------- |
| 0       | 5    | Action (69%), Adventure (52%), Indie (41%)                |
| 1       | 188  | Action (80%), Massively Multiplayer (60%), Strategy (40%) |

---

### Question 8

Pick the best pipeline and report two high-purity clusters with genres, purity, and representative games.

**Answer:**

We selected MiniLM + SVD(50) + Agglomerative because it performed the best in Task 1 and shows strong genre purity.

**Cluster 3 (13 games, 100% pure)**

Top genres: Action (100%), Massively Multiplayer (38%), Indie (23%)

| Game             | Genres                             |
| ---------------- | ---------------------------------- |
| Counter-Strike 2 | Action, Massively Multiplayer      |
| Dota 2           | Action, Massively Multiplayer, RPG |
| Team Fortress 2  | Action, Massively Multiplayer      |

These are competitive online multiplayer first-person-shooter games. All have the Action tag.

**Cluster 1 (62 games, 85% pure)**

Top genres: Action (85%), Adventure (56%), RPG (44%)

| Game          | Genres                   |
| ------------- | ------------------------ |
| Dying Light   | Action, RPG              |
| Black Mesa    | Action, Adventure, Indie |
| Borderlands 3 | Action, RPG              |

These are mostly narrative-driven action games with RPG elements.

---

## Task 3 - Held-Out Game Profiling and Theme Discovery

### Question 9

Report assigned cluster ID, top 3 genres, and 3 representative games.

**Answer:**

The best Task 2 pipeline is MiniLM + SVD + Agglomerative.

**Cluster ID:** 4

Top genres: Action (80%), Adventure (55%), RPG (35%)

| Game          | Genres                   |
| ------------- | ------------------------ |
| Dying Light   | Action, RPG              |
| Black Mesa    | Action, Adventure, Indie |
| Borderlands 3 | Action, RPG              |

This is genre-estimation, because the games themselves are multi-label. Rather than predicting one genre, we assign the held-out game to its nearest cluster and inherit that cluster's genre distribution. The held-out game shares Action / Adventure / RPG characteristics because it clusters semantically with games having those tags.

---

### Question 10

For negative reviews: report 3-5 clusters with top terms, exemplar reviews, and labels.

**Answer:**

We used MiniLM + SVD(50) + Agglomerative because it achieved the best performance in Task 1 (V-Measure 0.648, ARI 0.742) and Task 2 (up to 100% genre purity).

**Cluster 0 - Boss-Focused Gameplay Issues (48 reviews)**

Terms: game, like, just, boss, fun, games, combat, boring, good, don

> "First of all it's a beautiful game. The visuals are great, It also has some unique combat that when you get used to it, it's enjoyable. But beyond that...this game is a boss simulator. I'm sure a lot of people will enjoy that aspect. I was hoping for more exploration. The game is linear, which isn't bad in itself, but it's boss fight after boss fight after boss fight..."

> "Is a fun game but with a lot of problems. First of all performance is not great and not consistent at all through the chapters. There's a beautiful world to look at but full of invisible walls that takes me out of the immersion..."

Label: Boss-focused gameplay issues

---

**Cluster 1 - Frustrating Combat & Boss Design (21 reviews)**

Terms: game, boss, like, just, bosses, combat, design, really, attacks, feel

> "The other negative reviews have already covered most of it. I can just add that I too find the game very frustrating. I want to love it, but the poor combat design, getting locked into animations far too often, which punishes you constantly..."

> "I really wanted to like this game, but it falls victim to many poor design decisions and I found the game far more tedious and frustrating than it should be..."

Label: Frustrating Combat and Boss Design

---

**Cluster 2 - Misogynistic Developers Criticism (6 reviews)**

Terms: developers misogynists, developers, country, misogynists, taiwan, taiwan country, game, want, talking, feets

> "If you want to support a super misogynistic games company then this is the game for you"

> "The developers are misogynists"

Label: Misogynistic Developers Criticism

---

**Cluster 3 - Wukong Game Criticism (8 reviews)**

Terms: game, boss, just, like, wukong, story, design, level, black, black myth

> "Do be aware that the majority of the reviews for Wukong are from China, even those in English, and as such this review will probably trigger some..."

> "Is Black Myth: Wukong the best game ever made and ever will be made? ...... A sudden uproar of positive feedback for this game brings me here..."

Label: Wukong game criticism

---

**Cluster 4 - Performance Issues (17 reviews)**

Terms: game, fps, low, crashes, settings, issues, like, amd, 7900xtx, low fps

> "I wanted to share my recent experience with this where I have seen that's been getting positive reviews. I'm running this on an i7 13th gen, RTX 4080, and 32 gigs RAM, which should be more than capable. However, the game crashes consistently..."

> "Not really sure what these purchased reviews are all about. Game runs terribly.1440p, Cinematic, 4090, 32GB Ram, 13900kConstant stutters."

Label: Performance Issues

---

### Question 11

For positive reviews: repeat with praise clusters.

**Answer:**

We used MiniLM + SVD(50) + Agglomerative (same logic as Q10).

**Cluster 0 - Positive Gameplay Experience (49 reviews)**

Terms: game, like, just, good, games, really, graphics, combat, 10, play

> "This game has the big awesome set pieces and aesthetic from the old school God of War games, the core gameplay of a Soul-like (albeit a tad more forgiving), really sick and fast paced Sekiro / Bloodborne combat..."

> "(Review in progress, 8h so far, 2 chapters completed)This game surprised me more than I expected. The level of polish and care put into it is quite impressive and I absolutely loved my time with it so far..."

Label: Positive gameplay experience

---

**Cluster 1 - Mythical Chinese Game Excellence (15 reviews)**

Terms: game, wukong, chinese, myth, good, black, black myth, games, myth wukong, like

> "Black Myth: Wukong was released under a lot of anticipation after gameplay showcases had been shown over the past few years, leading people to even question if the game would really come out like that. Thankfully, it really did – and overcame all expectations..."

> "Black Myth Wukong is an absolute masterclass of a game. All the hype, all the praise, all the positive buzz surrounding this game...they're absolutely deserved..."

Label: Mythical Chinese Game Excellence

---

**Cluster 2 - Perfect 10s and Sweet Baby Praise (16 reviews)**

Terms: 10, sweet baby, baby, sweet, ign, game, add, monke, ape, game ape

> "※This game does not include Sweet Baby"

> "Story: 10/10Gameplay: 10/10Graphic & Sound: 10/10Animation: 10/10Game's World: 10/10Character: 10/10Combat: 10/10Enemy's AI: 10/10Soundtrack: 10/10Game Optimize: 10/10It's Wukong: 10/10OVERALL: 10+/10"

Label: Perfect 10s and Sweet Baby praise

---

**Cluster 3 - Grind and Repetitive Gameplay (15 reviews)**

Terms: good, grind, just, bad, 10, long, average, bugs, game, hard

> "---{ Graphics }---☑ You forget what reality is☑ Beautiful☑ Good☑ Decent☑ Bad☑ Don't look too long at it☐ MS-DOS---{ Gameplay }---☑ Very good☐ Good☐ It's just gameplay☐ Mehh☐ Watch paint dry instead☐ Just don't..."

Label: Grind and repetitive gameplay

---

**Cluster 4 - Monke Embrace Approved (5 reviews)**

Terms: monke, reject, approves, monke approves, embrace, embrace monke, modernityreturn monke, modernityreturn, reject modernityreturn, dei

> "MONKE APPROVES..."

> "REJECT MODERNITYRETURN TO MONKE"

Label: Monke Embrace Approved

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

Terms: game, like, just, boss, fun, games, combat, boring, good, don
Exemplar: "First of all it's a beautiful game. The visuals are great, It also has some unique combat that when you get used to it, it's enjoyable. But beyond that...this game is a boss simulator..."

LLM output: "Boss-focused gameplay issues"

**Example 2 (Negative Cluster 4)**

Terms: game, fps, low, crashes, settings, issues, like, amd, 7900xtx, low fps
Exemplar: "I'm running this on an i7 13th gen, RTX 4080, and 32 gigs RAM... the game crashes consistently..."

LLM output: "Performance Issues"

**Example 3 (Positive Cluster 1)**

Terms: game, wukong, chinese, myth, good, black, black myth, games, myth wukong, like
Exemplar: "Black Myth: Wukong was released under a lot of anticipation..."

LLM output: "Mythical Chinese Game Excellence"
