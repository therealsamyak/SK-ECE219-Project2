# Part 3 - Clustering using both Image and Text

## Question 20

Construct text queries to find Pokemon images by type. Report top 5 for Bug, Fire, Grass, Dark, and Dragon.

**Answer:**

**Query Template:** `"a photo of a {type} type Pokemon"`

This worked best because CLIP was trained on internet image-text pairs, and "a photo of" matches typical captions.

**Top 5 per type:**

| Type | Top 5 Pokemon |
|------|---------------|
| Bug | Yanmega, Ariados, Yanma, Butterfree, Anorith |
| Fire | Simisear, Magmar, Magmortar, Quilava, Delphox |
| Grass | Oddish, Tropius, Bulbasaur, Leafeon, Bellsprout |
| Dark | Umbreon, Darkrai, Pincurchin, Gastly, Salandit |
| Dragon | Dragonite, Rhydon, Druddigon, Nidorino, Kingdra |

Files: [Bug](outputs/part3_q20_bug.png), [Fire](outputs/part3_q20_fire.png), [Grass](outputs/part3_q20_grass.png), [Dark](outputs/part3_q20_dark.png), [Dragon](outputs/part3_q20_dragon.png)

Bug, Fire, and Grass types retrieved pretty coherent results. Fire Pokemon tend to have warm colors (red, orange, yellow) and flame-like features. Grass types are mostly green with plant-like body parts. Bug types have insectoid body structures that CLIP picks up on easily.

Dark and Dragon were trickier. Dark-type visual identity isn't very consistent—it includes shadowy creatures like Umbreon and Darkrai, but also Pokemon that just don't fit other categories. Dragon types are visually all over the place, from serpentine (Kingdra) to dinosaur-like (Dragonite). The fact that Rhydon (Ground/Rock) and Nidorino (Poison) showed up for Dragon queries suggests CLIP associates "dragon-like features" (horns, bulky bodies) with the Dragon type concept, regardless of the official type classification.

The main difference: Bug, Fire, and Grass have strong visual-semantic patterns that match common language descriptions. Dark and Dragon are more abstract game mechanics with weaker visual signatures.

---

## Question 21

Randomly select 10 Pokemon. For each, plot and show predicted types.

**Answer:**

Seed: 42

Pokemon: Cosmog, Ledyba, Zubat, Wynaut, Spoink, Makuhita, Quagsire, Meganium, Thievul, Espurr

File: [predictions](outputs/part3_q21_predictions.png)

| Pokemon | Actual | Top-1 | Top-5 | Correct? |
|---------|--------|-------|-------|----------|
| Cosmog | Psychic | Psychic (29.8%) | Psychic, Dark, Fairy, Normal, Poison | Yes |
| Ledyba | Bug/Flying | Bug (31.1%) | Bug, Normal, Steel, Fighting, Dark | Yes |
| Zubat | Poison/Flying | Dark (41.5%) | Dark, Flying, Normal, Fighting, Fairy | No |
| Wynaut | Psychic | Ice (17.0%) | Ice, Normal, Psychic, Water, Ghost | Top-5 |
| Spoink | Psychic | Normal (19.6%) | Normal, Dark, Psychic, Poison, Fighting | Top-5 |
| Makuhita | Fighting | Fighting (50.4%) | Fighting, Dark, Normal, Psychic, Electric | Yes |
| Quagsire | Water/Ground | Normal (33.4%) | Normal, Water, Psychic, Ground, Ice | Top-5 |
| Meganium | Grass | Dragon (33.5%) | Dragon, Normal, Grass, Psychic, Poison | Top-5 |
| Thievul | Dark | Dark (38.9%) | Dark, Normal, Psychic, Ghost, Fairy | Yes |
| Espurr | Psychic | Psychic (29.7%) | Psychic, Normal, Ghost, Dark, Fairy | Yes |

Top-1: 30%, Top-5: 90%

CLIP struggles with visually ambiguous Pokemon. Zubat (Poison) gets called Dark—purple bat matches Dark-type look. Meganium (Grass) gets called Dragon because its dinosaur body looks more "dragon" than "plant."

---

## Question 22

Report Accuracy@1 and Hit@5 for all Pokemon using Type1 ground truth.

**Answer:**

Dataset: 754 Pokemon, 18 candidate types

| Metric | Value |
|--------|-------|
| CLIP Acc@1 | 32.76% (247/754) |
| CLIP Hit@5 | 73.34% (553/754) |

Per-type Acc@1:

| Type | Acc@1 | Type | Acc@1 |
|------|-------|------|-------|
| Fire | 68.18% | Bug | 36.23% |
| Dark | 66.67% | Steel | 45.83% |
| Ice | 54.17% | Normal | 31.52% |
| Dragon | 50.00% | Ghost | 28.00% |
| Fighting | 46.88% | Rock | 29.27% |
| Fairy | 40.00% | Grass | 18.99% |
| Water | 33.02% | Electric | 15.79% |
| | | Poison | 13.79% |
| | | Psychic | 10.87% |
| | | Ground | 3.70% |
| | | Flying | 0.00% |

Best: Fire (68%), Dark (67%), Ice (54%), Dragon (50%)
Worst: Flying (0%), Ground (4%), Psychic (11%), Poison (14%)

Why the gap between Acc@1 and Hit@5:

1. **Dual-encoder limitation:** CLIP encodes images and text independently into a shared space using contrastive learning. Predictions are made by nearest-neighbor similarity, without any reasoning step.

2. **Visual ambiguity:** Many Pokemon share visual features across types. A Water-type might look similar to Ice-type (both blue), or a Grass-type might look like Bug-type (both green with plant/insect features).

3. **Prompt sensitivity:** The template "a photo of a {type} type Pokemon" might not optimally separate visually similar type concepts in CLIP's embedding space.

4. **Game mechanics vs. visual reality:** Pokemon types are game mechanics, not purely visual categories. Flying-type Pokemon have diverse appearances (birds, dragons, insects) with no unifying visual look, hence 0% accuracy.

---

## Question 23

VLM Reranking of CLIP Top-5. Report Reranked Acc@1.

**Answer:**

Model: Qwen3-VL-2B-Instruct

Protocol:
1. CLIP provides top-5 candidates
2. VLM receives image + candidates, prompted to select one type as JSON
3. Invalid output falls back to CLIP top-1

| Method | Acc@1 | Hit@5 |
|--------|-------|-------|
| CLIP | 32.76% | 73.34% |
| VLM Rerank | 43.50% | — |
| Improvement | +10.74% | — |

VLM fallbacks: 0

Does VLM reranking help?

Yes. VLM reranking bumps Acc@1 from 32.76% to 43.50%, a 10.74 percentage point improvement (32.8% relative gain).

Why VLM helps:

1. **Cross-attention reasoning:** Unlike CLIP's independent encoding, the VLM attends jointly over image tokens and text tokens. It can compare the Pokemon's visual features against each candidate type description and make an informed selection.

2. **Discrete decision-making:** CLIP produces continuous similarity scores that can be noisy at the decision boundary. The VLM makes an explicit, discrete choice from the candidate set, forcing it to commit to one answer.

3. **Instruction following:** The VLM understands the task (select the most likely type) and can apply world knowledge about Pokemon characteristics that CLIP's contrastive training might not capture.

4. **Constrained candidate set:** By restricting the VLM to CLIP's top-5 candidates, we use CLIP's broad retrieval capability while letting the VLM refine the ranking. Since CLIP's Hit@5 is 73.34%, the correct answer is available to the VLM in nearly 3 out of 4 cases.

Limitations:

The VLM is still constrained by CLIP's candidate pool. For the 26.66% of Pokemon where the correct type isn't in the top-5, the VLM can't recover the right answer. Also, VLM inference is significantly slower than CLIP (0.6s vs 0.01s per image on MPS), so it's better suited for post-hoc refinement than real-time prediction.
