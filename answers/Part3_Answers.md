# Part 3 - Clustering using both Image and Text

> **Note:** We use the `open_clip` library instead of the official OpenAI `clip` library. Both load the same ViT-L-14 model with identical OpenAI pretrained weights, but `open_clip` is an easily installable Python package (`pip install open-clip-torch`) with no additional setup required.

## Question 20

Construct text queries to find Pokemon images by type. Report top 5 for Bug, Fire, Grass, Dark, and Dragon.

**Answer:**

**Query Template:** `"a photo of a {type} type Pokemon"`

This worked best because CLIP was trained on internet image-text pairs, and "a photo of" matches typical captions.

**Top 5 per type:**

| Type   | Top 5 Pokemon                                   |
| ------ | ----------------------------------------------- |
| Bug    | Yanmega, Ariados, Yanma, Butterfree, Anorith    |
| Fire   | Simisear, Magmar, Magmortar, Quilava, Delphox   |
| Grass  | Oddish, Tropius, Bulbasaur, Leafeon, Bellsprout |
| Dark   | Umbreon, Darkrai, Pincurchin, Gastly, Salandit  |
| Dragon | Dragonite, Rhydon, Druddigon, Nidorino, Kingdra |

Files: [Bug](outputs/Q20_bug.png), [Fire](outputs/Q20_fire.png), [Grass](outputs/Q20_grass.png), [Dark](outputs/Q20_dark.png), [Dragon](outputs/Q20_dragon.png)

Bug, Fire, and Grass types retrieved pretty coherent results. Fire Pokemon tend to have warm colors (red, orange, yellow) and flame-like features. Grass types are mostly green with plant-like body parts. Bug types have insectoid body structures that CLIP picks up on easily.

Dark and Dragon were trickier. Dark-type visual identity isn't very consistent—it includes shadowy creatures like Umbreon and Darkrai, but also Pokemon that just don't fit other categories. Dragon types are visually all over the place, from serpentine (Kingdra) to dinosaur-like (Dragonite). The fact that Rhydon (Ground/Rock) and Nidorino (Poison) showed up for Dragon queries suggests CLIP associates "dragon-like features" (horns, bulky bodies) with the Dragon type concept, regardless of the official type classification.

The main difference: Bug, Fire, and Grass have strong visual-semantic patterns that match common language descriptions. Dark and Dragon are more abstract game mechanics with weaker visual signatures.

---

## Question 21

Randomly select 10 Pokemon. For each, plot and show predicted types.

**Answer:**

Seed: 42

Pokemon: Cosmog, Ledyba, Zubat, Wynaut, Spoink, Makuhita, Quagsire, Meganium, Thievul, Espurr

File: [predictions](outputs/Q21_predictions.png)

| Pokemon  | Actual        | Top-1            | Top-5                                     | Correct? |
| -------- | ------------- | ---------------- | ----------------------------------------- | -------- |
| Cosmog   | Psychic       | Dark (73.68%)    | Dark, Psychic, Electric, Normal, Poison   | Top-5    |
| Ledyba   | Bug/Flying    | Bug (78.64%)     | Bug, Normal, Fighting, Steel, Rock        | Yes      |
| Zubat    | Poison/Flying | Dark (57.62%)    | Dark, Normal, Fighting, Flying, Bug       | No       |
| Wynaut   | Psychic       | Normal (22.44%)  | Normal, Water, Psychic, Dark, Ice         | Top-5    |
| Spoink   | Psychic       | Dark (31.37%)    | Dark, Normal, Psychic, Fighting, Steel    | Top-5    |
| Makuhita | Fighting      | Fighting (59.52%)| Fighting, Dark, Normal, Psychic, Electric | Yes      |
| Quagsire | Water/Ground  | Normal (24.69%)  | Normal, Water, Psychic, Ice, Ground       | Top-5    |
| Meganium | Grass         | Dragon (22.35%)  | Dragon, Poison, Psychic, Normal, Water    | No       |
| Thievul  | Dark          | Dark (47.24%)    | Dark, Fire, Fighting, Normal, Psychic     | Yes      |
| Espurr   | Psychic       | Dark (65.42%)    | Dark, Normal, Psychic, Ghost, Fairy       | Top-5    |

Top-1: 30%, Top-5: 70%

CLIP struggles with visually ambiguous Pokemon. Zubat (Poison) gets called Dark—purple bat matches Dark-type look. Meganium (Grass) gets called Dragon because its dinosaur body looks more "dragon" than "plant."

---

## Question 22

Report Accuracy@1 and Hit@5 for all Pokemon using Type1 ground truth.

**Answer:**

Dataset: 754 Pokemon, 18 candidate types

| Metric     | Value            |
| ---------- | ---------------- |
| CLIP Acc@1 | 33.29% (251/754) |
| CLIP Hit@5 | 74.27% (560/754) |

Per-type Acc@1:

| Type     | Acc@1  | Type     | Acc@1  |
| -------- | ------ | -------- | ------ |
| Dark     | 76.67% | Bug      | 44.93% |
| Ice      | 58.33% | Rock     | 41.46% |
| Fire     | 54.55% | Dragon   | 36.36% |
| Fairy    | 50.00% | Normal   | 23.91% |
| Water    | 46.23% | Poison   | 24.14% |
| Fighting | 46.88% | Electric | 18.42% |
| Steel    | 45.83% | Psychic  | 10.87% |
|          |        | Grass    | 7.59%  |
|          |        | Ghost    | 4.00%  |
|          |        | Ground   | 3.70%  |
|          |        | Flying   | 0.00%  |

Best: Dark (77%), Ice (58%), Fire (55%), Fairy (50%)
Worst: Flying (0%), Ground (4%), Ghost (4%), Grass (8%)

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

| Method      | Acc@1   | Hit@5  |
| ----------- | ------- | ------ |
| CLIP        | 33.29%  | 74.27% |
| VLM Rerank  | 44.96%  | —      |
| Improvement | +11.67% | —      |

VLM fallbacks: 1

Does VLM reranking help?

Yes. VLM reranking bumps Acc@1 from 33.29% to 44.96%, an 11.67 percentage point improvement (35.1% relative gain).

Why VLM helps:

1. **Cross-attention reasoning:** Unlike CLIP's independent encoding, the VLM attends jointly over image tokens and text tokens. It can compare the Pokemon's visual features against each candidate type description and make an informed selection.

2. **Discrete decision-making:** CLIP produces continuous similarity scores that can be noisy at the decision boundary. The VLM makes an explicit, discrete choice from the candidate set, forcing it to commit to one answer.

3. **Instruction following:** The VLM understands the task (select the most likely type) and can apply world knowledge about Pokemon characteristics that CLIP's contrastive training might not capture.

4. **Constrained candidate set:** By restricting the VLM to CLIP's top-5 candidates, we use CLIP's broad retrieval capability while letting the VLM refine the ranking. Since CLIP's Hit@5 is 74.27%, the correct answer is available to the VLM in nearly 3 out of 4 cases.

Limitations:

The VLM is still constrained by CLIP's candidate pool. For the 25.73% of Pokemon where the correct type isn't in the top-5, the VLM can't recover the right answer. Also, VLM inference is significantly slower than CLIP (0.6s vs 0.01s per image on MPS), so it's better suited for post-hoc refinement than real-time prediction.
