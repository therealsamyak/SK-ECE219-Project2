# Part 3 - Clustering using both Image and Text

> **Note:** We use the `open_clip` library instead of the official OpenAI `clip` library. Both load the same ViT-L-14 model with identical OpenAI pretrained weights, but `open_clip` is an easily installable Python package (`pip install open-clip-torch`) with no additional setup required.

## Question 20

Construct text queries to find Pokemon images by type. Report top 5 for Bug, Fire, Grass, Dark, and Dragon.

**Answer:**

**Queries Tried:**

- `"type: {type}"`
- `"{type} type Pokemon"`
- `"Pokemon with {type} abilities"`
- `"a photo of a {type} type Pokemon"`

**Best one:** `"a photo of a {type} type Pokemon"`. This is likely the best due to most internet photo captions being in this format.

**Top 5 per type:**

| Type   | Top 5 Pokemon                                   |
| ------ | ----------------------------------------------- |
| Bug    | Yanmega, Ariados, Yanma, Butterfree, Anorith    |
| Fire   | Simisear, Magmar, Magmortar, Quilava, Delphox   |
| Grass  | Oddish, Tropius, Bulbasaur, Leafeon, Bellsprout |
| Dark   | Umbreon, Darkrai, Pincurchin, Gastly, Salandit  |
| Dragon | Dragonite, Rhydon, Druddigon, Nidorino, Kingdra |

![Bug type](outputs/Q20_bug.png)
![Fire type](outputs/Q20_fire.png)
![Grass type](outputs/Q20_grass.png)
![Dark type](outputs/Q20_dark.png)
![Dragon type](outputs/Q20_dragon.png)

Bug, Fire, and Grass types retrieved pretty coherent results. Fire Pokemon tend to have warm colors (red, orange, yellow) and flame-like features. Grass types are mostly green with plant-like body parts. Bug types have insectoid body structures that CLIP picks up on easily. However, Dark and Dragon were trickier. Dark-type visual identity isn't very consistent. Dragon types are visually all over the place, from serpentine (Kingdra) to dinosaur-like (Dragonite). The fact that Rhydon (Ground/Rock) and Nidorino (Poison) showed up for Dragon queries suggests CLIP associates "dragon-like features" (horns, bulky bodies) with the Dragon type.

The performance difference matches the uniformity of appearances within the classes: Bug, Fire, and Grass have clear visual patterns that match how people describe them, and Dark and Dragon are more abstract categories with weaker visual cues.

---

## Question 21

Randomly select 10 Pokemon. For each, plot and show predicted types.

**Answer:**

Seed: 42

Pokemon: Cosmog, Ledyba, Zubat, Wynaut, Spoink, Makuhita, Quagsire, Meganium, Thievul, Espurr

![Predictions](outputs/Q21_predictions.png)

| Pokemon  | Actual        | Top-1             | Top-5                                     | Correct? |
| -------- | ------------- | ----------------- | ----------------------------------------- | -------- |
| Cosmog   | Psychic       | Dark (73.68%)     | Dark, Psychic, Electric, Normal, Poison   | Top-5    |
| Ledyba   | Bug/Flying    | Bug (78.64%)      | Bug, Normal, Fighting, Steel, Rock        | Yes      |
| Zubat    | Poison/Flying | Dark (57.62%)     | Dark, Normal, Fighting, Flying, Bug       | No       |
| Wynaut   | Psychic       | Normal (22.44%)   | Normal, Water, Psychic, Dark, Ice         | Top-5    |
| Spoink   | Psychic       | Dark (31.37%)     | Dark, Normal, Psychic, Fighting, Steel    | Top-5    |
| Makuhita | Fighting      | Fighting (59.52%) | Fighting, Dark, Normal, Psychic, Electric | Yes      |
| Quagsire | Water/Ground  | Normal (24.69%)   | Normal, Water, Psychic, Ice, Ground       | Top-5    |
| Meganium | Grass         | Dragon (22.35%)   | Dragon, Poison, Psychic, Normal, Water    | No       |
| Thievul  | Dark          | Dark (47.24%)     | Dark, Fire, Fighting, Normal, Psychic     | Yes      |
| Espurr   | Psychic       | Dark (65.42%)     | Dark, Normal, Psychic, Ghost, Fairy       | Top-5    |

Top-1: 30%, Top-5: 70%

The results line up with what we found in Q20. For example, Zubat (Poison) gets called Dark type because its apperance of a purple bat matches Dark-type look. Similarly, Meganium (Grass) gets called Dragon because its dinosaur body looks more "dragon" than "plant."

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

---

## Question 23

VLM Reranking of CLIP Top-5. Report Reranked Acc@1.

**Answer:**

Model: Qwen3-VL-2B-Instruct

| Method      | Acc@1   | Hit@5  |
| ----------- | ------- | ------ |
| CLIP        | 33.29%  | 74.27% |
| VLM Rerank  | 44.96%  | —      |
| Improvement | +11.67% | —      |

Files: [metrics](outputs/Q23_metrics.json), [detailed](outputs/Q23_detailed.json)

Yes, VLM reranking helps. It bumps Acc@1 from 33.29% to 44.96%, an 11.67 percentage point improvement. The VLM works because it can attend jointly over the image and the candidate type names, whereas CLIP encodes them separately and just compares similarity scores. Since CLIP already finds the right type somewhere in the top 5 about 74% of the time, the VLM just needs to pick the best one from that list. It's basically a second opinion that's better at the final decision. Only 1 case needed a fallback to CLIP's top-1.

The major drawback is that VLM can't recover the right answer if CLIP's top-5 never included it in the first place. VLM inference is also much slower (0.6s per image vs CLIP's 0.01s), making this approach less ideal in real-time classification scenarios.
