# Part 3 - Clustering using both Image and Text

> **Note:** We use the `open_clip` library instead of the official OpenAI `clip` library. Both load the same ViT-L-14 model with identical OpenAI pretrained weights, but `open_clip` is an easily installable Python package (`pip install open-clip-torch`) with no additional setup required.

## Question 20

Try to construct various text queries regarding types of Pokemon (such as "type: Bug", "electric type Pokémon" or "Pokémon with fire abilities") to find the relevant images from the dataset. Once you have found the most suitable template for queries, please find the top five most relevant Pokemon for type Bug, Fire and Grass. For each of the constructed query, please plot the five most relevant Pokemon horizontally in one figure with following specifications:

- the title of the figure should be the query you used;
- the title of each Pokemon should be the name of the Pokemon and its first and second type.

Repeat this process for Pokemon of Dark and Dragon types. Assess the effectiveness of your queries in these cases as well and try to explain any differences.

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

Randomly select 10 Pokemon images from the dataset and use CLIP to find the most relevant types (use your preferred template, e.g "type: Bug"). For each selected Pokemon, please plot it and indicate:

- its name and first and second type;
- the five most relevant types predicted by CLIP and their predicted similarities.

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

In this question, reuse the exact same CLIP setup from the previous question on all the Pokemon images (same model, same type prompt template, and the same set of candidate types), but instead of returning only the most relevant type, return the top-5 most relevant types for each Pokemon image. Using the ground-truth primary type label Type1, report:

- **Accuracy@1 (Acc@1):** the fraction of images whose top-1 predicted type matches the ground-truth primary type (Type1).
- **Hit@5 (a.k.a. Recall@5):** the fraction of images whose ground-truth primary type appears anywhere in the top-5 predicted types.

In practice, you will likely observe that CLIP's Acc@1 for predicting the primary type (Type1) is relatively low, while Hit@5 is often reasonably good. This gap suggests that CLIP frequently retrieves the correct type somewhere in a short candidate list, but does not reliably rank it as the top-1 prediction. A key reason is that CLIP is a dual-encoder model trained with a contrastive objective: it independently embeds images and text prompts into a shared vector space, and predictions are made purely by embedding similarity (e.g., dot-product/cosine similarity). This is powerful for coarse semantic alignment, but it lacks an explicit reasoning step and can be sensitive to prompt wording and visually similar concepts.

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

VLM Reranking of CLIP Top-5 Candidates. Reuse the exact same CLIP setup and evaluation protocol from the previous question. For each Pokemon image, first use CLIP to obtain the top-5 predicted types (and their probabilities/similarities). Then, use a modern vision-language model (VLM), e.g., Qwen/Qwen3-VL-2B-Instruct, to select the single most likely primary type from only these five candidates. You can check the helper code for how to use Qwen vision language models (please use Google-colab or if you have your local pc with GPU)

Concretely, for each image $i$, let CLIP return a candidate set $C_i$ containing the top-5 types. Prompt the VLM with the image and the candidate list $C_i$, and force the VLM to output exactly one type from $C_i$ (e.g., as a JSON field `{"type1": "..."}`). If the VLM output is invalid (not in $C_i$), fall back to CLIP's top-1 type.

Report:

- **Reranked Accuracy@1:** the fraction of images whose final predicted type (after VLM selection) matches the ground-truth Type1.
- A comparison table of CLIP Acc@1, CLIP Hit@5, and VLM-reranked Acc@1.
- Briefly discuss: Does VLM reranking help?

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
