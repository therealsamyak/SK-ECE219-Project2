# Part 3 - Clustering using both Image and Text

---

## Question 20

Try to construct various text queries regarding types of Pokemon to find the relevant images from the dataset. Once you have found the most suitable template for queries, please find the top five most relevant Pokemon for type Bug, Fire and Grass. Repeat this process for Pokemon of Dark and Dragon types.

#### Answer

**Query Template:** `"a photo of a {type} type Pokemon"`

This template proved most effective because it follows natural language patterns that CLIP was trained on during its internet-scale pretraining. The phrase "a photo of" matches common image captioning formats, and "type Pokemon" provides clear semantic context.

**Top 5 Retrieved Pokemon per Type:**

| Type | Top 5 Pokemon |
|------|---------------|
| Bug | Yanmega, Ariados, Yanma, Butterfree, Anorith |
| Fire | Simisear, Magmar, Magmortar, Quilava, Delphox |
| Grass | Oddish, Tropius, Bulbasaur, Leafeon, Bellsprout |
| Dark | Umbreon, Darkrai, Pincurchin, Gastly, Salandit |
| Dragon | Dragonite, Rhydon, Druddigon, Nidorino, Kingdra |

**Visualizations:** `outputs/part3_q20_bug.png`, `outputs/part3_q20_fire.png`, `outputs/part3_q20_grass.png`, `outputs/part3_q20_dark.png`, `outputs/part3_q20_dragon.png`

**Assessment of Query Effectiveness:**

- **Bug, Fire, Grass:** These types show strong visual coherence. Fire-type Pokemon often have warm colors (red, orange, yellow) and flame-like features. Grass-types share green coloration and plant-like features. Bug-types have insectoid body structures that CLIP easily recognizes.

- **Dark and Dragon:** These types show more variability. Dark-type visual identity is less consistent—it encompasses shadowy creatures (Umbreon, Darkrai) but also Pokemon that simply don't fit other categories. Dragon-types are visually diverse, ranging from serpentine (Kingdra) to dinosaur-like (Dragonite). The retrieval of Rhydon (Ground/Rock) and Nidorino (Poison) for Dragon queries suggests CLIP associates dragon-like features (horns, bulky bodies) beyond the official type classification.

The key difference is that Bug, Fire, and Grass have strong visual-semantic correlations that match common natural language descriptions, while Dark and Dragon are more abstract game mechanics concepts with weaker visual signatures.

---

## Question 21

Randomly select 10 Pokemon images from the dataset and use CLIP to find the most relevant types. For each selected Pokemon, please plot it and indicate its name, types, and the five most relevant types predicted by CLIP.

#### Answer

**Random Seed:** 42 (for reproducibility)

**Selected Pokemon:** Cosmog, Ledyba, Zubat, Wynaut, Spoink, Makuhita, Quagsire, Meganium, Thievul, Espurr

**Visualization:** `outputs/part3_q21_predictions.png`

**Detailed Predictions:**

| Pokemon | Actual Type | Top-1 Prediction | Top-5 Predictions (with scores) | Top-1 Correct? | Top-5 Correct? |
|---------|-------------|------------------|--------------------------------|----------------|----------------|
| Cosmog | Psychic | Psychic (29.8%) | Psychic, Dark, Fairy, Normal, Poison | ✓ | ✓ |
| Ledyba | Bug/Flying | Bug (31.1%) | Bug, Normal, Steel, Fighting, Dark | ✓ | ✓ |
| Zubat | Poison/Flying | Dark (41.5%) | Dark, Flying, Normal, Fighting, Fairy | ✗ | ✗ |
| Wynaut | Psychic | Ice (17.0%) | Ice, Normal, Psychic, Water, Ghost | ✗ | ✓ |
| Spoink | Psychic | Normal (19.6%) | Normal, Dark, Psychic, Poison, Fighting | ✗ | ✓ |
| Makuhita | Fighting | Fighting (50.4%) | Fighting, Dark, Normal, Psychic, Electric | ✓ | ✓ |
| Quagsire | Water/Ground | Normal (33.4%) | Normal, Water, Psychic, Ground, Ice | ✗ | ✓ |
| Meganium | Grass | Dragon (33.5%) | Dragon, Normal, Grass, Psychic, Poison | ✗ | ✓ |
| Thievul | Dark | Dark (38.9%) | Dark, Normal, Psychic, Ghost, Fairy | ✓ | ✓ |
| Espurr | Psychic | Psychic (29.7%) | Psychic, Normal, Ghost, Dark, Fairy | ✓ | ✓ |

**Summary:**
- Top-1 Accuracy: 3/10 = 30%
- Top-5 Accuracy: 9/10 = 90%

**Observations:**

CLIP struggles with visually ambiguous Pokemon. Zubat (Poison) gets misclassified as Dark—likely because its purple coloration and bat-like appearance match the Dark-type aesthetic more than Poison. Meganium (Grass) gets classified as Dragon, probably due to its large, dinosaur-like body and neck features that resemble dragon aesthetics. The model performs best when the Pokemon's visual appearance strongly correlates with its type (e.g., Makuhita's fighting stance, Ledyba's insect wings).

---

## Question 22

Using the ground-truth primary type label Type1, report Accuracy@1 and Hit@5 for all Pokemon images.

#### Answer

**Dataset:** 754 Pokemon (distinct IDs, using 0.jpg image only)

**Candidate Types:** 18 types (Bug, Dark, Dragon, Electric, Fairy, Fighting, Fire, Flying, Ghost, Grass, Ground, Ice, Normal, Poison, Psychic, Rock, Steel, Water)

**Results:**

| Metric | Value |
|--------|-------|
| **CLIP Acc@1** | 32.76% (247/754 correct) |
| **CLIP Hit@5** | 73.34% (553/754 correct type in top-5) |
| Total Pokemon | 754 |

**Per-Type Accuracy:**

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

**Best Performing Types:** Fire (68.18%), Dark (66.67%), Ice (54.17%), Dragon (50.00%)

**Worst Performing Types:** Flying (0.00%), Ground (3.70%), Psychic (10.87%), Poison (13.79%)

**Why Acc@1 is Low but Hit@5 is Reasonably Good:**

The gap between Acc@1 (32.76%) and Hit@5 (73.34%) confirms that CLIP frequently retrieves the correct type in its top candidates but doesn't reliably rank it first. This happens because:

1. **Dual-encoder limitation:** CLIP independently encodes images and text into a shared space using contrastive learning. Predictions are made by nearest-neighbor similarity without any reasoning step.

2. **Visual ambiguity:** Many Pokemon share visual features across types. A Water-type might look similar to Ice-type (both blue), or a Grass-type might resemble Bug-type (both green with plant/insect features).

3. **Prompt sensitivity:** The template "a photo of a {type} type Pokemon" may not optimally separate visually similar type concepts in CLIP's embedding space.

4. **Game mechanics vs. visual reality:** Pokemon types are game mechanics, not purely visual categories. Flying-type Pokemon have diverse appearances (birds, dragons, insects) with no unifying visual signature—hence 0% accuracy.

---

## Question 23

VLM Reranking of CLIP Top-5 Candidates. Report Reranked Accuracy@1 and a comparison table.

#### Answer

**VLM Model:** Qwen3-VL-2B-Instruct

**Reranking Protocol:**
1. For each Pokemon, CLIP provides top-5 candidate types
2. VLM receives the image and candidate list with prompt: "This is a Pokemon. From these types: {top5}, which is the most likely primary type? Return only the type name as JSON: {"type1": "..."}"
3. If VLM output is invalid (not in candidate list), fall back to CLIP's top-1

**Results:**

| Method | Acc@1 | Hit@5 |
|--------|-------|-------|
| CLIP | 32.76% | 73.34% |
| VLM Rerank | 43.50% | N/A |
| **Improvement** | **+10.74%** | - |

**Statistics:**
- Total Pokemon: 754
- VLM Fallbacks: 0 (all responses parsed successfully)

**Does VLM Reranking Help?**

Yes, significantly. VLM reranking improves Acc@1 from 32.76% to 43.50%, a **10.74 percentage point improvement (32.8% relative gain)**.

**Why VLM Helps:**

1. **Cross-attention reasoning:** Unlike CLIP's independent encoding, the VLM attends jointly over image tokens and text tokens. It can compare the Pokemon's visual features against each candidate type description and make an informed selection.

2. **Discrete decision-making:** CLIP produces continuous similarity scores that can be noisy at the decision boundary. The VLM makes an explicit, discrete choice from the candidate set, forcing it to commit to one answer.

3. **Instruction following:** The VLM understands the task (select the most likely type) and can apply world knowledge about Pokemon characteristics that may not be captured in CLIP's contrastive training.

4. **Constrained candidate set:** By restricting the VLM to CLIP's top-5 candidates, we leverage CLIP's broad retrieval capability while using the VLM's reasoning to refine the ranking. Since CLIP's Hit@5 is 73.34%, the correct answer is available to the VLM in nearly 3 out of 4 cases.

**Limitations:**

The VLM is still constrained by CLIP's candidate pool. For the 26.66% of Pokemon where the correct type isn't in the top-5, the VLM cannot recover the correct answer. Additionally, VLM inference is significantly slower than CLIP (0.6s vs 0.01s per image on MPS), making it more suitable for post-hoc refinement than real-time prediction.

---

## Summary

| Question | Key Finding |
|----------|-------------|
| Q20 | CLIP retrieves visually consistent Pokemon for type queries; effectiveness varies by type visual coherence |
| Q21 | 30% top-1 accuracy, 90% top-5 accuracy on random 10 Pokemon |
| Q22 | Full dataset: 32.76% Acc@1, 73.34% Hit@5—CLIP finds correct type often but not reliably first |
| Q23 | VLM reranking improves Acc@1 to 43.50% (+10.74%), demonstrating value of cross-attention reasoning |
