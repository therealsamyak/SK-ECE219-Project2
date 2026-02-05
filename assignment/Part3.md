## Part 3 - Clustering using both image and text

In part 1 and part 2, we have practived the art of clustering text and images separately. However, can we map image and text to the same space? In the Pokemon world, Pokedex catalogs Pokemon's appearances and various metadata. We will build our Pokedex from image dataset link and meta metadata link. Fortunately, ECE 219 Gym kindly provides new Pokemon trainers with the helper code for data preprocessing and inferencing. Please find the code on Bruinlearn modules Week 4.

Each Pokémon may be represented by multiple images and up to two types (for example, Bulbasaur is categorized as both Grass and Poison types). In this section, we will focus on the first image (named 0.jpg) in each folder for our analysis.

We will use the pre-trained CLIP [?] to illustrate the idea of multimodal clustering. CLIP (Contrastive Language–Image Pretraining) is an innovative model developed by OpenAI, designed to understand and connect concepts from both text and images. CLIP is trained on a vast array of internet-sourced text-image pairs. This extensive training enables the model to understand a broad spectrum of visual concepts and their textual descriptions.

![CLIP training summary](Figure 1)

**Figure 1:** CLIP training summary

CLIP consists of two primary components: a text encoder and an image encoder. The text encoder processes textual data, converting sentences and phrases into numerical representations. Simultaneously, the image encoder transforms visual inputs into a corresponding set of numerical values. These encoders are trained to map both text and images into a shared embedding space, allowing the model to compare and relate the two different types of data directly. The training employs a contrastive learning approach, where the model learns to match corresponding text and image pairs against numerous non-matching pairs. This approach helps the model in accurately associating images with their relevant textual descriptions and vice versa.

**QUESTION 20:** Try to construct various text queries regarding types of Pokemon (such as "type: Bug", "electric type Pokémon" or "Pokémon with fire abilities") to find the relevant images from the dataset. Once you have found the most suitable template for queries, please find the top five most relevant Pokemon for type Bug, Fire and Grass. For each of the constructed query, please plot the five most relevant Pokemon horizontally in one figure with following specifications:

- the title of the figure should be the query you used;
- the title of each Pokemon should be the name of the Pokemon and its first and second type.

Repeat this process for Pokemon of Dark and Dragon types. Assess the effectiveness of your queries in these cases as well and try to explain any differences.

**QUESTION 21:** Randomly select 10 Pokemon images from the dataset and use CLIP to find the most relevant types (use your preferred template, e.g "type: Bug"). For each selected Pokemon, please plot it and indicate:

- its name and first and second type;
- the five most relevant types predicted by CLIP and their predicted similarities.

**QUESTION 22:** In this question, reuse the exact same CLIP setup from the previous question on all the Pokemon images (same model, same type prompt template, and the same set of candidate types), but instead of returning only the most relevant type, return the top-5 most relevant types for each Pokemon image. Using the ground-truth primary type label Type1, report:

- **Accuracy@1 (Acc@1):** the fraction of images whose top-1 predicted type matches the ground-truth primary type (Type1).
- **Hit@5 (a.k.a. Recall@5):** the fraction of images whose ground-truth primary type appears anywhere in the top-5 predicted types.

In practice, you will likely observe that CLIP's Acc@1 for predicting the primary type (Type1) is relatively low, while Hit@5 is often reasonably good. This gap suggests that CLIP frequently retrieves the correct type somewhere in a short candidate list, but does not reliably rank it as the top-1 prediction. A key reason is that CLIP is a dual-encoder model trained with a contrastive objective: it independently embeds images and text prompts into a shared vector space, and predictions are made purely by embedding similarity (e.g., dot-product/cosine similarity). This is powerful for coarse semantic alignment, but it lacks an explicit reasoning step and can be sensitive to prompt wording and visually similar concepts.

Modern vision-language models (VLMs) address this limitation by tokenizing the image: a vision encoder converts the image into a sequence of visual tokens, which are then fed alongside text tokens into a large language model [8]. The LLM attends jointly over image and text tokens and generates outputs autoregressively, enabling it to use additional context (such as a candidate list of types) and make a more discrete, instruction-following decision rather than relying solely on nearest-neighbor similarity. Figure 2 illustrates this architecture (we will use a lightweight VLM in the next question to rerank CLIP's top-5 candidate types and re-evaluate Acc@1).

![Qwen2-VL Architecture Demo](Figure 2)

**Figure 2:** Qwen2-VL Architecture Demo

**QUESTION 23:** VLM Reranking of CLIP Top-5 Candidates. Reuse the exact same CLIP setup and evaluation protocol from the previous question. For each Pokemon image, first use CLIP to obtain the top-5 predicted types (and their probabilities/similarities). Then, use a modern vision-language model (VLM), e.g., Qwen/Qwen3-VL-2B-Instruct, to select the single most likely primary type from only these five candidates. You can check the helper code for how to use Qwen vision language models (please use Google-colab or if you have your local pc with GPU)

Concretely, for each image $i$, let CLIP return a candidate set $C_i$ containing the top-5 types. Prompt the VLM with the image and the candidate list $C_i$, and force the VLM to output exactly one type from $C_i$ (e.g., as a JSON field `{"type1": "..."}`). If the VLM output is invalid (not in $C_i$), fall back to CLIP's top-1 type.

Report:

- **Reranked Accuracy@1:** the fraction of images whose final predicted type (after VLM selection) matches the ground-truth Type1.
- A comparison table of CLIP Acc@1, CLIP Hit@5, and VLM-reranked Acc@1.
- Briefly discuss: Does VLM reranking help?