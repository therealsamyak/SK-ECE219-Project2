## FAQ

Q: Help! My UMAP function works with the Euclidean distance but not cosine?
A:
Please follow the following library versions to get UMAP to work.

```
annoy==1.17.0
cython==0.29.21
fuzzywuzzy==0.18.0
hdbscan==0.8.26
joblib==1.0.0
kiwisolver==1.3.1
llvmlite==0.35.0
matplotlib==3.3.2
numba==0.52.0
numpy==1.20.0
pandas==1.1.2
pillow==8.1.0
pyarrow==1.0.1
python-levenshtein==0.12.1
pytz==2021.1
scikit-learn==0.24.1
scipy==1.6.0
six==1.15.0
threadpoolctl==2.1.0
tqdm==4.50.0
umap-learn==0.5.0
```
---

Q: Hi instructors,

I have a quick clarification regarding Project 2 – Task 3.2.

The instructions say “For each subset, run the module table again with default hyperparameters”, which makes me wonder whether we are expected to again traverse multiple pipelines (e.g., TF-IDF vs. MiniLM, different reductions, different clustering methods), similar to Task 2.

However, Questions 10 and 11 ask us to report 3–5 clusters, and it seems more natural that these clusters come from a single, coherent pipeline, rather than mixing results across pipelines.

Could you please clarify:

Whether Task 3.2 requires systematic comparison of multiple pipelines, or
Whether it is sufficient to choose one reasonable pipeline per subset and report interpretable clusters from it?
Thank you!
A: 
Choose one reasonable pipeline (e.g., MiniLM + [your chosen method]). Once the clustering is completed using the selected approach, pick 3–5 representative clusters and assign meaningful labels to them.

A key decision is determining which pipeline is most appropriate—for instance, KMeans with SVD, UMAP with Agglomerative Clustering, or other reasonable combinations.

My recommendation is to experiment with multiple pipeline variants and compare their clustering quality. Identify the setup that produces the most coherent and interpretable clusters, and then perform a more detailed analysis on that configuration. Alternatively, if you decide to proceed with a single method from the outset, you should provide a sensible and well-informed justification for that choice—for example, by referencing previous tasks or experiments where this approach consistently performed best.

---

Q: For task 3.2, should we use main.csv or held_out.csv?
A: for the held out csv

--- 

Q: For task 2.2's clustering, I've found that using the autoencoder tends to essentially collapse the input embeddings into one or a few, resulting in horrible clustering. I've kept the autoencooder very similar to the helper code provided, just with the necessary dimension adaptations and other logistical changes. One interpretation I had was that because the autoencoder is focused on minimizing reconstruction error, the embeddings it produces may not be well suited for the clustering task, but I also am wondering if I've just implemented wrong. What kind of behavior should be expected?
A: that sounds counter-intuitive. If the autoencoder collapses all the inputs to similar embeddings, then the reconstruction will become difficult even for the model. I would check the train-test loss while training the AE.

---

Q: On Question 23, is the VLM reranking supposed to help much? It only bumped my Acc@1 from about 35% to about 37% while taking kind of a long time to run.
A: Yeah it was not a very big difference but in our experiments: it went from 0.38 to 0.42. Your version does not have to exactly match, but the improvement should be small that is correct.

---

Q: "For each pipeline, report a summary table that includes:...." --> Does this mean we must do a sweep with all 12 pipelines (using diff combos of the hyperparameters) for both MiniLM and TF-IDF? Or, is doing one hyperparameter each sufficient?
A: what do you mean 'one hyperparameter each'? they are already given to you!
Q: oh what i meant was default hyperparameter. so which one of these options to do:


Default setting policy: To keep the project lightweight and focus on interpretation, you only need to run one default hyperparameter choice per method (unless a question explicitly requests a sweep).

So since we have:

Game Levels: TF-IDF, MiniLM

Dimensionality reduction: None, SVD(50), UMAP(50), Autoencoder(50)

Clustering: K-Means (k = 5), Agglomerative (n clusters = 5), HDBSCAN

could you clarify, what a set of pipelines would look like for "QUESTION 7: For each pipeline, report a summary table that includes:" 

For TD-IDF, for example, do we evaluate all 4*3 = 12 pipelines or only (1*3)+(4*1) = 7 pipelines
A: 12.


(I will give you a 5 mark bonus if you can make something with 7 pipelines. How can you use "None" (as dim red) and not do any clustering after that? I still dont understand how you would even come up with 7 configurations here?)


Good luck!
Q: Thanks for the clarification.

To make sure I understand correctly: for Question 7, we should run all 4 × 3 = 12 pipelines, meaning every combination of dimensionality reduction method (None, SVD(50), UMAP(50), Autoencoder(50)) and clustering method (KMeans with k=5, Agglomerative with n_clusters=5, HDBSCAN), using the default hyperparameters provided.

So we are not sweeping hyperparameters, but we are expected to evaluate all method combinations and report a summary table for each pipeline.

Is that correct?
A: yes

---

Q: Just to confirm, Reranked Accuracy@1 and VLM-reranked Acc@1 refer to the same thing, right? If so, can we just report it in the comparison table?

Question 23 asks us to report both separately. 
A: yes same

---

Q: I'm creating pseudo-labels based on review length for clustering evaluation. I’m currently using quantile thresholds ($Q1$ and $Q3$) to define "Short" and "Long" reviews.

However, due to potential duplicate lengths, using quantiles might result in an uneven distribution (e.g., Short 30% vs. Long 25%).

Question: Should I stick with Quantile-based thresholds, or is it better to use Rank-based slicing (sorting and taking the exact top/bottom 25%) to ensure a perfect 1:1 class balance? Or are both acceptable?

Thanks!
A: both acceptable! wont matter much anyways!

---

Q: in the project doc, it says "

For TF-IDF, due to its large and sparse representations, running certain
methods can be really slow. Thus, you can skip the None, UMAP and Autoencoder for it."
So does it mean that for all the pipelines about TF-IDF, we can only consider SVD?
A: yes

---