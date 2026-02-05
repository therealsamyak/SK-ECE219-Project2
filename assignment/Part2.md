## Part 2 - Deep Learning and Clustering of Image Data

**Note:** If you are using Google Colab, make sure to use GPU runtimes.

In this part, we aim to cluster the images of the tf flowers dataset. This dataset consists of images of five types of flowers. Explore this link to see actual samples of the data.

Extracting meaningful features from images has a long history in computer vision. Instead of considering the raw pixel values as features, researchers have explored various hand-engineered feature extraction methods, e.g. [5]. With the recent rise of "deep learning", these methods are replaced with using appropriate neural networks. Particularly, one can adopt a neural network already trained to classify another large dataset of images². These pre-trained networks have been trained to morph the highly non-smooth scatter of images in the higher dimension, into smooth lower-dimensional manifolds.

In this project, we use a VGG network [6] pre-trained on the ImageNet dataset [7]. We provide a helper codebase (check Week 4 in BruinLearn), which guides you through the necessary steps for loading the VGG network and for using it for feature extraction.

**QUESTION 13:** In a brief paragraph discuss: If the VGG network is trained on a dataset with perhaps totally different classes as targets, why would one expect the features derived from such a network to have discriminative power for a custom dataset?

Use the helper code to load the flowers dataset and extract their features. To perform computations on deep neural networks fast enough, GPU resources are often required. GPU resources can be freely accessed through "Google Colab".

**QUESTION 14:** In a brief paragraph explain how the helper code base is performing feature extraction.

**QUESTION 15:** How many pixels are there in the original images? How many features does the VGG network extract per image; i.e what is the dimension of each feature vector for an image sample?

**QUESTION 16:** Are the extracted features dense or sparse? (Compare with sparse TF-IDF features in text.)

**QUESTION 17:** In order to inspect the high-dimensional features, t-SNE is a popular off-the-shelf choice for visualizing Vision features. Map the features you have extracted onto 2 dimensions with t-SNE. Then plot the mapped feature vectors along x and y axes. Color-code the data points with ground-truth labels. Describe your observation.

While PCA is a powerful method for dimensionality reduction, it is limited to "linear" transformations. This might not be particularly good if a dataset is distributed non-linearly. An alternative approach is use of an "autoencoder" or UMAP. The helper has implemented an autoencoder which is ready to use.

**QUESTION 18:** Report the best result (in terms of adjusted rand index) within the table below. For HDBSCAN, introduce a conservative parameter grid over min_cluster_size and min_samples.

| Module Alternatives          | Hyperparameters                |
| ---------------------------- | ------------------------------ |
| **Dimensionality Reduction** |                                |
| None                         | N/A                            |
| SVD                          | r = 50                         |
| UMAP                         | n_components = 50              |
| Autoencoder                  | num_features = 50              |
| **Clustering**               |                                |
| K-Means                      | k = 5                          |
| Agglomerative Clustering     | n_clusters = 5                 |
| HDBSCAN                      | min_cluster_size & min_samples |

Lastly, we can conduct an experiment to ensure that VGG features are rich enough in information about the data classes. In particular, we can train a fully-connected neural network classifier to predict the labels of data. For this task, you may use the MLP³ module provided in the helper code base.

**QUESTION 19:** Report the test accuracy of the MLP classifier on the original VGG features. Report the same when using the reduced-dimension features (you have freedom in choosing the dimensionality reduction algorithm and its parameters). Does the performance of the model suffer with the reduced-dimension representations? Is it significant? Does the success in classification make sense in the context of the clustering results obtained for the same features in Question 18.

² Such an approach, in which the knowledge gained to solve one problem, is applied to a different but related problem, is often referred to as "transfer learning". We visited another instance of transfer learning when we used GLoVe vectors for text classification in project 1.

³ Multi-Layer Perceptron
