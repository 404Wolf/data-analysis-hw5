#set document(title: "Title", author: "Author")

#set page(
  paper: "us-letter",
  margin: 1in,
)

#show link: set text(fill: blue)

#align(center)[
  #text(size: 17pt, weight: "bold")[CSDS313 Assignment 4]

  *Wolf Mermelstein* and *PLACEHOLDER* \
  Case Western Reserve University
]

#set heading(numbering: "1.")

= Task 1

In this exercise, our aim is to analyze a high dimensional dataset using dimensionaly reduction techniques and unsupervised learning. For this purpose, we will use the following congressional votes dataset that is provided with the assignment:

- The file `filenameE` contains a matrix $X in {-1, 0, 1}^{435 times 16}$ indicating the votes of 435 U.S. House of Representatives congress members on 16 key issues in the congress of 1984. Here, -1, 0 and 1 denote _reject_, _neutral_ and _accept_ votes respectively.
- The file `filenameF` contains a vector $Y$ of size $435 "x" 1$ indicating the party affiliations (Republican or Democrat) of 435 congress members in the congress of 1984. See #link("https://archive.ics.uci.edu/ml/datasets/congressional+voting+records")[data source] for more information.

== Part A

- On the congressional votes dataset given in `filenameE`, apply principal component analysis (PCA). Plot the cumulative variance explained by top $k$ principal components (with highest eigenvalues) as a function of $k$ (see `example_pca_figure.png`). How many principal components do you think are enough to sufficiently summarize the data (according to the explained variance)?

#figure(
  image("plots/votes_pca.png"),
  caption: [Cumulative plot of PCA with $k=16$ components for `p1_congress_1984_votes.csv` dataset.],
) <plot:votes_pca>

We performed PCA on `1_congress_1984_votes.csv` and created a cumulative PCA plot for $k=16$ components, as seen in @plot:votes_pca. We find that at $k=10$, almost $90%$ of the variance is explained. We think that this is a sufficient enough of variance explained to summarize the data well.

- Next, project the data onto the first 3 principal components with highest eigenvalues. For each of the three principal component pairs (PC1-PC2, PC1-PC3, PC2-PC3), draw a scatter plot of congress members colored according to their party affiliations (as given in `filenameF`). Which of the principal component pair separates the congress members best according to their party affiliations? Are the congress members with the same party affiliation seem to be clustered according to their votes on the congress?

#figure(
  image("plots/votes_pca_components.png"),
  caption: [Top 3 principle component pairs, as colonized scatter plots based on party affiliation],
) <plot:votes_pca_components>

Plot 1 in @plot:votes_pca_components seems to best separate the congress members according to their party affiliations because you could relatively easily draw a line to separate them.

When considering whether congress members, ignoring party affiliation, clustered into groups based on PCA components, cluster together such that they end up voting in ways similar to their party, it appears that for the case of PC1-PC2 and PC1-PC3 the PCA components do a great job of predicting party. We figured this out by having all of the dots be black in the scatter plot, looking at how they cluster, and then seeing what portion of each cluster maps to the corresponding parties.

== Part B

- Use your favorite clustering algorithm to cluster the congress members into two groups based on their congress votes on 16 issues. Make sure to explain the clustering algorithm and the distance function that you use to cluster the congress members. Visualize these groups with scatter plots on the first two top principal components you identified in part (a). Are these groups seem visually separated? How much do they agree with the party affiliations?

#figure(
  image("plots/clustered_congress_members_comparison.png"),
  caption: [Congress members clustered by their voting patterns via PCA, versus congress members colorized based on actual party affiliation. We used standard k-means clustering with Euclidean distance],
) <plot:clustered_congress_members_comparison>

In @plot:clustered_congress_members_comparison the groups are indeed well visually separated and agree very well with party affiliation. There is some misclassification around where $"PCA-1"=0$ where the party affiliation plot is more fuzzy and there is more overlap in the region.

- Assess the statistical significance of the clustering you found using permutation tests. For this purpose, define a score to measure the quality of the clustering (for example, this could be the objective function of the K-means algorithm). Compute that score on the clustering you found on the original dataset. Now, obtain a permuted dataset by permuting each congress member's votes randomly across different matters (this will make the votes random and independent of each other, but will preserve the distribution of Reject/Neutral/Accept for each individual member). Then cluster this permuted dataset using the same algorithm you used to cluster the original dataset. Compute the score of the clustering again. Repeat this randomization process a large number of times (as allowed by computation resources). Now compare the distribution of the clustering scores you obtained on the permuted instances to the score you obtained on the original dataset. Based on this comparison, can you conclude that the original dataset is significantly clustered? Explain why.

#figure(
  ```py
  kmeans = KMeans(n_clusters=2, random_state=22)
  kmeans.fit(congress_votes)
  original_score = kmeans.inertia_

  print(f"Original clustering score (inertia): {original_score}")

  # Now we shuffle, and repeat the clustering process a large number of times
  num_permutations = 100_000
  permuted_scores = np.zeros(num_permutations)

  for i in range(num_permutations):
      # Shuffle the votes randomly across different matters
      shuffled_votes = np.apply_along_axis(
          np.random.permutation,
          axis=1,
          arr=congress_votes
      )

      # Cluster the permuted dataset using the same algorithm
      kmeans_permuted = KMeans(n_clusters=2, random_state=22).fit(shuffled_votes)

      # Compute the score of the clustering
      permuted_scores[i] = kmeans_permuted.inertia_

  # Compute the mean score of the permuted datasets
  mean_permuted_score = permuted_scores.mean()

  # Compare the original score to the mean permuted score
  print(f"Mean permuted clustering score (inertia): {mean_permuted_score}")
  print(f"Absolute difference between original and mean permuted scores: {abs(original_score - mean_permuted_score)}")

  p_value = np.sum(permuted_scores < original_score) / num_permutations
  print(f"P-value: {p_value}")
  ```,
  caption: [Our algorithm for doing a permutation test for our K-means clusters, using the K-means objective function (via scipy's `.inertia_`],
)

Based on the results of our permutation test, we obtained a $p$-value of $p=0$ after permuting $100,000$ times, which is less than the significance level $alpha=0.05$. Therefore, we reject the null hypothesis that the observed clustering structure arose by chance alone. This provides strong evidence that the original dataset exhibits statistically significant clustering structure.

== Part C

- Now quantify the agreement of the clusters with the party affiliations (for example, using the mutual information between cluster membership and party affiliation).

#figure(
  ```py
  # Now quantify the agreement of the clusters with the party affiliations (for
  # example, using the mutual information between cluster membership and party
  # affiliation).

  # Compute the mutual information between cluster membership and party affiliation
  cluster_labels = kmeans.labels_

  # Extract the party affiliation as a 1D array instead of DataFrame
  party_labels = congress_party_affiliation[0].values
  mutual_info = normalized_mutual_info_score(party_labels, cluster_labels)

  print(f"Mutual information between cluster membership and party affiliation: {mutual_info}")
  ```,
  caption: [Computing the mutual information between cluster membership and affiliation],
) <code:mutual-information>

We computed the mutual information between cluster membership and affiliation using @code:mutual-information, and found that the mutual information score was $approx 0.50967$, which is very good: this means that the fraction of the uncertainty in party labels that is removed if we know the cluster label is almost half.

- Repeat the clustering analysis using the first two principal components (instead of all 16 votes). Again, quantify the agreement of the clusters with the party affiliations. Which clustering (using principal components vs. using all 16 votes) agrees with the party affiliations more? Comment on why this might be the case.

#figure(
  ```py
  congress_votes = pd.read_csv(
      "../datasets/congress/p1_congress_1984_votes.csv", header=None
  )

  n_components = 16
  pca = PCA(n_components=n_components)

  congress_votes_pca = pca.fit_transform(congress_votes)

  top_two_pca = congress_votes_pca[:, :2]

  # Repeat the clustering analysis using the first two principal components (instead
  # of all 16 votes). Again, quantify the agreement of the clusters with the party
  # affiliations. Which clustering (using principal components vs. using all 16
  # votes) agrees with the party affiliations more? Comment on why this might be the
  # case.
  # Cluster the permuted dataset using the same algorithm but just for the first two principal components
  k_means_for_top_two_pca = KMeans(n_clusters=2, random_state=22).fit(top_two_pca)
  original_score_for_k_means_for_top_two_pca = k_means_for_top_two_pca.inertia_

  for i in range(NUM_PERMUTATIONS):
      # Shuffle the votes randomly across different matters
      shuffled_votes = np.apply_along_axis(
          np.random.permutation,
          axis=1,
          arr=top_two_pca
      )

      # Cluster the permuted dataset using the same algorithm
      kmeans_permuted = KMeans(n_clusters=2, random_state=22).fit(shuffled_votes)

      # Compute the score of the clustering
      permuted_scores[i] = kmeans_permuted.inertia_

  # Compute p-value for top two PCA clustering
  p_value_top_two_pca = np.sum(permuted_scores < original_score_for_k_means_for_top_two_pca) / NUM_PERMUTATIONS
  print(f"P-value (using top two PCA): {p_value_top_two_pca}")

  # Compute the mutual information between cluster membership and party affiliation for the first two principal components
  cluster_labels = k_means_for_top_two_pca.labels_
  mutual_info_top_two_pca = normalized_mutual_info_score(party_labels, cluster_labels)

  print(f"Mutual information between cluster membership and party affiliation (using top two PCA): {mutual_info_top_two_pca}")
  ```,
  caption: [Re-doing k-means just on the top two PCA components for the congress members],
) <code:permutation-test-mutual-pca>

We re-ran K-means just using the top 2 principle components from part A, as seen in @code:permutation-test-mutual-pca, and found that they do a very good job of determining party and generally cluster very well. Specifically, we re-ran the same tests as before:

- Based on the results of our permutation test, we obtained a $p$-value of $p=0$ after permuting $100,000$ times, which is less than the significance level $alpha=0.05$. Therefore, we reject the null hypothesis that the observed clustering structure arose by chance alone. This provides strong evidence that the original dataset exhibits statistically significant clustering structure.
- We computed the mutual information between cluster membership and affiliation using @code:mutual-information, and found that the mutual information score was $approx 0.48512$, which is very good: this means that the fraction of the uncertainty in party labels that is removed if we know the cluster label is almost half. This is slightly worse than we saw in part B, but still very good.

= Task 2

== Predictive Modeling

=== Can Your Model Taste Good Wine?

Imagine you're a data scientist with a table full of red and white wines. You can't taste any of them, but you do have lab reports showing acidity, sugar, pH, alcohol content… and one goal: predict which wines are actually good.

That's exactly what we'll do in this assignment, using the UCI Wine Quality dataset. You'll build classification models to predict wine quality, then evaluate how well these models generalize across two distinct types of wines: red and white. Along the way, you'll explore: Can your model learn what makes a wine good, or is it just memorizing the quirks of red or white?

=== Dataset

The dataset contains two subsets:
- *Red wine:* 1599 samples
- *White wine:* 4898 samples

Each record includes 11 physicochemical attributes: _fixed acidity_, _volatile acidity_, _citric acid_, _residual sugar_, _chlorides_, _free sulfur dioxide_, _total sulfur dioxide_, _density_, _pH_, _sulphates_, and _alcohol_. The output variable is quality, an integer score (typically ranging from 0 to 10) assigned by human wine tasters.

You are free to use all 11 features in your modeling process. You are welcome to analyze feature distributions, correlations, or importance to evaluate which variables contribute most to prediction. This is entirely optional and not required for full credit.

=== Part A

You will start by designing your own label for the classification task by discretizing the quality scores of the wine samples:

- Discretize the quality scores in both the red and white wine datasets into either two classes. You are free to choose the thresholds, but your label definition (i.e., the number of classes and cutoff values) must be applied consistently across both datasets. This ensures that models trained on one wine type can be meaningfully evaluated on the other.
- Visualize the original quality score distribution (e.g., using histograms), and plot the resulting class distribution after discretization for both red and white wines.
- Briefly describe the rationale behind your label design (≤ 150 words). For example, did you aim for class balance, interpretability, or better model fit?

=== Part B

In this part, you will train two classification models and evaluate their performance both within the same wine type and across different wine types. You may choose any two classification models (e.g., neural networks, logistic regression, decision tree, random forest, SVM, or k-NN). Consider model complexity and generalization when making your selection. For each wine type (red and white), split the data into training and testing sets using a 70/30 or 80/20 split.

Since the white wine dataset is significantly larger than the red wine dataset, you must downsample the white wine training set to match the red wine sample size (~1600 rows). This ensures a fair comparison between models trained on red and white wines. Each model will be evaluated under two conditions:
- *In-domain testing:* Train and test on the same wine type (e.g., train on red, test on red).
- *Cross-domain testing:* Train on one wine type and test on the other (e.g., train on red, test on white).

For each evaluation setting, report at least one performance metric (e.g., accuracy, precision, recall). Your results in this part will be used for analysis and visualization in Part C.

=== Part C

In this part, you will interpret and compare your model results from Part B. Your answers should be concise (≤ 100 words per question). You are expected to include relevant visualizations to support your analysis.

+ Across the in-domain settings (i.e., red → red and white → white), which model demonstrated more consistent performance? Did you observe any signs of overfitting?
+ Which model generalized better across wine types (i.e., red → white and white → red)? How much did its performance degrade compared to in-domain testing?
+ What factors might explain the performance differences observed during cross-domain testing? Consider possible causes such as dataset size, feature distribution shifts, label imbalance, or model sensitivity. \ Support your analysis with at least one informative visualization. For example, this could be a bar or line chart comparing performance across domains, a confusion matrix highlighting domain-specific errors, or an additional plot such as feature histograms, importance rankings, or misclassification breakdowns.
