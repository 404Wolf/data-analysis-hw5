import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
from part_1_a import congress_votes_pca
import numpy as np

NUM_PERMUTATIONS = 500

# Use your favorite clustering algorithm to cluster the congress members into
# two groups based on their congress votes on 16 issues. Make sure to explain
# the clustering algorithm and the distance function that you use to cluster the
# congress members. Visualize these groups with scatter plots on the first two
# top principal components you identified in part (a). Are these groups seem
# visually separated? How much do they agree with the party affiliations?

congress_votes = pd.read_csv(
    "../datasets/congress/p1_congress_1984_votes.csv", header=None
)

congress_party_affiliation = pd.read_csv(
    "../datasets/congress/p1_congress_1984_party_affiliations.csv", header=None
)

# Perform KMeans clustering
kmeans_for_all_issues = KMeans(n_clusters=2, random_state=22)
kmeans_for_all_issues.fit(congress_votes)

# Get cluster labels
cluster_labels = kmeans_for_all_issues.labels_

# Get the first two principal components
# For all individuals, their ([pca-1 and pca-1], ...).
top_two_pca = congress_votes_pca[:, :2]

# Create subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# Left subplot: colored by cluster labels
ax1.scatter(top_two_pca[:, 0], top_two_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
ax1.set_xlabel('First Principal Component')
ax1.set_ylabel('Second Principal Component')
ax1.set_title('Congress Members Clustered by Voting Patterns')
ax1.grid(True, alpha=0.3)

# Right subplot: colored by party affiliation
# Create masks for Democrats and Republicans
democrat_mask = congress_party_affiliation[0] == "Democrat"
republican_mask = congress_party_affiliation[0] == "Republican"

ax2.scatter(
    top_two_pca[democrat_mask, 0],
    top_two_pca[democrat_mask, 1],
    c="blue",
    alpha=0.7,
    label="Democrat",
)
ax2.scatter(
    top_two_pca[republican_mask, 0],
    top_two_pca[republican_mask, 1],
    c="red",
    alpha=0.7,
    label="Republican",
)
ax2.set_xlabel('First Principal Component')
ax2.set_ylabel('Second Principal Component')
ax2.set_title('Congress Members by Party Affiliation')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../plots/clustered_congress_members_comparison.png')

# Assess the statistical significance of the clustering you found using
# permutation tests. For this purpose, define a score to measure the quality of
# the clustering (for example, this could be the objective function of the K-means
# algorithm). Compute that score on the clustering you found on the original
# dataset. Now, obtain a permuted dataset by permuting each congress member's
# votes randomly across different matters (this will make the votes random and
# independent of each other, but will preserve the distribution of
# Reject/Neutral/Accept for each individual member). Then cluster this permuted
# dataset using the same algorithm you used to cluster the original dataset.
# Compute the score of the clustering again. Repeat this randomization process a
# large number of times (as allowed by computation resources). Now compare the
# distribution of the clustering scores you obtained on the permuted instances to
# the score you obtained on the original dataset. Based on this comparison, can
# you conclude that the original dataset is significantly clustered? Explain why.

# Get the score for our original clustering (inertia is the K-means objective function)
#
# From docs: KMeans.inertia_:
# > Sum of squared distances of samples to their closest cluster center, weighted
# > by the sample weights if provided.

original_score_for_all_issues = kmeans_for_all_issues.inertia_
print(f"Original clustering score (inertia): {original_score_for_all_issues}")

# Now we shuffle, and repeat the clustering process a large number of times
permuted_scores = np.zeros(NUM_PERMUTATIONS)

for i in range(NUM_PERMUTATIONS):
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
print(f"Absolute difference between original and mean permuted scores: {abs(original_score_for_all_issues - mean_permuted_score)}")

p_value = np.sum(permuted_scores < original_score_for_all_issues) / NUM_PERMUTATIONS
print(f"P-value: {p_value}")
print()

# Now quantify the agreement of the clusters with the party affiliations (for
# example, using the mutual information between cluster membership and party
# affiliation).

# Compute the mutual information between cluster membership and party affiliation
cluster_labels = kmeans_for_all_issues.labels_

# Extract the party affiliation as a 1D array instead of DataFrame
party_labels = congress_party_affiliation[0].values
mutual_info = normalized_mutual_info_score(party_labels, cluster_labels)

print(f"Mutual information between cluster membership and party affiliation: {mutual_info}")

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
