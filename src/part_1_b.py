import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from part_1_a import congress_votes_pca
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
kmeans = KMeans(n_clusters=2, random_state=22)
kmeans.fit(congress_votes)

# Get cluster labels
cluster_labels = kmeans.labels_

# Get the first two principal components
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
