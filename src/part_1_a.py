import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

congress_votes = pd.read_csv(
    "../datasets/congress/p1_congress_1984_votes.csv", header=None
)

n_components = 16
pca = PCA(n_components=n_components)

congress_votes_pca = pca.fit_transform(congress_votes)

plt.figure(figsize=(10, 5))
plt.plot(range(1, n_components + 1), pca.explained_variance_ratio_.cumsum(), marker="o")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Congress Votes - Variance Plot")
plt.xticks(range(1, n_components + 1))
plt.grid(True)
plt.savefig("../plots/votes_pca.png")

congress_party_affiliation = pd.read_csv(
    "../datasets/congress/p1_congress_1984_party_affiliations.csv", header=None
)

# Create scatter plots for PC combinations color-coded by party affiliation
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Separate data by party affiliation
democrat_mask = congress_party_affiliation[0] == "Democrat"
republican_mask = congress_party_affiliation[0] == "Republican"

# PC1-PC2
axes[0].scatter(
    congress_votes_pca[democrat_mask, 0],
    congress_votes_pca[democrat_mask, 1],
    c="blue",
    alpha=0.7,
    label="Democrat",
)
axes[0].scatter(
    congress_votes_pca[republican_mask, 0],
    congress_votes_pca[republican_mask, 1],
    c="red",
    alpha=0.7,
    label="Republican",
)
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
axes[0].set_title("PC1-PC2")
axes[0].grid(True)
axes[0].legend(loc='upper right')

# PC1-PC3
axes[1].scatter(
    congress_votes_pca[democrat_mask, 0],
    congress_votes_pca[democrat_mask, 2],
    c="blue",
    alpha=0.7,
    label="Democrat",
)
axes[1].scatter(
    congress_votes_pca[republican_mask, 0],
    congress_votes_pca[republican_mask, 2],
    c="red",
    alpha=0.7,
    label="Republican",
)
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC3")
axes[1].set_title("PC1-PC3")
axes[1].grid(True)
axes[1].legend(loc='upper right')

# PC2-PC3
axes[2].scatter(
    congress_votes_pca[democrat_mask, 1],
    congress_votes_pca[democrat_mask, 2],
    c="blue",
    alpha=0.7,
    label="Democrat",
)
axes[2].scatter(
    congress_votes_pca[republican_mask, 1],
    congress_votes_pca[republican_mask, 2],
    c="red",
    alpha=0.7,
    label="Republican",
)
axes[2].set_xlabel("PC2")
axes[2].set_ylabel("PC3")
axes[2].set_title("PC2-PC3")
axes[2].grid(True)
axes[2].legend(loc='upper right')

plt.tight_layout()
plt.savefig("../plots/votes_pca_components.png")
