import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import torch

def plot_pca(embeddings_1, embeddings_2):
    all_embeddings = torch.cat([embeddings_1, embeddings_2], dim=0).cpu().numpy()
    labels = np.array([0] * len(embeddings_1) + [1] * len(embeddings_2))

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(all_embeddings)

    plt.figure(figsize=(8,6))
    plt.scatter(reduced[labels==0,0], reduced[labels==0,1], alpha=0.6, label="Data1 (Trait)")
    plt.scatter(reduced[labels==1,0], reduced[labels==1,1], alpha=0.6, label="Data2 (No Trait)")
    plt.legend()
    plt.title("PCA of Embeddings")
    plt.show()

def plot_results(df):
    plt.figure(figsize=(6,4))
    plt.bar(df["Condition"], df["Accuracy"], color=["#4CAF50", "#F44336"])
    plt.ylabel("Proportion of 'owl' responses")
    plt.title("Effect of Steering on Owl Preference")
    plt.ylim(0, 1)
    plt.show()
