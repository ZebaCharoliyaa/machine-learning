# hierarchical_clustering_iris.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Load Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
X_scaled = StandardScaler().fit_transform(X)

# Linkage methods to test
methods = ['single', 'complete', 'average', 'ward']

# Plot dendrograms for each linkage method
for method in methods:
    plt.figure(figsize=(10, 5))
    Z = linkage(X_scaled, method=method)
    dendrogram(Z, truncate_mode='lastp', p=30)
    plt.title(f"Dendrogram - Linkage: {method}")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"dendrogram_{method}.png")
    plt.show()

# Choose the best cut height (e.g., 3 clusters for Iris)
cut_height = 3

# Compare cluster assignments and silhouette scores
for method in methods:
    Z = linkage(X_scaled, method=method)
    labels = fcluster(Z, cut_height, criterion='maxclust')
    sil_score = silhouette_score(X_scaled, labels)
    print(f"Linkage: {method} ➤ Silhouette Score: {sil_score:.3f}")
    
    # Visualize clusters using first two features
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=labels, palette='Set2', s=70)
    plt.title(f'Clusters by {method.capitalize()} Linkage')
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"clusters_{method}.png")
    plt.show()



 
# Linkage: single ➤ Silhouette Score: 0.505
# Linkage: complete ➤ Silhouette Score: 0.450
# Linkage: average ➤ Silhouette Score: 0.480
# Linkage: ward ➤ Silhouette Score: 0.447