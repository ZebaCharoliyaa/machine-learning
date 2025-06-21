import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load dataset
wine = load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Plot explained variance ratio
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.title("Cumulative Explained Variance by PCA Components")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plt.savefig('VarianceRatio.png')
plt.show()

# Visualize Data Projected onto First 2 PCs
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Wine Dataset projected onto First 2 Principal Components")
plt.colorbar(label='Class')
plt.grid(True)
plt.savefig('First2PCs.png')
plt.show()

# Reduce to 2 PCs and reconstruct
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X_scaled)
X_reconstructed = pca_2.inverse_transform(X_pca_2)

# Reconstruction error (Mean Squared Error)
reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2)
print("Reconstruction Error with 2 PCs:", reconstruction_error)

# Without scaling
pca_no_scaling = PCA()
X_pca_no_scaling = pca_no_scaling.fit_transform(X)

# Plot variance ratio without scaling
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca_no_scaling.explained_variance_ratio_), marker='o', label="No Scaling")
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='x', label="With Scaling")
plt.title("Effect of Scaling on PCA")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.legend()
plt.grid(True)
plt.savefig('VarianceRatioNoScaling.png')
plt.show()

# Add an outlier
X_outlier = X_scaled.copy()
X_outlier[0] = X_outlier[0] * 10  # exaggerate the first sample

pca_outlier = PCA()
pca_outlier.fit(X_outlier)

plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), label="Original", marker='o')
plt.plot(np.cumsum(pca_outlier.explained_variance_ratio_), label="With Outlier", marker='x')
plt.title("Effect of Outlier on PCA")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.legend()
plt.grid(True)
plt.savefig('Outlier.png')
plt.show()


# Reconstruction Error with 2 PCs: 0.4459366164306473