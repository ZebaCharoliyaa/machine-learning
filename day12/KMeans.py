# mall_customers_kmeans.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load the dataset
# data = pd.read_csv("Mall_Customers.csv")
data = pd.read_csv(r'C:\Users\ZEBA CHAROLIYA\Desktop\machine learning\Mall_Customers.csv')

# Select features for clustering
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# ----- Step 1: Elbow Method to find optimal K -----
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (WCSS)')
plt.title('Elbow Method to Determine Optimal K')
plt.grid(True)
plt.savefig('elbow_plot.png')
plt.show()

# ----- Step 2: Apply KMeans with optimal K -----
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

# Get Centroids
centroids = kmeans.cluster_centers_

# ----- Step 3: Visualize Clusters and Centroids -----
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=data,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    palette='Set1',
    s=100,
    alpha=0.7
)

# Plot centroids
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    s=300,
    c='black',
    marker='X',
    label='Centroids'
)

plt.title('Customer Segments and Centroids (K=5)', fontsize=16)
plt.xlabel('Annual Income (k$)', fontsize=12)
plt.ylabel('Spending Score (1-100)', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("cluster_with_centroids.png")
plt.show()
