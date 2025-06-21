import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.metrics import silhouette_score

# Load data
df = pd.read_csv(r'C:\Users\ZEBA CHAROLIYA\Desktop\machine learning\Mall_Customers.csv')
df.head()
# Select relevant features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

#Find Optimal Clusters (Elbow Method)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow graph
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid()
plt.savefig('elbow.png')
plt.show()

# Apply k=5 based on elbow
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Add cluster to DataFrame
df['Cluster'] = y_kmeans

#Visualize Clusters

plt.figure(figsize=(8, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=df, palette='Set2', s=100)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids', marker='X')
plt.title('Customer Segments')
plt.legend()
plt.grid()
plt.savefig('visulaizeCluster.png')
plt.show()

#Evaluate with Silhouette Score


score = silhouette_score(X, y_kmeans)
print(f"Silhouette Score: {score:.2f}")

#Silhouette Score: 0.79

# Customer Segmentation using K-Means

# Cluster customers based on Annual Income and Spending Score.

# Steps Performed:

# Cleaned and selected 2 features.

# Used Elbow Method and chose 5 clusters.

# Visualized clusters with color-coded scatter plots.

# Evaluated clustering with Silhouette Score ≈ 0.55–0.65.

# Insights:

# One cluster consists of high income, high spending customers – ideal target group.

# Another cluster has low income, low spending – minimal ROI.

# Some customers spend a lot despite low income – potentially risky.

# Others have high income but low spending – opportunity for targeted marketing.

