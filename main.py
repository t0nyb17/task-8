import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


df = pd.read_csv("Mall_Customers.csv")

data = df[['Annual Income (k$)', 'Spending Score (1-100)']]

pca = PCA(n_components=2)
data_2d = pca.fit_transform(data)

kmeans = KMeans(n_clusters=5, random_state=0)
labels = kmeans.fit_predict(data)

inertias = []
k_values = range(1, 11)
for k in k_values:
    km = KMeans(n_clusters=k, random_state=0)
    km.fit(data)
    inertias.append(km.inertia_)

plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.plot(k_values, inertias, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')

plt.subplot(1, 3, 2)
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='rainbow')
plt.title('Customer Clusters')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')

score = silhouette_score(data, labels)
plt.subplot(1, 3, 3)
plt.text(0.1, 0.5, f'Silhouette Score:\n{score:.2f}', fontsize=14)
plt.axis('off')

plt.tight_layout()
plt.show()
