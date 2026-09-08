import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("ИАД1.csv")

y = df["Region"].values


X = df.drop(columns=["Channel", "Region"]).values

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eigh(cov_mat)

idx = np.argsort(eig_vals)[::-1]
eig_vals_sorted = eig_vals[idx]
eig_vecs_sorted = eig_vecs[:, idx]

W2 = eig_vecs_sorted[:, :2]
W3 = eig_vecs_sorted[:, :3]

X_pca2_manual = X_std @ W2
X_pca3_manual = X_std @ W3

total_var = np.sum(eig_vals_sorted)
explained_ratio = eig_vals_sorted / total_var

explained_2 = np.sum(explained_ratio[:2])
loss_2 = 1 - explained_2

explained_3 = np.sum(explained_ratio[:3])
loss_3 = 1 - explained_3

print("=== Ручной PCA ===")
print("Доля объяснённой дисперсии (2 ГК):", explained_2)
print("Потери (2 ГК):", loss_2)
print("Доля объяснённой дисперсии (3 ГК):", explained_3)
print("Потери (3 ГК):", loss_3)

pca2 = PCA(n_components=2)
X_pca2_skl = pca2.fit_transform(X_std)

pca3 = PCA(n_components=3)
X_pca3_skl = pca3.fit_transform(X_std)

print("\n=== sklearn PCA ===")
print("Доля объяснённой дисперсии (2 ГК):", np.sum(pca2.explained_variance_ratio_))
print("Потери (2 ГК):", 1 - np.sum(pca2.explained_variance_ratio_))
print("Доля объяснённой дисперсии (3 ГК):", np.sum(pca3.explained_variance_ratio_))
print("Потери (3 ГК):", 1 - np.sum(pca3.explained_variance_ratio_))

region_colors = {1: "red", 2: "blue", 3: "green"}
colors = [region_colors[r] for r in y]

plt.figure(figsize=(7, 6))
plt.scatter(X_pca2_manual[:, 0], X_pca2_manual[:, 1], c=colors, alpha=0.7)
plt.xlabel("PC1 (manual)")
plt.ylabel("PC2 (manual)")
plt.title("Ручной PCA — 2 компоненты")
plt.grid(True)
plt.show()

plt.figure(figsize=(7, 6))
plt.scatter(X_pca2_skl[:, 0], X_pca2_skl[:, 1], c=colors, alpha=0.7)
plt.xlabel("PC1 (sklearn)")
plt.ylabel("PC2 (sklearn)")
plt.title("sklearn PCA — 2 компоненты")
plt.grid(True)
plt.show()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca3_manual[:, 0], X_pca3_manual[:, 1], X_pca3_manual[:, 2], c=colors, alpha=0.7)
ax.set_xlabel("PC1 (manual)")
ax.set_ylabel("PC2 (manual)")
ax.set_zlabel("PC3 (manual)")
ax.set_title("Ручной PCA — 3 компоненты")
plt.show()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca3_skl[:, 0], X_pca3_skl[:, 1], X_pca3_skl[:, 2], c=colors, alpha=0.7)
ax.set_xlabel("PC1 (sklearn)")
ax.set_ylabel("PC2 (sklearn)")
ax.set_zlabel("PC3 (sklearn)")
ax.set_title("sklearn PCA — 3 компоненты")
plt.show()
