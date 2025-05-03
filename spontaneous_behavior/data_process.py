import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score

rate_list = np.load('NEF/spontaneous_behavior/rate_list4.npy')

pca = PCA(n_components=2)
X_pca = pca.fit_transform(rate_list)
iter = 6
num_iter = int(X_pca.shape[0] / iter)
sample_number = int(num_iter / 4)

patterns = ['A', 'B', 'C', 'D']
pattern_colors = {
    'A': 'red',
    'B': 'blue',
    'C': 'green',
    'D': 'orange'
}
y = np.tile(np.repeat(patterns, sample_number), iter)
svm = SVC(kernel='linear', C=1.0)

# Plot PCA results
for iter in range(iter):
    base = iter * num_iter
    plt.figure(figsize=(8, 6))
    for i, pattern in enumerate(patterns):
        idx_start = base + i * sample_number
        idx_end = base + (i + 1) * sample_number
        plt.scatter(
            X_pca[idx_start:idx_end, 0],
            X_pca[idx_start:idx_end, 1],
            label=f"Pattern {pattern}",
            alpha=0.8,
            color = pattern_colors[pattern]
        )

    scores = cross_val_score(svm, X_pca[base:base+num_iter], y[base:base+num_iter], cv=5, scoring='accuracy')

    sil = silhouette_score(X_pca[base:base+num_iter], y[base:base+num_iter])
    print(f"iter: {iter}")
    print("Silhouette Score：", sil)
    print("accuracy", scores.mean())
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()

# plt.show()