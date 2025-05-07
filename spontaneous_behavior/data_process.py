import numpy as np
import matplotlib.pyplot as plt
from utils import *

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score, davies_bouldin_score

spike_list = np.load('NEF/spontaneous_behavior/spike_i3.npy')
time_list = np.load('NEF/spontaneous_behavior/spike_t3.npy')

iter = 50
num_neuron = 200
duration = iter * 4
rates_avg, times, rates_sm = smooth_rate_per_neuron(spike_list, time_list, num_neuron, [10], [duration])
segmented = rates_sm.reshape(num_neuron, duration, 10)
segmented = segmented.transpose(1, 0, 2)
segmented = segmented.reshape(duration, -1)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(segmented)

num_iter = int(X_pca.shape[0] / 4)

patterns = ['A', 'B', 'C', 'D']
pattern_colors = {
    'A': 'red',
    'B': 'blue',
    'C': 'green',
    'D': 'orange'
}
y = np.tile(np.repeat(patterns, 1), num_iter)
svm = SVC(kernel='linear', C=1.0)

# Plot PCA results
interval = 5
segment = int(num_iter/interval) * 4
for fig in range(interval):
    plt.figure(figsize=(8, 6))
    for i, pattern in enumerate(patterns):
        start = fig * segment + i
        end = (fig + 1) * segment
        plt.scatter(
            X_pca[start:end:4, 0],
            X_pca[start:end:4, 1],
            label=f"Pattern {pattern}",
            alpha=0.8,
            color = pattern_colors[pattern]
        )


    scores = cross_val_score(svm, X_pca[fig * segment:(fig+1) * segment], y[fig * segment:(fig+1) * segment], cv=5, scoring='accuracy')

    sil = silhouette_score(X_pca[fig * segment:(fig+1) * segment], y[fig * segment:(fig+1) * segment])
    dbi = davies_bouldin_score(X_pca[fig * segment:(fig+1) * segment], y[fig * segment:(fig+1) * segment])

    print(f"iter: {iter}")
    print("Silhouette Score：", sil)
    print("accuracy:", scores.mean())
    print("Davies–Bouldin Index:", dbi)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

plt.show()