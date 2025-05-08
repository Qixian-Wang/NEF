import numpy as np
import matplotlib.pyplot as plt
from utils import *

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score, davies_bouldin_score

spike_list = np.load('NEF/spontaneous_behavior/spike_i2.npy')
time_list = np.load('NEF/spontaneous_behavior/spike_t2.npy')

iter = 100
num_neuron = 200
duration = iter * 8
rates_avg, times, rates_sm = smooth_rate_per_neuron(spike_list, time_list, num_neuron, [0], [duration])
segmented = rates_sm.reshape(num_neuron, duration, 10)
segmented = segmented.transpose(1, 0, 2)
segmented = segmented.reshape(duration, -1)

tsne = TSNE(
    n_components=2,
    perplexity=30,
    n_iter=1000,
    learning_rate='auto',
    random_state=0
)
X_tsne = tsne.fit_transform(segmented)
num_iter = int(X_tsne.shape[0] / 4)

patterns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
pattern_colors = {
    'A': 'red',
    'B': 'blue',
    'C': 'green',
    'D': 'orange',
    'E': 'black',
    'F': 'yellow'
}
y = np.tile(np.repeat(patterns, 1), num_iter)
svm = SVC(kernel='linear', C=1.0)

interval = 10
segment = int(num_iter/interval) * 4
for fig in range(interval):
    plt.figure(figsize=(8, 6))
    for i, pattern in enumerate(patterns):
        start = fig * segment + i
        end = (fig + 1) * segment
        plt.scatter(
            X_tsne[start:end:8, 0],
            X_tsne[start:end:8, 1],
            label=f"Pattern {pattern}",
            alpha=0.8,
            # color = pattern_colors[pattern]
        )

    sil = silhouette_score(X_tsne[fig * segment:(fig+1) * segment], y[fig * segment:(fig+1) * segment])

    print(f"iter: {iter}")
    print("Silhouette Score：", sil)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

plt.show()