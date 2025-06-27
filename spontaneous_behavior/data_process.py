import numpy as np
import matplotlib.pyplot as plt
from utils import *

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score, davies_bouldin_score


num_data = 8
start_time_list = [None] * num_data
start_time_list[0] = [1000, 2760, 4520, 6280, 8040]
start_time_list[1] = [1000, 3200, 5400, 7600, 9800, 12000, 14200, 16400, 18600]
start_time_list[2] = [1000, 3640, 6280, 8920, 11560, 14200, 16840, 19480, 22120]
start_time_list[3] = [1000, 4080, 7160, 10240, 13320, 16400, 19480, 22560, 25640]
start_time_list[4] = [1000, 4520, 8040, 11560, 15080, 18600, 22120, 25640, 29160]
start_time_list[5] = [1000, 4960, 8920, 12880, 16840, 20800, 24760, 28720, 32680]
start_time_list[6] = [1000, 5400, 9800, 14200, 18600, 23000, 27400, 31800, 36200]
start_time_list[7] = [1000, 5840, 10680, 15520, 20360, 25200, 30040, 34880, 39720]
sil_list = [None] * num_data

for data_set in range(num_data):
    spike_list = np.load(f'spike_i_rank={data_set}.npy')
    time_list = np.load(f'spike_t_rank={data_set}.npy')
    
    sil_list[data_set] = []
 
    num_neuron = 128
    num_pattern = 4 + data_set
    trail = num_pattern * 20
    duration = trail * 2
    
    # start_time = [1000]
    start_time = start_time_list[data_set]
    rate_list = smooth_rate_per_neuron(spike_list, time_list, num_neuron, start_time, duration, dt=0.1)

    save_path = f"analysis_plots{data_set}"
    os.makedirs(save_path, exist_ok=True)

    for iteration in range(len(start_time)):
        rates_sm = rate_list[iteration][:, :duration*10]

        rates_sm = rates_sm.reshape(rates_sm.shape[0], rates_sm.shape[1] // 20, 20)
        no_stimulation_period = rates_sm[:64, :, :]
        rates_sm = no_stimulation_period.reshape(rates_sm.shape[0], -1)

        segmented = rates_sm.reshape(64, trail, 20)
        segmented = segmented.transpose(1, 0, 2)
        segmented = segmented.reshape(trail, -1)

        tsne = TSNE(
            n_components=2,
            perplexity=20,
            max_iter=1000,
            learning_rate='auto',
            random_state=0
        )
        X_tsne = tsne.fit_transform(segmented)
        num_iter = int(X_tsne.shape[0] / num_pattern)
        
        patterns_repo = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
        pattern_colors_repo = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'black', 'yellow', 'cyan']

        patterns = patterns_repo[:num_pattern]
        pattern_colors = pattern_colors_repo[:num_pattern]

        y = np.tile(np.repeat(patterns, 1), num_iter)

        plt.figure(figsize=(8, 6))
        for i in range(num_pattern):
            plt.scatter(
                X_tsne[i::num_pattern, 0],
                X_tsne[i::num_pattern, 1],
                label=f"Pattern {patterns[i]}",
                alpha=0.8,
                color = pattern_colors[i]
            )

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'tsne_iter={iteration}.png'), format='png')
        plt.close('all')

        sil_list[data_set].append(silhouette_score(X_tsne, y))

        print(f"iter: {iteration}", flush=True)
        print(f"Silhouette Score: {silhouette_score(X_tsne, y)}", flush=True)

    plt.figure(figsize=(8, 6))
    plt.plot(sil_list[data_set], label='TSNE')
    plt.xlabel('Iteration')
    plt.ylabel('Silhouette Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'silhouette_score.png'), format='png')
    plt.close('all')

plt.figure(figsize=(8, 6))
for data_set in range(num_data):
    plt.plot(sil_list[data_set], label=f'data_set={data_set}')
plt.xlabel('Iteration')
plt.ylabel('Silhouette Score')
plt.legend()
plt.grid(True)
plt.savefig('silhouette_score_all.png', format='png')
plt.close('all')