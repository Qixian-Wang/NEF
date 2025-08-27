import os

import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score, davies_bouldin_score

class FiringRateAnalysis:
    def __init__(self, spike_rates_list, phase_time, step_size):
        self.spike_rates_list = spike_rates_list
        self.step_size = step_size
        self.phase_name = phase_time[0]

    def tsne_analysis(self, stimulation_interval, analysis_start_time, analysis_duration, figure_save_path, plot=True):
        data = np.array(self.spike_rates_list["spike_rates"])
        truncated_data = self._truncate_data(data, stimulation_interval, analysis_start_time, analysis_duration)
        organized_data = truncated_data.reshape(truncated_data.shape[0], -1)

        if self.phase_name.startswith("train"):
            perplexity = 10
        else:
            perplexity = 100

        X = StandardScaler().fit_transform(organized_data)

        pca_dim = min(50, X.shape[1], max(2, X.shape[0]-1))
        X_pca = PCA(n_components=pca_dim, random_state=0).fit_transform(X)

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            max_iter=1000,
            learning_rate='auto',
            random_state=0
        )

        X_tsne = tsne.fit_transform(X_pca)

        if plot:
            self._plot_results(X_tsne, figure_save_path, test_name="TSNE")
                
    def pca_analysis(self, stimulation_interval, analysis_start_time, analysis_duration, figure_save_path, plot=True):
        data = np.array(self.spike_rates_list["spike_rates"])

        truncated_data = self._truncate_data(data, stimulation_interval, analysis_start_time, analysis_duration)
        organized_data = truncated_data.reshape(truncated_data.shape[0], -1)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(organized_data)

        if plot:
            self._plot_results(X_pca, figure_save_path, test_name="PCA")

    def _truncate_data(self, data, stimulation_interval, analysis_start_time, analysis_duration):
        stimulation_size = int(stimulation_interval / self.step_size)
        num_stimulation = int(np.ceil(data.shape[1] / stimulation_size))
        starting_data_size = int(analysis_start_time / self.step_size)
        truncated_data_size = int(analysis_duration / self.step_size)

        truncated_data = []
        for stimulation_idx in range(num_stimulation):
            start_idx = stimulation_idx*stimulation_size + starting_data_size
            end_idx = start_idx + truncated_data_size
            
            truncated_data.append(data[:, start_idx:end_idx])

        return np.array(truncated_data)

    def _plot_results(self, results, figure_save_path, test_name):   

        if self.phase_name.startswith("train"):
            num_pattern = 6
        else:
            num_pattern = 4

        tsne_path = os.path.join(figure_save_path, f"{test_name}_figures")
        os.makedirs(tsne_path, exist_ok=True)
        
        patterns_repo = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
        pattern_colors_repo = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'black', 'yellow', 'cyan']

        patterns = patterns_repo[:num_pattern]
        pattern_colors = pattern_colors_repo[:num_pattern]

        plt.figure(figsize=(8, 6))
        for i in range(num_pattern):
            plt.scatter(
                results[i::num_pattern, 0],
                results[i::num_pattern, 1],
                label=f"Pattern {patterns[i]}",
                alpha=0.8,
                color = pattern_colors[i]
            )

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(tsne_path, f'{self.phase_name}.png'), format='png')
        plt.close('all')
