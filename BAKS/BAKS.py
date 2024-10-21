import os
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.special as sps

from typing import List
from miv.core.operator import Operator, DataLoader
from miv.core.pipeline import Pipeline
from miv.io.openephys import Data, DataManager
from miv.signal.filter import ButterBandpass, MedianFilter
from miv.signal.spike import ThresholdCutoff
from miv.core.operator import OperatorMixin
from miv.core.operator.wrapper import cache_call

from BAKS_benchmark import spike_times

# Download the sample data
path: str = "/Users/aia/Downloads/2024-08-25_19-49-12"
print('file path:', path)

# Create data modules:
dataset: DataManager = DataManager(data_collection_path=path)
data: DataLoader = dataset[0]

# Create operator modules:
bandpass_filter: Operator = ButterBandpass(lowcut=300, highcut=3000, order=4, tag="bandpass")
spike_detection: Operator = ThresholdCutoff(cutoff=4.0, dead_time=0.002, tag="spikes", progress_bar=True)

data >> bandpass_filter >> spike_detection

# BAKS Estimate the firing rate
def _bayesian_adaptive_kernel_smoother(spike_times, time, a, b):
    spike_number_total = len(spike_times)
    numerator: float = 0
    denumerator: float = 0
    # Calculate the adaptive bandwidth h
    for i in range(spike_number_total):
        val = (((time - spike_times[i]) ** 2) / 2 + 1 / b)
        numerator += val ** -a
        denumerator += val ** -(a + 0.5)

    bandwidth = (sps.gamma(a) / sps.gamma(a + 0.5)) * (numerator / denumerator)

    firing_rate = 0
    for j in range(spike_number_total):
        power = (1 / (np.sqrt(2 * np.pi) * bandwidth)) * np.exp(-((time - spike_times[j]) ** 2) / (2 * bandwidth ** 2))
        firing_rate += power

    return firing_rate, bandwidth

@dataclass
class BAKSFiringRate(OperatorMixin):
    shape_parameter: float = 0.32
    probe_time: np.ndarray = np.linspace(10, 1000, 3)
    tag: str = "BAKS firing rate"

    def __post_init__(self):
        super().__init__()
        self.endtime: float = 2000.

    # @cache_call
    def __call__(self, spikestamps) -> List[float]:
        num_channels = len(spikestamps)
        probe_num = len(self.probe_time)
        firing_rate_array = np.zeros((num_channels, probe_num))
        self.rate_ref_array = np.zeros((num_channels, probe_num))

        # Firing rate for each channel
        for channel, spikestamp in enumerate(spikestamps):
            spike_times_local = spikestamp - spikestamp[0]

            assert self.probe_time[-1] < spike_times_local[-1], "Ending time is longer than recording time"

            for probe in range (probe_num):
                total_duration = self.probe_time[probe] - spike_times_local[0]
                spike_times = spike_times_local[spike_times_local < total_duration]

                # Calculate firing rate using BAKS function
                scale_parameter_coefficient: float = 4. / 5.
                a = self.shape_parameter
                b = len(spikestamp) ** scale_parameter_coefficient

                firing_rate, bandwidth= _bayesian_adaptive_kernel_smoother(spike_times, total_duration, a, b)
                firing_rate_array[channel, probe] = firing_rate


                rate_ref = np.sum(
                    (spike_times_local > probe - 5) & (spike_times_local < probe + 5)) / 10

                self.rate_ref_array[channel, probe] = rate_ref

        return firing_rate_array, self.rate_ref_array

    # after_call process
    def after_run_print(self, output):
        firing_rate_array = output[0]
        rate_ref_array = output[1]
        num_channels = len(firing_rate_array)
        probe_num = len(firing_rate_array[0])

        comparison_plots_dir = os.path.join(self.analysis_path, 'comparison_plots')
        os.makedirs(comparison_plots_dir, exist_ok=True)

        # Create figures for each probe time
        for probe_idx in range(probe_num):
            rate = firing_rate_array[:, probe_idx]
            rate_ref = rate_ref_array[:, probe_idx]

            channel = np.arange(num_channels)

            plt.figure(figsize=(15, 5))
            plt.errorbar(channel, rate, xerr=0.2, fmt='none', ecolor='blue', capsize=2, label='BAKS_rate')
            plt.errorbar(channel, rate_ref, xerr=0.2, fmt='none', ecolor='red', capsize=2, label='Reference Rate')

            for i in range(len(channel)):
                plt.vlines(channel[i], rate[i], rate_ref[i], colors='gray', label='Connection' if i == 0 else "")

            plt.xlabel('Channels')
            plt.ylabel('Firing Rate')
            plt.title(f'Firing Rate Comparison at Probe Time {self.probe_time[probe_idx]:.1f}s')
            plt.legend()

            file_name = os.path.join(comparison_plots_dir,
                                     f'firing_rate_comparison_{self.probe_time[probe_idx]:.1f}s.png')
            plt.savefig(file_name, dpi=300)
            plt.close()

            # Calculate difference between Metric and BAKS
            MISE = 0
            for i in range(len(channel)):
                MISE += np.sqrt((rate[i] - rate_ref[i]) ** 2)
                print(f'MISE: for rank {i} is {MISE}')

        # Save data
        os.makedirs(self.analysis_path, exist_ok=True)
        summary_file_path = os.path.join(self.analysis_path, 'firing_rate_summary_sorted.txt')

        col_width = 25

        with open(summary_file_path, 'w') as summary_file:
            header = "channel".ljust(col_width)
            header += "".join(
                [f"probe_time: {probe_time:.1f}s".ljust(col_width) for probe_time in self.probe_time]
            ) + "\n"
            summary_file.write(header)

            for ch in range(num_channels):
                line = str(ch).ljust(col_width)
                line += "".join([f"{rate:.8f}".ljust(col_width) for rate in firing_rate_array[ch]]) + "\n"
                summary_file.write(line)

        return output

# System structure
if __name__ == '__main__':
    firing_rate = BAKSFiringRate()
    spike_detection >> firing_rate

    pipeline = Pipeline(firing_rate)
    pipeline.run(working_directory="results/", verbose=True)