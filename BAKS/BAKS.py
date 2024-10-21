import os
import sys
from dataclasses import dataclass
from mpi4py import MPI

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
from miv.core.operator.policy import StrictMPIRunner

# Download the sample data
path: str = "/Users/aia/Downloads/2024-08-25_19-49-12"
# path: str = "/home1/10197/qxwang/BAKS_test/2024-08-25_19-49-12"
print('file path:', path)

# Create data modules:
dataset: DataManager = DataManager(data_collection_path=path)
data: DataLoader = dataset[0]

# Create operator modules:
bandpass_filter: Operator = ButterBandpass(lowcut=300, highcut=3000, order=4, tag="bandpass")
spike_detection: Operator = ThresholdCutoff(cutoff=4.0, dead_time=0.002, tag="spikes", progress_bar=True)

data >> bandpass_filter >> spike_detection
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
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
    shape_parameter: np.ndarray = 0.52
    probe_time: np.ndarray = np.linspace(100, 1000, 10)
    tag: str = "BAKS firing rate"
    alpha_testing = False

    def __post_init__(self):
        super().__init__()

    # @cache_call
    def __call__(self, spikestamps) -> List[float]:
        print(f"Hello from rank {rank} of {size} total ranks.")
        comm.Barrier()
        spikestamps = comm.bcast(spikestamps, root=0)
        num_channels = len(spikestamps)
        probe_num = len(self.probe_time)
        scale_parameter_coefficient = 4. / 5.

        # Determine the shape of the arrays based on whether alpha testing is enabled
        if self.alpha_testing:
            firing_rate_array = np.zeros((len(self.shape_parameter), num_channels, probe_num))
            self.rate_ref_array = np.zeros((len(self.shape_parameter), num_channels, probe_num))
        else:
            firing_rate_array = np.zeros((num_channels, probe_num))
            self.rate_ref_array = np.zeros((num_channels, probe_num))

        for channel in range(rank, len(spikestamps), size):
            spikestamp = spikestamps[channel]
            spike_times_local = spikestamp - spikestamp[0]
            assert self.probe_time[-1] < spike_times_local[-1], "Ending time is longer than recording time"

            # Iterate over probes and calculate firing rates and reference rates
            for probe in range(probe_num):
                total_duration = self.probe_time[probe] - spike_times_local[0]
                spike_times = spike_times_local[spike_times_local < total_duration]

                # Rate reference calculations
                rate_ref = np.sum((spike_times_local > self.probe_time[probe] - 0.5) &
                    (spike_times_local < self.probe_time[probe] + 0.5)
                    ) / 1

                # Store the reference rate in the array
                if self.alpha_testing:
                    for a_index, a in enumerate(self.shape_parameter):
                        b = len(spikestamp) ** scale_parameter_coefficient
                        firing_rate, _ = _bayesian_adaptive_kernel_smoother(spike_times, total_duration, a, b)
                        firing_rate_array[a_index, channel, probe] = firing_rate
                        self.rate_ref_array[a_index, channel, probe] = rate_ref
                else:
                    a = self.shape_parameter
                    b = len(spikestamp) ** scale_parameter_coefficient
                    firing_rate, _ = _bayesian_adaptive_kernel_smoother(spike_times, total_duration, a, b)
                    firing_rate_array[channel, probe] = firing_rate
                    self.rate_ref_array[channel, probe] = rate_ref
            print(f"Channel {channel} finished")

        gathered_firing_rates = comm.gather(firing_rate_array, root=0)
        gathered_rate_refs = comm.gather(self.rate_ref_array, root=0)

        if rank == 0:
            firing_rate_array = np.vstack(gathered_firing_rates)
            self.rate_ref_array = np.vstack(gathered_rate_refs)

        comm.Barrier()

        return firing_rate_array, self.rate_ref_array

    # after_call process
    def after_run_print(self, output):
        firing_rate_array, rate_ref_array = output

        if self.alpha_testing:
            mise_list = []
            num_channels = firing_rate_array.shape[1]
            probe_num = firing_rate_array.shape[2]
            for a_index, a in enumerate(self.shape_parameter):
                rate_diff = firing_rate_array[a_index] - rate_ref_array[a_index]
                MISE = np.sqrt(np.sum(rate_diff ** 2))
                mise_list.append((a, MISE))

            # Save MISE values for different alpha values
            summary_file_path = os.path.join(self.analysis_path, 'alpha_comparison.txt')
            with open(summary_file_path, 'w') as summary_file:
                summary_file.write("Alpha, MISE\n")
                for alpha, MISE in mise_list:
                    summary_file.write(f"{alpha}, {MISE}\n")

            alpha_values, MISE_values = zip(*mise_list)
            plt.figure(figsize=(8, 6))
            plt.plot(alpha_values, MISE_values, marker='o', linestyle='-', color='b', label='MISE vs Alpha')
            plt.xlabel('Alpha')
            plt.ylabel('MISE')
            plt.title('MISE as a function of Alpha')
            plt.grid(True)
            plt.legend()
            plot_path = os.path.join(self.analysis_path, 'alpha_vs_MISE.png')
            plt.savefig(plot_path, dpi=300)
            plt.close()

        else:
            num_channels = firing_rate_array.shape[0]
            probe_num = firing_rate_array.shape[1]
            comparison_plots_dir = os.path.join(self.analysis_path, 'comparison_plots')
            os.makedirs(comparison_plots_dir, exist_ok=True)
            for probe_idx in range(probe_num):
                # Extract rates for the current probe time
                rate = firing_rate_array[:, :, probe_idx] if self.alpha_testing else firing_rate_array[:, probe_idx]
                rate_ref = rate_ref_array[:, :, probe_idx] if self.alpha_testing else rate_ref_array[:, probe_idx]

                plt.figure(figsize=(15, 5))
                plt.errorbar(np.arange(num_channels), rate, xerr=0.2, fmt='none', ecolor='blue', capsize=2,
                             label='BAKS_rate')
                plt.errorbar(np.arange(num_channels), rate_ref, xerr=0.2, fmt='none', ecolor='red', capsize=2,
                             label='Reference Rate')
                plt.vlines(np.arange(num_channels), rate, rate_ref, colors='gray',
                           label='Connection' if probe_idx == 0 else "")

                plt.xlabel('Channels')
                plt.ylabel('Firing Rate')
                plt.title(f'Firing Rate Comparison at Probe Time {self.probe_time[probe_idx]:.1f}s')
                plt.legend()
                file_name = os.path.join(comparison_plots_dir,
                                         f'firing_rate_comparison_{self.probe_time[probe_idx]:.1f}s.png')
                plt.savefig(file_name, dpi=300)
                plt.close()

            # Save firing rate data
            summary_file_path = os.path.join(self.analysis_path, 'firing_rate_summary_sorted.txt')
            col_width = 25

            with open(summary_file_path, 'w') as summary_file:
                header = "channel".ljust(col_width)
                header += "".join([f"probe_time: {pt:.1f}s".ljust(col_width) for pt in self.probe_time]) + "\n"
                summary_file.write(header)

                for ch in range(num_channels):
                    line = str(ch).ljust(col_width)
                    line += "".join([f"{rate:.8f}".ljust(col_width) for rate in firing_rate_array[ch]]) + "\n"
                    summary_file.write(line)


# System structure
if __name__ == '__main__':
    firing_rate = BAKSFiringRate()
    spike_detection >> firing_rate
    firing_rate.runner = StrictMPIRunner()
    pipeline = Pipeline(firing_rate)
    pipeline.run(working_directory="results/", verbose=True)