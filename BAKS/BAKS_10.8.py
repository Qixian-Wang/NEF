import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.special as sps

from miv.core.operator import Operator, DataLoader
from miv.core.pipeline import Pipeline
from miv.io.openephys import Data, DataManager
from miv.signal.filter import ButterBandpass, MedianFilter
from miv.signal.spike import ThresholdCutoff
from miv.core.operator import OperatorMixin
from miv.core.operator.wrapper import cache_call

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


def _adaptive_bandwidth(spike_times, time, a, b):
    N = len(spike_times)
    sumnum = 0
    sumdenum = 0
    # Calculate the adaptive bandwidth h
    for i in range(N):
        basicterm = (((time - spike_times[i]) ** 2) / 2 + 1 / b)
        numerator = basicterm ** -a
        denumerator = basicterm ** -(a + 0.5)
        sumnum += numerator
        sumdenum += denumerator

    bandwidth = (sps.gamma(a) / sps.gamma(a + 0.5)) * (sumnum / sumdenum)
    return bandwidth, N

# BAKS Estimate the firing rate
def _baysian___(spike_times, time, bandwidth, N):
    firing_rate = 0
    for j in range(N):
        power = (1 / (np.sqrt(2 * np.pi) * bandwidth)) * np.exp(-((time - spike_times[j]) ** 2) / (2 * bandwidth ** 2))
        firing_rate += power

    return firing_rate


@dataclass
class BAKSFiringRate(OperatorMixin):
    """
    baks = BAKSFiringRate(name_param1=0.32)
    """
    name_param1: float = 0.32
    name_param2: float = (4. / 5.)

    tag: str = "BAKS firing rate"

    def __post_init__(self):
        super().__init__()
        self.firing_rate_list: list[float] = []
        self.bandwidth_list: list[float] = []
        self.endtime: float = 2000.

    @cache_call
    def __call__(self, spike_detection) -> list[float]:

        spikestamp.number_of_channels

        _adaptive_bandwidth(...)

        spike_times = spike_detection.output()
        spikestamps_view = spike_times.get_view(0, self.endtime)
        spikes_last_time = spike_times.get_last_spikestamp()

        # Firing rate for each channel
        for spikstamp in spikestamps:
        for ch in range(0, len(spikestamps_view)):
            spikestamp = spikestamps_view[ch]

            # Calculate firing rate using BAKS function
            spike_times_local = spikestamp - spikestamp.get_first_spikestamp()
            a = self.name_param1
            b_power = self.name_param2
            b = len(spikestamp) ** b_power

            bandwidth, N = adaptive_bandwidth(spike_times_local, spikes_last_time[ch], a, b)
            firing_rate= BAKS(spike_times_local, spikes_last_time[ch], bandwidth, N)
            rate_ref = len(spikestamp) / spikes_last_time[ch]
            self.firing_rate_list.append((ch, firing_rate, rate_ref))
            self.bandwidth_list.append((ch, bandwidth))

        # Organize firing_rate_list and bandwidth_list according to firing rate
        self.firing_rate_list.sort(key=lambda x: x[1], reverse=True)
        sorted_channels = [x[0] for x in self.firing_rate_list]
        self.bandwidth_list = sorted(self.bandwidth_list, key=lambda x: sorted_channels.index(x[0]))

        return self.firing_rate_list

    # after_call process
    def after_run_print(self, output):

        # Get channel information and firing rate from BAKS and Metric
        channel = [item[0] for item in output]
        rate = [item[1] for item in output]


        file_path = "results/spikes/firing_rate_histogram.csv"
        df = pd.read_csv(file_path)

        rate_em_ordered = df.set_index('channel').loc[channel].reset_index()['firing_rate_hz']

        # Plot comparison
        plt.figure(figsize=(15,5))
        plt.errorbar(channel, rate, xerr=0.2, fmt='none', ecolor='blue', capsize=2, label='BAKS_rate')
        plt.errorbar(channel, rate_em_ordered, xerr=0.2, fmt='none', ecolor='red', capsize=2, label='Metric')
        for i in range(len(channel)):
            plt.vlines(channel[i], rate[i], rate_em_ordered[i], colors='gray',
                       label='Connection' if i == 0 else "")
        plt.xlabel('channels')
        plt.ylabel('firing rate')
        plt.legend()
        file_name = os.path.join(self.analysis_path, ".png")
        plt.savefig(file_name, dpi=300)
        plt.close('all')

        # Calculate difference between Metric and BAKS
        MISE = 0
        for i in range(len(channel)):
            MISE += np.sqrt((rate[i] - rate_em_ordered[i]) ** 2)
        print('MISE:', MISE)

        # Save data
        os.makedirs(self.analysis_path, exist_ok=True)
        summary_file_path = os.path.join(self.analysis_path, 'firing_rate_summary_sorted.txt')

        with open(summary_file_path, 'w') as summary_file:
            summary_file.write("channel, firing_rate_hz, ref_firing_rate_spikes/time\n")

            for ch, firing_rate, rate_ref in self.firing_rate_list:
                summary_file.write(f"{ch}, {firing_rate}, {rate_ref}\n")

        return MISE

# System structure
if __name__ == '__main__':
    firing_rate = BAKSFiringRate()
    spike_detection >> firing_rate

    pipeline = Pipeline(firing_rate)
    pipeline.run(working_directory="results/", verbose=True)
