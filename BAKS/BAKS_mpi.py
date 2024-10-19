import os
import sys
from dataclasses import dataclass

from miv.typing import SignalType
from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.special as sps
import scipy.signal as sps_sig

from typing import List
from miv.core.datatype import Signal
from miv.core.operator import Operator, DataLoader
from miv.io.openephys import Data, DataManager
from miv.core.operator import OperatorMixin
from BAKS_MPI_filter import ButterBandpass

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
    scale_parameter_coefficient: float = 4. / 5.
    tag: str = "BAKS firing rate"

    def __post_init__(self):
        super().__init__()
        self.firing_rate_list: List[float] = []
        self.bandwidth_list: List[float] = []
        self.endtime: float = 2000.

    # @cache_call
    def __call__(self, spikestamps) -> List[float]:

        comm.Barrier()
        spikestamps = comm.bcast(spikestamps, root=0)

        # Firing rate for each channel
        for channel in range(rank, len(spikestamps), size):
            spikestamp = spikestamps[channel]
            total_duration = spikestamp[-1] - spikestamp[0]
            spike_times_local = spikestamp - spikestamp[0]

            # Calculate firing rate using BAKS function
            a = self.shape_parameter
            b = len(spikestamp) ** self.scale_parameter_coefficient

            firing_rate, bandwidth= _bayesian_adaptive_kernel_smoother(spike_times_local, total_duration, a, b)
            rate_ref = len(spikestamp) / total_duration
            self.firing_rate_list.append((channel, firing_rate, rate_ref))
            self.bandwidth_list.append((channel, bandwidth))
            print(f'channel {channel} is handled by rank {rank}')

        # Organize firing_rate_list and bandwidth_list according to firing rate
        comm.Barrier()
        gathered_firing_rate = comm.gather(self.firing_rate_list, root=0)

        if rank != 0:
            MPI.Finalize()
            sys.exit()

        flattened_firing_rate = [item for sublist in gathered_firing_rate for item in sublist]
        flattened_firing_rate.sort(key=lambda x: x[1], reverse=True)

        return flattened_firing_rate

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
        file_name = os.path.join(self.analysis_path, 'firing_rate_comparison.png')
        plt.savefig(file_name, dpi = 300)
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

            for ch, firing_rate, rate_ref in output:
                summary_file.write(f"{ch}, {firing_rate}, {rate_ref}\n")

        return MISE
# System structure
if __name__ == '__main__':

    # def bandpass_filter(signal):
    #     rate = signal.rate
    #     critical_frequency = [300, 3000]
    #     sos = sps_sig.butter(
    #         5,
    #         critical_frequency,
    #         fs = rate,
    #         btype = "bandpass",
    #         output = "sos",
    #     )
    #     y = signal.data.copy()
    #     y[:, 0] = sps_sig.sosfiltfilt(sos, signal.data[:, 0])
    #
    #     return Signal(data=y, timestamps=signal.timestamps, rate=rate)


    def process_channel(data):
        bandpass_filter: Operator = ButterBandpass(lowcut=300, highcut=3000, order=4, tag="bandpass")
        bandpass_filter.cacher.policy = "OFF"
        filtered_data = bandpass_filter(data)
        print(f"data from rank {rank} filtered")
        return filtered_data

    # def data_segment(signal, parts):
    #     length = signal.data.shape[0]
    #     signal_truncate = Signal(data=signal.data[:int(length/parts), :], timestamps=signal.timestamps[:int(length/parts)], rate=signal.rate)
    #     for i in range (1, parts):
    #         signal_truncate.data =  np.append(signal_truncate.data, signal.data[int(length/parts)*i:int(length/parts)*(i+1), :], axis=1)
    #         print(signal_truncate.data.shape)
    #
    #     return signal_truncate

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        data_processing_time_start = MPI.Wtime()
        # path: str = "/Users/aia/Downloads/2024-08-25_19-49-12"
        path: str = "/home1/10197/qxwang/BAKS_test/2024-08-25_19-49-12"
        print('file path:', path)

        dataset: DataManager = DataManager(data_collection_path=path)
        data: DataLoader = dataset[0]

        data_iterator = data.load()
        data_full = next(data_iterator)  # Get the first signal

        # Optionally append more data if needed
        for current_signal in data_iterator:
            data_full.data = np.append(data_full.data, current_signal.data, axis=0)
            data_full.timestamps = np.append(data_full.timestamps, current_signal.timestamps)

        number_of_channels = data_full.number_of_channels
    else:
        data_full = None
        number_of_channels = None

    number_of_channels = comm.bcast(number_of_channels, root=0)
    data_full_rearrange = None

    for channel in range(number_of_channels):
        target_rank = channel % size
        if target_rank != 0:
            if rank == 0:
                data_channel = data_full.data[:, channel]
                comm.send((data_channel, data_full.timestamps, data_full.rate), dest=target_rank)
                print(f"data {channel} sent")

            elif rank == target_rank:
                data_channel, timestamps, rate = comm.recv(source=0)
                print(f"data {channel} recieved")
                if data_full is None:
                    data_full = Signal(data=data_channel[:, np.newaxis], timestamps=timestamps, rate=rate)
                else:
                    data_full.append(data_channel[:, np.newaxis])

        if target_rank == 0:
            if rank == 0:
                data_channel = data_full.data[:, channel]
                if data_full_rearrange is None:
                    data_full_rearrange = Signal(data=data_channel[:, np.newaxis], timestamps=data_full.timestamps, rate=data_full.rate)
                else:
                    data_full_rearrange.append(data_channel[:, np.newaxis])

    if rank == 0:
        data_full = data_full_rearrange

    comm.barrier()
    if rank == 0:
        data_processing_time_end =MPI.Wtime()
        data_processing_time = data_processing_time_end - data_processing_time_start
        filtering_time_start = MPI.Wtime()

    if data_full != None:
        filtered_data = process_channel(data_full)

    if rank == 0:
        filtering_time_end = MPI.Wtime()
        filtering_time = filtering_time_end - filtering_time_start
        print(data_processing_time)
        summary_file_path = os.path.join('/home1/10197/qxwang/BAKS_test/', 'firing_rate_summary_sorted.txt')
        with open(summary_file_path, 'w') as summary_file:
            summary_file.write(f"data processing time: {data_processing_time}\n")
            summary_file.write(f"filtering time: {filtering_time}\n")
            summary_file.write(f"total time: {data_processing_time + filtering_time}\n")
