import os
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt


class MovingAverageFiringRate:
    def __init__(self, spiketrains, comm, phase_time):
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.comm = comm

        self.spiketrains = spiketrains
        self.spike_data = list(self.spiketrains.data)

        self.phase_name = phase_time[0]
        self.duration = phase_time[1]


    def select_reading_channels(self, excluded_reading_channels, reading_channels):
        self.reading_spike_data = []
        assert len(self.spike_data) == self.spiketrains.number_of_channels, "wrong number of channels"
        for channel in reading_channels:
            channel_index = int(channel[-3:])
            if channel[0] == "A" and channel not in excluded_reading_channels:
                self.reading_spike_data.append([channel, self.spike_data[channel_index]])
            elif channel[0] == "B" and channel not in excluded_reading_channels:
                self.reading_spike_data.append([channel, self.spike_data[channel_index + 32]])
            elif channel[0] == "C" and channel not in excluded_reading_channels:
                self.reading_spike_data.append([channel, self.spike_data[channel_index + 64]])
            elif channel[0] == "D" and channel not in excluded_reading_channels:
                self.reading_spike_data.append([channel, self.spike_data[channel_index + 96]])

        self.data_for_rank = self.reading_spike_data[self.rank::self.size]

    
    def moving_average_firing_rate(self, figure_save_path, window_size=0.25, step_size=0.1, gaussian_sigma=1, plot_firing_rate=False): 
        phase_result = self._calculate_phase_firing_rate(
            window_size, step_size, gaussian_sigma
        )

        if plot_firing_rate:
            self._plot_phase_firing_rate(
                phase_result, figure_save_path, step_size
            )            

        self.comm.Barrier()
        merged_phase_result = self._collect_results(phase_result)
        
        return merged_phase_result
    
    def _calculate_phase_firing_rate(self, window_size, step_size, gaussian_sigma):
        window_starts = np.arange(0, self.duration - window_size + step_size, step_size)
        window_centers = window_starts + window_size / 2
        num_windows = len(window_starts)
        
        channel_names = [ch_data[0] for ch_data in self.data_for_rank]
        spike_times_list = [ch_data[1] for ch_data in self.data_for_rank]
        num_channels = len(channel_names)
        
        spike_rates_matrix = np.zeros((num_channels, num_windows))
        
        for ch_idx, spike_times in enumerate(spike_times_list):
            start_indices = np.searchsorted(spike_times, window_starts, side='left')
            end_indices = np.searchsorted(spike_times, window_starts + window_size, side='right')
            spike_counts = end_indices - start_indices
            spike_rates_matrix[ch_idx, :] = spike_counts / window_size
        
        spike_rates_matrix = gaussian_filter1d(spike_rates_matrix, sigma=gaussian_sigma, mode='reflect')
        
        return {
            'channel_names': channel_names,
            'spike_rates': spike_rates_matrix,
            'window_centers': window_centers
        }
    
    def _collect_results(self, phase_result):
        all_results = self.comm.gather(phase_result, root=0)
        
        if self.rank == 0:
            merged_channel_names = []
            merged_spike_rates = []
            
            for result in all_results:
                merged_channel_names.extend(result['channel_names'])
                merged_spike_rates.extend(result['spike_rates'])
            
            merged_result = {
                'channel_names': merged_channel_names,
                'spike_rates': merged_spike_rates,
            }
        else:
            merged_result = None
        
        merged_result = self.comm.bcast(merged_result, root=0)
        self.comm.Barrier()

        return merged_result
    

    def _plot_phase_firing_rate(self, phase_result, data_save_path, step_size, image_time=10):
        firing_rate_path = os.path.join(data_save_path, "firing_rate")
        os.makedirs(firing_rate_path, exist_ok=True)
        
        channel_names = phase_result['channel_names']
        spike_rates_matrix = phase_result['spike_rates']
        window_centers = phase_result['window_centers']

        phase_path = os.path.join(firing_rate_path, f"{self.phase_name}")
        os.makedirs(phase_path, exist_ok=True)
        
        num_images = int(np.ceil(self.duration / image_time))
        
        for image_idx in range(num_images):
            image_start_time = image_time * image_idx
            actual_image_duration = min(image_time, self.duration - image_idx * image_time)

            time_interval_path = os.path.join(phase_path, f"starting_from_{image_start_time}")
            os.makedirs(time_interval_path, exist_ok=True)
            
            windows_per_step = int(1 / step_size) if step_size > 0 else 1
            num_windows_per_image = int(actual_image_duration * windows_per_step)
            start_window = int(image_idx * image_time * windows_per_step)
            end_window = min(start_window + num_windows_per_image, len(window_centers))
            
            t_slice = window_centers[start_window:end_window]
            spike_data_slice = spike_rates_matrix[:, start_window:end_window]
            
            for ch_idx, channel_name in enumerate(channel_names):
                if spike_data_slice.shape[1] > 0:
                    plt.figure(figsize=(12, 8))
                    plt.plot(t_slice, spike_data_slice[ch_idx, :], 
                            linewidth=2, color='red', alpha=0.8)
                    plt.ylabel("Firing rate (Hz)", fontsize=12)
                    plt.xlabel("Time (s)", fontsize=12)
                    plt.title(f"Channel {channel_name} Firing rate", fontsize=14)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    save_path = os.path.join(time_interval_path, f"channel_{channel_name}_rank_{self.rank}.png")
                    plt.savefig(save_path, dpi=150)
                    plt.close('all')