from brian2 import *
import matplotlib.pyplot as plt
import os
import math

def plot_figures(model, spike_time, group_index, n_per_group, n_groups, dt_bin=2*ms, smooth_width=200*ms):
    save_path = 'plots'
    os.makedirs(save_path, exist_ok=True)

    # Plot weight
    plt.figure(figsize=(20, 10))
    monitor = model['S2_monitor']
    subplot(211)
    plot(monitor.t / ms, monitor.w.T)
    ylabel('synaptic weight')
    subplot(212)
    plot(monitor.t / ms, monitor.ge.T)
    ylabel('synaptic conductance')
    plt.savefig(os.path.join(save_path, 'weight.png'), format='png')
    plt.close('all')

    # Plot firing rate
    monitor = model['excitatory_spike']
    for fig_index in range(math.ceil(max(monitor.t)/10)):
        t_all = monitor.t
        i_all = monitor.i
        mask = (t_all >= fig_index * 10 * second) & (t_all <= (fig_index+1) * 10 * second)

        t_all_masked = t_all[mask]
        i_all_masked = i_all[mask]

        group_all = i_all_masked // n_per_group

        t_all_ms = t_all_masked / ms
        dt_bin_ms = dt_bin / ms
        bins = np.arange(t_all_ms.min(), t_all_ms.max() + dt_bin_ms, dt_bin_ms)

        counts = np.zeros((n_groups, len(bins) - 1))
        for g in range(n_groups):
            mask = (group_all == g)
            counts[g], _ = np.histogram(t_all_ms[mask], bins=bins)

        bin_sec = dt_bin_ms / 1000.0
        rates = counts / (n_per_group * bin_sec)

        window_bins = int((smooth_width / ms) / dt_bin_ms)
        if window_bins > 1:
            kernel = np.ones(window_bins) / window_bins
            rates_smooth = np.array([np.convolve(r, kernel, mode='same') for r in rates])
        else:
            rates_smooth = rates

        time_centers = bins[:-1] + dt_bin_ms / 2
        plt.figure(figsize=(20, 6))
        plt.plot(time_centers, rates_smooth.mean(axis=0))
        plt.xlabel('Time (ms)')
        plt.ylabel('Firing rate (Hz)')
        plt.title('Grouped firing rates')
        plt.savefig(os.path.join(save_path, f'firing_rate{fig_index}.png'), format='png')
        plt.close('all')

    # Plot rastor plot
    for fig_index in range(math.ceil(max(spike_time)/10)):
        mask = (spike_time >= fig_index * 10) & (spike_time <= (fig_index+1) * 10)
        spike_time_masked = spike_time[mask]
        group_index_maksed = group_index[mask]

        plt.figure(figsize=(12, 4))
        plt.plot(spike_time_masked / ms, group_index_maksed, '.', color='r', markersize=1)
        plt.xlabel('Time (ms)')
        plt.ylabel('Group index')
        plt.title('Spike raster (grouped)')
        plt.savefig(os.path.join(save_path, f'rastor_plot{fig_index}.png'), format='png', dpi=600)
        plt.close('all')


# def plot_v(ESM, ISM, neuron=13):
#     plt.rcParams["figure.figsize"] = (20, 6)
#     cnt = -50000  # tail
#     plot(ESM.t[cnt:] / ms, ESM.v[neuron][cnt:] / mV, label='exc', color='r')
#     plot(ISM.t[cnt:] / ms, ISM.v[neuron][cnt:] / mV, label='inh', color='b')
#     plt.axhline(y=v_thresh_e / mV, color='pink', label='v_thresh_e')
#     plt.axhline(y=v_thresh_i / mV, color='silver', label='v_thresh_i')
#     legend()
#     ylabel('v')
#     show()

def process_spikes(model):
    monitor = model['excitatory_spike']
    groups = model.neuron_group
    spike_trains = monitor.spike_trains()

    times_list = []
    group_id_list = []
    for g in range(len(groups)):
        neuron_list = groups[g]
        if len(neuron_list) == 0:
            continue
        group_times = np.concatenate([spike_trains[i] for i in neuron_list])
        group_ids = np.full_like(group_times, g, dtype=int)
        times_list.append(group_times)
        group_id_list.append(group_ids)

    all_times = np.concatenate(times_list)
    all_groups = np.concatenate(group_id_list)
    return all_times, all_groups

def ColoredGaussianDataset(D=64, top_eigs=[7, 6, 5, 4], low=(0, 0.5), seed=42):
    D = D
    top_eigs = top_eigs
    low = low
    np.random.seed(seed)
    m = len(top_eigs)
    Q, _ = np.linalg.qr(np.random.randn(D, D))
    low_vals = np.random.uniform(low[0], low[1], size=D-m)
    eigvals = np.concatenate([top_eigs, low_vals])
    eigvals = np.sort(eigvals)[::-1]
    C = Q @ np.diag(eigvals) @ Q.T
    L = np.linalg.cholesky(C)
    z = np.random.randn(D)
    x = L @ z
    return x


def smooth_rate_per_neuron(spike_i, spike_t, num_neurons,
                           stim_start, stim_duration,
                           dt=0.1,
                           sigma=0.1,
                           remove_burst=False
                           ):

    for stimulation in range(len(stim_start)):
        t_start = stim_start[stimulation]
        t_end = stim_start[stimulation] + stim_duration[stimulation]

        bins = np.arange(t_start, t_end + dt, dt)
        times = bins[:-1] + dt/2

        half = int(3 * sigma / dt)
        kernel_times = np.linspace(-half*dt, half*dt, 2*half+1)
        gauss = np.exp(-kernel_times**2/(2*sigma**2))
        gauss /= (gauss.sum() * dt)

        n_bins = len(bins) - 1
        rates_sm = np.zeros((num_neurons, n_bins))

        for nid in range(num_neurons):
            mask = (spike_i == nid) & (spike_t >= t_start) & (spike_t < t_end)
            ts = spike_t[mask]
            counts, _ = np.histogram(ts, bins=bins)
            inst_rate = counts / dt

            valid_rate = inst_rate[inst_rate>0]
            mean_rate = valid_rate.mean()
            std_rate = valid_rate.std()
            threshold = mean_rate + 0.5 * std_rate
            avalanche_average = inst_rate[inst_rate>threshold].mean()
            inst_rate[inst_rate > threshold] -= avalanche_average if remove_burst else 0
            inst_rate[inst_rate < 0] = 0

            rates_sm[nid, :] = np.convolve(inst_rate, gauss, mode='same')

        rates_avg = rates_sm.mean(axis=1)
        # plt.plot(rates_sm[0, :])
        # plt.plot(rates_sm[180, :])
        # plt.show()
        return rates_avg, times, rates_sm

def compute_rates(spike_mon, n_per_group, n_groups, dt_bin=10*ms, smooth_width=200*ms):
    t_all = spike_mon.t
    i_all = spike_mon.i

    group_all = i_all // n_per_group

    t_all_ms = t_all / ms
    dt_bin_ms = dt_bin / ms
    t_max = t_all_ms.max()
    bins = np.arange(0, t_max + dt_bin_ms, dt_bin_ms)

    counts = np.zeros((n_groups, len(bins)-1))
    for g in range(n_groups):
        mask = (group_all == g)
        counts[g], _ = np.histogram(t_all_ms[mask], bins=bins)

    bin_sec = dt_bin_ms / 1000.0
    rates = counts / (n_per_group * bin_sec)  # shape (n_groups, n_time_bins)

    window_bins = int((smooth_width/ms) / dt_bin_ms)
    if window_bins > 1:
        kernel = np.ones(window_bins) / window_bins
        rates_smooth = np.array([np.convolve(r, kernel, mode='same') for r in rates])
    else:
        rates_smooth = rates
    bin_centers_ms = (bins[:-1] + bins[1:]) / 2.0
    t_seconds = bin_centers_ms / 1000.0

    return t_seconds, rates_smooth.mean(axis=0)