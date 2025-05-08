from brian2 import *
import matplotlib.pyplot as plt

def plot_w(S1M):
    plt.rcParams["figure.figsize"] = (20, 10)
    subplot(311)
    plot(S1M.t / ms, S1M.w.T / 0.5)
    ylabel('w / wmax')
    subplot(312)
    plot(S1M.t / ms, S1M.ge.T / 0.5)
    ylabel('w / wmax')
    show()
#
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

def plot_rates(ERM):
    plt.rcParams["figure.figsize"] = (20, 6)
    plot(ERM.t / ms, ERM.smooth_rate(window='flat', width=200* ms) * Hz, color='r')
    ylabel('Rate')
    show()

def plot_spikes(ESP):
    plt.rcParams["figure.figsize"] = (20, 6)
    plot(ESP.t / ms, ESP.i, '.r')
    ylabel('Neuron index')
    show()

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
                           sigma=0.1
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
            inst_rate[inst_rate > threshold] -= avalanche_average
            inst_rate[inst_rate < 0] = 0

            rates_sm[nid, :] = np.convolve(inst_rate, gauss, mode='same')

        rates_avg = rates_sm.mean(axis=1)
        # plt.plot(rates_sm[0, :])
        # plt.plot(rates_sm[180, :])
        # plt.show()
        return rates_avg, times, rates_sm

