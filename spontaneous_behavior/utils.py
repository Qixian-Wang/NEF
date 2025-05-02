from brian2 import *
import matplotlib.pyplot as plt

# def plot_w(S1M):
#     plt.rcParams["figure.figsize"] = (20, 10)
#     subplot(311)
#     plot(S1M.t / ms, S1M.w.T / gmax)
#     ylabel('w / wmax')
#     subplot(312)
#     plot(S1M.t / ms, S1M.Apre.T)
#     ylabel('apre')
#     subplot(313)
#     plot(S1M.t / ms, S1M.Apost.T)
#     ylabel('apost')
#     tight_layout()
#     show()
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
    plot(ERM.t / ms, ERM.smooth_rate(window='flat', width=20* ms) * Hz, color='r')
    ylabel('Rate')
    show()

def plot_spikes(ESP):
    plt.rcParams["figure.figsize"] = (20, 6)
    plot(ESP.t / ms, ESP.i, '.r')
    ylabel('Neuron index')
    show()


def weight_rate_spikes(model, T):
    spike_monitor = model['excitatory_spike']
    num_neurons = len(spike_monitor.count)

    # Get spike count in the last T seconds
    spike_counts = np.zeros(num_neurons)
    current_time = spike_monitor.t[-1]  # Last spike time
    start_time = current_time - T

    for i in range(num_neurons):
        times = spike_monitor.spike_trains()[i]
        spike_counts[i] = np.sum((times > start_time) & (times <= current_time))

    rates = spike_counts / T  # in Hz
    return rates
