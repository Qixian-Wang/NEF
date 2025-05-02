from NEF.spontaneous_behavior.utils import *
from numpy.random import lognormal

class Model:
    def __init__(self, hebbian_lr, synaptic_delay, refractory, sigma_ge=0, sigma_v=0.0, stimulation_amp=4, mode="spontaneous"):
        app = {}
        self.mode = mode

        # input poisson group
        app['poisson_group'] = PoissonGroup(
            n_input,
            rates=np.zeros(n_input) * Hz,
            name='poisson_group'
        )

        @network_operation(dt=defaultclock.dt)
        def update_rates():
            if self.mode == "spontaneous":
                rates = np.zeros(n_input) * Hz
                app['poisson_group'].rates = rates
            elif self.mode == "patternA":
                rates = np.zeros(n_input) * Hz
                rates[:50] = stimulation_amp * Hz
                app['poisson_group'].rates = rates
            elif self.mode == "patternB":
                rates = np.zeros(n_input) * Hz
                rates[25:75] = stimulation_amp * Hz
                app['poisson_group'].rates = rates
            elif self.mode == "patternC":
                rates = np.zeros(n_input) * Hz
                rates[50:100] = stimulation_amp * Hz
                app['poisson_group'].rates = rates
            elif self.mode == "patternD":
                rates = np.zeros(n_input) * Hz
                rates[70:125] = stimulation_amp * Hz
                app['poisson_group'].rates = rates

        # excitatory group
        neuron_e = '''
            dv/dt = (ge*(0*mV-v) + (v_rest_e-v))/ (25*ms) + sigma_v * xi_v * sqrt(1/ms): volt
            dge/dt = (-ge + sigma_ge * xi_e * sqrt(ms)) / (5*ms) : 1
            sigma_ge : 1
            sigma_v : volt
            '''

        app['excitatory_group'] = NeuronGroup(num_neuron, neuron_e, threshold='v>v_thresh_e', refractory=refractory * ms, reset='v=v_reset_e',
                                              method='euler', name='excitatory_group')
        app['excitatory_group'].v = v_rest_e
        app['excitatory_group'].sigma_ge = sigma_ge
        app['excitatory_group'].sigma_v = sigma_v * mV

        # poisson generators one-to-all excitatory neurons with plastic connections
        app['S1'] = Synapses(app['poisson_group'],
                             app['excitatory_group'],
                             stdp, on_pre=pre,
                             on_post=post,
                             method='euler',
                             name='S1')

        app['S1'].connect(j = 'i')
        app['S1'].w = 'rand()*gmax'  # random weights initialisation
        app['S1'].lr = 0

        # excitatory neurons to excitatory neurons
        app['S2'] = Synapses(app['excitatory_group'],
                             app['excitatory_group'],
                             stdp_depr,
                             on_pre=pre_depr,
                             on_post=post_depr,
                             method='euler',
                             name='S2')
        app['S2'].connect(condition='i != j', p=p_conn)
        app['S2'].delay = synaptic_delay * ms

        mean = np.log(0.5*gmax)
        sigma = 1.0
        app['S2'].w = clip(lognormal(mean, sigma, size=len(app['S2'].w)), 0, gmax)
        app['S2'].lr = hebbian_lr

        # Monitors
        app['poisson_monitor'] = StateMonitor(app['poisson_group'], ['rates'], record=True, name='poisson_monitor')
        app['excitatory_spike'] = SpikeMonitor(app['excitatory_group'], name='excitatory_spike')
        app['ESM'] = StateMonitor(app['excitatory_group'], ['v'], record=True, name='ESM')
        app['ERM'] = PopulationRateMonitor(app['excitatory_group'], name='excitatory_group_rate')
        app['S2M'] = StateMonitor(app['S2'], ['w', 'Apre', 'Apost'], record=range(5), name='S2M')

        self.net = Network(app.values(), update_rates)


    def __getitem__(self, key):
        return self.net[key]

    def set_mode(self, new_mode):
        self.mode = new_mode

    def train(self, duration):
        self.net.run(duration * second)


if __name__ == '__main__':
    defaultclock.dt = 1 * ms
    seed(42)
    n_input = 200
    num_neuron = 200

    v_rest_e = -65. * mV
    v_reset_e = -75. * mV
    v_thresh_e = -50. * mV

    K = 4
    p_conn = K / (num_neuron - 1)
    taupre = 20 * ms
    taupost = taupre
    gmax = 1.7
    dApre = .01
    dApost = -dApre * taupre / taupost * 1
    dApost *= gmax
    dApre *= gmax
    U = 0.3
    tau_rec = 300 * ms

    # Apre and Apost - presynaptic and postsynaptic traces
    stdp = '''
    w : 1
    lr : 1 (shared)
    dApre/dt = -Apre / taupre : 1 (event-driven)
    dApost/dt = -Apost / taupost : 1 (event-driven)
    '''
    pre = '''
    ge += w
    Apre += dApre
    w = clip(w + lr*Apost, 0, gmax)
    '''
    post = '''
    Apost += dApost
    w = clip(w + lr*Apre, 0, gmax)
    '''

    stdp_depr = '''
    w : 1
    lr : 1 (shared)
    dApre/dt  = -Apre / taupre  : 1 (event-driven)
    dApost/dt = -Apost / taupost: 1 (event-driven)
    dres/dt = (1 - res)/tau_rec     : 1 (event-driven)
    '''
    pre_depr = f'''
    ge += w * res
    Apre += dApre
    w = clip(w + lr*Apost, 0, gmax)
    res *= (1 - {U})
    '''
    post_depr = '''
    Apost += dApost
    w = clip(w + lr*Apre, 0, gmax)
    '''

    model = Model(hebbian_lr=0, synaptic_delay=0, refractory=5, sigma_ge=0, sigma_v=1.2, stimulation_amp=4, mode="spontaneous")
    time_spon = 30
    time_sti = 1
    model.train(duration=time_spon)
    # model.set_mode("patternA")
    # model.train(duration=5)
    # model.set_mode("patternB")
    # model.train(duration=5)
    # model.set_mode("patternC")
    # model.train(duration=5)
    # model.set_mode("patternA")
    # model.train(duration=5)
    # model.set_mode("patternB")
    # model.train(duration=5)
    # model.set_mode("patternC")
    # model.train(duration=5)

    # sample_number = 5
    # rate_list = []
    # for mode in ["patternA", "patternB", "patternC", "patternD"]:
    #     model.set_mode(mode)
    #     for k in range(sample_number):
    #         model.train(duration=time_sti)
    #         rate = weight_rate_spikes(model, T=time_sti*second)
    #         rate_list.append(rate)
    # for mode in ["patternA", "patternB", "patternC", "patternD"]:
    #     model.set_mode(mode)
    #     for k in range(sample_number):
    #         model.train(duration=time_sti)
    #         rate = weight_rate_spikes(model, T=time_sti*second)
    #         rate_list.append(rate)
    # for mode in ["patternA", "patternB", "patternC", "patternD"]:
    #     model.set_mode(mode)
    #     for k in range(sample_number):
    #         model.train(duration=time_sti)
    #         rate = weight_rate_spikes(model, T=time_sti*second)
    #         rate_list.append(rate)
    #
    # rate_list = np.array(rate_list)
    # np.save('rate_list2.npy', rate_list)
    # plot_w(model["S2M"])
    plot_rates(model['excitatory_group_rate'])
    plot_spikes(model['excitatory_spike'])