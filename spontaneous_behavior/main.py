from NEF.spontaneous_behavior.utils import *
from numpy.random import lognormal, default_rng
from sklearn.decomposition import PCA

class Model:
    def __init__(self, hebbian_lr, refractory, sigma_i=0.0, stimulation_amp=4):
        self.app = {}
        self.stimulation_amp = stimulation_amp

        # input poisson group
        self.app['poisson_group'] = PoissonGroup(
            n_input,
            rates=np.zeros(n_input) * Hz,
            name='poisson_group'
        )
        self.rates = np.zeros(n_input) * Hz
        self.app['poisson_group'].rates = self.rates

        # excitatory group
        neuron_e = '''
        dv/dt = (ge*(0*mV - v) + (v_rest_e - v)) / tau_m + (- theta * v + sigma_i * xi_i * sqrt(ms)) / tau_m: volt
        dge/dt = -ge / tau_syn : 1
        dr/dt  = -r/tau_r : Hz
        sigma_i : volt
        tau_m : second
        tau_syn : second
        tau_r   : second
        theta : 1
        '''

        self.app['excitatory_group'] = NeuronGroup(num_neuron, neuron_e, threshold='v>v_thresh_e', refractory=refractory * ms, reset='v=v_reset_e; r+=1/tau_r',
                                              method='euler', name='excitatory_group')
        self.app['excitatory_group'].v = v_rest_e
        self.app['excitatory_group'].sigma_i = sigma_i * mV
        self.app['excitatory_group'].tau_m = 25 * ms
        self.app['excitatory_group'].tau_syn = 5 * ms
        self.app['excitatory_group'].tau_r = 50 * ms
        self.app['excitatory_group'].theta = 1e-1

        oja = Equations('''
        w  : 1
        lr : 1 (shared)
        ''')

        # poisson generators one-to-all excitatory neurons with plastic connections
        self.app['S1'] = Synapses(self.app['poisson_group'],
                             self.app['excitatory_group'],
                             model=oja,
                             on_pre='ge_post += w',
                             method='euler',
                             name='S1')

        self.app['S1'].connect(j = 'i')
        self.app['S1'].w = 'rand()*gmax'  # random weights initialisation
        self.app['S1'].lr = hebbian_lr
        self.app['S1'].run_regularly('''
        x = rates_pre/Hz
        y = r_post/Hz
        w += lr*(y*x - y**2*w)*(dt/second)
        w = clip(w, 0, gmax)
        ''', dt=defaultclock.dt)


        # excitatory neurons to excitatory neurons
        self.app['S2'] = Synapses(self.app['excitatory_group'],
                             self.app['excitatory_group'],
                             model=oja,
                             on_pre='ge_post += w',
                             method='euler',
                             name='S2')
        self.app['S2'].connect(condition='i != j', p=p_conn)
        self.app['S2'].w = 'rand()*gmax'
        self.app['S2'].lr = hebbian_lr
        self.app['S2'].run_regularly('''
        x = r_pre/Hz
        y = r_post/Hz
        w += lr*(y*x - y**2*w)*(dt/second)
        w = clip(w, 0, gmax)
        ''', dt=defaultclock.dt)

        # Monitors
        self.app['poisson_monitor'] = StateMonitor(self.app['poisson_group'], ['rates'], record=True, name='poisson_monitor')
        self.app['excitatory_spike'] = SpikeMonitor(self.app['excitatory_group'], name='excitatory_spike')
        self.app['ESM'] = StateMonitor(self.app['excitatory_group'], ['v'], record=True, name='ESM')
        self.app['ERM'] = PopulationRateMonitor(self.app['excitatory_group'], name='excitatory_group_rate')
        self.app['S2M'] = StateMonitor(self.app['S2'], ['w', 'ge'], record=range(5), name='S2M')

        self.net = Network(self.app.values())


    def __getitem__(self, key):
        return self.net[key]

    def set_mode(self, new_mode, input_signal):
        if new_mode == "spontaneous":
            rates = np.zeros(n_input) * Hz
            self.app['poisson_group'].rates = rates
        elif new_mode == "patternA":
            rates = np.zeros(n_input) * Hz
            rates[:50] = self.stimulation_amp * Hz
            self.app['poisson_group'].rates = rates
        elif new_mode == "patternB":
            rates = np.zeros(n_input) * Hz
            rates[50:100] = self.stimulation_amp * Hz
            self.app['poisson_group'].rates = rates
        elif new_mode == "patternC":
            rates = np.zeros(n_input) * Hz
            rates[100:150] = self.stimulation_amp * Hz
            self.app['poisson_group'].rates = rates
        elif new_mode == "patternD":
            rates = np.zeros(n_input) * Hz
            rates[150:200] = self.stimulation_amp * Hz
            self.app['poisson_group'].rates = rates
        elif new_mode == "random_channel":
            self.app['poisson_group'].rates = input_signal * Hz

    def train(self, duration):
        self.net.run(duration * second)


if __name__ == '__main__':
    defaultclock.dt = 1 * ms
    n_input = 200
    num_neuron = 200
    num_catagoty = 8
    rate_list = []
    for catagory in range(num_catagoty):
        rates = np.zeros(n_input)
        ones_idx = np.random.choice(n_input, size=10, replace=False)
        rates[ones_idx] = 15 * Hz
        rate_list.append(rates)

    seed(42)

    v_rest_e = -65. * mV
    v_reset_e = -75. * mV
    v_thresh_e = -50. * mV

    K = 20
    p_conn = K / num_neuron
    gmax = 0.4

    model = Model(hebbian_lr=1e-2, refractory=5, sigma_i=20, stimulation_amp=20)
    time_spon = 10
    time_sti = 1

    # model.train(duration=time_spon)
    # model.set_mode("random_channel", rates1)
    # model.train(duration=10)
    # model.set_mode("random_channel", rates2)
    # model.train(duration=10)
    # model.set_mode("random_channel", rates3)
    # model.train(duration=10)
    # model.set_mode("random_channel", rates4)
    # model.train(duration=10)
    # model.set_mode("patternA")
    # model.train(duration=10)
    # model.set_mode("patternB")
    # model.train(duration=10)
    # model.set_mode("patternC")
    # model.train(duration=10)
    # model.set_mode("patternD")
    # model.train(duration=10)
    # model.set_mode("patternA")
    # model.train(duration=10)
    # model.set_mode("patternB")
    # model.train(duration=10)
    # model.set_mode("patternC")
    # model.train(duration=10)
    # model.set_mode("patternD")
    # model.train(duration=10)
    # a = model['S2'].w[:200]
    # model.set_mode("patternA")
    # model.train(duration=300)
    # b = model['S2'].w[:200]
    # print(b - a)


    iteration = 100
    for iter in range(iteration):
        for rate in rate_list:
            model.set_mode('random_channel', rate)
            model.train(duration=1)

    spike_mon = model['excitatory_spike']
    i_arr = spike_mon.i
    t_arr = spike_mon.t / second
    np.save('spike_i2.npy', i_arr)
    np.save('spike_t2.npy', t_arr)

    # model.set_mode('noise')
    # model.train(duration=0)
    #
    # iteration = 3
    # for iter in range(iteration):
    #     for mode in ["patternA", "patternB", "patternC", "patternD"]:
    #         model.set_mode(mode)
    #         for k in range(sample_number):
    #             model.train(duration=time_sti)
    #             rate = weight_rate_spikes(model, T=time_sti*second)
    #             rate_list.append(rate)

    # print("simulation finished")
    # rate_list = np.array(rate_list)
    # np.save('rate_list7.npy', rate_list)

    plot_w(model["S2M"])
    plot_rates(model['excitatory_group_rate'])
    plot_spikes(model['excitatory_spike'])
