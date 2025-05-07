from NEF.spontaneous_behavior.utils import *
from numpy.random import lognormal
from sklearn.decomposition import PCA

class Model:
    def __init__(self, hebbian_lr, synaptic_delay, refractory, sigma_ge=0, sigma_i=0.0, stimulation_amp=4, mode="spontaneous"):
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
                rates[50:100] = stimulation_amp * Hz
                app['poisson_group'].rates = rates
            elif self.mode == "patternC":
                rates = np.zeros(n_input) * Hz
                rates[100:150] = stimulation_amp * Hz
                app['poisson_group'].rates = rates
            elif self.mode == "patternD":
                rates = np.zeros(n_input) * Hz
                rates[150:200] = stimulation_amp * Hz
                app['poisson_group'].rates = rates
            elif self.mode == "noise":
                rates = stimulation_amp * np.random.rand(n_input) * Hz
                app['poisson_group'].rates = rates
            elif self.mode == "stimulate":
                rates = stimulation_amp * Hz
                app['poisson_group'].rates = rates
            elif self.mode == "colored_gaussian":
                data = ColoredGaussianDataset(D=num_neuron)
                data = data + data.min() + 10
                app['poisson_group'].rates = data * Hz

        # excitatory group
        neuron_e = '''
        dv/dt = (ge*(0*mV - v) + (v_rest_e - v)) / tau_m + I_ou/C : volt
        dge/dt = (-ge + sigma_ge * xi_e * sqrt(ms)) / tau_syn : 1
        dI_ou/dt = (-I_ou + amplitude * sin(2*pi*freq*t))/tau_ou + sigma_i * xi_i * sqrt(1/ms) : amp
        dr/dt  = -r/tau_r : Hz
        sigma_ge : 1
        sigma_i : amp
        amplitude : amp
        freq : Hz
        tau_ou : second
        tau_m : second
        tau_syn : second
        tau_r   : second
        C : farad
        '''

        app['excitatory_group'] = NeuronGroup(num_neuron, neuron_e, threshold='v>v_thresh_e', refractory=refractory * ms, reset='v=v_reset_e; r+=1/tau_r',
                                              method='euler', name='excitatory_group')
        app['excitatory_group'].v = v_rest_e
        app['excitatory_group'].sigma_ge = sigma_ge
        app['excitatory_group'].sigma_i = sigma_i * pA
        app['excitatory_group'].tau_m = 25 * ms
        app['excitatory_group'].tau_syn = 5 * ms
        app['excitatory_group'].amplitude = 5 * pA  # amplitude of periodic modulation
        app['excitatory_group'].freq = 0.5 * Hz  # frequency of modulation
        app['excitatory_group'].tau_ou = 30 * ms
        app['excitatory_group'].C = 200 * pF
        app['excitatory_group'].tau_r = 50 * ms

        oja = Equations('''
        w  : 1
        lr : 1 (shared)
        ''')

        # poisson generators one-to-all excitatory neurons with plastic connections
        app['S1'] = Synapses(app['poisson_group'],
                             app['excitatory_group'],
                             model=oja,
                             on_pre='ge_post += w',
                             method='euler',
                             name='S1')

        app['S1'].connect(j = 'i')
        app['S1'].w = 'rand()*gmax'  # random weights initialisation
        app['S1'].lr = hebbian_lr
        app['S1'].run_regularly('''
        x = rates_pre/Hz
        y = r_post/Hz
        w += lr*(y*x - y**2*w)*(dt/second)
        w = clip(w, 0, gmax)
        ''', dt=defaultclock.dt)


        # excitatory neurons to excitatory neurons
        app['S2'] = Synapses(app['excitatory_group'],
                             app['excitatory_group'],
                             model=oja,
                             on_pre='ge_post += w',
                             method='euler',
                             name='S2')
        app['S2'].connect(condition='i != j', p=p_conn)
        mean = np.log(0.5*gmax)
        sigma = 1.0
        app['S2'].w = 'rand()*gmax'
        app['S2'].lr = hebbian_lr
        app['S2'].run_regularly('''
        x = r_pre/Hz
        y = r_post/Hz
        w += lr*(y*x - y**2*w)*(dt/second)
        w = clip(w, 0, gmax)
        ''', dt=defaultclock.dt)

        # Monitors
        app['poisson_monitor'] = StateMonitor(app['poisson_group'], ['rates'], record=True, name='poisson_monitor')
        app['excitatory_spike'] = SpikeMonitor(app['excitatory_group'], name='excitatory_spike')
        app['ESM'] = StateMonitor(app['excitatory_group'], ['v'], record=True, name='ESM')
        app['ERM'] = PopulationRateMonitor(app['excitatory_group'], name='excitatory_group_rate')
        app['S2M'] = StateMonitor(app['S2'], ['w'], record=range(5), name='S2M')

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

    K = 20
    p_conn = K / num_neuron
    gmax = 0.5

    model = Model(hebbian_lr=1e-2, synaptic_delay=5, refractory=5, sigma_ge=0, sigma_i=15, stimulation_amp=30, mode="spontaneous")
    time_spon = 10
    time_sti = 1

    model.train(duration=time_spon)
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


    iteration = 50
    for iter in range(iteration):
        for mode in ["patternA", "patternB", "patternC", "patternD"]:
            model.set_mode(mode)
            model.train(duration=1)

    spike_mon = model['excitatory_spike']
    i_arr = spike_mon.i
    t_arr = spike_mon.t / second
    np.save('spike_i3.npy', i_arr)
    np.save('spike_t3.npy', t_arr)

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

    pattern = np.zeros((num_neuron, 4))
    pattern[:50, 0] = 1
    pattern[50:100, 1] = 1
    pattern[100:150, 2] = 1
    pattern[150:200, 3] = 1
    mean_p = pattern.mean(axis=1, keepdims=True)
    pattern_centered = pattern - mean_p
    C_in = (pattern @ pattern.T) / 4
    eigvals, eigvecs = np.linalg.eigh(C_in)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    print("C_in top 4 eigenvalues:", eigvals[:10])


    W = np.zeros((num_neuron, num_neuron))
    W[model['S2'].i, model['S2'].j] = model['S2'].w
    Wc = W - W.mean(axis=0, keepdims=True)

    pca = PCA(n_components=5)
    pcs = pca.fit_transform(W)
    print("Explained variance ratio:", pca.explained_variance_ratio_)

    C = Wc.T @ Wc
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.sort(eigvals)[::-1]
    print(eigvals)