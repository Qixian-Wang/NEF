from NEF.spontaneous_behavior.utils import *

class Model:
    def __init__(self, hebbian_lr, refractory, sigma_i=0.0, connectivity=0, transmitter_leakage=0,
                 taupre=0, taupost=0,
                 transmitter_restoration=0, groups_per_row=10, n_per_group=10, gmax=0.3):
        seed(42)
        self.app = {}
        self.stimulation_frequency = 0
        stimulation_voltage = 0
        self.n_groups = groups_per_row ** 2
        self.num_electrodes = self.n_groups
        num_neuron = self.n_groups * n_per_group
        electrode_spacing = 100 * umetre
        electrode_radius = 30 * umetre

        v_rest_e = -65
        v_reset_e = -75
        v_thresh_e = -50
        taupre = taupre * ms
        taupost = taupost * ms
        dApre = 0.1
        dApost = -dApre * 0.7
        dApost *= gmax
        dApre *= gmax
        U = transmitter_leakage
        tau_rec = transmitter_restoration * ms

        stdp = '''
            w : 1
            lr : 1 (shared)
            dApre    : 1 (shared)
            dApost   : 1 (shared)            
            taupre   : second   (shared)
            taupost  : second   (shared)
            tau_rec : second (shared)
            gmax     : 1        (shared)
            dApre/dt = -Apre / taupre : 1 (event-driven)
            dApost/dt = -Apost / taupost : 1 (event-driven)
            dres/dt = (1 - res)/tau_rec     : 1 (event-driven)
            '''
        pre = f'''
            ge += w * res
            Apre += dApre
            w = clip(w + lr*Apost, 0, gmax)
            res *= (1 - {U})
            '''
        post = f'''
            Apost += dApost
            w = clip(w + lr*Apre, 0, gmax)
            '''

        # excitatory group
        neuron_e = '''
        dv/dt = (ge*(0*mV - v) + (v_rest_e - v)) / tau_m + (theta * (v_rest_e - v) + sigma_i * xi_i * sqrt(ms)) / tau_m: volt
        dge/dt = -ge / tau_syn : 1
        sigma_i : volt
        tau_m : second
        tau_syn : second
        theta : 1
        x_pos : meter
        y_pos : meter
        v_rest_e: volt
        v_reset_e: volt
        v_thresh_e: volt
        '''

        elec_eqs = '''
        rates : Hz
        x_elec : meter (constant)
        y_elec : meter (constant)
        '''

        #### excitatory group
        self.app['excitatory_group'] = NeuronGroup(num_neuron, neuron_e, threshold='v>v_thresh_e',
                                                   refractory=refractory * ms, reset='v=v_reset_e',
                                                   method='euler', name='excitatory_group')
        self.app['excitatory_group'].v = v_rest_e * mV
        self.app['excitatory_group'].sigma_i = sigma_i * mV
        self.app['excitatory_group'].tau_m = 20 * ms
        self.app['excitatory_group'].tau_syn = 5 * ms
        self.app['excitatory_group'].theta = 0.05
        self.app['excitatory_group'].v_rest_e = v_rest_e * mV
        self.app['excitatory_group'].v_reset_e = v_reset_e * mV
        self.app['excitatory_group'].v_thresh_e = v_thresh_e * mV

        width, height = (groups_per_row + 1) * 100 * umetre, (groups_per_row + 1) * 100 * umetre
        self.app['excitatory_group'].x_pos = 'rand()*width'
        self.app['excitatory_group'].y_pos = 'rand()*height'


        #### input electrodes
        self.app['electrodes_input'] = NeuronGroup(
            self.num_electrodes, elec_eqs,
            threshold='rand()<rates*dt', reset='', method='euler', name='electrodes_input'
        )
        self.app['electrodes_input'].rates = np.zeros(self.num_electrodes) * Hz
        self.app['electrodes_input'].x_elec = '((i%groups_per_row)+0.5) * electrode_spacing'
        self.app['electrodes_input'].y_elec = '((i//groups_per_row)+0.5) * electrode_spacing'

        # Synapses
        self.app['S1'] = Synapses(
            self.app['electrodes_input'], self.app['excitatory_group'],
            on_pre=f'v_post += {stimulation_voltage}*mV', method='euler', name='elec_syn'
        )
        self.app['S1'].connect(
            condition=f'(x_pos_post - x_elec_pre)**2 + (y_pos_post - y_elec_pre)**2 <= electrode_radius**2'
        )

        self.app['S2'] = Synapses(self.app['excitatory_group'],
                             self.app['excitatory_group'],
                             stdp,
                             on_pre=pre,
                             on_post=post,
                             method='euler',
                             name='S2')
        self.app['S2'].connect(
            condition='i!=j',
            p= f'{connectivity} * (1 - sqrt((x_pos_pre-x_pos_post)**2 + (y_pos_pre-y_pos_post)**2) / (600 * umetre))'
        )
        self.app['S2'].gmax = gmax
        mean = np.log(0.5 * gmax)
        sigma = 1.0
        self.app['S2'].w = clip(lognormal(mean, sigma, size=len(self.app['S2'].w)), 0, gmax)
        self.app['S2'].lr = hebbian_lr
        self.app['S2'].taupre = taupre
        self.app['S2'].taupost = taupost
        self.app['S2'].Apre = 0
        self.app['S2'].Apost = 0
        self.app['S2'].dApre = dApre
        self.app['S2'].dApost = dApost
        self.app['S2'].tau_rec = tau_rec

        # Monitors
        self.app['excitatory_spike'] = SpikeMonitor(self.app['excitatory_group'], name='excitatory_spike')
        self.neuron_group = []
        for i in range(self.num_electrodes):
            x0 = self.app['electrodes_input'].x_elec[i]
            y0 = self.app['electrodes_input'].y_elec[i]
            mask = ((self.app['excitatory_group'].x_pos - x0)**2 +
                    (self.app['excitatory_group'].y_pos - y0)**2) <= (electrode_radius)**2
            self.neuron_group.append(np.where(mask)[0])

        self.app['ESM'] = StateMonitor(self.app['excitatory_group'], ['v'], record=True, name='ESM')
        self.app['ERM'] = PopulationRateMonitor(self.app['excitatory_group'], name='excitatory_group_rate')
        self.app['S2M'] = StateMonitor(self.app['S2'], ['w', 'ge', 'Apre'], record=range(5), name='S2M')

        self.net = Network(self.app.values())


    def __getitem__(self, key):
        return self.net[key]

    def set_mode(self, new_mode, input_signal, freqeuncy):
        if new_mode == "spontaneous":
            rates = np.zeros(self.n_groups) * Hz
            self.app['electrodes_input'].rates = rates
        elif new_mode == "patternA":
            rates = np.zeros(self.n_groups) * Hz
            rates[:20] = self.stimulation_frequency * Hz
            self.app['electrodes_input'].rates = rates
        elif new_mode == "patternB":
            rates = np.zeros(self.n_groups) * Hz
            rates[10:20] = self.stimulation_frequency * Hz
            self.app['electrodes_input'].rates = rates * 2
        elif new_mode == "patternC":
            rates = np.zeros(self.n_groups) * Hz
            rates[100:150] = self.stimulation_frequency * Hz
            self.app['electrodes_input'].rates = rates
        elif new_mode == "patternD":
            rates = np.zeros(self.n_groups) * Hz
            rates[150:200] = self.stimulation_frequency * Hz
            self.app['electrodes_input'].rates = rates
        elif new_mode == "selected":
            rates = np.zeros(self.n_groups)
            rates[input_signal] = freqeuncy
            self.app['electrodes_input'].rates = rates * Hz

    def train(self, duration):
        self.net.run(duration * second)
