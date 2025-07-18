from utils import *

class Model:
    def __init__(self, 
                 hebbian_lr, 
                 refractory, 
                 sigma_noise=0.0,
                 connectivity=0,
                 stimulation_voltage=4,
                 stimulation_frequency=4,
                 transmitter_leakage=0.2,
                 transmitter_restoration=600,
                 distance_factor=600,
                 dApre=0.1,
                 taupre=0,
                 taupost=0,
                 tau_noise=0,
                 layout=None,
                 gmax=0.3
                 ):

        seed(42)
        self.app = {}
        self.stimulation_frequency = stimulation_frequency
        electrode_positions, neuron_positions = layout
        electrode_positions = electrode_positions * umetre
        neuron_positions = neuron_positions * umetre
        num_neuron = len(neuron_positions)
        self.num_electrodes = len(electrode_positions)
        electrode_radius = 30 * umetre

        v_rest_e = -65
        v_reset_e = -75
        v_thresh_e = -50
        taupre = taupre * ms
        taupost = taupost * ms
        dApre = 0.1
        dApost = -dApre * 0.8
        dApost *= gmax
        dApre *= gmax
        U = transmitter_leakage  # now set to 0 -> no transmitter leakage
        tau_rec = transmitter_restoration * ms

        # Homustatic plasticity: ref: https://brian2.readthedocs.io/en/stable/examples/synapses.spike_based_homeostasis.html
        dw_plus = 0.001
        dw_minus = 0.001

        stdp = '''
            w : 1
            lr : 1 (shared)
            dApre    : 1 (shared)
            dApost   : 1 (shared)            
            taupre   : second   (shared)
            taupost  : second   (shared)
            tau_rec : second (shared)
            dw_plus : 1 (shared)
            dw_minus : 1 (shared)
            gmax     : 1        (shared)
            dApre/dt = -Apre / taupre : 1 (event-driven)
            dApost/dt = -Apost / taupost : 1 (event-driven)
            dres/dt = (1 - res)/tau_rec     : 1 (event-driven)
            '''
        pre = f'''
            ge += w * res
            Apre += dApre
            w = clip(w + lr*Apost + dw_plus, 0, gmax)
            res *= (1 - {U})
            '''
        post = f'''
            Apost += dApost
            w = clip(w + lr*Apre - dw_minus, 0, gmax)
            '''

        # excitatory group
        neuron_e = '''
        dv/dt = (ge*(0*mV - v) + (v_rest_e - v)) / tau_m + noise / (20*ms): volt
        dnoise/dt = - noise/tau_noise + sqrt(2*sigma_noise**2/tau_noise)*xi: volt
        dge/dt = -ge / tau_syn : 1
        sigma_noise : volt
        tau_m : second
        tau_syn : second
        tau_noise : second
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
        self.app['excitatory_group'].sigma_noise = sigma_noise * mV
        self.app['excitatory_group'].tau_m = 5 * ms
        self.app['excitatory_group'].tau_syn = 5 * ms
        self.app['excitatory_group'].tau_noise = tau_noise * ms
        self.app['excitatory_group'].theta = 0.01
        self.app['excitatory_group'].v_rest_e = v_rest_e * mV
        self.app['excitatory_group'].v_reset_e = v_reset_e * mV
        self.app['excitatory_group'].v_thresh_e = v_thresh_e * mV

        x_array, y_array = zip(*neuron_positions)
        self.app['excitatory_group'].x_pos = x_array
        self.app['excitatory_group'].y_pos = y_array


        #### input electrodes
        self.app['electrodes_input'] = NeuronGroup(
            self.num_electrodes, elec_eqs,
            threshold='rand()<rates*dt', reset='', method='euler', name='electrodes_input'
        )
        self.app['electrodes_input'].rates = np.zeros(self.num_electrodes) * Hz

        x_array_elec, y_array_elec = zip(*electrode_positions)
        self.app['electrodes_input'].x_elec = x_array_elec
        self.app['electrodes_input'].y_elec = y_array_elec

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
                             delay=15*ms,
                             name='S2')
        self.app['S2'].connect(
            condition='i!=j',
            p = f'{connectivity} * exp(-sqrt((x_pos_pre-x_pos_post)**2 + (y_pos_pre-y_pos_post)**2) / ({distance_factor} * umetre))',
            # p = f'{connectivity} * (1 - sqrt((x_pos_pre-x_pos_post)**2 + (y_pos_pre-y_pos_post)**2) / ({distance_factor} * umetre))'
        )
        self.app['S2'].gmax = gmax
        mean = 0
        sigma = 1
        self.app['S2'].w = clip(np.random.normal(loc=mean, scale=sigma, size=len(self.app['S2'].w)), 0, gmax)
        self.app['S2'].lr = hebbian_lr
        self.app['S2'].taupre = taupre
        self.app['S2'].taupost = taupost
        self.app['S2'].Apre = 0
        self.app['S2'].Apost = 0
        self.app['S2'].dApre = dApre
        self.app['S2'].dApost = dApost
        self.app['S2'].tau_rec = tau_rec
        self.app['S2'].dw_plus = dw_plus
        self.app['S2'].dw_minus = dw_minus

        # Monitors
        self.app['excitatory_spike'] = SpikeMonitor(self.app['excitatory_group'], name='excitatory_spike')
        self.neuron_group = []
        for i in range(self.num_electrodes):
            x0 = self.app['electrodes_input'].x_elec[i]
            y0 = self.app['electrodes_input'].y_elec[i]
            mask = ((self.app['excitatory_group'].x_pos - x0)**2 +
                    (self.app['excitatory_group'].y_pos - y0)**2) <= (electrode_radius)**2
            self.neuron_group.append(np.where(mask)[0])

        self.app['S2_monitor'] = StateMonitor(self.app['S2'], ['w', 'ge', 'Apre'], record=range(15), name='S2_monitor')

        self.net = Network(self.app.values())


    def __getitem__(self, key):
        return self.net[key]

    def set_mode(self, new_mode, input_signal, freqeuncy):
        if new_mode == "spontaneous":
            rates = np.zeros(self.num_electrodes) * Hz
            self.app['electrodes_input'].rates = rates
        elif new_mode == "patternA":
            rates = np.zeros(self.num_electrodes) * Hz
            rates[:20] = self.stimulation_frequency * Hz
            self.app['electrodes_input'].rates = rates
        elif new_mode == "patternB":
            rates = np.zeros(self.num_electrodes) * Hz
            rates[10:20] = self.stimulation_frequency * Hz
            self.app['electrodes_input'].rates = rates * 2
        elif new_mode == "patternC":
            rates = np.zeros(self.num_electrodes) * Hz
            rates[100:150] = self.stimulation_frequency * Hz
            self.app['electrodes_input'].rates = rates
        elif new_mode == "patternD":
            rates = np.zeros(self.num_electrodes) * Hz
            rates[150:200] = self.stimulation_frequency * Hz
            self.app['electrodes_input'].rates = rates
        elif new_mode == "selected":
            rates = np.zeros(self.num_electrodes)
            rates[input_signal] = freqeuncy
            self.app['electrodes_input'].rates = rates * Hz

    def train(self, duration):
        self.net.run(duration * second)
