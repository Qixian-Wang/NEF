from NEF.spontaneous_behavior.utils import *

class Model:
    def __init__(self, hebbian_lr, refractory, sigma_i=0.0, stimulation_amp=4, stimulation_frequency=4, groups_per_row=10, n_per_group=10, gmax=0.3):
        seed(42)
        self.app = {}
        self.stimulation_frequency = stimulation_frequency
        self.n_groups = groups_per_row ** 2
        self.num_electrodes = self.n_groups
        num_neuron = self.n_groups * n_per_group
        electrode_spacing = 100 * umetre
        electrode_radius = 30 * umetre

        v_rest_e = -65
        v_reset_e = -75
        v_thresh_e = -50

        # excitatory group
        neuron_e = f'''
        dv/dt = (ge*(0*mV - v) + ({v_rest_e}*mV - v)) / tau_m + (- theta * v + sigma_i * xi_i * sqrt(ms)) / tau_m: volt
        dge/dt = -ge / tau_syn : 1
        dr/dt  = -r/tau_r : Hz
        sigma_i : volt
        tau_m : second
        tau_syn : second
        tau_r   : second
        theta : 1
        x_pos : meter
        y_pos : meter
        '''

        oja = '''
        w  : 1
        lr : 1 (shared)
        '''

        elec_eqs = '''
        rates : Hz
        x_elec : meter (constant)
        y_elec : meter (constant)
        '''

        #### excitatory group
        self.app['excitatory_group'] = NeuronGroup(num_neuron, neuron_e, threshold=f'v>{v_thresh_e}*mV',
                                                   refractory=refractory * ms, reset=f'v={v_reset_e}*mV; r+=1/tau_r',
                                                   method='euler', name='excitatory_group')
        self.app['excitatory_group'].v = v_rest_e * mV
        self.app['excitatory_group'].sigma_i = sigma_i * mV
        self.app['excitatory_group'].tau_m = 25 * ms
        self.app['excitatory_group'].tau_syn = 5 * ms
        self.app['excitatory_group'].tau_r = 500 * ms
        self.app['excitatory_group'].theta = 1e-1

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
        sigma_con = 100 * umetre
        self.app['S1'] = Synapses(
            self.app['electrodes_input'], self.app['excitatory_group'],
            on_pre=f'ge_post += {stimulation_amp}', method='euler', name='elec_syn'
        )
        self.app['S1'].connect(
            condition=f'(x_pos_post - x_elec_pre)**2 + (y_pos_post - y_elec_pre)**2 <= electrode_radius**2'
        )

        self.app['S2'] = Synapses(self.app['excitatory_group'],
                             self.app['excitatory_group'],
                             model=oja,
                             on_pre='ge_post += w',
                             method='euler',
                             name='S2')
        self.app['S2'].connect(
            condition='i!=j',
            p='exp(-((x_pos_pre-x_pos_post)**2 + (y_pos_pre-y_pos_post)**2)/(2*sigma_con**2))'
        )
        self.app['S2'].w = f'rand()*{gmax}'
        self.app['S2'].lr = hebbian_lr
        self.app['S2'].run_regularly(f'''
        x = r_pre/Hz
        y = r_post/Hz
        w += lr*(y*x - y**2*w)*(dt/second)
        w = clip(w, 0, {gmax})
        ''', dt=defaultclock.dt)


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
        self.app['S2M'] = StateMonitor(self.app['S2'], ['w', 'ge'], record=range(5), name='S2M')

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