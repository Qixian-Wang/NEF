from NEF.spontaneous_behavior.utils import *
from model_stdp import Model

defaultclock.dt = 1 * ms
groups_per_row = 8
n_groups = groups_per_row ** 2
neuron_per_group = 10
num_catagoty = 4
columns = [list(range(c, n_groups, groups_per_row)) for c in range(groups_per_row)]
block1 = [start + offset for start in range(0, 32, 8) for offset in range(2)]
block2 = [start + offset for start in range(4, 32, 8) for offset in range(2)]
block3 = [start + offset for start in range(32, 64, 8) for offset in range(2)]
block4 = [start + offset for start in range(36, 64, 8) for offset in range(2)]
# channel_list = [block1+block4, block2+block3, block1+block3, block2+block4]
channel_list = [block1+block2+block3, block1+block2+block4, block1+block3+block4, block2+block3+block4]

gmax = 0.6
noise_amp = 30
connectivity = 0.1
taupre = 500
taupost = 500
transmitter_leakage = 0.5
transmitter_restoration = 600

model = Model(hebbian_lr=1e-3, refractory=50, sigma_i=noise_amp,
              connectivity=connectivity,
              groups_per_row=groups_per_row,
              n_per_group=neuron_per_group,
              taupre=taupre,
              taupost=taupost,
              transmitter_leakage=transmitter_leakage,
              transmitter_restoration=transmitter_restoration,
              stimulation_voltage=80,
              stimulation_frequency=6,
              gmax=gmax)
time_spon = 10
model.train(duration=time_spon)


iteration = 20
for iter in range(iteration):
    for channel in channel_list:
        model.set_mode('selected', channel, freqeuncy=6)
        model.train(duration=0.5)
        model.set_mode('spontaneous', channel, freqeuncy=4)
        model.train(duration=0.5)
    print(f"finish iter {iter}")

for iter in range(100):
    for channel in [block1, block2, block3, block4]:
        model.set_mode('selected', channel, freqeuncy=15)
        model.train(duration=0.5)
        model.set_mode('spontaneous', channel, freqeuncy=6)
        model.train(duration=0.5)

for iter in range(100):
    for channel in [block1+block2, block2+block4, block3+block1, block4+block3]:
        model.set_mode('selected', channel, freqeuncy=15)
        model.train(duration=0.5)
        model.set_mode('spontaneous', channel, freqeuncy=6)
        model.train(duration=0.5)

iteration = 20
for iter in range(iteration):
    for channel in channel_list:
        model.set_mode('selected', channel, freqeuncy=6)
        model.train(duration=0.5)
        model.set_mode('spontaneous', channel, freqeuncy=4)
        model.train(duration=0.5)
    print(f"finish iter {iter}")

spike_time, group_index = process_spikes(model)
np.save('spike_i3.npy', group_index)
np.save('spike_t3.npy', spike_time)

plot_figures(model, spike_time, group_index, n_groups=n_groups, n_per_group=neuron_per_group)
