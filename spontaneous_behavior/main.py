from utils import *
from model_stdp import Model
import os
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

from mea_layout import generate_layout


defaultclock.dt = 2 * ms
n_groups = 64

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

figure_save_path = "plots"
if rank == 0:
    os.makedirs(figure_save_path, exist_ok=True)
    print("start", flush=True)
data_save_path = "data"
if rank == 0:
    os.makedirs(data_save_path, exist_ok=True)

comm.Barrier()

# random select channel
channel_list = []
pre_training_channel_list = []
num_class = rank + 4
for i in range(num_class):
    rng = np.random.default_rng(i)
    pattern = rng.choice(n_groups, size=(int(n_groups/num_class)))
    channel_list.append(pattern)
    pre_training_channel_list.append(pattern)

# parameters
gmax = 5
noise_amp = 20
connectivity = 0.9
tau_noise = 5
distance_factor = 200

taupre = 100
taupost = 300
dApre = 0.1
transmitter_leakage = 0
transmitter_restoration = 600

# neuron and electrodelayout
layout = generate_layout(height=1600,
                           electrode_t=64,
                           electrode_b=64,
                           electrode_per_row_b=8,
                           electrode_per_row_t=8,
                           num_neuron=1000,
                           spacing=100,
                           plot_result=False)

# model
model = Model(hebbian_lr=1e-2, 
              refractory=30,  
              sigma_noise=noise_amp,
              connectivity=connectivity,
              taupre=taupre,
              taupost=taupost,
              dApre=dApre,
              tau_noise=tau_noise,
              transmitter_leakage=transmitter_leakage,
              transmitter_restoration=transmitter_restoration,
              distance_factor=distance_factor,
              stimulation_voltage=80,
              stimulation_frequency=6,
              layout=layout,
              gmax=gmax)

def execute_training_sequence(model, sequence, channel_list, pre_training_channel_list):
    training_start_time_list = []
    current_time = 0
    i = 0

    while i < len(sequence):
        mode = sequence[i][0]
        value = int(sequence[i][1])
        i += 1
        
        if mode == "spontaneous":
            model.set_mode('spontaneous', None, freqeuncy=8)
            model.train(duration=value)
            current_time += value
            
        elif mode == "pretrain":
            for iter in range(value):
                for channel in pre_training_channel_list:
                    model.set_mode('selected', channel, freqeuncy=10)
                    model.train(duration=1)
                    model.set_mode('spontaneous', channel, freqeuncy=4)
                    model.train(duration=1)
            current_time += 2 * value * len(pre_training_channel_list)
                    
        elif mode == "train":
            for iter in range(int(value)):
                for channel in channel_list:
                    model.set_mode('selected', channel, freqeuncy=10) # original 20
                    model.train(duration=1)
                    model.set_mode('spontaneous', channel, freqeuncy=4)
                    model.train(duration=1)
            training_start_time_list.append(current_time)
            current_time += 2 * value * len(channel_list)

    return training_start_time_list

# training_sequence = [("spontaneous", "1000"), ("train", "20"), ("pretrain", "200"), ("train", "20"), ("pretrain", "200"), 
#                      ("train", "20"), ("pretrain", "200"), ("train", "20"), ("spontaneous", "1000"), ("train", "20"), ("spontaneous", "1000"), ("train", "20"),
#                      ("spontaneous", "1000"), ("train", "20"), ("spontaneous", "1000"), ("train", "20"), ("spontaneous", "1000"), ("train", "20")]

training_sequence = [("spontaneous", "1000"), ("train", "20"), ("pretrain", "200"), ("train", "20"), ("pretrain", "200"), 
                     ("train", "20"), ("pretrain", "200"), ("train", "20"), ("pretrain", "200"), ("train", "20")]

# training_sequence = [("spontaneous", "300"), ("train", "5")]

training_start_time_list = execute_training_sequence(model, training_sequence, channel_list, pre_training_channel_list)
print(f"rank:{rank} training_start_time_list: {training_start_time_list}", flush=True)

spike_time, group_index = process_spikes(model)
np.save(os.path.join(data_save_path, f'spike_i_rank={rank}.npy'), group_index)
np.save(os.path.join(data_save_path, f'spike_t_rank={rank}.npy'), spike_time)

plot_figures(rank, model, spike_time, group_index, n_groups=64, figure_save_path=figure_save_path)
