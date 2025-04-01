from datetime import datetime
import torch
import matplotlib.pyplot as plt

from model_method2 import VAE_RC, train_combined, test_combined
from model_method1 import VAE_regression, train_regression, test_regression
from model_basic import train_basic_pca, test_basic_pca
from model_basic2 import train_rc, test_rc
from dataloader import data_generator
from config_file import Config as Configs
import time

seed = 42
torch.manual_seed(seed)

configs = Configs()

train_dataset, test_dataset = data_generator(configs, training_mode=True, subset=False)

iteration = 5
dt = 0.02
loss1 = []
loss2 = []

# Basic
# start_time = time.time()
# model, matrix = train_basic_pca(train_dataset, configs)
# test_basic_pca(model, matrix, test_dataset, configs)
# end_time = time.time()
# time = end_time - start_time
# print(f"time usage: {time}s")

# Basic 2
# start_time = time.time()
# model = train_rc(train_dataset, configs)
# test_rc(model, test_dataset)
# end_time = time.time()
# time = end_time - start_time
# print(f"time usage: {time}s")

# method 1
# start_time = time.time()
# model = VAE_regression(configs).to(configs.device)
# train_regression(model, train_dataset, configs)
# test_regression(model, test_dataset, configs)
# end_time = time.time()
# time = end_time - start_time
# print(f"time usage: {time}s")

# method 2
start_time = time.time()
model = VAE_RC(configs).to(configs.device)
train_combined(model, train_dataset, configs)
test_combined(model, test_dataset, configs)
end_time = time.time()
time = end_time - start_time
print(f"time usage: {time}s")



# for i in range(iteration):
#     RC = ReservoirComputing(num_neuron=300, spectral_radius=0.8, sigma=0.1, sparsity=0.98, beta=1e-6,
#                             mask_prob=0.3, mask_interval=25, dt=dt, lr_recurrent=1e-3, device='cpu')
#     RC2 = ReservoirComputing(num_neuron=300, spectral_radius=0.8, sigma=0.1, sparsity=0.98, beta=1e-6,
#                             mask_prob=0.3, mask_interval=25, dt=dt, lr_recurrent=0, device='cpu')
#
#     model_pretrain(RC, train_dataset, configs)
#     loss1.append(model_test(RC, test_dataset))
#
#     loss2.append(model_test(RC2, test_dataset))
#
# plt.figure(figsize=(8, 4))
# plt.plot(loss1, label="loss with pretrain")
# plt.plot(loss2, label="loss without pretrain")
# plt.title("Loss")
# plt.legend()
# plt.xlabel("Test Number")
# plt.ylabel("Loss")
# plt.grid()
# plt.savefig("loss.png")
# print(f"ave_loss with pretrain = {sum(loss1)/len(loss1)}")
# print(f"ave_loss without pretrain = {sum(loss2)/len(loss2)}")
