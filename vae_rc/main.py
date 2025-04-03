from datetime import datetime
import torch
import matplotlib.pyplot as plt

from model_method2 import VAE_RC, train_combined, test_combined
from model_method1 import VAE_regression, train_regression, test_regression
from model_basic1 import train_basic_pca, test_basic_pca
from model_basic2 import train_rc, test_rc
from dataloader import data_generator
from config_file import Config as Configs
import time

seed = 42
torch.manual_seed(seed)

configs = Configs()

train_dataset, test_dataset = data_generator(configs, subset=True)

def run(method):
    start_time = time.time()

    if method == "Basic1":
        model, matrix = train_basic_pca(train_dataset, configs)
        test_basic_pca(model, matrix, test_dataset)

    if method == "Basic2":
        model = train_rc(train_dataset, configs)
        test_rc(model, test_dataset)

    if method == "Method1":
        model = VAE_regression(configs).to(configs.device)
        train_regression(model, train_dataset, configs)
        test_regression(model, test_dataset, configs)

    if method == "Method2":
        model = VAE_RC(configs).to(configs.device)
        train_combined(model, train_dataset, configs)
        test_combined(model, test_dataset, configs)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"time usage: {total_time}s")

run(method="Method2")

