from datetime import datetime
import torch
import matplotlib.pyplot as plt
import numpy as np

from method_base import MethodBase
from method_vae_rc import MethodVAERC
from method_vae_ridge import MethodVAERegression
from method_pca import MethodPCA
from method_naiverc import NaiveRC

from dataloader import DatasetMNIST, data_generator
from config_file import Configs
import time


def main(epoch: int, dataset: DatasetMNIST, method: MethodBase, seed: int | None = None) -> None:

    for i in range(epoch):
        start_time = time.time()
        train_accuracy = method.train(dataset, seed)
        print(f"training duration: {time.time() - start_time:.2f}s")
        print(f"Train Accuracy: {train_accuracy * 100:.2f}%")

        start_time = time.time()
        test_accuracy = method.test(dataset, seed)
        print(f"test duration: {time.time() - start_time:.2f}s")
        print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")

    # elif method == "Method1":
    #     model = VAE_regression(configs).to(configs.device)
    #     train_regression(model, train_dataset, configs)
    #     test_regression(model, test_dataset, configs)
    #
    # elif method == "Method2":
    #     model = VAE_RC(configs).to(configs.device)
    #     train_combined(model, train_dataset, configs)
    #     test_combined(model, test_dataset, configs)


if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)

    configs = Configs()
    dataset = data_generator(configs)
    method = MethodVAERegression(configs)

    main(dataset=dataset, method=method, seed=seed, epoch=10)
