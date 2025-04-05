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
import config_file
import time


def main(
    epoch: int, dataset: DatasetMNIST, method: MethodBase, seed: int | None = None
) -> None:

    for epoch_idx in range(epoch):
        start_time = time.time()
        train_accuracy = method.train(epoch_idx, dataset, seed)
        print(f"training duration: {time.time() - start_time:.2f}s")
        print(f"Train Accuracy: {train_accuracy * 100:.2f}%")

        start_time = time.time()
        test_accuracy = method.test(epoch_idx, dataset, seed)
        print(f"test duration: {time.time() - start_time:.2f}s")
        print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")


if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    dataset = data_generator(config_file.configs)
    method = NaiveRC(config_file.configs)

    main(dataset=dataset, method=method, seed=seed, epoch=config_file.configs.num_epoch)
    config_file.configs.writer.close()
