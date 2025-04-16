import torch

from NEF.vae_rc.method.method_base import MethodBase
# from NEF.vae_rc.method.method_vae_rc import MethodVAERC
from NEF.vae_rc.method.method_hebbian_rc import MethodHebbian_RC
from NEF.vae_rc.implement.pca_linear_network import HebbianComparisonPCA

from dataloader import DatasetMNIST, data_generator
import config_file
import time


def main(
    epoch: int, dataset: DatasetMNIST, method: MethodBase, seed: int | None = None
) -> None:

    for epoch_idx in range(epoch):
        start_time = time.time()
        method.train(epoch_idx, dataset, seed)
        print(f"training duration: {time.time() - start_time:.2f}s")

        start_time = time.time()
        method.validate(epoch_idx, dataset, seed)
        print(f"test duration: {time.time() - start_time:.2f}s")

    # method.test(dataset, seed)

if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    dataset = data_generator(config_file.configs)
    method = HebbianComparisonPCA(config_file.configs)

    main(dataset=dataset, method=method, seed=seed, epoch=config_file.configs.num_epoch)

    config_file.configs.writer.close()
