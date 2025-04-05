import os
import torch
import numpy as np
from pydantic import BaseModel, ConfigDict
from torch.utils.tensorboard import SummaryWriter


class Configs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir: str = os.path.join("runs", "test1")
    writer: SummaryWriter = SummaryWriter(log_dir=log_dir)

    # Parameters
    alphas: np.ndarray = np.linspace(1e-4, 10, 20)
    batch_size: int = 200
    num_epoch: int = 100
    data_length: int = 0

    # Param for PCA
    n_components: int = 16

    # Param for VAE
    VAE_learning_rate: float = 1e-3
    num_hidden: int = 32

    # Subset configuration
    use_subset: bool = False
    train_subset_size: int = 5000
    test_subset_size: int = 1000

    # Param for Reservoir
    num_neuron: int = 100
    input_dim_ridge: int = 784
    spectral_radius: float = 0.8
    reservoir_sigma: float = 0.1
    reservoir_sparsity: float = 0.98


configs = Configs()
