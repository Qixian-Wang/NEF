import os
import torch
import numpy as np
from pydantic import BaseModel, ConfigDict
from torch.utils.tensorboard import SummaryWriter

# Types of LFB kernels
LFB_GAUSS = 'gauss'
LFB_DoG = 'DoG'
LFB_EXP = 'exp'
LFB_DoE = 'DoE'


class Configs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir: str = os.path.join("runs", "test1")
    writer: SummaryWriter = SummaryWriter(log_dir=log_dir)

    ff_activate: bool = True

    # Parameters
    alphas: np.ndarray = np.linspace(1e-4, 10, 20)
    batch_size: int = 64
    num_epoch: int = 20
    data_length: int = 0

    # Param for PCA
    n_components: int = 32

    # Param for VAE
    VAE_learning_rate: float = 1e-3
    num_hidden: int = 16

    # Subset configuration
    use_subset: bool = True
    train_subset_size: int = 20000
    test_subset_size: int = 10000

    # Param for Reservoir
    num_neuron: int = 50
    input_dim_ridge: int = 784
    spectral_radius: float = 0
    reservoir_sigma: float = 0.1
    reservoir_sparsity: float = 0.9

    # Param for Hebbian
    in_channels: int = 1
    kernel_size: int = 28
    alpha: float = 3.0
    HEBB_UPD_GRP: int = 8


configs = Configs()
