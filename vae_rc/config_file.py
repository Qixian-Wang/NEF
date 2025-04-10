import os
import torch
import numpy as np
from pydantic import BaseModel, ConfigDict
from torch.utils.tensorboard import SummaryWriter

RULE_BASE = 'base'  # r = lfb
RULE_HEBB = 'hebb'  # r = y * lfb

# Type of reconstruction scheme
REC_QNT = 'qnt'  # reconst = w
REC_QNT_SGN = 'qnt_sgn'  # reconst = sign(lfb) * w
REC_LIN_CMB = 'lin_cmb'  # reconst = sum_i y_i w_i

# Types of LFB kernels
LFB_GAUSS = 'gauss'
LFB_DoG = 'DoG'
LFB_EXP = 'exp'
LFB_DoE = 'DoE'

# Types of weight initialization schemes
INIT_BASE = 'base'
INIT_NORM = 'norm'

# Type of update reduction scheme
RED_AVG = 'avg'  # average
RED_W_AVG = 'w_avg'  # weighted average


class Configs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir: str = os.path.join("runs", "test1")
    writer: SummaryWriter = SummaryWriter(log_dir=log_dir)

    # Parameters
    alphas: np.ndarray = np.linspace(1e-4, 10, 20)
    batch_size: int = 64
    num_epoch: int = 30
    data_length: int = 0

    # Param for PCA
    n_components: int = 32

    # Param for VAE
    VAE_learning_rate: float = 1e-3
    num_hidden: int = 32

    # Subset configuration
    use_subset: bool = False
    train_subset_size: int = 640
    test_subset_size: int = 100

    # Param for Reservoir
    num_neuron: int = 50
    input_dim_ridge: int = 784
    spectral_radius: float = 0
    reservoir_sigma: float = 0.1
    reservoir_sparsity: float = 0.9

    # Param for Hebbian
    in_channels: int = 1
    kernel_size: int = 28
    lfb_value: float = 0
    competitive: bool = False
    random_abstention: bool = False
    alpha: float = 1.0
    weight_upd_rule: str = RULE_HEBB
    weight_init: str = INIT_BASE
    reconstruction: str = REC_LIN_CMB
    reduction: str = RED_AVG
    HEBB_UPD_GRP: int = 2

    tau: int = 1000


configs = Configs()
