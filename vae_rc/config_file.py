import torch
import numpy as np
from pydantic import BaseModel, ConfigDict


class Configs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Param for basic
    n_components: int = 16
    alphas: np.ndarray = np.linspace(1e-4, 10, 20)

    # Param for method1 method2
    sequence_length: int = 784
    batch_size: int = 64
    target_batch_size: int = 32
    dt: float = 0.02
    num_epoch: int = 10

    VAE_learning_rate: float = 1e-3
    rc_loss_weight: int = 5
    num_hidden: int = 32

    # Subset configuration
    use_subset: bool = True
    train_subset_size: int = 5000
    test_subset_size: int = 1000

    # Param for RNN
    num_neuron: int = 100
    input_dim_ridge: int = 784
    input_dim_rnn: int = 32
    spectral_radius: float = 0.8
    sigma: float = 0.1
    sparsity: float = 0.98
    n_classes: int = 10
    rls_init: float = 1e2

    data_dim: int = 0
    train_data_size: int = 0
    test_data_size: int = 0

    experiment_name: str = "test1"
