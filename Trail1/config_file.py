import torch
class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Param for basic
        self.n_components = 16

        # Param for method1 method2
        self.sequence_length = 784
        self.batch_size = 64
        self.target_batch_size = 32
        self.dt = 0.02
        self.num_epoch = 10
        self.VAE_learning_rate = 1e-3
        self.rc_loss_weight = 4
        self.subset_size = 3000
        self.num_hidden = 16

        # Param for RNN
        self.num_neuron = 100,
        self.input_dim_ridge = 784,
        self.input_dim_rnn = 16,
        self.spectral_radius = 0.8,
        self.sigma = 0.1,
        self.sparsity = 0.98,
        self.n_classes = 10,
        self.rls_init = 1e2,
        self.beta = 0.8
