import torch
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import accuracy_score

from method_base import MethodBase


class NaiveRC(MethodBase):
    def __init__(self, config):
        self.config = config
        self.alphas = config.alphas
        self.ridge_model = RidgeCV

        self.Win = torch.FloatTensor(
            config.num_neuron, config.input_dim_ridge
        ).uniform_(-config.sigma, config.sigma)

        self.Wrec = torch.rand(config.num_neuron, config.num_neuron) - 0.5
        mask = torch.rand(config.num_neuron, config.num_neuron) < config.sparsity
        self.Wrec[mask] = 0.0

        eigvals = torch.linalg.eigvals(self.Wrec)
        rho_A = torch.max(torch.abs(eigvals)).item()
        self.Wrec *= config.spectral_radius / rho_A

        self.W_out = torch.zeros((config.n_classes, config.num_neuron))
        self.P_rls = [
            torch.eye(config.num_neuron) * config.rls_init
            for _ in range(config.n_classes)
        ]

    def update_state(self, data):
        state = torch.zeros(self.config.num_neuron)
        state_history = torch.zeros(data.shape[0], self.config.num_neuron)

        for fig in range(data.shape[0]):
            update_in = torch.matmul(self.Win, data[fig, :].reshape(-1, 1))
            state = torch.tanh(self.Wrec @ state.reshape(-1, 1) + update_in)
            state_history[fig, :] = state.squeeze(1)

        return state_history

    def compute_accuracy(self, state_history, label):
        y_pred = self.ridge_model.predict(state_history)
        return accuracy_score(
            y_true=np.argmax(label, axis=1), y_pred=np.argmax(y_pred, axis=1)
        )

    def train(self, epoch, dataset, seed: int | None = None):
        data = torch.from_numpy(dataset.X_train)
        label = torch.from_numpy(dataset.y_train)
        state_history = self.update_state(data)

        self.ridge_model = RidgeCV(alphas=self.alphas, fit_intercept=True)
        self.ridge_model.fit(state_history, label)

        return self.compute_accuracy(state_history, label)

    def test(self, epoch, dataset, seed: int | None = None):
        data = torch.from_numpy(dataset.X_test)
        label = dataset.y_test
        state_history = self.update_state(data)

        return self.compute_accuracy(state_history, label)
