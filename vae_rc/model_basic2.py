import torch
import numpy as np
from sklearn.linear_model import RidgeCV


class ReservoirComputing:
    def __init__(self, config):
        self.num_neuron = config.num_neuron[0]
        self.n_classes = config.n_classes[0]
        self.input_dim_ridge = config.input_dim_ridge[0]
        self.sigma = config.sigma[0]
        self.sparsity = config.sparsity[0]
        self.spectral_radius = config.spectral_radius[0]
        self.rls_init = int(config.rls_init[0])
        self.alphas = config.alphas

        self.Win = torch.FloatTensor(self.num_neuron, self.input_dim_ridge).uniform_(-self.sigma, self.sigma)

        Wrec = (torch.rand(self.num_neuron, self.num_neuron) - 0.5)
        mask = (torch.rand(self.num_neuron, self.num_neuron) < self.sparsity)
        Wrec[mask] = 0.0

        eigvals = torch.linalg.eigvals(Wrec)
        if eigvals.numel() == 0:
            rho_A = 1e-8
        else:
            rho_A = torch.max(torch.abs(eigvals)).item()
        Wrec *= (self.spectral_radius / max(rho_A,1e-8))
        self.Wrec = Wrec
        self.W_out = torch.zeros((self.n_classes, self.num_neuron))
        self.P_rls = [torch.eye(self.num_neuron)*self.rls_init for _ in range(self.n_classes)]


    def train(self, x_seq, label):
        size, fig_dim = x_seq.shape
        state = torch.zeros(self.num_neuron)
        state_history = torch.zeros(size, self.num_neuron)

        for fig in range(size):
            update_in = torch.matmul(self.Win, x_seq[fig, :].reshape(-1, 1))
            state = torch.tanh(self.Wrec @ state.reshape(-1, 1) + update_in)
            state_history[fig, :] = state.squeeze(1)

        self.model = RidgeCV(alphas=self.alphas, fit_intercept=True)
        self.model.fit(state_history, label)


    def test(self, data):
        size, fig_dim = data.shape
        state = torch.zeros(self.num_neuron)
        state_history = torch.zeros(size, self.num_neuron)

        for fig in range(size):
            update_in = torch.matmul(self.Win, data[fig, :].reshape(-1, 1))
            state = torch.tanh(self.Wrec @ state.reshape(-1, 1) + update_in)
            state_history[fig, :] = state.squeeze(1)

        prediction = self.model.predict(state_history)
        return prediction


def train_rc(train_dataset, config):
    model = ReservoirComputing(config)

    data = torch.from_numpy(train_dataset.X_train)
    label = torch.from_numpy(train_dataset.y_train)
    model.train(data, label)

    return model

def test_rc(model, test_dataset):
    data = torch.from_numpy(test_dataset.X_test)
    label = torch.from_numpy(test_dataset.y_test)

    prediction = model.test(data)
    result = np.argmax(prediction, axis=1)

    correct_predict = (result == label).sum()
    total = result.shape[0]

    print(f"accuracy: {correct_predict/total * 100}%")
    return model
