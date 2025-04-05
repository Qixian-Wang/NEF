import torch
import torch.nn as nn
from sklearn.linear_model import RidgeCV


class VAE(nn.Module):
    def __init__(self, config):
        super(VAE, self).__init__()
        self.data_length = config.data_length
        self.num_hidden = config.num_hidden
        self.encoder = nn.Sequential(
            nn.Linear(self.data_length, int(self.data_length / 2)),
            nn.ReLU(),
            nn.Linear(int(self.data_length / 2), self.num_hidden),
            nn.ReLU(),
        )

        self.mu = nn.Linear(self.num_hidden, self.num_hidden)
        self.log_var = nn.Linear(self.num_hidden, self.num_hidden)
        self.decoder = nn.Sequential(
            nn.Linear(self.num_hidden, int(self.data_length / 2)),
            nn.ReLU(),
            nn.Linear(int(self.data_length / 2), self.data_length),
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(z)
        return encoded, mu, log_var, z, decoded


class Regression:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.ridge_model = RidgeCV

    def regression_forward(self, latent_data, label):
        self.ridge_model = RidgeCV(alphas=self.config.alphas, fit_intercept=True)
        self.ridge_model.fit(
            latent_data.detach().cpu().numpy(), label.detach().cpu().numpy()
        )
        y_pred = self.ridge_model.predict(latent_data.detach().cpu().numpy())
        a = self.ridge_model.coef_

        return torch.from_numpy(y_pred).to(self.device)

    def regression_predict(self, data):
        y_pred = self.ridge_model.predict(data.detach().cpu().numpy())

        return torch.from_numpy(y_pred).to(self.device)


class ReservoirComputing:
    def __init__(self, config):
        self.config = config
        self.device = config.device

        self.Win = (
            torch.FloatTensor(config.num_neuron, config.num_hidden)
            .uniform_(-config.reservoir_sigma, config.reservoir_sigma)
            .to(self.device)
        )

        Wrec = torch.rand(config.num_neuron, config.num_neuron) - 0.5
        mask = (
            torch.rand(config.num_neuron, config.num_neuron) < config.reservoir_sparsity
        )
        Wrec[mask] = 0.0

        eigvals = torch.linalg.eigvals(Wrec)
        if eigvals.numel() == 0:
            rho_A = 1e-8
        else:
            rho_A = torch.max(torch.abs(eigvals)).item()
        Wrec *= config.spectral_radius / max(rho_A, 1e-8)
        self.Wrec = Wrec.to(self.device)
        self.ridge_model = RidgeCV

    def update_states(self, latent_data):
        size, fig_dim = latent_data.shape
        state = torch.zeros(self.config.num_neuron, device=self.device)
        state_history = torch.zeros(size, self.config.num_neuron, device=self.device)

        for fig in range(size):
            update_in = torch.matmul(self.Win, latent_data[fig, :].reshape(-1, 1))
            state = torch.tanh(self.Wrec @ state.reshape(-1, 1) + update_in)
            state_history[fig, :] = state.squeeze(1)

        return state_history

    def rc_train(self, latent_data, label):
        state_history = self.update_states(latent_data)

        self.ridge_model = RidgeCV(alphas=self.config.alphas, fit_intercept=False)
        self.ridge_model.fit(
            state_history.detach().cpu().numpy(), label.detach().cpu().numpy()
        )
        y_pred = self.ridge_model.predict(state_history.detach().cpu().numpy())

        return torch.from_numpy(y_pred).to(self.device)

    def rc_predict(self, latent_data):
        state_history = self.update_states(latent_data)
        y_pred = self.ridge_model.predict(state_history.detach().cpu().numpy())

        return torch.from_numpy(y_pred).to(self.device)
