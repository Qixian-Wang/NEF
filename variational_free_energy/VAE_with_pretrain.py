import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
writer = SummaryWriter(log_dir="runs/experiment1")

def gaussian_fit_loss(data, mean, rho):
    sigma = torch.log(1 + torch.exp(rho))
    nll = 0.5 * ((data - mean) ** 2 / sigma ** 2) + torch.log(sigma)
    return torch.mean(nll)


def generate_data(n_samples):
    means = [1, 3, 5, 7]
    sds = [1, 2, 3, 4]
    means = np.array(means)
    sds = np.array(sds)

    samples = np.random.normal(loc=means, scale=sds, size=(n_samples, len(means)))
    return samples


def batch_iterator(x):
    n_samples = x.shape[0]

    def _iterator(batch_size):
        sample_indices = np.random.randint(0, high=n_samples, size=batch_size)
        return x[sample_indices, :]

    return _iterator


class CombinedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=2048, latent_dim=4):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.mean_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, pretrain=False):
        encoded = self.encoder(x)
        mu = self.mean_head(encoded)
        log_var = self.logvar_head(encoded)

        if pretrain:
            return mu, log_var
        else:
            z = self.reparameterize(mu, log_var)
            decoded = self.decoder(z)
            return mu, log_var, z, decoded


def train_combined_model(
        pretrain=True,
        n_pretrain_epochs=100,
        lr=1e-4
):
    n_train_data = 5000
    batch_size = 512
    train_data = generate_data(n_train_data)
    train_iterator = batch_iterator(train_data)

    model = CombinedModel(input_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    pretrain_loss = []
    losses = []

    if pretrain:
        for epoch in range(n_pretrain_epochs):
            batch_data = train_iterator(batch_size)
            batch_data_torch = torch.tensor(batch_data, dtype=torch.float32)
            mu, log_var = model(batch_data_torch, pretrain=True)

            mle = gaussian_fit_loss(batch_data_torch, mu, log_var)
            kld = 0.5 * torch.mean(mu.pow(2) + log_var.exp() - 1.0 - log_var)

            loss = torch.mean(mle + kld)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for name, param in model.named_parameters():
                writer.add_histogram(f'weights/{name}', param, epoch)

            writer.add_scalar("Loss/mle", mle, epoch)
            writer.add_scalar("Loss/kld", kld, epoch)
            writer.add_scalar("Loss/loss", loss, epoch)

            pretrain_loss.append(loss.item())
        # samples = np.random.normal(loc=mu.item(), scale=torch.log(1 + torch.exp(log_var)).item(), size=(5000,))
    writer.close()
    # for epoch in range(n_vae_epochs):
    #     batch_data_torch = torch.tensor(train_data, dtype=torch.float32)
    #
    #     mu, log_var, z, decoded = model(batch_data_torch, pretrain=False)
    #     recon_loss = F.mse_loss(decoded, batch_data_torch, reduction='sum')
    #
    #     kld = 0.5 * torch.mean(mu.pow(2) + log_var.exp() - 1.0 - log_var)
    #
    #     loss = recon_loss + kld
    #
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #
    #     losses.append(loss.item())
    # decoded = np.random.normal(loc=mu.item(), scale=torch.log(1 + torch.exp(log_var)).item(), size=(5000,))

    return train_data, losses, pretrain_loss

train_data, loss_pretrain, pretrain_loss = train_combined_model(pretrain=True)
#
# fig, ax = plt.subplots(1, 1, figsize=(12, 4))
#
# ax[0].semilogy(pretrain_loss, label="Loss with pretrain")
# ax[0].set_xlabel("Update steps")
# ax[0].set_ylabel("Loss")
# ax[0].set_title("Training Loss (Pretrain)")
# ax[0].grid(True)
# ax[0].legend()
#
# plt.savefig(f"compare.png", format="png")
