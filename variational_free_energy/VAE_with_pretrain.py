import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

def gaussian_fit_loss(data, mean, rho):
    sigma = torch.log(1 + torch.exp(rho))
    nll = 0.5 * ((data - mean) ** 2 / sigma ** 2) + torch.log(sigma)
    return torch.mean(nll)


def generate_data(n_samples, random_seed=42):
    np.random.seed(random_seed)
    w = np.array([0.2, 0.8])
    mu = np.array([-0.1, 0.5])
    sd = np.array([0.1, 0.1])

    component_choices = np.random.choice(len(w), size=n_samples, p=w)
    samples = np.random.normal(loc=mu[component_choices], scale=sd[component_choices])
    return samples


def batch_iterator(x):
    n_samples = x.shape[0]

    def _iterator(batch_size):
        sample_indices = np.random.randint(0, high=n_samples, size=batch_size)
        return x[sample_indices]

    return _iterator


class CombinedModel(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, latent_dim=1):
        super().__init__()

        # ----- Encoder -----
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
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
        n_vae_epochs=200,
        batch_size=256,
        lr=5e-3
):

    n_train_data = 500
    train_data = generate_data(n_train_data)
    train_iterator = batch_iterator(train_data)

    # Model
    model = CombinedModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    pretrain_loss = []
    losses = []
    samples = 0
    if pretrain:
        for epoch in range(n_pretrain_epochs):
            batch_data = train_iterator(batch_size)
            batch_data_torch = torch.tensor(batch_data, dtype=torch.float32)
            mu, log_var = model(batch_data_torch, pretrain=True)

            mle = gaussian_fit_loss(batch_data_torch, mu, log_var)
            kld = 0.5 * torch.mean(mu.pow(2) + log_var.exp() - 1.0 - log_var)

            loss = mle + kld

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pretrain_loss.append(loss.item())
        samples = np.random.normal(loc=mu.item(), scale=torch.log(1 + torch.exp(log_var)).item(), size=(5000,))


    for epoch in range(n_vae_epochs):
        batch_data = train_iterator(batch_size)
        batch_data_torch = torch.tensor(batch_data, dtype=torch.float32)

        mu, log_var, z, decoded = model(batch_data_torch, pretrain=False)
        recon_loss = F.mse_loss(decoded, batch_data_torch, reduction='sum')

        kld = 0.5 * torch.mean(mu.pow(2) + log_var.exp() - 1.0 - log_var)

        loss = recon_loss + kld

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return train_data, losses, decoded, pretrain_loss, samples

train_data, loss_pretrain, decoded1, pretrain_loss, samples = train_combined_model(pretrain=True)
_, loss_no, decoded2, _, _ = train_combined_model(pretrain=False)

fig, ax = plt.subplots(1, 4, figsize=(12, 4))
sns.kdeplot(train_data, color="blue", label="Real PDF", linewidth=2, ax=ax[0])
sns.kdeplot(samples, color="green", label="pretrain PDF", linewidth=2, ax=ax[0])
sns.kdeplot(decoded1.detach().cpu().numpy().flatten(), color="red", label="final PDF (pretrain)", linewidth=2, ax=ax[0])
sns.kdeplot(decoded2.detach().cpu().numpy().flatten(), color="orange", label="final PDF (no pretrain)", linewidth=2, ax=ax[0])
ax[0].set_xlabel("Data values")
ax[0].set_ylabel("Density")
ax[0].set_title("PDF Comparison")
ax[0].grid(True)
ax[0].legend()

ax[1].plot(loss_pretrain, label="Loss with pretrain")
ax[1].set_xlabel("Update steps")
ax[1].set_ylabel("Loss")
ax[1].set_title("Training Loss (Pretrain)")
ax[1].grid(True)
ax[1].legend()

ax[2].plot(loss_no, label="Loss without pretrain")
ax[2].set_xlabel("Update steps")
ax[2].set_ylabel("Loss")
ax[2].set_title("Training Loss (No Pretrain)")
ax[2].grid(True)
ax[2].legend()
ax[2].sharey(ax[1])
#
# ax[3].plot(pretrain_loss, label="pretrain loss")
# ax[3].set_xlabel("Update steps")
# ax[3].set_ylabel("Loss")
# ax[3].set_title("Training Loss (No Pretrain)")
# ax[3].grid(True)
# ax[3].legend()
# ax[3].sharey(ax[1])

plt.savefig(f"compare.png", format="png")
