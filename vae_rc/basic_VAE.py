import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from NEF.vae_rc.model.model import VAE
from utils import show_reconstructions, batch_generate, plot_latent_gaussians
from NEF.vae_rc.method.method_base import MethodBase


class NaiveVAE(MethodBase, nn.Module):
    def __init__(self, config):
        super(NaiveVAE, self).__init__()
        self.vae = VAE(config)
        self.config = config

        self.optimizer = optim.Adam(self.vae.parameters(), lr=config.VAE_learning_rate)
        self.writer = self.config.writer
        self.to(self.config.device)

    def forward(self, data):
        encoded, mu, log_var, z, decoded = self.vae(data)

        return encoded, mu, log_var, z, decoded

    def hebbian_projection_loss(self, x, z, activation_fn=torch.nn.Identity()):
        B, D = z.shape
        x = x.view(B, -1)

        fz = activation_fn(z)  # (B, D)

        W = torch.matmul(x.T, fz)  # (input_dim, D)
        W = torch.nn.functional.normalize(W, dim=0)  # 每列归一化

        Y = fz.unsqueeze(2)  # (B, D, 1)
        WT = W.T.unsqueeze(0)  # (1, D, input_dim)

        proj_components = Y * WT
        proj_cumsum = torch.cumsum(proj_components, dim=1)

        x_expanded = x.unsqueeze(1)  # (B, 1, input_dim)
        residuals = x_expanded - proj_cumsum  # (B, D, input_dim)
        loss_per_i = torch.mean(residuals ** 2, dim=(0, 2))
        projection_loss = torch.mean(loss_per_i)

        return projection_loss

    def hebbian_loss(self, z, activation_fn=torch.nn.Identity()):
        """
        z: (batch_size, latent_dim)
        f(y) = activation function
        """
        fz = activation_fn(z)  # f(y)
        B, D = fz.shape
        cov = (fz.T @ fz) / B  # shape: (D, D)

        # 减去对角项（只保留维度之间的相关性）
        loss = torch.sum((cov - torch.diag(torch.diag(cov))) ** 2)
        return loss

    def train(self, epoch: int, dataset, seed: int | None = None):
        self.vae.train()
        total_recon_loss = 0
        total_kld_loss = 0
        total_vae_loss = 0
        total_hebb_loss = 0
        for batch_idx, data in enumerate(
            batch_generate(
                dataset, self.config.batch_size, mode="train", config=self.config
            )
        ):
            encoded, mu, log_var, z, decoded = self.forward(data)

            recon_loss = F.mse_loss(decoded, data)

            kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            hebb_loss = self.hebbian_projection_loss(encoded, z, activation_fn=torch.nn.ReLU())

            vae_loss_batch = 10 * recon_loss + 0.001 * kld_loss + 0 * hebb_loss

            self.optimizer.zero_grad()
            vae_loss_batch.backward()
            self.optimizer.step()

            total_recon_loss += recon_loss.item()
            total_kld_loss += kld_loss.item()
            total_vae_loss += vae_loss_batch.item()
            total_hebb_loss += hebb_loss.item()

        self.writer.add_scalar("Reconstruction_loss", total_recon_loss, epoch)
        self.writer.add_scalar("KLD_loss", total_kld_loss, epoch)
        self.writer.add_scalar("VAE_loss_sum", total_vae_loss, epoch)
        self.writer.add_scalar("Hebb_sum", total_hebb_loss, epoch)
        print(f"Epoch{epoch}")
        print(f"Reconstruction_loss {total_recon_loss:.4f}")
        print(f"KLD_loss {total_kld_loss:.4f}")
        print(f"Hebbian_loss_sum {total_hebb_loss:.4f}")
        print(f"VAE_loss_sum {total_vae_loss:.4f}")

        return None


    def test(self, epoch: int, dataset, seed: int | None = None):
        self.vae.eval()
        toral_rc_loss = 0
        for batch_idx, data in enumerate(
            batch_generate(
                dataset, self.config.batch_size, mode="test", config=self.config
            )
        ):
            encoded, mu, log_var, z, decoded = self.forward(data)

            recon_loss = F.mse_loss(decoded, data)

            toral_rc_loss += recon_loss * self.config.batch_size

        self.writer.add_scalar("Recon loss", toral_rc_loss, epoch)
        if epoch == self.config.num_epoch:
            self.writer.close()
        print(f"Validation_loss {toral_rc_loss:.4f}")
        if epoch % 10 == 0:
            show_reconstructions(self, data)
            plot_latent_gaussians(mu, log_var)

        return None

    def validate(self, dataset):
        self.vae.eval()
