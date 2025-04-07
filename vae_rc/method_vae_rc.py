import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

from model import VAE, ReservoirComputing
from utils import batch_generate, show_reconstructions, plot_y_pred
from method_base import MethodBase


class MethodVAERC(MethodBase, nn.Module):
    def __init__(self, config):
        super(MethodVAERC, self).__init__()
        self.vae = VAE(config)
        self.rc = ReservoirComputing(config)
        self.config = config

        self.optimizer = optim.Adam(self.vae.parameters(), lr=config.VAE_learning_rate)
        self.writer = self.config.writer
        self.to(self.config.device)
        self.readout = nn.Linear(config.num_neuron, config.data_length, bias=False).to(config.device)

    def forward(self, data, mode):
        encoded, mu, log_var, z, decoded = self.vae(data)

        if mode == "train":
            rc_reconstruction = self.rc.rc_train_costom(z, data)
        if mode == "test":
            rc_reconstruction = self.rc.rc_predict_costom(z)

        return encoded, mu, log_var, z, decoded, rc_reconstruction

    def train(self, epoch: int, dataset, seed: int | None = None):
        self.vae.train()
        total_recon_loss = 0
        total_kld_loss = 0
        total_vae_loss = 0
        total_rc_loss = 0
        for batch_idx, data in enumerate(
            batch_generate(
                dataset, self.config.batch_size, mode="train", config=self.config
            )
        ):
            encoded, mu, log_var, z, decoded, rc_reconstruction = self.forward(
                data, mode="train"
            )

            recon_loss = F.mse_loss(decoded, data)
            kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            vae_loss_batch = 10 * recon_loss + 0.01 * kld_loss
            rc_loss = F.mse_loss(rc_reconstruction, data)

            total_loss_batch = 1 * vae_loss_batch + 0.01 * rc_loss
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            self.optimizer.step()

            total_recon_loss += recon_loss.item()
            total_kld_loss += kld_loss.item()
            total_vae_loss += vae_loss_batch.item()
            total_rc_loss += rc_loss.item() * self.config.batch_size


        loss = total_vae_loss + rc_loss
        self.writer.add_scalar("Reconstruction_loss", total_recon_loss, epoch)
        self.writer.add_scalar("KLD_loss", total_kld_loss, epoch)
        self.writer.add_scalar("VAE_loss_sum", total_vae_loss, epoch)
        self.writer.add_scalar("RC_loss_sum", total_rc_loss, epoch)
        self.writer.add_scalar("Training Accuracy", loss, epoch)
        print(f"Epoch{epoch}")
        print(f"Reconstruction_loss {total_recon_loss:.4f}")
        print(f"KLD_loss {total_kld_loss:.4f}")
        print(f"VAE_loss_sum {total_vae_loss:.4f}")
        print(f"RC_loss_sum {total_rc_loss:.4f}")

        return loss

    def test(self, epoch: int, dataset, seed: int | None = None):
        self.vae.eval()
        toral_rc_loss = 0
        for batch_idx, data in enumerate(
            batch_generate(
                dataset, self.config.batch_size, mode="test", config=self.config
            )
        ):
            encoded, mu, log_var, z, decoded, rc_reconstruction = self.forward(
                data, mode="test"
            )

            rc_loss = F.mse_loss(rc_reconstruction, data)
            toral_rc_loss += rc_loss.item() * self.config.batch_size

        self.writer.add_scalar("Validating loss", toral_rc_loss, epoch)
        if epoch == self.config.num_epoch:
            self.writer.close()

        # if epoch % 2 == 0:
        #     plot_y_pred(rc_reconstruction, n_images=5, config=self.config)
        #     show_reconstructions(self, data)
        return toral_rc_loss
