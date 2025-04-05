import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
import numpy as np

from model import VAE, ReservoirComputing
from utils import batch_generate, show_reconstructions
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

    def forward(self, data, label, mode):
        encoded, mu, log_var, z, decoded = self.vae(data)
        if mode == "train":
            prediction = self.rc.rc_train(z, label)
        if mode == "test":
            prediction = self.rc.rc_predict(z)

        return encoded, mu, log_var, z, decoded, prediction

    def train(self, epoch: int, dataset, seed: int | None = None):
        self.vae.train()
        result_list = []
        label_list = []
        totoal_recon_loss = 0
        totoal_kld_loss = 0
        totoal_vae_loss = 0
        for batch_idx, (data, label) in enumerate(
            batch_generate(
                dataset, self.config.batch_size, mode="train", config=self.config
            )
        ):
            encoded, mu, log_var, z, decoded, prediction = self.forward(
                data, label, mode="train"
            )

            recon_loss = F.mse_loss(decoded, data)
            kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            vae_loss_batch = 10 * recon_loss + 1 * kld_loss
            _, result = prediction.max(dim=1)

            label_list.extend(label.detach().cpu().numpy())
            result_list.extend(result.detach().cpu().numpy())

            total_loss_batch = vae_loss_batch
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            self.optimizer.step()

            totoal_recon_loss += recon_loss.item()
            totoal_kld_loss += kld_loss.item()
            totoal_vae_loss += vae_loss_batch.item()

        accuracy = accuracy_score(
            y_true=np.argmax(label_list, axis=1), y_pred=np.array(result_list)
        )
        self.writer.add_scalar("Reconstruction_loss", totoal_recon_loss, epoch)
        self.writer.add_scalar("KLD_loss", totoal_kld_loss, epoch)
        self.writer.add_scalar("VAE_loss_sum", totoal_vae_loss, epoch)
        self.writer.add_scalar("Training Accuracy", accuracy, epoch)

        return accuracy

    def test(self, epoch: int, dataset, seed: int | None = None):
        self.vae.eval()
        result_list = []
        label_list = []
        for batch_idx, (data, label) in enumerate(
            batch_generate(
                dataset, self.config.batch_size, mode="test", config=self.config
            )
        ):
            encoded, mu, log_var, z, decoded, prediction = self.forward(
                data, label, mode="test"
            )

            _, pred_label = prediction.max(dim=1)
            label_list.extend(label.detach().cpu().numpy())
            result_list.extend(pred_label.detach().cpu().numpy())

        accuracy = accuracy_score(
            y_true=np.argmax(label_list, axis=1), y_pred=np.array(result_list)
        )
        self.writer.add_scalar("Testing Accuracy", accuracy, epoch)

        # show_reconstructions(self, data, label)
        return accuracy
