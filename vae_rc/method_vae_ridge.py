import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from sklearn.metrics import accuracy_score

from method_base import MethodBase
from utils import batch_generate, show_reconstructions
from model import VAE, Regression


class MethodVAERegression(MethodBase, nn.Module):
    def __init__(self, config):
        super(MethodVAERegression, self).__init__()
        self.config = config
        self.vae = VAE(config)
        self.regression = Regression(config)
        self.optimizer = optim.Adam(
            self.vae.parameters(), lr=self.config.VAE_learning_rate
        )
        self.to(self.config.device)

    def forward(self, data, label, mode):
        encoded, mu, log_var, z, decoded = self.vae(data)

        if mode == "train":
            prediction = self.regression.regression_forward(z, label)
        if mode == "test":
            prediction = self.regression.regression_predict(z)

        return encoded, mu, log_var, z, decoded, prediction

    def train(self, epoch, dataset, seed: int | None = None):
        self.vae.train()
        result_list = []
        label_list = []

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
            vae_loss_batch = 10 * recon_loss + kld_loss
            _, result = prediction.max(dim=1)

            label_list.extend(label.detach().cpu().numpy())
            result_list.extend(result.detach().cpu().numpy())

            total_loss_batch = vae_loss_batch
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            self.optimizer.step()

        return accuracy_score(
            y_true=np.argmax(label_list, axis=1), y_pred=np.array(result_list)
        )

    def test(self, epoch, dataset, seed: int | None = None):
        self.vae.eval()
        result_list = []
        label_list = []
        correct_predict = 0
        total_samples = 0
        for batch_idx, (data, label) in enumerate(
            batch_generate(
                dataset, self.config.batch_size, mode="test", config=self.config
            )
        ):

            encoded, mu, log_var, z, decoded, prediction = self.forward(
                data, label, mode="test"
            )

            pred_label = prediction.argmax(dim=1)
            label_list.extend(label.detach().cpu().numpy())
            result_list.extend(pred_label.detach().cpu().numpy())

        # show_reconstructions(self, data, label)
        return accuracy_score(
            y_true=np.argmax(label_list, axis=1), y_pred=np.array(result_list)
        )
