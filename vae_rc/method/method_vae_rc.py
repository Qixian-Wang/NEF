import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.linear_model import LogisticRegression

from NEF.vae_rc.model.model import VAE, ReservoirComputing
from NEF.vae_rc.utils import batch_generate, show_reconstructions
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

        self.classifier = LogisticRegression(
            solver='lbfgs', multi_class='auto', max_iter=1000
        )

    def forward(self, data):
        encoded, mu, log_var, z, decoded = self.vae(data)

        return encoded, mu, log_var, z, decoded

    def hebbian_projection_loss(self, x, z, activation_fn=torch.nn.Identity()):
        B, D = z.shape
        x = x.view(B, -1)
        fz = activation_fn(z)

        W = torch.matmul(x.T, fz)
        W = torch.nn.functional.normalize(W, dim=0)

        Y = fz.unsqueeze(2)
        WT = W.T.unsqueeze(0)

        proj_components = Y * WT
        proj_cumsum = torch.cumsum(proj_components, dim=1)
        x_expanded = x.unsqueeze(1)
        residuals = x_expanded - proj_cumsum

        loss_per_i = torch.mean(residuals ** 2, dim=(0, 2))
        projection_loss = torch.mean(loss_per_i)

        return projection_loss

    def train(self, epoch: int, dataset, seed: int | None = None):
        self.vae.train()
        total_recon_loss = 0
        total_kld_loss = 0
        total_vae_loss = 0
        total_hebb_loss = 0

        for batch_idx, (data, label) in enumerate(
            batch_generate(
                dataset, self.config.batch_size, mode="train", config=self.config
            )
        ):
            encoded, mu, log_var, z, decoded = self.forward(data)

            recon_loss = F.mse_loss(decoded, data)
            kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

            hebb_loss = self.hebbian_projection_loss(encoded, z, activation_fn=torch.nn.ReLU())

            vae_loss_batch = 10 * recon_loss + 0.001 * kld_loss + 0.5 * hebb_loss

            total_loss_batch = 1 * vae_loss_batch
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            self.optimizer.step()

            total_recon_loss += recon_loss.item()
            total_kld_loss += kld_loss.item()
            total_vae_loss += vae_loss_batch.item()
            total_hebb_loss += hebb_loss.item()

        loss = total_vae_loss + total_hebb_loss
        self.writer.add_scalar("[train] Reconstruction_loss", total_recon_loss, epoch)
        self.writer.add_scalar("[train] KLD_loss", total_kld_loss, epoch)
        self.writer.add_scalar("[train] VAE_loss_sum", total_vae_loss, epoch)
        self.writer.add_scalar("[train] Hebbian_loss_sum", total_hebb_loss, epoch)
        self.writer.add_scalar("[train] Training loss", loss, epoch)
        print(f"Epoch{epoch}")
        print(f"[train] Reconstruction_loss {total_recon_loss:.4f}")
        print(f"[train] KLD_loss {total_kld_loss:.4f}")
        print(f"[train] Hebbian_loss {total_hebb_loss:.4f}")
        print(f"[train] VAE_loss_sum {total_vae_loss:.4f}")


    def data_collection(self, dataset, mode):
        latent_list = []
        data_list = []
        labels_list = []
        for batch_idx, (data, label) in enumerate(
                batch_generate(
                    dataset, self.config.batch_size, mode=mode, config=self.config
                )
        ):
            encoded, mu, log_var, z, decoded = self.forward(data)
            latent_list.append(z)
            data_list.append(data)
            labels_list.append(label)

        return torch.cat(latent_list, dim=0), torch.cat(data_list, dim=0), torch.cat(labels_list, dim=0)


    def validate(self, epoch: int, dataset, seed: int | None = None):
        self.vae.eval()
        latent_list, data_list, _ = self.data_collection(dataset, mode="train")
        rc_reconstruction_train = self.rc.rc_train_costom(latent_list, data_list)

        latent_list_valid, data_list_valid, _ = self.data_collection(dataset, mode="test")
        rc_reconstruction = self.rc.rc_predict_costom(latent_list_valid)

        test_mse = F.mse_loss(rc_reconstruction, data_list_valid)

        self.writer.add_scalar("[validate] Validation loss", test_mse, epoch)
        print(f"[validation] RC reconstruction MSE: {test_mse.item():.6f}")
        return test_mse.item()

    def test(self, dataset, seed: int | None = None):
        self.vae.eval()
        latent_list, _, label_list = self.data_collection(dataset, mode="train")
        prediction_train = self.rc.rc_train_costom(latent_list, label_list)

        latent_list_test, _, label_list_test = self.data_collection(dataset, mode="train")
        prediction = self.rc.rc_predict_costom(latent_list_test)
        accuracy = accuracy_score(
            y_true=np.argmax(label_list_test.detach().cpu().numpy(), axis=1), y_pred=np.argmax(prediction.detach().cpu().numpy(), axis=1)
        )
        print(f"[test] Final accuracy: {accuracy * 100:.6f}%")