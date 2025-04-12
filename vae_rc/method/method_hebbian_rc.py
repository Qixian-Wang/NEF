import torch
import torch.nn as nn
from NEF.vae_rc.utils import batch_generate
from NEF.vae_rc.method.method_base import MethodBase
from NEF.vae_rc.model.model import ReservoirComputing
from NEF.vae_rc.model.Hebbian_model import HebbianLearning
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import itertools

class MethodHebbian_RC(MethodBase, nn.Module):
    def __init__(self, config):
        super(MethodHebbian_RC, self).__init__()
        self.config = config

        self.ff_layers = nn.ModuleList([
            HebbianLearning(config, in_channels=1,  out_size=16, kernel_size=28),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # HebbianLearning(config, in_channels=128, out_size=64, kernel_size=5),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # HebbianLearning(config, in_channels=64, out_size=16, kernel_size=3),
        ])

        self.rc = ReservoirComputing(config)
        self.optimizer = torch.optim.Adam(self.ff_layers.parameters(), lr=1e-4)
        self.writer = self.config.writer
        self.to(self.config.device)

    def forward(self, data, label, epoch):
        batch_size = data.size(0)
        data = data.view(batch_size, 1, 28, 28)

        for layer in self.ff_layers:
            if isinstance(layer, HebbianLearning):
                data = layer(data, label, epoch)
                data = self.detach_norm(data)
            else:
                data = layer(data)
        return data

    def detach_norm(self, z):
        z = z.detach()
        eps = 1e-8
        z = z / (z.pow(2).mean(dim=[1, 2, 3], keepdim=True).sqrt() + eps)
        return z

    def train(self, epoch: int, dataset, seed: int = None):
        self.ff_layers.train()
        recon_loss_total = 0
        kld_loss_total = 0
        latents = []
        labels = []
        posneg_labels = []
        for batch_idx, (data, label, posneg_label) in enumerate(
            batch_generate(dataset, self.config.batch_size, mode="train", ff_data=True, config=self.config)
        ):
            latent = self.forward(data, posneg_label, epoch)
            latents.append(latent)
            labels.append(label)
            posneg_labels.append(posneg_label)
            self.optimizer.zero_grad()
            for layer in self.ff_layers:
                if isinstance(layer, HebbianLearning):
                    recon_loss = layer.local_update()
                    recon_loss_total += recon_loss.item()
            self.optimizer.step()

        latents = torch.cat(latents, dim=0)
        posneg_labels = torch.cat(posneg_labels, dim=0)
        mask = (posneg_labels == 1).bool()

        for i in range(self.config.num_hidden):
            x_i = latents[:, i].squeeze(2)
            x_i = x_i[mask]
            mu_i, std_i = x_i.mean(), x_i.std()
            kld_loss_total += 0.5 * torch.sum(mu_i.pow(2) + std_i.exp() - 1 - std_i).mean()

        recon_loss_total = recon_loss_total / dataset.X_train.shape[0]
        kld_loss_total = recon_loss_total / dataset.X_train.shape[0]
        self.writer.add_scalar("[train] recon_loss_total", recon_loss_total, epoch)
        self.writer.add_scalar("[train] kld_loss_total", kld_loss_total, epoch)
        print(f"Epoch{epoch}")
        print(f"[train] Recon_total {recon_loss_total:.4f}")
        print(f"[train] KLD_total {kld_loss_total:.4f}")


    def data_collection(self, dataset, mode: str):
        self.ff_layers.eval()
        latents = []
        labels = []
        with torch.no_grad():
            for data, label, posneg_labels in batch_generate(dataset, self.config.batch_size, mode=mode, ff_data=False, config=self.config):
                z = self.forward(data, label, 0)
                latents.append(z)
                labels.append(label)

        return torch.cat(latents, dim=0), torch.cat(labels, dim=0)

    def validate(self, epoch: int, dataset, seed: int = None):
        latent_train, label_train = self.data_collection(dataset, mode="train")
        latent_train = latent_train.reshape(latent_train.shape[0], -1)
        self.rc.rc_train(latent_train, label_train)

        latent_val, label_val = self.data_collection(dataset, mode="test")
        latent_val = latent_val.reshape(latent_val.shape[0], -1)
        pred_val = self.rc.rc_predict(latent_val)

        y_true = label_val.argmax(dim=1).cpu().numpy()
        y_pred = pred_val.argmax(dim=1).cpu().numpy()
        acc = accuracy_score(y_true, y_pred)

        self.writer.add_scalar("[train] val_accuracy", acc, epoch)
        print(f"[validation] Accuracy: {acc * 100:.2f}%")

        latent_np = latent_val.detach().cpu().numpy()

        mask = (y_true == 0)  # shape (N,), bool
        latent0 = latent_np[mask]

        kmeans = KMeans(n_clusters=10, random_state=0).fit(latent_np)
        labels_pred = kmeans.labels_
        sil = silhouette_score(latent_np, labels_pred)
        ari = adjusted_rand_score(y_true, labels_pred)
        nmi = normalized_mutual_info_score(y_true, labels_pred)
        self.writer.add_scalar("[validation] sil", sil, epoch)
        self.writer.add_scalar("[validation] ari", ari, epoch)
        self.writer.add_scalar("[validation] nmi", nmi, epoch)
        print(f"Silhouette: {sil:.3f}, ARI: {ari:.3f}, NMI: {nmi:.3f}")

        # plt.figure(figsize=(10, 6))
        # x = np.linspace(-8, 8, 500)
        # for i in range(16):
        #     x_i = latent0[:, i]
        #     mu_i, std_i = x_i.mean(), x_i.std()
        #
        #     pdf = norm.pdf(x, loc=mu_i, scale=std_i)
        #     plt.plot(x, pdf, label=f'z[{i}]', linewidth=1.2)
        #
        # plt.xlabel("z value")
        # plt.ylabel("Probability Density")
        # plt.grid(True)
        # plt.legend(ncol=2, fontsize=8)
        # plt.tight_layout()
        # plt.show()

        return acc

    def test(self, dataset, seed: int = None):
        latent_train, label_train = self.data_collection(dataset, mode="train")
        self.rc.rc_train_costom(latent_train.to(self.config.device), label_train.to(self.config.device))

        latent_test, label_test = self.data_collection(dataset, mode="test")
        pred_test = self.rc.rc_predict(latent_test.to(self.config.device))

        y_true = label_test.argmax(dim=1).cpu().numpy()
        y_pred = pred_test.argmax(dim=1).cpu().numpy()
        acc = accuracy_score(y_true, y_pred)

        print(f"[Test] Final RC accuracy = {acc*100:.2f}%")
        return acc
