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
            # HebbianLearning(config, in_channels=128, out_size=(64,1), kernel_size=5),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # HebbianLearning(config, in_channels=64, out_size=(16,1), kernel_size=3),
        ])

        self.rc = ReservoirComputing(config)
        self.optimizer = torch.optim.Adam(self.ff_layers.parameters(), lr=1e-4)
        self.writer = self.config.writer
        self.to(self.config.device)

    def forward(self, data, label):
        batch_size = data.size(0)
        z = data.view(batch_size, 1, 28, 28)

        for layer in self.ff_layers:
            if isinstance(layer, HebbianLearning):
                z = layer(z, label)
                z = self.detach_norm(z)
            else:
                z = layer(z)
        return z

    def detach_norm(self, z):
        z = z.detach()
        eps = 1e-8
        z = z / (z.pow(2).mean(dim=[1, 2, 3], keepdim=True).sqrt() + eps)
        return z

    def train(self, epoch: int, dataset, seed: int = None):
        self.ff_layers.train()
        ff_loss_total = 0
        for batch_idx, (data, label, posneg_labels) in enumerate(
            batch_generate(dataset, self.config.batch_size, mode="train", ff_data=True, config=self.config)
        ):
            self.forward(data, posneg_labels)
            self.optimizer.zero_grad()
            for layer in self.ff_layers:
                if isinstance(layer, HebbianLearning):
                    hebbian_loss, ff_loss = layer.local_update()
            self.optimizer.step()

            ff_loss_total += ff_loss.item()

        self.writer.add_scalar("[train] ff_loss_total", ff_loss_total, epoch)
        print(f"Epoch{epoch}")
        print(f"[train] ff_loss_total {ff_loss_total:.4f}")


    def data_collection(self, dataset, mode: str):
        self.ff_layers.eval()
        latents = []
        labels = []
        with torch.no_grad():
            for data, label, posneg_labels in batch_generate(dataset, self.config.batch_size, mode=mode, ff_data=False, config=self.config):
                z = self.forward(data, label)
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
        print(f"Silhouette: {sil:.3f}, ARI: {ari:.3f}, NMI: {nmi:.3f}")

        plt.figure(figsize=(10, 6))
        x = np.linspace(-8, 8, 500)
        for i in range(16):
            x_i = latent0[:, i]
            mu_i, std_i = x_i.mean(), x_i.std()

            pdf = norm.pdf(x, loc=mu_i, scale=std_i)
            plt.plot(x, pdf, label=f'z[{i}]', linewidth=1.2)

        plt.xlabel("z value")
        plt.ylabel("Probability Density")
        plt.grid(True)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.show()

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
