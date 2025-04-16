import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA

from NEF.vae_rc.utils import batch_generate
from NEF.vae_rc.method.method_base import MethodBase
from NEF.vae_rc.model.model import ReservoirComputing
from NEF.vae_rc.model.Hebbian_model import HebbianLearning
from NEF.vae_rc.dataloader import DatasetMNIST, data_generator
from NEF.vae_rc.dataloader import ColoredGaussianDataset

import NEF.vae_rc.config_file as config_file


class HebbianComparisonPCA(MethodBase, nn.Module):
    def __init__(self, config):
        super(HebbianComparisonPCA, self).__init__()
        self.config = config

        self.hebb = HebbianLearning(config, input_dim=64,  out_size=config.num_hidden)
        self.writer = self.config.writer
        self.to(self.config.device)

    def forward(self, data):
        data = self.hebb(data)
        return data

    def train(self, epoch: int, dataset, seed: int = None):
        self.hebb.train()
        latents = []
        labels = []
        datas = []
        if __name__ == "__main__":
            for batch_idx, data in enumerate(gauss_loader):
                data = data.to(self.config.device)
                latent = self.hebb(data)
                latents.append(latent)
                datas.append(data)
                recon_loss = self.hebb.local_update(data)
                if batch_idx > 64:
                    break
            datas = torch.cat(datas, 0).detach().cpu().numpy()
            latents = torch.cat(latents, 0).detach().cpu().numpy()

        else:
            for batch_idx, (data, label, posneg_label) in enumerate(
                batch_generate(dataset, self.config.batch_size, mode="train", ff_data=True, config=self.config)
            ):
                latent = self.forward(data)
                latents.append(latent)
                labels.append(label)
                datas.append(data)
                recon_loss = self.hebb.local_update(data)
            datas = torch.cat(datas, dim=0).detach().cpu().numpy()
            latents = torch.cat(latents, dim=0).detach().cpu().numpy()

        self.compute_pca_metrics(
            epoch, datas, latents, self.hebb.feedforward_weights.detach().cpu().numpy(),
            self.hebb.lateral_weights.detach().cpu().numpy(), num_components=self.config.num_hidden
        )

    def validate(self, epoch: int, dataset, seed: int | None = None):
        pass

    def test(self, dataset, seed: int | None = None):
        pass

    def compute_pca_metrics(self, epoch, data, latent, w_yx, w_yy, num_components):
        data_central = data - data.mean(axis=0, keepdims=True)
        latent_central = latent - latent.mean(axis=0, keepdims=True)

        pca = PCA(n_components=num_components, svd_solver='randomized', whiten=False)
        pca.fit(data_central)

        # EigenErr
        offline_eigs = pca.explained_variance_
        cov_latent = latent_central.T @ latent_central / data.shape[0]
        cov_latent = (cov_latent + cov_latent.T) * 0.5
        cov_latent += 1e-6 * np.eye(cov_latent.shape[0])
        online_eigs = np.linalg.eigvalsh(cov_latent)[::-1][:num_components]

        eigen_err = 10 * np.log10(np.sum((online_eigs - offline_eigs)**2) + 1e-12)

        # SubspaceErr
        vector = pca.components_.T
        mat1 = np.eye(w_yx.shape[1]) + w_yy
        f_full_trans = np.linalg.solve(mat1, w_yx.T)
        f_full = f_full_trans.T

        base_vector, _, _ = np.linalg.svd(f_full, full_matrices=False)
        f_main_component = base_vector[:, :num_components]
        diff_subspace = f_main_component @ f_main_component.T - vector @ vector.T
        subspace_err = 10 * np.log10(np.linalg.norm(diff_subspace, 'fro') ** 2 + 1e-12)

        # non_orthogonal error
        mat2 = np.eye(w_yx.shape[1]) + w_yy
        f_full_trans = np.linalg.solve(mat2, w_yx.T)
        f_full = f_full_trans.T
        f_mul = f_full.T @ f_full
        diff_ortho = f_mul - np.eye(f_mul.shape[0])
        non_ortho = 10 * np.log10(np.linalg.norm(diff_ortho, 'fro') ** 2 + 1e-12)

        self.writer.add_scalar("[validation] eigen_err", eigen_err, epoch)
        self.writer.add_scalar("[validation] subspace_err", subspace_err, epoch)
        self.writer.add_scalar("[validation] non_ortho_err", non_ortho, epoch)
        print(f"Epoch {epoch} EigenErr: {eigen_err:.2f} dB")
        print(f"Epoch {epoch} SubspaceErr: {subspace_err:.2f} dB")
        print(f"Epoch {epoch} non_ortho_err: {non_ortho:.2f} dB")


if __name__ == "__main__":
    gauss_ds = ColoredGaussianDataset(D=64, top_eigs=[7, 6, 5, 4], low=(0, 0.5), seed=42)
    gauss_loader = DataLoader(gauss_ds, batch_size=64)

    seed = 42
    torch.manual_seed(seed)
    dataset = data_generator(config_file.configs)
    method = HebbianComparisonPCA(config_file.configs)

    for epoch_idx in range(1000):
        print(f"epoch {epoch_idx}")
        method.train(epoch_idx, dataset, seed)
        method.validate(epoch_idx, dataset, seed)

    config_file.configs.writer.close()
