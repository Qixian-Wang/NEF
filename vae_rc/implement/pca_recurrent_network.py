import torch
from torch.utils.data import DataLoader
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from NEF.vae_rc.model.Hebbian_model import HebbianLearning
from NEF.vae_rc.dataloader import ColoredGaussianDataset
import NEF.vae_rc.config_file as config_file


def compute_pca_metrics(data, latent, w_yx, w_yy, num_components):
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

    eigen_err = 10 * np.log10(np.sum((online_eigs - offline_eigs) ** 2) + 1e-12)

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

    print(f"EigenErr: {eigen_err:.2f} dB")
    print(f"SubspaceErr: {subspace_err:.2f} dB")
    print(f"non_ortho_err: {non_ortho:.2f} dB")

def recurrent_forward(data):
    state_linear = (data @ hebb.feedforward_weights + hebb.state @ hebb.lateral_weights) / hebb.feedforward_weights.size(0)
    state = torch.tanh(state_linear)
    hebb.current_output = state

    hebb.state = state.detach()
    return state

def train_rnn(T_max, gauss_iter):
    for t in range(T_max):
        x_t = next(gauss_iter)
        x_t = x_t.to(device)
        y_t = hebb(x_t)
        optimizer.zero_grad()
        loss = hebb.local_update(x_t)
        optimizer.step()

def loop(hebb, data, steps=50):
    for _ in range(steps):
        state = hebb(data)
    return state.detach()

seed = 42
torch.manual_seed(seed)
config = config_file.configs
device = config.device
gauss_ds = ColoredGaussianDataset(D=64, top_eigs=[7, 6, 5, 4], low=(0, 0.5), seed=42)
gauss_loader = DataLoader(gauss_ds, batch_size=200)

hebb = HebbianLearning(config, input_dim=64, out_size=config.num_hidden)
hebb.to(device)
hebb.state = torch.zeros(1, config.num_hidden, device=device)

hebb.forward = recurrent_forward

optimizer = torch.optim.SGD(hebb.parameters(), lr=1e-6)

T_max = 200
gauss_iter = iter(gauss_loader)
train_rnn(T_max, gauss_iter)
print("Training finished.")

data_base = next(gauss_iter).to(device)
state_base = loop(hebb, data_base, steps=100)

num_dirs = 128
dim = data_base.size(1)
eps = 1e-3
perturb = torch.randn(num_dirs, dim, device=device)
perturb = perturb / perturb.norm(dim=1, keepdim=True)

state_dot_list = []
for i in range(num_dirs):
    hebb.state = state_base.clone()
    perturb_vec = perturb[i:i+1]
    data_perturbed = data_base + eps * perturb_vec
    state_perturbed = loop(hebb, data_perturbed, steps=100)
    state_dot = (state_perturbed - state_base) / eps
    state_dot_list.append(state_dot.cpu().numpy()[0])

state_dot = np.stack(state_dot_list, axis=0)
Jacobian = np.linalg.lstsq(perturb.cpu().numpy(), state_dot, rcond=None)[0].T
U, S, VT = linalg.svd(Jacobian, full_matrices=False)

offline_data = next(gauss_iter).cpu().numpy()
pca = PCA(n_components=config.num_hidden)
pca.fit(offline_data)
true_dirs = pca.components_[:num_dirs]

ali_mat = np.abs(VT[:num_dirs] @ true_dirs.T)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].plot(S, 'o-')
axs[0].set_title("Jacobian singular values")
im = axs[1].imshow(ali_mat, vmin=0, vmax=1, cmap='viridis')
axs[1].set_title("Alignment")
plt.colorbar(im, ax=axs[1])
plt.tight_layout()
plt.show()
