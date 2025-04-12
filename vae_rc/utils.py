import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.stats import norm

def show_reconstructions(model, X, num_images=8):
    with torch.no_grad():
        _, _, _, _, decoded = model.forward(X)

    originals = X.view(-1, 28, 28).cpu().numpy()
    reconstructions = decoded.view(-1, 28, 28).cpu().numpy()

    fig, axs = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
    for i in range(num_images):
        axs[0, i].imshow(originals[i], cmap="gray")
        axs[0, i].axis("off")
        axs[0, i].set_title("Original")

        axs[1, i].imshow(reconstructions[i], cmap="gray")
        axs[1, i].axis("off")
        axs[1, i].set_title("Reconstruction")

    plt.tight_layout()
    plt.show()

def show_reconstructions_CIFAR(model, X, num_images=8):
    with torch.no_grad():
        _, _, _, _, decoded = model.forward(X.to(model.config.device))

    originals = X.permute(0, 2, 3, 1).cpu().numpy()
    reconstructions = decoded.permute(0, 2, 3, 1).cpu().numpy()

    fig, axs = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
    for i in range(num_images):
        axs[0, i].imshow(originals[i])
        axs[0, i].axis("off")
        axs[0, i].set_title("Original")

        axs[1, i].imshow(reconstructions[i])
        axs[1, i].axis("off")
        axs[1, i].set_title("Reconstruction")

    plt.tight_layout()
    plt.show()

def batch_generate(dataset, batch_size, ff_data, mode, config):
    if ff_data and config.ff_activate:
        posneg_labels = np.zeros((dataset.X_train.shape[0] + dataset.hybrid_dataset.shape[0]))
        posneg_labels[: dataset.X_train.shape[0]] = 1
        posneg_labels = torch.from_numpy(posneg_labels).to(config.device)

    if mode == "train":
        data = (np.concatenate([dataset.X_train, dataset.hybrid_dataset], axis=0) if ff_data and config.ff_activate else dataset.X_train)
        label = (np.concatenate([dataset.y_train, dataset.y_train], axis=0) if ff_data and config.ff_activate else dataset.y_train)
    if mode == "test":
        data = dataset.X_test
        label = dataset.y_test

    data = torch.from_numpy(data).to(config.device)
    label = torch.from_numpy(label).to(config.device)
    num_samples = data.shape[0]
    indices = torch.randperm(num_samples, device=config.device)
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i : i + batch_size]
        batch_data = data[batch_indices]
        batch_label = label[batch_indices]
        batch_posneg_labels = posneg_labels[batch_indices] if ff_data and config.ff_activate else torch.ones((batch_data.shape[0]))
        yield batch_data, batch_label, batch_posneg_labels


def lorenz_system(state, t):
    x, y, z = state
    a = 10
    b = 28
    c = 8.0 / 3.0
    return [a * (y - x), x * (b - z) - y, x * y - c * z]

def plot_y_pred(Y_pred, n_images=5, config=None):
    Y_pred = Y_pred.detach().cpu()
    n = min(n_images, Y_pred.shape[0])

    if config is not None and hasattr(config, 'data_shape'):
        shape = config.data_shape
    else:
        D = Y_pred.shape[1]
        side = int(np.sqrt(D))
        shape = (side, side)

    if len(shape) == 3:
        C, H, W = shape
    else:
        H, W = shape

    fig, axs = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axs = [axs]

    for i in range(n):
        img = Y_pred[i].view(*shape)
        if len(shape) == 3 and shape[0] in [1, 3]:
            if shape[0] == 1:
                img = img.squeeze(0)
                axs[i].imshow(img, cmap='gray')
            else:
                img = img.permute(1, 2, 0)
                axs[i].imshow(img)
        else:
            axs[i].imshow(img, cmap='gray')
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()

def plot_latent_gaussians(mu, log_var, name="Latent Gaussian PDFs"):
    mu = mu.detach().cpu().numpy()
    log_var = log_var.detach().cpu().numpy()
    dim = mu.shape[1]

    mu_mean = mu.mean(axis=0)
    std_mean = np.exp(0.5 * log_var).mean(axis=0)

    x = np.linspace(-4, 4, 500)
    plt.figure(figsize=(10, 6))
    for i in range(dim):
        pdf = norm.pdf(x, loc=mu_mean[i], scale=std_mean[i])
        plt.plot(x, pdf, label=f'z[{i}]', linewidth=1.2)

    plt.title(name)
    plt.xlabel("z value")
    plt.ylabel("Probability Density")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()


def display_hybrid_images(imageA, imageB, hybrid):
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(imageA, cmap='gray')
    axes[1].imshow(imageB, cmap='gray')
    axes[2].imshow(hybrid, cmap='gray')
    plt.show()
