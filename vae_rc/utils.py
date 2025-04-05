import matplotlib.pyplot as plt
import torch


def show_reconstructions(model, X, label, num_images=8):
    with torch.no_grad():
        _, _, _, _, decoded, _ = model.forward(X, label, mode="test")

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


def batch_generate(dataset, batch_size, mode, config):
    if mode == "train":
        data = dataset.X_train
        label = dataset.y_train
    if mode == "test":
        data = dataset.X_test
        label = dataset.y_test

    data = torch.from_numpy(data).to(config.device)
    label = torch.from_numpy(label).to(config.device)

    num_samples = data.shape[0]
    indices = torch.randperm(num_samples)
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i : i + batch_size]
        batch_data = data[batch_indices]
        batch_label = label[batch_indices]
        yield batch_data, batch_label
