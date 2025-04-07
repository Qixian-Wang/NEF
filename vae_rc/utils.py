import matplotlib.pyplot as plt
import torch
import numpy as np

def show_reconstructions(model, X, num_images=8):
    with torch.no_grad():
        _, _, _, _, decoded, _ = model.forward(X, mode="test")

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
    if mode == "test":
        data = dataset.X_test

    data = torch.from_numpy(data).to(config.device)

    num_samples = data.shape[0]
    indices = torch.randperm(num_samples)
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i : i + batch_size]
        batch_data = data[batch_indices]
        yield batch_data

def lorenz_system(state, t):
    x, y, z = state
    a = 10
    b = 28
    c = 8.0 / 3.0
    return [a * (y - x), x * (b - z) - y, x * y - c * z]

def plot_y_pred(Y_pred, n_images=5, config=None):
    """
    绘制 Y_pred 的前 n_images 张图像。

    参数:
      Y_pred: torch.Tensor, 形状为 [N, D] 的预测结果，每一行是展平后的图像。
      n_images: int, 要绘制的图像数量（默认 5）。
      config: 可选参数，如果提供，应该包含 data_shape 属性，例如 (C, H, W)。
    """
    # 将 Y_pred 转为 CPU 上的张量并 detach，确保不记录梯度
    Y_pred = Y_pred.detach().cpu()
    n = min(n_images, Y_pred.shape[0])

    # 如果提供 config.data_shape, 使用它恢复图像；否则假设图像为正方形，单通道
    if config is not None and hasattr(config, 'data_shape'):
        shape = config.data_shape  # 例如 (C, H, W)
    else:
        D = Y_pred.shape[1]
        side = int(np.sqrt(D))
        shape = (side, side)

    # 如果是多通道图像（例如 RGB），处理方式略有不同
    if len(shape) == 3:
        C, H, W = shape
    else:
        H, W = shape

    # 创建绘图
    fig, axs = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axs = [axs]  # 保证 axs 可迭代

    for i in range(n):
        img = Y_pred[i].view(*shape)
        # 如果多通道，将数据转换为 (H,W,C) 格式以便 plt.imshow 正确显示
        if len(shape) == 3 and shape[0] in [1, 3]:
            # 若是单通道， squeeze 后以灰度图显示；若 RGB，则 permute 到 (H, W, C)
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