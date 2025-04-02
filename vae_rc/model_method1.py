import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Regression:
    def __init__(self, config, n_classes=10, rls_init=1e2):

        self.n_classes = n_classes
        self.latent_dim = config.num_hidden
        self.device = config.device

        self.W_out = torch.zeros((n_classes, self.latent_dim)).to(self.device)
        self.P_rls = torch.stack([torch.eye(self.latent_dim, device=self.device) * rls_init for _ in range(n_classes)])

    def regression_forward(self, x_seq):
        preds = x_seq @ self.W_out.T

        return preds

    def update_readout_RLS(self, x_seq, label, mode):
        batch_size, dim = x_seq.shape
        if mode == "train":
            for i in range(batch_size):
                x_t = x_seq[i]  # (latent_dim,)
                y_pred = torch.mv(self.W_out, x_t)
                y_true = label[i]  # (n_classes,)

                x_expand = x_t.expand(self.n_classes, -1)  # (n_classes, latent_dim)
                P_x = torch.matmul(self.P_rls, x_t.unsqueeze(1)).squeeze(2)
                denom = 1 + torch.sum(x_expand * P_x, dim=1)
                k_vec = P_x / denom.unsqueeze(1)
                e = y_pred - y_true

                # update W_out: (n_classes, latent_dim)
                self.W_out -= e.unsqueeze(1) * k_vec

                update_term = k_vec.unsqueeze(2) * P_x.unsqueeze(1)  # shape: (n_classes, flat_dim, flat_dim)
                self.P_rls = self.P_rls - update_term


class VAE(nn.Module):
    def __init__(self, config):
        super(VAE, self).__init__()
        self.data_length = config.sequence_length
        self.num_hidden = config.num_hidden
        self.encoder = nn.Sequential(
            nn.Linear(self.data_length, int(self.data_length / 2)),
            nn.ReLU(),
            nn.Linear(int(self.data_length / 2), self.num_hidden),
            nn.ReLU(),
        )

        self.mu = nn.Linear(self.num_hidden, self.num_hidden)
        self.log_var = nn.Linear(self.num_hidden, self.num_hidden)
        self.decoder = nn.Sequential(
            nn.Linear(self.num_hidden, int(self.data_length / 2)),
            nn.ReLU(),
            nn.Linear(int(self.data_length / 2), self.data_length)
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(z)
        return encoded, mu, log_var, z, decoded


class VAE_regression(nn.Module):
    def __init__(self, config):
        super(VAE_regression, self).__init__()
        self.vae = VAE(config)
        self.regression = Regression(config)

    def forward(self, data, label, mode):
        encoded, mu, log_var, z, decoded = self.vae(data)

        with torch.no_grad():
            self.regression.update_readout_RLS(z, label, mode)
        prediction = self.regression.regression_forward(z)

        return encoded, mu, log_var, z, decoded, prediction


def batch_generate(dataset, batch_size, mode):
    if mode == "train":
        data = dataset.X_train
        label = dataset.y_train
    if mode == "test":
        data = dataset.X_test
        label = dataset.y_test

    num_samples = data.shape[0]
    indices = torch.randperm(num_samples)
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_data = data[batch_indices]
        batch_label = label[batch_indices]
        yield batch_data, batch_label


def train_regression(model, train_dataset, config):
    optimizer = optim.Adam(model.parameters(), lr=config.VAE_learning_rate)

    for epoch in range(config.num_epoch):
        total_loss = 0.0
        RC_loss = 0.0
        vae_loss = 0.0
        correct_predict = 0
        recon_loss_total = 0
        kld_loss_total = 0
        total = 0

        for batch_idx, (data, label) in enumerate(batch_generate(train_dataset, config.batch_size, mode="train")):

            data = torch.from_numpy(data).to(config.device)
            label = torch.from_numpy(label).to(config.device)
            encoded, mu, log_var, z, decoded, prediction = model(data, label, mode="train")

            recon_loss = F.mse_loss(decoded, data)
            kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            vae_loss_batch = recon_loss + 2 * kld_loss

            RC_loss_batch = F.cross_entropy(prediction, label.argmax(dim=1))
            _, result = prediction.max(dim=1)

            total_loss_batch = vae_loss_batch + config.rc_loss_weight * RC_loss_batch

            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()

            total_loss += total_loss_batch.item()
            RC_loss += RC_loss_batch
            vae_loss += vae_loss_batch
            recon_loss_total += recon_loss
            kld_loss_total += kld_loss

            correct_predict += (result == label.argmax(dim=1)).sum()
            total += result.shape[0]

        # if epoch % 10 == 0:
        #     show_reconstructions(model, data, label)

        print(f"Epoch {epoch}")
        print(f"accuracy: {correct_predict}/{total}")
        print(f"Total Loss: {total_loss:.4f}")
        print(f"VAE Loss: {vae_loss:.4f}")
        print(f"Reconstruction Loss: {recon_loss_total:.4f}")
        print(f"KLD Loss: {kld_loss_total:.4f}")
        print(f"RC Loss: {RC_loss:.4f}")
    return model

def test_regression(model, test_dataset, config):
    model.eval()
    with torch.no_grad():
        correct_predict = 0
        total_samples = 0
        for batch_idx, (data, label) in enumerate(batch_generate(test_dataset, config.batch_size, mode="test")):
            data = torch.from_numpy(data).to(config.device)
            label = torch.from_numpy(label).to(config.device)
            encoded, mu, log_var, z, decoded, prediction = model(data, label, mode="test")
            _, pred_label = prediction.max(dim=1)

            correct_predict += (pred_label == label).sum().item()
            total_samples += label.shape[0]
        accuracy = correct_predict / total_samples
        print(f"Test Accuracy: {accuracy * 100:.2f}%")


def show_reconstructions(model, X, label, num_images=8, epoch=0):
    model.eval()
    with torch.no_grad():
        _, _, _, _, decoded, _ = model(X, label, epoch)

    originals = X.view(-1, 28, 28).cpu().numpy()
    reconstructions = decoded.view(-1, 28, 28).cpu().numpy()

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
    for i in range(num_images):
        axs[0, i].imshow(originals[i], cmap='gray')
        axs[0, i].axis("off")
        axs[0, i].set_title("Original")

        axs[1, i].imshow(reconstructions[i], cmap='gray')
        axs[1, i].axis("off")
        axs[1, i].set_title("Reconstruction")

    plt.tight_layout()
    plt.show()


