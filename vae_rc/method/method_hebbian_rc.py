import torch
import torch.nn as nn
from NEF.vae_rc.utils import batch_generate
from NEF.vae_rc.method.method_base import MethodBase
from NEF.vae_rc.model.model import ReservoirComputing
from NEF.vae_rc.model.Hebbian_model import HebbianLearning
from sklearn.metrics import accuracy_score


class MethodHPCA_RC(MethodBase, nn.Module):
    def __init__(self, config):
        super(MethodHPCA_RC, self).__init__()
        self.config = config

        self.hpca = HebbianLearning(config)
        self.rc = ReservoirComputing(config)
        self.optimizer = torch.optim.Adam(self.hpca.parameters(), lr=1e-4)
        self.to(self.config.device)

    def forward(self, data):
        batch_size = data.size(0)
        data = data.view(batch_size, 1, 28, 28)
        latent = self.hpca(data)
        return latent

    def train(self, epoch: int, dataset, seed: int = None):
        self.hpca.train()
        for batch_idx, (data, _) in enumerate(
            batch_generate(dataset, self.config.batch_size, mode="train", config=self.config)
        ):

            self.forward(data)
            self.optimizer.zero_grad()
            self.hpca.local_update()
            self.optimizer.step()

        print(f"[Train] Epoch {epoch}: pretraining done.")

    def data_collection(self, dataset, mode: str):
        self.hpca.eval()
        latents = []
        labels = []
        with torch.no_grad():
            for data, label in batch_generate(dataset, self.config.batch_size, mode=mode, config=self.config):
                z = self.forward(data)
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

        print(f"[Validate] Epoch {epoch}: RC classification accuracy = {acc*100:.2f}%")
        return acc

    def test(self, dataset, seed: int = None):
        latent_train, label_train = self.data_collection(dataset, mode="train")
        self.rc.rc_train_costom(latent_train.to(self.config.device), label_train.to(self.config.device))

        latent_test, label_test = self.data_collection(dataset, mode="test")
        pred_test = self.rc.rc_predict(latent_test.to(self.config.device))

        y_true = label_test.argmax(dim=1).cpu().numpy()
        y_pred = pred_test.argmax(dim=1).cpu().numpy()
        acc = accuracy_score(y_true, y_pred)

        print(f"[test] Final RC accuracy = {acc*100:.2f}%")
        return acc
