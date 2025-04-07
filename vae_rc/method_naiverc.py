import torch
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from method_base import MethodBase
from model import ReservoirComputing
from utils import plot_y_pred

class NaiveRC(MethodBase):
    def __init__(self, config):
        data_property = {"num_hidden": config.data_length}
        config = config.copy(update=data_property)
        self.config = config
        self.rc = ReservoirComputing(config)
        self.device = config.device

    def train(self, epoch, dataset, seed: int | None = None):
        data = torch.from_numpy(dataset.X_train).to(self.device)
        rc_reconstruction = self.rc.rc_train_costom(data, data)

        return F.mse_loss(rc_reconstruction, data) * dataset.X_train.shape[0]

    def test(self, epoch, dataset, seed: int | None = None):
        data = torch.from_numpy(dataset.X_test).to(self.device)
        rc_reconstruction = self.rc.rc_predict_costom(data)
        plot_y_pred(rc_reconstruction, n_images=5, config=self.config)

        return F.mse_loss(rc_reconstruction, data) * dataset.X_test.shape[0]
