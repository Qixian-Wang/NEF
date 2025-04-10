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
         label = torch.from_numpy(dataset.y_train).to(self.device)
         prediction = self.rc.rc_train_costom(data, label)

         accuracy_score(
             y_true=np.argmax(label.detach().cpu().numpy(), axis=1), y_pred=np.argmax(prediction.detach().cpu().numpy(), axis=1)
         )

    def validate(self, epoch: int, dataset, seed: int | None = None):
        data = torch.from_numpy(dataset.X_test).to(self.device)
        label = dataset.y_test
        prediction = self.rc.rc_predict_costom(data)
        accuracy = accuracy_score(
            y_true=np.argmax(label, axis=1), y_pred=np.argmax(prediction.detach().cpu().numpy(), axis=1)
        )

        print(f"final acc: {accuracy}")

    def test(self, dataset, seed: int | None = None):
        pass