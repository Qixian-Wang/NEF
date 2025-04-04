from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import accuracy_score
import numpy as np

from method_base import MethodBase


class MethodPCA(MethodBase):
    def __init__(self, config):
        self.config = config
        self.pca: PCA
        self.ridge_model: RidgeCV

    def train(self, dataset, seed: int | None = None) -> float:
        self.pca = PCA(n_components=self.config.n_components)
        train_data_pca = self.pca.fit_transform(dataset.X_train)

        self.ridge_model = RidgeCV(alphas=self.config.alphas, fit_intercept=False)
        self.ridge_model.fit(train_data_pca, dataset.y_train)
        y_pred = self.ridge_model.predict(train_data_pca)

        return accuracy_score(
            y_true=np.argmax(dataset.y_train, axis=1), y_pred=np.argmax(y_pred, axis=1)
        )

    def test(self, dataset, seed: int | None = None) -> float:
        X_pca = self.pca.transform(dataset.X_test)
        y_pred = self.ridge_model.predict(X_pca)

        return accuracy_score(y_true=np.argmax(dataset.y_test, axis=1), y_pred=np.argmax(y_pred, axis=1))
