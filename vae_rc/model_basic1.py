from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
import numpy as np

def train_basic_pca(train_dataset, config):
    n_components = config.n_components
    pca = PCA(n_components=n_components)

    train_data_pca = pca.fit_transform(train_dataset.X_train)

    ridge_model = RidgeCV(alphas=config.alphas, fit_intercept=False)
    ridge_model.fit(train_data_pca, train_dataset.y_train)
    matrix_a = ridge_model.coef_

    return pca, matrix_a

def test_basic_pca(model, matrix, test_dataset):
    test_data_pca = model.transform(test_dataset.X_test)
    Y_pred = test_data_pca @ matrix.T
    predicted_labels = np.argmax(Y_pred, axis=1)
    accuracy = np.mean(predicted_labels == test_dataset.y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

