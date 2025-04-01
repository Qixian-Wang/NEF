import os
import torch
import torch.utils.data as data
import numpy as np
from scipy.integrate import odeint
from urllib import request
import gzip
from sklearn.preprocessing import OneHotEncoder

def lorenz_system(state, t):
    x, y, z = state
    a = 10
    b = 28
    c = 8.0 / 3.0
    return [a * (y - x), x * (b - z) - y, x * y - c * z]

class Load_Dataset_Mnist(data.Dataset):
    def __init__(self, configs, training_mode, subset):
        self.training_mode = training_mode

        url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
        filenames = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
                     't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

        save_dir = "mnist_data"
        os.makedirs(save_dir, exist_ok=True)
        data_dic = []
        for filename in filenames:
            file_path = os.path.join(save_dir, filename)
            if not os.path.exists(file_path):
                request.urlretrieve(url + filename, file_path)

            with gzip.open(file_path, 'rb') as f:
                if 'labels' in filename:
                    data_dic.append(np.frombuffer(f.read(), np.uint8, offset=8))
                else:
                    data_dic.append(np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28))

        encoder = OneHotEncoder()
        X_train, y_train, X_test, y_test = data_dic
        y_train = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
        # y_test = encoder.fit_transform(y_test.reshape(-1, 1)).toarray()

        if subset:
            self.X_train = X_train[:5 * configs.subset_size].astype(np.float32) / 255.0
            self.X_test = X_test[:configs.subset_size].astype(np.float32) / 255.0

            self.y_train = y_train[:5 * configs.subset_size].astype(np.int64)
            self.y_test = y_test[:configs.subset_size].astype(np.int64)

        else:
            self.X_train = X_train.astype(np.float32) / 255.0
            self.X_test = X_test.astype(np.float32) / 255.0

            self.y_train = y_train.astype(np.int64)
            self.y_test = y_test.astype(np.int64)

    def __getitem__(self, index):
        if self.training_mode:
            return self.X_train[index], self.y_train[index]
        else:
            return self.X_test[index], self.y_test[index]

    def __len__(self):
        return len(self.X_train) if self.training_mode else len(self.X_test)


def data_generator(configs, training_mode, subset=True):
    train_dataset = Load_Dataset_Mnist(configs, training_mode=True, subset=subset)
    test_dataset = Load_Dataset_Mnist(configs, training_mode=False, subset=subset)

    print("data loaded")
    return train_dataset, test_dataset
