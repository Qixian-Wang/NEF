import os
import torch.utils.data as data
import numpy as np
from urllib import request
import gzip
from sklearn.preprocessing import OneHotEncoder
import config_file


class DatasetMNIST(data.Dataset):
    def __init__(self, configs):
        url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
        filenames = [
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz",
        ]

        save_dir = "mnist_data"
        os.makedirs(save_dir, exist_ok=True)
        data_dic = []
        for filename in filenames:
            file_path = os.path.join(save_dir, filename)
            if not os.path.exists(file_path):
                request.urlretrieve(url + filename, file_path)

            with gzip.open(file_path, "rb") as f:
                if "labels" in filename:
                    data_dic.append(np.frombuffer(f.read(), np.uint8, offset=8))
                else:
                    data_dic.append(
                        np.frombuffer(f.read(), np.uint8, offset=16).reshape(
                            -1, 28 * 28
                        )
                    )

        encoder = OneHotEncoder()
        X_train, y_train, X_test, y_test = data_dic
        y_train = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
        y_test = encoder.fit_transform(y_test.reshape(-1, 1)).toarray()

        if configs.use_subset:
            self.X_train = (
                X_train[: configs.train_subset_size].astype(np.float32) / 255.0
            )
            self.X_test = X_test[: configs.test_subset_size].astype(np.float32) / 255.0

            self.y_train = y_train[: configs.train_subset_size].astype(np.int64)
            self.y_test = y_test[: configs.test_subset_size].astype(np.int64)

        else:
            self.X_train = X_train.astype(np.float32) / 255.0
            self.X_test = X_test.astype(np.float32) / 255.0

            self.y_train = y_train.astype(np.int64)
            self.y_test = y_test.astype(np.int64)

        data_property = {"data_length": X_train.shape[1]}

        config_file.configs = config_file.configs.copy(update=data_property)

    def __getitem__(self, index):
        return (
            self.X_train[index],
            self.y_train[index],
            self.X_test[index],
            self.y_test[index],
        )

    def __len__(self):
        return len(self.X_train)


def data_generator(configs):
    dataset = DatasetMNIST(configs)

    print("data loaded")
    return dataset
