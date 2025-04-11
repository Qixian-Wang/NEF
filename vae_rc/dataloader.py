import os
import torch.utils.data as data
import torch
import numpy as np
from urllib import request
import gzip
from sklearn.preprocessing import OneHotEncoder
from utils import lorenz_system, display_hybrid_images
import config_file
from scipy.integrate import odeint
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import cv2

class DatasetMNIST(data.Dataset):
    def __init__(self, configs):
        self.config = configs
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

        if configs.ff_activate:
            num_samples = X_train.shape[0]
            height, width = 28, 28
            hybrid_dataset = np.zeros_like(X_train, dtype=np.float32)

            for i in range(int(num_samples/3)):
                imgA = X_train[i].reshape(height, width)
                idx = np.random.choice(np.delete(np.arange(num_samples), i))
                imgB = X_train[idx].reshape(height, width)

                # 直接把上半张 imgA、下半张 imgB 拼起来
                top = imgA[: height // 2, :]
                bottom = imgB[height // 2:, :]
                hybrid_img = np.vstack((top, bottom))

                hybrid_dataset[i] = hybrid_img.reshape(-1)
                # display_hybrid_images(imgA, imgB, hybrid_img)

        y_train = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
        y_test = encoder.fit_transform(y_test.reshape(-1, 1)).toarray()

        self.X_train = (X_train[:configs.train_subset_size] if configs.use_subset else X_train).astype(np.float32) / 255.0
        self.X_test = (X_test[:configs.test_subset_size] if configs.use_subset else X_test).astype(np.float32) / 255.0
        if configs.ff_activate:
            self.hybrid_dataset = (hybrid_dataset[:configs.train_subset_size] if configs.use_subset else hybrid_dataset).astype(
                np.float32) / 255.0
        self.y_train = (y_train[:configs.train_subset_size] if configs.use_subset else y_train).astype(np.int64)
        self.y_test = (y_test[:configs.test_subset_size] if configs.use_subset else y_test).astype(np.int64)

        data_property = {"data_length": X_train.shape[1]}
        config_file.configs = config_file.configs.copy(update=data_property)

    def __getitem__(self, index):
        if self.config.ff_activate:
            return (
                self.X_train[index],
                self.y_train[index],
                self.X_test[index],
                self.y_test[index],
                self.hybrid_dataset[index]
            )
        else:
            return (
                self.X_train[index],
                self.y_train[index],
                self.X_test[index],
                self.y_test[index],
            )

    def __len__(self):
        return len(self.X_train)


class DatasetLorenz(data.Dataset):
    def __init__(self, config):
        dt = config.dt
        t = np.linspace(0, (config.sequence_length - 1) * dt, config.sequence_length)

        samples = []
        for i in range(config.num_samples):
            init_state = np.random.uniform(-20, 20, size=3)
            sol = odeint(lorenz_system, init_state, t)
            samples.append(sol.T)

        samples = np.stack(samples, axis=0)   # shape: (num_samples, 3, sequence_length)

        if config.use_subset:
            subset_size = int(config.batch_size * 10)
            samples = samples[:subset_size]

        self.X_train = torch.tensor(samples[:int(0.8 * config.num_samples)], dtype=torch.float32)
        self.X_test = torch.tensor(samples[-int(0.2 * config.num_samples):], dtype=torch.float32)
        self.len = self.X_train.shape[2]

        data_property = {"data_length": self.len}
        config_file.configs = config_file.configs.copy(update=data_property)

    def __getitem__(self, index):
            return self.X_train[index], self.X_test[index]

    def __len__(self):
        return self.len

class DatasetCIFAR:
    def __init__(self, config):
        self.config = config
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        full_train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)

        val_size = int(0.1 * len(full_train_data))  # e.g. 5000
        train_size = len(full_train_data) - val_size
        self.train_data, self.val_data = random_split(full_train_data, [train_size, val_size],
                                                      generator=torch.Generator().manual_seed(42))


        self.test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform)

    def get_loader(self, mode):
        if mode == "train":
            return DataLoader(self.train_data, batch_size=self.config.batch_size, shuffle=True)
        elif mode == "val":
            return DataLoader(self.val_data, batch_size=self.config.batch_size, shuffle=False)


def data_generator(configs):
    dataset = DatasetMNIST(configs)

    print("data loaded")
    return dataset
