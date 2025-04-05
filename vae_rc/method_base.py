from abc import ABC, abstractmethod


class MethodBase(ABC):
    @abstractmethod
    def train(self, epoch: int, dataset, seed: int | None = None):
        pass

    @abstractmethod
    def test(self, epoch: int, dataset, seed: int | None = None):
        pass
