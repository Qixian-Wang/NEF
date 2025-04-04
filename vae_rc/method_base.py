from abc import ABC, abstractmethod


class MethodBase(ABC):
    @abstractmethod
    def train(self, dataset, seed: int | None = None):
        pass

    @abstractmethod
    def test(self, dataset, seed: int | None = None):
        pass
