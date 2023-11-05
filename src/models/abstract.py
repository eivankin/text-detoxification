import abc
from pathlib import Path


class BaseModel(abc.ABC):
    @abc.abstractmethod
    def predict(self, inputs: list[str]) -> list[str]:
        ...

    @abc.abstractmethod
    def train(self, inputs: list[str], target: list[str]) -> list[str]:
        ...

    @abc.abstractmethod
    def load(self, from_path: Path) -> None:
        ...
