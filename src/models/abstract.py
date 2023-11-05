import abc
from pathlib import Path


class BaseModel(abc.ABC):
    def predict(self, inputs: list[str]) -> list[str]:
        return [self.predict_single(sentence) for sentence in inputs]

    @abc.abstractmethod
    def predict_single(self, input_sentence: str) -> str:
        ...
