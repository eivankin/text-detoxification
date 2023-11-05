from pathlib import Path

from src.models.abstract import BaseModel


class IdentityModel(BaseModel):
    """Just returns the original sentence as prediction"""

    def predict(self, inputs: list[str]) -> list[str]:
        return inputs

    def train(self, inputs: list[str], target: list[str]) -> list[str]:
        return inputs

    def load(self, from_path: Path) -> None:
        ...
