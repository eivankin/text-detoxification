from pathlib import Path

from src.models.abstract import BaseModel


class IdentityModel(BaseModel):
    """Just returns the original sentence as prediction"""

    def predict_single(self, input_sentence: str) -> str:
        return input_sentence