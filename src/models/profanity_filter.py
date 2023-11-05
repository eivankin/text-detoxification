from pathlib import Path

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from src.models.abstract import BaseModel

nltk.download("wordnet")
nltk.download("punkt")


class ProfanityFilter(BaseModel):
    def __init__(self):
        self.blacklist = set()

    def predict_single(self, input_sentence: str) -> str:
        if not self.blacklist:
            raise ValueError("blacklist is not loaded, call `load` first")

        words = word_tokenize(input_sentence)

        lemmatizer = WordNetLemmatizer()

        filtered = []
        for word in words:
            lemma = lemmatizer.lemmatize(word.lower())
            if lemma not in self.blacklist:
                filtered.append(word)

        censored_sentence = " ".join(filtered)

        return censored_sentence

    def predict(self, inputs: list[str]) -> list[str]:
        return [self.predict_single(s) for s in inputs]

    def train(self, inputs: list[str], target: list[str]) -> list[str]:
        return self.predict(inputs)

    def load(self, from_path: Path) -> None:
        self.blacklist.clear()
        with open(from_path, "r") as file:
            for line in file:
                self.blacklist.add(line.strip().lower())


if __name__ == "__main__":
    pf = ProfanityFilter()
    pf.load(Path("./data/interim/bad_words.txt"))
    print(pf.predict_single("this shit fucking sucks"))
