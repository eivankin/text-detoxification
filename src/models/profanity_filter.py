from pathlib import Path

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from src.models.abstract import BaseModel

nltk.download("wordnet")
nltk.download("punkt")


class ProfanityFilter(BaseModel):
    def __init__(self,  blacklist_path: Path):
        self.blacklist = set()
        self._load_blacklist(blacklist_path)

    def predict_single(self, input_sentence: str) -> str:
        if not self.blacklist:
            raise ValueError("empty blacklist")

        words = word_tokenize(input_sentence)

        lemmatizer = WordNetLemmatizer()

        filtered = []
        for word in words:
            lemma = lemmatizer.lemmatize(word.lower())
            if lemma not in self.blacklist:
                filtered.append(word)

        censored_sentence = " ".join(filtered)

        return censored_sentence

    def _load_blacklist(self, from_path: Path) -> None:
        self.blacklist.clear()
        with open(from_path, "r") as file:
            for line in file:
                self.blacklist.add(line.strip().lower())


if __name__ == "__main__":
    # TODO: cli interface
    pf = ProfanityFilter(Path("./data/interim/bad_words.txt"))
    print(pf.predict_single("this shit fucking sucks"))
