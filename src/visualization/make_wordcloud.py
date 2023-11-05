import re
from collections import Counter

import pandas as pd
import wordcloud
import nltk.corpus

nltk.download("stopwords", quiet=True)
stopwords = set(nltk.corpus.stopwords.words("english"))


def make_wordcloud(column: pd.Series) -> wordcloud.WordCloud:
    joined_text = " ".join(column.tolist())
    lowercase_words = [
        word
        for word in re.findall("[a-z]+", joined_text.lower())
        if word not in stopwords
    ]
    word_counts = Counter(lowercase_words)
    return wordcloud.WordCloud(background_color="white").fit_words(word_counts)
