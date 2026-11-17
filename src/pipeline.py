import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return text


def clean_all(texts):
    return [clean_text(t) for t in texts]


class LostFoundMatcher:
    def __init__(self, csv_path="data/found_items.csv",
                 ngram_range=(1, 1), use_stopwords=True, include_location=False):
        self.df = pd.read_csv(csv_path)

        texts = self.df["description"]
        if include_location:
            texts = self.df["description"] + " " + self.df["location"]

        self.pipe = Pipeline([
            ("clean", FunctionTransformer(clean_all)),
            ("tfidf", TfidfVectorizer(
                stop_words="english" if use_stopwords else None,
                ngram_range=ngram_range)),
        ])
        self.found_vectors = self.pipe.fit_transform(texts)

    def match(self, lost_text, top_n=3):
        query_vector = self.pipe.transform([lost_text])
        scores = cosine_similarity(query_vector, self.found_vectors)[0]
        top_idx = scores.argsort()[::-1][:top_n]
        results = self.df.iloc[top_idx].copy()
        results["score"] = scores[top_idx]
        return results[["id", "description", "location", "score"]]