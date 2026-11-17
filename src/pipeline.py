import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return text


class LostFoundMatcher:
    def __init__(self, csv_path="data/found_items.csv"):
        self.df = pd.read_csv(csv_path)
        self.df["clean"] = self.df["description"].apply(clean_text)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.found_vectors = self.vectorizer.fit_transform(self.df["clean"])

    def match(self, lost_text, top_n=3):
        query_vector = self.vectorizer.transform([clean_text(lost_text)])
        scores = cosine_similarity(query_vector, self.found_vectors)[0]
        top_idx = scores.argsort()[::-1][:top_n]
        results = self.df.iloc[top_idx].copy()
        results["score"] = scores[top_idx]
        return results[["id", "description", "location", "score"]]