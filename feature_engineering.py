"""
feature_engineering.py
Build TF-IDF features for email spam detection.

Outputs:
- X_train, X_test: TF-IDF feature matrices (sparse)
- y_train, y_test: labels
- vectorizer: fitted TF-IDF vectorizer
"""

from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def build_tfidf_features(
    texts,
    labels,
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 20000,
    ngram_range: Tuple[int, int] = (1, 2),
):
    """
    Convert raw text -> TF-IDF features and split train/test.

    texts: list/iterable of strings
    labels: list/iterable of 0/1 or "ham"/"spam" style labels
    """
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.5,
        random_state=random_state,
        stratify=None
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=max_features,
        ngram_range=ngram_range
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    return X_train, X_test, y_train, y_test, vectorizer