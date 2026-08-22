"""A small, dependency-free Multinomial Naive Bayes text classifier.

The implementation is intentionally kept in plain Python so the project is
easy to understand and can be deployed without installing ML libraries.
"""

from __future__ import annotations

import csv
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


LABELS = ("ham", "spam")
TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    """Convert a message into lower-case word tokens."""
    return TOKEN_PATTERN.findall(text.lower())


def normalise_label(label: str) -> str:
    """Accept common dataset labels and return either ``ham`` or ``spam``."""
    value = label.strip().lower()
    if value in {"spam", "1", "yes"}:
        return "spam"
    if value in {"ham", "not spam", "0", "no"}:
        return "ham"
    raise ValueError(f"Unsupported label: {label!r}. Use ham or spam.")


def load_records(path: str | Path) -> list[tuple[str, str]]:
    """Load a labelled CSV or the native UCI SMS Spam Collection text file.

    CSV files need ``label,message`` columns. The common ``v1,v2`` names are
    also accepted. If no recognised header is present, each tab-separated line
    is treated as ``label<TAB>message``, which matches UCI's download format.
    """
    records: list[tuple[str, str]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as source:
        first_line = source.readline()
        source.seek(0)
        first_fields = next(csv.reader([first_line]), [])
        fields = {name.strip().lower(): name for name in first_fields}
        label_key = fields.get("label") or fields.get("v1") or fields.get("category")
        message_key = fields.get("message") or fields.get("v2") or fields.get("text")
        if label_key and message_key:
            reader = csv.DictReader(source)
            for row_number, row in enumerate(reader, start=2):
                message = (row.get(message_key) or "").strip()
                if not message:
                    continue
                try:
                    records.append((normalise_label(row.get(label_key) or ""), message))
                except ValueError as error:
                    raise ValueError(f"Row {row_number}: {error}") from error
        else:
            for row_number, line in enumerate(source, start=1):
                label, separator, message = line.partition("\t")
                if not separator:
                    raise ValueError(
                        f"Row {row_number}: expected CSV header or label followed by a tab."
                    )
                try:
                    records.append((normalise_label(label), message.strip()))
                except ValueError as error:
                    raise ValueError(f"Row {row_number}: {error}") from error
    if not records:
        raise ValueError("No usable messages were found in the CSV file.")
    return records


def stratified_split(
    records: Iterable[tuple[str, str]], test_fraction: float = 0.2, seed: int = 42
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Make a reproducible train/test split while keeping both labels present."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for record in records:
        grouped[record[0]].append(record)
    if set(grouped) != set(LABELS):
        raise ValueError("The dataset must contain both ham and spam examples.")

    rng = random.Random(seed)
    train: list[tuple[str, str]] = []
    test: list[tuple[str, str]] = []
    for label in LABELS:
        examples = grouped[label][:]
        if len(examples) < 2:
            raise ValueError(f"Need at least two {label} examples.")
        rng.shuffle(examples)
        test_count = max(1, round(len(examples) * test_fraction))
        test_count = min(test_count, len(examples) - 1)
        test.extend(examples[:test_count])
        train.extend(examples[test_count:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


class SpamClassifier:
    """Multinomial Naive Bayes classifier with Laplace smoothing."""

    def __init__(self) -> None:
        self.document_counts: Counter[str] = Counter()
        self.token_counts: dict[str, Counter[str]] = {label: Counter() for label in LABELS}
        self.total_tokens: Counter[str] = Counter()
        self.vocabulary: set[str] = set()

    def fit(self, records: Iterable[tuple[str, str]]) -> "SpamClassifier":
        self.__init__()
        for label, message in records:
            if label not in LABELS:
                raise ValueError(f"Unexpected label: {label}")
            tokens = tokenize(message)
            self.document_counts[label] += 1
            self.token_counts[label].update(tokens)
            self.total_tokens[label] += len(tokens)
            self.vocabulary.update(tokens)
        if not all(self.document_counts[label] for label in LABELS):
            raise ValueError("Training data must contain ham and spam messages.")
        return self

    def predict_proba(self, message: str) -> dict[str, float]:
        if not self.vocabulary:
            raise RuntimeError("The classifier has not been trained.")
        total_documents = sum(self.document_counts.values())
        vocabulary_size = len(self.vocabulary)
        scores: dict[str, float] = {}
        for label in LABELS:
            score = math.log(self.document_counts[label] / total_documents)
            denominator = self.total_tokens[label] + vocabulary_size
            for token in tokenize(message):
                score += math.log((self.token_counts[label][token] + 1) / denominator)
            scores[label] = score
        largest_score = max(scores.values())
        weights = {label: math.exp(score - largest_score) for label, score in scores.items()}
        normaliser = sum(weights.values())
        return {label: weights[label] / normaliser for label in LABELS}

    def predict(self, message: str) -> tuple[str, float]:
        probabilities = self.predict_proba(message)
        label = max(probabilities, key=probabilities.get)
        return label, probabilities[label]

    def to_dict(self) -> dict:
        return {
            "document_counts": dict(self.document_counts),
            "token_counts": {label: dict(self.token_counts[label]) for label in LABELS},
            "total_tokens": dict(self.total_tokens),
            "vocabulary": sorted(self.vocabulary),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SpamClassifier":
        model = cls()
        model.document_counts = Counter(data["document_counts"])
        model.token_counts = {label: Counter(data["token_counts"][label]) for label in LABELS}
        model.total_tokens = Counter(data["total_tokens"])
        model.vocabulary = set(data["vocabulary"])
        return model


def evaluate(model: SpamClassifier, records: Iterable[tuple[str, str]]) -> dict[str, float | int]:
    """Return easy-to-report classification metrics for the spam class."""
    true_positive = false_positive = true_negative = false_negative = 0
    for actual, message in records:
        predicted, _ = model.predict(message)
        if actual == "spam" and predicted == "spam":
            true_positive += 1
        elif actual == "ham" and predicted == "spam":
            false_positive += 1
        elif actual == "ham" and predicted == "ham":
            true_negative += 1
        else:
            false_negative += 1
    total = true_positive + false_positive + true_negative + false_negative
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "test_messages": total,
        "accuracy": (true_positive + true_negative) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }
