"""Train, evaluate, and save the email spam classifier."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from classifier import SpamClassifier, evaluate, load_records, stratified_split


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the email spam detector.")
    parser.add_argument("--data", default=ROOT / "data" / "sample_messages.csv", type=Path)
    parser.add_argument("--model", default=ROOT / "model" / "model.json", type=Path)
    parser.add_argument("--test-size", default=0.2, type=float, help="Fraction held back for testing.")
    args = parser.parse_args()

    records = load_records(args.data)
    train_records, test_records = stratified_split(records, args.test_size)
    validation_model = SpamClassifier().fit(train_records)
    metrics = evaluate(validation_model, test_records)

    # After measuring honestly on unseen records, train the deployable model on
    # all available labelled examples.
    final_model = SpamClassifier().fit(records)
    payload = {
        "project": "Email Spam Detection",
        "algorithm": "Multinomial Naive Bayes with Laplace smoothing",
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "training_messages": len(records),
        "dataset": str(args.data),
        "evaluation": metrics,
        "model": final_model.to_dict(),
    }
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.model.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Trained on {len(train_records)} messages and tested on {len(test_records)} messages.")
    print("Evaluation results (spam is the positive class):")
    for name in ("accuracy", "precision", "recall", "f1_score"):
        print(f"  {name.replace('_', ' ').title()}: {metrics[name]:.1%}")
    print(f"Saved deployable model to {args.model}")


if __name__ == "__main__":
    main()
