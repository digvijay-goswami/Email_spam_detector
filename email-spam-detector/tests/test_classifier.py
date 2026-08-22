import unittest
from pathlib import Path

from classifier import SpamClassifier, evaluate, load_records, stratified_split, tokenize


EXAMPLES = [
    ("ham", "Please bring the notes to class tomorrow"),
    ("ham", "Meeting moved to Monday afternoon"),
    ("ham", "Your library book is due next week"),
    ("ham", "Can we talk after lunch"),
    ("spam", "Claim your free cash prize now"),
    ("spam", "Winner alert click now for free bonus"),
    ("spam", "Urgent verify account and collect reward"),
    ("spam", "Free vacation offer ends today"),
]


class ClassifierTests(unittest.TestCase):
    def test_tokenize_lowercases_and_removes_punctuation(self):
        self.assertEqual(tokenize("Hello, WORLD! 123"), ["hello", "world", "123"])

    def test_split_keeps_each_label_in_train_and_test(self):
        train, test = stratified_split(EXAMPLES, test_fraction=0.25)
        self.assertEqual({label for label, _ in train}, {"ham", "spam"})
        self.assertEqual({label for label, _ in test}, {"ham", "spam"})

    def test_spam_message_is_detected(self):
        model = SpamClassifier().fit(EXAMPLES)
        label, confidence = model.predict("You are a winner claim your free prize now")
        self.assertEqual(label, "spam")
        self.assertGreater(confidence, 0.5)

    def test_evaluation_returns_metrics(self):
        model = SpamClassifier().fit(EXAMPLES)
        metrics = evaluate(model, EXAMPLES)
        self.assertEqual(metrics["test_messages"], len(EXAMPLES))
        self.assertGreaterEqual(metrics["accuracy"], 0)

    def test_loads_native_uci_tab_separated_format(self):
        dataset = Path(__file__).parent / "fixtures" / "SMSSpamCollection"
        self.assertEqual(
            load_records(dataset),
            [("ham", "Can we meet tomorrow?"), ("spam", "Claim a free prize now!")],
        )


if __name__ == "__main__":
    unittest.main()
