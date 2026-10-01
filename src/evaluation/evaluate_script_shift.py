from pathlib import Path
from collections import Counter
import re
import math

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "train_clean.csv"
SHIFT_FILE = PROJECT_ROOT / "data" / "script_shift" / "validation_script_shift_full.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TEXT TOKENIZATION
# ============================================================

def tokenize(text):
    text = str(text).lower()
    tokens = re.findall(r"\w+", text, flags=re.UNICODE)

    # Unigrams
    features = tokens.copy()

    # Bigrams
    for i in range(len(tokens) - 1):
        features.append(tokens[i] + " " + tokens[i + 1])

    return features


# ============================================================
# TF-IDF
# ============================================================

class SimpleTfidf:

    def __init__(self, min_df=2, max_features=10000):
        self.min_df = min_df
        self.max_features = max_features
        self.vocabulary = {}
        self.idf = None

    def fit(self, texts):

        document_frequency = Counter()
        tokenized_documents = []

        for text in texts:
            tokens = tokenize(text)
            tokenized_documents.append(tokens)

            unique_tokens = set(tokens)

            for token in unique_tokens:
                document_frequency[token] += 1

        # Keep terms appearing in at least min_df documents
        valid_terms = [
            token
            for token, count in document_frequency.items()
            if count >= self.min_df
        ]

        # Similar idea to max_features:
        # keep highest document-frequency terms
        valid_terms.sort(
            key=lambda token: (-document_frequency[token], token)
        )

        valid_terms = valid_terms[:self.max_features]

        self.vocabulary = {
            token: index
            for index, token in enumerate(valid_terms)
        }

        n_documents = len(texts)
        n_features = len(self.vocabulary)

        # Smooth IDF similar to sklearn:
        # log((1+n)/(1+df)) + 1
        self.idf = np.ones(n_features, dtype=np.float64)

        for token, index in self.vocabulary.items():
            df = document_frequency[token]
            self.idf[index] = math.log(
                (1 + n_documents) / (1 + df)
            ) + 1

        print("Vocabulary size:", n_features)

        return self

    def transform(self, texts):

        n_documents = len(texts)
        n_features = len(self.vocabulary)

        X = np.zeros(
            (n_documents, n_features),
            dtype=np.float32
        )

        for row, text in enumerate(texts):

            tokens = tokenize(text)
            counts = Counter(tokens)

            for token, count in counts.items():

                if token in self.vocabulary:

                    col = self.vocabulary[token]

                    # Term frequency
                    tf = float(count)

                    # TF-IDF
                    X[row, col] = tf * self.idf[col]

            # L2 normalization
            norm = np.linalg.norm(X[row])

            if norm > 0:
                X[row] /= norm

        return X


# ============================================================
# CLASS-WEIGHTED LOGISTIC REGRESSION
# ============================================================

class SimpleLogisticRegression:

    def __init__(
        self,
        learning_rate=0.1,
        epochs=300,
        regularization=0.01,
        random_state=42
    ):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization = regularization
        self.random_state = random_state

        self.weights = None
        self.bias = 0.0

    def sigmoid(self, z):

        z = np.clip(z, -50, 50)

        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):

        rng = np.random.default_rng(self.random_state)

        n_samples, n_features = X.shape

        self.weights = np.zeros(
            n_features,
            dtype=np.float64
        )

        self.bias = 0.0

        # Class weights equivalent in spirit to
        # sklearn class_weight="balanced"
        class_counts = np.bincount(y)

        total = len(y)
        n_classes = 2

        class_weights = np.ones(2)

        for cls in range(2):

            if class_counts[cls] > 0:
                class_weights[cls] = (
                    total /
                    (n_classes * class_counts[cls])
                )

        sample_weights = np.array(
            [class_weights[label] for label in y],
            dtype=np.float64
        )

        for epoch in range(self.epochs):

            # Shuffle training samples
            indices = rng.permutation(n_samples)

            X_shuffled = X[indices]
            y_shuffled = y[indices]
            w_shuffled = sample_weights[indices]

            # Prediction
            logits = (
                X_shuffled @ self.weights
                + self.bias
            )

            probabilities = self.sigmoid(logits)

            errors = probabilities - y_shuffled

            weighted_errors = errors * w_shuffled

            # Gradients
            grad_w = (
                X_shuffled.T @ weighted_errors
            ) / n_samples

            grad_b = np.mean(weighted_errors)

            # L2 regularization
            grad_w += (
                self.regularization *
                self.weights
            )

            # Update
            self.weights -= (
                self.learning_rate * grad_w
            )

            self.bias -= (
                self.learning_rate * grad_b
            )

            if (epoch + 1) % 50 == 0:

                loss = -np.mean(
                    w_shuffled *
                    (
                        y_shuffled *
                        np.log(
                            probabilities + 1e-10
                        )
                        +
                        (1 - y_shuffled) *
                        np.log(
                            1 - probabilities + 1e-10
                        )
                    )
                )

                print(
                    f"Epoch {epoch + 1:3d} "
                    f"| Loss: {loss:.4f}"
                )

        return self

    def predict_proba(self, X):

        logits = (
            X @ self.weights
            + self.bias
        )

        probability_class_1 = self.sigmoid(logits)

        probability_class_0 = (
            1 - probability_class_1
        )

        return np.column_stack(
            [
                probability_class_0,
                probability_class_1
            ]
        )

    def predict(self, X):

        probabilities = self.predict_proba(X)

        return np.argmax(
            probabilities,
            axis=1
        )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    classes = np.unique(
        np.concatenate([y_true, y_pred])
    )

    f1_scores = []
    precision_scores = []
    recall_scores = []

    recalls = []

    for cls in classes:

        true_positive = np.sum(
            (y_true == cls) &
            (y_pred == cls)
        )

        false_positive = np.sum(
            (y_true != cls) &
            (y_pred == cls)
        )

        false_negative = np.sum(
            (y_true == cls) &
            (y_pred != cls)
        )

        precision = (
            true_positive /
            (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0
        )

        recall = (
            true_positive /
            (true_positive + false_negative)
            if true_positive + false_negative > 0
            else 0
        )

        f1 = (
            2 * precision * recall /
            (precision + recall)
            if precision + recall > 0
            else 0
        )

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        recalls.append(recall)

    accuracy = np.mean(
        y_true == y_pred
    )

    balanced_accuracy = np.mean(
        recalls
    )

    return {
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_accuracy,
        "Macro F1": np.mean(f1_scores),
        "Macro Precision": np.mean(precision_scores),
        "Macro Recall": np.mean(recall_scores)
    }


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SCRIPT-CERT SCRIPT-SHIFT BASELINE EVALUATION")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FILE)

shift_df = pd.read_csv(SHIFT_FILE)

print("\nTraining data:")
print(train_df.shape)

print("\nScript-shift validation data:")
print(shift_df.shape)

print("\nColumns:")
print(shift_df.columns.tolist())


# ============================================================
# PREPARE TRAINING DATA
# ============================================================

X_train_text = train_df["text"].astype(str)

labels = train_df["label"].astype(str)

# Explicit binary mapping
label_to_int = {
    "hate": 0,
    "non-hate": 1
}

y_train = labels.map(label_to_int).values


# ============================================================
# TF-IDF
# ============================================================

print("\n" + "=" * 70)
print("BUILDING TF-IDF")
print("=" * 70)

tfidf = SimpleTfidf(
    min_df=2,
    max_features=10000
)

tfidf.fit(X_train_text)

X_train = tfidf.transform(
    X_train_text
)

print(
    "Training matrix:",
    X_train.shape
)


# ============================================================
# TRAIN LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 70)

model = SimpleLogisticRegression(
    learning_rate=0.1,
    epochs=300,
    regularization=0.01,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("\nModel training complete.")


# ============================================================
# SCRIPT-SHIFT CONDITIONS
# ============================================================

conditions = [
    "original",
    "romanized",
    "telugu_script",
    "mixed_script",
    "mild_noise",
    "severe_noise"
]


# ============================================================
# EVALUATE
# ============================================================

results = []

for condition in conditions:

    print("\n" + "=" * 70)
    print("CONDITION:", condition)
    print("=" * 70)

    X_text = shift_df[
        condition
    ].astype(str)

    y_true_text = shift_df[
        "label"
    ].astype(str)

    y_true = y_true_text.map(
        label_to_int
    ).values

    X_test = tfidf.transform(
        X_text
    )

    y_pred = model.predict(
        X_test
    )

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    result = {
        "Condition": condition,
        **metrics
    }

    results.append(result)

    print(
        f"Accuracy:          "
        f"{metrics['Accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{metrics['Balanced Accuracy']:.4f}"
    )

    print(
        f"Macro F1:          "
        f"{metrics['Macro F1']:.4f}"
    )

    print(
        f"Macro Precision:   "
        f"{metrics['Macro Precision']:.4f}"
    )

    print(
        f"Macro Recall:      "
        f"{metrics['Macro Recall']:.4f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

output_file = (
    OUTPUT_DIR /
    "script_shift_baseline_results_numpy.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 70)
print("SCRIPT-SHIFT EVALUATION COMPLETE")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

print("\nSaved:")
print(output_file)