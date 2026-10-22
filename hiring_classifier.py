"""Predict whether a job applicant gets hired from two interview scores with logistic regression.

Every idea used here is explained with pictures in README.md.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

DATA_PATH = Path(__file__).parent / "data" / "hiring_scores.csv"
FEATURES = ["score_1", "score_2"]


def load_data() -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(DATA_PATH)
    return data[FEATURES].to_numpy(), data["hired"].to_numpy()


def train(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    return LogisticRegression().fit(X, y)


def plot_boundary(model: LogisticRegression, X: np.ndarray, y: np.ndarray) -> None:
    hired = y == 1
    plt.figure()
    plt.scatter(X[hired, 0], X[hired, 1], marker="o", label="Hired")
    plt.scatter(X[~hired, 0], X[~hired, 1], marker="x", label="Rejected")

    # P(hired) = 0.5 exactly where b0 + b1*s1 + b2*s2 = 0, so solve that for s2
    b0 = model.intercept_[0]
    b1, b2 = model.coef_[0]
    s1 = np.array([X[:, 0].min(), X[:, 0].max()])
    plt.plot(s1, -(b0 + b1 * s1) / b2, "k--", label="Decision boundary")

    plt.xlabel("Question 1 score")
    plt.ylabel("Question 2 score")
    plt.title("Hire above the line, reject below it")
    plt.legend()


def main() -> None:
    X, y = load_data()
    print(f"{len(y)} applicants, {X.shape[1]} scores each")

    model = train(X, y)
    errors = int((model.predict(X) != y).sum())
    print(f"Training accuracy: {model.score(X, y):.0%} ({errors} of {len(y)} misclassified)")
    print(f"P(hired | scores 45, 85): {model.predict_proba([[45, 85]])[0, 1]:.3f}")

    plot_boundary(model, X, y)
    plt.show()


if __name__ == "__main__":
    main()
