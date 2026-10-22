"""Compare two ways to stretch a yes/no classifier to 10 classes: One-vs-Rest and One-vs-One.

Every idea used here is explained with pictures in README.md.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_strategies() -> dict[str, OneVsRestClassifier | OneVsOneClassifier]:
    base = LogisticRegression(max_iter=1000)
    return {
        "One-vs-Rest": OneVsRestClassifier(base),
        "One-vs-One": OneVsOneClassifier(base),
    }


def evaluate(
    strategy: OneVsRestClassifier | OneVsOneClassifier, X: np.ndarray, y: np.ndarray
) -> tuple[int, float, float]:
    # Scaling inside the pipeline means each fold is scaled using only its own training data
    pipeline = make_pipeline(StandardScaler(), strategy)
    scores = cross_val_score(pipeline, X, y, cv=5)
    n_models = len(strategy.fit(X, y).estimators_)
    return n_models, scores.mean(), scores.std()


def plot_comparison(names: list[str], accuracies: list[float]) -> None:
    plt.figure()
    plt.bar(names, accuracies)
    plt.ylim(0.9, 0.96)
    plt.ylabel("5-fold accuracy")
    plt.title("One-vs-One buys accuracy with more models")


def main() -> None:
    X, y = load_digits(return_X_y=True)
    n_classes = len(set(y))
    print(f"{X.shape[0]} digit images, {X.shape[1]} pixels each, {n_classes} classes")

    names = []
    accuracies = []
    for name, strategy in build_strategies().items():
        n_models, mean, std = evaluate(strategy, X, y)
        print(f"{name}: {n_models} binary models, accuracy {mean:.1%} ± {std:.1%}")
        names.append(name)
        accuracies.append(mean)

    plot_comparison(names, accuracies)
    plt.show()


if __name__ == "__main__":
    main()
