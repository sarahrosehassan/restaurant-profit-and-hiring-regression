"""Predict a restaurant's profit from the population of its city with linear regression.

Every idea used here is explained with pictures in README.md.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

DATA_PATH = Path(__file__).parent / "data" / "restaurant_profit.csv"


def load_data() -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(DATA_PATH)
    # scikit-learn expects a 2D table of features, even when there is only one column
    X = data[["population_10k"]].to_numpy()
    y = data["profit_10k"].to_numpy()
    return X, y


def train(X: np.ndarray, y: np.ndarray) -> LinearRegression:
    return LinearRegression().fit(X, y)


def plot_fit(model: LinearRegression, X: np.ndarray, y: np.ndarray) -> None:
    plt.figure()
    plt.scatter(X, y, label="Restaurants")
    plt.plot(X, model.predict(X), "r-", label="Fitted line")
    plt.xlabel("City population (10k)")
    plt.ylabel("Profit ($10k)")
    plt.title("Bigger city, bigger profit")
    plt.legend()


def main() -> None:
    X, y = load_data()
    print(f"{len(y)} restaurants, 1 measurement each")

    model = train(X, y)
    print(f"Fitted line: profit = {model.intercept_:.4f} + {model.coef_[0]:.4f} * population")
    print(f"R^2: {model.score(X, y):.3f}")
    # Both columns are in units of 10k, so 18 means a city of 180,000 people
    print(f"Predicted profit at population 18: {model.predict([[18]])[0]:.4f}")

    plot_fit(model, X, y)
    plt.show()


if __name__ == "__main__":
    main()
