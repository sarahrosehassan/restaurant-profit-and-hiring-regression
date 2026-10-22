"""Generate the figures used in the README.

Run from the repo root:
    .venv/bin/python docs/make_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent / "images"

# Colors (validated categorical slots 1 and 2, plus neutral text/surface tokens)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_MUTED,
    "axes.titlecolor": TEXT,
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "font.size": 11,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
})


def save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / name, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT_DIR / name)


def restaurant_fit_figure(X: np.ndarray, y: np.ndarray) -> None:
    """Restaurant profit vs city population, with the least-squares line through it."""
    model = LinearRegression().fit(X, y)
    line_x = np.array([[X.min()], [X.max()]])

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.scatter(X, y, s=28, color=BLUE, alpha=0.75, edgecolor=SURFACE, linewidth=1, label="restaurant")
    ax.plot(line_x, model.predict(line_x), color=ORANGE, linewidth=2,
            label=f"y = {model.intercept_:.2f} + {model.coef_[0]:.2f}x")
    ax.axhline(0, color=TEXT_MUTED, linewidth=0.8)
    ax.set_xlabel("City population (10k)")
    ax.set_ylabel("Profit ($10k)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Bigger city, bigger profit", loc="left")
    save(fig, "restaurant_fit.png")


def hiring_boundary_figure(X: np.ndarray, y: np.ndarray) -> None:
    """Hired vs rejected applicants, shaded by the side of the 50% boundary they fall on."""
    model = LogisticRegression().fit(X, y)
    grid_x, grid_y = np.meshgrid(np.linspace(28, 102, 300), np.linspace(28, 102, 300))
    probability = model.predict_proba(np.c_[grid_x.ravel(), grid_y.ravel()])[:, 1].reshape(grid_x.shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.contourf(grid_x, grid_y, probability, levels=[0, 0.5, 1], colors=[ORANGE, BLUE], alpha=0.08)
    ax.contour(grid_x, grid_y, probability, levels=[0.5], colors=TEXT_MUTED, linewidths=1.5, linestyles="--")
    ax.scatter(*X[y == 1].T, marker="o", s=34, color=BLUE, edgecolor=SURFACE, linewidth=1, label="hired")
    ax.scatter(*X[y == 0].T, marker="X", s=40, color=ORANGE, edgecolor=SURFACE, linewidth=1, label="rejected")
    ax.set_xlim(28, 102)
    ax.set_ylim(28, 102)
    ax.set_xlabel("Question 1 score")
    ax.set_ylabel("Question 2 score")
    ax.legend(frameon=False, loc="lower left")
    ax.set_title(f"Decision boundary  ·  {model.score(X, y):.0%} training accuracy", loc="left")
    save(fig, "hiring_boundary.png")


def multiclass_tradeoff_figure() -> None:
    """Models trained vs accuracy for One-vs-Rest and One-vs-One on the 10 digits."""
    X, y = load_digits(return_X_y=True)
    base = LogisticRegression(max_iter=1000)
    names = ["One-vs-Rest", "One-vs-One"]
    model_counts: list[int] = []
    accuracies: list[float] = []
    for strategy in [OneVsRestClassifier(base), OneVsOneClassifier(base)]:
        accuracies.append(cross_val_score(make_pipeline(StandardScaler(), strategy), X, y, cv=5).mean())
        model_counts.append(len(strategy.fit(X, y).estimators_))

    fig, (count_ax, acc_ax) = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, values, title, fmt in [
        (count_ax, model_counts, "Binary models trained", "{:.0f}"),
        (acc_ax, accuracies, "5-fold accuracy", "{:.1%}"),
    ]:
        bars = ax.barh(names, values, color=[BLUE, ORANGE], height=0.5)
        ax.bar_label(bars, labels=[fmt.format(v) for v in values], padding=6, color=TEXT)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.25)
        ax.set_title(title, loc="left")
    acc_ax.set_yticklabels([])
    fig.suptitle("10-class digits: OvO buys accuracy with 4.5x the models", x=0.02, ha="left",
                 color=TEXT, fontsize=14, fontweight="bold", y=1.06)
    save(fig, "multiclass_tradeoff.png")


def main() -> None:
    restaurants = pd.read_csv(DATA_DIR / "restaurant_profit.csv")
    applicants = pd.read_csv(DATA_DIR / "hiring_scores.csv")

    restaurant_fit_figure(restaurants[["population_10k"]].to_numpy(), restaurants["profit_10k"].to_numpy())
    hiring_boundary_figure(applicants[["score_1", "score_2"]].to_numpy(), applicants["hired"].to_numpy())
    multiclass_tradeoff_figure()


if __name__ == "__main__":
    main()
