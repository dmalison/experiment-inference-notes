"""Illustrate the sign-specific Fieller confidence sets.

The figure plots the left-hand sides of the inequalities defining
C_{1-alpha,n}^{%,+} and C_{1-alpha,n}^{%,-}. Values below zero are inside the
corresponding confidence set.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
from scipy import stats


N = 10_000
CI_LEVEL = 0.95
Z = stats.norm.ppf(1 - (1 - CI_LEVEL) / 2)

# Illustrative estimates. The animation varies only SIGMA0_HAT; the true value
# and point estimate are fixed in percentage-lift units.
Y0_HAT = 1.0
POINT_ESTIMATE = 0.30
TRUE_VALUE = 0.20
DELTA_HAT = POINT_ESTIMATE * Y0_HAT
SIGMA1_HAT = 10.0
BOUNDED_SIGMA0_HAT = 10.0
UNBOUNDED_SIGMA0_HAT = 100.0
CASES = (
    ("bounded", BOUNDED_SIGMA0_HAT, "lift_fieller_confidence_sets_bounded.png"),
    ("unbounded", UNBOUNDED_SIGMA0_HAT, "lift_fieller_confidence_sets_unbounded.png"),
)

BOUNDED_X_LIM = (-3.0, 3.0)
UNBOUNDED_X_LIM = (-3.0, 3.0)
Y_LIM = (-3.0, 3.0)
N_GRID = 2_000

OUTSIDE_COLOR = "0.55"
PLUS_COLOR = "#4c72b0"
MINUS_COLOR = "#dd8452"
ZERO_COLOR = "0.15"
TRUTH_COLOR = "0.2"
POINT_COLOR = "black"


def sigma_delta_hat(sigma0_hat: float) -> float:
    return sigma0_hat + SIGMA1_HAT


def baseline_statistic(sigma0_hat: float) -> float:
    return abs(Y0_HAT) / (sigma0_hat / np.sqrt(N))


def frame_label(sigma0_hat: float) -> str:
    return "bounded" if baseline_statistic(sigma0_hat) > Z else "unbounded"


def fieller_plus(delta: np.ndarray, sigma0_hat: float) -> np.ndarray:
    k = Z**2 / N
    return (
        (Y0_HAT**2 - k * sigma0_hat**2) * delta**2
        - 2 * (Y0_HAT * DELTA_HAT + k * sigma0_hat**2) * delta
        - 2 * k * sigma0_hat * SIGMA1_HAT * (np.abs(1 + delta) - 1)
        + DELTA_HAT**2
        - k * sigma_delta_hat(sigma0_hat)**2
    )


def fieller_minus(delta: np.ndarray, sigma0_hat: float) -> np.ndarray:
    k = Z**2 / N
    return (
        (Y0_HAT**2 - k * sigma0_hat**2) * delta**2
        + 2 * (Y0_HAT * DELTA_HAT + k * sigma0_hat**2) * delta
        - 2 * k * sigma0_hat * SIGMA1_HAT * (np.abs(1 - delta) - 1)
        + DELTA_HAT**2
        - k * sigma_delta_hat(sigma0_hat)**2
    )


def intervals_below_zero(x: np.ndarray, y: np.ndarray) -> list[tuple[float, float]]:
    inside = y <= 0
    intervals: list[tuple[float, float]] = []
    start: float | None = x[0] if inside[0] else None

    for i in range(len(x) - 1):
        if inside[i] == inside[i + 1]:
            continue
        root = x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
        if inside[i]:
            intervals.append((start if start is not None else x[i], root))
            start = None
        else:
            start = root

    if inside[-1]:
        intervals.append((start if start is not None else x[-1], x[-1]))

    return intervals


def plot_panel(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    xlabel: str,
    ci_label: str,
    true_value: float,
    point_estimate: float,
    true_label: str,
    estimate_label: str,
) -> None:
    intervals = intervals_below_zero(x, y)

    ax.axhline(0, color=ZERO_COLOR, lw=1.0, zorder=1)
    ax.axvline(true_value, color=TRUTH_COLOR, lw=1.4, linestyle=":", zorder=1,
               label=true_label)
    ax.plot(x, y, color=OUTSIDE_COLOR, lw=2.0, zorder=2)

    ax.axvline(point_estimate, color=POINT_COLOR, lw=1.4, linestyle="--",
               zorder=6, label=estimate_label)

    for lo, hi in intervals:
        ax.axvspan(lo, hi, color=color, alpha=0.18, zorder=0)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Fieller inequality value", fontsize=10)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    handles, labels = ax.get_legend_handles_labels()
    if intervals:
        handles.append(Patch(facecolor=color, edgecolor="none", alpha=0.18))
        labels.append(ci_label)
    legend = ax.legend(handles, labels, frameon=True, fontsize=9, loc="upper right",
                       bbox_to_anchor=(1.0, 1.0), borderaxespad=0.0,
                       handlelength=2.4, facecolor="white", edgecolor="white",
                       framealpha=1.0)
    legend.set_zorder(10)
    legend.get_frame().set_zorder(10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=9)


def y_limits(x: np.ndarray, sigma0_hats: np.ndarray) -> tuple[float, float]:
    return Y_LIM


def draw_frame(
    axes: np.ndarray,
    x: np.ndarray,
    sigma0_hat: float,
    y_lim: tuple[float, float],
    x_lim: tuple[float, float],
) -> None:
    y_plus = fieller_plus(x, sigma0_hat)
    y_minus = fieller_minus(x, sigma0_hat)

    for ax in axes:
        ax.clear()
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)

    plot_panel(
        axes[0], x, y_plus, PLUS_COLOR,
        r"$\rho_n^+$",
        r"$C_{1-\alpha,n}^{+}$",
        TRUE_VALUE,
        POINT_ESTIMATE,
        "true value",
        r"$\widehat{\rho}_n^+$",
    )
    plot_panel(
        axes[1], x, y_minus, MINUS_COLOR,
        r"$\rho_n^-$",
        r"$C_{1-\alpha,n}^{-}$",
        -TRUE_VALUE,
        -POINT_ESTIMATE,
        "true value",
        r"$\widehat{\rho}_n^-$",
    )
    axes[0].tick_params(labelbottom=True)


def main() -> None:
    x_for_limits = np.linspace(*UNBOUNDED_X_LIM, N_GRID)
    y_lim = y_limits(x_for_limits, np.array([case[1] for case in CASES]))

    for _, sigma0_hat, out_name in CASES:
        x_lim = UNBOUNDED_X_LIM if sigma0_hat == UNBOUNDED_SIGMA0_HAT else BOUNDED_X_LIM
        x = np.linspace(*x_lim, N_GRID)
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.0), sharex=True, sharey=True)
        draw_frame(axes, x, sigma0_hat, y_lim, x_lim)
        fig.tight_layout()
        out = Path(__file__).parent / out_name
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()