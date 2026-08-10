"""Compare proposed and Neyman studentization under unequal variances."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


N_UNITS = 1_000
TREATMENT_PROBABILITY = 0.5
TREATMENT_SCALE = 3.0
N_DRAWS = 50_000
CHUNK_SIZE = 500
POPULATION_SEED = 8
ASSIGNMENT_SEED = 0

PROPOSED_COLOR = "#ee964b"
NEYMAN_COLOR = "#4c78a8"
NORMAL_COLOR = "#1b4332"


def make_population() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(POPULATION_SEED)
    y0 = rng.standard_normal(N_UNITS)
    y0 -= y0.mean()
    y1 = TREATMENT_SCALE * y0
    return y0, y1


def simulate_studentized_statistics(
    y0: np.ndarray, y1: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(ASSIGNMENT_SEED)
    y0_total = y0.sum()
    y0_sq_total = np.square(y0).sum()
    y1_sq = np.square(y1)
    ate = y1.mean() - y0.mean()
    proposed_statistics: list[np.ndarray] = []
    neyman_statistics: list[np.ndarray] = []

    completed = 0
    while completed < N_DRAWS:
        batch_size = min(CHUNK_SIZE, N_DRAWS - completed)
        treatment = rng.binomial(
            1,
            TREATMENT_PROBABILITY,
            size=(batch_size, N_UNITS),
        ).astype(bool)
        treatment_count = treatment.sum(axis=1).astype(float)
        control_count = N_UNITS - treatment_count
        valid = (treatment_count > 1) & (control_count > 1)
        treatment = treatment[valid]
        treatment_count = treatment_count[valid]
        control_count = control_count[valid]

        treatment_sum = np.einsum(
            "bi,i->b", treatment, y1, optimize=False
        )
        treatment_sq_sum = np.einsum(
            "bi,i->b", treatment, y1_sq, optimize=False
        )
        assigned_control_sum = np.einsum(
            "bi,i->b", treatment, y0, optimize=False
        )
        assigned_control_sq_sum = np.einsum(
            "bi,i->b", treatment, np.square(y0), optimize=False
        )
        control_sum = y0_total - assigned_control_sum
        control_sq_sum = y0_sq_total - assigned_control_sq_sum

        treatment_mean = treatment_sum / treatment_count
        control_mean = control_sum / control_count
        treatment_variance = np.maximum(
            treatment_sq_sum / treatment_count - treatment_mean**2,
            0.0,
        )
        control_variance = np.maximum(
            control_sq_sum / control_count - control_mean**2,
            0.0,
        )
        treatment_share = treatment_count / N_UNITS

        proposed_variance = np.square(
            np.sqrt(treatment_share / (1.0 - treatment_share))
            * np.sqrt(control_variance)
            + np.sqrt((1.0 - treatment_share) / treatment_share)
            * np.sqrt(treatment_variance)
        ) / N_UNITS
        neyman_variance = (
            control_variance / (control_count - 1.0)
            + treatment_variance / (treatment_count - 1.0)
        )
        estimation_error = treatment_mean - control_mean - ate

        proposed_statistics.append(
            estimation_error / np.sqrt(proposed_variance)
        )
        neyman_statistics.append(
            estimation_error / np.sqrt(neyman_variance)
        )
        completed += batch_size

    return (
        np.concatenate(proposed_statistics),
        np.concatenate(neyman_statistics),
    )


def draw_histogram_figure(
    proposed_statistics: np.ndarray,
    neyman_statistics: np.ndarray,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.0, 4.3), dpi=150)
    bins = np.linspace(-4.0, 4.0, 80)
    ax.hist(
        proposed_statistics,
        bins=bins,
        density=True,
        color=PROPOSED_COLOR,
        alpha=0.55,
        edgecolor="white",
        linewidth=0.4,
        label="Cauchy-Schwarz",
    )
    ax.hist(
        neyman_statistics,
        bins=bins,
        density=True,
        histtype="step",
        color=NEYMAN_COLOR,
        linewidth=2.2,
        label="Neyman",
    )
    t_grid = np.linspace(bins[0], bins[-1], 400)
    ax.plot(
        t_grid,
        stats.norm.pdf(t_grid),
        color=NORMAL_COLOR,
        linewidth=2.0,
        linestyle="--",
        label=r"$N(0,1)$",
    )
    ax.set_xlim(-4.0, 4.0)
    ax.set_xlabel("studentized ATE estimator", fontsize=12)
    ax.set_ylabel("density", fontsize=11)
    ax.legend(frameon=False, fontsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    return fig


def main() -> None:
    y0, y1 = make_population()
    proposed_statistics, neyman_statistics = simulate_studentized_statistics(
        y0, y1
    )

    assert np.isclose(y0.mean(), 0.0)
    assert np.isclose(y1.mean(), 0.0)
    assert np.isclose((y1 - y0).mean(), 0.0)
    assert np.isclose(y1.var(), TREATMENT_SCALE**2 * y0.var())
    assert abs(proposed_statistics.mean()) < 0.02
    assert abs(proposed_statistics.std() - 1.0) < 0.03
    assert abs(neyman_statistics.std() - np.sqrt(0.8)) < 0.03

    fig = draw_histogram_figure(proposed_statistics, neyman_statistics)
    output = Path(__file__).parent / "ate_neyman_studentized_histogram.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {output}")
    print(
        "Cauchy--Schwarz studentized standard deviation: "
        f"{proposed_statistics.std():.4f}"
    )
    print(
        "Neyman studentized standard deviation: "
        f"{neyman_statistics.std():.4f}"
    )


if __name__ == "__main__":
    main()