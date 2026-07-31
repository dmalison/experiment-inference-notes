"""Fieller non-coverage with one weak denominator.

One fixed population of n = 10,000 units is generated with

    y_{i,1} = rho * zbar_1 + eps_{y,i},
    z_{i,1} = zbar_1 + eps_{z,i},
    y_{i,0} = eps_{y,i},
    z_{i,0} = 1 + eps_{z,i}.

The shared residual sequences have exactly zero means, unit variances, and zero
covariance. Only Bernoulli(1/2) treatment assignment is rerandomized. The
treatment denominator strength is varied while the ratio treatment effect rho
is fixed and the control denominator remains strong.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from scipy import stats

SEED = 42
ASSIGNMENT_SEED = 2024
N = 10_000
P = 0.5
ALPHA = 0.05
DRAWS = 20_000
CHUNK_SIZE = 500
RATIO_TREATMENT_EFFECT = 10.0
CONTROL_Y_MEAN = 0.0
CONTROL_Z_MEAN = 1.0
CONTROL_RESIDUAL_SCALE = 1.0

DENOMINATOR_STRENGTHS = np.arange(1.0, 21.0)
Z_FIELLER = stats.norm.ppf(1.0 - ALPHA / 4.0)

SHARP_NULL_COLOR = "#2f2f2f"
NOMINAL_COLOR = "0.5"
WIDTH_Y_MAX = 60.0
NONCOVERAGE_Y_MAX = 0.20


def normalized_residuals(
    rng: np.random.Generator, n: int
) -> tuple[np.ndarray, np.ndarray]:
    eps_z = rng.standard_normal(n)
    eps_z -= eps_z.mean()
    eps_z /= eps_z.std()

    eps_y = rng.standard_normal(n)
    eps_y -= eps_y.mean()
    eps_y -= np.mean(eps_y * eps_z) * eps_z
    eps_y /= eps_y.std()
    return eps_y, eps_z


def make_population() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rng = np.random.default_rng(SEED)
    eps_y, eps_z = normalized_residuals(rng, N)
    y0 = CONTROL_Y_MEAN + CONTROL_RESIDUAL_SCALE * eps_y
    z0_residual = CONTROL_RESIDUAL_SCALE * eps_z
    y1 = eps_y
    return y0, z0_residual, y1, eps_z


def assignment_summaries(
    y0: np.ndarray,
    eps_z0: np.ndarray,
    y1: np.ndarray,
    eps_z1: np.ndarray,
    draws: int,
) -> dict[str, np.ndarray]:
    control_moments = np.column_stack(
        (y0, y0**2, eps_z0, eps_z0**2, y0 * eps_z0)
    )
    treatment_moments = np.column_stack(
        (y1, y1**2, eps_z1, eps_z1**2, y1 * eps_z1)
    )
    control_population_totals = control_moments.sum(axis=0)
    summaries: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "dbar",
            "y1",
            "y0",
            "z1_residual",
            "z0_residual",
            "syy1",
            "syy0",
            "szz1",
            "szz0",
            "syz1",
            "syz0",
        )
    }

    rng = np.random.default_rng(ASSIGNMENT_SEED)
    completed = 0
    while completed < draws:
        batch_size = min(CHUNK_SIZE, draws - completed)
        treatment = rng.random((batch_size, y0.size)) < P
        treatment_count = treatment.sum(axis=1).astype(float)
        control_count = y0.size - treatment_count
        treatment_totals = np.column_stack(
            [
                np.einsum(
                    "bi,i->b",
                    treatment,
                    treatment_moments[:, column],
                    optimize=False,
                )
                for column in range(treatment_moments.shape[1])
            ]
        )
        assigned_control_totals = np.column_stack(
            [
                np.einsum(
                    "bi,i->b",
                    treatment,
                    control_moments[:, column],
                    optimize=False,
                )
                for column in range(control_moments.shape[1])
            ]
        )
        control_totals = (
            control_population_totals - assigned_control_totals
        )

        treatment_means = treatment_totals / treatment_count[:, None]
        control_means = control_totals / control_count[:, None]

        summaries["dbar"].append(treatment_count / y0.size)
        summaries["y1"].append(treatment_means[:, 0])
        summaries["y0"].append(control_means[:, 0])
        summaries["z1_residual"].append(treatment_means[:, 2])
        summaries["z0_residual"].append(control_means[:, 2])
        summaries["syy1"].append(
            treatment_means[:, 1] - treatment_means[:, 0] ** 2
        )
        summaries["syy0"].append(
            control_means[:, 1] - control_means[:, 0] ** 2
        )
        summaries["szz1"].append(
            treatment_means[:, 3] - treatment_means[:, 2] ** 2
        )
        summaries["szz0"].append(
            control_means[:, 3] - control_means[:, 2] ** 2
        )
        summaries["syz1"].append(
            treatment_means[:, 4]
            - treatment_means[:, 0] * treatment_means[:, 2]
        )
        summaries["syz0"].append(
            control_means[:, 4]
            - control_means[:, 0] * control_means[:, 2]
        )
        completed += batch_size

    return {
        key: np.concatenate(value)
        for key, value in summaries.items()
    }


def fieller_draws(
    summaries: dict[str, np.ndarray],
    denominator_strength: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dbar = summaries["dbar"]
    treatment_weight = (1.0 - dbar) / dbar
    control_weight = dbar / (1.0 - dbar)
    treatment_zbar = denominator_strength / np.sqrt(N)
    treatment_ybar = RATIO_TREATMENT_EFFECT * treatment_zbar

    y1 = treatment_ybar + summaries["y1"]
    y0 = summaries["y0"]
    z1 = treatment_zbar + summaries["z1_residual"]
    z0 = CONTROL_Z_MEAN + summaries["z0_residual"]
    true_ratio_difference = (
        treatment_ybar / treatment_zbar
        - CONTROL_Y_MEAN / CONTROL_Z_MEAN
    )

    sigma_y1_sq = treatment_weight * summaries["syy1"]
    sigma_y0_sq = control_weight * summaries["syy0"]
    sigma_z1_sq = treatment_weight * summaries["szz1"]
    sigma_z0_sq = control_weight * summaries["szz0"]
    sigma_yz1 = treatment_weight * summaries["syz1"]
    sigma_yz0 = control_weight * summaries["syz0"]

    denominator_test = np.minimum(
        np.sqrt(N) * np.abs(z1) / np.sqrt(sigma_z1_sq),
        np.sqrt(N) * np.abs(z0) / np.sqrt(sigma_z0_sq),
    )
    bounded = denominator_test > Z_FIELLER
    misses = np.zeros(dbar.size, dtype=bool)
    widths = np.full(dbar.size, np.inf)

    for mask, y_mean, z_mean, sigma_y_sq, sigma_z_sq, sigma_yz in (
        (
            bounded,
            y1,
            z1,
            sigma_y1_sq,
            sigma_z1_sq,
            sigma_yz1,
        ),
        (
            bounded,
            y0,
            z0,
            sigma_y0_sq,
            sigma_z0_sq,
            sigma_yz0,
        ),
    ):
        coefficient = z_mean**2 - Z_FIELLER**2 * sigma_z_sq / N
        midpoint_numerator = (
            y_mean * z_mean - Z_FIELLER**2 * sigma_yz / N
        )
        constant = y_mean**2 - Z_FIELLER**2 * sigma_y_sq / N
        discriminant = np.maximum(
            midpoint_numerator**2 - coefficient * constant,
            0.0,
        )
        radius = np.sqrt(discriminant)
        lower = np.full(dbar.size, np.nan)
        upper = np.full(dbar.size, np.nan)
        lower[mask] = (
            midpoint_numerator[mask] - radius[mask]
        ) / coefficient[mask]
        upper[mask] = (
            midpoint_numerator[mask] + radius[mask]
        ) / coefficient[mask]

        if y_mean is y1:
            treatment_lower, treatment_upper = lower, upper
        else:
            control_lower, control_upper = lower, upper

    difference_lower = treatment_lower - control_upper
    difference_upper = treatment_upper - control_lower
    misses[bounded] = (
        (true_ratio_difference < difference_lower[bounded])
        | (true_ratio_difference > difference_upper[bounded])
    )
    widths[bounded] = (
        treatment_upper[bounded]
        - treatment_lower[bounded]
        + control_upper[bounded]
        - control_lower[bounded]
    )
    return misses, widths, denominator_test


def fieller_summary(
    summaries: dict[str, np.ndarray],
    denominator_strength: float,
) -> tuple[float, float]:
    misses, widths, _ = fieller_draws(
        summaries, denominator_strength
    )
    return float(misses.mean()), float(np.median(widths))


def main() -> None:
    y0, z0_residual, y1, z1_residual = make_population()
    sharp_null_summaries = assignment_summaries(
        y0, z0_residual, y1, z1_residual, DRAWS
    )
    simulation_summaries = [
        fieller_summary(sharp_null_summaries, strength)
        for strength in DENOMINATOR_STRENGTHS
    ]
    sharp_null_noncoverage = np.array(
        [summary[0] for summary in simulation_summaries]
    )
    median_width = np.array(
        [summary[1] for summary in simulation_summaries]
    )

    fig, ax_noncoverage = plt.subplots(figsize=(8.5, 4.2))

    ax_noncoverage.axhline(
        ALPHA,
        color=NOMINAL_COLOR,
        lw=1.0,
        linestyle="--",
        label="nominal",
    )
    ax_noncoverage.plot(
        DENOMINATOR_STRENGTHS,
        sharp_null_noncoverage,
        color=SHARP_NULL_COLOR,
        marker="o",
        ms=4,
        lw=1.5,
        label="_nolegend_",
    )
    ax_noncoverage.set_xlim(0.5, 20.5)
    ax_noncoverage.set_ylim(0.0, NONCOVERAGE_Y_MAX)
    ax_noncoverage.set_xticks(np.arange(2.0, 21.0, 2.0))
    ax_noncoverage.set_xlabel(
        r"treatment denominator strength "
        r"$\sqrt{n}\,|\overline{z}_{1,n}|/\sigma_{z,1,n}$",
        fontsize=11,
    )
    ax_noncoverage.set_ylabel(
        r"$95\%$ interval non-coverage", fontsize=10
    )
    ax_noncoverage.yaxis.set_major_formatter(
        PercentFormatter(xmax=1.0, decimals=0)
    )
    ax_noncoverage.legend(frameon=False, fontsize=9, loc="upper right")
    for spine in ("top", "right"):
        ax_noncoverage.spines[spine].set_visible(False)
    ax_noncoverage.tick_params(labelsize=8)
    fig.tight_layout()

    output = Path(__file__).with_name(
        "ratio_sharp_null_noncoverage.png"
    )
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {output}")
    print("strength  noncoverage  median_width")
    for strength, sharp_rate, width in zip(
        DENOMINATOR_STRENGTHS,
        sharp_null_noncoverage,
        median_width,
    ):
        print(f"{strength:4.1f}      {sharp_rate:.4f}  {width:g}")


if __name__ == "__main__":
    main()