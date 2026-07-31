"""Delta-method non-coverage on the Figure 1 weak-denominator DGP.

The population, denominator-strength grid, and Bernoulli assignments are exactly
those used by ``ratio_sharp_null_noncoverage_simulation.py``. Only the confidence
interval changes.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from scipy import stats

import ratio_sharp_null_noncoverage_simulation as fig1

Z_DELTA = stats.norm.ppf(1.0 - fig1.ALPHA / 2.0)
DELTA_COLOR = "#4c72b0"
NOMINAL_COLOR = "0.5"


def delta_draws(
    summaries: dict[str, np.ndarray],
    denominator_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    dbar = summaries["dbar"]
    treatment_weight = (1.0 - dbar) / dbar
    control_weight = dbar / (1.0 - dbar)
    treatment_zbar = denominator_strength / np.sqrt(fig1.N)
    treatment_ybar = fig1.RATIO_TREATMENT_EFFECT * treatment_zbar

    y1 = treatment_ybar + summaries["y1"]
    y0 = summaries["y0"]
    z1 = treatment_zbar + summaries["z1_residual"]
    z0 = fig1.CONTROL_Z_MEAN + summaries["z0_residual"]
    true_ratio_difference = (
        treatment_ybar / treatment_zbar
        - fig1.CONTROL_Y_MEAN / fig1.CONTROL_Z_MEAN
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        q1 = y1 / z1
        q0 = y0 / z0
        ratio_difference = q1 - q0

        residual_variance1 = (
            summaries["syy1"]
            - 2.0 * q1 * summaries["syz1"]
            + q1**2 * summaries["szz1"]
        ) / z1**2
        residual_variance0 = (
            summaries["syy0"]
            - 2.0 * q0 * summaries["syz0"]
            + q0**2 * summaries["szz0"]
        ) / z0**2

        sigma_ell1 = np.sqrt(
            np.maximum(treatment_weight * residual_variance1, 0.0)
        )
        sigma_ell0 = np.sqrt(
            np.maximum(control_weight * residual_variance0, 0.0)
        )
        half_width = (
            Z_DELTA * (sigma_ell1 + sigma_ell0) / np.sqrt(fig1.N)
        )

    misses = (
        np.abs(ratio_difference - true_ratio_difference) > half_width
    )
    widths = 2.0 * half_width
    return misses, widths


def delta_summary(
    summaries: dict[str, np.ndarray],
    denominator_strength: float,
) -> tuple[float, float]:
    misses, widths = delta_draws(summaries, denominator_strength)
    return float(misses.mean()), float(np.median(widths))


def main() -> None:
    y0, z0_residual, y1, z1_residual = fig1.make_population()
    summaries = fig1.assignment_summaries(
        y0, z0_residual, y1, z1_residual, fig1.DRAWS
    )

    simulation_summaries = [
        delta_summary(summaries, strength)
        for strength in fig1.DENOMINATOR_STRENGTHS
    ]
    noncoverage = np.array(
        [summary[0] for summary in simulation_summaries]
    )
    median_width = np.array(
        [summary[1] for summary in simulation_summaries]
    )

    figure, noncoverage_axis = plt.subplots(figsize=(8.5, 4.2))

    noncoverage_axis.axhline(
        fig1.ALPHA,
        color=NOMINAL_COLOR,
        lw=1.0,
        linestyle="--",
        label="nominal",
    )
    noncoverage_axis.plot(
        fig1.DENOMINATOR_STRENGTHS,
        noncoverage,
        color=DELTA_COLOR,
        marker="o",
        ms=4,
        lw=1.5,
        label="_nolegend_",
    )
    noncoverage_axis.set_xlim(0.5, 20.5)
    noncoverage_axis.set_ylim(0.0, fig1.NONCOVERAGE_Y_MAX)
    noncoverage_axis.set_xticks(np.arange(2.0, 21.0, 2.0))
    noncoverage_axis.set_xlabel(
        r"treatment denominator strength "
        r"$\sqrt{n}\,|\overline{z}_{1,n}|/\sigma_{z,1,n}$",
        fontsize=11,
    )
    noncoverage_axis.set_ylabel(
        r"$95\%$ interval non-coverage", fontsize=10
    )
    noncoverage_axis.yaxis.set_major_formatter(
        PercentFormatter(xmax=1.0, decimals=0)
    )
    noncoverage_axis.legend(frameon=False, fontsize=9, loc="upper right")

    for spine in ("top", "right"):
        noncoverage_axis.spines[spine].set_visible(False)
    noncoverage_axis.tick_params(labelsize=8)

    figure.tight_layout()
    output = Path(__file__).with_name(
        "ratio_delta_sharp_null_noncoverage.png"
    )
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)

    print(f"wrote {output}")
    print("strength  delta_noncoverage  delta_width")
    for strength, rate, width in zip(
        fig1.DENOMINATOR_STRENGTHS,
        noncoverage,
        median_width,
    ):
        print(f"{strength:4.1f}      {rate:.4f}  {width:g}")


if __name__ == "__main__":
    main()
