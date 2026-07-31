"""Hybrid-interval non-coverage on the Figure 1 weak-denominator DGP.

The population, denominator-strength grid, and Bernoulli assignments are exactly
those used by ``ratio_sharp_null_noncoverage_simulation.py``. The hybrid uses the
delta-method interval when T_n^z is at least the maximum of K_alpha and
sqrt(log(n)), and the Fieller interval otherwise.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

import ratio_delta_sharp_null_simulation as fig2
import ratio_sharp_null_noncoverage_simulation as fig1

K_ALPHA = float(
    fig2.Z_DELTA
    * fig1.Z_FIELLER
    / (fig1.Z_FIELLER - fig2.Z_DELTA)
)
BASE_HYBRID_THRESHOLD = float(np.sqrt(np.log(fig1.N)))
HYBRID_THRESHOLD = max(K_ALPHA, BASE_HYBRID_THRESHOLD)
HYBRID_COLOR = "#2a7f62"
NOMINAL_COLOR = "0.5"


def hybrid_summary(
    summaries: dict[str, np.ndarray],
    denominator_strength: float,
) -> tuple[float, float, float]:
    fieller_misses, fieller_widths, denominator_test = (
        fig1.fieller_draws(summaries, denominator_strength)
    )
    delta_misses, delta_widths = fig2.delta_draws(
        summaries, denominator_strength
    )
    selects_delta = denominator_test >= HYBRID_THRESHOLD
    hybrid_misses = np.where(
        selects_delta, delta_misses, fieller_misses
    )
    hybrid_widths = np.where(
        selects_delta, delta_widths, fieller_widths
    )
    return (
        float(hybrid_misses.mean()),
        float(np.median(hybrid_widths)),
        float(selects_delta.mean()),
    )


def main() -> None:
    y0, z0_residual, y1, z1_residual = fig1.make_population()
    summaries = fig1.assignment_summaries(
        y0, z0_residual, y1, z1_residual, fig1.DRAWS
    )

    simulation_summaries = [
        hybrid_summary(summaries, strength)
        for strength in fig1.DENOMINATOR_STRENGTHS
    ]
    noncoverage = np.array(
        [summary[0] for summary in simulation_summaries]
    )
    median_width = np.array(
        [summary[1] for summary in simulation_summaries]
    )
    delta_selection_rate = np.array(
        [summary[2] for summary in simulation_summaries]
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
        color=HYBRID_COLOR,
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
        "ratio_hybrid_sharp_null_noncoverage.png"
    )
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)

    print(f"wrote {output}")
    print(f"K_alpha: {K_ALPHA:.4f}")
    print(f"sqrt(log(n)): {BASE_HYBRID_THRESHOLD:.4f}")
    print(f"hybrid threshold: {HYBRID_THRESHOLD:.4f}")
    print("strength  noncoverage  median_width  selects_delta")
    for strength, rate, width, selection_rate in zip(
        fig1.DENOMINATOR_STRENGTHS,
        noncoverage,
        median_width,
        delta_selection_rate,
    ):
        print(
            f"{strength:4.1f}      {rate:.4f}  "
            f"{width:g}  {selection_rate:.4f}"
        )


if __name__ == "__main__":
    main()
