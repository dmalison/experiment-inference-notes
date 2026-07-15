"""Figure 1, with the delta-method interval in place of the scaled-ATE interval.

Replicates ``lift_scaled_ate_simulation.py`` -- same fixed potential-outcomes DGP
(baseline ybar0 = 1, n = 10,000, lifts Delta^% from -1 to 1), the same normalized
eps, and the same assignment draws (the module's rng / ASSIGN_SEED are consumed
identically) -- but studentizes and builds the interval with the delta-method
plug-in SD

    sigma_hat = (|1 + sgn(y0_hat) Delta_hat^%| sigma_{n,0} + sigma_{n,1}) / |y0_hat|

from (eq-delta-method-interval) instead of the scaled-ATE SD
sigma_{n,Delta}/|y0_hat|.

* Top row: sampling distribution of the studentized delta-method statistic
  sqrt(n)(Delta_hat^% - Delta^%)/sigma_hat at Delta^% = 1% and 100%. Because the
  baseline is far from zero, sigma_hat carries the missing |1 + Delta^%| factor,
  so the statistic is close to N(0,1) at BOTH lifts -- unlike the scaled-ATE
  statistic, which is over-dispersed at Delta^% = 100%.
* Bottom row: empirical non-coverage of the nominal 95% delta-method interval as
  a function of Delta^%. The interval covers iff the studentized statistic lies in
  [-z, z], so the (well-separated) delta method stays near nominal across the
  whole lift range, where the scaled-ATE interval's non-coverage grew with
  Delta^%.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy import stats

from lib.simulation import SimulationConfig, simulate_summary_stats

import lift_scaled_ate_simulation as fig1

NORMAL_COLOR = "#1b4332"
CURVE_COLOR = fig1.CURVE_COLOR


def _studentized_delta_method(summary, lift):
    d_bar = summary.treatment_count / summary.count
    lift_hat = (summary.treatment_mean - summary.control_mean) / abs(summary.control_mean)
    sigma0 = summary.control_std * np.sqrt(d_bar / (1 - d_bar))
    sigma1 = summary.treatment_std * np.sqrt((1 - d_bar) / d_bar)
    sd_delta = (
        (abs(1 + np.sign(summary.control_mean) * lift_hat) * sigma0 + sigma1)
        / abs(summary.control_mean)
    )
    return np.sqrt(summary.count) * (lift_hat - lift) / sd_delta


def main() -> None:
    # Reproduce Figure 1's rng consumption exactly: one normalized eps, then the
    # top-panel draws in order, so the assignments match that figure's top row.
    rng = np.random.default_rng(fig1.SEED)
    eps = fig1.normalized_eps(rng, fig1.N)

    XLIM = (-6.0, 6.0)
    bins = np.linspace(XLIM[0], XLIM[1], 80)
    t_grid = np.linspace(XLIM[0], XLIM[1], 400)

    fig, axd = plt.subplot_mosaic(
        [["top_left", "top_right"], ["bottom", "bottom"]],
        figsize=(9, 8), height_ratios=[1.0, 1.1])

    # --- Top row: studentized delta-method statistic ------------------------
    for key, lift in zip(("top_left", "top_right"), fig1.TOP_LIFTS):
        ax = axd[key]
        population = fig1.make_population(lift, eps)
        z = np.array([
            _studentized_delta_method(summary, lift)
            for summary in simulate_summary_stats(
                SimulationConfig(
                    population=population,
                    treatment_probability=fig1.P,
                    ci_level=fig1.CI_LEVEL,
                    rng=rng,
                    n_draws=fig1.DRAWS_TOP,
                )
            )
        ])

        ax.hist(z, bins=bins, density=True, color=fig1.LIFT_COLORS[lift],
                alpha=0.75, edgecolor="white", linewidth=0.4)
        ax.plot(t_grid, stats.norm.pdf(t_grid),
                color=NORMAL_COLOR, lw=2, linestyle="--", label="$N(0,1)$")
        ax.set_xlim(XLIM)
        ax.set_title(rf"$\rho_n = {lift * 100:g}\%$", fontsize=12)
        ax.set_xlabel(
            r"$(\widehat{\rho}_n - \rho_n)"
            r"/\widehat{\mathrm{se}}^{\mathrm{DM}}$",
            fontsize=11)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelsize=8)
    axd["top_left"].set_ylabel("density", fontsize=10)
    axd["top_left"].legend(frameon=False, fontsize=9, loc="upper right")

    # --- Bottom row: non-coverage of the delta-method interval vs lift ------
    # Re-seed the assignment RNG per lift so every dot shares the same draws.
    noncoverage = np.empty(fig1.LIFT_GRID.size)
    for k, lift in enumerate(fig1.LIFT_GRID):
        population = fig1.make_population(lift, eps)
        z = np.array([
            _studentized_delta_method(summary, lift)
            for summary in simulate_summary_stats(
                SimulationConfig(
                    population=population,
                    treatment_probability=fig1.P,
                    ci_level=fig1.CI_LEVEL,
                    rng=np.random.default_rng(fig1.ASSIGN_SEED),
                    n_draws=fig1.DRAWS_COV,
                )
            )
        ])
        noncoverage[k] = np.mean(np.abs(z) > fig1.Z)

    axb = axd["bottom"]
    axb.axhline(0.05, color="0.5", lw=1, linestyle="--", label="nominal")
    axb.plot(fig1.LIFT_GRID, noncoverage, marker="o", ms=4,
             color=CURVE_COLOR, lw=1.5, zorder=2, label="delta method")
    # Highlight the points matching the two top panels in their panel colors.
    for lift in fig1.TOP_LIFTS:
        noncov = noncoverage[np.isclose(fig1.LIFT_GRID, lift)][0]
        axb.plot(lift, noncov, marker="o", ms=10, color=fig1.LIFT_COLORS[lift],
                 markeredgecolor="white", markeredgewidth=1.0, zorder=3)
    axb.set_ylim(0.0, 0.2)
    axb.set_xlim(-1.03, 1.03)
    axb.set_xlabel(r"$\rho_n$", fontsize=11)
    axb.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    axb.set_ylabel(r"$95\%$ CI non-coverage", fontsize=10)
    axb.legend(frameon=False, fontsize=9, loc="upper left")
    for spine in ("top", "right"):
        axb.spines[spine].set_visible(False)
    axb.tick_params(labelsize=8)

    fig.tight_layout()

    out = Path(__file__).parent / "lift_delta_studentized.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
