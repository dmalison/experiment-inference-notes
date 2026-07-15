"""Non-coverage of the simple scaled-ATE interval as a function of the lift size.

Sibling of ``lift_skewness_simulation.py``. Here the baseline is held fixed at
ybar0 = 1 (far from zero, so the denominator is never the problem) and the
percentage lift Delta^% is varied. The scaled-ATE interval divides the ATE
interval by |y0_hat|, i.e. its half-width uses sigma_{n,Delta} = sigma_{n,0} +
sigma_{n,1} with a coefficient 1 on sigma_{n,0}; the correct delta-method SD
carries the coefficient |1 + Delta^%|. The scaled interval therefore omits a
term proportional to Delta^% * sigma_{n,0}, so it increasingly under-covers as
the lift grows.

Two-part figure, all at n = 10,000 in the fixed-potential-outcomes framework
(one population per lift, only the Bernoulli(1/2) assignment is re-randomized):

* Top row: sampling distribution of the studentized scaled-ATE statistic
  sqrt(n)(Delta_hat^% - Delta^%) / (sigma_hat_{n,Delta} / |y0_hat|) for
  Delta^% = 0.01 (near nominal) and Delta^% = 1.00 (badly under-covering), with
  the N(0,1) density overlaid. The scaled interval covers iff this statistic
  lies in [-z, z], so a distribution wider than N(0,1) means under-coverage.
* Bottom row: empirical non-coverage of the nominal 95% scaled-ATE interval as a
  function of Delta^%.

Potential outcomes follow y_{i,0} = 1 + eps_i, y_{i,1} = 1 + Delta^% + eps_i,
with eps normalized to exactly mean zero and unit variance. The same normalized
eps is reused for every lift.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy import stats

from lib.simulation import Population, SimulationConfig, simulate_results

SEED = 42
N = 10_000               # units, fixed across the whole figure
P = 0.5                  # treatment probability
YBAR0 = 1.0              # baseline, held fixed (far from zero)
CI_LEVEL = 0.95
Z = stats.norm.ppf(0.975)

# Assignments depend only on this seed and n (not on the lift), so re-seeding it
# for every point makes all bottom-panel dots share the same assignment set.
ASSIGN_SEED = 2024

TOP_LIFTS = [0.01, 1.00]                          # left, right top panels
# Distinct color per top-panel lift, reused to highlight the matching point on
# the non-coverage curve below (orange = the badly under-covering large lift).
LIFT_COLORS = {0.01: "#4c72b0", 1.00: "#dd8452"}

# Bottom-panel x-grid: Delta^% from -1 to 1, forced to include both highlights.
LIFT_GRID = np.unique(np.concatenate([np.linspace(-1.0, 1.0, 21), TOP_LIFTS]))

DRAWS_TOP = 50_000
DRAWS_COV = 20_000

NORMAL_COLOR = "#1b4332"
CURVE_COLOR = "#2ca02c"      # scaled-ATE line (matches the green line in Figure 1)


def normalized_eps(rng: np.random.Generator, n: int) -> np.ndarray:
    eps = rng.standard_normal(n)
    eps -= eps.mean()
    eps /= eps.std()          # exactly mean 0, variance 1
    return eps


def make_population(lift: float, eps: np.ndarray) -> Population:
    y0 = YBAR0 + eps
    y1 = YBAR0 + lift + eps     # constant treatment effect, Delta^% = lift
    return Population(y0=y0, y1=y1)


def main() -> None:
    rng = np.random.default_rng(SEED)
    eps = normalized_eps(rng, N)                    # one normalized draw, reused

    XLIM = (-6.0, 6.0)
    bins = np.linspace(XLIM[0], XLIM[1], 80)
    t_grid = np.linspace(XLIM[0], XLIM[1], 400)

    fig, axd = plt.subplot_mosaic(
        [["top_left", "top_right"], ["bottom", "bottom"]],
        figsize=(9, 8), height_ratios=[1.0, 1.1])

    # --- Top row: studentized scaled-ATE statistic --------------------------
    for key, lift in zip(("top_left", "top_right"), TOP_LIFTS):
        ax = axd[key]
        population = make_population(lift, eps)
        zstat = np.array([
            (result.lift_hat - lift)
            / (result.ate_se_hat / abs(result.control_mean_hat))
            for result in simulate_results(
                SimulationConfig(
                    population=population,
                    treatment_probability=P,
                    ci_level=CI_LEVEL,
                    rng=rng,
                    n_draws=DRAWS_TOP,
                )
            )
        ])

        ax.hist(zstat, bins=bins, density=True, color=LIFT_COLORS[lift],
                alpha=0.75, edgecolor="white", linewidth=0.4)
        ax.plot(t_grid, stats.norm.pdf(t_grid),
                color=NORMAL_COLOR, lw=2, linestyle="--", label="$N(0,1)$")
        ax.set_xlim(XLIM)
        ax.set_title(rf"$\rho_n = {lift * 100:g}\%$", fontsize=12)
        ax.set_xlabel(
            r"$(\widehat{\rho}_n - \rho_n)"
            r"/\widehat{\mathrm{se}}^{\mathrm{scaled}}$",
            fontsize=11)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelsize=8)
    axd["top_left"].set_ylabel("density", fontsize=10)
    axd["top_left"].legend(frameon=False, fontsize=9, loc="upper right")

    # --- Bottom row: non-coverage vs lift -----------------------------------
    # Re-seed the assignment RNG per lift so every dot is evaluated on the
    # identical set of treatment assignments (only the lift changes).
    noncoverage = np.empty(LIFT_GRID.size)
    for k, lift in enumerate(LIFT_GRID):
        population = make_population(lift, eps)
        zstat = np.array([
            (result.lift_hat - lift)
            / (result.ate_se_hat / abs(result.control_mean_hat))
            for result in simulate_results(
                SimulationConfig(
                    population=population,
                    treatment_probability=P,
                    ci_level=CI_LEVEL,
                    rng=np.random.default_rng(ASSIGN_SEED),
                    n_draws=DRAWS_COV,
                )
            )
        ])
        noncoverage[k] = np.mean(np.abs(zstat) > Z)

    axb = axd["bottom"]
    axb.axhline(0.05, color="0.5", lw=1, linestyle="--", label="nominal")
    axb.plot(LIFT_GRID, noncoverage, marker="o", ms=4,
             color=CURVE_COLOR, lw=1.5, zorder=2, label="scaled ATE")
    # Highlight the points matching the two top panels in their panel colors.
    for lift in TOP_LIFTS:
        noncov = noncoverage[np.isclose(LIFT_GRID, lift)][0]
        axb.plot(lift, noncov, marker="o", ms=10, color=LIFT_COLORS[lift],
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

    out = Path(__file__).parent / "lift_scaled_ate_noncoverage.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
