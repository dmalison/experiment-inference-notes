"""Non-coverage of the uniformly valid confidence set, recreated on the exact
data-generating processes of Figures 1 and 2.

Two stacked panels, both at n = 10,000 in the fixed-potential-outcomes framework
(one population per point, only the Bernoulli(1/2) assignment re-randomized):

* Top: non-coverage of the uniform set as a function of the percentage lift
  Delta^%, on the DGP of ``lift_scaled_ate_simulation.py`` (baseline ybar0 = 1
  held fixed, lift varied). Recreates the bottom panel of that figure for the
  uniform set instead of the scaled-ATE interval.
* Bottom: non-coverage as a function of the number of standard errors
  sqrt(n) ybar0 / sigma_{n,0} separating the baseline from zero, on the DGP of
  ``lift_skewness_simulation.py`` (lift Delta^% = 10% held fixed, baseline
  varied). Recreates the bottom panel of that figure for the uniform set instead
  of the delta-method interval.

Both panels reuse the originals' populations and assignment RNG streams (same
SEED / ASSIGN_SEED / DRAWS_COV via the imported modules); only the interval rule
changes. The uniform set is C^%_{n,1-alpha} of (eq-final-confidence-set): a sign
pretest T_{n,0} = sqrt(n) y0_hat / sigma_{n,0} at level alpha^sign, then the
Fieller set C^{%,+}/C^{%,-} at level alpha^Fiellers, or the real line when the
sign test is inconclusive. alpha^sign = 0.001 and alpha^Fiellers = 0.049
(matching the coverage table). The additional solid curves show the known-
baseline-sign intervals I^+ and I^- defined in the known baseline sign section,
using the same alpha split.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy import stats

import lift_scaled_ate_simulation as fig1   # Delta^% varied at ybar0 = 1
import lift_skewness_simulation as fig2      # baseline varied at Delta^% = 10%

# Uniform-set tuning, matching lift_coverage_table.py.
ALPHA = 0.05
ALPHA_SIGN = 0.001
ALPHA_FIELLER = ALPHA - ALPHA_SIGN
Z_SIGN = stats.norm.ppf(1 - ALPHA_SIGN / 2)
Z_FIELLER = stats.norm.ppf(1 - ALPHA_FIELLER / 2)

P = fig1.P                       # = 0.5, shared by both DGPs
UNIFORM_COLOR = "#6a51a3"        # purple, distinct from the Fig 1/2 curves
POSITIVE_SIGN_COLOR = "#1b9e77"  # teal line for I^+
NEGATIVE_SIGN_COLOR = "#d95f02"  # orange line for I^-
NOMINAL_COLOR = "0.5"


def noncoverage_rates(rng, y0, y1, true_lift, draws, chunk=4000):
    """Monte-Carlo non-coverage of the uniform and known-sign intervals.

    Both rates are for the true percentage lift `true_lift` (= Delta^%, signed),
    over `draws` Bernoulli(P) assignments of the fixed population (y0, y1).
    """
    n = y0.size
    r = true_lift
    baseline_positive = y0.mean() > 0
    uniform_miss = 0
    known_interval_miss = 0
    total = 0
    done = 0
    while done < draws:
        b = min(chunk, draws - done)
        D = rng.random((b, n)) < P
        n_t = D.sum(1)
        n_c = n - n_t
        ok = (n_t > 1) & (n_c > 1)
        D, n_t, n_c = D[ok], n_t[ok], n_c[ok]
        notD = ~D

        mu0 = (notD * y0).sum(1) / n_c
        mu1 = (D * y1).sum(1) / n_t
        S0 = (notD * (y0[None, :] - mu0[:, None]) ** 2).sum(1) / n_c
        S1 = (D * (y1[None, :] - mu1[:, None]) ** 2).sum(1) / n_t
        Dbar = n_t / n
        dh = mu1 - mu0                              # ATE estimate

        sig0 = np.sqrt(Dbar * S0 / (1 - Dbar))      # sigma_{n,0}
        sig1 = np.sqrt((1 - Dbar) * S1 / Dbar)      # sigma_{n,1}
        se0, se1 = sig0 / np.sqrt(n), sig1 / np.sqrt(n)
        T0 = mu0 / se0                              # sign pretest statistic
        TDelta = dh / (se0 + se1)                   # ATE sign statistic

        # Membership of the true Delta^% (= r) in the selected sign-specific set;
        # the real-line branch always covers.
        cp = np.abs(mu0 * r - dh) <= Z_FIELLER * (abs(1 + r) * se0 + se1)
        cm = np.abs(mu0 * r + dh) <= Z_FIELLER * (abs(1 - r) * se0 + se1)
        uniform_covered = np.where(T0 > Z_SIGN, cp, np.where(T0 < -Z_SIGN, cm, True))

        known_fieller_covered = cp if baseline_positive else cm
        known_interval_covered = np.where(
            np.abs(T0) > Z_FIELLER,
            known_fieller_covered,
            np.where(
                TDelta > Z_SIGN,
                known_fieller_covered & (r >= 0),
                np.where(TDelta < -Z_SIGN, known_fieller_covered & (r <= 0), True),
            ),
        )

        uniform_miss += int((~uniform_covered).sum())
        known_interval_miss += int((~known_interval_covered).sum())
        total += uniform_covered.size
        done += b
    known_interval_noncoverage = known_interval_miss / total
    if baseline_positive:
        return uniform_miss / total, known_interval_noncoverage, np.nan
    return uniform_miss / total, np.nan, known_interval_noncoverage


def lift_panel():
    """Fig 1 DGP: ybar0 = 1 fixed, Delta^% = lift varied."""
    eps = fig1.normalized_eps(np.random.default_rng(fig1.SEED), fig1.N)
    grid = fig1.LIFT_GRID
    nc = np.empty(grid.size)
    nc_sign_known_pos = np.empty(grid.size)
    nc_sign_known_neg = np.empty(grid.size)
    for k, lift in enumerate(grid):
        population = fig1.make_population(lift, eps)      # ybar0 = 1, so Delta^% = lift
        nc[k], nc_sign_known_pos[k], nc_sign_known_neg[k] = noncoverage_rates(
            np.random.default_rng(fig1.ASSIGN_SEED), population.y0, population.y1, lift, fig1.DRAWS_COV)
    return grid, nc, nc_sign_known_pos, nc_sign_known_neg


def baseline_panel():
    """Fig 2 DGP: Delta^% = 10% fixed, baseline ybar0 varied."""
    eps = fig2.normalized_eps(np.random.default_rng(fig2.SEED), fig2.N)
    bases = fig2.COV_BASELINES
    se_grid = fig2.standard_errors(bases)
    nc = np.empty(bases.size)
    nc_sign_known_pos = np.empty(bases.size)
    nc_sign_known_neg = np.empty(bases.size)
    for k, ybar0 in enumerate(bases):
        y0, y1 = fig2.population(ybar0, eps)
        nc[k], nc_sign_known_pos[k], nc_sign_known_neg[k] = noncoverage_rates(
            np.random.default_rng(fig2.ASSIGN_SEED), y0, y1,
            fig2.true_lift(ybar0), fig2.DRAWS_COV)
    return bases, se_grid, nc, nc_sign_known_pos, nc_sign_known_neg


def main() -> None:
    lift_grid, nc_lift, nc_lift_sign_known_pos, nc_lift_sign_known_neg = lift_panel()
    bases, se_grid, nc_base, nc_base_sign_known_pos, nc_base_sign_known_neg = baseline_panel()

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 8))

    # --- Top: vs lift (Fig 1 DGP) ------------------------------------------
    ax_top.axhline(0.05, color=NOMINAL_COLOR, lw=1, linestyle="--", label="nominal")
    ax_top.plot(lift_grid, nc_lift, marker="o", ms=4, color=UNIFORM_COLOR,
                lw=1.5, zorder=2, label="uniform set")
    for lift in fig1.TOP_LIFTS:                      # highlight Fig 1's two panels
        v = nc_lift[np.isclose(lift_grid, lift)][0]
        ax_top.plot(lift, v, marker="o", ms=10, color=fig1.LIFT_COLORS[lift],
                    markeredgecolor="white", markeredgewidth=1.0, zorder=3)
    ax_top.set_ylim(0.0, 0.1)
    ax_top.set_xlim(-1.03, 1.03)
    ax_top.set_xlabel(r"$\rho_n$", fontsize=11)
    ax_top.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax_top.set_ylabel(r"$95\%$ set non-coverage", fontsize=10)
    ax_top.set_title(r"varying the lift at $\overline{y}_0 = 1$", fontsize=11)
    ax_top.legend(frameon=False, fontsize=9, loc="upper left")

    # --- Bottom: vs baseline separation (Fig 2 DGP) ------------------------
    ax_bot.axhline(0.05, color=NOMINAL_COLOR, lw=1, linestyle="--", label="nominal")
    ax_bot.axvline(0, color="0.8", lw=1, zorder=0)   # denominator singularity
    ax_bot.plot(se_grid, nc_base, marker="o", ms=4, color=UNIFORM_COLOR,
                lw=1.5, zorder=2, label="uniform set")
    ax_bot.plot(se_grid, nc_base_sign_known_pos, marker="o", ms=4,
                color=POSITIVE_SIGN_COLOR, lw=1.5, zorder=3,
                label=r"known positive sign ($I^+$)")
    if np.isfinite(nc_base_sign_known_neg).any():
        ax_bot.plot(se_grid, nc_base_sign_known_neg, marker="o", ms=4,
                    color=NEGATIVE_SIGN_COLOR, lw=1.5, zorder=3,
                    label=r"known negative sign ($I^-$)")
    for ybar0 in fig2.TOP_BASELINES:                 # highlight Fig 2's two panels
        m = np.isclose(bases, ybar0)
        if m.any():
            ax_bot.plot(se_grid[m][0], nc_base[m][0], marker="o", ms=10,
                        color=fig2.BASELINE_COLORS[ybar0], markeredgecolor="white",
                        markeredgewidth=1.0, zorder=3)
    ax_bot.set_ylim(0.0, 0.1)
    ax_bot.set_xlim(-10.5, 10.5)
    ax_bot.set_xticks(range(-10, 11))
    ax_bot.set_xlabel(r"$\sqrt{n}\,\overline{y}_0/\sigma_{n,0}$", fontsize=11)
    ax_bot.set_ylabel(r"$95\%$ set non-coverage", fontsize=10)
    ax_bot.set_title(r"varying the baseline at $\rho_n = 10\%$", fontsize=11)
    ax_bot.legend(frameon=False, fontsize=9, loc="upper left")

    for ax in (ax_top, ax_bot):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    out = Path(__file__).parent / "lift_uniform_noncoverage.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
