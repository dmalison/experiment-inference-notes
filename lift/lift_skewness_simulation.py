"""Skewness of the percentage-lift estimator and its effect on delta-method coverage.

Two-part figure, all at a single sample size n = 10,000 in the fixed-potential-
outcomes framework (one population is drawn and only the Bernoulli(1/2) assignment
is re-randomized):

* Top row: the sampling distribution of the standardized delta-method statistic
  sqrt(n)(Delta_hat^% - Delta^%)/sigma for two baselines, y_bar_0 = 0.1 (well
  separated from zero, ~10 standard errors) and y_bar_0 = 0.01 (close to zero,
  ~1 standard error), with the N(0,1) density overlaid.
* Bottom row: the empirical coverage of the nominal 95% delta-method interval as a
  function of y_bar_0, showing coverage collapse as the baseline approaches zero.

Potential outcomes follow y_{i,0} = ybar0 + eps_i, y_{i,1} = 1.1 ybar0 + eps_i,
with eps normalized to exactly mean zero and unit variance, so Delta^% = 10%. The
same normalized eps is reused for every baseline. With sigma_{n,0} = 1 at p = 1/2,
the baseline sits 100 * ybar0 standard errors from zero.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

SEED = 42
N = 10_000               # units, fixed across the whole figure
P = 0.5                  # treatment probability
LIFT = 0.10              # Delta^% = (1 + LIFT) - 1 = 10%
Z = stats.norm.ppf(0.975)

# Assignments depend only on this seed and n (not on ybar0), so re-seeding it for
# every coverage point makes all bottom-panel dots share the same assignment set.
ASSIGN_SEED = 2024

TOP_BASELINES = [0.01, 0.1]                      # left, right top panels
# Distinct color per top-panel baseline, reused to highlight the matching point
# on the coverage curve below.
BASELINE_COLORS = {0.01: "#dd8452", 0.1: "#4c72b0"}

# Standard errors separating the baseline from zero: sqrt(n) ybar0 / sigma_{n,0},
# with sigma_{n,0} = sqrt(p/(1-p)) S = 1 for the normalized DGP at p = 1/2.
SIGMA_N0 = np.sqrt(P / (1 - P))

# Bottom-panel x-grid: signed standard errors -10, ..., -1, 1, ..., 10.
# Both highlighted baselines (0.01 and 0.1) fall on the grid.
COV_BASELINE_SES = np.array([*range(-10, 0), *range(1, 11)], dtype=float)
COV_BASELINES = COV_BASELINE_SES * SIGMA_N0 / np.sqrt(N)


def standard_errors(ybar0):
    return np.sqrt(N) * np.asarray(ybar0) / SIGMA_N0


DRAWS_TOP = 50_000
DRAWS_COV = 20_000

# Population delta-method asymptotic SD factor: Var = fac / (n * ybar0^2).
FAC = (1 - P) / P + (1 + LIFT) ** 2 * P / (1 - P) + 2 * (1 + LIFT)

NORMAL_COLOR = "#1b4332"
CURVE_COLOR = "#2f2f2f"      # delta-method line (near-black; highlight dots sit on it)


def normalized_eps(rng: np.random.Generator, n: int) -> np.ndarray:
    eps = rng.standard_normal(n)
    eps -= eps.mean()
    eps /= eps.std()          # exactly mean 0, variance 1
    return eps


def population(ybar0: float, eps: np.ndarray):
    y0 = ybar0 + eps
    y1 = (1.0 + LIFT) * ybar0 + eps     # constant treatment effect
    return y0, y1


def true_lift(ybar0: float) -> float:
    return np.sign(ybar0) * LIFT


def _draw(rng, y0, y1, draws, chunk=4000):
    """Over `draws` assignments, return dhat and the delta-method (conservative)
    interval half-width.
    """
    n = y0.size
    dhats, hws = [], []
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

        dhat = (mu1 - mu0) / np.abs(mu0)
        sig0 = np.sqrt(Dbar * S0 / (1 - Dbar))      # sigma_{n,0}
        sig1 = np.sqrt((1 - Dbar) * S1 / Dbar)      # sigma_{n,1}
        base = Z / (np.sqrt(n) * np.abs(mu0))
        hw = base * (np.abs(1 + np.sign(mu0) * dhat) * sig0 + sig1)   # delta method

        dhats.append(dhat)
        hws.append(hw)
        done += b
    return np.concatenate(dhats), np.concatenate(hws)


def main() -> None:
    rng = np.random.default_rng(SEED)
    eps = normalized_eps(rng, N)                    # one normalized draw, reused

    XLIM = (-6.0, 6.0)
    bins = np.linspace(XLIM[0], XLIM[1], 80)
    t_grid = np.linspace(XLIM[0], XLIM[1], 400)

    fig, axd = plt.subplot_mosaic(
        [["top_left", "top_right"], ["bottom", "bottom"]],
        figsize=(9, 8), height_ratios=[1.0, 1.1])

    # --- Top row: standardized sampling distributions -----------------------
    sd_dhat = lambda ybar0: np.sqrt(FAC) / (abs(ybar0) * np.sqrt(N))
    for key, ybar0 in zip(("top_left", "top_right"), TOP_BASELINES):
        ax = axd[key]
        y0, y1 = population(ybar0, eps)
        dhat, _ = _draw(rng, y0, y1, DRAWS_TOP)
        z = (dhat - true_lift(ybar0)) / sd_dhat(ybar0)

        ax.hist(z, bins=bins, density=True, color=BASELINE_COLORS[ybar0],
                alpha=0.75, edgecolor="white", linewidth=0.4)
        ax.plot(t_grid, stats.norm.pdf(t_grid),
                color=NORMAL_COLOR, lw=2, linestyle="--", label="$N(0,1)$")
        ax.set_xlim(XLIM)
        se = float(standard_errors(ybar0))
        ax.set_title(
            rf"$\overline{{y}}_0 = {ybar0:g},\ "
            rf"\sqrt{{n}}\,\overline{{y}}_0/\sigma_{{n,0}} = {se:g}$",
            fontsize=11)
        ax.set_xlabel(r"$\sqrt{n}\,(\widehat{\rho}_n - \rho_n)/\sigma$",
                      fontsize=11)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelsize=8)
    axd["top_left"].set_ylabel("density", fontsize=10)
    axd["top_left"].legend(frameon=False, fontsize=9, loc="upper right")

    # --- Bottom row: empirical coverage vs baseline -------------------------
    # Re-seed the assignment RNG per baseline so every dot is evaluated on the
    # identical set of treatment assignments (only the baseline ybar0 changes).
    coverage = np.empty(COV_BASELINES.size)
    for k, ybar0 in enumerate(COV_BASELINES):
        y0, y1 = population(ybar0, eps)
        dhat, hw = _draw(np.random.default_rng(ASSIGN_SEED), y0, y1, DRAWS_COV)
        err = np.abs(true_lift(ybar0) - dhat)
        coverage[k] = np.mean(err <= hw)

    noncoverage = 1.0 - coverage

    axb = axd["bottom"]
    se_grid = standard_errors(COV_BASELINES)
    axb.axhline(0.05, color="0.5", lw=1, linestyle="--", label="nominal")
    axb.axvline(0, color="0.8", lw=1, zorder=0)            # denominator singularity
    axb.plot(se_grid, noncoverage, marker="o", ms=4,
             color=CURVE_COLOR, lw=1.5, zorder=2, label="delta method")
    # Highlight the points matching the two top panels in their panel colors.
    for ybar0 in TOP_BASELINES:
        se = standard_errors(ybar0)
        noncov = noncoverage[np.isclose(COV_BASELINES, ybar0)][0]
        axb.plot(se, noncov, marker="o", ms=10, color=BASELINE_COLORS[ybar0],
                 markeredgecolor="white", markeredgewidth=1.0, zorder=3)
    axb.set_ylim(0.0, 0.2)
    axb.set_xlim(-10.5, 10.5)        # small margin so boundary markers aren't clipped
    axb.set_xticks(range(-10, 11))
    axb.set_xlabel(r"$\sqrt{n}\,\overline{y}_0/\sigma_{n,0}$", fontsize=11)
    axb.set_ylabel(r"$95\%$ CI non-coverage", fontsize=10)
    axb.legend(frameon=False, fontsize=9, loc="upper right")
    for spine in ("top", "right"):
        axb.spines[spine].set_visible(False)
    axb.tick_params(labelsize=8)

    fig.tight_layout()

    out = Path(__file__).parent / "lift_skewness.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
