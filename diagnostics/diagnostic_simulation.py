"""Variance-share diagnostic simulation: compare a population where the
diagnostic flags the normal approximation with one where it does not."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from lib.simulation import Population, SimulationConfig, simulate_results

N_UNITS = 1_000
N_DRAWS = 50_000
TREATMENT_PROBABILITY = 0.5
CI_LEVEL = 0.95
SEED_Y_FAIL = 7
SEED_Y_PASS = 11
SEED_SIM = 42


POP_COLOR = "#4c72b0"
Z_COLOR = "#ee964b"
NORMAL_COLOR = "#1b4332"
TRUTH_COLOR = "#555"


def make_population_figure(y: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    y_range = (float(y.min()), float(y.max()))
    ax.hist(y, bins=60, range=y_range, color=POP_COLOR, alpha=0.75,
            edgecolor="white", linewidth=0.4)
    ax.set_xlabel(r"$y_i$", fontsize=12)
    ax.set_ylabel("count", fontsize=11)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def make_studentized_distribution_figure(
    studentized: np.ndarray,
    out_path: Path,
) -> None:
    fig, (density_ax, cdf_ax) = plt.subplots(
        nrows=2,
        figsize=(7.5, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": (1.15, 1.0)},
        facecolor="white",
    )
    density_ax.set_facecolor("white")
    cdf_ax.set_facecolor("white")
    lo = max(float(studentized.min()), -6.0)
    hi = min(float(studentized.max()), 6.0)
    bins = np.linspace(min(lo, -4), max(hi, 4), 80)
    density_ax.hist(
        studentized,
        bins=bins.tolist(),
        density=True,
        color=Z_COLOR,
        alpha=0.65,
        edgecolor="white",
        linewidth=0.4,
        label="Simulation",
    )
    t_grid = np.linspace(bins[0], bins[-1], 400)
    density_ax.plot(
        t_grid,
        stats.norm.pdf(t_grid),
        color=NORMAL_COLOR,
        lw=2.0,
        linestyle="--",
        label=r"$N(0,1)$",
    )
    density_ax.set_ylabel("density", fontsize=11)
    density_ax.legend(frameon=False, fontsize=10)

    sorted_studentized = np.sort(studentized)
    empirical_cdf = (
        np.searchsorted(sorted_studentized, t_grid, side="right")
        / studentized.size
    )
    cdf_ax.plot(
        t_grid,
        empirical_cdf,
        color=Z_COLOR,
        lw=2.0,
        drawstyle="steps-post",
        label="Empirical CDF",
    )
    cdf_ax.plot(
        t_grid,
        stats.norm.cdf(t_grid),
        color=NORMAL_COLOR,
        lw=2.0,
        linestyle="--",
        label=r"$\Phi$",
    )
    cdf_ax.set_xlabel(
        r"$\sqrt{n}(\widehat\psi_n-\psi_n)/\widehat\sigma_n$",
        fontsize=12,
    )
    cdf_ax.set_ylabel("cumulative probability", fontsize=11)
    cdf_ax.set_ylim(-0.02, 1.02)
    cdf_ax.legend(frameon=False, fontsize=10)
    for ax in (density_ax, cdf_ax):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def max_variance_share(y: np.ndarray) -> float:
    centered = y - y.mean()
    return float(np.max(centered**2) / np.sum(centered**2))


# Failing case: bulk of mass near zero with a handful of extreme outliers.
# Assignment of the outliers dominates the test statistic, producing a
# visibly non-Gaussian sampling distribution.
rng_y_fail = np.random.default_rng(SEED_Y_FAIL)
y_fail = np.concatenate([
    stats.norm.rvs(loc=0.0, scale=1.0, size=N_UNITS - 3, random_state=rng_y_fail),
    np.array([60.0, 60.0, 60.0]),
])
population_fail = Population(y0=y_fail, y1=y_fail)
print(f"FAIL: max_variance_share = {max_variance_share(y_fail):.3f}")
studentized_fail = np.array([
    result.studentized_ate
    for result in simulate_results(
        SimulationConfig(
            population=population_fail,
            treatment_probability=TREATMENT_PROBABILITY,
            ci_level=CI_LEVEL,
            rng=np.random.default_rng(SEED_SIM),
            n_draws=N_DRAWS,
        )
    )
])

# Passing case: light-tailed log-normal (sigma = 0.25)
rng_y_pass = np.random.default_rng(SEED_Y_PASS)
y_pass = np.asarray(
    stats.lognorm.rvs(
        s=0.25,
        scale=1.0,
        size=N_UNITS,
        random_state=rng_y_pass,
    ),
    dtype=float,
)
population_pass = Population(y0=y_pass, y1=y_pass)
print(f"PASS: max_variance_share = {max_variance_share(y_pass):.3f}")
studentized_pass = np.array([
    result.studentized_ate
    for result in simulate_results(
        SimulationConfig(
            population=population_pass,
            treatment_probability=TREATMENT_PROBABILITY,
            ci_level=CI_LEVEL,
            rng=np.random.default_rng(SEED_SIM + 1),
            n_draws=N_DRAWS,
        )
    )
])

out_dir = Path(__file__).parent
make_population_figure(
    y_fail,
    out_path=out_dir / "outlier_potential_outcomes.png",
)
make_studentized_distribution_figure(
    studentized_fail,
    out_path=out_dir / "outlier_sampling_distribution.png",
)
make_population_figure(
    y_pass,
    out_path=out_dir / "log_normal_potential_outcomes.png",
)
make_studentized_distribution_figure(
    studentized_pass,
    out_path=out_dir / "log_normal_sampling_distribution.png",
)
