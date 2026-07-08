"""Berry-Esseen diagnostic simulation: compare a population where the
diagnostic flags the normal approximation (heavy-tailed log-normal) with
one where it does not (light-tailed log-normal close to a constant)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from lib.diagnostics import generate_diagnostics
from lib.experiment import ExperimentConfig, Population, simulate_experiment
from lib.simulation import SimulationConfig, simulate_studentized_ate

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


def make_studentized_density_figure(studentized: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor="white")
    ax.set_facecolor("white")
    lo = max(float(studentized.min()), -6.0)
    hi = min(float(studentized.max()), 6.0)
    bins = np.linspace(min(lo, -4), max(hi, 4), 80)
    ax.hist(studentized, bins=bins, density=True, color=Z_COLOR, alpha=0.65,
            edgecolor="white", linewidth=0.4)
    t_grid = np.linspace(bins[0], bins[-1], 400)
    ax.plot(t_grid, stats.norm.pdf(t_grid), color=NORMAL_COLOR, lw=2.0,
            linestyle="--", label=r"$N(0,1)$")
    ax.set_xlabel(
        r"$\sqrt{n}(\widehat\psi_n-\psi_n)/\widehat\sigma_n$",
        fontsize=12,
    )
    ax.set_ylabel("density", fontsize=11)
    ax.legend(frameon=False, fontsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


# Failing case: bulk of mass near zero with a handful of extreme outliers.
# This pushes lambda_n high enough that Z_n becomes visibly non-Gaussian
# (bimodal: assignment of the outliers dominates the test statistic).
rng_y_fail = np.random.default_rng(SEED_Y_FAIL)
y_fail = np.concatenate([
    stats.norm.rvs(loc=0.0, scale=1.0, size=N_UNITS - 3, random_state=rng_y_fail),
    np.array([60.0, 60.0, 60.0]),
])
population_fail = Population(y0=y_fail, y1=y_fail)
experiment_fail = simulate_experiment(
    ExperimentConfig(TREATMENT_PROBABILITY, population_fail, np.random.default_rng(SEED_SIM))
)
diagnostics_fail = generate_diagnostics(experiment_fail)
print(f"FAIL: berry_esseen_bound = {diagnostics_fail.berry_esseen_bound:.3f}, max_variance_share = {diagnostics_fail.max_variance_share:.3f}")
studentized_fail = simulate_studentized_ate(
    SimulationConfig(
        population=population_fail,
        treatment_probability=TREATMENT_PROBABILITY,
        ci_level=CI_LEVEL,
        rng=np.random.default_rng(SEED_SIM),
        n_draws=N_DRAWS,
    )
)

# Passing case: light-tailed log-normal (sigma = 0.3)
rng_y_pass = np.random.default_rng(SEED_Y_PASS)
y_pass = stats.lognorm.rvs(s=0.3, scale=1.0, size=N_UNITS, random_state=rng_y_pass)
population_pass = Population(y0=y_pass, y1=y_pass)
experiment_pass = simulate_experiment(
    ExperimentConfig(TREATMENT_PROBABILITY, population_pass, np.random.default_rng(SEED_SIM + 1))
)
diagnostics_pass = generate_diagnostics(experiment_pass)
print(f"PASS: berry_esseen_bound = {diagnostics_pass.berry_esseen_bound:.3f}, max_variance_share = {diagnostics_pass.max_variance_share:.3f}")
studentized_pass = simulate_studentized_ate(
    SimulationConfig(
        population=population_pass,
        treatment_probability=TREATMENT_PROBABILITY,
        ci_level=CI_LEVEL,
        rng=np.random.default_rng(SEED_SIM + 1),
        n_draws=N_DRAWS,
    )
)

out_dir = Path(__file__).parent
make_population_figure(
    y_fail,
    out_path=out_dir / "outlier_potential_outcomes.png",
)
make_studentized_density_figure(
    studentized_fail,
    out_path=out_dir / "outlier_sampling_distribution.png",
)
make_population_figure(
    y_pass,
    out_path=out_dir / "log_normal_potential_outcomes.png",
)
make_studentized_density_figure(
    studentized_pass,
    out_path=out_dir / "log_normal_sampling_distribution.png",
)
