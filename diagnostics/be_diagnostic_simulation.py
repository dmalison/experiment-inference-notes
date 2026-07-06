"""Berry-Esseen diagnostic simulation: compare a population where the
diagnostic flags the normal approximation (heavy-tailed log-normal) with
one where it does not (light-tailed log-normal close to a constant)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

N = 1_000
B = 50_000
P = 0.5
C_BE = 0.5583
SEED_Y_FAIL = 7
SEED_Y_PASS = 11
SEED_SIM = 42


def diagnostic(y: np.ndarray) -> tuple[float, float]:
    n = y.size
    y_bar = float(y.mean())
    Sn = float(np.sqrt(((y - y_bar) ** 2).mean()))
    lambda_n = float((np.abs(y - y_bar) ** 3).sum() / (n ** 1.5 * Sn ** 3))
    eps_n = (
        C_BE
        * (P ** 2 + (1 - P) ** 2)
        / np.sqrt(P * (1 - P))
        * lambda_n
    )
    return lambda_n, eps_n


def simulate(y: np.ndarray, B: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = y.size
    D = rng.binomial(1, P, size=(B, n))
    n_t = D.sum(axis=1)
    keep = (n_t > 0) & (n_t < n)
    D = D[keep]

    y_bar = float(y.mean())
    Sn = float(np.sqrt(((y - y_bar) ** 2).mean()))
    Z_n = ((D - P) * (y[None, :] - y_bar)).sum(axis=1) / (
        Sn * np.sqrt(n * P * (1 - P))
    )
    return Z_n


def simulate_studentized_ate(y: np.ndarray, B: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = y.size
    D = rng.binomial(1, P, size=(B, n)).astype(float)
    n_t = D.sum(axis=1)
    keep = (n_t > 0) & (n_t < n)
    D = D[keep]
    n_t = n_t[keep]
    n_c = n - n_t

    y_sum = float(y.sum())
    y2_sum = float((y ** 2).sum())
    y_t_sum = (D * y[None, :]).sum(axis=1)
    y2_t_sum = (D * (y[None, :] ** 2)).sum(axis=1)
    y_c_sum = y_sum - y_t_sum
    y2_c_sum = y2_sum - y2_t_sum

    y1_hat = y_t_sum / n_t
    y0_hat = y_c_sum / n_c
    psi_hat = y1_hat - y0_hat

    S1_hat = np.sqrt(np.maximum(y2_t_sum / n_t - y1_hat ** 2, 0.0))
    S0_hat = np.sqrt(np.maximum(y2_c_sum / n_c - y0_hat ** 2, 0.0))
    D_bar = n_t / n
    sigma_hat_sq = D_bar * (1 - D_bar) * (
        S0_hat / (1 - D_bar) + S1_hat / D_bar
    ) ** 2

    return np.sqrt(n) * psi_hat / np.sqrt(sigma_hat_sq)


POP_COLOR = "#4c72b0"
Z_COLOR = "#ee964b"
NORMAL_COLOR = "#1b4332"
TRUTH_COLOR = "#555"


def make_population_figure(y, lambda_n, eps_n, title, out, y_hist_clip=None):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    y_range = (
        (float(np.percentile(y, 0)), float(y_hist_clip))
        if y_hist_clip is not None
        else (float(y.min()), float(y.max()))
    )
    ax.hist(y, bins=60, range=y_range, color=POP_COLOR, alpha=0.75,
            edgecolor="white", linewidth=0.4)
    ax.set_xlabel(r"$y_i$", fontsize=12)
    ax.set_ylabel("count", fontsize=11)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def make_diagnostic_figure(Z_n, lambda_n, eps_n, title, out, diff_ylim=None):
    fig = plt.figure(figsize=(12, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.45, wspace=0.3)
    ax_pdf = fig.add_subplot(gs[:, 0])
    ax_cdf = fig.add_subplot(gs[0, 1])
    ax_diff = fig.add_subplot(gs[1, 1], sharex=ax_cdf)

    # Left panel: empirical PDF of Z_n vs N(0, 1)
    lo = max(float(Z_n.min()), -6.0)
    hi = min(float(Z_n.max()), 6.0)
    bins_z = np.linspace(min(lo, -4), max(hi, 4), 80)
    ax_pdf.hist(Z_n, bins=bins_z, density=True, color=Z_COLOR, alpha=0.65,
                edgecolor="white", linewidth=0.4)
    t_grid = np.linspace(bins_z[0], bins_z[-1], 400)
    ax_pdf.plot(t_grid, stats.norm.pdf(t_grid), color=NORMAL_COLOR, lw=2.0,
                linestyle="--", label=r"$N(0,1)$")
    ax_pdf.set_xlabel(r"$Z_n$", fontsize=12)
    ax_pdf.set_ylabel("density", fontsize=11)
    ax_pdf.legend(frameon=False, fontsize=10)

    # Top-right: empirical CDF of Z_n vs N(0, 1)
    sorted_Z = np.sort(Z_n)
    ecdf = np.arange(1, sorted_Z.size + 1) / sorted_Z.size
    ax_cdf.plot(sorted_Z, ecdf, color=Z_COLOR, lw=2.0, label=r"$\widehat F_n$")
    t_lo = min(float(sorted_Z[0]), -4)
    t_hi = max(float(sorted_Z[-1]), 4)
    t_grid_cdf = np.linspace(t_lo, t_hi, 400)
    ax_cdf.plot(t_grid_cdf, stats.norm.cdf(t_grid_cdf), color=NORMAL_COLOR,
                lw=2.0, linestyle="--", label=r"$N(0,1)$")
    ax_cdf.set_ylabel(r"$\mathbb{P}(Z_n\leq t)$", fontsize=11)
    ax_cdf.legend(frameon=False, fontsize=10)
    ax_cdf.tick_params(axis="x", labelbottom=False)

    # Bottom-right: difference between empirical and N(0, 1) CDFs, with eps_n band
    diff_grid = np.linspace(t_lo, t_hi, 1000)
    ecdf_on_grid = np.searchsorted(sorted_Z, diff_grid, side="right") / sorted_Z.size
    diff = ecdf_on_grid - stats.norm.cdf(diff_grid)
    ax_diff.axhline(eps_n, color=NORMAL_COLOR, lw=1.2, linestyle=":",
                    label=rf"$\pm\widehat\varepsilon_n={eps_n:.3f}$")
    ax_diff.axhline(-eps_n, color=NORMAL_COLOR, lw=1.2, linestyle=":")
    ax_diff.axhline(0, color=TRUTH_COLOR, lw=0.8, linestyle="--")
    ax_diff.plot(diff_grid, diff, color="#c0392b", lw=1.8,
                 label=r"$\widehat F_n - \Phi$")
    ax_diff.set_xlabel(r"$t$", fontsize=12)
    ax_diff.set_ylabel(r"$\widehat F_n(t)-\Phi(t)$", fontsize=11)
    if diff_ylim is not None:
        ax_diff.set_ylim(diff_ylim)
    ax_diff.legend(frameon=False, fontsize=10, loc="upper right")

    for ax in (ax_pdf, ax_cdf, ax_diff):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelsize=9)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def make_studentized_density_figure(stat, title, out):
    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor="white")
    ax.set_facecolor("white")
    lo = max(float(stat.min()), -6.0)
    hi = min(float(stat.max()), 6.0)
    bins = np.linspace(min(lo, -4), max(hi, 4), 80)
    ax.hist(stat, bins=bins, density=True, color=Z_COLOR, alpha=0.65,
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
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


# Failing case: bulk of mass near zero with a handful of extreme outliers.
# This pushes lambda_n high enough that Z_n becomes visibly non-Gaussian
# (bimodal: assignment of the outliers dominates the test statistic).
rng_y_fail = np.random.default_rng(SEED_Y_FAIL)
y_fail = np.concatenate([
    stats.norm.rvs(loc=0.0, scale=1.0, size=N - 3, random_state=rng_y_fail),
    np.array([60.0, 60.0, 60.0]),
])
lambda_fail, eps_fail = diagnostic(y_fail)
print(f"FAIL: lambda_n = {lambda_fail:.2f}, eps_n = {eps_fail:.3f}")
Z_fail = simulate(y_fail, B, SEED_SIM)
studentized_fail = simulate_studentized_ate(y_fail, B, SEED_SIM)

# Passing case: light-tailed log-normal (sigma = 0.3)
rng_y_pass = np.random.default_rng(SEED_Y_PASS)
y_pass = stats.lognorm.rvs(s=0.3, scale=1.0, size=N, random_state=rng_y_pass)
lambda_pass, eps_pass = diagnostic(y_pass)
print(f"PASS: lambda_n = {lambda_pass:.2f}, eps_n = {eps_pass:.3f}")
Z_pass = simulate(y_pass, B, SEED_SIM + 1)
studentized_pass = simulate_studentized_ate(y_pass, B, SEED_SIM + 1)

out_dir = Path(__file__).parent
make_population_figure(
    y_fail, lambda_fail, eps_fail,
    title="Failing population: bulk near 0 plus 3 extreme outliers",
    out=out_dir / "outlier_potential_outcomes.png",
)
# Use the larger eps_n (failing case) to set a common diff-panel y-range,
# rounded up to a friendly number.
DIFF_YLIM = (-0.3, 0.3)
make_studentized_density_figure(
    studentized_fail,
    title="Failing case",
    out=out_dir / "outlier_sampling_distribution.png",
)
make_population_figure(
    y_pass, lambda_pass, eps_pass,
    title="Passing population: log-normal$(\\sigma=0.3)$",
    out=out_dir / "log_normal_potential_outcomes.png",
)
make_studentized_density_figure(
    studentized_pass,
    title="Passing case",
    out=out_dir / "log_normal_sampling_distribution.png",
)
