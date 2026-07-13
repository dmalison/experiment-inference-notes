"""Animated GIFs: CI coverage under homogeneous vs heterogeneous treatment effects.

Generates two comparable animations from the same seed:
  * homogeneous_coverage_animation.gif   (y_{i,1} = y_{i,0} + 1)
  * heterogeneous_coverage_animation.gif (y_{i,1} = -y_{i,0} + 1)
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy import stats

from lib.ate import compute_ate
from lib.experiment import (
    ExperimentConfig,
    Population,
    simulate_experiment,
)
from lib.simulation import SimulationConfig, simulate_studentized_ate

N_UNITS = 1000
N_TABLE_ROWS = 20  # units shown in the table (of N_UNITS total)
TREATMENT_PROBABILITY = 0.5
N_EXPERIMENTS = 10
SEED = 3
CI_LEVEL = 0.95

# Shared x-axis range and ticks across the homogeneous/heterogeneous figures
# so the two animations are visually comparable.
CI_XLIM = (-0.4, 2.4)
CI_XTICKS = [0.0, 0.5, 1.0, 1.5, 2.0]

TRUTH_COLOR = "#555"

# Studentized-histogram figure
N_HISTOGRAM_DRAWS = 50_000
HIST_COLOR = "#ee964b"
NORMAL_COLOR = "#1b4332"


def fmt(x: float) -> str:
    """Format with 2 decimals, mapping near-zero to '0.00' (no '-0.00')."""
    return "0.00" if abs(x) < 0.005 else f"{x:.2f}"


def draw_studentized_histogram(ax: plt.Axes, studentized: np.ndarray) -> None:
    """Plot the studentized-estimator histogram against the standard normal on ax."""
    lo = max(float(studentized.min()), -6.0)
    hi = min(float(studentized.max()), 6.0)
    bins = np.linspace(min(lo, -4.0), max(hi, 4.0), 80)
    ax.hist(studentized, bins=bins, density=True, color=HIST_COLOR, alpha=0.65,
            edgecolor="white", linewidth=0.4)
    t_grid = np.linspace(bins[0], bins[-1], 400)
    ax.plot(t_grid, stats.norm.pdf(t_grid), color=NORMAL_COLOR, lw=2.0,
            linestyle="--", label=r"$N(0,1)$")
    ax.set_xlabel(r"$\sqrt{n}(\widehat{\psi}_n - \psi_n) / \widehat{\sigma}_n$", fontsize=12)
    ax.set_ylabel("density", fontsize=11)
    ax.legend(frameon=False, fontsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=10)


def make_coverage_animation(
    y1_from_y0: Callable[[np.ndarray], np.ndarray], out_name: str
) -> None:
    rng = np.random.default_rng(SEED)

    # ---------- generate potential outcomes ----------
    y0_raw = rng.normal(0, 1, size=N_UNITS)
    y0 = y0_raw - y0_raw.mean()
    y1 = y1_from_y0(y0)

    y0_bar = y0.mean()
    y1_bar = y1.mean()
    population = Population(y0=y0, y1=y1)
    ate = population.ate

    # ---------- generate experiment results ----------
    config = ExperimentConfig(TREATMENT_PROBABILITY, population, rng)
    assignments, y0_bar_hats, y1_bar_hats, psi_hats, cis, covered_flags = [], [], [], [], [], []
    for _ in range(N_EXPERIMENTS):
        experiment = simulate_experiment(config)
        while experiment.d.sum() in (0, N_UNITS):
            experiment = simulate_experiment(config)
        assignments.append(experiment.d)
        result = compute_ate(experiment, CI_LEVEL)
        y0_bar_hats.append(result.y0_bar_hat)
        y1_bar_hats.append(result.y1_bar_hat)
        psi_hats.append(result.ate_hat)
        cis.append((result.ci_lo, result.ci_hi))
        covered_flags.append(result.ci_lo <= ate <= result.ci_hi)

    # ---------- studentized sampling distribution (bottom panel) ----------
    studentized = simulate_studentized_ate(
        SimulationConfig(
            population=population,
            treatment_probability=TREATMENT_PROBABILITY,
            ci_level=CI_LEVEL,
            rng=np.random.default_rng(0),
            n_draws=N_HISTOGRAM_DRAWS,
        )
    )

    # ---------- plot setup ----------
    # Table panel geometry
    ROW_OFFSET = 0.4
    ROW_SPACING = 1.3
    ELLIPSIS_Y = (N_TABLE_ROWS - 1) * ROW_SPACING + ROW_OFFSET + 0.95
    BOTTOM_LINE_Y = (N_TABLE_ROWS - 1) * ROW_SPACING + ROW_OFFSET + 1.7
    TABLE_DATA_RANGE = BOTTOM_LINE_Y + 10.0  # 7 below bottom line, 3 above header
    FIG_HEIGHT = 9.5 * TABLE_DATA_RANGE / 40.0  # 40 = original data range
    HIST_HEIGHT = 4.0  # full-width histogram strip below the table and CI panels

    fig = plt.figure(figsize=(13, FIG_HEIGHT + HIST_HEIGHT))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.4, 1.3],
        height_ratios=[FIG_HEIGHT, HIST_HEIGHT],
        hspace=0.12,
        wspace=0.08,
    )
    ax_table = fig.add_subplot(gs[0, 0])
    gs_ci = gs[0, 1].subgridspec(2, 1, height_ratios=[3, 1])
    ax_ci = fig.add_subplot(gs_ci[0, 0])
    ax_hist = fig.add_subplot(gs[1, :])
    fig.subplots_adjust(top=0.98, bottom=0.05)

    # table panel: 5 data columns at x = 0.5, 1.5, 2.5, 3.5, 4.5; footer ψ column at x = 4.3
    ax_table.set_xlim(-0.3, 5.5)
    ax_table.set_ylim(BOTTOM_LINE_Y + 7.0, -3.0)
    ax_table.axis("off")

    # Headers — potential outcomes first, then assignment / realized outcome
    ax_table.text(0.5, -2.0, "i", ha="center", fontsize=12, fontweight="bold")
    ax_table.text(1.5, -2.0, r"$y_{i,0}$", ha="center", fontsize=14, fontweight="bold")
    ax_table.text(2.5, -2.0, r"$y_{i,1}$", ha="center", fontsize=14, fontweight="bold")
    ax_table.text(3.5, -2.0, r"$D_i$", ha="center", fontsize=14, fontweight="bold")
    ax_table.text(4.5, -2.0, r"$Y_i$", ha="center", fontsize=14, fontweight="bold")
    ax_table.plot([0.1, 4.9], [-1.4, -1.4], color="gray", lw=0.6)

    cell0_artists, cell1_artists, di_artists, yi_artists = [], [], [], []
    for i in range(N_TABLE_ROWS):
        y_pos = i * ROW_SPACING + ROW_OFFSET
        ax_table.text(0.5, y_pos, str(i + 1), ha="center", fontsize=10, color="gray")
        t0 = ax_table.text(1.5, y_pos, f"{y0[i]:+.2f}", ha="center", fontsize=11, family="monospace")
        t1 = ax_table.text(2.5, y_pos, f"{y1[i]:+.2f}", ha="center", fontsize=11, family="monospace")
        td = ax_table.text(3.5, y_pos, "", ha="center", fontsize=11, family="monospace")
        ty = ax_table.text(4.5, y_pos, "", ha="center", fontsize=11, family="monospace")
        cell0_artists.append(t0)
        cell1_artists.append(t1)
        di_artists.append(td)
        yi_artists.append(ty)

    # Ellipsis row indicating the remaining N_UNITS - N_TABLE_ROWS units
    ax_table.text(0.5, ELLIPSIS_Y, "\u22ee", ha="center", va="center", fontsize=13, color="gray")
    ell0 = ax_table.text(1.5, ELLIPSIS_Y, "\u22ee", ha="center", va="center", fontsize=13, color="#2a2a2a")
    ell1 = ax_table.text(2.5, ELLIPSIS_Y, "\u22ee", ha="center", va="center", fontsize=13, color="#2a2a2a")
    elld = ax_table.text(3.5, ELLIPSIS_Y, "", ha="center", va="center", fontsize=13, color="#2a2a2a")
    elly = ax_table.text(4.5, ELLIPSIS_Y, "", ha="center", va="center", fontsize=13, color="#2a2a2a")

    # Bottom line of the values table
    ax_table.plot([0.1, 4.9], [BOTTOM_LINE_Y, BOTTOM_LINE_Y], color="gray", lw=0.6)

    # Truth row: ȳ_{0,n}/ȳ_{1,n} inline under the y columns + ψ_n to the right.
    # Stored as artists so they can fade with the potential outcome columns once realizations start.
    truth_y = BOTTOM_LINE_Y + 1.3
    ybar_text = ax_table.text(
        2.0, truth_y,
        rf"$\overline{{y}}_{{0,n}} = {fmt(y0_bar)}$     $\overline{{y}}_{{1,n}} = {fmt(y1_bar)}$",
        ha="center", fontsize=12, color=TRUTH_COLOR,
    )
    dbar_text = ax_table.text(4.3, truth_y, rf"$\psi_n = {fmt(ate)}$",
                              ha="center", fontsize=12, color=TRUTH_COLOR)

    # Estimate row: ŷ_{0,n}/ŷ_{1,n} inline + ψ̂_n under ψ_n
    estimate_y = BOTTOM_LINE_Y + 3.1
    mu_text = ax_table.text(2.0, estimate_y, "", ha="center", fontsize=12, color="#2a2a2a")
    dhat_text = ax_table.text(4.3, estimate_y, "", ha="center", fontsize=12, fontweight="bold")

    # Realization label
    exp_label = ax_table.text(2.5, BOTTOM_LINE_Y + 6.3, "", ha="center", fontsize=12, color="black")

    # ---------- CI panel ----------
    CI_SPACING = 0.7
    LAST_CI_Y = (N_EXPERIMENTS - 1) * CI_SPACING
    ax_ci.set_xlim(*CI_XLIM)
    ax_ci.set_xticks(CI_XTICKS)
    ax_ci.set_ylim(LAST_CI_Y + 0.5, -2.0)
    ax_ci.set_xlabel(r"$\widehat{\psi}_n$ and $C_{1-\alpha,\,n}$", fontsize=13)
    ax_ci.plot([ate, ate], [-0.5, LAST_CI_Y + 0.5],
               color=TRUTH_COLOR, linestyle="--", lw=1.5, alpha=0.7)
    ax_ci.text(ate, -1.5, r"$\psi_n$",
               ha="center", va="center", fontsize=14, color=TRUTH_COLOR)
    ax_ci.set_yticks([])
    for spine in ("left", "right", "top"):
        ax_ci.spines[spine].set_visible(False)
    ax_ci.tick_params(axis="x", labelsize=10)

    # ---------- histogram bottom panel ----------
    draw_studentized_histogram(ax_hist, studentized)

    INTRO_FRAMES = 5
    OUTRO_FRAMES = 6
    TOTAL = INTRO_FRAMES + N_EXPERIMENTS + OUTRO_FRAMES

    def update(frame: int) -> None:
        if frame < INTRO_FRAMES:
            exp_label.set_text("Fixed potential outcomes")
            for i in range(N_TABLE_ROWS):
                cell0_artists[i].set_alpha(1.0)
                cell1_artists[i].set_alpha(1.0)
                di_artists[i].set_text("")
                yi_artists[i].set_text("")
            ell0.set_alpha(1.0)
            ell1.set_alpha(1.0)
            elld.set_text("")
            elly.set_text("")
            ybar_text.set_alpha(1.0)
            dbar_text.set_alpha(1.0)
            mu_text.set_text("")
            dhat_text.set_text("")
            return

        k = frame - INTRO_FRAMES

        if k >= N_EXPERIMENTS:
            # Hold the final realization on screen (no state changes)
            return

        assignment = assignments[k]
        for i in range(N_TABLE_ROWS):
            di_artists[i].set_text(str(int(assignment[i])))
            Y_i = y1[i] if assignment[i] == 1 else y0[i]
            yi_artists[i].set_text(f"{Y_i:+.2f}")
            cell0_artists[i].set_alpha(0.3)
            cell1_artists[i].set_alpha(0.3)
        ell0.set_alpha(0.3)
        ell1.set_alpha(0.3)
        elld.set_text("\u22ee")
        elly.set_text("\u22ee")
        ybar_text.set_alpha(0.3)
        dbar_text.set_alpha(0.3)

        cov = covered_flags[k]
        color = "#2ca02c" if cov else "#d62728"
        lo, hi = cis[k]
        est = psi_hats[k]
        y_k = k * CI_SPACING
        ax_ci.plot([lo, hi], [y_k, y_k], color=color, lw=2.2, solid_capstyle="round")
        ax_ci.plot([est], [y_k], "o", color=color, markersize=4.5)

        mu_text.set_text(
            rf"$\widehat{{y}}_{{0,n}} = {fmt(y0_bar_hats[k])}$     $\widehat{{y}}_{{1,n}} = {fmt(y1_bar_hats[k])}$"
        )
        dhat_text.set_text(rf"$\widehat{{\psi}}_n = {fmt(est)}$")
        dhat_text.set_color(color)

        exp_label.set_text(f"Possible realization {k + 1} of {N_EXPERIMENTS}")

    anim = FuncAnimation(fig, update, frames=TOTAL, interval=800, blit=False)
    out = str(Path(__file__).parent / out_name)
    anim.save(out, writer=PillowWriter(fps=1.25))
    plt.close(fig)
    print(f"wrote {out}")
    print(f"y0.mean = {y0.mean():.4f}, y1.mean = {y1.mean():.4f}, psi = {ate:.4f}")


if __name__ == "__main__":
    # Homogeneous treatment effects: y_{i,1} = y_{i,0} + 1
    make_coverage_animation(lambda y0: y0 + 1.0, "homogeneous_coverage_animation.gif")
    # Heterogeneous treatment effects: y_{i,1} = -y_{i,0} + 1
    make_coverage_animation(lambda y0: -y0 + 1.0, "heterogeneous_coverage_animation.gif")
