"""Animated GIF: CI coverage when treatment effects are perfectly homogeneous."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

RNG = np.random.default_rng(42)

N = 30
P = 0.5
N_EXPERIMENTS = 20
Z = 1.959963984540054  # 0.975 quantile of N(0,1)

# Shared x-axis range and ticks across the homogeneous/heterogeneous figures
# so the two animations are visually comparable.
CI_XLIM = (-0.4, 2.4)
CI_XTICKS = [0.0, 0.5, 1.0, 1.5, 2.0]

# Homogeneous treatment effects: y_{i,1} = y_{i,0} + 1 for every i,
# so Δ_i = 1 exactly for all i.
y0_raw = RNG.normal(0, 1, size=N)
y0 = (y0_raw - y0_raw.mean()).round(2)
y0[0] = (y0[0] - y0.sum()).round(2)
y0[-1] = round(y0[-1] - y0.sum(), 2)
y1 = (y0 + 1.0).round(2)

y0_bar = y0.mean()
y1_bar = y1.mean()
delta_bar = y1_bar - y0_bar


def fmt(x: float) -> str:
    """Format with 2 decimals, mapping near-zero to '0.00' (no '-0.00')."""
    return "0.00" if abs(x) < 0.005 else f"{x:.2f}"


def run_experiment(D: np.ndarray) -> tuple[float, float, float, float, float, bool]:
    D_bar = D.mean()
    n_t, n_c = D.sum(), N - D.sum()
    mu1 = (D * y1).sum() / n_t
    mu0 = ((1 - D) * y0).sum() / n_c
    delta_hat = mu1 - mu0
    S1_sq = (D * (y1 - mu1) ** 2).sum() / n_t
    S0_sq = ((1 - D) * (y0 - mu0) ** 2).sum() / n_c
    sigma_sq_hat = S1_sq / D_bar + S0_sq / (1 - D_bar)
    se = np.sqrt(sigma_sq_hat / N)
    lo, hi = delta_hat - Z * se, delta_hat + Z * se
    return float(mu0), float(mu1), float(delta_hat), float(lo), float(hi), bool(lo <= delta_bar <= hi)


assignments, mu0s, mu1s, estimates, cis, covered_flags = [], [], [], [], [], []
for _ in range(N_EXPERIMENTS):
    D = RNG.binomial(1, P, size=N)
    while D.sum() in (0, N):
        D = RNG.binomial(1, P, size=N)
    assignments.append(D)
    m0, m1, est, lo, hi, cov = run_experiment(D)
    mu0s.append(m0)
    mu1s.append(m1)
    estimates.append(est)
    cis.append((lo, hi))
    covered_flags.append(cov)

# ---------- plot setup ----------
TRUTH_COLOR = "#555"

# Table panel geometry
ROW_OFFSET = 0.4
ROW_SPACING = 1.3
BOTTOM_LINE_Y = (N - 1) * ROW_SPACING + ROW_OFFSET + 0.6
TABLE_DATA_RANGE = BOTTOM_LINE_Y + 10.0  # 7 below bottom line, 3 above header
FIG_HEIGHT = 9.5 * TABLE_DATA_RANGE / 40.0  # 40 = original data range

fig = plt.figure(figsize=(13, FIG_HEIGHT))
gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.3], wspace=0.08)
ax_table = fig.add_subplot(gs[0, 0])
gs_right = gs[0, 1].subgridspec(2, 1, height_ratios=[20, 1])
ax_ci = fig.add_subplot(gs_right[0, 0])
fig.subplots_adjust(top=0.98, bottom=0.06)

# table panel: 5 data columns at x = 0.5, 1.5, 2.5, 3.5, 4.5; footer Δ column at x = 4.3
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
for i in range(N):
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

# Bottom line of the values table
ax_table.plot([0.1, 4.9], [BOTTOM_LINE_Y, BOTTOM_LINE_Y], color="gray", lw=0.6)

# Truth row: ȳ_0/ȳ_1 inline under the y columns + Δ̄ to the right.
# Stored as artists so they can fade with the potential outcome columns once realizations start.
truth_y = BOTTOM_LINE_Y + 1.3
ybar_text = ax_table.text(
    2.0, truth_y,
    rf"$\overline{{y_0}} = {fmt(y0_bar)}$     $\overline{{y_1}} = {fmt(y1_bar)}$",
    ha="center", fontsize=12, color=TRUTH_COLOR,
)
dbar_text = ax_table.text(4.3, truth_y, rf"$\psi_n = {fmt(delta_bar)}$",
                          ha="center", fontsize=12, color=TRUTH_COLOR)

# Estimate row: μ̂_0/μ̂_1 inline + Δ̂_n under Δ̄
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
ax_ci.set_xlabel(r"$\widehat{\psi}_n$ and $C_{n,\,1-\alpha}$", fontsize=13)
ax_ci.plot([delta_bar, delta_bar], [-0.5, LAST_CI_Y + 0.5],
           color=TRUTH_COLOR, linestyle="--", lw=1.5, alpha=0.7)
ax_ci.text(delta_bar, -1.5, r"$\psi_n$",
           ha="center", va="center", fontsize=14, color=TRUTH_COLOR)
ax_ci.set_yticks([])
for spine in ("left", "right", "top"):
    ax_ci.spines[spine].set_visible(False)
ax_ci.tick_params(axis="x", labelsize=10)

INTRO_FRAMES = 5
OUTRO_FRAMES = 6
TOTAL = INTRO_FRAMES + N_EXPERIMENTS + OUTRO_FRAMES


def update(frame: int) -> None:
    if frame < INTRO_FRAMES:
        exp_label.set_text("Fixed potential outcomes")
        for i in range(N):
            cell0_artists[i].set_alpha(1.0)
            cell1_artists[i].set_alpha(1.0)
            di_artists[i].set_text("")
            yi_artists[i].set_text("")
        ybar_text.set_alpha(1.0)
        dbar_text.set_alpha(1.0)
        mu_text.set_text("")
        dhat_text.set_text("")
        return

    k = frame - INTRO_FRAMES

    if k >= N_EXPERIMENTS:
        # Hold the final realization on screen (no state changes)
        return

    D = assignments[k]
    for i in range(N):
        di_artists[i].set_text(str(int(D[i])))
        Y_i = y1[i] if D[i] == 1 else y0[i]
        yi_artists[i].set_text(f"{Y_i:+.2f}")
        cell0_artists[i].set_alpha(0.3)
        cell1_artists[i].set_alpha(0.3)
    ybar_text.set_alpha(0.3)
    dbar_text.set_alpha(0.3)

    cov = covered_flags[k]
    color = "#2ca02c" if cov else "#d62728"
    lo, hi = cis[k]
    est = estimates[k]
    y_k = k * CI_SPACING
    ax_ci.plot([lo, hi], [y_k, y_k], color=color, lw=2.2, solid_capstyle="round")
    ax_ci.plot([est], [y_k], "o", color=color, markersize=4.5)

    mu_text.set_text(
        rf"$\widehat{{y_0}} = {fmt(mu0s[k])}$     $\widehat{{y_1}} = {fmt(mu1s[k])}$"
    )
    dhat_text.set_text(rf"$\widehat{{\psi}}_n = {fmt(est)}$")
    dhat_text.set_color(color)

    exp_label.set_text(f"Possible realization {k + 1} of {N_EXPERIMENTS}")


anim = FuncAnimation(fig, update, frames=TOTAL, interval=800, blit=False)
out = str(Path(__file__).parent / "homogeneous_coverage_animation.gif")
anim.save(out, writer=PillowWriter(fps=1.25))
plt.close(fig)
print(f"wrote {out}")
print(f"y0.mean = {y0.mean():.4f}, y1.mean = {y1.mean():.4f}, delta_bar = {delta_bar:.4f}")
