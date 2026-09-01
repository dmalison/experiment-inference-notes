"""
Illustrate the gap between an asymptotically valid test (the likelihood
ratio test) and an always-valid test (the e-process from index.qmd).

Simulates n_seqs sequences of length n_steps from the balanced K=2 null and
animates T_n and E_n over time, marking the first threshold crossing of
each sequence with a star.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import chi2
from scipy.special import xlogy, gammaln


# ---- Parameters ---------------------------------------------------------
SEED = 2038
K = 2
pi = np.array([0.5, 0.5])
n_seqs = 20
n_steps = 100
frame_step = 1  # animation advances by this many time steps per frame
alpha = 0.05

c_alpha = chi2.ppf(1 - alpha, df=K - 1)        # LR test threshold
e_threshold = 1.0 / alpha                       # Ville threshold

# E-process prior pseudo-counts: a_k = A_bar * pi_k with A_bar = K
A_bar = K
a = A_bar * pi
A = a.sum()


# ---- Simulation ---------------------------------------------------------
def simulate(n_steps, pi, a, rng):
    """Return (T, E) trajectories of length n_steps under the null."""
    A_ = a.sum()
    K_ = len(pi)
    D = rng.choice(K_, size=n_steps, p=pi)
    counts = np.zeros(K_)
    T = np.zeros(n_steps)
    E = np.zeros(n_steps)
    log_E = 0.0
    for n in range(n_steps):
        # Predictive q_{n-1,k}(a) = (a_k + N_{n-1,k}) / (A + n - 1)
        # In 0-indexed terms: at step n we have observed n previous units,
        # so denominator is A_ + n (matches A + (n+1) - 1 in 1-indexed text).
        q = (a + counts) / (A_ + n)
        d = D[n]
        log_E += np.log(q[d] / pi[d])
        counts[d] += 1
        n_obs = n + 1
        # Likelihood-ratio statistic; xlogy handles 0 log 0 = 0.
        T[n] = 2.0 * np.sum(xlogy(counts, counts / (n_obs * pi)))
        E[n] = np.exp(log_E)
    return T, E


rng = np.random.default_rng(SEED)
T_seqs = np.zeros((n_seqs, n_steps))
E_seqs = np.zeros((n_seqs, n_steps))
for i in range(n_seqs):
    T_seqs[i], E_seqs[i] = simulate(n_steps, pi, a, rng)


def first_crossing(seq, threshold):
    idx = np.where(seq >= threshold)[0]
    return int(idx[0]) if len(idx) else None


T_cross = [first_crossing(T_seqs[i], c_alpha) for i in range(n_seqs)]
E_cross = [first_crossing(E_seqs[i], e_threshold) for i in range(n_seqs)]


# Closed-form constant C in the asymptotic expansion
#   log E_n = T_n/2 - (K-1)/2 log n + C + o_p(1)
# obtained by Stirling-expanding the marginal Dirichlet code length.
C_const = (
    gammaln(A) - gammaln(a).sum()
    + ((a - 0.5) * np.log(pi)).sum()
    + 0.5 * (K - 1) * np.log(2 * np.pi)
)
gap_seqs = np.log(E_seqs) - T_seqs / 2.0
n_axis = np.arange(1, n_steps + 1)
expansion = -0.5 * (K - 1) * np.log(n_axis) + C_const


def jittered_star_positions(seqs, crossings, jitter=0.4):
    """For each crossing sequence, return (x, y). Sequences crossing at the
    same (n, value) are spread horizontally so their stars don't fully overlap.
    """
    raw = [
        (j + 1, seqs[i, j]) if j is not None else None
        for i, j in enumerate(crossings)
    ]
    groups = {}
    for i, pos in enumerate(raw):
        if pos is not None:
            groups.setdefault(pos, []).append(i)
    star_x = [None] * len(crossings)
    star_y = [None] * len(crossings)
    for (x_, y_), idxs in groups.items():
        if len(idxs) == 1:
            star_x[idxs[0]] = x_
            star_y[idxs[0]] = y_
        else:
            offsets = np.linspace(-jitter, jitter, len(idxs))
            for k, idx in enumerate(idxs):
                star_x[idx] = x_ + offsets[k]
                star_y[idx] = y_
    return star_x, star_y


logE_seqs = np.log(E_seqs)
log_e_threshold = np.log(e_threshold)

T_star_x, T_star_y = jittered_star_positions(T_seqs, T_cross)
E_star_x, E_star_y = jittered_star_positions(logE_seqs, E_cross)


# ---- Figure -------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, figsize=(9, 10), sharex=True, constrained_layout=True
)
colors = plt.cm.tab20(np.linspace(0, 1, n_seqs))

ax1.axhline(
    c_alpha, color="black", ls="--", lw=1.2,
    label=fr"$c_\alpha = {c_alpha:.2f}$",
)
ax1.set_ylabel(r"$T_n$")
ax1.set_title("Likelihood ratio")
ax1.legend(loc="upper left")

ax2.axhline(
    log_e_threshold, color="black", ls="--", lw=1.2,
    label=fr"$\log(1/\alpha) = {log_e_threshold:.2f}$",
)
ax2.set_ylabel(r"$\log E_n$")
ax2.set_title("Evidence process")
ax2.legend(loc="upper left")

ax3.plot(
    n_axis, expansion, color="black", ls="--", lw=1.2,
    label=r"$-\frac{K-1}{2}\log n + C$",
)
ax3.set_ylabel(r"$\log E_n - T_n/2$")
ax3.set_xlabel(r"$n$")
ax3.set_title("Asymptotic expansion")
ax3.legend(loc="upper right")

ax1.set_xlim(1, n_steps)
y_lo = min(0.0, float(logE_seqs.min()))
y_hi = max(
    15.0,
    float(np.max(T_seqs)) * 1.05,
    float(logE_seqs.max()),
    c_alpha,
    log_e_threshold,
)
y_pad = 0.05 * (y_hi - y_lo)
ax1.set_ylim(y_lo - y_pad, y_hi + y_pad)
ax2.set_ylim(y_lo - y_pad, y_hi + y_pad)
gap_lo = min(float(gap_seqs.min()), float(expansion.min()))
gap_hi = max(float(gap_seqs.max()), float(expansion.max()))
gap_pad = 0.1 * (gap_hi - gap_lo) if gap_hi > gap_lo else 1.0
ax3.set_ylim(gap_lo - gap_pad, gap_hi + gap_pad)

lines1, lines2, lines3, stars1, stars2 = [], [], [], [], []
for i in range(n_seqs):
    (l1,) = ax1.plot([], [], color=colors[i], alpha=0.75, lw=1.2)
    (l2,) = ax2.plot([], [], color=colors[i], alpha=0.75, lw=1.2)
    (l3,) = ax3.plot([], [], color=colors[i], alpha=0.75, lw=1.2)
    (s1,) = ax1.plot(
        [], [], color=colors[i], marker="*", ms=16, ls="",
        mec="black", mew=0.8,
    )
    (s2,) = ax2.plot(
        [], [], color=colors[i], marker="*", ms=16, ls="",
        mec="black", mew=0.8,
    )
    lines1.append(l1)
    lines2.append(l2)
    lines3.append(l3)
    stars1.append(s1)
    stars2.append(s2)

counter1 = ax1.text(
    0.99, 0.95, "", transform=ax1.transAxes, ha="right", va="top",
    fontsize=11, family="monospace",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.85),
)
counter2 = ax2.text(
    0.99, 0.95, "", transform=ax2.transAxes, ha="right", va="top",
    fontsize=11, family="monospace",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.85),
)

x = np.arange(1, n_steps + 1)

# Animate one frame per `frame_step` time steps, then hold the last frame.
ANIM_FRAMES = (n_steps + frame_step - 1) // frame_step
HOLD_FRAMES = 24  # ~2 seconds at fps=12
TOTAL_FRAMES = ANIM_FRAMES + HOLD_FRAMES


def init():
    artists = []
    for i in range(n_seqs):
        lines1[i].set_data([], [])
        lines2[i].set_data([], [])
        lines3[i].set_data([], [])
        stars1[i].set_data([], [])
        stars2[i].set_data([], [])
        artists += [lines1[i], lines2[i], lines3[i], stars1[i], stars2[i]]
    counter1.set_text("")
    counter2.set_text("")
    artists += [counter1, counter2]
    return artists


def update(frame):
    artists = []
    n = min((frame + 1) * frame_step, n_steps)
    fp1 = 0
    fp2 = 0
    for i in range(n_seqs):
        lines1[i].set_data(x[:n], T_seqs[i, :n])
        lines2[i].set_data(x[:n], logE_seqs[i, :n])
        lines3[i].set_data(x[:n], gap_seqs[i, :n])
        if T_cross[i] is not None and T_cross[i] < n:
            stars1[i].set_data([T_star_x[i]], [T_star_y[i]])
            fp1 += 1
        if E_cross[i] is not None and E_cross[i] < n:
            stars2[i].set_data([E_star_x[i]], [E_star_y[i]])
            fp2 += 1
        artists += [lines1[i], lines2[i], lines3[i], stars1[i], stars2[i]]
    counter1.set_text(f"false positives: {fp1}/{n_seqs}")
    counter2.set_text(f"false positives: {fp2}/{n_seqs}")
    artists += [counter1, counter2]
    return artists


ani = animation.FuncAnimation(
    fig, update, frames=TOTAL_FRAMES, init_func=init, blit=True, interval=120
)
ani.save("animation.gif", writer=animation.PillowWriter(fps=12), dpi=72)
plt.close(fig)

print(f"Saved animation.gif")
print(f"Likelihood ratio crossings: {sum(c is not None for c in T_cross)} / {n_seqs}")
print(f"E-process crossings: {sum(c is not None for c in E_cross)} / {n_seqs}")
