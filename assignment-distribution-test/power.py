"""
Estimate by Monte Carlo, for K=2 under an iid alternative pi_1:
  - cumulative power of the always-valid e-process test, omega^c_{n,alpha},
  - pointwise power of the fixed-horizon LR test, omega_{n,alpha},
both as functions of horizon n on a fixed time grid.

Single panel, linear x axis. Results are cached on disk.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2


# ---- Parameters ---------------------------------------------------------
SEED = 2026
K = 2
pi_0 = np.array([0.5, 0.5])
pi_1 = np.array([0.45, 0.55])
n_steps = 5_000
n_seqs = 10_000
alpha = 0.05

A_bar = K
a = A_bar * pi_0
A = a.sum()
b_thresh = np.log(1.0 / alpha)
c_alpha = chi2.ppf(1 - alpha, df=K - 1)


# ---- Simulation ---------------------------------------------------------
def simulate(pi_0, pi_1, a, n_steps, n_seqs, b, c_alpha, rng):
    """Run n_seqs sequences of length n_steps under pi_1.

    Returns:
      omega_e_cum: (n_steps,) cumulative-rejection probability of the e-process
                   test estimated as fraction of sequences whose log_E first
                   crossed b by step n.
      omega_lr_pt: (n_steps,) pointwise rejection probability of the LR test
                   estimated as fraction of sequences whose T_n >= c_alpha at
                   step n.
    """
    A_ = float(a.sum())
    K_ = len(pi_0)
    counts = np.zeros((n_seqs, K_), dtype=np.int64)
    log_E = np.zeros(n_seqs)
    crossed_at = np.full(n_seqs, n_steps + 1, dtype=np.int64)
    crossed = np.zeros(n_seqs, dtype=bool)
    log_pi_0 = np.log(pi_0)
    idx = np.arange(n_seqs)
    cum_pi1 = np.cumsum(pi_1)
    lr_count = np.zeros(n_steps, dtype=np.int64)

    for n in range(n_steps):
        U = rng.random(n_seqs)
        d = np.searchsorted(cum_pi1, U)
        q = (a + counts) / (A_ + n)
        log_E += np.log(q[idx, d]) - log_pi_0[d]
        counts[idx, d] += 1

        # cumulative e-process crossing
        new = (~crossed) & (log_E >= b)
        crossed_at[new] = n + 1
        crossed |= new

        # pointwise LR statistic at step n+1
        n_now = n + 1
        with np.errstate(divide="ignore", invalid="ignore"):
            T = 2 * (
                np.where(counts[:, 0] > 0,
                         counts[:, 0] * np.log(counts[:, 0] / (n_now * pi_0[0])),
                         0.0)
                + np.where(counts[:, 1] > 0,
                           counts[:, 1] * np.log(counts[:, 1] / (n_now * pi_0[1])),
                           0.0)
            )
        lr_count[n] = int((T >= c_alpha).sum())

    n_grid = np.arange(1, n_steps + 1)
    sorted_times = np.sort(crossed_at)
    omega_e_cum = np.searchsorted(sorted_times, n_grid, side="right") / n_seqs
    omega_lr_pt = lr_count / n_seqs
    return omega_e_cum, omega_lr_pt


def cached_simulate(pi_0, pi_1, a, n_steps, n_seqs, b, c_alpha, seed):
    cache_dir = "power_cache"
    os.makedirs(cache_dir, exist_ok=True)
    key = (
        f"K{len(pi_0)}_pi0_{'-'.join(f'{x:g}' for x in pi_0)}"
        f"_pi1_{'-'.join(f'{x:g}' for x in pi_1)}"
        f"_a_{'-'.join(f'{x:g}' for x in a)}"
        f"_n{n_steps}_seqs{n_seqs}_b{b:g}_c{c_alpha:g}_seed{seed}_both.npz"
    )
    path = os.path.join(cache_dir, key)
    if os.path.exists(path):
        print(f"  loading cached {path}")
        d = np.load(path)
        return d["omega_e_cum"], d["omega_lr_pt"]
    print(f"  simulating (will cache to {path})")
    rng = np.random.default_rng(seed)
    omega_e_cum, omega_lr_pt = simulate(
        pi_0, pi_1, a, n_steps, n_seqs, b, c_alpha, rng
    )
    np.savez(path, omega_e_cum=omega_e_cum, omega_lr_pt=omega_lr_pt)
    return omega_e_cum, omega_lr_pt


# ---- Run ----------------------------------------------------------------
n_grid = np.arange(1, n_steps + 1)
print(f"pi_1 = {tuple(pi_1)}")
omega_e_cum, omega_lr_pt = cached_simulate(
    pi_0, pi_1, a, n_steps, n_seqs, b_thresh, c_alpha, SEED
)

fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
ax.plot(
    n_grid, omega_e_cum, color="tab:blue", lw=1.4,
    label="e-process (cumulative)",
)
ax.plot(
    n_grid, omega_lr_pt, color="tab:orange", lw=1.4, ls="--",
    label="likelihood ratio (pointwise)",
)
ax.set_xlabel(r"$n$")
ax.set_ylabel("Rejection probability")
ax.set_xlim(1, n_steps)
ax.set_ylim(-0.02, 1.02)
ax.grid(True, alpha=0.25)
ax.set_title("Power under the alternative")
ax.legend(loc="lower right", framealpha=0.9)

out_path = "power.png"
plt.savefig(out_path, dpi=150)
plt.close(fig)
print(f"Saved {out_path}")
