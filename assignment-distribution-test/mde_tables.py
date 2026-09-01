"""
MDE tables for the always-valid e-process test, computed by Monte Carlo.

For each (pi_0, n, alpha):
  - MDE^c:  smallest delta such that cumulative power of e-process at horizon n
            reaches 80%, estimated by MC.
  - MDE^pt: smallest delta such that pointwise power of LR test at horizon n
            reaches 80%, estimated by MC.

Strategy: for each (pi_0, n), pre-estimate MDE via the learning-corrected
closed-form approximation; run MC at a small grid of delta values bracketing
those estimates; track cumulative e-process and pointwise LR power for all
alphas in one pass per delta; interpolate to find the target-power delta.
"""

import json
import os
import sys
import time
import numpy as np
from scipy.stats import chi2, norm, ncx2
from scipy.special import gammaln
from scipy.optimize import brentq


TARGET = 0.80
ALPHAS = [0.05, 0.01, 0.001]
NS = [10**3, 10**4, 10**5, 10**6, 10**7]
K = 2
CACHE_DIR = "mde_tables_cache"
GRID_SIZE = 8


def alt(pi_0, delta):
    pi_1 = pi_0.copy().astype(np.float64)
    pi_1[0] = pi_0[0] + delta
    for j in range(1, len(pi_0)):
        pi_1[j] = pi_0[j] * (1 - delta / (1 - pi_0[0]))
    return pi_1


# --------------- closed-form starting points ---------------
def _drift_var(pi_0, pi_1):
    lr = np.log(pi_1 / pi_0)
    mu = float(np.sum(pi_1 * lr))
    var = float(np.sum(pi_1 * (lr - mu) ** 2))
    return mu, var


def omega_e_cf(n, alpha, pi_0, pi_1):
    a = K * pi_0
    mu, var = _drift_var(pi_0, pi_1)
    if var <= 0:
        return 1.0 if mu > 0 else 0.0
    sigma = np.sqrt(var)
    b = np.log(1.0 / alpha)
    A = float(np.sum(a))
    C = (
        gammaln(A)
        - float(np.sum(gammaln(a)))
        + float(np.sum((a - 0.5) * np.log(pi_1)))
        + (K - 1) / 2 * np.log(2 * np.pi)
    )
    b_n = b + (K - 1) / 2 * np.log(n) - C
    sqrt_n = np.sqrt(n)
    z1 = (mu * n - b_n) / (sigma * sqrt_n)
    z2 = -(mu * n + b_n) / (sigma * sqrt_n)
    log_term2 = 2 * mu * b_n / var + norm.logcdf(z2)
    return float(min(norm.cdf(z1) + np.exp(min(log_term2, 0.0)), 1.0))


def omega_lr_cf(n, alpha, pi_0, pi_1):
    c = chi2.ppf(1 - alpha, df=K - 1)
    lam = n * float(np.sum((pi_1 - pi_0) ** 2 / pi_0))
    return 1.0 - ncx2.cdf(c, df=K - 1, nc=lam)


def closed_form_mde(omega_fn, n, alpha, pi_0):
    eps = 1e-12
    max_d = 1.0 - pi_0[0] - eps

    def f(d):
        return omega_fn(n, alpha, pi_0, alt(pi_0, d)) - TARGET

    if f(eps) >= 0:
        return eps
    if f(max_d) < 0:
        return None
    return brentq(f, eps, max_d, xtol=max_d * 1e-4)


# --------------- MC core ---------------
def simulate_powers(pi_0, pi_1, n, n_seqs, alphas, seed, chunk_size):
    rng = np.random.default_rng(seed)
    a = K * pi_0
    A = float(a.sum())
    p1 = float(pi_1[1])

    b_levels = np.log(1.0 / np.asarray(alphas))
    c_alphas = np.array([chi2.ppf(1 - al, df=K - 1) for al in alphas])

    N_run = np.zeros((n_seqs, 2), dtype=np.int64)
    log_E = np.zeros(n_seqs)
    log_E_max = np.full(n_seqs, -np.inf)
    crossed = np.zeros((len(alphas), n_seqs), dtype=bool)

    n_done = 0
    while n_done < n:
        T = min(chunk_size, n - n_done)
        X = (rng.random((n_seqs, T)) < p1).astype(np.int64)
        cumX = X.cumsum(axis=1)
        cumX_prev = np.empty_like(cumX)
        cumX_prev[:, 0] = 0
        cumX_prev[:, 1:] = cumX[:, :-1]

        t_arr = np.arange(T)
        N_1_before = N_run[:, 1:2] + cumX_prev
        N_0_before = N_run[:, 0:1] + t_arr[None, :] - cumX_prev
        s_prev = n_done + t_arr

        num = np.where(X == 1, a[1] + N_1_before, a[0] + N_0_before)
        denom = (A + s_prev[None, :]) * np.where(X == 1, pi_0[1], pi_0[0])
        log_M = np.log(num / denom)

        log_E_chunk = log_E[:, None] + log_M.cumsum(axis=1)
        running_max_chunk = np.maximum.accumulate(log_E_chunk, axis=1)
        running_max_chunk = np.maximum(running_max_chunk, log_E_max[:, None])

        chunk_max = running_max_chunk[:, -1]
        for ai, b_lvl in enumerate(b_levels):
            crossed[ai] |= chunk_max >= b_lvl

        log_E = log_E_chunk[:, -1]
        log_E_max = chunk_max
        N_run[:, 1] += cumX[:, -1]
        N_run[:, 0] = (n_done + T) - N_run[:, 1]
        n_done += T

    cum_e = crossed.mean(axis=1)

    N_0 = N_run[:, 0]
    N_1 = N_run[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        T_stat = 2 * (
            np.where(N_0 > 0, N_0 * np.log(N_0 / (n * pi_0[0])), 0.0)
            + np.where(N_1 > 0, N_1 * np.log(N_1 / (n * pi_0[1])), 0.0)
        )
    pt_lr = np.array([(T_stat >= c).mean() for c in c_alphas])
    return cum_e, pt_lr


def interp_mde(deltas, powers, target):
    order = np.argsort(deltas)
    d = deltas[order]
    p = powers[order]
    above = p >= target
    if not above.any():
        return None
    first = int(above.argmax())
    if first == 0:
        return float(d[0])
    d0, d1 = d[first - 1], d[first]
    p0, p1 = p[first - 1], p[first]
    if p1 == p0:
        return float(d0)
    return float(d0 + (d1 - d0) * (target - p0) / (p1 - p0))


def n_seqs_for(n):
    if n <= 1_000:
        return 5000
    if n <= 10_000:
        return 3000
    if n <= 100_000:
        return 2000
    if n <= 1_000_000:
        return 1000
    return 300


def chunk_size_for(n, n_seqs):
    if n <= 50_000:
        return n
    target_elements = 8_000_000
    cs = max(1000, target_elements // n_seqs)
    return min(cs, n)


def mc_mdes_for_pi_n(pi_0, n, seed):
    cf = []
    for alpha in ALPHAS:
        m_e = closed_form_mde(omega_e_cf, n, alpha, pi_0)
        m_l = closed_form_mde(omega_lr_cf, n, alpha, pi_0)
        cf.append(m_e)
        cf.append(m_l)
    cf = [c for c in cf if c is not None and c > 0]
    if not cf:
        return {alpha: {"mde_e": None, "mde_lr": None} for alpha in ALPHAS}
    d_lo = max(min(cf) * 0.55, 1e-12)
    d_hi = min(max(cf) * 1.7, 1 - pi_0[0] - 1e-12)
    deltas = np.geomspace(d_lo, d_hi, GRID_SIZE)

    n_seqs = n_seqs_for(n)
    chunk = chunk_size_for(n, n_seqs)
    cum_powers = np.zeros((GRID_SIZE, len(ALPHAS)))
    pt_powers = np.zeros((GRID_SIZE, len(ALPHAS)))
    for i, d in enumerate(deltas):
        pi_1 = alt(pi_0, d)
        cum, pt = simulate_powers(
            pi_0, pi_1, n, n_seqs, ALPHAS, seed=seed + i, chunk_size=chunk
        )
        cum_powers[i] = cum
        pt_powers[i] = pt

    out = {}
    for ai, alpha in enumerate(ALPHAS):
        out[alpha] = {
            "mde_e": interp_mde(deltas, cum_powers[:, ai], TARGET),
            "mde_lr": interp_mde(deltas, pt_powers[:, ai], TARGET),
            "grid": [float(x) for x in deltas],
            "cum_powers": [float(x) for x in cum_powers[:, ai]],
            "pt_powers": [float(x) for x in pt_powers[:, ai]],
        }
    return out


def run_all(pi_0_list, cache_path):
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    results = {}
    for pi_0_arr in pi_0_list:
        pi_0 = np.asarray(pi_0_arr, dtype=np.float64)
        key = f"{pi_0[0]:g}_{pi_0[1]:g}"
        results[key] = {}
        for n in NS:
            t0 = time.time()
            seed = 9000 + int(np.log10(n)) * 100 + int(pi_0[0] * 1000)
            out = mc_mdes_for_pi_n(pi_0, n, seed)
            dt = time.time() - t0
            results[key][str(n)] = {
                str(alpha): {
                    "mde_e": out[alpha]["mde_e"],
                    "mde_lr": out[alpha]["mde_lr"],
                }
                for alpha in ALPHAS
            }
            print(
                f"pi_0=({pi_0[0]:g},{pi_0[1]:g}) n=10^{int(np.log10(n))} "
                f"[{dt:.1f}s]:",
                flush=True,
            )
            for alpha in ALPHAS:
                me = out[alpha]["mde_e"]
                ml = out[alpha]["mde_lr"]
                me_s = f"{me:.4g}" if me is not None else "n/a"
                ml_s = f"{ml:.4g}" if ml is not None else "n/a"
                print(f"  alpha={alpha:g}: MDE_e={me_s}, MDE_lr={ml_s}", flush=True)
            with open(cache_path, "w") as f:
                json.dump(results, f, indent=2)
    return results


# --------------- formatting ---------------
import math


def fmt_2sig(x):
    """Round to 2 significant figures, decimal notation (no scientific)."""
    if x is None:
        return "—"
    if x <= 0:
        return "0"
    exp = math.floor(math.log10(x))
    factor = 10 ** (exp - 1)
    rounded = math.floor(x / factor + 0.5) * factor
    decimals = max(0, -(exp - 1))
    return f"{rounded:.{decimals}f}"


def fmt_mde(d):
    return fmt_2sig(d)


def fmt_ratio(r):
    if r is None or not np.isfinite(r):
        return "—"
    return f"{fmt_2sig(r)}×"


def fmt_pct(pct):
    if pct is None:
        return "—"
    return f"{fmt_2sig(pct)}\\%"


def fmt_n(n):
    return f"$10^{{{int(np.log10(n))}}}$"


def make_table(results, pi_0_target, label):
    key = f"{pi_0_target[0]:g}_{pi_0_target[1]:g}"
    pi_0_str = f"({pi_0_target[0]:g}, {pi_0_target[1]:g})"
    pi_0_1 = pi_0_target[0]
    lines = []
    cols = ["$n$"] + [f"$\\alpha = {a:g}$" for a in ALPHAS]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for n in NS:
        cells = [fmt_n(n)]
        for alpha in ALPHAS:
            entry = results[key][str(n)][str(alpha)]
            mde_e = entry["mde_e"]
            mde_lr = entry["mde_lr"]
            ratio = (
                mde_e / mde_lr
                if (mde_e is not None and mde_lr is not None and mde_lr > 0)
                else None
            )
            pct = mde_e / pi_0_1 * 100 if mde_e is not None else None
            pct_lr = mde_lr / pi_0_1 * 100 if mde_lr is not None else None
            cells.append(
                f'<span style="line-height:1.6">'
                f"{fmt_mde(mde_e)} ({fmt_pct(pct)})<br>"
                f"{fmt_mde(mde_lr)} ({fmt_pct(pct_lr)})"
                f"</span>"
            )
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        f": Minimum detectable effect at $80\\%$ power under "
        f"$\\boldsymbol{{\\pi}}_0 = {pi_0_str}$, $K = 2$, $k = 1$, "
        f"pseudo-counts $\\bar A = K$. Each cell shows the e-process "
        f"cumulative-power MDE on top, the LR fixed-horizon pointwise "
        f"MDE on the bottom, and each MDE divided by $\\pi_{{0,1}}$ in "
        f"parentheses. Estimated by Monte Carlo simulation. {{#{label}}}"
    )
    return "\n".join(lines)


def main():
    cache_path = os.path.join(CACHE_DIR, "results.json")
    pi_0_list = [[0.5, 0.5], [0.01, 0.99]]
    results = run_all(pi_0_list, cache_path)
    print()
    print(make_table(results, [0.5, 0.5], "tbl-mde-balanced"))
    print()
    print(make_table(results, [0.01, 0.99], "tbl-mde-imbalanced"))


if __name__ == "__main__":
    main()
