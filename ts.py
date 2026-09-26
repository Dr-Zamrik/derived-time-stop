"""ts.py — the one solver behind every number and figure of *The Derived Time Stop*.

Everything is DIMENSIONLESS (Tim, 2026-09-26: no invented market numbers). With drifts mu_H > 0 > mu_L,
Delta = mu_H - mu_L and volatility sigma:
    theta = -mu_L / Delta                      the break-even belief
    s     = t * (Delta / sigma)^2              time in information units; the deadline is S = T (Delta/sigma)^2
    x     = X * Delta / sigma^2                return in units of sigma^2 / Delta
    y     = logit(pi)                          the belief's log-odds
Then dx = (mu/Delta) ds + dB with mu/Delta = 1 - theta (edge real) or -theta (no edge),
y = logit(p0) + x - (1/2 - theta) s, and dy = dB_hat + (pi - 1/2) ds under the observer's measure.
The value of holding on is v(s, y) = sup_tau E[ int_s^tau (pi_u - theta) du ], tau <= S; the dollar value is
(sigma^2 / Delta) v.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import solve_banded
from scipy.special import expit, logit
from scipy.stats import norm

BGK = 0.5826          # the continuity-correction constant, -zeta(1/2)/sqrt(2 pi)


# ── the obstacle problem, backward in time, on a uniform log-odds grid ─────────────────────────────
def solve(theta: float, S: float, N: int = 2401, L: float = 18.0, ds: float = 2e-4, keep: int = 81):
    """Projected implicit scheme: (I - ds A) v^n = v^{n+1} + ds (pi - theta), then v^n = max(v^n, 0).
    A v = 1/2 v_yy + (pi - 1/2) v_y, central differences (monotone since |pi - 1/2| dy < 1).
    Dirichlet: v = 0 at y = -L (certain there is no edge: stop); v = (pi - theta)(S - s) at y = +L."""
    y = np.linspace(-L, L, N); dy = y[1] - y[0]; pi = expit(y)
    M = max(1, int(round(S / ds))); ds = S / M
    d = pi[1:-1] - 0.5
    lo = 0.5 / dy**2 - d / (2 * dy); di = -1.0 / dy**2 * np.ones(N - 2); up = 0.5 / dy**2 + d / (2 * dy)
    ab = np.zeros((3, N - 2))
    ab[0, 1:] = -ds * up[:-1]; ab[1] = 1 - ds * di; ab[2, :-1] = -ds * lo[1:]
    v = np.zeros(N)
    s_b = np.empty(M + 1); y_b = np.empty(M + 1)
    s_b[M], y_b[M] = S, logit(theta)                          # at the deadline: stop exactly below theta
    snap_idx = set(np.linspace(0, M, keep).round().astype(int).tolist())
    snaps = {M: v.copy()}
    src = ds * (pi[1:-1] - theta)
    for n in range(M - 1, -1, -1):
        s = n * ds
        top = (pi[-1] - theta) * (S - s)
        rhs = v[1:-1] + src
        rhs[-1] += ds * up[-1] * top
        vi = solve_banded((1, 1), ab, rhs)
        v = np.concatenate(([0.0], np.maximum(vi, 0.0), [top]))
        s_b[n], y_b[n] = s, _boundary(y, v, dy)
        if n in snap_idx:
            snaps[n] = v.copy()
    order = sorted(snaps)
    return {"theta": theta, "S": S, "y": y, "pi": pi, "ds": ds, "dy": dy,
            "s_b": s_b, "y_b": y_b, "b": expit(y_b),
            "s_snap": np.array([k * ds for k in order]), "V": np.array([snaps[k] for k in order])}


def _boundary(y, v, dy, eps=1e-15):
    """Where v leaves zero. Smooth fit makes v ~ c (y - y_b)^2, so sqrt(v) is linear there."""
    i = int(np.argmax(v > eps))
    if i == 0:
        return y[0]
    r0, r1 = np.sqrt(v[i]), np.sqrt(v[i + 1])
    return y[i] - r0 * dy / max(r1 - r0, 1e-300)


def value_at(sol, p0: float, s: float = 0.0):
    """v(s, logit p0), linear in y between grid nodes, from the nearest stored time slice."""
    k = int(np.argmin(abs(sol["s_snap"] - s)))
    return float(np.interp(logit(p0), sol["y"], sol["V"][k]))


def y_boundary(sol, s):
    return np.interp(s, sol["s_b"], sol["y_b"])


def stop_line(sol, p0: float, s):
    """The exit boundary in the trader's coordinates: return (units sigma^2/Delta) against time s."""
    return y_boundary(sol, s) - logit(p0) + (0.5 - sol["theta"]) * np.asarray(s)


# ── rules that look once, or never: closed forms (Theorems 'unconditional' and 'one look') ─────────
def v_unconditional(theta, p0, h):
    """A time stop that ignores the price: exit at h whatever happened. pi is a martingale."""
    return (p0 - theta) * h


def v_one_look(theta, p0, S, h, k=None):
    """Hold to h, look once, continue to S iff the log-likelihood ratio Lambda_h exceeds k.
    Lambda_h ~ N(+h/2, h) if the edge is real, N(-h/2, h) if not. k = None: the optimal k = logit(theta) - logit(p0)."""
    if k is None:
        k = logit(theta) - logit(p0)
    sh = np.sqrt(h)
    PH, PL = norm.cdf((h / 2 - k) / sh), norm.cdf((-h / 2 - k) / sh)
    return (p0 - theta) * h + (S - h) * (p0 * (1 - theta) * PH - theta * (1 - p0) * PL)


def k_trader(theta, h):
    """The trader's rule 'exit at h unless the trade has worked' (x_h > 0) as a threshold on Lambda_h."""
    return -(0.5 - theta) * h


def v_n_looks(theta, p0, S, n, N=6401, L=16.0, q=96):
    """n evenly spaced looks (the last at the deadline), optimal thresholds by backward induction."""
    y = np.linspace(-L, L, N); pi = expit(y); dt = S / n
    z, w = np.polynomial.hermite_e.hermegauss(q); w = w / w.sum()
    def expect(f):
        up = np.interp(y[:, None] + dt / 2 + np.sqrt(dt) * z[None, :], y, f)
        dn = np.interp(y[:, None] - dt / 2 + np.sqrt(dt) * z[None, :], y, f)
        return pi * (up @ w) + (1 - pi) * (dn @ w)
    W = np.zeros(N)                                            # after the last look nothing is left
    for _ in range(n - 1, 0, -1):
        W = np.maximum(0.0, (pi - theta) * dt + expect(W))
    return float(np.interp(logit(p0), y, (pi - theta) * dt + expect(W)))


# ── simulation: the derived rule and the trader's fixed stops, on the same paths ───────────────────
def simulate(theta, p0, S, sol=None, n_paths=100_000, ds=1e-3, seed=0, h_list=()):
    """Every rule on one set of paths. Returns exit returns x_tau and exit times per rule, and the truth."""
    rng = np.random.default_rng(seed)
    M = int(round(S / ds)); ds = S / M
    real = rng.random(n_paths) < p0
    drift = np.where(real, 1 - theta, -theta)
    x = np.zeros(n_paths)
    out = {"real": real}
    alive = np.ones(n_paths, bool); xt = np.zeros(n_paths); tt = np.full(n_paths, S)
    snaps = {}
    hs = {int(round(h / ds)): h for h in h_list}
    for m in range(1, M + 1):
        x = x + drift * ds + np.sqrt(ds) * rng.standard_normal(n_paths)
        s = m * ds
        if m in hs:
            snaps[hs[m]] = x.copy()
        if sol is not None and m < M:
            y = logit(p0) + x - (0.5 - theta) * s
            #: continuity correction (Broadie–Glasserman–Kou): checking a continuous boundary only at grid
            #: times misses paths that dip and recover between checks; shifting the boundary by 0.5826 sqrt(ds)
            #: removes the leading bias. Verified against the forward equation (exit_density) to within 1 s.e.
            hit = alive & (y < y_boundary(sol, s) + BGK * np.sqrt(ds))
            xt[hit], tt[hit] = x[hit], s
            alive &= ~hit
    xt[alive] = x[alive]
    out["x_S"] = x
    if sol is not None:
        out["derived"] = (xt, tt)
    for h, xh in snaps.items():
        stay = xh > 0                                          # the trader's rule: "has it worked by h?"
        out[("trader", h)] = (np.where(stay, x, xh), np.where(stay, S, h))
        yh = logit(p0) + xh - (0.5 - theta) * h
        stay = yh > logit(theta)                               # the best single look at h
        out[("onelook", h)] = (np.where(stay, x, xh), np.where(stay, S, h))
    return out


def filter_paths(theta, p0, S, n_paths=2000, ds=1e-4, seed=1):
    """The belief two ways on the same noise: Wonham's filter d pi = pi(1-pi)(dx - (pi-theta) ds), stepped by
    Euler IN pi, against the explicit formula pi = expit(logit p0 + x - (1/2-theta) s). (Stepping in log-odds
    would be exact by construction and prove nothing.) Returns the largest gap over all paths and times."""
    rng = np.random.default_rng(seed)
    M = int(round(S / ds)); ds = S / M
    real = rng.random(n_paths) < p0
    drift = np.where(real, 1 - theta, -theta)
    x = np.zeros(n_paths); p = np.full(n_paths, p0); err = 0.0
    for m in range(1, M + 1):
        dx = drift * ds + np.sqrt(ds) * rng.standard_normal(n_paths)
        p = np.clip(p + p * (1 - p) * (dx - (p - theta) * ds), 0.0, 1.0)
        x = x + dx
        err = max(err, float(np.max(np.abs(p - expit(logit(p0) + x - (0.5 - theta) * m * ds)))))
    return err


# ── the exit-time law: the FORWARD (Fokker–Planck) equation through the HJB boundary ───────────────
def smooth_g(sol, K=4):
    """The universal curve g(u) = logit b - logit theta, fitted by sum_k a_k u^{k/2} (k = 1..K).
    The numerical boundary carries an O(dy) sawtooth; the fit sits inside it (rms ~ 7e-4) and can be differentiated."""
    u = sol["S"] - sol["s_b"]; g = sol["y_b"] - logit(sol["theta"])
    a, *_ = np.linalg.lstsq(np.stack([u ** (k / 2) for k in range(1, K + 1)], 1), g, rcond=None)
    val = lambda uu: sum(a[k - 1] * np.asarray(uu, float) ** (k / 2) for k in range(1, K + 1))
    der = lambda uu: sum(a[k - 1] * (k / 2) * np.asarray(uu, float) ** (k / 2 - 1) for k in range(1, K + 1))
    return val, der, a


def exit_density(sol, p0, real: bool, ds=1e-4, dxi=2e-3, Xi=14.0, s0=0.02):
    """Density of the exit time under one hypothesis. The log-likelihood ratio z is a Brownian motion with drift
    +1/2 (edge real) or -1/2 (no edge); the trader exits when z falls to z_b(s) = logit theta + g(S-s) - logit p0.
    In the moving frame xi = z - z_b(s): d xi = (m - z_b'(s)) ds + dB, absorbed at xi = 0. Crank–Nicolson for
        p_s = -(m - z_b') p_xi + 1/2 p_xixi,
    started from the exact Gaussian at s0 (absorption before s0 is below 1e-20). Exit density = 1/2 p_xi(s, 0)."""
    th, S = sol["theta"], sol["S"]
    gv, gd, _ = smooth_g(sol)
    zb = lambda s: logit(th) + gv(S - s) - logit(p0)
    zbd = lambda s: -gd(np.maximum(S - s, 1e-12))                # d/ds g(S-s)
    m = 0.5 if real else -0.5
    xi = np.arange(0.0, Xi + dxi / 2, dxi); n = xi.size - 2       # interior nodes
    z0 = m * s0; var = s0
    p = np.exp(-((xi + zb(s0) - z0) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var); p[0] = p[-1] = 0.0
    M = int(round((S - s0) / ds)); ds = (S - s0) / M
    s_out = [s0]; f_out = [0.5 * (4 * p[1] - p[2]) / (2 * dxi)]; Q_out = [np.trapezoid(p, xi)]
    for k in range(M):
        s_mid = s0 + (k + 0.5) * ds
        c = m - zbd(s_mid)
        lo = 0.5 / dxi**2 + c / (2 * dxi); di = -1.0 / dxi**2; up = 0.5 / dxi**2 - c / (2 * dxi)
        Ap = lo * p[:-2] + di * p[1:-1] + up * p[2:]
        rhs = p[1:-1] + 0.5 * ds * Ap
        ab = np.zeros((3, n)); ab[0, 1:] = -0.5 * ds * up; ab[1] = 1 - 0.5 * ds * di; ab[2, :-1] = -0.5 * ds * lo
        p = np.concatenate(([0.0], solve_banded((1, 1), ab, rhs), [0.0]))
        s_out.append(s0 + (k + 1) * ds)
        f_out.append(0.5 * (4 * p[1] - p[2]) / (2 * dxi))          # one-sided derivative at the absorbing wall
        Q_out.append(np.trapezoid(p, xi))
    s_out, f_out, Q_out = map(np.array, (s_out, f_out, Q_out))
    return {"s": s_out, "f": f_out, "Q": Q_out, "held": float(Q_out[-1]),
            "cut": float(np.trapezoid(f_out, s_out)), "mass_err": float(abs(np.trapezoid(f_out, s_out) + Q_out[-1] - 1))}


def exit_times_mc(sol, theta, p0, S, real: bool, n_paths=400_000, ds=5e-4, seed=41):
    """Exit times of the derived rule under one hypothesis, by simulation against the numerical boundary."""
    rng = np.random.default_rng(seed)
    M = int(round(S / ds)); ds = S / M
    x = np.zeros(n_paths); t = np.full(n_paths, np.inf); alive = np.ones(n_paths, bool)
    d = (1 - theta) if real else -theta
    for k in range(1, M + 1):
        x += d * ds + np.sqrt(ds) * rng.standard_normal(n_paths)
        if k < M:
            s = k * ds
            hit = alive & (logit(p0) + x - (0.5 - theta) * s < y_boundary(sol, s) + BGK * np.sqrt(ds))
            t[hit] = s; alive &= ~hit
    return t
