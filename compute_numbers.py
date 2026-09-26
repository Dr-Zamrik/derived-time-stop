"""compute_numbers.py — every number the text of *The Derived Time Stop* states, written to numbers.json.

Python owns the numbers: no atom types a value this file did not print. Dimensionless throughout (see ts.py).
The illustrative point is theta = 0.3, S = 1, p0 = 0.5; every claim is also swept over theta and S.
"""
from __future__ import annotations
import json, time
import numpy as np
from scipy.special import expit, logit
import ts

TH, S0, P0 = 0.3, 1.0, 0.5
OUT = {"base": {"theta": TH, "S": S0, "p0": P0}}
t_start = time.time()


def stamp(k):
    print(f"  [{time.time() - t_start:6.1f}s] {k}", flush=True)


# 1 ── convergence of the boundary and the value under refinement ───────────────────────────────────
conv = []
for N in (1201, 2401, 4801, 9601):
    sol = ts.solve(TH, S0, N=N, ds=1e-4)
    conv.append({"N": N, "dy": float(sol["dy"]), "b0": float(sol["b"][0]), "v0": ts.value_at(sol, P0),
                 "worst_dip": float(np.min(np.diff(sol["b"][:-1])))})
tconv = []
for ds in (4e-4, 2e-4, 1e-4, 5e-5):
    sol = ts.solve(TH, S0, N=4801, ds=ds)
    tconv.append({"ds": ds, "b0": float(sol["b"][0]), "v0": ts.value_at(sol, P0)})
OUT["converge"] = {"space": conv, "time": tconv}
stamp("convergence")

REF = ts.solve(TH, S0, N=9601, ds=5e-5)                       # the reference solve for the base point
V0 = ts.value_at(REF, P0)
OUT["base"].update({"b0": float(REF["b"][0]), "b_half": float(np.interp(S0 / 2, REF["s_b"], REF["b"])),
                    "b_S": float(REF["b"][-1]), "v0": V0, "hold": ts.v_unconditional(TH, P0, S0),
                    "y_b0": float(REF["y_b"][0]),
                    "x_b0": float(ts.stop_line(REF, P0, 0.0)), "x_bS": float(ts.stop_line(REF, P0, S0))})

# 2 ── patience: theta - b over the life of the trade, for every deadline; b(S) = theta ─────────────
pat = {}
for S in (0.25, 0.5, 1.0, 2.0, 4.0):
    sol = ts.solve(TH, S, N=4801, ds=min(1e-4, S / 4000))
    pat[str(S)] = {"b0": float(sol["b"][0]), "b_mid": float(np.interp(S / 2, sol["s_b"], sol["b"])),
                   "b_end_minus": float(sol["b"][-2]), "b_S": float(sol["b"][-1]),
                   "patience0": TH - float(sol["b"][0]), "worst_dip": float(np.min(np.diff(sol["b"][:-1]))),
                   "v0": ts.value_at(sol, P0), "hold": ts.v_unconditional(TH, P0, S)}
OUT["patience"] = pat
stamp("patience")

# 3 ── long deadlines: the boundary at entry sinks toward zero ───────────────────────────────────────
inf = []
for S in (1.0, 4.0, 16.0, 64.0, 256.0):
    sol = ts.solve(TH, S, N=4801, L=24.0, ds=min(1e-4 * S, 2e-3))
    inf.append({"S": S, "b0": float(sol["b"][0]), "y_b0": float(sol["y_b"][0])})
Ss = np.array([r["S"] for r in inf]); b0s = np.array([r["b0"] for r in inf])
slope = float(np.polyfit(np.log(Ss[2:]), np.log(b0s[2:]), 1)[0])
OUT["long_deadline"] = {"rows": inf, "loglog_slope_S16_256": slope}
stamp("long deadlines")

# 4 ── the fan: the boundary for every (theta, S) ─────────────────────────────────────────────────────
fan = {}
for th in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
    for S in (0.25, 0.5, 1.0, 2.0, 4.0):
        sol = ts.solve(th, S, N=2401, ds=min(2e-4, S / 2000))
        fan[f"{th}|{S}"] = {"b0": float(sol["b"][0]), "v_p0": ts.value_at(sol, P0)}
OUT["fan"] = fan
stamp("fan")

# 5 ── Monte Carlo against the PDE, paired against holding (a control variate with a known mean) ─────
mc = {}
for ds in (2e-3, 1e-3, 5e-4):
    o = ts.simulate(TH, P0, S0, REF, n_paths=1_000_000, ds=ds, seed=11)
    xt, tt = o["derived"]
    d = xt - o["x_S"]                                            # paired: derived minus hold
    est = ts.v_unconditional(TH, P0, S0) + d.mean()
    mc[str(ds)] = {"raw": float(xt.mean()), "raw_se": float(xt.std() / np.sqrt(xt.size)),
                   "paired": float(est), "paired_se": float(d.std() / np.sqrt(d.size))}
OUT["mc"] = {"pde": V0, "by_ds": mc, "paths": 1_000_000}
stamp("monte carlo")

# 5b ── scale invariance: two dimensional parameterisations with the same (theta, S) ─────────────────
scale = []
for sig, dmu in ((1.0, 1.0), (0.5, 2.0)):
    T = S0 * sig**2 / dmu**2
    muL = -TH * dmu; muH = muL + dmu
    rng = np.random.default_rng(5); n = 400_000; M = 1000; dt = T / M
    real = rng.random(n) < P0; mu = np.where(real, muH, muL)
    X = np.zeros(n); Xt = np.zeros(n); alive = np.ones(n, bool)
    for m in range(1, M + 1):
        X = X + mu * dt + sig * np.sqrt(dt) * rng.standard_normal(n)
        if m < M:
            s = m * dt * dmu**2 / sig**2
            y = logit(P0) + X * dmu / sig**2 - (0.5 - TH) * s
            hit = alive & (y < ts.y_boundary(REF, s) + ts.BGK * np.sqrt(dt * dmu**2 / sig**2))
            Xt[hit] = X[hit]; alive &= ~hit
    Xt[alive] = X[alive]
    scale.append({"sigma": sig, "dmu": dmu, "T": T, "E_X": float(Xt.mean()),
                  "E_X_scaled": float(Xt.mean() * dmu / sig**2), "se_scaled": float(Xt.std() / np.sqrt(n) * dmu / sig**2)})
OUT["scale"] = scale
stamp("scale invariance")

# 6 ── the filter, path by path ────────────────────────────────────────────────────────────────────────
OUT["filter"] = [{"ds": ds, "max_gap": ts.filter_paths(TH, P0, S0, n_paths=2000, ds=ds)} for ds in (1e-2, 1e-3, 1e-4)]
stamp("filter")

# 7 ── the stop line: signs of the average drift, and conviction at entry ──────────────────────────────
signs = {}
SIGN_S = 4.0
for th in (0.3, 0.5, 0.7):
    sol = ts.solve(th, SIGN_S, N=9601, ds=1e-4)
    s = np.linspace(0, SIGN_S, 201); xb = ts.stop_line(sol, 0.8, s)
    signs[str(th)] = {"x0": float(xb[0]), "x_half": float(xb[100]), "xS": float(xb[-1]),
                      "min_x": float(xb.min()), "s_at_min": float(s[int(np.argmin(xb))]), "b0": float(sol["b"][0])}
OUT["signs"] = {"p0": 0.8, "S": SIGN_S, "rows": signs}
conv_rows = {}
for p in (0.3, 0.4, 0.5, 0.6, 0.7):
    conv_rows[str(p)] = {"x_b0": float(ts.stop_line(REF, p, 0.0)), "logit_p0": float(logit(p)),
                         "enters": bool(p > REF["b"][0]), "v0": ts.value_at(REF, p)}
OUT["conviction"] = conv_rows
stamp("stop lines")

# 8 ── looking: never, once, n times ──────────────────────────────────────────────────────────────────
hs = [S0 / 8, S0 / 4, S0 / 2, 3 * S0 / 4]
o = ts.simulate(TH, P0, S0, REF, n_paths=1_000_000, ds=1e-3, seed=21, h_list=hs)
one = []
for h in hs:
    a = o[("onelook", h)][0]; b_ = o[("trader", h)][0]
    one.append({"h": h, "closed_opt": float(ts.v_one_look(TH, P0, S0, h)),
                "mc_opt": float(ts.v_unconditional(TH, P0, S0) + (a - o["x_S"]).mean()),
                "mc_opt_se": float((a - o["x_S"]).std() / np.sqrt(a.size)),
                "closed_trader": float(ts.v_one_look(TH, P0, S0, h, ts.k_trader(TH, h))),
                "mc_trader": float(ts.v_unconditional(TH, P0, S0) + (b_ - o["x_S"]).mean()),
                "mc_trader_se": float((b_ - o["x_S"]).std() / np.sqrt(b_.size))})
hgrid = np.linspace(0.01, S0 - 0.01, 99)
vals = [ts.v_one_look(TH, P0, S0, h) for h in hgrid]
OUT["one_look"] = {"rows": one, "h_star": float(hgrid[int(np.argmax(vals))]), "v_star": float(max(vals))}
OUT["n_looks"] = [{"n": n, "v": ts.v_n_looks(TH, P0, S0, n)} for n in (1, 2, 3, 4, 8, 16, 32, 64, 128)]
stamp("looks")

# 9 ── what the trader's fixed stops cost: the base table, then the whole (theta, S) plane ─────────────
xt, tt = o["derived"]; real = o["real"]
rows = {"derived": (xt, tt), "hold": (o["x_S"], np.full(xt.size, S0))}
for h in (S0 / 4, S0 / 2, 3 * S0 / 4):
    rows[f"trader {h:g}"] = o[("trader", h)]
tab = []
for name, (x_, t_) in rows.items():
    tab.append({"rule": name, "value": float(ts.v_unconditional(TH, P0, S0) + (x_ - o["x_S"]).mean()),
                "se": float((x_ - o["x_S"]).std() / np.sqrt(x_.size)), "mean_hold": float(t_.mean()),
                "cut_real": float(np.mean(t_[real] < S0)), "held_false": float(np.mean(t_[~real] >= S0))})
OUT["table"] = tab
plane = []
for th in np.round(np.linspace(0.05, 0.45, 9), 3):
    for S in (0.25, 0.5, 1.0, 2.0, 4.0):
        sol = ts.solve(float(th), S, N=2401, ds=min(2e-4, S / 2000))
        v = ts.value_at(sol, P0)
        plane.append({"theta": float(th), "S": S, "v": v, "hold": ts.v_unconditional(th, P0, S),
                      **{f"trader_{f}": float(ts.v_one_look(th, P0, S, f * S, ts.k_trader(th, f * S))) for f in (0.25, 0.5, 0.75)}})
OUT["plane"] = plane
stamp("fixed stops")

# 10 ── holding times and exit returns (for the figure and the text) ───────────────────────────────────
OUT["holding"] = {name: {"median_t": float(np.median(t_)), "p_exit_before_S": float(np.mean(t_ < S0)),
                         "mean_x_exit_early": float(x_[t_ < S0].mean()) if np.any(t_ < S0) else None}
                  for name, (x_, t_) in rows.items()}

json.dump(OUT, open("numbers.json", "w"), indent=1)
stamp("wrote numbers.json")

# 11 ── the universal curve g(tau) = logit b - logit theta: the collapse across theta, its table, its sqrt shape ──
uni = {"collapse": []}
for N in (2401, 4801, 9601):
    Gs = []
    for th in (0.1, 0.3, 0.5, 0.7, 0.9):
        sol = ts.solve(th, 4.0, N=N, ds=1e-4)
        tau = np.linspace(0.05, 3.95, 79)
        Gs.append(np.interp(4.0 - tau, sol["s_b"], sol["y_b"]) - logit(th))
    Gs = np.array(Gs)
    uni["collapse"].append({"N": N, "dy": 36.0 / (N - 1), "max_spread": float(np.max(Gs.max(0) - Gs.min(0)))})
sol = ts.solve(0.5, 16.0, N=9601, L=24.0, ds=2e-4)
taus = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
uni["table"] = [{"tau": t, "g": float(np.interp(16.0 - t, sol["s_b"], sol["y_b"])),
                 "g_over_sqrt": float(np.interp(16.0 - t, sol["s_b"], sol["y_b"]) / np.sqrt(t))} for t in taus]
OUT["universal"] = uni
json.dump(OUT, open("numbers.json", "w"), indent=1)
stamp("universal curve; wrote numbers.json again")

# 12 ── the exit-time law: forward (Fokker–Planck) equation through the HJB boundary, against simulation ──────
ex = {}
for real in (True, False):
    r = ts.exit_density(REF, P0, real)
    t_mc = ts.exit_times_mc(REF, TH, P0, S0, real, n_paths=400_000, ds=5e-4)
    cut = float(np.mean(np.isfinite(t_mc)))
    bands = [(0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.95)]
    ex["real" if real else "no_edge"] = {
        "fp_cut": r["cut"], "fp_held": r["held"], "mass_err": r["mass_err"],
        "mc_cut": cut, "mc_se": float(np.sqrt(cut * (1 - cut) / t_mc.size)),
        "median_exit_fp": float(np.interp(0.5 * r["cut"], np.cumsum(r["f"]) * (r["s"][1] - r["s"][0]), r["s"])),
        "mode_fp": float(r["s"][int(np.argmax(r["f"][r["s"] < 0.98]))]),
        "bands": [{"a": a, "b": b, "fp": float(np.trapezoid(r["f"][(r["s"] >= a) & (r["s"] < b)], r["s"][(r["s"] >= a) & (r["s"] < b)])),
                   "mc": float(np.mean((t_mc >= a) & (t_mc < b)))} for a, b in bands]}
gv, gd, coef = ts.smooth_g(REF)
ex["g_fit"] = {"coef": [float(c) for c in coef], "rms": float(np.sqrt(np.mean((gv(REF["S"] - REF["s_b"]) - (REF["y_b"] - logit(TH)))**2)))}
OUT["exit_law"] = ex
json.dump(OUT, open("numbers.json", "w"), indent=1)
stamp("exit law; wrote numbers.json again")
