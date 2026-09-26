"""compute_forward.py — the forward-equation section's numbers, added to numbers.json under "forward".
  (a) the ratio identity f_H(s) = exp(z_b(s)) f_L(s), i.e. P(edge real | exit at s) = b(s): forward equation and simulation
  (b) the exit return is on the stop line: its range
  (c) estimating the share of setups with a real edge from exit data: Fisher information with and without timing,
      and a simulation of the maximum-likelihood estimator. theta=0.3, S=1, p0=0.5 (the rule's prior)."""
import json, numpy as np, ts
from scipy.special import logit, expit
from scipy.optimize import brentq

TH, S, P0 = 0.3, 1.0, 0.5
sol = ts.solve(TH, S, N=9601, ds=5e-5)
FH, FL = ts.exit_density(sol, P0, True), ts.exit_density(sol, P0, False)
s = FH["s"]; fH, fL = FH["f"], FL["f"]
gv, gd, _ = ts.smooth_g(sol)
zb = logit(TH) + gv(S - s) - logit(P0)
out = {}
# (a) the identity, where both densities are well resolved
m = (s > 0.25) & (s < 0.99)
rel = np.abs(fH[m] / fL[m] * np.exp(-zb[m]) - 1)
out["ratio"] = {"max_rel_err": float(rel.max()), "median_rel_err": float(np.median(rel)),
                "at": [{"s": float(x), "fH": float(np.interp(x, s, fH)), "fL": float(np.interp(x, s, fL)),
                        "ratio": float(np.interp(x, s, fH) / np.interp(x, s, fL)), "exp_zb": float(np.exp(np.interp(x, s, zb))),
                        "b": float(np.interp(x, sol["s_b"], sol["b"]))} for x in (0.3, 0.5, 0.7, 0.9)],
                "ratio_range": [float(np.exp(zb[0])), float(np.exp(logit(TH) - logit(P0)))]}
# the same identity from simulation: among trades cut near s, the share that were real
tH = ts.exit_times_mc(sol, TH, P0, S, True, n_paths=400_000, ds=5e-4, seed=51)
tL = ts.exit_times_mc(sol, TH, P0, S, False, n_paths=400_000, ds=5e-4, seed=52)
win = []
for a, b_ in ((0.3, 0.5), (0.5, 0.7), (0.7, 0.9)):
    nH, nL = np.sum((tH >= a) & (tH < b_)), np.sum((tL >= a) & (tL < b_))
    share = nH / (nH + nL)                                     # equal numbers of each: p = 1/2 = P0
    win.append({"a": a, "b": b_, "share_real": float(share), "se": float(np.sqrt(share * (1 - share) / (nH + nL))),
                "b_mid": float(np.mean(np.interp(np.linspace(a, b_, 50), sol["s_b"], sol["b"])))})
out["ratio_mc"] = win
# (b) the exit return is the stop line at the exit time
xb = ts.stop_line(sol, P0, s)
out["exit_return_range"] = [float(xb[s > 0.1].min()), float(xb.max())]
# (c) the hit-rate estimator: exit data = a cut at time s (density p fH + (1-p) fL) or survival (mass p QH + (1-p) QL)
cH, cL = FH["cut"], FL["cut"]; QH, QL = FH["held"], FL["held"]
def fisher(p):
    mix = p * fH + (1 - p) * fL
    ok = mix > 1e-12
    return float(np.trapezoid(np.where(ok, (fH - fL) ** 2 / np.where(ok, mix, 1), 0), s) + (QH - QL) ** 2 / (p * QH + (1 - p) * QL))
def fisher_cut(p):
    c = p * cH + (1 - p) * cL
    return float((cH - cL) ** 2 / (c * (1 - c)))
out["hit_rate"] = {"cH": cH, "cL": cL, "rows": [{"p": p, "I_full": fisher(p), "I_cut": fisher_cut(p),
                                                    "gain_pct": 100 * (fisher(p) / fisher_cut(p) - 1),
                                                    "se_100_full": 1 / np.sqrt(100 * fisher(p)), "se_100_cut": 1 / np.sqrt(100 * fisher_cut(p))}
                                                   for p in (0.3, 0.5, 0.7)]}
# simulate the MLE on books of 100 trades drawn from the two simulated populations, true share 0.5
rng = np.random.default_rng(61); est_full, est_cut = [], []
fHi = lambda t: np.interp(t, s, fH); fLi = lambda t: np.interp(t, s, fL)
for _ in range(4000):
    nreal = rng.binomial(100, 0.5)
    t = np.concatenate([rng.choice(tH, nreal), rng.choice(tL, 100 - nreal)])
    cut = np.isfinite(t); tc = t[cut]; nsurv = np.sum(~cut)
    a_, b2 = fHi(tc), fLi(tc)
    score = lambda p: np.sum((a_ - b2) / (p * a_ + (1 - p) * b2)) + nsurv * (QH - QL) / (p * QH + (1 - p) * QL)
    est_full.append(brentq(score, 1e-6, 1 - 1e-6) if score(1e-6) > 0 and score(1 - 1e-6) < 0 else (0.0 if score(1e-6) <= 0 else 1.0))
    est_cut.append(float(np.clip((cL - cut.mean()) / (cL - cH), 0, 1)))
out["hit_rate"]["mc_100"] = {"true_p": 0.5, "reps": 4000, "sd_full": float(np.std(est_full)), "mean_full": float(np.mean(est_full)),
                             "sd_cut": float(np.std(est_cut)), "mean_cut": float(np.mean(est_cut))}
n = json.load(open("numbers.json")); n["forward"] = out; json.dump(n, open("numbers.json", "w"), indent=1)
print(json.dumps(out, indent=1)[:3000])
