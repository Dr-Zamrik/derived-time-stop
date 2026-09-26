"""compute_bias.py — the simulation's grid-monitoring bias, with and without the continuity correction, against the
forward equation (Proposition 6). Adds `exit_law_bias` to numbers.json. theta=0.3, S=1, p0=0.5, edge real."""
import json, numpy as np, ts

sol = ts.solve(0.3, 1.0, N=9601, ds=5e-5)
fp = ts.exit_density(sol, 0.5, True)["cut"]
out = {"fp_cut_real": fp, "rows": []}
for ds in (5e-4, 1e-4):
    for corr in (False, True):
        keep = ts.BGK
        ts.BGK = keep if corr else 0.0
        t = ts.exit_times_mc(sol, 0.3, 0.5, 1.0, True, n_paths=400_000, ds=ds, seed=41)
        ts.BGK = keep
        c = float(np.mean(np.isfinite(t)))
        out["rows"].append({"ds": ds, "corrected": corr, "cut": c, "se": float(np.sqrt(c * (1 - c) / t.size))})
        print(f"  ds={ds:g} corrected={corr}: {100*c:.2f}% ± {100*out['rows'][-1]['se']:.2f}   (forward equation {100*fp:.2f}%)")
n = json.load(open("numbers.json")); n["exit_law_bias"] = out; json.dump(n, open("numbers.json", "w"), indent=1)
