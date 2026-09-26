"""fig_price — the exit boundary as a stop line on the return chart, with simulated trades. theta=0.3, S=1, p0=0.5."""
import numpy as np, ts, ts_plot as P
from scipy.special import logit

TH, S, P0 = 0.3, 1.0, 0.5
sol = ts.solve(TH, S, N=9601, ds=5e-5)
s = np.linspace(0, S, 1001); xb = ts.stop_line(sol, P0, s)
rng = np.random.default_rng(7)
fig, ax = P.figure()
for a, (real, col, lab) in zip(ax, ((True, P.TEAL, "edge real"), (False, P.GREY, "no edge"))):
    for i in range(14):
        x = np.r_[0, np.cumsum(((1 - TH) if real else -TH) * (S / 1000) + np.sqrt(S / 1000) * rng.standard_normal(1000))]
        below = np.where(x[1:-1] < xb[1:-1])[0]
        if below.size:
            j = below[0] + 1
            a.plot(s[:j + 1], x[:j + 1], color=col, lw=1.3, alpha=0.9); a.plot(s[j], x[j], "x", color=P.RED, ms=8, mew=2)
        else:
            a.plot(s, x, color=col, lw=1.3, alpha=0.9)
    a.plot(s, xb, color=P.NAVY, lw=3, label="derived stop line")
    a.axhline(0, color="#bbbbbb", lw=0.8)
    a.set_xlabel(r"time since entry, $s$", fontsize=P.FS_LABEL)
    a.set_ylabel(r"return since entry, $x$ (units of $\sigma^2/\Delta$)", fontsize=P.FS_LABEL)
    a.plot([], [], color=col, lw=1.3, label=f"paths, {lab}"); a.plot([], [], "x", color=P.RED, label="exit")
    a.legend(fontsize=P.FS_LEGEND, loc="upper left"); a.set_ylim(-2.2, 2.6)
P.finish(fig, ax, "fig_price")
