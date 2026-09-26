"""fig_holding — holding times and early-exit returns: the derived rule against the trader's stop at S/2. theta=0.3, S=1, p0=0.5."""
import numpy as np, ts, ts_plot as P

TH, S, P0 = 0.3, 1.0, 0.5
sol = ts.solve(TH, S, N=9601, ds=5e-5)
o = ts.simulate(TH, P0, S, sol, n_paths=300_000, ds=1e-3, seed=31, h_list=(0.5,))
real = o["real"]
(xd, td), (xt, tt) = o["derived"], o[("trader", 0.5)]
fig, ax = P.figure()
grid = np.linspace(0, S, 401)
for t_, lab, col in ((td, "derived rule", P.TEAL), (tt, "trader's stop at $S/2$", P.RED)):
    for mask, ls, who in ((real, "-", "edge real"), (~real, "--", "no edge")):
        cut = np.array([np.mean(t_[mask] <= g) if g < S else np.mean(t_[mask] < S) for g in grid])
        ax[0].plot(grid, 100 * cut, color=col, ls=ls, lw=2.4, label=f"{lab}: {who} ({100*cut[-1]:.0f}% cut)")
ax[0].set_xlabel(r"time since entry, $s$", fontsize=P.FS_LABEL); ax[0].set_ylabel("share of trades cut by time $s$ (%)", fontsize=P.FS_LABEL)
ax[0].legend(fontsize=10.5, loc="upper left")
xb = np.linspace(-2.5, 1.5, 61)
for x_, t_, lab, col in ((xd, td, "derived rule", P.TEAL), (xt, tt, "trader's stop at $S/2$", P.RED)):
    ax[1].hist(x_[t_ < S], bins=xb, histtype="step", lw=2.2, color=col, label=f"{lab}: {np.mean(t_ < S):.0%} exit early")
ax[1].axvline(0, color=P.NAVY, lw=0.8)
ax[1].set_xlabel(r"return at an early exit, $x_\tau$", fontsize=P.FS_LABEL); ax[1].set_ylabel("paths", fontsize=P.FS_LABEL)
ax[1].legend(fontsize=P.FS_LEGEND)
P.finish(fig, ax, "fig_holding")
