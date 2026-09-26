"""fig_signs — the stop line when the average drift is positive (theta<1/2), zero and negative (theta>1/2). p0 = 0.8, S = 4."""
import numpy as np, ts, ts_plot as P

P0, S = 0.8, 4.0
s = np.linspace(0, S, 401)
fig, ax = P.figure()
for k, (th, lab) in enumerate(((0.3, "positive"), (0.5, "zero"), (0.7, "negative"))):
    sol = ts.solve(th, S, N=9601, ds=1e-4)
    col = (P.TEAL, P.GREY, P.NAVY)[k]
    ax[0].plot(s, ts.stop_line(sol, P0, s), color=col, lw=2.6, label=rf"$\theta={th}$: average drift {lab}")
    if th == 0.5:                                   # the log-odds part is the same for every theta (the universal curve)
        ax[1].plot(s, ts.y_boundary(sol, s) - ts.y_boundary(sol, 0.0), color=P.NAVY, lw=2.8, ls="--",
                   label="log-odds part, the same for every $\\theta$")
    ax[1].plot(s, (0.5 - th) * s, color=col, lw=1.8, label=rf"drift part $(\frac{{1}}{{2}}-\theta)s$, $\theta={th}$")
ax[0].set_xlabel(r"time since entry, $s$", fontsize=P.FS_LABEL); ax[0].set_ylabel(r"stop line $x_b(s)$", fontsize=P.FS_LABEL)
ax[0].legend(fontsize=P.FS_LEGEND)
ax[1].axhline(0, color="#bbbbbb", lw=0.8)
ax[1].set_xlabel(r"time since entry, $s$", fontsize=P.FS_LABEL); ax[1].set_ylabel("the two parts of the stop line's move", fontsize=P.FS_LABEL)
ax[1].legend(fontsize=10.5, loc="upper left")
P.finish(fig, ax, "fig_signs")
