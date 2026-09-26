"""fig_value — the value of holding on over (time, belief), with the exit boundary. theta = 0.3, S = 1."""
import numpy as np, ts, ts_plot as P
from scipy.special import expit

TH, S = 0.3, 1.0
sol = ts.solve(TH, S, N=9601, ds=5e-5, keep=201)
fig, ax = P.figure()
Pi = sol["pi"]; m = (Pi > 0.002) & (Pi < 0.998)
cs = ax[0].contourf(sol["s_snap"], Pi[m], sol["V"][:, m].T, levels=20, cmap=P.HOUSE)
cb = fig.colorbar(cs, ax=ax[0]); cb.set_label(r"value of holding on, $v(s,\pi)$", fontsize=P.FS_LABEL)
ax[0].plot(sol["s_b"], sol["b"], color=P.SAND, lw=3, label=r"exit boundary $b(s)$")
ax[0].axhline(TH, color=P.NAVY, ls=":", lw=1.4, label=r"break-even belief $\theta$")
ax[0].set_xlabel(r"time since entry, $s$ (information units)", fontsize=P.FS_LABEL)
ax[0].set_ylabel(r"belief that the edge is real, $\pi$", fontsize=P.FS_LABEL)
ax[0].text(0.05, 0.08, "exit", color=P.NAVY, fontsize=13, fontweight="bold")
ax[0].text(0.05, 0.62, "hold", color="white", fontsize=13, fontweight="bold")
ax[0].legend(loc="upper right", fontsize=P.FS_LEGEND, framealpha=0.95)
for k, (s, c) in enumerate(((0.0, P.NAVY), (0.5, P.TEAL), (0.9, P.GREY))):
    j = int(np.argmin(abs(sol["s_snap"] - s)))
    ax[1].plot(Pi, sol["V"][j], color=c, lw=2.4, label=f"$s={s:g}$")
    bb = float(np.interp(s, sol["s_b"], sol["b"]))
    ax[1].plot([bb], [0], "o", color=c, ms=8)
ax[1].plot(Pi, np.maximum(Pi - TH, 0) * S, color=P.NAVY, ls="--", lw=1.2, label=r"myopic $(\pi-\theta)^+ S$")
ax[1].set_xlim(0, 0.8); ax[1].set_ylim(-0.01, 0.4)
ax[1].set_xlabel(r"belief, $\pi$", fontsize=P.FS_LABEL); ax[1].set_ylabel(r"$v(s,\pi)$", fontsize=P.FS_LABEL)
ax[1].legend(fontsize=P.FS_LEGEND, loc="upper left")
P.finish(fig, ax, "fig_value")
