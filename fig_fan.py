"""fig_fan — the exit boundary for five deadlines (theta = 0.3) and for six break-even beliefs (against time left)."""
import numpy as np, ts, ts_plot as P

fig, ax = P.figure()
Ss = (0.25, 0.5, 1.0, 2.0, 4.0)
for k, S in enumerate(Ss):
    sol = ts.solve(0.3, S, N=9601, ds=min(1e-4, S / 4000))
    ax[0].plot(sol["s_b"], sol["b"], color=P.blend(k, len(Ss)), lw=2.4, label=f"$S={S:g}$")
ax[0].axhline(0.3, color=P.GREY, ls=":", lw=1.2)
ax[0].set_xlabel(r"time since entry, $s$", fontsize=P.FS_LABEL); ax[0].set_ylabel(r"exit boundary $b(s)$", fontsize=P.FS_LABEL)
ax[0].legend(fontsize=P.FS_LEGEND, title="deadline", title_fontsize=P.FS_LEGEND, loc="lower right")
ths = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
for k, th in enumerate(ths):
    sol = ts.solve(th, 4.0, N=9601, ds=1e-4)
    tau = 4.0 - sol["s_b"]
    ax[1].plot(tau, sol["b"], color=P.blend(k, len(ths)), lw=2.4, label=rf"$\theta={th:g}$")
    ax[1].plot([0], [th], "o", color=P.blend(k, len(ths)), ms=6)
ax[1].set_xlabel(r"time left to the deadline, $S-s$", fontsize=P.FS_LABEL); ax[1].set_ylabel(r"exit boundary", fontsize=P.FS_LABEL)
ax[1].legend(fontsize=P.FS_LEGEND, ncol=2, loc="upper right")
P.finish(fig, ax, "fig_fan")
