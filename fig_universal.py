"""fig_universal — every exit boundary is one curve shifted in log-odds: g(tau) = logit b - logit theta, tau = time left."""
import numpy as np, ts, ts_plot as P
from scipy.special import logit

S = 4.0
tau = np.linspace(0.0, S, 401)
fig, ax = P.figure()
ths = (0.1, 0.3, 0.5, 0.7, 0.9)
for k, th in enumerate(ths):
    sol = ts.solve(th, S, N=9601, ds=1e-4)
    yb = np.interp(S - tau, sol["s_b"], sol["y_b"])
    ax[0].plot(tau, yb, color=P.blend(k, len(ths)), lw=2.2, label=rf"$\theta={th}$")
    ax[1].plot(tau, yb - logit(th), color=P.blend(k, len(ths)), lw=[5, 4, 3, 2, 1.2][k], alpha=0.85, label=rf"$\theta={th}$")
ax[0].set_xlabel(r"time left to the deadline, $u=S-s$", fontsize=P.FS_LABEL); ax[0].set_ylabel(r"exit boundary in log-odds, $\mathrm{logit}\,b$", fontsize=P.FS_LABEL)
ax[0].legend(fontsize=P.FS_LEGEND, ncol=2)
ax[1].plot(tau, -0.61 * np.sqrt(tau), color=P.RED, ls="--", lw=1.4, label=r"$-0.61\sqrt{u}$ for scale")
ax[1].set_xlabel(r"time left to the deadline, $u$", fontsize=P.FS_LABEL); ax[1].set_ylabel(r"$g(u)=\mathrm{logit}\,b-\mathrm{logit}\,\theta$", fontsize=P.FS_LABEL)
ax[1].legend(fontsize=P.FS_LEGEND, ncol=2, loc="lower left")
P.finish(fig, ax, "fig_universal")
