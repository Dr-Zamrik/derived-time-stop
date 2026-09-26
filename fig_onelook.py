"""fig_onelook — the value of watching: no look, one look, n looks, every instant. theta = 0.3, S = 1, p0 = 0.5."""
import numpy as np, ts, ts_plot as P

TH, S, P0 = 0.3, 1.0, 0.5
v = ts.value_at(ts.solve(TH, S, N=9601, ds=5e-5), P0)
ns = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128])
vn = np.array([ts.v_n_looks(TH, P0, S, int(n)) for n in ns])
fig, ax = P.figure()
ax[0].plot(ns, vn, "o-", color=P.NAVY, lw=2.2, ms=6, label="best rule with $n$ evenly spaced looks")
ax[0].axhline(v, color=P.TEAL, lw=2.4, label="derived rule (watching every instant)")
ax[0].axhline(ts.v_unconditional(TH, P0, S), color=P.GREY, ls="--", lw=1.4, label="no look: hold to the deadline")
ax[0].set_xscale("log", base=2); ax[0].set_xticks(ns[[0, 1, 3, 5, 7, 9, 10, 11]]); ax[0].set_xticklabels([str(n) for n in ns[[0, 1, 3, 5, 7, 9, 10, 11]]])
ax[0].set_xlabel("number of looks before the deadline", fontsize=P.FS_LABEL); ax[0].set_ylabel(r"value at entry, $v(0,p_0)$", fontsize=P.FS_LABEL)
ax[0].legend(fontsize=P.FS_LEGEND, loc="lower right")
h = np.linspace(0.01, 0.99, 197)
ax[1].plot(h, [ts.v_one_look(TH, P0, S, x) for x in h], color=P.NAVY, lw=2.4, label="one look, optimal threshold")
ax[1].plot(h, [ts.v_one_look(TH, P0, S, x, ts.k_trader(TH, x)) for x in h], color=P.RED, lw=2.4, label="one look, 'has it worked?'")
ax[1].axhline(v, color=P.TEAL, lw=2.0, label="derived rule"); ax[1].axhline(ts.v_unconditional(TH, P0, S), color=P.GREY, ls="--", lw=1.4, label="hold to the deadline")
ax[1].set_xlabel(r"day of the single look, $h$ (information units)", fontsize=P.FS_LABEL); ax[1].set_ylabel(r"value at entry", fontsize=P.FS_LABEL)
ax[1].legend(fontsize=P.FS_LEGEND, loc="lower right")
P.finish(fig, ax, "fig_onelook")
