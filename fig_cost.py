"""fig_cost — what fixed rules give up against the derived rule, over the break-even belief and the deadline. p0 = 0.5."""
import numpy as np, ts, ts_plot as P

P0 = 0.5
ths = np.linspace(0.05, 0.45, 25); Ss = np.geomspace(0.25, 4.0, 19)
loss_tr = np.zeros((Ss.size, ths.size)); loss_hold = np.zeros_like(loss_tr)
for i, S in enumerate(Ss):
    for j, th in enumerate(ths):
        v = ts.value_at(ts.solve(float(th), float(S), N=1601, ds=min(5e-4, S / 1000)), P0)
        loss_tr[i, j] = 100 * (v - ts.v_one_look(th, P0, S, S / 2, ts.k_trader(th, S / 2))) / v
        loss_hold[i, j] = 100 * (v - ts.v_unconditional(th, P0, S)) / v
fig, ax = P.figure()
for a, Z, lab in ((ax[0], loss_tr, "trader's stop at $S/2$: value given up (%)"), (ax[1], loss_hold, "holding to the deadline: value given up (%)")):
    cs = a.contourf(ths, Ss, Z, levels=16, cmap=P.HOUSE)
    cl = a.contour(ths, Ss, Z, levels=[1, 5, 10, 25, 50], colors=P.NAVY, linewidths=1.0)
    a.clabel(cl, fmt="%g%%", fontsize=10)
    a.set_yscale("log"); a.set_yticks([0.25, 0.5, 1, 2, 4]); a.set_yticklabels(["0.25", "0.5", "1", "2", "4"])
    a.set_xlabel(r"break-even belief, $\theta$", fontsize=P.FS_LABEL); a.set_ylabel(r"deadline, $S$", fontsize=P.FS_LABEL)
    cb = fig.colorbar(cs, ax=a); cb.set_label(lab, fontsize=11)
P.finish(fig, ax, "fig_cost")
