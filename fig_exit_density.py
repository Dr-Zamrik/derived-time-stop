"""fig_exit_density — when the derived rule exits: the exit-time density from the FORWARD (Fokker–Planck) equation
through the HJB boundary, edge real and no edge, with continuity-corrected simulation behind it. theta=0.3, S=1, p0=0.5."""
import numpy as np, ts, ts_plot as P

TH, S, P0 = 0.3, 1.0, 0.5
sol = ts.solve(TH, S, N=9601, ds=5e-5)
fig, ax = P.figure()
for a, real, col, lab in ((ax[0], True, P.TEAL, "edge real"), (ax[1], False, P.NAVY, "no edge")):
    r = ts.exit_density(sol, P0, real)
    t = ts.exit_times_mc(sol, TH, P0, S, real, n_paths=400_000, ds=2e-4, seed=43 if real else 44)
    fin = t[np.isfinite(t)]
    a.hist(fin, bins=np.linspace(0, S, 81), weights=np.full(fin.size, 80 / (S * t.size)), color=P.GREY, alpha=0.35,
           label=f"simulation, 400,000 paths ({100*fin.size/t.size:.1f}% cut)")
    a.plot(r["s"], r["f"], color=col, lw=2.8, label=f"Fokker–Planck density ({100*r['cut']:.1f}% cut)")
    top = 1.15 * float(np.max(r["f"][r["s"] < 0.995]))
    a.annotate(f"held to the deadline:\n{100*r['held']:.1f}% of trades", xy=(S, 0.5 * top), xytext=(0.52 * S, 0.62 * top),
               fontsize=11.5, color=col, arrowprops=dict(arrowstyle="->", color=col, lw=1.4))
    a.axvline(S, color=col, lw=1.0, ls=":")
    a.set_xlim(0, S * 1.02); a.set_ylim(0, top)
    a.set_xlabel(r"exit time since entry, $s$ (information units)", fontsize=P.FS_LABEL)
    a.set_ylabel(f"exit-time density, {lab}", fontsize=P.FS_LABEL)
    a.legend(fontsize=P.FS_LEGEND, loc="upper left")
P.finish(fig, ax, "fig_exit_density")
