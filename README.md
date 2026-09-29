# The Derived Time Stop — the code behind the paper

<!-- @@ZAMRIK-LINKS:BEGIN@@ -->
The paper this code belongs to: <https://zamrik.com/research-items/wp-2026-80309110/>
PDF: <https://zamrik.com/wp-content/uploads/research/derived-time-stop.pdf>
Zenodo: <https://zenodo.org/records/22972016> · DOI 10.5281/zenodo.22972015
Disclaimer: research and education, not advice: <https://zamrik.com/disclaimer/>

`WP-2026-80309110` · T. Zamrik
<!-- @@ZAMRIK-LINKS:END@@ -->

Everything that produces the numbers and figures in **The Derived Time Stop: Exiting a
Swing Trade under Drift Uncertainty and a Deadline** (`WP-2026-80309110`), published at
<https://zamrik.com/research-items/wp-2026-80309110/>.

A swing trade is a hypothesis with a deadline. The price drifts at one of two constant
rates, `mu_H > 0` if the setup's edge is real and `mu_L < 0` if it is not, and the trader
does not know which. The setup expires at a known date. The trader's belief that the edge
is real is filtered from the return alone, holding on earns the expected drift under that
belief, and when to close becomes a finite-horizon optimal stopping problem with two
parameters: the break-even belief `theta = -mu_L / (mu_H - mu_L)` and the deadline `S`.

Two partial differential equations answer it, and `ts.py` solves both:

- **Hamilton–Jacobi–Bellman**, backward from the deadline. A variational inequality (an
  obstacle problem) for the value of holding on. Its free boundary `b(s)` is the exit
  rule: *where* to get out.
- **Fokker–Planck**, forward from entry, once with the edge real and once with no edge,
  with `b(s)` as an absorbing wall. The flux through the wall is the law of the exit
  time: *when* trades are actually cut.

Everything is dimensionless. With `Delta = mu_H - mu_L` and volatility `sigma`, time is
measured in units of `(sigma / Delta)^2` and return in units of `sigma^2 / Delta`, so no
market number is assumed anywhere.

## Files

| file | what it does |
|---|---|
| `ts.py` | the one solver: the belief filter, the HJB obstacle problem (a projected implicit scheme in log-odds), the forward equations with an absorbing boundary, the one-look and n-look values, and the simulator |
| `compute_numbers.py` | every number quoted in the paper; writes `numbers.json` |
| `compute_forward.py` | the forward-equation section: the ratio identity, the exit return, and estimating the share of real edges from exit data |
| `compute_bias.py` | the simulation's monitoring bias, with and without the continuity correction |
| `fig_value.py` | the value of holding on over time and belief, with the exit boundary |
| `fig_fan.py` | the boundary for five deadlines and for six break-even beliefs |
| `fig_universal.py` | every boundary collapses onto one curve of the time left |
| `fig_price.py` | the boundary as a stop line on the return chart, with simulated trades |
| `fig_signs.py` | the stop line when the average drift is positive, zero and negative |
| `fig_exit_density.py` | the exit-time densities from the forward equations, edge real and no edge, with simulation behind them |
| `fig_holding.py` | holding times and early-exit returns: the derived rule against the trader's stop at `S/2` |
| `fig_onelook.py` | the value of watching: no look, one look, n looks, every instant |
| `fig_cost.py` | what fixed rules give up against the derived rule, over `theta` and `S` |
| `ts_plot.py` | house plotting defaults |

## Reproducing

```
python3 compute_numbers.py     # ~220 s, writes numbers.json
python3 compute_forward.py     # adds "forward" to numbers.json
python3 compute_bias.py        # adds "exit_law_bias" to numbers.json
for f in fig_*.py; do python3 "$f"; done     # each figure script writes its own .png
```

Requires `numpy`, `scipy` and `matplotlib`. The two `compute_` scripts after the first
extend its `numbers.json`, so run them in this order. The figure scripts call `ts.py`
directly and run in any order. Every figure in the paper is regenerated from these scripts
on every build of the paper; none is stored and reused.

## What it should reproduce

At `theta = 0.3`, `S = 1`, `p0 = 0.5`:

| quantity | equation | independent check |
|---|---|---|
| value of holding on at entry, `v(0, p0)` | 0.207671 (HJB) | 0.20752 ± 0.00032, a million simulated paths |
| holding to the deadline, `(p0 - theta) S` | 0.2, exact | the derived rule adds 0.0077 |
| exit boundary at entry, `b(0)` | 0.1862 | rises to `theta = 0.3` exactly at the deadline |
| real edges cut before the deadline | 0.1379 (Fokker–Planck) | 0.1379 ± 0.0005, simulation |
| dead trades cut before the deadline | 0.4474 (Fokker–Planck) | 0.4471 ± 0.0008, simulation |
| `f_H(s) / f_L(s)` against `exp(z_b(s))` | equal (the ratio identity) | largest relative gap 1.7 × 10⁻⁵ |
| value with one look at `S/2` | 0.20437, closed form | 0.20435 ± 0.00027, simulation |
| `g(u) / sqrt(u)`, time left `u` from 0.01 to 4 | between −0.63 and −0.58 | the collapse tightens with the grid: spread 0.0073, 0.0043, 0.0030 |

The ratio identity says that a trade cut at time `s` had a real edge with probability
exactly `b(s)`, the boundary it was cut on.

## Where the grid shows

**A simulated path that is checked only at grid times cuts too few trades.** It misses the
crossings that happen between steps. At step `5e-4` the raw simulation cuts 0.1346 of
real edges against the forward equation's 0.1379. Moving the stop `0.5826 * sqrt(ds)`
closer to the path (the continuity correction of Broadie, Glasserman and Kou) gives
0.1379 ± 0.0005. The correction is *derived*, not fitted, and every simulation here uses
it; `compute_bias.py` prints both.

**The boundary carries the grid; the value does not.** The value at entry is converged to
seven digits. The computed boundary shows small downward steps as it moves between nodes,
of 4.1e-3, 2.2e-3, 9.6e-4 and 5.5e-4 as the log-odds grid doubles from 1201 to 9601
nodes. They halve with the grid spacing, which is the signature of a discretisation error
and not of a real turn: the boundary is monotone, as the paper proves, to within them.
