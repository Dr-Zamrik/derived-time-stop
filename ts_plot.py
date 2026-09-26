"""ts_plot.py — the house figure style for *The Derived Time Stop* (conventions/FIGURE_PIPELINE.md).
White face, navy #1e3a5f primary, teal #2a9d8f accent, no title inside a figure (the chrome carries it)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

NAVY, TEAL, GREY, SAND, RED = "#1e3a5f", "#2a9d8f", "#8a94a6", "#e8b04a", "#b5452f"
FIGSIZE, DPI = (12.4, 6.4), 170
FS_LABEL, FS_LEGEND, FS_TICK = 12.5, 11.0, 11
#: value maps read from white (nothing to gain) through teal to navy (most to gain)
HOUSE = LinearSegmentedColormap.from_list("house", ["#ffffff", "#bfe3de", TEAL, NAVY])


def figure(ncols=2):
    fig, ax = plt.subplots(1, ncols, figsize=FIGSIZE, facecolor="white")
    return fig, ax


def finish(fig, axes, name):
    for a in (axes if hasattr(axes, "__iter__") else [axes]):
        a.tick_params(labelsize=FS_TICK)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout(pad=1.4)
    fig.savefig(f"{name}.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def blend(k, n):
    """k-th of n colours from navy to teal: an ordered family (longer deadline, higher theta …)."""
    from matplotlib.colors import to_rgb
    a, b = to_rgb(NAVY), to_rgb(TEAL)
    t = 0 if n == 1 else k / (n - 1)
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))
