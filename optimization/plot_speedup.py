#!/usr/bin/env python3
"""
plot_speedup.py
===============
Reads timing_results.csv produced by run_benchmark.sh and generates two
publication-quality figures:

  speedup_plot.png  —  Speed-up of each variant relative to the baseline
                        at the same rank count.  One line per variant.
  walltime_plot.png —  Raw wall-clock time bar chart grouped by rank count.
                        One bar per variant per rank count.

Usage:
    python3 plot_speedup.py [timing_results.csv]

Requirements:
    pip install matplotlib pandas numpy
"""

import sys
import os
import csv
import collections
import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CSV_FILE   = sys.argv[1] if len(sys.argv) > 1 else "timing_results.csv"
OUT_SPEED  = "speedup_plot.png"
OUT_TIME   = "walltime_plot.png"

VARIANT_LABELS = {
    "baseline": "Baseline (no opts)",
    "opt1":     "Opt1: comm/compute overlap",
    "opt2":     "Opt2: Opt1 + OpenMP",
    "opt3":     "Opt3: Opt2 + raw-ptr + amortised residual",
}

VARIANT_ORDER  = ["baseline", "opt1", "opt2", "opt3"]
VARIANT_COLORS = ["#6c757d", "#0d6efd", "#198754", "#dc3545"]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
if not os.path.exists(CSV_FILE):
    print(f"ERROR: {CSV_FILE} not found.  Run run_benchmark.sh first.")
    sys.exit(1)

# data[ranks][variant] = wall_time (average if multiple runs)
data = collections.defaultdict(lambda: collections.defaultdict(list))

with open(CSV_FILE) as f:
    reader = csv.DictReader(f)
    for row in reader:
        ranks   = int(row["ranks"])
        variant = row["variant"].strip()
        t       = float(row["wall_time_s"])
        data[ranks][variant].append(t)

# Average over any repeated runs
avg = {}   # avg[ranks][variant] = mean wall time
for ranks, vdict in data.items():
    avg[ranks] = {}
    for v, times in vdict.items():
        avg[ranks][v] = np.mean(times)

all_ranks    = sorted(avg.keys())
all_variants = [v for v in VARIANT_ORDER if v in
                set(v for vd in avg.values() for v in vd)]

print(f"Loaded data for ranks: {all_ranks}")
print(f"Variants found: {all_variants}")

# ---------------------------------------------------------------------------
# Figure 1 — Speed-up relative to baseline at same rank count
# ---------------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(8, 5))

for vi, variant in enumerate(all_variants):
    speedups = []
    valid_ranks = []
    for r in all_ranks:
        if variant in avg[r] and "baseline" in avg[r] and avg[r]["baseline"] > 0:
            speedups.append(avg[r]["baseline"] / avg[r][variant])
            valid_ranks.append(r)

    if speedups:
        ax1.plot(
            valid_ranks, speedups,
            marker="o", linewidth=2.0, markersize=7,
            color=VARIANT_COLORS[vi],
            label=VARIANT_LABELS.get(variant, variant),
        )

# Baseline reference line at 1.0
ax1.axhline(1.0, color="#6c757d", linestyle="--", linewidth=1.2, label="Baseline (1.0×)")

ax1.set_xlabel("Number of MPI Ranks", fontsize=12)
ax1.set_ylabel("Speed-up  (×  relative to baseline)", fontsize=12)
ax1.set_title("Optimisation Speed-up vs Baseline", fontsize=14, fontweight="bold")
ax1.set_xticks(all_ranks)
ax1.set_xticklabels([str(r) for r in all_ranks])
ax1.legend(fontsize=9, loc="upper left")
ax1.grid(True, linestyle=":", alpha=0.5)
ax1.set_ylim(bottom=0)

fig1.tight_layout()
fig1.savefig(OUT_SPEED, dpi=150)
print(f"Saved: {OUT_SPEED}")

# ---------------------------------------------------------------------------
# Figure 2 — Raw wall-clock time grouped bar chart
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(9, 5))

n_variants  = len(all_variants)
n_ranks     = len(all_ranks)
bar_width   = 0.18
group_gap   = 0.1
x_positions = np.arange(n_ranks) * (n_variants * bar_width + group_gap)

for vi, variant in enumerate(all_variants):
    times = [avg[r].get(variant, 0.0) for r in all_ranks]
    offsets = x_positions + vi * bar_width
    bars = ax2.bar(
        offsets, times,
        width=bar_width,
        color=VARIANT_COLORS[vi],
        label=VARIANT_LABELS.get(variant, variant),
        edgecolor="white",
        linewidth=0.5,
    )
    # Annotate bar tops with time in seconds
    for bar, t in zip(bars, times):
        if t > 0:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002 * max(
                    avg[r].get(v, 0) for r in all_ranks for v in all_variants
                ),
                f"{t:.2f}s",
                ha="center", va="bottom", fontsize=6.5, rotation=90,
            )

# Group tick labels centred on each rank group
group_centres = x_positions + (n_variants - 1) * bar_width / 2
ax2.set_xticks(group_centres)
ax2.set_xticklabels([f"{r} rank{'s' if r > 1 else ''}" for r in all_ranks], fontsize=10)

ax2.set_ylabel("Wall-clock time (s)", fontsize=12)
ax2.set_title("Wall-clock Time by Variant and Rank Count", fontsize=14, fontweight="bold")
ax2.legend(fontsize=9, loc="upper right")
ax2.grid(True, axis="y", linestyle=":", alpha=0.5)
ax2.set_ylim(bottom=0)

fig2.tight_layout()
fig2.savefig(OUT_TIME, dpi=150)
print(f"Saved: {OUT_TIME}")

# ---------------------------------------------------------------------------
# Print summary table to terminal
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print(f"{'Ranks':>6}  {'Variant':<12}  {'Time (s)':>10}  {'Speed-up':>10}")
print("-" * 68)
for r in all_ranks:
    baseline_t = avg[r].get("baseline", None)
    for v in all_variants:
        t = avg[r].get(v)
        if t is None:
            continue
        su = (baseline_t / t) if (baseline_t and baseline_t > 0) else float("nan")
        print(f"{r:>6}  {v:<12}  {t:>10.3f}  {su:>10.3f}×")
    print()
print("=" * 68)
