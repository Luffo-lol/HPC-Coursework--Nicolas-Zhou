#!/usr/bin/env python3
"""
plot_speedup.py  --  Plot speed-up and wall-time from timing_results.csv

Produces:
  speedup_plot.png   -- speed-up of each variant relative to baseline
  walltime_plot.png  -- raw wall-clock times grouped by rank count

Usage:
    python3 plot_speedup.py [timing_results.csv]
"""
import sys, os, csv, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_FILE  = sys.argv[1] if len(sys.argv) > 1 else "timing_results.csv"
OUT_SPEED = "speedup_plot.png"
OUT_TIME  = "walltime_plot.png"

VARIANT_LABELS = {
    "baseline": "Baseline",
    "opt1":     "Opt1: comm overlap",
    "opt2":     "Opt2: + OpenMP",
    "opt3":     "Opt3: + raw-ptr + amortised residual",
}
VARIANT_ORDER  = ["baseline", "opt1", "opt2", "opt3"]
VARIANT_COLORS = ["#343a40", "#0d6efd", "#198754", "#dc3545"]

if not os.path.exists(CSV_FILE):
    sys.exit(f"ERROR: {CSV_FILE} not found. Run run_benchmark.sh first.")

data = collections.defaultdict(lambda: collections.defaultdict(list))
with open(CSV_FILE) as f:
    for row in csv.DictReader(f):
        data[int(row["ranks"])][row["variant"].strip()].append(float(row["wall_time_s"]))

avg = {r: {v: np.mean(ts) for v, ts in vd.items()} for r, vd in data.items()}
all_ranks    = sorted(avg)
all_variants = [v for v in VARIANT_ORDER if any(v in avg[r] for r in all_ranks)]

print(f"Ranks: {all_ranks}    Variants: {all_variants}")

# --- Figure 1: speed-up ---
fig, ax = plt.subplots(figsize=(8, 5))
for vi, v in enumerate(all_variants):
    xs, ys = [], []
    for r in all_ranks:
        if v in avg[r] and "baseline" in avg[r] and avg[r]["baseline"] > 0:
            xs.append(r); ys.append(avg[r]["baseline"] / avg[r][v])
    if xs:
        ax.plot(xs, ys, marker="o", lw=2, ms=7,
                color=VARIANT_COLORS[vi], label=VARIANT_LABELS.get(v, v))

ax.axhline(1.0, color="#6c757d", ls="--", lw=1.2, label="Baseline (1x)")
ax.axhspan(0, 1.0, alpha=0.04, color="red")
ax.set_xlabel("MPI Ranks", fontsize=12)
ax.set_ylabel("Speed-up relative to baseline", fontsize=11)
ax.set_title("Optimisation Speed-up vs Baseline", fontsize=13, fontweight="bold")
ax.set_xticks(all_ranks)
ax.legend(fontsize=9)
ax.grid(True, ls=":", alpha=0.5)
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(OUT_SPEED, dpi=150)
print(f"Saved: {OUT_SPEED}")

# --- Figure 2: bar chart ---
fig2, ax2 = plt.subplots(figsize=(10, 5))
nv = len(all_variants); bw = 0.18; gap = 0.12
xpos = np.arange(len(all_ranks)) * (nv * bw + gap)
max_t = max(avg[r].get(v, 0) for r in all_ranks for v in all_variants)

for vi, v in enumerate(all_variants):
    times = [avg[r].get(v, 0) for r in all_ranks]
    offs  = xpos + vi * bw
    bars  = ax2.bar(offs, times, width=bw, color=VARIANT_COLORS[vi],
                    edgecolor="white", lw=0.5, label=VARIANT_LABELS.get(v, v))
    for bar, t in zip(bars, times):
        if t > 0:
            ax2.text(bar.get_x() + bw/2, bar.get_height() + 0.005*max_t,
                     f"{t:.2f}s", ha="center", va="bottom", fontsize=6.5, rotation=90)

centres = xpos + (nv-1)*bw/2
ax2.set_xticks(centres)
ax2.set_xticklabels([f"{r} rank{'s' if r>1 else ''}" for r in all_ranks], fontsize=10)
ax2.set_ylabel("Wall-clock time (s)", fontsize=12)
ax2.set_title("Wall-clock Time by Variant and Rank Count", fontsize=13, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(True, axis="y", ls=":", alpha=0.5)
ax2.set_ylim(bottom=0)
fig2.tight_layout()
fig2.savefig(OUT_TIME, dpi=150)
print(f"Saved: {OUT_TIME}")

# --- terminal table ---
print("\n" + "="*64)
print(f"{'Ranks':>6}  {'Variant':<14}  {'Time(s)':>9}  {'Speed-up':>10}")
print("-"*64)
for r in all_ranks:
    bt = avg[r].get("baseline")
    for v in all_variants:
        t = avg[r].get(v)
        if t is None: continue
        su = (bt/t) if bt else float("nan")
        flag = " <-- regression" if su < 0.98 else ""
        print(f"{r:>6}  {v:<14}  {t:>9.3f}  {su:>9.3f}x{flag}")
    print()
print("="*64)