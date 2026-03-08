#!/usr/bin/env bash
# =============================================================================
# run_benchmark.sh  —  Time baseline + opt1 + opt2 + opt3 across rank counts
#
# Writes timing_results.csv, then run:
#   python3 plot_speedup.py
#
# Usage:
#   bash run_benchmark.sh [OMP_THREADS]
#   OMP_THREADS — threads per rank for opt2/opt3  (default: 2)
#
# Uses test case 3 (32x64x128 = 262144 points) — large enough that OpenMP
# and raw-pointer optimisations show a real benefit.
# =============================================================================

set -euo pipefail

OMP=${1:-2}
CSV=timing_results.csv

# --- Edit to match your machine ----------------------------------------------
TC=3                 # test case 3: 32x64x128 grid
EPS=1e-5             # relaxed tolerance for faster runs
NREPS=1              # increase to 3 for stable timings
RANK_LIST=(1 2 4 8)  # must divide Nx=32; extend to (1 2 4 8 16 32) on big nodes
# -----------------------------------------------------------------------------

BASELINE=./poisson-mpi-baseline
OPT1=./poisson-mpi-opt1
OPT2=./poisson-mpi-opt2
OPT3=./poisson-mpi-opt3

for b in "$BASELINE" "$OPT1" "$OPT2" "$OPT3"; do
    [[ -x "$b" ]] || { echo "ERROR: $b not found. Run 'make all' first."; exit 1; }
done

time_run() {
    local bin=$1 np=$2 omp=$3; shift 3
    local wall
    wall=$( { TIMEFORMAT='%R'; time \
        OMP_NUM_THREADS="$omp" mpirun -n "$np" "$bin" "$@" >/dev/null 2>&1; } 2>&1 )
    printf "%.3f" "$wall"
}

echo ""
echo "============================================================"
echo " Benchmark: test case $TC | epsilon=$EPS | OMP=$OMP"
echo " Rank counts: ${RANK_LIST[*]}"
echo " Results -> $CSV"
echo "============================================================"
echo ""

echo "ranks,variant,wall_time_s" > "$CSV"

for NP in "${RANK_LIST[@]}"; do
    MPI_ARGS=(--test "$TC" --epsilon "$EPS" --Px "$NP" --Py 1 --Pz 1)
    echo "=== $NP rank(s) ==="

    for variant in baseline opt1 opt2 opt3; do
        case "$variant" in
            baseline) BIN=$BASELINE; OT=1   ;;
            opt1)     BIN=$OPT1;     OT=1   ;;
            opt2)     BIN=$OPT2;     OT=$OMP;;
            opt3)     BIN=$OPT3;     OT=$OMP;;
        esac

        total=0
        for ((rep=1; rep<=NREPS; rep++)); do
            t=$(time_run "$BIN" "$NP" "$OT" "${MPI_ARGS[@]}")
            total=$(echo "$total + $t" | bc -l)
        done
        avg=$(echo "scale=3; $total / $NREPS" | bc -l)

        printf "  %-10s  OMP=%-2d  %ss\n" "$variant" "$OT" "$avg"
        echo "$NP,$variant,$avg" >> "$CSV"
    done
    echo ""
done

echo "============================================================"
echo " Done.  Run: python3 plot_speedup.py"
echo "============================================================"