#!/usr/bin/env bash
# =============================================================================
# run_benchmark.sh
#
# Times the baseline MPI solver and all three optimised variants across a
# range of MPI rank counts.  Results are written to timing_results.csv which
# can then be plotted with:
#
#   python3 plot_speedup.py
#
# Usage:
#   bash run_benchmark.sh [TEST_CASE] [EPSILON] [OMP_THREADS]
#
#   TEST_CASE   1-5   (default 2 — 64^3 smooth solution, good timing target)
#   EPSILON           (default 1e-6 — looser than default so runs finish fast)
#   OMP_THREADS       (default 4 — threads per rank for opt2/opt3)
#
# Rank counts tested:  1  2  4  8  (edit RANK_LIST below for your machine)
# =============================================================================

set -euo pipefail

TC=${1:-2}
EPS=${2:-1e-6}
OMP=${3:-4}

# --- Edit this list to match how many cores your machine / cluster has -------
RANK_LIST=(1 2 4 8)

# --- Binaries (must be built first with: make all) ---------------------------
BASELINE=./poisson-mpi
OPT1=./poisson-mpi-opt1
OPT2=./poisson-mpi-opt2
OPT3=./poisson-mpi-opt3

CSV=timing_results.csv

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

check_binary() {
    if [[ ! -x "$1" ]]; then
        echo "ERROR: $1 not found.  Run 'make all' first." >&2
        exit 1
    fi
}

# Run a solver once and extract wall-clock seconds.
# The solver prints "Final residual: X.XXe-XX (after NNN iterations)"
# We wrap it with 'time' and capture the real-time line.
time_run() {
    local bin="$1"
    local np="$2"
    local omp="$3"
    shift 3
    local args=("$@")

    # Use bash's TIMEFORMAT to capture wall time in seconds
    local wall
    wall=$( { TIMEFORMAT='%R'; time \
        OMP_NUM_THREADS="$omp" mpirun -n "$np" "$bin" "${args[@]}" \
        > /dev/null 2>&1; } 2>&1 )
    echo "$wall"
}

# -----------------------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------------------
for b in "$BASELINE" "$OPT1" "$OPT2" "$OPT3"; do
    check_binary "$b"
done

echo ""
echo "============================================================"
echo " Benchmark: test case $TC  |  epsilon=$EPS  |  OMP=$OMP"
echo " Rank counts: ${RANK_LIST[*]}"
echo "============================================================"
echo ""

# Write CSV header
echo "ranks,variant,wall_time_s" > "$CSV"

# -----------------------------------------------------------------------------
# Main timing loop
# -----------------------------------------------------------------------------
for NP in "${RANK_LIST[@]}"; do
    echo "--- $NP rank(s) ---"

    for variant in baseline opt1 opt2 opt3; do
        case "$variant" in
            baseline) BIN="$BASELINE"; OTHREADS=1 ;;
            opt1)     BIN="$OPT1";     OTHREADS=1 ;;
            opt2)     BIN="$OPT2";     OTHREADS="$OMP" ;;
            opt3)     BIN="$OPT3";     OTHREADS="$OMP" ;;
        esac

        # Build MPI args based on NP (simple 1-D decomposition in x)
        # Adjust decomposition to be valid for this rank count
        PX=$NP; PY=1; PZ=1
        MPI_ARGS=(--test "$TC" --epsilon "$EPS" --Px "$PX" --Py "$PY" --Pz "$PZ")

        printf "  %-10s  NP=%-3d  OMP=%-3d  ... " "$variant" "$NP" "$OTHREADS"

        T=$(time_run "$BIN" "$NP" "$OTHREADS" "${MPI_ARGS[@]}")

        printf "%.3f s\n" "$T"
        echo "$NP,$variant,$T" >> "$CSV"
    done

    echo ""
done

echo "============================================================"
echo " Results written to: $CSV"
echo " Now run:  python3 plot_speedup.py"
echo "============================================================"
