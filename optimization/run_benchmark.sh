#!/usr/bin/env bash
# =============================================================================
# run_benchmark.sh
#
# Times the baseline and all three optimisation variants across a range of
# MPI rank counts and writes timing_results.csv for plotting with:
#
#   python3 plot_speedup.py
#
# Usage:
#   bash run_benchmark.sh [TEST_CASE] [EPSILON]
#
#   TEST_CASE   (default 2 — 64^3 smooth solution, good timing target)
#   EPSILON     (default 1e-6 — looser tolerance so runs finish quickly)
#
# Edit RANK_LIST below to match the core counts on your machine / cluster.
# =============================================================================
set -euo pipefail

TC=${1:-2}
EPS=${2:-1e-6}

# Edit this list to match your available core counts
RANK_LIST=(1 2 4 8)

BASELINE=./poisson-mpi-baseline
OPT1=./poisson-mpi-opt1
OPT2=./poisson-mpi-opt2
OPT3=./poisson-mpi-opt3
CSV=timing_results.csv

check_binary() {
    if [[ ! -x "$1" ]]; then
        echo "ERROR: $1 not found.  Run 'make all' first." >&2
        exit 1
    fi
}

for b in "$BASELINE" "$OPT1" "$OPT2" "$OPT3"; do
    check_binary "$b"
done

echo ""
echo "========================================================"
echo " Benchmark: test case $TC  |  epsilon=$EPS"
echo " Rank counts: ${RANK_LIST[*]}"
echo "========================================================"
echo ""

echo "ranks,variant,wall_time_s" > "$CSV"

for NP in "${RANK_LIST[@]}"; do
    echo "--- $NP rank(s) ---"

    for variant in baseline opt1 opt2 opt3; do
        case "$variant" in
            baseline) BIN="$BASELINE" ;;
            opt1)     BIN="$OPT1" ;;
            opt2)     BIN="$OPT2" ;;
            opt3)     BIN="$OPT3" ;;
        esac

        MPI_ARGS=(--test "$TC" --epsilon "$EPS" --Px "$NP" --Py 1 --Pz 1)

        printf "  %-10s  NP=%-3d  ... " "$variant" "$NP"

        wall=$( { TIMEFORMAT='%R'; time \
            mpirun -n "$NP" "$BIN" "${MPI_ARGS[@]}" > /dev/null 2>&1; } 2>&1 )

        printf "%.3f s\n" "$wall"
        echo "$NP,$variant,$wall" >> "$CSV"
    done
    echo ""
done

echo "========================================================"
echo " Results written to: $CSV"
echo " Now run:  python3 plot_speedup.py"
echo "========================================================"
