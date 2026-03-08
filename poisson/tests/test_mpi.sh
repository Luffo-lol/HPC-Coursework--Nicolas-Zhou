#!/bin/bash
# =============================================================================
# test_mpi.sh  —  Verification tests for the MPI Poisson solver
#
# Runs the three verification cases with multiple decompositions and compares
# the solution against the serial solver.  Must be run from the project root.
#
# Usage:
#   bash tests/test_mpi.sh
#
# Requirements:
#   - ./poisson       (serial binary, built with 'make poisson')
#   - ./poisson-mpi   (MPI binary,    built with 'make poisson-mpi')
#   - mpirun / mpiexec
# =============================================================================

set -e
PASS=0
FAIL=0
TOLERANCE=1e-6      # max L∞ difference between serial and parallel solutions

# ---- helpers ----------------------------------------------------------------

check() {
    local desc="$1"
    local result="$2"   # "PASS" or "FAIL"
    if [ "$result" = "PASS" ]; then
        echo "  [PASS] $desc"
        PASS=$((PASS+1))
    else
        echo "  [FAIL] $desc"
        FAIL=$((FAIL+1))
    fi
}

# Compare two solution files; print max L∞ error and return PASS/FAIL.
# Both files have header "Nx Ny Nz" then lines "x y z val".
compare_solutions() {
    local fileA="$1"
    local fileB="$2"
    python3 - "$fileA" "$fileB" "$TOLERANCE" <<'PYEOF'
import sys, math
a_path, b_path, tol = sys.argv[1], sys.argv[2], float(sys.argv[3])
def read(p):
    with open(p) as f:
        lines = f.read().split('\n')
    vals = {}
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) == 4:
            vals[(parts[0], parts[1], parts[2])] = float(parts[3])
    return vals
a, b = read(a_path), read(b_path)
maxErr = max(abs(a[k] - b[k]) for k in a if k in b)
print(f"  Max L∞ diff: {maxErr:.3e}", end='')
if maxErr < tol:
    print("  → PASS")
    sys.exit(0)
else:
    print(f"  → FAIL (tolerance {tol})")
    sys.exit(1)
PYEOF
}

# ---- test function ----------------------------------------------------------

run_test() {
    local tc="$1"       # test case number
    local Px="$2" Py="$3" Pz="$4"
    local P=$((Px*Py*Pz))
    local desc="Test case $tc  (${Px}x${Py}x${Pz} = $P ranks)"

    echo ""
    echo "[$desc]"

    # Run serial
    ./poisson --test "$tc" --epsilon 1e-7 > /dev/null
    cp solution.txt /tmp/serial_solution.txt

    # Run parallel
    mpirun -n "$P" ./poisson-mpi \
        --test "$tc" --epsilon 1e-7 \
        --Px "$Px" --Py "$Py" --Pz "$Pz" > /dev/null

    # Compare
    if compare_solutions /tmp/serial_solution.txt solution.txt; then
        check "$desc matches serial" "PASS"
    else
        check "$desc matches serial" "FAIL"
    fi
}

# ---- validate bad decomposition exits with error ----------------------------

echo ""
echo "[Decomposition validation]"
if ! mpirun -n 4 ./poisson-mpi --test 1 --Px 2 --Py 2 --Pz 2 > /dev/null 2>&1; then
    check "Px*Py*Pz != P is rejected" "PASS"
else
    check "Px*Py*Pz != P is rejected" "FAIL"
fi

# ---- run verification cases with various decompositions ---------------------
# (only run if small enough for quick CI; skip 64³ on >8 ranks)

run_test 1  1 1 1    # serial-equivalent
run_test 1  2 1 1    # split in x only
run_test 1  1 2 1    # split in y only
run_test 1  2 2 1    # 2-D slab (xy)
run_test 1  2 2 2    # full 3-D decomposition (8 ranks needed)

run_test 2  1 1 1
run_test 2  2 2 2

run_test 3  2 2 2

# ---- summary ----------------------------------------------------------------

echo ""
echo "========================================"
echo " MPI Test Results: $PASS passed, $FAIL failed"
echo "========================================"

[ "$FAIL" -eq 0 ]    # exit 0 iff all passed