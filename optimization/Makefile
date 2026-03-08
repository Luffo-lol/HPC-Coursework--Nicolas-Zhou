# =============================================================================
# optimization/Makefile
#
# Expected layout:
#   optimization/
#   ├── Makefile          <-- THIS FILE
#   ├── base.cpp          <-- baseline MPI solver  → ./poisson-mpi
#   ├── opt1.cpp          <-- comm/compute overlap  → ./poisson-mpi-opt1
#   ├── opt2.cpp          <-- + OpenMP              → ./poisson-mpi-opt2
#   ├── opt3.cpp          <-- + raw-ptr + amortised → ./poisson-mpi-opt3
#   ├── run_benchmark.sh
#   └── plot_speedup.py
#
# -----------------------------------------------------------------------
# Targets
# -----------------------------------------------------------------------
#   all        Build all four binaries                        [DEFAULT]
#   baseline   Build ./poisson-mpi only
#   opt1       Build ./poisson-mpi-opt1 only
#   opt2       Build ./poisson-mpi-opt2 only
#   opt3       Build ./poisson-mpi-opt3 only
#
#   verify     Correctness check — all 4 variants, 1 rank, test 1
#
#   benchmark  Time all 4 variants at NP ranks (prints to terminal)
#              Defaults: NP=4  TC=2  OMP=4  EPS=1e-6
#              Override: make benchmark NP=8 TC=2 OMP=4
#
#   clean      Remove all binaries and timing_results.csv
#
# -----------------------------------------------------------------------
# Full workflow for speedup plots
# -----------------------------------------------------------------------
#   make all
#   bash run_benchmark.sh       # → timing_results.csv
#   python3 plot_speedup.py     # → speedup_plot.png + walltime_plot.png
# =============================================================================

CXX      := mpic++
CXXFLAGS := -std=c++17 -O3 -march=native -Wall
OMPFLAGS := -fopenmp

# Source files — all live in this directory
BASELINE_SRC := base.cpp
OPT1_SRC     := opt1.cpp
OPT2_SRC     := opt2.cpp
OPT3_SRC     := opt3.cpp

# Output binaries
BASELINE_BIN := poisson-mpi
OPT1_BIN     := poisson-mpi-opt1
OPT2_BIN     := poisson-mpi-opt2
OPT3_BIN     := poisson-mpi-opt3

ALL_BINS := $(BASELINE_BIN) $(OPT1_BIN) $(OPT2_BIN) $(OPT3_BIN)

# Benchmark knobs — override on command line: make benchmark NP=8
NP  ?= 4        # number of MPI ranks
OMP ?= 4        # OMP_NUM_THREADS for opt2/opt3
TC  ?= 2        # test case (2 = 64^3 smooth, good timing target)
EPS ?= 1e-6     # convergence tolerance (looser = faster runs)

.PHONY: all baseline opt1 opt2 opt3 verify benchmark clean

# =============================================================================
# Build rules
# =============================================================================
all: $(ALL_BINS)

# Baseline — from base.cpp, no extra flags
baseline: $(BASELINE_BIN)
$(BASELINE_BIN): $(BASELINE_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

# Opt1 — computation/communication overlap only (no OpenMP)
opt1: $(OPT1_BIN)
$(OPT1_BIN): $(OPT1_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

# Opt2 — Opt1 + OpenMP threading
opt2: $(OPT2_BIN)
$(OPT2_BIN): $(OPT2_SRC)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<

# Opt3 — Opt2 + raw-pointer hot loop + amortised residual
opt3: $(OPT3_BIN)
$(OPT3_BIN): $(OPT3_SRC)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<

# =============================================================================
# Correctness check — all 4 variants, 1 rank, test 1 (polynomial, exact soln)
# =============================================================================
verify: all
	@echo ""
	@echo "=== [1/4] Verifying baseline (poisson-mpi) ==="
	mpirun -n 1 ./$(BASELINE_BIN) --test 1
	@echo "=== [2/4] Verifying opt1 ==="
	mpirun -n 1 ./$(OPT1_BIN) --test 1
	@echo "=== [3/4] Verifying opt2 (OMP_NUM_THREADS=2) ==="
	OMP_NUM_THREADS=2 mpirun -n 1 ./$(OPT2_BIN) --test 1
	@echo "=== [4/4] Verifying opt3 (OMP_NUM_THREADS=2) ==="
	OMP_NUM_THREADS=2 mpirun -n 1 ./$(OPT3_BIN) --test 1
	@echo ""
	@echo "All variants verified."

# =============================================================================
# Quick benchmark — runs all 4 at NP ranks, prints real/user/sys time each
# Uses bash -c so the shell built-in 'time' is available
# =============================================================================
benchmark: all
	@echo ""
	@echo "============================================================"
	@echo " Quick benchmark:  test case $(TC)  |  $(NP) MPI rank(s)"
	@echo "                   opt2/opt3: OMP_NUM_THREADS=$(OMP)"
	@echo "                   epsilon = $(EPS)"
	@echo "============================================================"
	@echo ""
	@echo "--- [1/4] Baseline (poisson-mpi) ---"
	@bash -c "time mpirun -n $(NP) ./$(BASELINE_BIN) --test $(TC) --epsilon $(EPS) --Px $(NP) --Py 1 --Pz 1"
	@echo ""
	@echo "--- [2/4] Opt1: comm/compute overlap ---"
	@bash -c "time mpirun -n $(NP) ./$(OPT1_BIN) --test $(TC) --epsilon $(EPS) --Px $(NP) --Py 1 --Pz 1"
	@echo ""
	@echo "--- [3/4] Opt2: Opt1 + OpenMP ($(OMP) threads/rank) ---"
	@bash -c "time OMP_NUM_THREADS=$(OMP) mpirun -n $(NP) ./$(OPT2_BIN) --test $(TC) --epsilon $(EPS) --Px $(NP) --Py 1 --Pz 1"
	@echo ""
	@echo "--- [4/4] Opt3: Opt2 + raw-pointer + amortised residual ---"
	@bash -c "time OMP_NUM_THREADS=$(OMP) mpirun -n $(NP) ./$(OPT3_BIN) --test $(TC) --epsilon $(EPS) --Px $(NP) --Py 1 --Pz 1"
	@echo ""
	@echo "============================================================"
	@echo " For full CSV + plots:  bash run_benchmark.sh"
	@echo "============================================================"

# =============================================================================
# Clean
# =============================================================================
clean:
	rm -f $(ALL_BINS) solution.txt timing_results.csv
