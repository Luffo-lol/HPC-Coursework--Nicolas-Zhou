# =============================================================================
# optimization/Makefile
#
# Builds the baseline MPI solver and all three optimised variants.
# Provides timing and correctness targets.
#
# Expected layout:
#   HPC-COURSEWORK/
#   ├── poisson/src/poisson-mpi.cpp   <-- baseline source
#   └── optimization/
#       ├── Makefile                  <-- THIS FILE
#       ├── opt1.cpp
#       ├── opt2.cpp
#       ├── opt3.cpp
#       ├── run_benchmark.sh          (run to collect timings → CSV)
#       └── plot_speedup.py           (run to draw speedup/time plots)
#
# -----------------------------------------------------------------------
# Targets
# -----------------------------------------------------------------------
#   all           Build baseline + opt1 + opt2 + opt3          [DEFAULT]
#   baseline      Build poisson-mpi-baseline only
#   opt1          Build poisson-mpi-opt1 only
#   opt2          Build poisson-mpi-opt2 only
#   opt3          Build poisson-mpi-opt3 only
#
#   verify        Quick correctness check (all 4 variants, 1 rank, test 1)
#
#   benchmark     Quick timing comparison at NP ranks (prints to terminal)
#                 Override defaults:  make benchmark NP=8 TC=2 OMP=4
#
#   clean         Remove all built binaries and timing_results.csv
#
# -----------------------------------------------------------------------
# Workflow to produce speedup plots
# -----------------------------------------------------------------------
#   make all
#   bash run_benchmark.sh          # writes timing_results.csv
#   python3 plot_speedup.py        # writes speedup_plot.png + walltime_plot.png
# =============================================================================

CXX      := mpic++
CXXFLAGS := -std=c++17 -O3 -march=native -Wall
OMPFLAGS := -fopenmp

# Source files
BASELINE_SRC := ../poisson/src/poisson-mpi.cpp
OPT1_SRC     := opt1.cpp
OPT2_SRC     := opt2.cpp
OPT3_SRC     := opt3.cpp

# Output binaries
BASELINE_BIN := poisson-mpi-baseline
OPT1_BIN     := poisson-mpi-opt1
OPT2_BIN     := poisson-mpi-opt2
OPT3_BIN     := poisson-mpi-opt3

ALL_BINS := $(BASELINE_BIN) $(OPT1_BIN) $(OPT2_BIN) $(OPT3_BIN)

# Benchmark knobs (override on command line)
NP  ?= 4        # MPI ranks for 'make benchmark'
OMP ?= 4        # OMP_NUM_THREADS for opt2/opt3
TC  ?= 2        # test case (2 = 64^3 smooth, good timing target)
EPS ?= 1e-6     # convergence tolerance (looser = faster runs)

.PHONY: all baseline opt1 opt2 opt3 verify benchmark clean

# =============================================================================
# Build rules
# =============================================================================
all: $(ALL_BINS)

# Baseline — compiled from the main src tree, no extra optimisations
baseline: $(BASELINE_BIN)
$(BASELINE_BIN): $(BASELINE_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

# Opt1 — computation/communication overlap only
opt1: $(OPT1_BIN)
$(OPT1_BIN): $(OPT1_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

# Opt2 — Opt1 + OpenMP
opt2: $(OPT2_BIN)
$(OPT2_BIN): $(OPT2_SRC)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<

# Opt3 — Opt2 + raw-pointer inner loop + amortised residual check
opt3: $(OPT3_BIN)
$(OPT3_BIN): $(OPT3_SRC)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<

# =============================================================================
# Correctness check — all variants, 1 rank, test case 1 (polynomial solution)
# =============================================================================
verify: all
	@echo ""
	@echo "=== Verifying baseline ==="
	mpirun -n 1 ./$(BASELINE_BIN) --test 1
	@echo "=== Verifying opt1 ==="
	mpirun -n 1 ./$(OPT1_BIN) --test 1
	@echo "=== Verifying opt2 (OMP_NUM_THREADS=2) ==="
	OMP_NUM_THREADS=2 mpirun -n 1 ./$(OPT2_BIN) --test 1
	@echo "=== Verifying opt3 (OMP_NUM_THREADS=2) ==="
	OMP_NUM_THREADS=2 mpirun -n 1 ./$(OPT3_BIN) --test 1
	@echo ""
	@echo "All variants verified."

# =============================================================================
# Quick benchmark — run all 4 variants once at NP ranks, print times
# For full scaling study use run_benchmark.sh instead
# =============================================================================
benchmark: all
	@echo ""
	@echo "============================================================"
	@echo " Quick benchmark:  test case $(TC)  $(NP) MPI rank(s)"
	@echo "                   opt2/opt3: OMP_NUM_THREADS=$(OMP)"
	@echo "                   epsilon = $(EPS)"
	@echo "============================================================"
	@echo ""
	@echo "--- [1/4] Baseline ---"
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
	@echo " For CSV output + plots:  bash run_benchmark.sh"
	@echo "============================================================"

# =============================================================================
# Clean
# =============================================================================
clean:
	rm -f $(ALL_BINS) solution.txt timing_results.csv
