# =============================================================================
# optimization/Makefile
#
# Builds all three progressive optimisation variants.
# Each is a self-contained version of the MPI Poisson solver:
#
#   opt1.cpp  — Initial implementation: blocking MPI_Sendrecv (face by face),
#               MPI_Barrier at start of each iteration, defensive BC re-stamp
#               before and after every sweep.
#
#   opt2.cpp  — Replace blocking Sendrecv with 12 simultaneous non-blocking
#               Isend/Irecv + MPI_Waitall.  Barrier and BC re-stamp retained.
#
#   opt3.cpp  — Remove MPI_Barrier and both BC re-stamp loops.  These are
#               provably redundant: MPI_Allreduce already synchronises ranks
#               and interior-only loop bounds prevent any Dirichlet node from
#               being overwritten by the sweep.
#
# Expected layout:
#   HPC-COURSEWORK/
#   ├── poisson/src/poisson-mpi.cpp   (baseline — same algorithm as opt3)
#   └── optimization/
#       ├── Makefile                  <-- THIS FILE
#       ├── opt1.cpp
#       ├── opt2.cpp
#       ├── opt3.cpp
#       ├── run_benchmark.sh          (time all 3 + baseline -> CSV)
#       └── plot_speedup.py           (draw speedup_plot.png + walltime_plot.png)
#
# Targets
# -------
#   all           Build opt1 + opt2 + opt3 + baseline          [DEFAULT]
#   opt1          Build poisson-mpi-opt1 only
#   opt2          Build poisson-mpi-opt2 only
#   opt3          Build poisson-mpi-opt3 only
#   baseline      Build poisson-mpi-baseline (from poisson/src) only
#   verify        Quick correctness check: all 4 binaries, 1 rank, test 1
#   benchmark     Quick terminal timing comparison (make benchmark NP=4 TC=2)
#   clean         Remove all built binaries and timing_results.csv
#
# Full workflow to produce speedup plots:
#   make all
#   bash run_benchmark.sh        # writes timing_results.csv
#   python3 plot_speedup.py      # writes speedup_plot.png + walltime_plot.png
# =============================================================================

CXX      := mpic++
CXXFLAGS := -std=c++17 -O3 -march=native -Wall

BASELINE_SRC := ../poisson/src/poisson-mpi.cpp
OPT1_SRC     := opt1.cpp
OPT2_SRC     := opt2.cpp
OPT3_SRC     := opt3.cpp

BASELINE_BIN := poisson-mpi-baseline
OPT1_BIN     := poisson-mpi-opt1
OPT2_BIN     := poisson-mpi-opt2
OPT3_BIN     := poisson-mpi-opt3

ALL_BINS := $(BASELINE_BIN) $(OPT1_BIN) $(OPT2_BIN) $(OPT3_BIN)

# Benchmark defaults (override: make benchmark NP=8 TC=2 EPS=1e-6)
NP  ?= 4
TC  ?= 2
EPS ?= 1e-6

.PHONY: all baseline opt1 opt2 opt3 verify benchmark clean

all: $(ALL_BINS)

baseline: $(BASELINE_BIN)
$(BASELINE_BIN): $(BASELINE_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

opt1: $(OPT1_BIN)
$(OPT1_BIN): $(OPT1_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

opt2: $(OPT2_BIN)
$(OPT2_BIN): $(OPT2_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

opt3: $(OPT3_BIN)
$(OPT3_BIN): $(OPT3_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

# =============================================================================
# Correctness - all variants produce the same answer on test 1 (serial)
# =============================================================================
verify: all
	@echo ""
	@echo "=== Verifying baseline (poisson-mpi.cpp) ==="
	mpirun -n 1 ./$(BASELINE_BIN) --test 1
	@echo "=== Verifying opt1 (blocking comms + barrier + re-stamp) ==="
	mpirun -n 1 ./$(OPT1_BIN) --test 1
	@echo "=== Verifying opt2 (non-blocking comms, barrier + re-stamp) ==="
	mpirun -n 1 ./$(OPT2_BIN) --test 1
	@echo "=== Verifying opt3 (non-blocking comms, no barrier, no re-stamp) ==="
	mpirun -n 1 ./$(OPT3_BIN) --test 1
	@echo ""
	@echo "All variants verified."

# =============================================================================
# Quick benchmark - run all 4 at NP ranks and print wall times
# For full CSV + plots use:  bash run_benchmark.sh
# =============================================================================
benchmark: all
	@echo ""
	@echo "=================================================================="
	@echo " Benchmark: test $(TC)  |  $(NP) MPI rank(s)  |  epsilon=$(EPS)"
	@echo "=================================================================="
	@echo ""
	@echo "--- [baseline] poisson-mpi (equivalent to opt3) ---"
	$(SHELL) -c 'time mpirun -n $(NP) ./$(BASELINE_BIN) --test $(TC) --epsilon $(EPS) --Px $(NP) --Py 1 --Pz 1'
	@echo ""
	@echo "--- [opt1] blocking Sendrecv + MPI_Barrier + BC re-stamp ---"
	$(SHELL) -c 'time mpirun -n $(NP) ./$(OPT1_BIN) --test $(TC) --epsilon $(EPS) --Px $(NP) --Py 1 --Pz 1'
	@echo ""
	@echo "--- [opt2] non-blocking Isend/Irecv, barrier + re-stamp ---"
	$(SHELL) -c 'time mpirun -n $(NP) ./$(OPT2_BIN) --test $(TC) --epsilon $(EPS) --Px $(NP) --Py 1 --Pz 1'
	@echo ""
	@echo "--- [opt3] non-blocking, no barrier, no re-stamp (final) ---"
	$(SHELL) -c 'time mpirun -n $(NP) ./$(OPT3_BIN) --test $(TC) --epsilon $(EPS) --Px $(NP) --Py 1 --Pz 1'
	@echo ""
	@echo "=================================================================="
	@echo " For CSV output + plots:  bash run_benchmark.sh"
	@echo "=================================================================="

clean:
	rm -f $(ALL_BINS) solution.txt timing_results.csv
