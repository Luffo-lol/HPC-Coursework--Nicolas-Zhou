# =============================================================================
# optimization/Makefile
#
# Builds all optimised MPI Poisson solver variants.
# Lives at:   HPC-COURSEWORK/optimization/Makefile
# Baseline:   HPC-COURSEWORK/poisson/src/poisson-mpi.cpp
#
# Targets
# -------
#   all      Build every variant (default)
#   baseline poisson-mpi-baseline  — unmodified baseline
#   opt1     poisson-mpi-opt1      — comm/compute overlap
#   opt2     poisson-mpi-opt2      — + OpenMP threading
#   opt3     poisson-mpi-opt3      — + raw-pointer hot loop & amortised residual
#   verify   Quick correctness smoke-test (test case 1, 1 rank)
#   clean    Remove all binaries and solution.txt
# =============================================================================

CXX      := mpic++
CXXFLAGS := -std=c++17 -O3 -march=native
OMPFLAGS := -fopenmp

BASELINE_SRC := ../poisson/src/poisson-mpi.cpp
OPT1_SRC     := opt1.cpp
OPT2_SRC     := opt2.cpp
OPT3_SRC     := opt3.cpp

BASELINE_BIN := poisson-mpi-baseline
OPT1_BIN     := poisson-mpi-opt1
OPT2_BIN     := poisson-mpi-opt2
OPT3_BIN     := poisson-mpi-opt3

.PHONY: all baseline opt1 opt2 opt3 verify clean

all: baseline opt1 opt2 opt3

baseline: $(BASELINE_BIN)
$(BASELINE_BIN): $(BASELINE_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

opt1: $(OPT1_BIN)
$(OPT1_BIN): $(OPT1_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

opt2: $(OPT2_BIN)
$(OPT2_BIN): $(OPT2_SRC)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<

opt3: $(OPT3_BIN)
$(OPT3_BIN): $(OPT3_SRC)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<

verify: all
	@echo "--- baseline ---"
	mpirun -n 1 ./$(BASELINE_BIN) --test 1
	@echo "--- opt1 ---"
	mpirun -n 1 ./$(OPT1_BIN) --test 1
	@echo "--- opt2 (2 threads) ---"
	OMP_NUM_THREADS=2 mpirun -n 1 ./$(OPT2_BIN) --test 1
	@echo "--- opt3 (2 threads) ---"
	OMP_NUM_THREADS=2 mpirun -n 1 ./$(OPT3_BIN) --test 1
	@echo "All variants verified."

clean:
	rm -f $(BASELINE_BIN) $(OPT1_BIN) $(OPT2_BIN) $(OPT3_BIN) solution.txt
