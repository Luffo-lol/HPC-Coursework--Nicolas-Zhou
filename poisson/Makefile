# =============================================================================
# Makefile for the 3D Poisson Solver
# Lives at: HPC-COURSEWORK/poisson/Makefile
#
# Targets:
#   all           - Build the serial solver (default)
#   poisson       - Build the serial solver
#   poisson-mpi   - Build the MPI parallel solver
#   poisson-cuda  - Build the MPI+CUDA solver (requires CUDA toolkit)
#   tests         - Build and run the serial unit test suite
#   tests-mpi     - Run the MPI verification tests (requires mpirun)
#   doc           - Generate Doxygen HTML documentation
#   optimization  - Build all optimised variants in ../optimization/
#   clean         - Remove all build artefacts (including optimization/ binaries)
# =============================================================================

# --- Compilers & flags -------------------------------------------------------
CXX      := g++
MPICXX   := mpic++
NVCC     := nvcc
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra
LDFLAGS  := -lm

# CUDA flags — adjust -arch to match your GPU:
#   sm_60  Pascal  (P100)
#   sm_70  Volta   (V100)
#   sm_80  Ampere  (A100, RTX 30xx)
#   sm_89  Ada     (RTX 40xx)
CUDA_ARCH   := sm_80
NVCCFLAGS   := -O3 -arch=$(CUDA_ARCH) -std=c++17 -allow-unsupported-compiler --compiler-bindir /usr/bin/gcc-12
# Grab MPI include/lib paths from the mpic++ wrapper
MPI_INCS    := $(addprefix -I,$(shell mpic++ --showme:incdirs 2>/dev/null))
MPI_CFLAGS  := $(shell mpic++ --showme:compile 2>/dev/null)
MPI_LDFLAGS := $(shell mpic++ --showme:link   2>/dev/null)

# --- Directories -------------------------------------------------------------
SRC_DIR  := src
TEST_DIR := tests
DOC_DIR  := doc
OPT_DIR  := ../optimization

# --- Source files ------------------------------------------------------------
SERIAL_SRC := $(SRC_DIR)/poisson.cpp
MPI_SRC    := $(SRC_DIR)/poisson-mpi.cpp
CUDA_SRC   := $(SRC_DIR)/poisson-cuda.cu
TEST_SRC   := $(TEST_DIR)/test_poisson.cpp

# --- Binaries ----------------------------------------------------------------
SERIAL_BIN := poisson
MPI_BIN    := poisson-mpi
CUDA_BIN   := poisson-cuda
TEST_BIN   := $(TEST_DIR)/test_poisson

# =============================================================================
# Default target
# =============================================================================
.PHONY: all
all: poisson

# =============================================================================
# Serial solver
# =============================================================================
.PHONY: poisson
poisson: $(SERIAL_SRC)
	$(CXX) $(CXXFLAGS) -o $(SERIAL_BIN) $^ $(LDFLAGS)

# =============================================================================
# MPI parallel solver
# =============================================================================
.PHONY: poisson-mpi
poisson-mpi: $(MPI_SRC)
	$(MPICXX) $(CXXFLAGS) -o $(MPI_BIN) $^ $(LDFLAGS)

# =============================================================================
# MPI + CUDA solver
#
# Uses nvcc as the compiler driver.  MPI headers and libs are obtained
# from mpic++ --showme flags so that the same MPI installation is used.
# The poisson-mpi target is NOT affected — it still builds without CUDA.
# =============================================================================
.PHONY: poisson-cuda
poisson-cuda: $(CUDA_SRC)
	$(NVCC) $(NVCCFLAGS) $(MPI_INCS) -Xcompiler "-pthread" -c $^ -o poisson-cuda.o
	$(MPICXX) -o $(CUDA_BIN) poisson-cuda.o -L/apps/cuda-hpc-sdk/24.9/Linux_x86_64/24.9/cuda/12.6/targets/x86_64-linux/lib -lcudart -Wl,-rpath,/apps/cuda-hpc-sdk/24.9/Linux_x86_64/24.9/cuda/12.6/targets/x86_64-linux/lib

# =============================================================================
# Tests — serial unit tests
# =============================================================================
.PHONY: tests
tests: $(TEST_BIN)
	@echo "--- Running serial test suite ---"
	./$(TEST_BIN)

$(TEST_BIN): $(TEST_SRC) $(SERIAL_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $(TEST_SRC) $(LDFLAGS)

# =============================================================================
# Tests — MPI verification tests
# =============================================================================
.PHONY: tests-mpi
tests-mpi: poisson poisson-mpi
	@echo "--- Running MPI verification tests ---"
	bash $(TEST_DIR)/test_mpi.sh

# =============================================================================
# Doxygen documentation
# =============================================================================
.PHONY: doc
doc:
	@if command -v doxygen > /dev/null 2>&1; then \
		doxygen Doxyfile; \
		echo "Documentation generated in $(DOC_DIR)/html/index.html"; \
	else \
		echo "Error: doxygen not found. Please install doxygen."; \
		exit 1; \
	fi

# =============================================================================
# Optimisation variants — delegates to ../optimization/Makefile
# =============================================================================
.PHONY: optimization
optimization:
	$(MAKE) -C $(OPT_DIR) all

# =============================================================================
# Clean — removes all binaries, docs, and optimization/ binaries
# =============================================================================
.PHONY: clean
clean:
	rm -f $(SERIAL_BIN) $(MPI_BIN) $(CUDA_BIN) $(TEST_BIN)
	rm -f solution.txt
	rm -rf $(DOC_DIR)/html $(DOC_DIR)/latex
	$(MAKE) -C $(OPT_DIR) clean
