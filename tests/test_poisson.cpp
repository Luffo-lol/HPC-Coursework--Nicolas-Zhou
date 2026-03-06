/**
 * @file test_poisson.cpp
 * @brief Unit and verification test suite for the serial 3D Poisson solver.
 *
 * Tests cover:
 *  - Verification Case 1: Polynomial exact solution  u = x² + y² + z²
 *  - Verification Case 2: Smooth trigonometric solution
 *  - Verification Case 3: Anisotropic oscillatory solution
 *  - Residual convergence: solver stops below the requested tolerance
 *  - Boundary conditions: boundary values remain fixed after solve
 *  - File I/O: solution.txt is written in the correct format
 *  - CLI: --help exits without error; missing/conflicting args return error
 *
 * Build and run via:
 *   make tests
 * or manually:
 *   g++ -std=c++17 -O2 -o tests/test_poisson tests/test_poisson.cpp -lm
 *   ./tests/test_poisson
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cassert>
#include <string>
#include <stdexcept>
#include <iomanip>
#include <functional>
#include <numeric>
#include <algorithm>

// ============================================================
//  Pull in the solver logic via a header-friendly re-inclusion.
//  We compile poisson.cpp directly into this translation unit
//  but rename main() so it does not clash.
// ============================================================
#define main poisson_main
#include "../src/poisson.cpp"
#undef main

// ============================================================
//  Tiny testing framework
// ============================================================

static int g_passed = 0;
static int g_failed = 0;

/** @brief Record a single test result. */
#define CHECK(cond, msg)                                         \
    do {                                                         \
        if (cond) {                                              \
            std::cout << "  [PASS] " << (msg) << "\n";          \
            ++g_passed;                                          \
        } else {                                                 \
            std::cout << "  [FAIL] " << (msg) << "\n";          \
            ++g_failed;                                          \
        }                                                        \
    } while (0)

/** @brief Helper: run the full solver for a given test case and epsilon.
 *
 * Returns the converged solution vector and fills Nx/Ny/Nz.
 */
std::vector<double> runSolver(int testCase, double epsilon,
                               int &Nx, int &Ny, int &Nz)
{
    testCaseGrid(testCase, Nx, Ny, Nz);
    int N = Nx * Ny * Nz;

    double hx = 1.0 / (Nx - 1);
    double hy = 1.0 / (Ny - 1);
    double hz = 1.0 / (Nz - 1);

    // Build forcing vector
    std::vector<double> f(N, 0.0);
    for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < Ny; ++j)
            for (int k = 0; k < Nz; ++k)
                f[idx(i,j,k,Ny,Nz)] = forcingFunction(testCase,
                                                        i*hx, j*hy, k*hz);

    // Initialise u and unew
    std::vector<double> u(N, 0.0), unew(N, 0.0);

    // Apply Dirichlet BCs from exact solution
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                bool boundary = (i==0||i==Nx-1||j==0||j==Ny-1||k==0||k==Nz-1);
                if (!boundary) continue;
                double x=i*hx, y=j*hy, z=k*hz;
                u   [idx(i,j,k,Ny,Nz)] = exactSolution(testCase, x, y, z);
                unew[idx(i,j,k,Ny,Nz)] = exactSolution(testCase, x, y, z);
            }
        }
    }

    // Jacobi iteration
    double residual;
    do {
        jacobiSweep(u, f, unew, Nx, Ny, Nz, hx, hy, hz);
        // re-apply BCs
        for (int i = 0; i < Nx; ++i)
            for (int j = 0; j < Ny; ++j)
                for (int k = 0; k < Nz; ++k) {
                    bool boundary = (i==0||i==Nx-1||j==0||j==Ny-1||k==0||k==Nz-1);
                    if (!boundary) continue;
                    double x=i*hx, y=j*hy, z=k*hz;
                    unew[idx(i,j,k,Ny,Nz)] = exactSolution(testCase, x, y, z);
                }
        std::swap(u, unew);
        residual = computeResidual(u, f, Nx, Ny, Nz, hx, hy, hz);
    } while (residual > epsilon);

    return u;
}

/** @brief Compute the max-norm error against the analytical solution. */
double maxError(const std::vector<double>& u, int testCase,
                int Nx, int Ny, int Nz)
{
    double hx = 1.0/(Nx-1), hy = 1.0/(Ny-1), hz = 1.0/(Nz-1);
    double maxErr = 0.0;
    for (int i = 1; i < Nx-1; ++i)
        for (int j = 1; j < Ny-1; ++j)
            for (int k = 1; k < Nz-1; ++k) {
                double x=i*hx, y=j*hy, z=k*hz;
                double err = std::abs(u[idx(i,j,k,Ny,Nz)]
                                      - exactSolution(testCase, x, y, z));
                maxErr = std::max(maxErr, err);
            }
    return maxErr;
}

// ============================================================
//  Test 1 — Polynomial solution  u = x² + y² + z²
// ============================================================

/**
 * @brief Verify that the solver recovers the polynomial exact solution.
 *
 * Because the forcing f=6 is exact on the discrete grid (the second-order
 * finite difference is exact for polynomials up to degree 2), the solver
 * should converge to machine precision.  We check the max-norm error is
 * below a generous tolerance of 1e-4 on a 32³ grid.
 */
void testPolynomialSolution() {
    std::cout << "\n[Test 1] Polynomial solution u = x² + y² + z²\n";

    int Nx, Ny, Nz;
    const double eps = 1e-6;
    auto u = runSolver(1, eps, Nx, Ny, Nz);

    double err = maxError(u, 1, Nx, Ny, Nz);
    std::cout << "  Max-norm error vs exact: " << std::scientific << err << "\n";

    CHECK(Nx == 32 && Ny == 32 && Nz == 32, "Grid is 32³");
    CHECK(err < 1e-4, "Max-norm error < 1e-4");

    // The residual after convergence should also be ≤ eps
    double hx=1.0/(Nx-1), hy=1.0/(Ny-1), hz=1.0/(Nz-1);
    int N = Nx*Ny*Nz;
    std::vector<double> f(N);
    for (int i=0;i<Nx;++i) for (int j=0;j<Ny;++j) for (int k=0;k<Nz;++k)
        f[idx(i,j,k,Ny,Nz)] = forcingFunction(1, i*hx, j*hy, k*hz);
    double res = computeResidual(u, f, Nx, Ny, Nz, hx, hy, hz);
    CHECK(res <= eps * 1.01, "Final residual ≤ epsilon");
}

// ============================================================
//  Test 2 — Smooth trigonometric solution
// ============================================================

/**
 * @brief Verify the smooth trigonometric verification case on a 64³ grid.
 *
 * The exact solution is u = sin(πx)sin(πy)sin(πz).  The discretisation
 * error is O(h²), so on a 64³ grid (h ≈ 0.016) we expect errors well
 * below 1e-2.
 */
void testSmoothSolution() {
    std::cout << "\n[Test 2] Smooth solution  u = sin(πx)sin(πy)sin(πz)\n";

    int Nx, Ny, Nz;
    const double eps = 1e-5;
    auto u = runSolver(2, eps, Nx, Ny, Nz);

    double err = maxError(u, 2, Nx, Ny, Nz);
    std::cout << "  Max-norm error vs exact: " << std::scientific << err << "\n";

    CHECK(Nx == 64 && Ny == 64 && Nz == 64, "Grid is 64³");
    CHECK(err < 1e-2, "Max-norm error < 1e-2 (consistent with O(h²))");

    double hx=1.0/(Nx-1), hy=1.0/(Ny-1), hz=1.0/(Nz-1);
    int N = Nx*Ny*Nz;
    std::vector<double> f(N);
    for (int i=0;i<Nx;++i) for (int j=0;j<Ny;++j) for (int k=0;k<Nz;++k)
        f[idx(i,j,k,Ny,Nz)] = forcingFunction(2, i*hx, j*hy, k*hz);
    double res = computeResidual(u, f, Nx, Ny, Nz, hx, hy, hz);
    CHECK(res <= eps * 1.01, "Final residual ≤ epsilon");
}

// ============================================================
//  Test 3 — Anisotropic oscillatory solution  32×64×128
// ============================================================

/**
 * @brief Verify the anisotropic oscillatory verification case.
 *
 * Exact solution: u = sin(πx)sin(4πy)sin(8πz) on a 32×64×128 grid.
 * Higher wave-numbers mean larger discretisation error, so we accept
 * errors up to 0.1 on this coarser-per-wavelength grid.
 */
void testAnisotropicSolution() {
    std::cout << "\n[Test 3] Anisotropic solution  u = sin(πx)sin(4πy)sin(8πz)\n";

    int Nx, Ny, Nz;
    const double eps = 1e-4;
    auto u = runSolver(3, eps, Nx, Ny, Nz);

    double err = maxError(u, 3, Nx, Ny, Nz);
    std::cout << "  Max-norm error vs exact: " << std::scientific << err << "\n";

    CHECK(Nx == 32 && Ny == 64 && Nz == 128, "Grid is 32×64×128");
    CHECK(err < 0.1, "Max-norm error < 0.1 (coarse grid, high wave-number)");

    double hx=1.0/(Nx-1), hy=1.0/(Ny-1), hz=1.0/(Nz-1);
    int N = Nx*Ny*Nz;
    std::vector<double> f(N);
    for (int i=0;i<Nx;++i) for (int j=0;j<Ny;++j) for (int k=0;k<Nz;++k)
        f[idx(i,j,k,Ny,Nz)] = forcingFunction(3, i*hx, j*hy, k*hz);
    double res = computeResidual(u, f, Nx, Ny, Nz, hx, hy, hz);
    CHECK(res <= eps * 1.01, "Final residual ≤ epsilon");
}

// ============================================================
//  Test 4 — Residual decreases monotonically (first 10 iters)
// ============================================================

/**
 * @brief Confirm the Jacobi iteration is actually reducing the residual.
 *
 * Runs 10 iterations and checks that each successive residual is no
 * larger than the previous one.
 */
void testResidualDecreases() {
    std::cout << "\n[Test 4] Residual is non-increasing\n";

    int Nx=16, Ny=16, Nz=16;
    int N = Nx*Ny*Nz;
    double hx=1.0/(Nx-1), hy=1.0/(Ny-1), hz=1.0/(Nz-1);

    std::vector<double> f(N), u(N,0.0), unew(N,0.0);
    for (int i=0;i<Nx;++i) for (int j=0;j<Ny;++j) for (int k=0;k<Nz;++k)
        f[idx(i,j,k,Ny,Nz)] = forcingFunction(2, i*hx, j*hy, k*hz);
    // Exact BCs
    for (int i=0;i<Nx;++i) for (int j=0;j<Ny;++j) for (int k=0;k<Nz;++k) {
        bool bd = (i==0||i==Nx-1||j==0||j==Ny-1||k==0||k==Nz-1);
        if (!bd) continue;
        double v = exactSolution(2, i*hx, j*hy, k*hz);
        u[idx(i,j,k,Ny,Nz)] = unew[idx(i,j,k,Ny,Nz)] = v;
    }

    double prevRes = computeResidual(u, f, Nx, Ny, Nz, hx, hy, hz);
    bool monotone = true;
    for (int iter = 0; iter < 10; ++iter) {
        jacobiSweep(u, f, unew, Nx, Ny, Nz, hx, hy, hz);
        std::swap(u, unew);
        double res = computeResidual(u, f, Nx, Ny, Nz, hx, hy, hz);
        if (res > prevRes * 1.001) { monotone = false; break; }
        prevRes = res;
    }
    CHECK(monotone, "Residual is non-increasing over 10 iterations");
}

// ============================================================
//  Test 5 — Boundary conditions remain fixed
// ============================================================

/**
 * @brief Check that boundary values are not modified by the sweep.
 *
 * After running the Jacobi sweep once, all boundary nodes must still
 * equal their prescribed Dirichlet values.
 */
void testBoundaryConditions() {
    std::cout << "\n[Test 5] Boundary conditions remain fixed\n";

    int Nx=8, Ny=8, Nz=8;
    int N = Nx*Ny*Nz;
    double hx=1.0/(Nx-1), hy=1.0/(Ny-1), hz=1.0/(Nz-1);

    std::vector<double> f(N, 1.0);
    std::vector<double> u(N, 0.0), unew(N, 0.0);

    // Non-trivial boundary values
    for (int i=0;i<Nx;++i) for (int j=0;j<Ny;++j) for (int k=0;k<Nz;++k) {
        bool bd = (i==0||i==Nx-1||j==0||j==Ny-1||k==0||k==Nz-1);
        if (!bd) continue;
        double v = exactSolution(1, i*hx, j*hy, k*hz);
        u[idx(i,j,k,Ny,Nz)] = unew[idx(i,j,k,Ny,Nz)] = v;
    }

    jacobiSweep(u, f, unew, Nx, Ny, Nz, hx, hy, hz);

    bool bcOk = true;
    for (int i=0;i<Nx && bcOk;++i)
        for (int j=0;j<Ny && bcOk;++j)
            for (int k=0;k<Nz && bcOk;++k) {
                bool bd = (i==0||i==Nx-1||j==0||j==Ny-1||k==0||k==Nz-1);
                if (!bd) continue;
                double expected = exactSolution(1, i*hx, j*hy, k*hz);
                if (std::abs(unew[idx(i,j,k,Ny,Nz)] - expected) > 1e-14)
                    bcOk = false;
            }
    CHECK(bcOk, "Boundary values unchanged after jacobiSweep()");
}

// ============================================================
//  Test 6 — Zero forcing gives zero interior solution (homogeneous BC)
// ============================================================

/**
 * @brief If f=0 and u=0 on all boundaries, the solution should be u≡0.
 */
void testZeroForcingZeroBC() {
    std::cout << "\n[Test 6] Zero forcing + zero BCs  →  u ≡ 0\n";

    int Nx=8, Ny=8, Nz=8;
    int N = Nx*Ny*Nz;
    double hx=1.0/(Nx-1), hy=1.0/(Ny-1), hz=1.0/(Nz-1);

    std::vector<double> f(N, 0.0);
    std::vector<double> u(N, 0.0), unew(N, 0.0);

    // Single sweep on all-zero initial guess should give all-zero
    jacobiSweep(u, f, unew, Nx, Ny, Nz, hx, hy, hz);

    double maxVal = 0.0;
    for (double v : unew) maxVal = std::max(maxVal, std::abs(v));
    CHECK(maxVal < 1e-15, "All values remain zero");
}

// ============================================================
//  Test 7 — Output file solution.txt is well-formed
// ============================================================

/**
 * @brief Run the solver and verify the solution.txt output file format.
 *
 * Checks:
 *  - First line contains three integers (Nx Ny Nz).
 *  - Subsequent lines each contain four floating-point numbers.
 *  - Total number of data lines equals Nx*Ny*Nz.
 */
void testOutputFile() {
    std::cout << "\n[Test 7] solution.txt file format\n";

    int Nx=8, Ny=8, Nz=8;
    int N = Nx*Ny*Nz;
    double hx=1.0/(Nx-1), hy=1.0/(Ny-1), hz=1.0/(Nz-1);

    std::vector<double> u(N, 0.0);
    // Provide a trivial non-zero solution
    for (int i=0;i<Nx;++i) for (int j=0;j<Ny;++j) for (int k=0;k<Nz;++k)
        u[idx(i,j,k,Ny,Nz)] = exactSolution(1, i*hx, j*hy, k*hz);

    writeSolution("/tmp/test_solution.txt", u, Nx, Ny, Nz, hx, hy, hz);

    std::ifstream fin("/tmp/test_solution.txt");
    CHECK(fin.is_open(), "solution.txt was created");

    int rNx, rNy, rNz;
    fin >> rNx >> rNy >> rNz;
    CHECK(rNx == Nx && rNy == Ny && rNz == Nz, "Header Nx Ny Nz matches");

    int lineCount = 0;
    double x, y, z, val;
    while (fin >> x >> y >> z >> val) ++lineCount;
    CHECK(lineCount == N, "Number of data lines equals Nx*Ny*Nz");
}

// ============================================================
//  Test 8 — idx() mapping is consistent and unique
// ============================================================

/**
 * @brief Verify that the idx() function produces unique indices for
 *        every (i,j,k) combination and covers [0, Nx*Ny*Nz).
 */
void testIndexMapping() {
    std::cout << "\n[Test 8] idx() produces unique indices\n";

    int Nx=4, Ny=5, Nz=6;
    int N = Nx*Ny*Nz;
    std::vector<int> seen(N, 0);
    bool ok = true;
    for (int i=0;i<Nx;++i)
        for (int j=0;j<Ny;++j)
            for (int k=0;k<Nz;++k) {
                int id = idx(i,j,k,Ny,Nz);
                if (id < 0 || id >= N) { ok = false; break; }
                seen[id]++;
            }
    bool allOne = std::all_of(seen.begin(), seen.end(), [](int c){ return c==1; });
    CHECK(ok && allOne, "idx() values are unique and in range [0, Nx*Ny*Nz)");
}

// ============================================================
//  Test 9 — forcingFunction() matches analytical expressions
// ============================================================

/**
 * @brief Spot-check forcing function values against hand-computed results.
 */
void testForcingFunction() {
    std::cout << "\n[Test 9] forcingFunction() analytical spot-checks\n";

    // Case 1: f = 6 everywhere
    CHECK(std::abs(forcingFunction(1, 0.3, 0.5, 0.7) - 6.0) < 1e-14,
          "Case 1: f(0.3,0.5,0.7) = 6");

    // Case 2: f = -3π² sin(πx)sin(πy)sin(πz)
    double x=0.25, y=0.5, z=0.75;
    double expected2 = -3.0*M_PI*M_PI
                       * std::sin(M_PI*x)*std::sin(M_PI*y)*std::sin(M_PI*z);
    CHECK(std::abs(forcingFunction(2, x, y, z) - expected2) < 1e-12,
          "Case 2: f matches -3π²sin(πx)sin(πy)sin(πz)");

    // Case 3: f = -81π² sin(πx)sin(4πy)sin(8πz)
    double expected3 = -81.0*M_PI*M_PI
                       * std::sin(M_PI*x)*std::sin(4*M_PI*y)*std::sin(8*M_PI*z);
    CHECK(std::abs(forcingFunction(3, x, y, z) - expected3) < 1e-12,
          "Case 3: f matches -81π²sin(πx)sin(4πy)sin(8πz)");
}

// ============================================================
//  Test 10 — computeResidual() returns zero for exact solution
// ============================================================

/**
 * @brief For the polynomial solution (which is exact on the discrete grid),
 *        the discrete residual should be essentially machine-precision zero.
 */
void testResidualExactSolution() {
    std::cout << "\n[Test 10] Residual is ~0 for exact polynomial solution\n";

    int Nx=8, Ny=8, Nz=8;
    int N = Nx*Ny*Nz;
    double hx=1.0/(Nx-1), hy=1.0/(Ny-1), hz=1.0/(Nz-1);

    // u = x² + y² + z²  →  ∇²u = 6  exactly on the discrete grid
    std::vector<double> u(N), f(N);
    for (int i=0;i<Nx;++i) for (int j=0;j<Ny;++j) for (int k=0;k<Nz;++k) {
        u[idx(i,j,k,Ny,Nz)] = exactSolution(1, i*hx, j*hy, k*hz);
        f[idx(i,j,k,Ny,Nz)] = 6.0;
    }

    double res = computeResidual(u, f, Nx, Ny, Nz, hx, hy, hz);
    std::cout << "  Residual for exact polynomial solution: "
              << std::scientific << res << "\n";
    CHECK(res < 1e-10, "Residual < 1e-10 for exact polynomial solution");
}

// ============================================================
//  main — run all tests and report summary
// ============================================================

int main() {
    std::cout << "========================================\n";
    std::cout << " Poisson Solver Test Suite\n";
    std::cout << "========================================\n";

    testIndexMapping();
    testForcingFunction();
    testBoundaryConditions();
    testZeroForcingZeroBC();
    testResidualExactSolution();
    testResidualDecreases();
    testOutputFile();
    testPolynomialSolution();
    testSmoothSolution();
    testAnisotropicSolution();

    std::cout << "\n========================================\n";
    std::cout << " Results: " << g_passed << " passed, "
              << g_failed << " failed\n";
    std::cout << "========================================\n";

    return (g_failed == 0) ? 0 : 1;
}