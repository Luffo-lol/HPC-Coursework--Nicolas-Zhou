/**
 * @file poisson.cpp
 * @brief Serial 3D Poisson equation solver using the Jacobi iterative method.
 *
 * Solves the equation:
 * @f[ \nabla^2 u(x,y,z) = f(x,y,z) \quad \text{on} \quad \Omega = [0,1]^3 @f]
 * with Dirichlet boundary conditions, using a second-order finite difference
 * discretisation and Jacobi iteration.
 *
 * @author Nicolas Zhou
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>
#include <iomanip>

// Command-line argument parsing (lightweight, no Boost needed)

/**
 * @brief Simple command-line option parser.
 *
 * Supports --flag and --key value style arguments.
 */
class Options {
public:
    bool        help    = false;
    std::string forcing = "";   ///< Path to an input forcing file (optional)
    int         test    = -1;   ///< Test case number 1–5 (-1 = not set)
    int         Nx      = 32;   ///< Number of grid points in x
    int         Ny      = 32;   ///< Number of grid points in y
    int         Nz      = 32;   ///< Number of grid points in z
    double      epsilon = 1e-8; ///< Convergence threshold for the residual

    /**
     * @brief Parse command-line arguments into an Options struct.
     * @param argc Argument count from main().
     * @param argv Argument values from main().
     * @return Populated Options object.
     * @throws std::invalid_argument on bad input.
     */
    static Options parse(int argc, char* argv[]) {
        Options opt;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--help") {
                opt.help = true;
            } else if (arg == "--forcing" && i + 1 < argc) {
                opt.forcing = argv[++i];
            } else if (arg == "--test" && i + 1 < argc) {
                opt.test = std::stoi(argv[++i]);
            } else if (arg == "--Nx" && i + 1 < argc) {
                opt.Nx = std::stoi(argv[++i]);
            } else if (arg == "--Ny" && i + 1 < argc) {
                opt.Ny = std::stoi(argv[++i]);
            } else if (arg == "--Nz" && i + 1 < argc) {
                opt.Nz = std::stoi(argv[++i]);
            } else if (arg == "--epsilon" && i + 1 < argc) {
                opt.epsilon = std::stod(argv[++i]);
            } else {
                throw std::invalid_argument("Unknown option: " + arg);
            }
        }
        return opt;
    }

    /// Print help text to stdout.
    static void printHelp() {
        std::cout <<
            "Allowed options:\n"
            "  --help            Print available options.\n"
            "  --forcing arg     Input forcing file\n"
            "  --test arg (=1)   Test case to use (1-5)\n"
            "  --Nx arg (=32)    Number of grid points (x)\n"
            "  --Ny arg (=32)    Number of grid points (y)\n"
            "  --Nz arg (=32)    Number of grid points (z)\n"
            "  --epsilon (=1e-8) Residual threshold\n";
    }
};

// Grid helper — maps 3D index (i,j,k) to 1D flat index

/**
 * @brief Inline 3-D to 1-D index mapping (row-major, k fastest).
 * @param i  Index in x direction (0 … Nx-1).
 * @param j  Index in y direction (0 … Ny-1).
 * @param k  Index in z direction (0 … Nz-1).
 * @param Ny Number of grid points in y.
 * @param Nz Number of grid points in z.
 * @return   Flat array index.
 */
inline int idx(int i, int j, int k, int Ny, int Nz) {
    return i * Ny * Nz + j * Nz + k;
}

// Forcing function definitions for built-in test cases

/**
 * @brief Exact solution for test cases that have one.
 * @param tc  Test case number (1–3).
 * @param x,y,z Grid coordinates.
 * @return Exact value of u at (x,y,z), or 0 if not applicable.
 */
double exactSolution(int tc, double x, double y, double z) {
    switch (tc) {
        case 1: return x*x + y*y + z*z;
        case 2: return std::sin(M_PI*x) * std::sin(M_PI*y) * std::sin(M_PI*z);
        case 3: return std::sin(M_PI*x) * std::sin(4*M_PI*y) * std::sin(8*M_PI*z);
        default: return 0.0;
    }
}

/**
 * @brief Forcing function f(x,y,z) for built-in test cases.
 * @param tc  Test case number (1–5).
 * @param x,y,z Grid coordinates.
 * @return Value of f(x,y,z).
 */
double forcingFunction(int tc, double x, double y, double z) {
    switch (tc) {
        case 1: return 6.0;
        case 2: return -3.0 * M_PI*M_PI * std::sin(M_PI*x) * std::sin(M_PI*y) * std::sin(M_PI*z);
        case 3: return -(1.0 + 16.0 + 64.0) * M_PI*M_PI
                        * std::sin(M_PI*x) * std::sin(4*M_PI*y) * std::sin(8*M_PI*z);
        case 4: return 100.0 * std::exp(-100.0 * ((x-0.5)*(x-0.5)
                                                  +(y-0.5)*(y-0.5)
                                                  +(z-0.5)*(z-0.5)));
        case 5: return (x < 0.5) ? 1.0 : -1.0;
        default: throw std::invalid_argument("Unknown test case");
    }
}

// Grid dimensions for each built-in test case

/**
 * @brief Return the prescribed grid dimensions for a built-in test case.
 * @param tc Test case number (1–5).
 * @param[out] Nx,Ny,Nz Grid dimensions.
 */
void testCaseGrid(int tc, int &Nx, int &Ny, int &Nz) {
    switch (tc) {
        case 1: Nx=32;  Ny=32;  Nz=32;  break;
        case 2: Nx=64;  Ny=64;  Nz=64;  break;
        case 3: Nx=32;  Ny=64;  Nz=128; break;
        case 4: Nx=64;  Ny=64;  Nz=64;  break;
        case 5: Nx=64;  Ny=64;  Nz=64;  break;
        default: throw std::invalid_argument("Unknown test case");
    }
}

// Core solver
/**
 * @brief Perform a single Jacobi sweep over all interior grid points.
 *
 * Updates every interior point according to:
 * @f[
 *   u^{(n+1)}_{i,j,k} = \frac{1}{2(h_x^{-2}+h_y^{-2}+h_z^{-2})}
 *   \left(
 *     \frac{u^{(n)}_{i+1,j,k}+u^{(n)}_{i-1,j,k}}{h_x^2}
 *    +\frac{u^{(n)}_{i,j+1,k}+u^{(n)}_{i,j-1,k}}{h_y^2}
 *    +\frac{u^{(n)}_{i,j,k+1}+u^{(n)}_{i,j,k-1}}{h_z^2}
 *    - f_{i,j,k}
 *   \right)
 * @f]
 *
 * @param u     Current solution vector (length Nx*Ny*Nz).
 * @param f     Forcing function values at interior nodes.
 * @param unew  Receives the updated solution (same size as u).
 * @param Nx,Ny,Nz Grid dimensions.
 * @param hx,hy,hz Grid spacings.
 */
void jacobiSweep(const std::vector<double>& u,
                 const std::vector<double>& f,
                 std::vector<double>&       unew,
                 int Nx, int Ny, int Nz,
                 double hx, double hy, double hz)
{
    const double ihx2 = 1.0 / (hx*hx);
    const double ihy2 = 1.0 / (hy*hy);
    const double ihz2 = 1.0 / (hz*hz);
    const double denom = 2.0 * (ihx2 + ihy2 + ihz2);

    for (int i = 1; i < Nx-1; ++i) {
        for (int j = 1; j < Ny-1; ++j) {
            for (int k = 1; k < Nz-1; ++k) {
                double rhs = (u[idx(i+1,j,k,Ny,Nz)] + u[idx(i-1,j,k,Ny,Nz)]) * ihx2
                           + (u[idx(i,j+1,k,Ny,Nz)] + u[idx(i,j-1,k,Ny,Nz)]) * ihy2
                           + (u[idx(i,j,k+1,Ny,Nz)] + u[idx(i,j,k-1,Ny,Nz)]) * ihz2
                           - f[idx(i,j,k,Ny,Nz)];
                unew[idx(i,j,k,Ny,Nz)] = rhs / denom;
            }
        }
    }
}

/**
 * @brief Compute the discrete L2 norm of the residual r = f - ∇²_h u.
 *
 * The discrete Laplacian is applied at every interior node and compared
 * with the forcing function value.
 *
 * @param u     Current solution vector.
 * @param f     Forcing function values.
 * @param Nx,Ny,Nz Grid dimensions.
 * @param hx,hy,hz Grid spacings.
 * @return      L2 norm of the residual.
 */
double computeResidual(const std::vector<double>& u,
                       const std::vector<double>& f,
                       int Nx, int Ny, int Nz,
                       double hx, double hy, double hz)
{
    const double ihx2 = 1.0 / (hx*hx);
    const double ihy2 = 1.0 / (hy*hy);
    const double ihz2 = 1.0 / (hz*hz);

    double sumSq = 0.0;
    for (int i = 1; i < Nx-1; ++i) {
        for (int j = 1; j < Ny-1; ++j) {
            for (int k = 1; k < Nz-1; ++k) {
                double lap = (u[idx(i+1,j,k,Ny,Nz)] - 2*u[idx(i,j,k,Ny,Nz)] + u[idx(i-1,j,k,Ny,Nz)]) * ihx2
                           + (u[idx(i,j+1,k,Ny,Nz)] - 2*u[idx(i,j,k,Ny,Nz)] + u[idx(i,j-1,k,Ny,Nz)]) * ihy2
                           + (u[idx(i,j,k+1,Ny,Nz)] - 2*u[idx(i,j,k,Ny,Nz)] + u[idx(i,j,k-1,Ny,Nz)]) * ihz2;
                double r = f[idx(i,j,k,Ny,Nz)] - lap;
                sumSq += r * r;
            }
        }
    }
    return std::sqrt(sumSq);
}

// I/O helpers

/**
 * @brief Read the forcing function from a text file.
 *
 * Expected file format:
 * - Line 1: Nx Ny Nz (integers, space-separated)
 * - Subsequent Nx*Ny*Nz lines: x y z f(x,y,z) (floats, space-separated)
 *
 * @param filename  Path to the input file.
 * @param[out] Nx,Ny,Nz Grid dimensions read from the file.
 * @param[out] f    Flat array of forcing values, indexed by idx().
 */
void readForcingFile(const std::string& filename,
                     int& Nx, int& Ny, int& Nz,
                     std::vector<double>& f)
{
    std::ifstream fin(filename);
    if (!fin) throw std::runtime_error("Cannot open forcing file: " + filename);

    fin >> Nx >> Ny >> Nz;
    int N = Nx * Ny * Nz;
    f.assign(N, 0.0);

    for (int n = 0; n < N; ++n) {
        double x, y, z, val;
        fin >> x >> y >> z >> val;
        // Determine integer indices from coordinates
        // (coordinates are on a uniform [0,1]^3 grid)
        double hx = 1.0 / (Nx - 1);
        double hy = 1.0 / (Ny - 1);
        double hz = 1.0 / (Nz - 1);
        int i = static_cast<int>(std::round(x / hx));
        int j = static_cast<int>(std::round(y / hy));
        int k = static_cast<int>(std::round(z / hz));
        f[idx(i, j, k, Ny, Nz)] = val;
    }
}

/**
 * @brief Write the solution u(x,y,z) to a text file.
 *
 * Output format matches the forcing file format:
 * - Line 1: Nx Ny Nz
 * - Subsequent lines: x y z u(x,y,z)
 *
 * @param filename  Output file path (typically "solution.txt").
 * @param u         Flat solution array.
 * @param Nx,Ny,Nz  Grid dimensions.
 * @param hx,hy,hz  Grid spacings.
 */
void writeSolution(const std::string& filename,
                   const std::vector<double>& u,
                   int Nx, int Ny, int Nz,
                   double hx, double hy, double hz)
{
    std::ofstream fout(filename);
    if (!fout) throw std::runtime_error("Cannot open output file: " + filename);

    fout << Nx << " " << Ny << " " << Nz << "\n";
    fout << std::scientific << std::setprecision(12);
    for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < Ny; ++j)
            for (int k = 0; k < Nz; ++k)
                fout << i*hx << " " << j*hy << " " << k*hz << " "
                     << u[idx(i,j,k,Ny,Nz)] << "\n";
}

// Main

/**
 * @brief Program entry point.
 *
 * Parses options, sets up the grid and boundary conditions, runs the
 * Jacobi iteration until the residual is below the tolerance, and
 * writes the solution to solution.txt.
 *
 * @param argc Argument count.
 * @param argv Argument values.
 * @return 0 on success, non-zero on error.
 */
int main(int argc, char* argv[]) {
    Options opt;
    try {
        opt = Options::parse(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error parsing options: " << e.what() << "\n";
        Options::printHelp();
        return 1;
    }

    if (opt.help) {
        Options::printHelp();
        return 0;
    }

    // Validate: exactly one of --forcing or --test must be given
    bool hasForcing = !opt.forcing.empty();
    bool hasTest    = (opt.test != -1);
    if (hasForcing == hasTest) {
        std::cerr << "Error: provide exactly one of --forcing or --test.\n";
        Options::printHelp();
        return 1;
    }

    // ---- Grid setup ----
    int Nx = opt.Nx, Ny = opt.Ny, Nz = opt.Nz;
    std::vector<double> f;

    if (hasTest) {
        testCaseGrid(opt.test, Nx, Ny, Nz);
        int N = Nx * Ny * Nz;
        f.assign(N, 0.0);
        double hx = 1.0 / (Nx - 1);
        double hy = 1.0 / (Ny - 1);
        double hz = 1.0 / (Nz - 1);
        for (int i = 0; i < Nx; ++i)
            for (int j = 0; j < Ny; ++j)
                for (int k = 0; k < Nz; ++k)
                    f[idx(i,j,k,Ny,Nz)] = forcingFunction(opt.test,
                                                           i*hx, j*hy, k*hz);
    } else {
        readForcingFile(opt.forcing, Nx, Ny, Nz, f);
    }

    double hx = 1.0 / (Nx - 1);
    double hy = 1.0 / (Ny - 1);
    double hz = 1.0 / (Nz - 1);
    int N = Nx * Ny * Nz;

    //  Allocate solution arrays 
    std::vector<double> u(N, 0.0);
    std::vector<double> unew(N, 0.0);

    // Apply Dirichlet boundary conditions 
    // For test cases 1–3 use the exact solution; otherwise u=0 on boundary.
    auto applyBC = [&](std::vector<double>& vec) {
        for (int i = 0; i < Nx; ++i) {
            for (int j = 0; j < Ny; ++j) {
                for (int k = 0; k < Nz; ++k) {
                    bool isBoundary = (i == 0 || i == Nx-1 ||
                                       j == 0 || j == Ny-1 ||
                                       k == 0 || k == Nz-1);
                    if (!isBoundary) continue;
                    double x = i*hx, y = j*hy, z = k*hz;
                    vec[idx(i,j,k,Ny,Nz)] = hasTest
                        ? exactSolution(opt.test, x, y, z)
                        : 0.0;
                }
            }
        }
    };

    applyBC(u);
    applyBC(unew);

    //  Jacobi iteration 
    double residual = 0.0;
    int iter = 0;
    do {
        jacobiSweep(u, f, unew, Nx, Ny, Nz, hx, hy, hz);
        // Enforce BCs on new iterate (boundary values must not drift)
        applyBC(unew);
        std::swap(u, unew);
        residual = computeResidual(u, f, Nx, Ny, Nz, hx, hy, hz);
        ++iter;
    } while (residual > opt.epsilon);

    //  Output 
    writeSolution("solution.txt", u, Nx, Ny, Nz, hx, hy, hz);
    std::cout << std::scientific << std::setprecision(6)
              << "Final residual: " << residual
              << " (after " << iter << " iterations)\n";

    return 0;
}