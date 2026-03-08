/**
 * @file poisson-mpi.cpp
 * @brief Parallel 3D Poisson solver using MPI with a 3-D Cartesian decomposition.
 *
 * Solves the equation:
 * @f[ \nabla^2 u(x,y,z) = f(x,y,z) \quad \text{on} \quad \Omega = [0,1]^3 @f]
 * with Dirichlet boundary conditions using a second-order finite difference
 * discretisation and the Jacobi iterative method.
 *
 * ## Array layout and domain decomposition
 *
 * The global grid has Nx x Ny x Nz points including boundary nodes.
 * These are partitioned into a Px x Py x Pz Cartesian process grid.
 *
 * Each rank owns a contiguous block of **interior** global grid points
 * (i.e. points that the Jacobi sweep actually updates).  The local array
 * also has one layer of ghost/halo cells on every face.  These ghost cells
 * serve two roles:
 *
 *  - For faces shared with a neighbour rank: filled by MPI halo exchange.
 *  - For faces on the global domain boundary: filled with the Dirichlet
 *    BC value and never overwritten by the sweep.
 *
 * With this layout the sweep always reads from ghost cells and writes to
 * owned cells, which is correct and avoids the need to special-case
 * boundary nodes inside the hot loop.
 *
 * ## Communication
 *
 * At every Jacobi iteration six non-blocking Isend/Irecv pairs exchange
 * halo faces using fully packed buffers and direction-unique tags.
 * A single MPI_Allreduce reduces the global L2 residual.
 *
 * @author HPC Assignment
 */

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>
#include <iomanip>
#include <algorithm>

// ============================================================
// Command-line option parser
// ============================================================

/**
 * @brief Command-line options including MPI decomposition parameters.
 */
class Options {
public:
    bool        help    = false;
    std::string forcing = "";
    int         test    = -1;
    int         Nx      = 32;
    int         Ny      = 32;
    int         Nz      = 32;
    double      epsilon = 1e-8;
    int         Px      = 1;
    int         Py      = 1;
    int         Pz      = 1;

    static Options parse(int argc, char* argv[]) {
        Options opt;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if      (a == "--help")                    opt.help    = true;
            else if (a == "--forcing"  && i+1 < argc)  opt.forcing = argv[++i];
            else if (a == "--test"     && i+1 < argc)  opt.test    = std::stoi(argv[++i]);
            else if (a == "--Nx"       && i+1 < argc)  opt.Nx      = std::stoi(argv[++i]);
            else if (a == "--Ny"       && i+1 < argc)  opt.Ny      = std::stoi(argv[++i]);
            else if (a == "--Nz"       && i+1 < argc)  opt.Nz      = std::stoi(argv[++i]);
            else if (a == "--epsilon"  && i+1 < argc)  opt.epsilon = std::stod(argv[++i]);
            else if (a == "--Px"       && i+1 < argc)  opt.Px      = std::stoi(argv[++i]);
            else if (a == "--Py"       && i+1 < argc)  opt.Py      = std::stoi(argv[++i]);
            else if (a == "--Pz"       && i+1 < argc)  opt.Pz      = std::stoi(argv[++i]);
            else throw std::invalid_argument("Unknown option: " + a);
        }
        return opt;
    }

    static void printHelp() {
        std::cout <<
            "Allowed options:\n"
            "  --help            Print available options.\n"
            "  --forcing arg     Input forcing file\n"
            "  --test arg (=1)   Test case to use (1-5)\n"
            "  --Nx arg (=32)    Number of grid points (x)\n"
            "  --Ny arg (=32)    Number of grid points (y)\n"
            "  --Nz arg (=32)    Number of grid points (z)\n"
            "  --epsilon (=1e-8) Residual threshold\n"
            "  --Px arg (=1)     Number of processes (x)\n"
            "  --Py arg (=1)     Number of processes (y)\n"
            "  --Pz arg (=1)     Number of processes (z)\n";
    }
};

// ============================================================
// Grid index helper
// ============================================================

/** @brief Row-major 3-D to 1-D index (k fastest). */
inline int idx(int i, int j, int k, int ny, int nz) {
    return i * ny * nz + j * nz + k;
}

// ============================================================
// Analytical test cases
// ============================================================

/** @brief Exact solution for verification cases 1-3. */
double exactSolution(int tc, double x, double y, double z) {
    switch (tc) {
        case 1: return x*x + y*y + z*z;
        case 2: return std::sin(M_PI*x)*std::sin(M_PI*y)*std::sin(M_PI*z);
        case 3: return std::sin(M_PI*x)*std::sin(4*M_PI*y)*std::sin(8*M_PI*z);
        default: return 0.0;
    }
}

/** @brief Forcing function f(x,y,z) for built-in test cases 1-5. */
double forcingFunction(int tc, double x, double y, double z) {
    switch (tc) {
        case 1: return 6.0;
        case 2: return -3.0*M_PI*M_PI*std::sin(M_PI*x)*std::sin(M_PI*y)*std::sin(M_PI*z);
        case 3: return -(1.0+16.0+64.0)*M_PI*M_PI
                        *std::sin(M_PI*x)*std::sin(4*M_PI*y)*std::sin(8*M_PI*z);
        case 4: return 100.0*std::exp(-100.0*((x-.5)*(x-.5)+(y-.5)*(y-.5)+(z-.5)*(z-.5)));
        case 5: return (x < 0.5) ? 1.0 : -1.0;
        default: throw std::invalid_argument("Unknown test case");
    }
}

/** @brief Prescribed grid dimensions for built-in test cases. */
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

// ============================================================
// Domain decomposition
// ============================================================

/**
 * @brief Compute the local start index and extent for one direction.
 *
 * Distributes the Ni-2 **interior** points of a dimension of size Ni
 * across P processes.  The two boundary nodes (index 0 and Ni-1) are
 * not owned by any rank; they live in halo cells.
 *
 * @param Ni      Total grid points in this direction (including boundaries).
 * @param P       Number of processes in this direction.
 * @param rank    Coordinate of this rank in this direction (0-based).
 * @param[out] gstart  Global index of first interior point owned by this rank.
 * @param[out] count   Number of interior points owned.
 */
void decompose1D(int Ni, int P, int rank, int &gstart, int &count) {
    int interior = Ni - 2;          // strip the two boundary nodes
    int base = interior / P;
    int rem  = interior % P;
    int local_start = rank * base + std::min(rank, rem);  // 0-based among interior
    count  = base + (rank < rem ? 1 : 0);
    gstart = 1 + local_start;       // global index (boundary is index 0)
}

// ============================================================
// Halo exchange
// ============================================================

/**
 * @brief Exchange one layer of halo cells with all face-neighbours.
 *
 * Packs all six faces into dedicated buffers before posting any
 * communication.  Direction-unique tags prevent cross-matching.
 *
 * @param u         Local array including ghost layer, size (lx+2)*(ly+2)*(lz+2).
 * @param lx,ly,lz  Number of locally owned (interior) points.
 * @param cart      Cartesian communicator.
 * @param nbrXm,nbrXp  Neighbour ranks (MPI_PROC_NULL at domain boundary).
 * @param nbrYm,nbrYp,nbrZm,nbrZp  Same for y and z.
 */
void exchangeHalos(std::vector<double>& u,
                   int lx, int ly, int lz,
                   MPI_Comm cart,
                   int nbrXm, int nbrXp,
                   int nbrYm, int nbrYp,
                   int nbrZm, int nbrZp)
{
    const int NY = ly + 2, NZ = lz + 2;
    const int szX = ly * lz, szY = lx * lz, szZ = lx * ly;

    std::vector<double> sXm(szX), sXp(szX), rXm(szX), rXp(szX);
    std::vector<double> sYm(szY), sYp(szY), rYm(szY), rYp(szY);
    std::vector<double> sZm(szZ), sZp(szZ), rZm(szZ), rZp(szZ);

    for (int j=1;j<=ly;++j) for (int k=1;k<=lz;++k) {
        sXm[(j-1)*lz+(k-1)] = u[idx(1,  j,k,NY,NZ)];
        sXp[(j-1)*lz+(k-1)] = u[idx(lx, j,k,NY,NZ)];
    }
    for (int i=1;i<=lx;++i) for (int k=1;k<=lz;++k) {
        sYm[(i-1)*lz+(k-1)] = u[idx(i,1, k,NY,NZ)];
        sYp[(i-1)*lz+(k-1)] = u[idx(i,ly,k,NY,NZ)];
    }
    for (int i=1;i<=lx;++i) for (int j=1;j<=ly;++j) {
        sZm[(i-1)*ly+(j-1)] = u[idx(i,j,1, NY,NZ)];
        sZp[(i-1)*ly+(j-1)] = u[idx(i,j,lz,NY,NZ)];
    }

    MPI_Request reqs[12];
    MPI_Isend(sXm.data(),szX,MPI_DOUBLE,nbrXm,11,cart,&reqs[0]);
    MPI_Irecv(rXm.data(),szX,MPI_DOUBLE,nbrXm,10,cart,&reqs[1]);
    MPI_Isend(sXp.data(),szX,MPI_DOUBLE,nbrXp,10,cart,&reqs[2]);
    MPI_Irecv(rXp.data(),szX,MPI_DOUBLE,nbrXp,11,cart,&reqs[3]);
    MPI_Isend(sYm.data(),szY,MPI_DOUBLE,nbrYm,21,cart,&reqs[4]);
    MPI_Irecv(rYm.data(),szY,MPI_DOUBLE,nbrYm,20,cart,&reqs[5]);
    MPI_Isend(sYp.data(),szY,MPI_DOUBLE,nbrYp,20,cart,&reqs[6]);
    MPI_Irecv(rYp.data(),szY,MPI_DOUBLE,nbrYp,21,cart,&reqs[7]);
    MPI_Isend(sZm.data(),szZ,MPI_DOUBLE,nbrZm,31,cart,&reqs[8]);
    MPI_Irecv(rZm.data(),szZ,MPI_DOUBLE,nbrZm,30,cart,&reqs[9]);
    MPI_Isend(sZp.data(),szZ,MPI_DOUBLE,nbrZp,30,cart,&reqs[10]);
    MPI_Irecv(rZp.data(),szZ,MPI_DOUBLE,nbrZp,31,cart,&reqs[11]);
    MPI_Waitall(12,reqs,MPI_STATUSES_IGNORE);

    for (int j=1;j<=ly;++j) for (int k=1;k<=lz;++k) {
        u[idx(0,   j,k,NY,NZ)] = rXm[(j-1)*lz+(k-1)];
        u[idx(lx+1,j,k,NY,NZ)] = rXp[(j-1)*lz+(k-1)];
    }
    for (int i=1;i<=lx;++i) for (int k=1;k<=lz;++k) {
        u[idx(i,0,   k,NY,NZ)] = rYm[(i-1)*lz+(k-1)];
        u[idx(i,ly+1,k,NY,NZ)] = rYp[(i-1)*lz+(k-1)];
    }
    for (int i=1;i<=lx;++i) for (int j=1;j<=ly;++j) {
        u[idx(i,j,0,   NY,NZ)] = rZm[(i-1)*ly+(j-1)];
        u[idx(i,j,lz+1,NY,NZ)] = rZp[(i-1)*ly+(j-1)];
    }
}

// ============================================================
// Local Jacobi sweep  (owned points only: i,j,k in [1..l*])
// ============================================================

/**
 * @brief One Jacobi sweep over all locally owned points.
 *
 * Ghost cells (index 0 and l*+1 in each direction) are read but
 * never written.  They contain either BC values (global boundary)
 * or values received from neighbours (internal faces).
 *
 * @param u,unew  Local arrays size (lx+2)*(ly+2)*(lz+2).
 * @param f       Local forcing (same size; ghost cells unused).
 * @param lx,ly,lz  Owned point counts.
 * @param hx,hy,hz  Grid spacings.
 */
void localJacobiSweep(const std::vector<double>& u,
                      const std::vector<double>& f,
                      std::vector<double>&       unew,
                      int lx, int ly, int lz,
                      double hx, double hy, double hz)
{
    const int NY=ly+2, NZ=lz+2;
    const double ihx2=1./(hx*hx), ihy2=1./(hy*hy), ihz2=1./(hz*hz);
    const double denom=2.*(ihx2+ihy2+ihz2);

    for (int i=1;i<=lx;++i)
        for (int j=1;j<=ly;++j)
            for (int k=1;k<=lz;++k) {
                double rhs =
                    (u[idx(i+1,j,k,NY,NZ)]+u[idx(i-1,j,k,NY,NZ)])*ihx2
                   +(u[idx(i,j+1,k,NY,NZ)]+u[idx(i,j-1,k,NY,NZ)])*ihy2
                   +(u[idx(i,j,k+1,NY,NZ)]+u[idx(i,j,k-1,NY,NZ)])*ihz2
                   -f[idx(i,j,k,NY,NZ)];
                unew[idx(i,j,k,NY,NZ)] = rhs/denom;
            }
}

// ============================================================
// Local residual (squared sum)
// ============================================================

/**
 * @brief Local contribution to the global squared L2 residual.
 * @return  Sum of r^2 over owned interior points.
 */
double localResidualSq(const std::vector<double>& u,
                       const std::vector<double>& f,
                       int lx, int ly, int lz,
                       double hx, double hy, double hz)
{
    const int NY=ly+2, NZ=lz+2;
    const double ihx2=1./(hx*hx), ihy2=1./(hy*hy), ihz2=1./(hz*hz);
    double s=0.;
    for (int i=1;i<=lx;++i)
        for (int j=1;j<=ly;++j)
            for (int k=1;k<=lz;++k) {
                double lap =
                    (u[idx(i+1,j,k,NY,NZ)]-2*u[idx(i,j,k,NY,NZ)]+u[idx(i-1,j,k,NY,NZ)])*ihx2
                   +(u[idx(i,j+1,k,NY,NZ)]-2*u[idx(i,j,k,NY,NZ)]+u[idx(i,j-1,k,NY,NZ)])*ihy2
                   +(u[idx(i,j,k+1,NY,NZ)]-2*u[idx(i,j,k,NY,NZ)]+u[idx(i,j,k-1,NY,NZ)])*ihz2;
                double r = f[idx(i,j,k,NY,NZ)] - lap;
                s += r*r;
            }
    return s;
}

// ============================================================
// I/O (rank 0 only)
// ============================================================

/** @brief Read forcing file. Format: "Nx Ny Nz\n" then Nx*Ny*Nz "x y z f" lines. */
void readForcingFile(const std::string& fn, int& Nx, int& Ny, int& Nz,
                     std::vector<double>& f)
{
    std::ifstream fin(fn);
    if (!fin) throw std::runtime_error("Cannot open: " + fn);
    fin >> Nx >> Ny >> Nz;
    f.assign(Nx*Ny*Nz, 0.);
    double hx=1./(Nx-1), hy=1./(Ny-1), hz=1./(Nz-1);
    for (int n=0; n<Nx*Ny*Nz; ++n) {
        double x,y,z,v; fin>>x>>y>>z>>v;
        int i=int(std::round(x/hx)), j=int(std::round(y/hy)), k=int(std::round(z/hz));
        f[i*Ny*Nz+j*Nz+k]=v;
    }
}

/** @brief Write solution file. Format: "Nx Ny Nz\n" then Nx*Ny*Nz "x y z u" lines. */
void writeSolution(const std::string& fn, const std::vector<double>& u,
                   int Nx, int Ny, int Nz, double hx, double hy, double hz)
{
    std::ofstream fout(fn);
    if (!fout) throw std::runtime_error("Cannot open: " + fn);
    fout << Nx << " " << Ny << " " << Nz << "\n" << std::scientific << std::setprecision(12);
    for (int i=0;i<Nx;++i) for (int j=0;j<Ny;++j) for (int k=0;k<Nz;++k)
        fout << i*hx << " " << j*hy << " " << k*hz << " " << u[i*Ny*Nz+j*Nz+k] << "\n";
}

// ============================================================
// Main
// ============================================================

/**
 * @brief MPI program entry point.
 *
 * Array layout:
 *  - Each rank owns the **interior** points of its sub-block.
 *    Global boundary nodes (i=0, i=Nx-1, etc.) are never owned;
 *    they live in the halo (ghost) layer at local index 0 or l*+1.
 *  - Before the first iteration the halo is initialised:
 *      * Global-boundary faces: set to the Dirichlet BC value (permanent).
 *      * Internal-boundary faces: zero (will be filled by halo exchange).
 *  - The halo exchange only updates internal faces; global-boundary halo
 *    cells are left alone (MPI_PROC_NULL sends/receives are no-ops).
 *
 * @param argc,argv  As passed from the OS.
 * @return 0 on success.
 */
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ---- Parse options ----
    Options opt;
    try { opt = Options::parse(argc, argv); }
    catch (const std::exception& e) {
        if (rank==0) { std::cerr<<"Error: "<<e.what()<<"\n"; Options::printHelp(); }
        MPI_Finalize(); return 1;
    }
    if (opt.help) { if (rank==0) Options::printHelp(); MPI_Finalize(); return 0; }
    if (opt.Px*opt.Py*opt.Pz != size) {
        if (rank==0) std::cerr<<"Error: Px*Py*Pz="<<opt.Px*opt.Py*opt.Pz<<" but P="<<size<<"\n";
        MPI_Finalize(); return 1;
    }
    bool hasForcing=!opt.forcing.empty(), hasTest=(opt.test!=-1);
    if (hasForcing==hasTest) {
        if (rank==0) std::cerr<<"Error: provide exactly one of --forcing or --test.\n";
        MPI_Finalize(); return 1;
    }

    // ---- Cartesian communicator (reorder=0: ranks stay the same) ----
    int dims[3]={opt.Px,opt.Py,opt.Pz}, periods[3]={0,0,0};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD,3,dims,periods,0,&cart);
    if(rank==0) std::cerr<<"[checkpoint] cart created\n"<<std::flush;
    int coords[3];
    MPI_Cart_coords(cart,rank,3,coords);
    if(rank==0) std::cerr<<"[checkpoint] coords ok\n"<<std::flush;
    int cx=coords[0], cy=coords[1], cz=coords[2];

    int nbrXm,nbrXp,nbrYm,nbrYp,nbrZm,nbrZp;
    MPI_Cart_shift(cart,0,1,&nbrXm,&nbrXp);
    MPI_Cart_shift(cart,1,1,&nbrYm,&nbrYp);
    MPI_Cart_shift(cart,2,1,&nbrZm,&nbrZp);

    // ---- Grid dimensions ----
    int Nx=0,Ny=0,Nz=0;
    std::vector<double> globalF;
    if (rank==0) {
        if (hasTest) {
            testCaseGrid(opt.test,Nx,Ny,Nz);
            globalF.resize(Nx*Ny*Nz);
            double hx=1./(Nx-1),hy=1./(Ny-1),hz=1./(Nz-1);
            for (int i=0;i<Nx;++i) for (int j=0;j<Ny;++j) for (int k=0;k<Nz;++k)
                globalF[i*Ny*Nz+j*Nz+k]=forcingFunction(opt.test,i*hx,j*hy,k*hz);
        } else {
            readForcingFile(opt.forcing,Nx,Ny,Nz,globalF);
        }
    }
    { int g[3]={Nx,Ny,Nz}; MPI_Bcast(g,3,MPI_INT,0,MPI_COMM_WORLD); Nx=g[0];Ny=g[1];Nz=g[2]; }
    if(rank==0) std::cerr<<"[checkpoint] bcast ok Nx="<<Nx<<" Ny="<<Ny<<" Nz="<<Nz<<"\n"<<std::flush;

    const double hx=1./(Nx-1), hy=1./(Ny-1), hz=1./(Nz-1);

    // ---- Local domain: interior points only ----
    // gxs = global index of first owned point in x, lx = count
    int gxs,lx, gys,ly, gzs,lz;
    decompose1D(Nx,opt.Px,cx,gxs,lx);
    decompose1D(Ny,opt.Py,cy,gys,ly);
    decompose1D(Nz,opt.Pz,cz,gzs,lz);

    // Local array includes 1 ghost cell on each side
    const int LNY=ly+2, LNZ=lz+2;
    const int localN=(lx+2)*LNY*LNZ;
    const int ownedN=lx*ly*lz;

    // ---- Scatter forcing to each rank ----
    std::vector<double> localF(localN,0.);
    {
        if (rank==0) {
            for (int r=1;r<size;++r) {
                int rc[3]; MPI_Cart_coords(cart,r,3,rc);
                int xs,lxr,ys,lyr,zs,lzr;
                decompose1D(Nx,opt.Px,rc[0],xs,lxr);
                decompose1D(Ny,opt.Py,rc[1],ys,lyr);
                decompose1D(Nz,opt.Pz,rc[2],zs,lzr);
                std::vector<double> buf(lxr*lyr*lzr);
                int off=0;
                for (int i=0;i<lxr;++i) for (int j=0;j<lyr;++j) for (int k=0;k<lzr;++k)
                    buf[off++]=globalF[(xs+i)*Ny*Nz+(ys+j)*Nz+(zs+k)];
                MPI_Send(buf.data(),lxr*lyr*lzr,MPI_DOUBLE,r,0,MPI_COMM_WORLD);
            }
            int off=0;
            for (int i=0;i<lx;++i) for (int j=0;j<ly;++j) for (int k=0;k<lz;++k)
                localF[idx(i+1,j+1,k+1,LNY,LNZ)]=globalF[(gxs+i)*Ny*Nz+(gys+j)*Nz+(gzs+k)];
            (void)off;
        } else {
            std::vector<double> buf(ownedN);
            MPI_Recv(buf.data(),ownedN,MPI_DOUBLE,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            int off=0;
            for (int i=0;i<lx;++i) for (int j=0;j<ly;++j) for (int k=0;k<lz;++k)
                localF[idx(i+1,j+1,k+1,LNY,LNZ)]=buf[off++];
        }
    }

    if(rank==0) std::cerr<<"[checkpoint] scatter done\n"<<std::flush;
    if (rank==0) std::cout << "Forcing data distributed. Initialising u...\n" << std::flush;

    // ---- Initialise u ----
    // Interior (owned) points: u = 0
    // Ghost cells on global boundary faces: u = Dirichlet BC value
    // Ghost cells on internal faces: u = 0  (will be filled by exchange)
    std::vector<double> u(localN,0.), unew(localN,0.);

    // BC value at global grid indices (gi, gj, gk)
    auto bcVal=[&](int gi,int gj,int gk)->double {
        double x=gi*hx,y=gj*hy,z=gk*hz;
        return hasTest ? exactSolution(opt.test,x,y,z) : 0.;
    };

    // Stamp BC into ghost cells on the global boundary
    // Global x=0 boundary: local ghost layer i=0, owned starts at global gxs
    if (cx==0)
        for (int j=1;j<=ly;++j) for (int k=1;k<=lz;++k)
            u[idx(0,j,k,LNY,LNZ)] = unew[idx(0,j,k,LNY,LNZ)]
                = bcVal(0, gys+j-1, gzs+k-1);
    // Global x=Nx-1 boundary: local ghost layer i=lx+1
    if (cx==opt.Px-1)
        for (int j=1;j<=ly;++j) for (int k=1;k<=lz;++k)
            u[idx(lx+1,j,k,LNY,LNZ)] = unew[idx(lx+1,j,k,LNY,LNZ)]
                = bcVal(Nx-1, gys+j-1, gzs+k-1);
    // Global y=0
    if (cy==0)
        for (int i=1;i<=lx;++i) for (int k=1;k<=lz;++k)
            u[idx(i,0,k,LNY,LNZ)] = unew[idx(i,0,k,LNY,LNZ)]
                = bcVal(gxs+i-1, 0, gzs+k-1);
    // Global y=Ny-1
    if (cy==opt.Py-1)
        for (int i=1;i<=lx;++i) for (int k=1;k<=lz;++k)
            u[idx(i,ly+1,k,LNY,LNZ)] = unew[idx(i,ly+1,k,LNY,LNZ)]
                = bcVal(gxs+i-1, Ny-1, gzs+k-1);
    // Global z=0
    if (cz==0)
        for (int i=1;i<=lx;++i) for (int j=1;j<=ly;++j)
            u[idx(i,j,0,LNY,LNZ)] = unew[idx(i,j,0,LNY,LNZ)]
                = bcVal(gxs+i-1, gys+j-1, 0);
    // Global z=Nz-1
    if (cz==opt.Pz-1)
        for (int i=1;i<=lx;++i) for (int j=1;j<=ly;++j)
            u[idx(i,j,lz+1,LNY,LNZ)] = unew[idx(i,j,lz+1,LNY,LNZ)]
                = bcVal(gxs+i-1, gys+j-1, Nz-1);

    // ---- Jacobi iteration ----
    if(rank==0) std::cerr<<"[checkpoint] BC init done\n"<<std::flush;
    if (rank==0) {
        std::cout << "Starting Jacobi iteration on " << size << " rank(s)\n"
                  << "  Grid: " << Nx << "x" << Ny << "x" << Nz
                  << "  Local: " << lx << "x" << ly << "x" << lz
                  << "  epsilon=" << opt.epsilon << "\n" << std::flush;
    }
    double residual=0.; int iter=0;
    do {
        // Halo exchange (internal faces only; global-boundary ghosts unchanged)
        exchangeHalos(u,lx,ly,lz,cart,nbrXm,nbrXp,nbrYm,nbrYp,nbrZm,nbrZp);

        // Sweep over owned points
        localJacobiSweep(u,localF,unew,lx,ly,lz,hx,hy,hz);

        std::swap(u,unew);

        // Global residual
        double lsq=localResidualSq(u,localF,lx,ly,lz,hx,hy,hz), gsq=0.;
        MPI_Allreduce(&lsq,&gsq,1,MPI_DOUBLE,MPI_SUM,cart);
        residual=std::sqrt(gsq);
        ++iter;

        if (rank==0 && iter%500==0) {
            std::cout << "  iter " << std::setw(7) << iter
                      << "  |r| = " << std::scientific << std::setprecision(3) << residual
                      << "  (target " << opt.epsilon << ")" << std::endl;
        }

    } while (residual > opt.epsilon);

    // ---- Gather and write solution ----
    {
        std::vector<double> sendBuf(ownedN);
        int off=0;
        for (int i=0;i<lx;++i) for (int j=0;j<ly;++j) for (int k=0;k<lz;++k)
            sendBuf[off++]=u[idx(i+1,j+1,k+1,LNY,LNZ)];

        if (rank==0) {
            std::vector<double> globalU(Nx*Ny*Nz,0.);

            // Boundary nodes: fill from BC
            // x faces
            for (int j=0;j<Ny;++j) for (int k=0;k<Nz;++k) {
                globalU[0*Ny*Nz+j*Nz+k]       = bcVal(0,   j,k);
                globalU[(Nx-1)*Ny*Nz+j*Nz+k]  = bcVal(Nx-1,j,k);
            }
            // y faces
            for (int i=0;i<Nx;++i) for (int k=0;k<Nz;++k) {
                globalU[i*Ny*Nz+0*Nz+k]       = bcVal(i,0,   k);
                globalU[i*Ny*Nz+(Ny-1)*Nz+k]  = bcVal(i,Ny-1,k);
            }
            // z faces
            for (int i=0;i<Nx;++i) for (int j=0;j<Ny;++j) {
                globalU[i*Ny*Nz+j*Nz+0]       = bcVal(i,j,0);
                globalU[i*Ny*Nz+j*Nz+(Nz-1)]  = bcVal(i,j,Nz-1);
            }

            // Rank 0 interior
            off=0;
            for (int i=0;i<lx;++i) for (int j=0;j<ly;++j) for (int k=0;k<lz;++k)
                globalU[(gxs+i)*Ny*Nz+(gys+j)*Nz+(gzs+k)]=sendBuf[off++];

            // Other ranks
            for (int r=1;r<size;++r) {
                int rc[3]; MPI_Cart_coords(cart,r,3,rc);
                int xs,lxr,ys,lyr,zs,lzr;
                decompose1D(Nx,opt.Px,rc[0],xs,lxr);
                decompose1D(Ny,opt.Py,rc[1],ys,lyr);
                decompose1D(Nz,opt.Pz,rc[2],zs,lzr);
                int nr=lxr*lyr*lzr;
                std::vector<double> buf(nr);
                MPI_Recv(buf.data(),nr,MPI_DOUBLE,r,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                off=0;
                for (int i=0;i<lxr;++i) for (int j=0;j<lyr;++j) for (int k=0;k<lzr;++k)
                    globalU[(xs+i)*Ny*Nz+(ys+j)*Nz+(zs+k)]=buf[off++];
            }

            writeSolution("solution.txt",globalU,Nx,Ny,Nz,hx,hy,hz);
            std::cout<<std::scientific<<std::setprecision(6)
                     <<"Final residual: "<<residual<<" (after "<<iter<<" iterations)\n";
        } else {
            MPI_Send(sendBuf.data(),ownedN,MPI_DOUBLE,0,1,MPI_COMM_WORLD);
        }
    }

    MPI_Comm_free(&cart);
    MPI_Finalize();
    return 0;
}