/**
 * @file poisson-cuda.cu
 * @brief MPI + CUDA Poisson solver.
 *
 * Extends the optimised MPI solver (opt3) to offload the two most
 * compute-intensive kernels — the Jacobi sweep and the residual
 * computation — to a GPU using CUDA.
 *
 * ## GPU offload strategy
 *
 * Each MPI rank owns one contiguous subdomain of the global 3-D grid.
 * At start-up the rank's local arrays (u, unew, localF) are allocated
 * in device memory and initialised with cudaMemcpy.  Every iteration:
 *
 *   1. Halo exchange:   device -> host (cudaMemcpy) for the six face
 *                       buffers, MPI send/recv as before, host ->device.
 *   2. Jacobi sweep:    jacobiKernel<<<grid,block>>>> writes unew on device.
 *   3. Swap:            device pointer swap (no data movement).
 *   4. Residual:        residualKernel<<<grid,block>>> + thrust::reduce
 *                       (or atomicAdd into a device scalar) for local sum.
 *   5. MPI_Allreduce:   sum squared residuals across ranks (host scalar).
 *
 * The host is only involved in MPI communication; all arithmetic stays
 * on the GPU.  This minimises PCIe traffic: only halo face data
 * (O(N²) doubles per face) cross the bus each iteration, rather than
 * the full O(N³) volume.
 *
 * ## Thread / block layout
 *
 * The local domain (lx+2)×(ly+2)×(lz+2) is mapped to a 3-D CUDA grid:
 *   - Each thread handles one (i,j,k) interior point.
 *   - Block size: BLOCK_X × BLOCK_Y × BLOCK_Z threads (default 8×8×8 = 512).
 *   - Grid dims:  ceil(lx/BLOCK_X) × ceil(ly/BLOCK_Y) × ceil(lz/BLOCK_Z).
 *
 * Shared memory is not used; the 7-point stencil has low reuse and the
 * working set for a typical subdomain fits comfortably in L2 cache on
 * modern GPUs (A100 has 40 MB L2; a 32³ subdomain needs 256 KB).
 *
 * ## Halo handling on the GPU
 *
 * The halo exchange must proceed via the host because MPI cannot directly
 * address device memory without CUDA-aware MPI.  The strategy is:
 *   - X faces are contiguous in row-major layout — copy the face slice
 *     directly with cudaMemcpy.
 *   - Y and Z faces are non-contiguous and must be packed/unpacked.
 *     Two auxiliary device kernels (packY, packZ, unpackY, unpackZ)
 *     perform the gather/scatter on the GPU before/after the MPI call,
 *     avoiding a full device→host copy of the entire local array.
 *
 * ## Build
 *   nvcc -O3 -arch=sm_80 -std=c++17 $(mpicxx --showme:compile) \
 *        -o poisson-cuda poisson-cuda.cu $(mpicxx --showme:libs)
 * Or via the Makefile target:
 *   make poisson-cuda
 */

#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>
#include <iomanip>
#include <algorithm>


// CUDA error-checking macro

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__   \
                      << " — " << cudaGetErrorString(err) << "\n";         \
            MPI_Abort(MPI_COMM_WORLD, 1);                                   \
        }                                                                   \
    } while(0)


// Block dimensions — tuned for 3-D stencils on modern GPUs.
// 8×8×8 = 512 threads/block; occupancy is high for double-
// precision stencils (register pressure is low: ~16 registers).

#define BLOCK_X 8
#define BLOCK_Y 8
#define BLOCK_Z 8


// Options

class Options {
public:
    bool        help    = false;
    std::string forcing = "";
    int         test    = -1;
    int         Nx=32,Ny=32,Nz=32;
    double      epsilon = 1e-8;
    int         Px=1,Py=1,Pz=1;

    static Options parse(int argc, char* argv[]) {
        Options o;
        for (int i=1;i<argc;++i) {
            std::string a=argv[i];
            if      (a=="--help")               o.help    =true;
            else if (a=="--forcing"&&i+1<argc)  o.forcing =argv[++i];
            else if (a=="--test"   &&i+1<argc)  o.test    =std::stoi(argv[++i]);
            else if (a=="--Nx"     &&i+1<argc)  o.Nx      =std::stoi(argv[++i]);
            else if (a=="--Ny"     &&i+1<argc)  o.Ny      =std::stoi(argv[++i]);
            else if (a=="--Nz"     &&i+1<argc)  o.Nz      =std::stoi(argv[++i]);
            else if (a=="--epsilon"&&i+1<argc)  o.epsilon =std::stod(argv[++i]);
            else if (a=="--Px"     &&i+1<argc)  o.Px      =std::stoi(argv[++i]);
            else if (a=="--Py"     &&i+1<argc)  o.Py      =std::stoi(argv[++i]);
            else if (a=="--Pz"     &&i+1<argc)  o.Pz      =std::stoi(argv[++i]);
            else throw std::invalid_argument("Unknown option: "+a);
        }
        return o;
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


// Flat index: k fastest (row-major, same layout as CPU code)

__host__ __device__ inline int idx(int i,int j,int k,int NY,int NZ){
    return i*NY*NZ + j*NZ + k;
}


// Test-case helpers (host only — used for initialisation)

double exactSolution(int tc,double x,double y,double z){
    switch(tc){
        case 1: return x*x+y*y+z*z;
        case 2: return std::sin(M_PI*x)*std::sin(M_PI*y)*std::sin(M_PI*z);
        case 3: return std::sin(M_PI*x)*std::sin(4*M_PI*y)*std::sin(8*M_PI*z);
        default: return 0.;
    }
}
double forcingFunction(int tc,double x,double y,double z){
    switch(tc){
        case 1: return 6.;
        case 2: return -3.*M_PI*M_PI*std::sin(M_PI*x)*std::sin(M_PI*y)*std::sin(M_PI*z);
        case 3: return -(1.+16.+64.)*M_PI*M_PI*std::sin(M_PI*x)*std::sin(4*M_PI*y)*std::sin(8*M_PI*z);
        case 4: return 100.*std::exp(-100.*((x-.5)*(x-.5)+(y-.5)*(y-.5)+(z-.5)*(z-.5)));
        case 5: return (x<.5)?1.:-1.;
        default: throw std::invalid_argument("Unknown test case");
    }
}
void testCaseGrid(int tc,int&Nx,int&Ny,int&Nz){
    switch(tc){
        case 1: Nx=32;Ny=32;Nz=32;  break;
        case 2: Nx=64;Ny=64;Nz=64;  break;
        case 3: Nx=32;Ny=64;Nz=128; break;
        case 4: Nx=64;Ny=64;Nz=64;  break;
        case 5: Nx=64;Ny=64;Nz=64;  break;
        default: throw std::invalid_argument("Unknown test case");
    }
}
void decompose1D(int N,int P,int rank,int&start,int&count){
    int base=N/P,rem=N%P;
    start=rank*base+std::min(rank,rem);
    count=base+(rank<rem?1:0);
}


// I/O helpers (rank 0 only)

void readForcingFile(const std::string&fn,int&Nx,int&Ny,int&Nz,
                     std::vector<double>&f){
    std::ifstream fin(fn);
    if(!fin) throw std::runtime_error("Cannot open: "+fn);
    fin>>Nx>>Ny>>Nz;
    f.assign(Nx*Ny*Nz,0.);
    double hx=1./(Nx-1),hy=1./(Ny-1),hz=1./(Nz-1);
    for(int n=0;n<Nx*Ny*Nz;++n){
        double x,y,z,v; fin>>x>>y>>z>>v;
        int i=int(std::round(x/hx)),j=int(std::round(y/hy)),k=int(std::round(z/hz));
        f[i*Ny*Nz+j*Nz+k]=v;
    }
}
void writeSolution(const std::string&fn,const std::vector<double>&u,
                   int Nx,int Ny,int Nz,double hx,double hy,double hz){
    std::ofstream fout(fn);
    fout<<Nx<<" "<<Ny<<" "<<Nz<<"\n";
    fout<<std::scientific<<std::setprecision(6);
    for(int i=0;i<Nx;++i) for(int j=0;j<Ny;++j) for(int k=0;k<Nz;++k)
        fout<<i*hx<<" "<<j*hy<<" "<<k*hz<<" "<<u[i*Ny*Nz+j*Nz+k]<<"\n";
}


// CUDA kernels


/**
 * @brief Jacobi sweep kernel — one thread per interior grid point.
 *
 * Reads u (with halo), writes unew.  Interior-only bounds [iLo,iHi]
 * × [jLo,jHi] × [kLo,kHi] are enforced: each thread checks whether
 * its (i,j,k) falls inside the interior range and returns early if not.
 * This avoids conditional branching in the hot path for interior threads.
 */
__global__ void jacobiKernel(
    const double* __restrict__ u,
    const double* __restrict__ f,
          double* __restrict__ unew,
    int lx, int ly, int lz,
    double ihx2, double ihy2, double ihz2, double denom,
    int iLo, int iHi, int jLo, int jHi, int kLo, int kHi)
{
    // Map thread indices to (i,j,k) in the local grid (1-based, halos at 0)
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i < iLo || i > iHi) return;
    if (j < jLo || j > jHi) return;
    if (k < kLo || k > kHi) return;

    const int NY = ly+2, NZ = lz+2;

    double rhs =
        (u[idx(i+1,j,k,NY,NZ)] + u[idx(i-1,j,k,NY,NZ)]) * ihx2
      + (u[idx(i,j+1,k,NY,NZ)] + u[idx(i,j-1,k,NY,NZ)]) * ihy2
      + (u[idx(i,j,k+1,NY,NZ)] + u[idx(i,j,k-1,NY,NZ)]) * ihz2
      - f[idx(i,j,k,NY,NZ)];

    unew[idx(i,j,k,NY,NZ)] = rhs / denom;
}

/**
 * @brief Residual kernel — computes local sum of squared residuals.
 *
 * Uses atomic addition into a device-side accumulator.  Each thread
 * evaluates the discrete Laplacian at its interior point, forms the
 * residual r = f - L(u), and atomically adds r² to *d_sum.
 * The caller resets *d_sum to 0 before launching.
 *
 * For large grids a parallel reduction (e.g. via shared memory) would
 * be faster, but atomicAdd on double is supported from sm_60 onwards
 * and is simpler to implement correctly.
 */
__global__ void residualKernel(
    const double* __restrict__ u,
    const double* __restrict__ f,
          double* __restrict__ d_sum,
    int lx, int ly, int lz,
    double ihx2, double ihy2, double ihz2,
    int iLo, int iHi, int jLo, int jHi, int kLo, int kHi)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i < iLo || i > iHi) return;
    if (j < jLo || j > jHi) return;
    if (k < kLo || k > kHi) return;

    const int NY = ly+2, NZ = lz+2;

    double lap =
        (u[idx(i+1,j,k,NY,NZ)] - 2*u[idx(i,j,k,NY,NZ)] + u[idx(i-1,j,k,NY,NZ)]) * ihx2
      + (u[idx(i,j+1,k,NY,NZ)] - 2*u[idx(i,j,k,NY,NZ)] + u[idx(i,j-1,k,NY,NZ)]) * ihy2
      + (u[idx(i,j,k+1,NY,NZ)] - 2*u[idx(i,j,k,NY,NZ)] + u[idx(i,j,k-1,NY,NZ)]) * ihz2;

    double r = f[idx(i,j,k,NY,NZ)] - lap;
    atomicAdd(d_sum, r * r);
}

/**
 * @brief Pack Y-direction face into a contiguous send buffer (device kernel).
 * Gathers u[i, jFace, k] for all i,k into buf[i*(lz+2)+k].
 */
__global__ void packYKernel(const double* u, double* buf,
                             int lx, int ly, int lz, int jFace)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > lx+1 || k > lz+1) return;
    const int NY=ly+2, NZ=lz+2;
    buf[i*(lz+2)+k] = u[idx(i, jFace, k, NY, NZ)];
}

/**
 * @brief Unpack Y-direction receive buffer into the halo layer.
 */
__global__ void unpackYKernel(double* u, const double* buf,
                               int lx, int ly, int lz, int jHalo)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > lx+1 || k > lz+1) return;
    const int NY=ly+2, NZ=lz+2;
    u[idx(i, jHalo, k, NY, NZ)] = buf[i*(lz+2)+k];
}

/**
 * @brief Pack Z-direction face.
 */
__global__ void packZKernel(const double* u, double* buf,
                             int lx, int ly, int lz, int kFace)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > lx+1 || j > ly+1) return;
    const int NY=ly+2, NZ=lz+2;
    buf[i*(ly+2)+j] = u[idx(i, j, kFace, NY, NZ)];
}

/**
 * @brief Unpack Z-direction receive buffer.
 */
__global__ void unpackZKernel(double* u, const double* buf,
                               int lx, int ly, int lz, int kHalo)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > lx+1 || j > ly+1) return;
    const int NY=ly+2, NZ=lz+2;
    u[idx(i, j, kHalo, NY, NZ)] = buf[i*(ly+2)+j];
}


// Host-side halo exchange
// (device→host face copies, MPI, host→device)

void exchangeHalos(
    double* d_u,                       // device array
    int lx, int ly, int lz,
    MPI_Comm cart,
    int nbrXm, int nbrXp,
    int nbrYm, int nbrYp,
    int nbrZm, int nbrZp,
    // pre-allocated host face buffers:
    std::vector<double>& sXm, std::vector<double>& sXp,
    std::vector<double>& rXm, std::vector<double>& rXp,
    std::vector<double>& sYm, std::vector<double>& sYp,
    std::vector<double>& rYm, std::vector<double>& rYp,
    std::vector<double>& sZm, std::vector<double>& sZp,
    std::vector<double>& rZm, std::vector<double>& rZp,
    // pre-allocated device face buffers:
    double* d_sYm, double* d_sYp, double* d_rYm, double* d_rYp,
    double* d_sZm, double* d_sZp, double* d_rZm, double* d_rZp)
{
    const int NY = ly+2, NZ = lz+2;
    const int xFace = NY*NZ;            // X face is contiguous
    const int yFace = (lx+2)*(lz+2);   // Y face needs pack/unpack
    const int zFace = (lx+2)*(ly+2);   // Z face needs pack/unpack

    // ── Pack Y and Z faces on device ──────────────────────────
    dim3 blk2(16, 16);
    dim3 grdY(((lx+2)+15)/16, ((lz+2)+15)/16);
    dim3 grdZ(((lx+2)+15)/16, ((ly+2)+15)/16);

    packYKernel<<<grdY, blk2>>>(d_u, d_sYm, lx, ly, lz, 1);
    packYKernel<<<grdY, blk2>>>(d_u, d_sYp, lx, ly, lz, ly);
    packZKernel<<<grdZ, blk2>>>(d_u, d_sZm, lx, ly, lz, 1);
    packZKernel<<<grdZ, blk2>>>(d_u, d_sZp, lx, ly, lz, lz);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Copy faces device ->host ───────────────────────────────
    // X faces (contiguous — direct pointer arithmetic)
    CUDA_CHECK(cudaMemcpy(sXm.data(),
        d_u + idx(1,  0, 0, NY, NZ), xFace*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sXp.data(),
        d_u + idx(lx, 0, 0, NY, NZ), xFace*sizeof(double), cudaMemcpyDeviceToHost));
    // Y and Z faces (already packed in device buffers)
    CUDA_CHECK(cudaMemcpy(sYm.data(), d_sYm, yFace*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sYp.data(), d_sYp, yFace*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sZm.data(), d_sZm, zFace*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sZp.data(), d_sZp, zFace*sizeof(double), cudaMemcpyDeviceToHost));

    // ── MPI exchange (12 non-blocking messages) ────────────────
    MPI_Request req[12]; int nr = 0;
    const int TAG = 0;

    MPI_Isend(sXm.data(), xFace, MPI_DOUBLE, nbrXm, TAG, cart, &req[nr++]);
    MPI_Irecv(rXm.data(), xFace, MPI_DOUBLE, nbrXm, TAG, cart, &req[nr++]);
    MPI_Isend(sXp.data(), xFace, MPI_DOUBLE, nbrXp, TAG, cart, &req[nr++]);
    MPI_Irecv(rXp.data(), xFace, MPI_DOUBLE, nbrXp, TAG, cart, &req[nr++]);
    MPI_Isend(sYm.data(), yFace, MPI_DOUBLE, nbrYm, TAG, cart, &req[nr++]);
    MPI_Irecv(rYm.data(), yFace, MPI_DOUBLE, nbrYm, TAG, cart, &req[nr++]);
    MPI_Isend(sYp.data(), yFace, MPI_DOUBLE, nbrYp, TAG, cart, &req[nr++]);
    MPI_Irecv(rYp.data(), yFace, MPI_DOUBLE, nbrYp, TAG, cart, &req[nr++]);
    MPI_Isend(sZm.data(), zFace, MPI_DOUBLE, nbrZm, TAG, cart, &req[nr++]);
    MPI_Irecv(rZm.data(), zFace, MPI_DOUBLE, nbrZm, TAG, cart, &req[nr++]);
    MPI_Isend(sZp.data(), zFace, MPI_DOUBLE, nbrZp, TAG, cart, &req[nr++]);
    MPI_Irecv(rZp.data(), zFace, MPI_DOUBLE, nbrZp, TAG, cart, &req[nr++]);
    MPI_Waitall(nr, req, MPI_STATUSES_IGNORE);

    // ── Copy halos host ->device ───────────────────────────────
    CUDA_CHECK(cudaMemcpy(d_u + idx(0,   0, 0, NY, NZ),
        rXm.data(), xFace*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u + idx(lx+1,0, 0, NY, NZ),
        rXp.data(), xFace*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rYm, rYm.data(), yFace*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rYp, rYp.data(), yFace*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rZm, rZm.data(), zFace*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rZp, rZp.data(), zFace*sizeof(double), cudaMemcpyHostToDevice));

    // ── Unpack Y and Z halos on device ────────────────────────
    unpackYKernel<<<grdY, blk2>>>(d_u, d_rYm, lx, ly, lz, 0);
    unpackYKernel<<<grdY, blk2>>>(d_u, d_rYp, lx, ly, lz, ly+1);
    unpackZKernel<<<grdZ, blk2>>>(d_u, d_rZm, lx, ly, lz, 0);
    unpackZKernel<<<grdZ, blk2>>>(d_u, d_rZp, lx, ly, lz, lz+1);
    CUDA_CHECK(cudaDeviceSynchronize());
}


// Main

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Assign one GPU per MPI rank (round-robin if fewer GPUs than ranks)
    int ngpu = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ngpu));
    if (ngpu == 0) {
        if (rank == 0) std::cerr << "Error: no CUDA device found.\n";
        MPI_Finalize(); return 1;
    }
    CUDA_CHECK(cudaSetDevice(rank % ngpu));
    if (rank == 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "GPU: " << prop.name
                  << " (" << prop.totalGlobalMem/1024/1024 << " MB)\n";
    }

    // ── Parse options ──────────────────────────────────────────
    Options opt;
    try { opt = Options::parse(argc, argv); }
    catch (const std::exception& e) {
        if (rank==0) { std::cerr<<"Error: "<<e.what()<<"\n"; Options::printHelp(); }
        MPI_Finalize(); return 1;
    }
    if (opt.help) { if(rank==0) Options::printHelp(); MPI_Finalize(); return 0; }
    if (opt.Px*opt.Py*opt.Pz != size) {
        if (rank==0) std::cerr<<"Error: Px*Py*Pz="<<opt.Px*opt.Py*opt.Pz<<" but P="<<size<<"\n";
        MPI_Finalize(); return 1;
    }
    bool hasForcing = !opt.forcing.empty(), hasTest = (opt.test!=-1);
    if (hasForcing == hasTest) {
        if(rank==0) std::cerr<<"Error: provide exactly one of --forcing or --test.\n";
        MPI_Finalize(); return 1;
    }

    // ── Cartesian communicator ─────────────────────────────────
    int dims[3]={opt.Px,opt.Py,opt.Pz}, periods[3]={0,0,0};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart);
    int coords[3];
    MPI_Cart_coords(cart, rank, 3, coords);
    int cx=coords[0], cy=coords[1], cz=coords[2];
    int nbrXm,nbrXp,nbrYm,nbrYp,nbrZm,nbrZp;
    MPI_Cart_shift(cart, 0, 1, &nbrXm, &nbrXp);
    MPI_Cart_shift(cart, 1, 1, &nbrYm, &nbrYp);
    MPI_Cart_shift(cart, 2, 1, &nbrZm, &nbrZp);

    // ── Grid setup ─────────────────────────────────────────────
    int Nx, Ny, Nz;
    std::vector<double> globalF;
    if (rank == 0) {
        if (hasTest) {
            testCaseGrid(opt.test, Nx, Ny, Nz);
            globalF.resize(Nx*Ny*Nz);
            double hx=1./(Nx-1), hy=1./(Ny-1), hz=1./(Nz-1);
            for(int i=0;i<Nx;++i) for(int j=0;j<Ny;++j) for(int k=0;k<Nz;++k)
                globalF[i*Ny*Nz+j*Nz+k]=forcingFunction(opt.test,i*hx,j*hy,k*hz);
        } else { readForcingFile(opt.forcing,Nx,Ny,Nz,globalF); }
    }
    { int g[3]={Nx,Ny,Nz}; MPI_Bcast(g,3,MPI_INT,0,MPI_COMM_WORLD); Nx=g[0];Ny=g[1];Nz=g[2]; }
    const double hx=1./(Nx-1), hy=1./(Ny-1), hz=1./(Nz-1);

    // ── Local extents ──────────────────────────────────────────
    int gxs,lx,gys,ly,gzs,lz;
    decompose1D(Nx, opt.Px, cx, gxs, lx);
    decompose1D(Ny, opt.Py, cy, gys, ly);
    decompose1D(Nz, opt.Pz, cz, gzs, lz);
    const int LNY=ly+2, LNZ=lz+2;
    const int localN=(lx+2)*LNY*LNZ, ownedN=lx*ly*lz;

    // ── Scatter forcing ────────────────────────────────────────
    std::vector<double> h_localF(localN, 0.);
    {
        if (rank == 0) {
            for (int r=1; r<size; ++r) {
                int rc[3]; MPI_Cart_coords(cart,r,3,rc);
                int xs,lxr,ys,lyr,zs,lzr;
                decompose1D(Nx,opt.Px,rc[0],xs,lxr);
                decompose1D(Ny,opt.Py,rc[1],ys,lyr);
                decompose1D(Nz,opt.Pz,rc[2],zs,lzr);
                std::vector<double> buf(lxr*lyr*lzr);
                int off=0;
                for(int i=0;i<lxr;++i) for(int j=0;j<lyr;++j) for(int k=0;k<lzr;++k)
                    buf[off++]=globalF[(xs+i)*Ny*Nz+(ys+j)*Nz+(zs+k)];
                MPI_Send(buf.data(),lxr*lyr*lzr,MPI_DOUBLE,r,0,MPI_COMM_WORLD);
            }
            for(int i=0;i<lx;++i) for(int j=0;j<ly;++j) for(int k=0;k<lz;++k)
                h_localF[idx(i+1,j+1,k+1,LNY,LNZ)]=globalF[(gxs+i)*Ny*Nz+(gys+j)*Nz+(gzs+k)];
        } else {
            std::vector<double> buf(ownedN);
            MPI_Recv(buf.data(),ownedN,MPI_DOUBLE,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            int off=0;
            for(int i=0;i<lx;++i) for(int j=0;j<ly;++j) for(int k=0;k<lz;++k)
                h_localF[idx(i+1,j+1,k+1,LNY,LNZ)]=buf[off++];
        }
    }

    // ── Interior-only loop bounds ──────────────────────────────
    const bool bndXm=(gxs==0),      bndXp=(gxs+lx==Nx);
    const bool bndYm=(gys==0),      bndYp=(gys+ly==Ny);
    const bool bndZm=(gzs==0),      bndZp=(gzs+lz==Nz);
    const int iLo=bndXm?2:1, iHi=bndXp?lx-1:lx;
    const int jLo=bndYm?2:1, jHi=bndYp?ly-1:ly;
    const int kLo=bndZm?2:1, kHi=bndZp?lz-1:lz;

    // ── Initialise host u ──────────────────────────────────────
    std::vector<double> h_u(localN, 0.), h_unew(localN, 0.);

    auto bcVal=[&](int gi,int gj,int gk)->double{
        double x=gi*hx, y=gj*hy, z=gk*hz;
        return hasTest ? exactSolution(opt.test,x,y,z) : 0.;
    };

    // Stamp Dirichlet boundary nodes
    if(bndXm) for(int j=1;j<=ly;++j) for(int k=1;k<=lz;++k){
        double v=bcVal(gxs,gys+j-1,gzs+k-1);
        h_u[idx(1,j,k,LNY,LNZ)]=h_unew[idx(1,j,k,LNY,LNZ)]=v;
        h_u[idx(0,j,k,LNY,LNZ)]=h_unew[idx(0,j,k,LNY,LNZ)]=v;
    }
    if(bndXp) for(int j=1;j<=ly;++j) for(int k=1;k<=lz;++k){
        double v=bcVal(gxs+lx-1,gys+j-1,gzs+k-1);
        h_u[idx(lx,  j,k,LNY,LNZ)]=h_unew[idx(lx,  j,k,LNY,LNZ)]=v;
        h_u[idx(lx+1,j,k,LNY,LNZ)]=h_unew[idx(lx+1,j,k,LNY,LNZ)]=v;
    }
    if(bndYm) for(int i=1;i<=lx;++i) for(int k=1;k<=lz;++k){
        double v=bcVal(gxs+i-1,gys,gzs+k-1);
        h_u[idx(i,1,k,LNY,LNZ)]=h_unew[idx(i,1,k,LNY,LNZ)]=v;
        h_u[idx(i,0,k,LNY,LNZ)]=h_unew[idx(i,0,k,LNY,LNZ)]=v;
    }
    if(bndYp) for(int i=1;i<=lx;++i) for(int k=1;k<=lz;++k){
        double v=bcVal(gxs+i-1,gys+ly-1,gzs+k-1);
        h_u[idx(i,ly,  k,LNY,LNZ)]=h_unew[idx(i,ly,  k,LNY,LNZ)]=v;
        h_u[idx(i,ly+1,k,LNY,LNZ)]=h_unew[idx(i,ly+1,k,LNY,LNZ)]=v;
    }
    if(bndZm) for(int i=1;i<=lx;++i) for(int j=1;j<=ly;++j){
        double v=bcVal(gxs+i-1,gys+j-1,gzs);
        h_u[idx(i,j,1,  LNY,LNZ)]=h_unew[idx(i,j,1,  LNY,LNZ)]=v;
        h_u[idx(i,j,0,  LNY,LNZ)]=h_unew[idx(i,j,0,  LNY,LNZ)]=v;
    }
    if(bndZp) for(int i=1;i<=lx;++i) for(int j=1;j<=ly;++j){
        double v=bcVal(gxs+i-1,gys+j-1,gzs+lz-1);
        h_u[idx(i,j,lz,  LNY,LNZ)]=h_unew[idx(i,j,lz,  LNY,LNZ)]=v;
        h_u[idx(i,j,lz+1,LNY,LNZ)]=h_unew[idx(i,j,lz+1,LNY,LNZ)]=v;
    }

    // ── Allocate device memory ─────────────────────────────────
    double *d_u, *d_unew, *d_f, *d_sum;
    CUDA_CHECK(cudaMalloc(&d_u,    localN*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_unew, localN*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_f,    localN*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sum,  sizeof(double)));

    // Face buffers on device (for pack/unpack kernels)
    const int yFace = (lx+2)*(lz+2);
    const int zFace = (lx+2)*(ly+2);
    const int xFace = LNY*LNZ;
    double *d_sYm, *d_sYp, *d_rYm, *d_rYp;
    double *d_sZm, *d_sZp, *d_rZm, *d_rZp;
    CUDA_CHECK(cudaMalloc(&d_sYm, yFace*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sYp, yFace*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rYm, yFace*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rYp, yFace*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sZm, zFace*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sZp, zFace*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rZm, zFace*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rZp, zFace*sizeof(double)));

    // Host face buffers (for MPI)
    std::vector<double> sXm(xFace),sXp(xFace),rXm(xFace),rXp(xFace);
    std::vector<double> sYm(yFace),sYp(yFace),rYm(yFace),rYp(yFace);
    std::vector<double> sZm(zFace),sZp(zFace),rZm(zFace),rZp(zFace);

    // Upload initial arrays to device
    CUDA_CHECK(cudaMemcpy(d_u,    h_u.data(),     localN*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_unew, h_unew.data(),  localN*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f,    h_localF.data(),localN*sizeof(double), cudaMemcpyHostToDevice));

    // ── CUDA kernel launch parameters ─────────────────────────
    const double ihx2  = 1./(hx*hx);
    const double ihy2  = 1./(hy*hy);
    const double ihz2  = 1./(hz*hz);
    const double denom = 2.*(ihx2+ihy2+ihz2);

    // Grid covers interior points [iLo..iHi] × [jLo..jHi] × [kLo..kHi]
    // Add 1 to shift from 0-based thread index to 1-based grid index
    const int gx = (iHi - iLo + BLOCK_X) / BLOCK_X;
    const int gy = (jHi - jLo + BLOCK_Y) / BLOCK_Y;
    const int gz = (kHi - kLo + BLOCK_Z) / BLOCK_Z;
    dim3 gridDim(gx, gy, gz);
    dim3 blockDim(BLOCK_X, BLOCK_Y, BLOCK_Z);

    // ── Jacobi iteration ───────────────────────────────────────
    double residual = 0.; int iter = 0;
    do {
        // 1. Halo exchange (device→host→MPI→host→device)
        exchangeHalos(d_u, lx, ly, lz, cart,
                      nbrXm, nbrXp, nbrYm, nbrYp, nbrZm, nbrZp,
                      sXm, sXp, rXm, rXp,
                      sYm, sYp, rYm, rYp,
                      sZm, sZp, rZm, rZp,
                      d_sYm, d_sYp, d_rYm, d_rYp,
                      d_sZm, d_sZp, d_rZm, d_rZp);

        // 2. Jacobi sweep on GPU
        jacobiKernel<<<gridDim, blockDim>>>(
            d_u, d_f, d_unew,
            lx, ly, lz,
            ihx2, ihy2, ihz2, denom,
            iLo, iHi, jLo, jHi, kLo, kHi);
        CUDA_CHECK(cudaGetLastError());

        // 3. Swap device pointers (zero-copy)
        std::swap(d_u, d_unew);

        // 4. Residual on GPU
        CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(double)));
        residualKernel<<<gridDim, blockDim>>>(
            d_u, d_f, d_sum,
            lx, ly, lz,
            ihx2, ihy2, ihz2,
            iLo, iHi, jLo, jHi, kLo, kHi);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 5. Copy local sum to host and MPI reduce
        double lsq = 0.;
        CUDA_CHECK(cudaMemcpy(&lsq, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
        double gsq = 0.;
        MPI_Allreduce(&lsq, &gsq, 1, MPI_DOUBLE, MPI_SUM, cart);
        residual = std::sqrt(gsq);
        ++iter;

    } while (residual > opt.epsilon);

    // ── Copy final solution back to host ───────────────────────
    CUDA_CHECK(cudaMemcpy(h_u.data(), d_u, localN*sizeof(double), cudaMemcpyDeviceToHost));

    // ── Gather and write (rank 0) ──────────────────────────────
    {
        std::vector<double> sendBuf(ownedN);
        { int off=0;
          for(int i=0;i<lx;++i) for(int j=0;j<ly;++j) for(int k=0;k<lz;++k)
              sendBuf[off++]=h_u[idx(i+1,j+1,k+1,LNY,LNZ)]; }

        if (rank == 0) {
            std::vector<double> globalU(Nx*Ny*Nz, 0.);
            for(int i=0;i<lx;++i) for(int j=0;j<ly;++j) for(int k=0;k<lz;++k)
                globalU[(gxs+i)*Ny*Nz+(gys+j)*Nz+(gzs+k)] = sendBuf[i*ly*lz+j*lz+k];
            for (int r=1; r<size; ++r) {
                int rc[3]; MPI_Cart_coords(cart,r,3,rc);
                int xs,lxr,ys,lyr,zs,lzr;
                decompose1D(Nx,opt.Px,rc[0],xs,lxr);
                decompose1D(Ny,opt.Py,rc[1],ys,lyr);
                decompose1D(Nz,opt.Pz,rc[2],zs,lzr);
                std::vector<double> rbuf(lxr*lyr*lzr);
                MPI_Recv(rbuf.data(),lxr*lyr*lzr,MPI_DOUBLE,r,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                int off=0;
                for(int i=0;i<lxr;++i) for(int j=0;j<lyr;++j) for(int k=0;k<lzr;++k)
                    globalU[(xs+i)*Ny*Nz+(ys+j)*Nz+(zs+k)]=rbuf[off++];
            }
            writeSolution("solution.txt", globalU, Nx, Ny, Nz, hx, hy, hz);
            std::cout << std::scientific << std::setprecision(6)
                      << "Final residual: " << residual
                      << " (after " << iter << " iterations)\n";
        } else {
            MPI_Send(sendBuf.data(), ownedN, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        }
    }

    // ── Cleanup ────────────────────────────────────────────────
    cudaFree(d_u); cudaFree(d_unew); cudaFree(d_f); cudaFree(d_sum);
    cudaFree(d_sYm); cudaFree(d_sYp); cudaFree(d_rYm); cudaFree(d_rYp);
    cudaFree(d_sZm); cudaFree(d_sZp); cudaFree(d_rZm); cudaFree(d_rZp);

    MPI_Comm_free(&cart);
    MPI_Finalize();
    return 0;
}
