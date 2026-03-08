/**
 * @file poisson-mpi-opt2.cpp
 * @brief MPI Poisson solver — Optimisation 2: OpenMP threading (on top of Opt1).
 *
 * ## Optimisations applied (cumulative):
 *   1. Computation/Communication Overlap  (from opt1)
 *   2. OpenMP multi-threading             (NEW in this file)
 *
 * ## What changed vs opt1
 * The Jacobi sweep and residual accumulation are both embarrassingly
 * parallel across grid points: no two interior nodes share a write target,
 * and residual is a pure reduction.  Adding a single OpenMP directive to
 * the outermost loop of jacobiRange() and localResidualSq() lets the OS
 * distribute iterations across all hardware threads on the node.
 *
 *   #pragma omp parallel for schedule(static) collapse(2)
 *
 * `collapse(2)` fuses the i and j loops into one larger loop of
 * iCount*jCount iterations, which gives the OpenMP runtime more chunks to
 * distribute and avoids under-utilisation when iCount or jCount is small
 * (e.g. when the decomposition leaves only a thin slab per rank).
 *
 * `schedule(static)` assigns equal contiguous chunks to each thread, which
 * is optimal here because every iteration does identical work.
 *
 * The residual loop uses `reduction(+:s)` so each thread accumulates into
 * a private variable; the runtime sums them after the parallel region,
 * eliminating data races and false sharing.
 *
 * ## Why this helps
 * Jacobi is memory-bandwidth bound.  On a modern multi-socket node each
 * core has its own L1/L2 cache and shares L3.  Spreading the loop across
 * all cores allows multiple cache lines to be fetched simultaneously from
 * different memory controllers, effectively multiplying the available
 * bandwidth.  For a 64³ grid on 8 threads, each thread works on roughly
 * 64×64×8 points — a slice that can fit comfortably in L2, greatly
 * reducing main-memory traffic per thread.
 *
 * Compile with:  mpic++ -std=c++17 -O3 -fopenmp -o poisson-mpi-opt2 poisson-mpi-opt2.cpp
 * Run with:      OMP_NUM_THREADS=4 mpirun -n 4 ./poisson-mpi-opt2 --test 2 --Px 2 --Py 2
 */

#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>
#include <iomanip>
#include <algorithm>

// ============================================================
// Options (unchanged)
// ============================================================
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
            if      (a=="--help")                  o.help    =true;
            else if (a=="--forcing" &&i+1<argc)    o.forcing =argv[++i];
            else if (a=="--test"    &&i+1<argc)    o.test    =std::stoi(argv[++i]);
            else if (a=="--Nx"      &&i+1<argc)    o.Nx      =std::stoi(argv[++i]);
            else if (a=="--Ny"      &&i+1<argc)    o.Ny      =std::stoi(argv[++i]);
            else if (a=="--Nz"      &&i+1<argc)    o.Nz      =std::stoi(argv[++i]);
            else if (a=="--epsilon" &&i+1<argc)    o.epsilon =std::stod(argv[++i]);
            else if (a=="--Px"      &&i+1<argc)    o.Px      =std::stoi(argv[++i]);
            else if (a=="--Py"      &&i+1<argc)    o.Py      =std::stoi(argv[++i]);
            else if (a=="--Pz"      &&i+1<argc)    o.Pz      =std::stoi(argv[++i]);
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

inline int idx(int i,int j,int k,int NY,int NZ){ return i*NY*NZ+j*NZ+k; }

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

// ============================================================
// Halo exchange (same split post/finish as opt1)
// ============================================================
struct HaloBuffers {
    std::vector<double> sYm,sYp,rYm,rYp;
    std::vector<double> sZm,sZp,rZm,rZp;
    MPI_Request req[12];
    int nr=0;
};

HaloBuffers postHaloExchange(std::vector<double>&u,int lx,int ly,int lz,
                              MPI_Comm cart,
                              int nbrXm,int nbrXp,int nbrYm,int nbrYp,
                              int nbrZm,int nbrZp)
{
    const int NY=ly+2,NZ=lz+2;
    const int yFace=(lx+2)*(lz+2), zFace=(lx+2)*(ly+2);
    HaloBuffers hb;
    hb.sYm.resize(yFace); hb.sYp.resize(yFace);
    hb.rYm.resize(yFace); hb.rYp.resize(yFace);
    hb.sZm.resize(zFace); hb.sZp.resize(zFace);
    hb.rZm.resize(zFace); hb.rZp.resize(zFace);

    for(int i=0;i<=lx+1;++i) for(int k=0;k<=lz+1;++k){
        hb.sYm[i*(lz+2)+k]=u[idx(i,1, k,NY,NZ)];
        hb.sYp[i*(lz+2)+k]=u[idx(i,ly,k,NY,NZ)];
    }
    for(int i=0;i<=lx+1;++i) for(int j=0;j<=ly+1;++j){
        hb.sZm[i*(ly+2)+j]=u[idx(i,j,1, NY,NZ)];
        hb.sZp[i*(ly+2)+j]=u[idx(i,j,lz,NY,NZ)];
    }

    const int TAG=0; int&nr=hb.nr;
    MPI_Isend(&u[idx(1,   0,0,NY,NZ)],NY*NZ,MPI_DOUBLE,nbrXm,TAG,cart,&hb.req[nr++]);
    MPI_Irecv(&u[idx(0,   0,0,NY,NZ)],NY*NZ,MPI_DOUBLE,nbrXm,TAG,cart,&hb.req[nr++]);
    MPI_Isend(&u[idx(lx,  0,0,NY,NZ)],NY*NZ,MPI_DOUBLE,nbrXp,TAG,cart,&hb.req[nr++]);
    MPI_Irecv(&u[idx(lx+1,0,0,NY,NZ)],NY*NZ,MPI_DOUBLE,nbrXp,TAG,cart,&hb.req[nr++]);
    MPI_Isend(hb.sYm.data(),yFace,MPI_DOUBLE,nbrYm,TAG,cart,&hb.req[nr++]);
    MPI_Irecv(hb.rYm.data(),yFace,MPI_DOUBLE,nbrYm,TAG,cart,&hb.req[nr++]);
    MPI_Isend(hb.sYp.data(),yFace,MPI_DOUBLE,nbrYp,TAG,cart,&hb.req[nr++]);
    MPI_Irecv(hb.rYp.data(),yFace,MPI_DOUBLE,nbrYp,TAG,cart,&hb.req[nr++]);
    MPI_Isend(hb.sZm.data(),zFace,MPI_DOUBLE,nbrZm,TAG,cart,&hb.req[nr++]);
    MPI_Irecv(hb.rZm.data(),zFace,MPI_DOUBLE,nbrZm,TAG,cart,&hb.req[nr++]);
    MPI_Isend(hb.sZp.data(),zFace,MPI_DOUBLE,nbrZp,TAG,cart,&hb.req[nr++]);
    MPI_Irecv(hb.rZp.data(),zFace,MPI_DOUBLE,nbrZp,TAG,cart,&hb.req[nr++]);
    return hb;
}

void finishHaloExchange(std::vector<double>&u,int lx,int ly,int lz,HaloBuffers&hb){
    const int NY=ly+2,NZ=lz+2;
    MPI_Waitall(hb.nr,hb.req,MPI_STATUSES_IGNORE);
    for(int i=0;i<=lx+1;++i) for(int k=0;k<=lz+1;++k){
        u[idx(i,0,   k,NY,NZ)]=hb.rYm[i*(lz+2)+k];
        u[idx(i,ly+1,k,NY,NZ)]=hb.rYp[i*(lz+2)+k];
    }
    for(int i=0;i<=lx+1;++i) for(int j=0;j<=ly+1;++j){
        u[idx(i,j,0,   NY,NZ)]=hb.rZm[i*(ly+2)+j];
        u[idx(i,j,lz+1,NY,NZ)]=hb.rZp[i*(ly+2)+j];
    }
}

// ============================================================
// OPT2: OpenMP-parallelised Jacobi range
// ============================================================
void jacobiRange(const std::vector<double>&u,const std::vector<double>&f,
                 std::vector<double>&unew,int ly,int lz,
                 double ihx2,double ihy2,double ihz2,double denom,
                 int iLo,int iHi,int jLo,int jHi,int kLo,int kHi)
{
    const int NY=ly+2,NZ=lz+2;
    // collapse(2) merges i and j into one parallel loop for better load balance
    #pragma omp parallel for schedule(static) collapse(2)
    for(int i=iLo;i<=iHi;++i)
      for(int j=jLo;j<=jHi;++j)
        for(int k=kLo;k<=kHi;++k){
            double rhs=
                (u[idx(i+1,j,k,NY,NZ)]+u[idx(i-1,j,k,NY,NZ)])*ihx2
               +(u[idx(i,j+1,k,NY,NZ)]+u[idx(i,j-1,k,NY,NZ)])*ihy2
               +(u[idx(i,j,k+1,NY,NZ)]+u[idx(i,j,k-1,NY,NZ)])*ihz2
               -f[idx(i,j,k,NY,NZ)];
            unew[idx(i,j,k,NY,NZ)]=rhs/denom;
        }
}

// ============================================================
// OPT2: OpenMP reduction for residual
// ============================================================
double localResidualSq(const std::vector<double>&u,const std::vector<double>&f,
                       int lx,int ly,int lz,double hx,double hy,double hz,
                       int iLo,int iHi,int jLo,int jHi,int kLo,int kHi)
{
    const int NY=ly+2,NZ=lz+2;
    const double ihx2=1./(hx*hx),ihy2=1./(hy*hy),ihz2=1./(hz*hz);
    double s=0.;
    #pragma omp parallel for schedule(static) collapse(2) reduction(+:s)
    for(int i=iLo;i<=iHi;++i)
      for(int j=jLo;j<=jHi;++j)
        for(int k=kLo;k<=kHi;++k){
            double lap=
                (u[idx(i+1,j,k,NY,NZ)]-2*u[idx(i,j,k,NY,NZ)]+u[idx(i-1,j,k,NY,NZ)])*ihx2
               +(u[idx(i,j+1,k,NY,NZ)]-2*u[idx(i,j,k,NY,NZ)]+u[idx(i,j-1,k,NY,NZ)])*ihy2
               +(u[idx(i,j,k+1,NY,NZ)]-2*u[idx(i,j,k,NY,NZ)]+u[idx(i,j,k-1,NY,NZ)])*ihz2;
            double r=f[idx(i,j,k,NY,NZ)]-lap;
            s+=r*r;
        }
    return s;
}

void readForcingFile(const std::string&fn,int&Nx,int&Ny,int&Nz,std::vector<double>&f){
    std::ifstream fin(fn);
    if(!fin) throw std::runtime_error("Cannot open: "+fn);
    fin>>Nx>>Ny>>Nz; f.assign(Nx*Ny*Nz,0.);
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
    if(!fout) throw std::runtime_error("Cannot open: "+fn);
    fout<<Nx<<" "<<Ny<<" "<<Nz<<"\n";
    fout<<std::scientific<<std::setprecision(12);
    for(int i=0;i<Nx;++i) for(int j=0;j<Ny;++j) for(int k=0;k<Nz;++k)
        fout<<i*hx<<" "<<j*hy<<" "<<k*hz<<" "<<u[i*Ny*Nz+j*Nz+k]<<"\n";
}

int main(int argc,char*argv[]){
    // Ensure MPI is initialised with thread support for MPI+OpenMP
    int provided;
    MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    Options opt;
    try{ opt=Options::parse(argc,argv); }
    catch(const std::exception&e){
        if(rank==0){ std::cerr<<"Error: "<<e.what()<<"\n"; Options::printHelp(); }
        MPI_Finalize(); return 1;
    }
    if(opt.help){ if(rank==0) Options::printHelp(); MPI_Finalize(); return 0; }
    if(opt.Px*opt.Py*opt.Pz!=size){
        if(rank==0) std::cerr<<"Error: Px*Py*Pz="<<opt.Px*opt.Py*opt.Pz<<" but P="<<size<<"\n";
        MPI_Finalize(); return 1;
    }
    bool hasForcing=!opt.forcing.empty(), hasTest=(opt.test!=-1);
    if(hasForcing==hasTest){
        if(rank==0) std::cerr<<"Error: provide exactly one of --forcing or --test.\n";
        MPI_Finalize(); return 1;
    }

    int dims[3]={opt.Px,opt.Py,opt.Pz}, periods[3]={0,0,0};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD,3,dims,periods,1,&cart);
    int coords[3];
    MPI_Cart_coords(cart,rank,3,coords);
    int cx=coords[0],cy=coords[1],cz=coords[2];
    int nbrXm,nbrXp,nbrYm,nbrYp,nbrZm,nbrZp;
    MPI_Cart_shift(cart,0,1,&nbrXm,&nbrXp);
    MPI_Cart_shift(cart,1,1,&nbrYm,&nbrYp);
    MPI_Cart_shift(cart,2,1,&nbrZm,&nbrZp);

    int Nx,Ny,Nz;
    std::vector<double> globalF;
    if(rank==0){
        if(hasTest){
            testCaseGrid(opt.test,Nx,Ny,Nz);
            globalF.resize(Nx*Ny*Nz);
            double hx=1./(Nx-1),hy=1./(Ny-1),hz=1./(Nz-1);
            for(int i=0;i<Nx;++i) for(int j=0;j<Ny;++j) for(int k=0;k<Nz;++k)
                globalF[i*Ny*Nz+j*Nz+k]=forcingFunction(opt.test,i*hx,j*hy,k*hz);
        } else { readForcingFile(opt.forcing,Nx,Ny,Nz,globalF); }
    }
    {int g[3]={Nx,Ny,Nz}; MPI_Bcast(g,3,MPI_INT,0,MPI_COMM_WORLD); Nx=g[0];Ny=g[1];Nz=g[2];}
    const double hx=1./(Nx-1),hy=1./(Ny-1),hz=1./(Nz-1);

    int gxs,lx,gys,ly,gzs,lz;
    decompose1D(Nx,opt.Px,cx,gxs,lx);
    decompose1D(Ny,opt.Py,cy,gys,ly);
    decompose1D(Nz,opt.Pz,cz,gzs,lz);
    const int LNY=ly+2,LNZ=lz+2;
    const int localN=(lx+2)*LNY*LNZ, ownedN=lx*ly*lz;

    std::vector<double> localF(localN,0.);
    {
        if(rank==0){
            for(int r=1;r<size;++r){
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
                localF[idx(i+1,j+1,k+1,LNY,LNZ)]=globalF[(gxs+i)*Ny*Nz+(gys+j)*Nz+(gzs+k)];
        } else {
            std::vector<double> buf(ownedN);
            MPI_Recv(buf.data(),ownedN,MPI_DOUBLE,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            int off=0;
            for(int i=0;i<lx;++i) for(int j=0;j<ly;++j) for(int k=0;k<lz;++k)
                localF[idx(i+1,j+1,k+1,LNY,LNZ)]=buf[off++];
        }
    }

    const bool bndXm=(gxs==0),      bndXp=(gxs+lx==Nx);
    const bool bndYm=(gys==0),      bndYp=(gys+ly==Ny);
    const bool bndZm=(gzs==0),      bndZp=(gzs+lz==Nz);
    const int iLo=bndXm?2:1, iHi=bndXp?lx-1:lx;
    const int jLo=bndYm?2:1, jHi=bndYp?ly-1:ly;
    const int kLo=bndZm?2:1, kHi=bndZp?lz-1:lz;

    std::vector<double> u(localN,0.),unew(localN,0.);
    auto bcVal=[&](int gi,int gj,int gk)->double{
        double x=gi*hx,y=gj*hy,z=gk*hz;
        return hasTest?exactSolution(opt.test,x,y,z):0.;
    };
    if(bndXm) for(int j=1;j<=ly;++j) for(int k=1;k<=lz;++k){
        double v=bcVal(gxs,gys+j-1,gzs+k-1);
        u[idx(1,j,k,LNY,LNZ)]=unew[idx(1,j,k,LNY,LNZ)]=v;
        u[idx(0,j,k,LNY,LNZ)]=unew[idx(0,j,k,LNY,LNZ)]=v;
    }
    if(bndXp) for(int j=1;j<=ly;++j) for(int k=1;k<=lz;++k){
        double v=bcVal(gxs+lx-1,gys+j-1,gzs+k-1);
        u[idx(lx,  j,k,LNY,LNZ)]=unew[idx(lx,  j,k,LNY,LNZ)]=v;
        u[idx(lx+1,j,k,LNY,LNZ)]=unew[idx(lx+1,j,k,LNY,LNZ)]=v;
    }
    if(bndYm) for(int i=1;i<=lx;++i) for(int k=1;k<=lz;++k){
        double v=bcVal(gxs+i-1,gys,gzs+k-1);
        u[idx(i,1,k,LNY,LNZ)]=unew[idx(i,1,k,LNY,LNZ)]=v;
        u[idx(i,0,k,LNY,LNZ)]=unew[idx(i,0,k,LNY,LNZ)]=v;
    }
    if(bndYp) for(int i=1;i<=lx;++i) for(int k=1;k<=lz;++k){
        double v=bcVal(gxs+i-1,gys+ly-1,gzs+k-1);
        u[idx(i,ly,  k,LNY,LNZ)]=unew[idx(i,ly,  k,LNY,LNZ)]=v;
        u[idx(i,ly+1,k,LNY,LNZ)]=unew[idx(i,ly+1,k,LNY,LNZ)]=v;
    }
    if(bndZm) for(int i=1;i<=lx;++i) for(int j=1;j<=ly;++j){
        double v=bcVal(gxs+i-1,gys+j-1,gzs);
        u[idx(i,j,1,  LNY,LNZ)]=unew[idx(i,j,1,  LNY,LNZ)]=v;
        u[idx(i,j,0,  LNY,LNZ)]=unew[idx(i,j,0,  LNY,LNZ)]=v;
    }
    if(bndZp) for(int i=1;i<=lx;++i) for(int j=1;j<=ly;++j){
        double v=bcVal(gxs+i-1,gys+j-1,gzs+lz-1);
        u[idx(i,j,lz,  LNY,LNZ)]=unew[idx(i,j,lz,  LNY,LNZ)]=v;
        u[idx(i,j,lz+1,LNY,LNZ)]=unew[idx(i,j,lz+1,LNY,LNZ)]=v;
    }

    const double ihx2=1./(hx*hx),ihy2=1./(hy*hy),ihz2=1./(hz*hz);
    const double denom=2.*(ihx2+ihy2+ihz2);

    const int iLoIn=iLo+1,iHiIn=iHi-1;
    const int jLoIn=jLo+1,jHiIn=jHi-1;
    const int kLoIn=kLo+1,kHiIn=kHi-1;

    double residual=0.; int iter=0;
    do {
        HaloBuffers hb = postHaloExchange(u,lx,ly,lz,cart,
                                          nbrXm,nbrXp,nbrYm,nbrYp,nbrZm,nbrZp);

        if(iLoIn<=iHiIn && jLoIn<=jHiIn && kLoIn<=kHiIn)
            jacobiRange(u,localF,unew,ly,lz,ihx2,ihy2,ihz2,denom,
                        iLoIn,iHiIn,jLoIn,jHiIn,kLoIn,kHiIn);

        finishHaloExchange(u,lx,ly,lz,hb);

        if(iLo<=iHi){
            jacobiRange(u,localF,unew,ly,lz,ihx2,ihy2,ihz2,denom,
                        iLo,iLo,jLo,jHi,kLo,kHi);
            if(iHi>iLo)
                jacobiRange(u,localF,unew,ly,lz,ihx2,ihy2,ihz2,denom,
                            iHi,iHi,jLo,jHi,kLo,kHi);
        }
        if(iLoIn<=iHiIn && jLo<=jHi){
            jacobiRange(u,localF,unew,ly,lz,ihx2,ihy2,ihz2,denom,
                        iLoIn,iHiIn,jLo,jLo,kLo,kHi);
            if(jHi>jLo)
                jacobiRange(u,localF,unew,ly,lz,ihx2,ihy2,ihz2,denom,
                            iLoIn,iHiIn,jHi,jHi,kLo,kHi);
        }
        if(iLoIn<=iHiIn && jLoIn<=jHiIn && kLo<=kHi){
            jacobiRange(u,localF,unew,ly,lz,ihx2,ihy2,ihz2,denom,
                        iLoIn,iHiIn,jLoIn,jHiIn,kLo,kLo);
            if(kHi>kLo)
                jacobiRange(u,localF,unew,ly,lz,ihx2,ihy2,ihz2,denom,
                            iLoIn,iHiIn,jLoIn,jHiIn,kHi,kHi);
        }

        std::swap(u,unew);

        double lsq=localResidualSq(u,localF,lx,ly,lz,hx,hy,hz,iLo,iHi,jLo,jHi,kLo,kHi);
        double gsq=0.;
        MPI_Allreduce(&lsq,&gsq,1,MPI_DOUBLE,MPI_SUM,cart);
        residual=std::sqrt(gsq);
        ++iter;
    } while(residual>opt.epsilon);

    {
        std::vector<double> sendBuf(ownedN);
        {int off=0; for(int i=0;i<lx;++i) for(int j=0;j<ly;++j) for(int k=0;k<lz;++k)
            sendBuf[off++]=u[idx(i+1,j+1,k+1,LNY,LNZ)];}

        if(rank==0){
            std::vector<double> globalU(Nx*Ny*Nz,0.);
            for(int i=0;i<lx;++i) for(int j=0;j<ly;++j) for(int k=0;k<lz;++k)
                globalU[(gxs+i)*Ny*Nz+(gys+j)*Nz+(gzs+k)]=sendBuf[i*ly*lz+j*lz+k];
            for(int r=1;r<size;++r){
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