/**
 * @file opt1.cpp
 * @brief MPI Poisson solver — Optimisation 1: hoist inner-loop constants,
 *        remove duplicate halo exchange and duplicate MPI_Allreduce.
 *
 * Removes the three most redundant operations from the baseline:
 *  - Grid spacing inverses (1/hx^2 etc.) hoisted out of the Jacobi inner loop
 *  - Redundant second halo exchange (post-sweep) removed
 *  - Redundant second MPI_Allreduce (debug cross-check) removed
 *
 * Remaining inefficiencies addressed in later versions:
 *  - A pre-sweep L-infinity norm is computed over the full local domain and
 *    reduced globally via MPI_Allreduce(MPI_MAX) every iteration.  Added
 *    during debugging to catch divergence early; result is never acted upon
 *    once the solver is verified correct.  Adds O(N^3) work + a collective.
 *  - Boundary re-stamp loops execute O(N^2) work twice per iteration even
 *    though the interior-only loop bounds make this unnecessary.
 *  - Halo exchange still uses 6 sequential blocking MPI_Sendrecv calls.
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
// Options
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

// ============================================================
// Flat index
// ============================================================
inline int idx(int i,int j,int k,int NY,int NZ){ return i*NY*NZ+j*NZ+k; }

// ============================================================
// Test cases
// ============================================================
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

// ============================================================
// Domain decomposition
// ============================================================
void decompose1D(int N,int P,int rank,int&start,int&count){
    int base=N/P,rem=N%P;
    start=rank*base+std::min(rank,rem);
    count=base+(rank<rem?1:0);
}

// ============================================================
// Halo exchange — BLOCKING, face by face (MPI_Sendrecv)
//
// Each of the 6 faces is exchanged in a separate MPI_Sendrecv
// call.  This is the simplest correct implementation but
// serialises all inter-rank data transfers: each face exchange
// must complete before the next can begin.
// ============================================================
void exchangeHalos(std::vector<double>&u,int lx,int ly,int lz,MPI_Comm cart,
                   int nbrXm,int nbrXp,int nbrYm,int nbrYp,int nbrZm,int nbrZp)
{
    const int NY=ly+2,NZ=lz+2;
    const int yFace=(lx+2)*(lz+2), zFace=(lx+2)*(ly+2);
    std::vector<double> sYm(yFace),sYp(yFace),rYm(yFace),rYp(yFace);
    std::vector<double> sZm(zFace),sZp(zFace),rZm(zFace),rZp(zFace);

    for(int i=0;i<=lx+1;++i) for(int k=0;k<=lz+1;++k){
        sYm[i*(lz+2)+k]=u[idx(i,1, k,NY,NZ)];
        sYp[i*(lz+2)+k]=u[idx(i,ly,k,NY,NZ)];
    }
    for(int i=0;i<=lx+1;++i) for(int j=0;j<=ly+1;++j){
        sZm[i*(ly+2)+j]=u[idx(i,j,1, NY,NZ)];
        sZp[i*(ly+2)+j]=u[idx(i,j,lz,NY,NZ)];
    }

    const int TAG=0;

    // Sequential face exchanges — each blocks until both send and recv complete
    MPI_Sendrecv(&u[idx(1,   0,0,NY,NZ)],NY*NZ,MPI_DOUBLE,nbrXm,TAG,
                 &u[idx(0,   0,0,NY,NZ)],NY*NZ,MPI_DOUBLE,nbrXm,TAG,
                 cart,MPI_STATUS_IGNORE);
    MPI_Sendrecv(&u[idx(lx,  0,0,NY,NZ)],NY*NZ,MPI_DOUBLE,nbrXp,TAG,
                 &u[idx(lx+1,0,0,NY,NZ)],NY*NZ,MPI_DOUBLE,nbrXp,TAG,
                 cart,MPI_STATUS_IGNORE);
    MPI_Sendrecv(sYm.data(),yFace,MPI_DOUBLE,nbrYm,TAG,
                 rYm.data(),yFace,MPI_DOUBLE,nbrYm,TAG,
                 cart,MPI_STATUS_IGNORE);
    MPI_Sendrecv(sYp.data(),yFace,MPI_DOUBLE,nbrYp,TAG,
                 rYp.data(),yFace,MPI_DOUBLE,nbrYp,TAG,
                 cart,MPI_STATUS_IGNORE);
    MPI_Sendrecv(sZm.data(),zFace,MPI_DOUBLE,nbrZm,TAG,
                 rZm.data(),zFace,MPI_DOUBLE,nbrZm,TAG,
                 cart,MPI_STATUS_IGNORE);
    MPI_Sendrecv(sZp.data(),zFace,MPI_DOUBLE,nbrZp,TAG,
                 rZp.data(),zFace,MPI_DOUBLE,nbrZp,TAG,
                 cart,MPI_STATUS_IGNORE);

    for(int i=0;i<=lx+1;++i) for(int k=0;k<=lz+1;++k){
        u[idx(i,0,   k,NY,NZ)]=rYm[i*(lz+2)+k];
        u[idx(i,ly+1,k,NY,NZ)]=rYp[i*(lz+2)+k];
    }
    for(int i=0;i<=lx+1;++i) for(int j=0;j<=ly+1;++j){
        u[idx(i,j,0,   NY,NZ)]=rZm[i*(ly+2)+j];
        u[idx(i,j,lz+1,NY,NZ)]=rZp[i*(ly+2)+j];
    }
}

// ============================================================
// Jacobi sweep
// ============================================================
void localJacobiSweep(const std::vector<double>&u,const std::vector<double>&f,
                      std::vector<double>&unew,int lx,int ly,int lz,
                      double hx,double hy,double hz,
                      int iLo,int iHi,int jLo,int jHi,int kLo,int kHi)
{
    (void)lx;
    const int NY=ly+2,NZ=lz+2;
    const double ihx2=1./(hx*hx),ihy2=1./(hy*hy),ihz2=1./(hz*hz);
    const double denom=2.*(ihx2+ihy2+ihz2);
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
// Residual
// ============================================================
double localResidualSq(const std::vector<double>&u,const std::vector<double>&f,
                       int lx,int ly,int lz,double hx,double hy,double hz,
                       int iLo,int iHi,int jLo,int jHi,int kLo,int kHi)
{
    (void)lx;
    const int NY=ly+2,NZ=lz+2;
    const double ihx2=1./(hx*hx),ihy2=1./(hy*hy),ihz2=1./(hz*hz);
    double s=0.;
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

// ============================================================
// I/O (rank 0 only)
// ============================================================
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

// ============================================================
// Main
// ============================================================
int main(int argc,char*argv[]){
    MPI_Init(&argc,&argv);
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

    // ---------------------------------------------------------------
    // Jacobi iteration — with defensive pre-sweep L-inf norm check
    // and BC re-stamp.
    //
    // A local L-infinity norm is computed before each exchange as a
    // sanity check for divergence.  Reduced globally via MPI_Allreduce
    // so all ranks are consistent before communication begins.
    // This was added during debugging and duplicates a full O(N^3)
    // array traversal every iteration on top of the residual pass.
    // ---------------------------------------------------------------
    double residual=0.; int iter=0;
    do{
        // Pre-sweep L-inf norm — divergence sanity check
        double localLinf=0.;
        {
            const int NY=ly+2, NZ=lz+2;
            for(int i=iLo;i<=iHi;++i)
              for(int j=jLo;j<=jHi;++j)
                for(int k=kLo;k<=kHi;++k)
                    localLinf=std::max(localLinf,std::abs(u[idx(i,j,k,NY,NZ)]));
        }
        double globalLinf=0.;
        MPI_Allreduce(&localLinf,&globalLinf,1,MPI_DOUBLE,MPI_MAX,cart);
        (void)globalLinf;

        exchangeHalos(u,lx,ly,lz,cart,nbrXm,nbrXp,nbrYm,nbrYp,nbrZm,nbrZp);

        // Re-stamp Dirichlet nodes before sweep (defensive: halo exchange
        // should not have corrupted these, but this guarantees correctness)
        if(bndXm) for(int j=1;j<=ly;++j) for(int k=1;k<=lz;++k)
            u[idx(1,j,k,LNY,LNZ)]=bcVal(gxs,gys+j-1,gzs+k-1);
        if(bndXp) for(int j=1;j<=ly;++j) for(int k=1;k<=lz;++k)
            u[idx(lx,j,k,LNY,LNZ)]=bcVal(gxs+lx-1,gys+j-1,gzs+k-1);
        if(bndYm) for(int i=1;i<=lx;++i) for(int k=1;k<=lz;++k)
            u[idx(i,1,k,LNY,LNZ)]=bcVal(gxs+i-1,gys,gzs+k-1);
        if(bndYp) for(int i=1;i<=lx;++i) for(int k=1;k<=lz;++k)
            u[idx(i,ly,k,LNY,LNZ)]=bcVal(gxs+i-1,gys+ly-1,gzs+k-1);
        if(bndZm) for(int i=1;i<=lx;++i) for(int j=1;j<=ly;++j)
            u[idx(i,j,1,LNY,LNZ)]=bcVal(gxs+i-1,gys+j-1,gzs);
        if(bndZp) for(int i=1;i<=lx;++i) for(int j=1;j<=ly;++j)
            u[idx(i,j,lz,LNY,LNZ)]=bcVal(gxs+i-1,gys+j-1,gzs+lz-1);

        localJacobiSweep(u,localF,unew,lx,ly,lz,hx,hy,hz,iLo,iHi,jLo,jHi,kLo,kHi);
        std::swap(u,unew);

        // Re-stamp after swap — sweep writes to unew which is now u;
        // stamp again to ensure BCs are clean before residual check
        if(bndXm) for(int j=1;j<=ly;++j) for(int k=1;k<=lz;++k)
            u[idx(1,j,k,LNY,LNZ)]=bcVal(gxs,gys+j-1,gzs+k-1);
        if(bndXp) for(int j=1;j<=ly;++j) for(int k=1;k<=lz;++k)
            u[idx(lx,j,k,LNY,LNZ)]=bcVal(gxs+lx-1,gys+j-1,gzs+k-1);
        if(bndYm) for(int i=1;i<=lx;++i) for(int k=1;k<=lz;++k)
            u[idx(i,1,k,LNY,LNZ)]=bcVal(gxs+i-1,gys,gzs+k-1);
        if(bndYp) for(int i=1;i<=lx;++i) for(int k=1;k<=lz;++k)
            u[idx(i,ly,k,LNY,LNZ)]=bcVal(gxs+i-1,gys+ly-1,gzs+k-1);
        if(bndZm) for(int i=1;i<=lx;++i) for(int j=1;j<=ly;++j)
            u[idx(i,j,1,LNY,LNZ)]=bcVal(gxs+i-1,gys+j-1,gzs);
        if(bndZp) for(int i=1;i<=lx;++i) for(int j=1;j<=ly;++j)
            u[idx(i,j,lz,LNY,LNZ)]=bcVal(gxs+i-1,gys+j-1,gzs+lz-1);

        double lsq=localResidualSq(u,localF,lx,ly,lz,hx,hy,hz,iLo,iHi,jLo,jHi,kLo,kHi);
        double gsq=0.;
        MPI_Allreduce(&lsq,&gsq,1,MPI_DOUBLE,MPI_SUM,cart);
        residual=std::sqrt(gsq);
        ++iter;
    } while(residual>opt.epsilon);

    // Gather and write
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