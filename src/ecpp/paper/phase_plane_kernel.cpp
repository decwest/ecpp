// Test 3 of the ECPP paper: phase-plane sweep of initial conditions on an
// infinite straight path.  Identical to the review-time kernel
// recovery_phase_plane_kernel.cpp (sha256 49aa9b1a4923385094705e16d544465bb31be3c25d70b012b217baac35d38027)
// apart from this header comment.  Driven by ecpp.paper.test3_phase_plane.
// Offline straight-path exploration; no manuscript or controller writes.
// Compile with g++ -O3 -std=c++17 -fopenmp (intentionally no fast-math).
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

constexpr double PI = 3.14159265358979323846;
constexpr double V = 0.5, LIMIT = 1.5;
constexpr double Y_TOL = 0.02, TH_TOL = PI / 180.0, HOLD = 5.0;
constexpr int NF = 14;
using Result = std::array<float, NF>;

struct Config {
    std::string name;
    int method; // 0 PP, 1 DPP, 2 ungated ECPP, 3 ECPP
    double ld, wn, zeta, dt, horizon;
    double ymin, dy; int ny;
    double thmin, dth; int nth;
};

inline double wrap(double t) {
    if (t >= PI) t -= 2 * PI;
    if (t < -PI) t += 2 * PI;
    return t;
}

Result simulate(const Config& c, double y, double th) {
    const int n = std::lround(c.horizon / c.dt);
    const double ld2 = c.ld*c.ld;
    const double ky = std::pow(c.wn/V, 2), kth = 2*c.zeta*c.wn/V;
    const double dky = ky - 2/ld2, dkth = kth - 2/c.ld;
    const double slope = 2*std::log(99.0)/0.4;
    const int n30 = std::lround(30/c.dt), n60 = std::lround(60/c.dt);
    int last_out = -1, sat = 0, tail_sat = 0, tail_n = 0;
    double max_y = std::abs(y), peak_raw = 0, net_angle = 0;
    double tail_min = y, tail_max = y;
    Result out{}; out[0] = out[1] = out[2] = -1;
    // Analytically exact equilibria must not escape because sin(pi) rounds
    // to 1e-16. No other states or small perturbations are snapped to zero.
    const bool equilibrium = y == 0.0 && (th == 0.0 || th == -PI || th == PI);
    for (int i=0; i<=n; ++i) {
        if (std::abs(y)>Y_TOL || std::abs(th)>TH_TOL) last_out=i;
        auto settle = [&]() -> float {
            const double ts = (last_out+1)*c.dt;
            return i*c.dt-ts >= HOLD-1e-9 ? float(ts) : -1.f;
        };
        if(i==n30) out[0]=settle();
        if(i==n60) out[1]=settle();
        if(i==n) {out[2]=settle(); break;}
        double s, co; sincos(th, &s, &co);
        if(equilibrium) s=0.0;
        double raw;
        if(c.method==1) raw=-V*(ky*y+kth*s);
        else {
            const double pp=-2*(y*co+c.ld*s)/(ld2+y*y);
            double sig=0;
            if(c.method==2) sig=1;
            if(c.method==3) {
                const double a=slope*(y*y/ld2-0.3);
                if(a>=0) {const double e=std::exp(-a); sig=e/(1+e);}
                else sig=1/(1+std::exp(a));
            }
            raw=V*(pp-sig*(dky*y+dkth*s));
        }
        peak_raw=std::max(peak_raw, std::abs(raw));
        const bool clipped=std::abs(raw)>LIMIT;
        sat+=clipped;
        if(i*c.dt>=c.horizon-10-1e-9) {
            if(tail_n==0) tail_min=tail_max=y;
            tail_min=std::min(tail_min,y); tail_max=std::max(tail_max,y);
            tail_sat+=clipped; ++tail_n;
        }
        const double om=std::clamp(raw,-LIMIT,LIMIT);
        const double angle=om*c.dt;
        // Exact constant-twist unicycle update under sample-and-hold.
        // Midpoint/sinc avoids cancellation when omega approaches zero.
        const double h=0.5*angle;
        const double sinc=std::abs(h)<1e-5 ? 1-h*h/6+h*h*h*h/120 : std::sin(h)/h;
        if(!equilibrium) y+=V*c.dt*std::sin(th+h)*sinc;
        th=wrap(th+angle); net_angle+=angle;
        max_y=std::max(max_y,std::abs(y));
    }
    out[3]=float(double(sat)/n); out[4]=float(max_y);
    out[5]=float(y); out[6]=float(th);
    out[7]=float(tail_min); out[8]=float(tail_max);
    out[9]=float(double(tail_sat)/std::max(1,tail_n));
    out[10]=float(net_angle/(2*PI)); out[11]=float(peak_raw);
    out[12]=equilibrium ? 1.f : 0.f;
    out[13]=float((last_out+1)*c.dt);
    return out;
}

int main(int argc, char** argv) {
    if(argc!=3 && argc!=4) {
        std::cerr<<"usage: kernel config.tsv output_dir [points.tsv]\n"; return 2;
    }
    std::ifstream in(argv[1]); Config c;
    if(!in) return 3;
    std::vector<std::array<double,2>> points;
    if(argc==4) {
        std::ifstream p(argv[3]); double y,t;
        while(p>>y>>t) points.push_back({y,t*PI/180});
        if(points.empty()) return 4;
    }
    while(in>>c.name>>c.method>>c.ld>>c.wn>>c.zeta>>c.dt>>c.horizon
            >>c.ymin>>c.dy>>c.ny>>c.thmin>>c.dth>>c.nth) {
        const auto start=std::chrono::steady_clock::now();
        const long count=points.empty()?long(c.ny)*c.nth:points.size();
        std::vector<Result> results(count);
        #pragma omp parallel for schedule(dynamic, 32)
        for(long j=0;j<count;++j) {
            const double y=points.empty()?c.ymin+(j/c.nth)*c.dy:points[j][0];
            const double th=points.empty()?(c.thmin+(j%c.nth)*c.dth)*PI/180:points[j][1];
            results[j]=simulate(c,y,th);
        }
        std::ofstream file(std::string(argv[2])+"/"+c.name+".bin",std::ios::binary);
        file.write(reinterpret_cast<const char*>(results.data()),count*sizeof(Result));
        if(!file) return 5;
        long settled=0; for(auto& r:results) settled+=r[2]>=0;
        const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::cout<<c.name<<" "<<settled<<"/"<<count<<" settled; "
                 <<std::fixed<<std::setprecision(2)<<elapsed<<" s\n"<<std::flush;
    }
}
