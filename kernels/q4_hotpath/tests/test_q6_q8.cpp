#include "q6_q8.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

using namespace q4_hotpath;
static void fill(block_q6_k &x, block_q8_k &y, std::mt19937 &g) {
    std::uniform_int_distribution<int> u8(0,255), s8(-128,127), sc(-127,127);
    for(auto &v:x.ql)v=u8(g);
    for(auto &v:x.qh)v=u8(g);
    for(auto &v:x.scales)v=sc(g);
    x.d=0x3c00; y.d=0.00390625f; for(auto &v:y.qs)v=s8(g);
    for(int j=0;j<16;++j){int sum=0;for(int k=0;k<16;++k)sum+=y.qs[j*16+k];y.bsums[j]=sum;}
}
int main(){
    std::mt19937 g(0x8a11ce); std::size_t trials=0,mismatches=0,bit_mismatches=0,ref_mismatches=0; double max_abs=0,max_rel=0,mean=0;
    for(int mode=0;mode<4;++mode) for(int t=0;t<500;++t){std::size_t n=1+(t%16);std::vector<block_q6_k>x(n);std::vector<block_q8_k>y(n);for(std::size_t i=0;i<n;++i)fill(x[i],y[i],g);
        if(mode==0){std::memset(x.data(),0,x.size()*sizeof(*x.data()));std::memset(y.data(),0,y.size()*sizeof(*y.data()));}
        if(mode==1)for(auto&b:x)std::fill(std::begin(b.scales),std::end(b.scales),int8_t(127));
        if(mode==2)for(auto&b:x)std::fill(std::begin(b.scales),std::end(b.scales),int8_t(-128));
        float r=reference(x.data(),y.data(),n),b=baseline(x.data(),y.data(),n),o=optimized(x.data(),y.data(),n); double ae=std::abs(double(r)-o),re=ae/std::max(1.0,std::abs(double(r)));max_abs=std::max(max_abs,ae);max_rel=std::max(max_rel,re);mean+=ae;++trials;if(b!=o)++bit_mismatches;if(ae>1e-3+5e-4*std::abs(r))++ref_mismatches;}
    mismatches=bit_mismatches+ref_mismatches;mean/=trials;std::cout<<"trials="<<trials<<" mismatches="<<mismatches<<" bit_mismatches="<<bit_mismatches<<" ref_mismatches="<<ref_mismatches<<" max_abs="<<max_abs<<" max_rel="<<max_rel<<" mean_abs="<<mean<<"\n";return mismatches?1:0;}
