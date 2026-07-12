#include "q6_q8.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
using namespace q4_hotpath;
int main(){std::mt19937 g(8);volatile float sink=0;for(int blocks:{1,4,8,16}){std::vector<block_q6_k>x(blocks);std::vector<block_q8_k>y(blocks);for(auto&b:x){for(auto&v:b.ql)v=g();for(auto&v:b.qh)v=g();for(auto&v:b.scales)v=g();b.d=0x3c00;}for(auto&b:y){b.d=.01f;for(auto&v:b.qs)v=g();for(int j=0;j<16;++j){int s=0;for(int k=0;k<16;++k)s+=b.qs[j*16+k];b.bsums[j]=s;}}
 for(auto name:{"reference","baseline","optimized"}){auto fn=name[0]=='r'?reference:(name[0]=='b'?baseline:optimized);for(int i=0;i<1000;++i)sink+=fn(x.data(),y.data(),blocks);std::vector<double> ns;for(int rep=0;rep<9;++rep){auto a=std::chrono::steady_clock::now();for(int i=0;i<20000;++i)sink+=fn(x.data(),y.data(),blocks);auto z=std::chrono::steady_clock::now();ns.push_back(std::chrono::duration<double,std::nano>(z-a).count()/20000);}std::sort(ns.begin(),ns.end());double mean=0;for(double v:ns)mean+=v;mean/=ns.size();double var=0;for(double v:ns)var+=(v-mean)*(v-mean);double cv=std::sqrt(var/(ns.size()-1))/mean*100;std::cout<<"shape=blocks"<<blocks<<" kernel="<<name<<" median_ns="<<std::fixed<<std::setprecision(2)<<ns[4]<<" cv_percent="<<cv<<" blocks_per_s="<<blocks/(ns[4]*1e-9)<<"\n";}}
 std::cerr<<"checksum="<<sink<<"\n";}
