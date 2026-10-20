// CPU numerical-reference helper only. Not part of the Unity runtime or model.
#include <immintrin.h>
#include <algorithm>
#include <cstdint>
#include <thread>
#include <vector>
#ifdef _MSC_VER
#include <intrin.h>
#define EG_EXPORT extern "C" __declspec(dllexport)
#else
#include <cpuid.h>
#define EG_EXPORT extern "C" __attribute__((visibility("default")))
#endif

EG_EXPORT int eg_ref_capabilities() {
    unsigned a,b,c,d;
#ifdef _MSC_VER
    int regs[4]; __cpuidex(regs,1,0); c=static_cast<unsigned>(regs[2]);
#else
    if(!__get_cpuid(1,&a,&b,&c,&d)) return 0;
#endif
    if((c & ((1u<<12)|(1u<<27)|(1u<<28))) != ((1u<<12)|(1u<<27)|(1u<<28))) return 0;
    if((_xgetbv(0)&6u)!=6u) return 0;
#ifdef _MSC_VER
    __cpuidex(regs,7,0); b=static_cast<unsigned>(regs[1]);
#else
    __cpuid_count(7,0,a,b,c,d);
#endif
    return (b&(1u<<5)) ? 1 : 0;
}
EG_EXPORT unsigned eg_ref_mxcsr() { return _mm_getcsr(); }

template<class Function> int parallel(std::int64_t rows,int requested,Function fn) {
    const int count=static_cast<int>(std::min<std::int64_t>(rows,requested));
    const unsigned environment=_mm_getcsr();
    auto run=[&](int index) {
        const unsigned previous=_mm_getcsr(); _mm_setcsr(environment);
        for(std::int64_t row=index;row<rows;row+=count) fn(row);
        _mm_setcsr(previous);
    };
    std::vector<std::thread> threads;
    try {
        threads.reserve(count-1);
        for(int index=1;index<count;++index) threads.emplace_back(run,index);
        run(0);
    } catch(...) {
        for(auto& thread:threads) if(thread.joinable()) thread.join();
        return -2;
    }
    for(auto& thread:threads) thread.join();
    return 0;
}

EG_EXPORT int eg_ref_matmul(const float* left,const float* right,const float* bias,float* output,
    std::int64_t batches,std::int64_t rows,std::int64_t width,std::int64_t columns,int threads) {
    if(!left||!right||!output||batches<1||rows<1||width<1||columns<1||threads<1||threads>4) return -1;
    const __m256 zero=_mm256_setzero_ps();
    return parallel(batches*rows,threads,[&](std::int64_t row) {
        const std::int64_t batch=row/rows;
        const float* a=left+row*width;
        const float* b=right+batch*width*columns;
        float* result=output+row*columns;
        std::int64_t column=0;
        for(;column+8<=columns;column+=8) {
            __m256 total=zero;
            for(std::int64_t start=0;start<width;start+=192) {
                const auto end=std::min(start+192,width);
                __m256 partial=zero;
                for(auto k=start;k<end;++k)
                    partial=_mm256_fmadd_ps(_mm256_set1_ps(a[k]),_mm256_loadu_ps(b+k*columns+column),partial);
                if(end-start<192) partial=_mm256_add_ps(partial,zero);
                if(start==0) total=bias?_mm256_add_ps(partial,_mm256_loadu_ps(bias+column)):partial;
                else total=_mm256_add_ps(total,partial);
            }
            _mm256_storeu_ps(result+column,total);
        }
        for(;column<columns;++column) {
            float total=0;
            for(std::int64_t start=0;start<width;start+=192) {
                const auto end=std::min(start+192,width);
                __m128 partial=_mm_setzero_ps();
                for(auto k=start;k<end;++k)
                    partial=_mm_fmadd_ss(_mm_set_ss(a[k]),_mm_set_ss(b[k*columns+column]),partial);
                if(end-start<192) partial=_mm_add_ss(partial,_mm_setzero_ps());
                const float value=_mm_cvtss_f32(partial);
                total=start==0?(bias?value+bias[column]:value):total+value;
            }
            result[column]=total;
        }
    });
}

EG_EXPORT int eg_ref_fma(const float* a,const float* b,const float* c,float* output,std::int64_t size,int threads) {
    if(!a||!b||!c||!output||size<1||threads<1||threads>4) return -1;
    const auto blocks=(size+7)/8;
    return parallel(blocks,threads,[&](std::int64_t block) {
        const auto index=block*8;
        if(index+8<=size) _mm256_storeu_ps(output+index,_mm256_fmadd_ps(_mm256_loadu_ps(a+index),_mm256_loadu_ps(b+index),_mm256_loadu_ps(c+index)));
        else for(auto i=index;i<size;++i) output[i]=_mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(a[i]),_mm_set_ss(b[i]),_mm_set_ss(c[i])));
    });
}
