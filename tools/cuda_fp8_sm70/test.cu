#include "devices/cuda/fastllm-fp8-sm70.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(2); }
}
static void require(bool ok, const char *message) {
    if (!ok) { std::fprintf(stderr, "%s\n", message); std::exit(3); }
}
static double fp8(unsigned code) {
    const int exponent = (code >> 3) & 15, mantissa = code & 7;
    const double value = exponent ? std::ldexp(1.0 + mantissa / 8.0, exponent - 7)
                                  : std::ldexp(double(mantissa), -9);
    return code & 128 ? -value : value;
}
static bool launch(bool gdn, const half *x, const uint8_t *w, const float *scales,
                   const half *bias, half *y, int K, int N, int T) {
    if (!gdn) return fastllm::fp8sm70::Try(x,w,scales,bias,y,K,N,T,K,1,cudaStreamPerThread);
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
    fastllm::fp8sm70::GdnQuadSplitKernel<<<(N+7)/8,256,0,cudaStreamPerThread>>>(
        w,scales,x,N,K,T,fastllm::fp8sm70::Output{y,bias,N});
#endif
    return true;
}
static void run(bool gdn) {
    setenv("FASTLLM_CUDA_FP8_SM70", "1", 1);
    int device = 0; check(cudaGetDevice(&device));
    require(fastllm::fp8sm70::Supported(1024, 33, 8, device), "SM70 GPU and kernel required");
    int cases = 0; double largest_error = 0;
    // Tail output tiles, an uneven split of K, all supported token counts,
    // row scales, optional bias, and all 254 finite E4M3 bit patterns.
    for (int K : {1024, 1152}) for (int N : {1, 31, 33, 129}) {
        std::vector<half> hx(size_t(K) * 8), hb(N), hy(size_t(N) * 8), single(N);
        std::vector<unsigned char> hw(size_t(K) * N); std::vector<float> hs(N);
        for (size_t i = 0; i < hx.size(); ++i) hx[i] = __float2half(float(int((i*17+3)%101)-50)/64);
        for (size_t i = 0; i < hw.size(); ++i) { unsigned c=(i*37+i/K*11)&255; if((c&127)==127)c--; hw[i]=c; }
        for (int i = 0; i < N; ++i) { hs[i]=(i%11+1)/8192.0f; hb[i]=__float2half((i%13-6)/16.0f); }
        half *x,*bias,*y; uint8_t *w; float *scales;
        check(cudaMalloc(&x,hx.size()*2)); check(cudaMalloc(&bias,N*2)); check(cudaMalloc(&y,hy.size()*2));
        check(cudaMalloc(&w,hw.size())); check(cudaMalloc(&scales,N*4));
        check(cudaMemcpy(x,hx.data(),hx.size()*2,cudaMemcpyHostToDevice));
        check(cudaMemcpy(w,hw.data(),hw.size(),cudaMemcpyHostToDevice));
        check(cudaMemcpy(bias,hb.data(),N*2,cudaMemcpyHostToDevice));
        check(cudaMemcpy(scales,hs.data(),N*4,cudaMemcpyHostToDevice));
        for (bool use_bias : {false,true}) for (int T=1; T<=8; ++T) {
            require(launch(gdn,x,w,scales,use_bias?bias:nullptr,y,K,N,T), "Eligible dispatch rejected");
            check(cudaGetLastError()); check(cudaMemcpy(hy.data(),y,size_t(T)*N*2,cudaMemcpyDeviceToHost));
            for (int row=0;row<T;++row) for(int col=0;col<N;++col) {
                double sum=0, magnitude=0;
                for(int k=0;k<K;++k) { double p=__half2float(hx[size_t(row)*K+k])*fp8(hw[size_t(col)*K+k]);sum+=p;magnitude+=std::abs(p); }
                const double expected=sum*hs[col]+(use_bias?__half2float(hb[col]):0);
                const double actual=__half2float(hy[size_t(row)*N+col]);
                const double error=std::abs(actual-expected);
                const double bound=0.0005*std::abs(expected)+1e-6*magnitude*hs[col]+1e-6;
                largest_error=std::max(largest_error,error);
                require(std::isfinite(actual)&&error<=bound,"FP64 reference mismatch");
            }
            if(T==1)std::copy(hy.begin(),hy.begin()+N,single.begin());
            else for(int i=0;i<N;++i)require(__half2float(single[i])==__half2float(hy[i]),"First row differs between single-token and batched verification");
            ++cases;
        }
        if (!gdn) {
            auto rejected=[&](const half *a,const uint8_t *b,half*c,int t,int bm,int bk) {
                return !fastllm::fp8sm70::Try(a,b,scales,nullptr,c,K,N,t,bm,bk,cudaStreamPerThread);
            };
            require(rejected(x,w,y,0,K,1)&&rejected(x,w,y,9,K,1),"Token guard failed");
            require(rejected(x,w,y,6,128,128),"Block-scale guard failed");
            require(rejected(x+1,w,y,6,K,1)&&rejected(x,w+1,y,6,K,1),"Alignment guard failed");
            require(rejected(x,w,x,6,K,1),"Aliasing guard failed");
            setenv("FASTLLM_CUDA_FP8_SM70","0",1);require(rejected(x,w,y,6,K,1),"Fallback switch failed");
            setenv("FASTLLM_CUDA_FP8_SM70","1",1);
        }
        cudaFree(x);cudaFree(bias);cudaFree(y);cudaFree(w);cudaFree(scales);
    }
    // Isolate every finite code: mixed-sign dot products could conceal a lost
    // tiny E4M3 value through cancellation. In this fixture each output is
    // exactly the represented FP8 value (or its negative), including subnormals.
    {
        constexpr int K=1024,N=32,T=2;
        std::vector<half> hx(K*T),hy(N*T);std::vector<uint8_t> hw(K*N);std::vector<float> hs(N,1.0f/K);
        for(int k=0;k<K;++k){hx[k]=__float2half(1);hx[K+k]=__float2half(-1);}
        half *x,*y;uint8_t*w;float*s;
        check(cudaMalloc(&x,hx.size()*2));check(cudaMalloc(&y,hy.size()*2));check(cudaMalloc(&w,hw.size()));check(cudaMalloc(&s,N*4));
        check(cudaMemcpy(x,hx.data(),hx.size()*2,cudaMemcpyHostToDevice));check(cudaMemcpy(s,hs.data(),N*4,cudaMemcpyHostToDevice));
        for(int group=0;group<8;++group){
            for(int n=0;n<N;++n){unsigned code=group*32+n;if((code&127)==127)--code;std::fill(hw.begin()+n*K,hw.begin()+(n+1)*K,code);}
            check(cudaMemcpy(w,hw.data(),hw.size(),cudaMemcpyHostToDevice));
            require(launch(gdn,x,w,s,nullptr,y,K,N,T),"Isolated-code dispatch failed");
            check(cudaMemcpy(hy.data(),y,hy.size()*2,cudaMemcpyDeviceToHost));
            for(int t=0;t<T;++t)for(int n=0;n<N;++n)require(__half2float(hy[t*N+n])==(t?-1:1)*fp8(hw[n*K]),"Isolated finite FP8 code or subnormal mismatch");
            ++cases;
        }
        cudaFree(x);cudaFree(y);cudaFree(w);cudaFree(s);
    }
    std::printf("PASS %s: %d numeric cases; isolated finite FP8 codes/subnormals, T=1..8, bias, tails, uneven K split, exact first-row parity, and dispatch guards; max absolute error %.8g\n",gdn ? "GDN quad split" : "row FP8",cases,largest_error);
}

static void run_production_gdn() {
    constexpr int K = 5120, N = 16384, MaxT = 8;
    int device = 0; check(cudaGetDevice(&device));
    require(fastllm::fp8sm70::Supported(K, N, MaxT, device), "SM70 GPU and kernel required");
    std::vector<half> hx(size_t(K) * MaxT), hb(N), hy(size_t(N) * MaxT), single(N);
    std::vector<uint8_t> hw(size_t(K) * N);
    std::vector<float> hs(N);
    uint32_t state = 0x637b29a5u;
    auto random = [&]() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; };
    for (auto &value : hx) value = __float2half(float(int(random() % 2049) - 1024) / 1024);
    for (auto &code : hw) { code = random() & 255; if ((code & 127) == 127) --code; }
    for (int col = 0; col < N; ++col) {
        hs[col] = float(random() % 11 + 1) / 8192;
        hb[col] = __float2half(float(int(random() % 33) - 16) / 16);
    }
    // Cache an independent FP64 reference for every output. Reuse it across
    // token counts and bias modes rather than repeating the large CPU GEMM.
    double decoded[256] = {};
    for (unsigned code = 0; code < 256; ++code) if ((code & 127) != 127) decoded[code] = fp8(code);
    std::vector<double> sums(size_t(MaxT) * N), magnitudes(sums.size()), dx(hx.size());
    for (size_t i = 0; i < hx.size(); ++i) dx[i] = __half2float(hx[i]);
    for (int row = 0; row < MaxT; ++row) for (int col = 0; col < N; ++col) {
        double sum = 0, magnitude = 0;
        for (int k = 0; k < K; ++k) {
            const double product = dx[size_t(row) * K + k] * decoded[hw[size_t(col) * K + k]];
            sum += product; magnitude += std::abs(product);
        }
        sums[size_t(row) * N + col] = sum * hs[col];
        magnitudes[size_t(row) * N + col] = magnitude * hs[col];
    }
    half *x = nullptr, *bias, *y; uint8_t *w; float *scales;
    check(cudaMalloc(&bias, hb.size() * sizeof(half)));
    check(cudaMalloc(&y, hy.size() * sizeof(half))); check(cudaMalloc(&w, hw.size()));
    check(cudaMalloc(&scales, hs.size() * sizeof(float)));
    check(cudaMemcpy(bias, hb.data(), hb.size() * sizeof(half), cudaMemcpyHostToDevice));
    check(cudaMemcpy(w, hw.data(), hw.size(), cudaMemcpyHostToDevice));
    check(cudaMemcpy(scales, hs.data(), hs.size() * sizeof(float), cudaMemcpyHostToDevice));
    int cases = 0; size_t checked_outputs = 0; double largest_error = 0, largest_ratio = 0;
    unsetenv("FASTLLM_CUDA_FP8_SM70");
    for (bool use_bias : {false, true}) for (int T = 1; T <= MaxT; ++T) {
        // Use exactly T rows, including T=1, so memcheck can catch padded reads.
        check(cudaFree(x)); check(cudaMalloc(&x, size_t(T) * K * sizeof(half)));
        check(cudaMemcpy(x, hx.data(), size_t(T) * K * sizeof(half), cudaMemcpyHostToDevice));
        check(cudaMemset(y, 0xff, hy.size() * sizeof(half)));
        require(fastllm::fp8sm70::Try(x, w, scales, use_bias ? bias : nullptr, y,
                                    K, N, T, K, 1, cudaStreamPerThread), "Production GDN dispatch rejected");
        check(cudaGetLastError());
        check(cudaMemcpy(hy.data(), y, hy.size() * sizeof(half), cudaMemcpyDeviceToHost));
        for (int row = 0; row < T; ++row) for (int col = 0; col < N; ++col) {
            const size_t i = size_t(row) * N + col;
            const double expected = sums[i] + (use_bias ? __half2float(hb[col]) : 0);
            const double actual = __half2float(hy[i]), error = std::abs(actual - expected);
            const double bound = 0.0005 * std::abs(expected) + 1e-6 * magnitudes[i] + 1e-6;
            largest_error = std::max(largest_error, error);
            largest_ratio = std::max(largest_ratio, error / bound);
            if (!std::isfinite(actual) || error > bound) {
                std::fprintf(stderr, "Production GDN mismatch T=%d bias=%d row=%d col=%d actual=%.12g expected=%.12g bound=%.12g\n",
                             T, int(use_bias), row, col, actual, expected, bound);
                std::exit(3);
            }
            ++checked_outputs;
        }
        for (size_t i = size_t(T) * N; i < hy.size(); ++i)
            require(std::isnan(__half2float(hy[i])), "Production GDN wrote beyond T rows");
        if (T == 1) std::copy(hy.begin(), hy.begin() + N, single.begin());
        else for (int col = 0; col < N; ++col)
            require(__half2float(single[col]) == __half2float(hy[col]), "Production GDN first-row parity failed");
        ++cases;
    }
    // Disabled dispatch must not launch a kernel or change the output buffer.
    for (const char *flag : {"0", "false"}) {
        setenv("FASTLLM_CUDA_FP8_SM70", flag, 1);
        check(cudaMemset(y, 0xff, hy.size() * sizeof(half)));
        require(!fastllm::fp8sm70::Try(x, w, scales, nullptr, y, K, N, 6, K, 1,
                                     cudaStreamPerThread), "Production GDN fallback switch failed");
        check(cudaMemcpy(hy.data(), y, hy.size() * sizeof(half), cudaMemcpyDeviceToHost));
        for (half value : hy) require(std::isnan(__half2float(value)), "Rejected GDN dispatch modified output");
    }
    unsetenv("FASTLLM_CUDA_FP8_SM70");
    check(cudaFree(x)); check(cudaFree(bias)); check(cudaFree(y)); check(cudaFree(w)); check(cudaFree(scales));
    std::printf("PASS production GDN Try: K=%d N=%d, %d numeric cases, %zu outputs checked against FP64; default dispatch, T=1..8, bias, exact first-row parity, output bounds, and 0/false fallback; max absolute error %.8g, max error/bound %.8g\n",
                K, N, cases, checked_outputs, largest_error, largest_ratio);
}

int main(int argc, char **argv) {
    require(argc == 1 || (argc == 2 && std::strcmp(argv[1], "--production-only") == 0),
            "Usage: test [--production-only]");
    if (argc == 1) { run(false); run(true); }
    run_production_gdn();
}
