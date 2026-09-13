// Optional CPU-reference arithmetic on CUDA. Keep explicit FP32 rounding
// boundaries; this file is compiled with --fmad=false, with FMA only where
// the reference BLAS/BRGEMM uses it. The normal CUDA operators stay available.
#include "deepseekv41-reference.cuh"
#include "deepseekv41-reference-math.cuh"
#include "fastllm-cuda.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
using namespace fastllm;
namespace {
    using namespace fastllm::v41ref;
    void Prepare(Data &o, DataType type, const std::vector<int> &dims) {
        o.dataType = type;
        o.Resize(dims);
        o.ToDevice(DataDevice::CUDA, {FastllmCudaGetDevice()}, false);
        o.Allocate(false);
    }
    __global__ void CastIn(const __nv_bfloat16 *x, float *y, size_t n) {
        size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
        if (i < n)
            y[i] = __bfloat162float(x[i]);
    }
    __global__ void CastOut(const float *x, __nv_bfloat16 *y, size_t n) {
        size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
        if (i < n)
            y[i] = __float2bfloat16_rn(x[i]);
    }
    size_t LogicalCount(const Data &x) {
        size_t n = 1;
        for (int d : x.dims)
            n *= d;
        return n;
    }
    const Data &FloatInput(const Data &x, Data &tmp) {
        if (x.dataType == FLOAT32)
            return x;
        Prepare(tmp, FLOAT32, x.dims);
        size_t n = LogicalCount(x);
        CastIn<<<(n + 255) / 256, 256>>>((const __nv_bfloat16 *)x.cudaData, (float *)tmp.cudaData, n);
        return tmp;
    }
    void Store(const Data &raw, Data &o, const std::vector<int> &dims) {
        Prepare(o, BFLOAT16, dims);
        CastOut<<<(raw.Count(0) + 255) / 256, 256>>>((const float *)raw.cudaData, (__nv_bfloat16 *)o.cudaData,
                                                     raw.Count(0));
    }
    bool Valid(const Data &x) {
        return x.dataDevice == DataDevice::CUDA && x.cudaData && !x.multiDeviceData &&
               (x.dataType == BFLOAT16 || x.dataType == FLOAT32);
    }
    bool Finish() { return cudaGetLastError() == cudaSuccess; }
    __global__ void ExecuteReference(Task t) {
        size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < t.work)
            Execute(t, i);
    }
    __global__ void ReferenceAttentionParallel(Task t) {
        const int item = blockIdx.x, lane = threadIdx.x;
        const int token = item / t.heads, head = item % t.heads, hd = t.dim;
        const float *q = t.p[Input] + (size_t(token) * t.heads + head) * hd;
        float *num = t.p[Output] + (size_t(token) * t.heads + head) * hd;
        const int *positions = reinterpret_cast<const int *>(t.p[Indices]) + size_t(token) * t.width;
        __shared__ float scores[64], raw[64], probs[64];
        __shared__ int indices[64];
        __shared__ float maximum, denominator, correction;
        if (lane == 0) { maximum = -1e30f; denominator = 0; }
        for (int j = lane; j < hd; j += blockDim.x) num[j] = 0;
        __syncthreads();
        for (int first = 0; first < t.width; first += 64) {
            if (lane < 64) {
                int pos = first + lane < t.width ? positions[first + lane] : -1;
                if (pos < 0 || pos >= t.offset + t.visible) pos = -1;
                indices[lane] = pos;
                float dot = 0;
                if (pos >= 0) {
                    const float *key = pos < t.offset ? t.p[Keys] + size_t(pos) * hd
                                                      : t.p[Values] + size_t(pos - t.offset) * hd;
                    for (int firstK = 0; firstK < hd; firstK += 192) {
                        float partial = 0;
                        for (int j = firstK; j < MinI(firstK + 192, hd); ++j)
                            partial = fmaf(q[j], key[j], partial);
                        dot += partial;
                    }
                }
                scores[lane] = pos >= 0 ? dot * t.factor : -INFINITY;
            }
            __syncthreads();
            if (lane == 0) {
                float next = maximum;
                for (int i = 0; i < 64; ++i) next = Max(next, scores[i]);
                correction = TensorExp(maximum - next);
                maximum = next;
            }
            __syncthreads();
            if (lane < 64) {
                raw[lane] = TensorExp(scores[lane] - maximum);
                probs[lane] = BF(raw[lane]);
            }
            __syncthreads();
            if (lane == 0)
                denominator = denominator * correction + Sum(64, Load{raw});
            // Parallelize independent output channels, retaining the same
            // 64-slot sequential FMA and the online-softmax rounding boundary.
            for (int j = lane; j < hd; j += blockDim.x) {
                float dot = 0;
                for (int i = 0; i < 64; ++i) {
                    int pos = indices[i];
                    if (pos >= 0) {
                        float value = pos < t.offset ? t.p[Keys][size_t(pos) * hd + j]
                            : t.p[Values][size_t(pos - t.offset) * hd + j];
                        dot = fmaf(probs[i], value, dot);
                    }
                }
                num[j] = num[j] * correction + dot;
            }
            __syncthreads();
        }
        if (lane == 0) denominator += TensorExp(t.p[Sink][head] - maximum);
        __syncthreads();
        for (int j = lane; j < hd; j += blockDim.x) num[j] = BF(num[j] / denominator);
    }
    // The 32 threads represent the reference's four streams of eight lanes.
    // Carry propagation and the final scalar fold retain their original order.
    template<class Loader> __device__ float WarpReferenceSum(size_t count, const Loader &load) {
        int lane = threadIdx.x & 31;
        if (count < 8) {
            float total = lane == 0 ? Sum(count, load) : 0;
            return __shfl_sync(0xffffffff, total, 0);
        }
        size_t groups = count/32, bits = 0;
        for (size_t v = groups ? groups-1 : 0; v; v >>= 1) ++bits;
        size_t power = bits/4 > 4 ? bits/4 : 4, step = size_t(1) << power, group = 0;
        float sums[4] = {};
        while (group + step <= groups) {
            for (size_t end = group+step; group < end; ++group) sums[0] += load(group*32+lane);
            for (int level = 1; level < 4; ++level) {
                sums[level] += sums[level-1]; sums[level-1] = 0;
                if (group & ((step-1) << (level*power))) break;
            }
        }
        for (; group < groups; ++group) sums[0] += load(group*32+lane);
        for (int level = 1; level < 4; ++level) sums[0] += sums[level];
        if (lane < 8)
            for (size_t i = groups*32; i+8 <= count; i += 8) sums[0] += load(i+lane);
        for (int stream = 1; stream < 4; ++stream) {
            float value = __shfl_sync(0xffffffff, sums[0], (lane&7)+stream*8);
            if (lane < 8) sums[0] += value;
        }
        float total = 0;
        if (lane == 0) for (size_t i = count/8*8; i < count; ++i) total += load(i);
        for (int i = 0; i < 8; ++i) {
            float value = __shfl_sync(0xffffffff, sums[0], i);
            if (lane == 0) total += value;
        }
        return __shfl_sync(0xffffffff, total, 0);
    }
    // Keep one matrix element per lane. Each row/column fold still visits
    // elements 0, 1, 2, 3 in order; Sinkhorn iterations stay in registers.
    __device__ void ReferenceHCMixes4(Task t, int row, float inverse) {
        const int lane = threadIdx.x;
        if (lane == 0) t.p[Output4][row] = inverse;
        float mix = lane < 24 ? t.p[Mix][row*24+lane]*inverse : 0;
        if (lane < 24) t.p[Output3][row*24+lane] = mix;
        float postMix = __shfl_sync(0xffffffff, mix, lane+4 < 32 ? lane+4 : 0);
        if (lane < 4) {
            float pre = -(mix*t.p[Scale][0]+t.p[Base][lane]);
            float post = -(postMix*t.p[Scale][1]+t.p[Base][4+lane]);
            bool vector = row*4+lane < t.rows*4/32*32;
            t.p[Output][row*4+lane] = 1.0f/(1.0f+(vector ? ReductionExp(pre) : Exp(pre)))+t.hcEps;
            t.p[Output1][row*4+lane] = 2.0f/(1.0f+(vector ? ReductionExp(post) : Exp(post)));
        }
        float value = __shfl_sync(0xffffffff, mix, lane < 16 ? lane+8 : 0);
        value = lane < 16 ? value*t.p[Scale][2]+t.p[Base][8+lane] : 0;
        const int first = lane&~3, col = lane&3;
        float maximum = -INFINITY;
        for (int j = 0; j < 4; ++j)
            maximum = Max(maximum, __shfl_sync(0xffffffff, value, first+j));
        value = ReductionExp(value-maximum);
        float total = 0;
        for (int j = 0; j < 4; ++j) total += __shfl_sync(0xffffffff, value, first+j);
        float reciprocal = 1.0f/total;
        value = value*reciprocal+t.hcEps;
        for (int it = 0; it < t.iterations; ++it) {
            if (it) {
                total = 0;
                for (int j = 0; j < 4; ++j) total += __shfl_sync(0xffffffff, value, first+j);
                value /= total+t.hcEps;
            }
            total = 0;
            for (int i = 0; i < 4; ++i) total += __shfl_sync(0xffffffff, value, i*4+col);
            value /= total+t.hcEps;
        }
        if (lane < 16) t.p[Output2][row*16+lane] = value;
    }
    __global__ void ReferenceNormParallel(Task t) {
        const int row = blockIdx.x, lane = threadIdx.x;
        const bool pre = t.op == V41ReferenceOp::HCPreNorm;
        const int count = pre ? t.dim : t.cols;
        __shared__ float inverse;
        if (lane < 32) {
            float total = pre ? WarpReferenceSum(count, HCPreSquares{t, size_t(row)})
                : WarpReferenceSum(count, Load{t.p[Input]+size_t(row)*count, true});
            if (lane == 0) inverse = 1.0f/sqrtf(total/count+t.eps);
        }
        __syncthreads();
        if (t.op == V41ReferenceOp::HCMixes) {
            if (t.h == 4) {
                if (lane < 32) ReferenceHCMixes4(t, row, inverse);
            } else if (lane == 0) HCMixesFromInv(t, row, inverse);
        } else {
            for (int j = lane; j < count; j += blockDim.x) {
                float value = pre ? HCPreValue(t, row, j) : t.p[Input][size_t(row)*count+j];
                t.p[Output][size_t(row)*count+j] = BF((value*inverse)*Value(t,j));
            }
        }
    }
    __global__ void ReferenceLinearFringeCoalesced(Task t) {
        const int item = blockIdx.x, lane = threadIdx.x;
        const int row = item%t.out, token = item/t.out;
        if (row < t.out/16*16 && token < t.rows/2*2) return;
        const float *a = t.p[Input]+size_t(token)*t.cols+(row/(t.out/t.groups))*t.width;
        float partial = 0;
        const int prefix = t.rows == 1 ? MinI(4,t.width) : 0;
        if ((lane&3) == 0) for (int k = 0; k < prefix; ++k)
            partial += a[k]*Value(t,size_t(row)*t.width+k);
        // A warp loads contiguous K values. Replicate the four accumulators
        // across its eight quads, preserving every non-fused addition.
        for (int first = prefix; first < t.width; first += 32) {
            int k = first+lane;
            float product = k < t.width ? a[k]*Value(t,size_t(row)*t.width+k) : 0;
            for (int j = 0; j < 8; ++j) {
                float value = __shfl_sync(0xffffffff,product,j*4+(lane&3));
                if (first+j*4+(lane&3) < t.width) partial += value;
            }
        }
        float p0=__shfl_sync(0xffffffff,partial,0),p1=__shfl_sync(0xffffffff,partial,1);
        float p2=__shfl_sync(0xffffffff,partial,2),p3=__shfl_sync(0xffffffff,partial,3);
        if (lane == 0) t.p[Output][item]=(p0+p2)+(p1+p3);
    }
    __global__ void ReferenceLinearFringe(Task t) {
        const size_t item = (size_t(blockIdx.x)*blockDim.x+threadIdx.x)/4;
        const int lane = threadIdx.x&3;
        if (item >= t.work) return;
        const int row = item%t.out, token = item/t.out;
        if (row < t.out/16*16 && token < t.rows/2*2) return;
        const float *a = t.p[Input]+size_t(token)*t.cols+(row/(t.out/t.groups))*t.width;
        float partial = 0;
        const int prefix = t.rows == 1 ? MinI(4,t.width) : 0;
        if (lane == 0) for (int k = 0; k < prefix; ++k)
            partial += a[k]*Value(t,size_t(row)*t.width+k);
        for (int k = prefix+lane; k < t.width; k += 4)
            partial += a[k]*Value(t,size_t(row)*t.width+k);
        const unsigned active = __activemask();
        float p0 = __shfl_sync(active,partial,0,4), p1 = __shfl_sync(active,partial,1,4);
        float p2 = __shfl_sync(active,partial,2,4), p3 = __shfl_sync(active,partial,3,4);
        if (lane == 0) t.p[Output][item] = (p0+p2)+(p1+p3);
    }
    __global__ void ReferenceLinearFullBlocks(Task t) {
        const int item = blockIdx.x, lane = threadIdx.x;
        const int row = item%t.out, token = item/t.out;
        if (row >= t.out/16*16 || token >= t.rows/2*2) return;
        const float *a = t.p[Input]+size_t(token)*t.cols+(row/(t.out/t.groups))*t.width;
        const int blocks = (t.width+191)/192;
        extern __shared__ float partials[];
        for (int b = lane; b < blocks; b += blockDim.x) {
            float partial = 0;
            for (int k = b*192; k < MinI((b+1)*192,t.width); ++k)
                partial = fmaf(a[k],Value(t,size_t(row)*t.width+k),partial);
            partials[b] = partial;
        }
        __syncthreads();
        if (lane == 0) {
            float total = 0;
            for (int b = 0; b < blocks; ++b) total += partials[b];
            t.p[Output][item] = total;
        }
    }
    __global__ void ReferenceLinearBF16Blocks(Task t) {
        const int item = blockIdx.x, lane = threadIdx.x;
        const int row = item%t.out, token = item/t.out;
        const float *a = t.p[Input]+size_t(token)*t.cols+(row/(t.out/t.groups))*t.width;
        const int block = t.rows > 1 && t.width > 1024 ? 1024 : 512;
        const int blocks = (t.width+block-1)/block;
        extern __shared__ float partials[];
        for (int b = lane; b < blocks; b += blockDim.x) {
            float partial = 0;
            for (int k = b*block; k < MinI((b+1)*block,t.width); k += 2) {
                partial = fmaf(a[k+1],Value(t,size_t(row)*t.width+k+1),partial);
                partial = fmaf(a[k],Value(t,size_t(row)*t.width+k),partial);
            }
            partials[b] = partial;
        }
        __syncthreads();
        if (lane == 0) {
            float total = 0;
            for (int b = 0; b < blocks; ++b) total += partials[b];
            t.p[Output][item] = BF(total);
        }
    }
    void Launch(Task &t) {
        if (t.op == V41ReferenceOp::SparseAttention)
            ReferenceAttentionParallel<<<t.work, 128>>>(t);
        else if (t.op == V41ReferenceOp::RMSNorm || t.op == V41ReferenceOp::HCPreNorm || t.op == V41ReferenceOp::HCMixes)
            ReferenceNormParallel<<<t.work, 128>>>(t);
        else if (t.op == V41ReferenceOp::Linear && t.dtype == 1 && t.mode && t.rows == 1 && !(t.width&1)) {
            int block = t.rows > 1 && t.width > 1024 ? 1024 : 512;
            ReferenceLinearBF16Blocks<<<t.work,32,((t.width+block-1)/block)*sizeof(float)>>>(t);
        }
        else if (t.op == V41ReferenceOp::Linear && t.dtype < 2 && !t.mode && t.width <= 32768) {
            // Tiny HC projections would otherwise place all outputs on one SM.
            if ((t.rows == 1 || t.work <= 256) && t.width >= 1024)
                ReferenceLinearFringeCoalesced<<<t.work,32>>>(t);
            else
                ReferenceLinearFringe<<<(t.work+31)/32,128>>>(t);
            if (t.rows > 1 && t.out >= 16)
                ReferenceLinearFullBlocks<<<t.work,128,((t.width+191)/192)*sizeof(float)>>>(t);
        }
        else
            ExecuteReference<<<(t.work + 127) / 128, 128>>>(t);
    }

    __global__ void MakeIndices(int *out, const int *compressed, int rows, int window, int start, int slots,
                                int width, int offset) {
        int i = blockIdx.x * 256 + threadIdx.x;
        if (i >= rows * (slots + width))
            return;
        int token = i / (slots + width), col = i % (slots + width);
        if (col < slots) {
            int begin = start == 0 ? max(0, token - window + 1) : start + token - window + 1;
            int pos = begin + col;
            out[i] = pos < 0 || pos > start + token ? -1
                     : start == 0                   ? pos
                     : pos >= start                 ? window + pos - start
                                                    : pos % window;
        } else {
            int c = compressed[token * width + col - slots];
            out[i] = c < 0 ? -1 : offset + c;
        }
    }
} // namespace
extern "C" bool FastllmCudaV41ReferenceNorm(const Data &x, Data &w, float eps, Data &o) {
    if (!Valid(x) || w.dataType != FLOAT32 || x.dims.empty())
        return false;
    w.ToDevice(DataDevice::CUDA);
    Data temp, raw;
    const Data &input = FloatInput(x, temp);
    Prepare(raw, FLOAT32, x.dims);
    Task t;
    t.op = V41ReferenceOp::RMSNorm;
    t.cols = x.dims.back();
    t.rows = LogicalCount(x) / t.cols;
    t.work = t.rows;
    t.eps = eps;
    t.p[Input] = (float *)input.cudaData;
    t.p[Weight] = (float *)w.cudaData;
    t.p[Output] = (float *)raw.cudaData;
    Launch(t);
    Store(raw, o, x.dims);
    return Finish();
}
extern "C" bool FastllmCudaV41ReferencePreNorm(const Data &x, const Data &pre, Data &w, float eps, Data &o) {
    if (!Valid(x) || x.dims.size() != 4 || pre.dataDevice != DataDevice::CUDA || pre.dataType != FLOAT32 ||
        w.dataType != FLOAT32)
        return false;
    w.ToDevice(DataDevice::CUDA);
    Data temp, raw;
    const Data &input = FloatInput(x, temp);
    auto dims = x.dims;
    dims.erase(dims.begin() + 2);
    Prepare(raw, FLOAT32, dims);
    Task t;
    t.op = V41ReferenceOp::HCPreNorm;
    t.h = x.dims[2];
    t.dim = x.dims[3];
    t.cols = t.h * t.dim;
    t.rows = LogicalCount(x) / t.cols;
    t.work = t.rows;
    t.eps = eps;
    t.p[Input] = (float *)input.cudaData;
    t.p[Weight] = (float *)w.cudaData;
    t.p[Mix] = (float *)pre.cudaData;
    t.p[Output] = (float *)raw.cudaData;
    Launch(t);
    Store(raw, o, dims);
    return Finish();
}
extern "C" bool FastllmCudaV41ReferenceMix(const Data &x, Data &fn, Data &scale, Data &base, int h, int it,
                                           float eps, float normEps, Data &pre, Data &post, Data &comb) {
    if (!Valid(x) || x.dims.size() != 4 || h < 1 || h > 4 || x.dims[2] != h || fn.dataType != FLOAT32 ||
        scale.dataType != FLOAT32 || base.dataType != FLOAT32)
        return false;
    fn.ToDevice(DataDevice::CUDA);
    scale.ToDevice(DataDevice::CUDA);
    base.ToDevice(DataDevice::CUDA);
    Data temp, mix, normalized, inverse;
    const Data &input = FloatInput(x, temp);
    int rows = x.dims[0] * x.dims[1], m = 2 * h + h * h;
    Prepare(mix, FLOAT32, {rows, m});
    Prepare(normalized, FLOAT32, {rows, m});
    Prepare(inverse, FLOAT32, {rows, 1});
    Prepare(pre, FLOAT32, {x.dims[0], x.dims[1], h});
    Prepare(post, FLOAT32, pre.dims);
    Prepare(comb, FLOAT32, {x.dims[0], x.dims[1], h, h});
    Task t;
    t.op = V41ReferenceOp::Linear;
    t.rows = rows;
    t.cols = t.width = h * x.dims[3];
    t.out = m;
    t.work = rows * m;
    t.p[Input] = (float *)input.cudaData;
    t.p[Weight] = (float *)fn.cudaData;
    t.p[Output] = (float *)mix.cudaData;
    Launch(t);
    t.op = V41ReferenceOp::HCMixes;
    t.h = h;
    t.iterations = it;
    t.hcEps = eps;
    t.eps = normEps;
    t.work = rows;
    t.p[Mix] = (float *)mix.cudaData;
    t.p[Scale] = (float *)scale.cudaData;
    t.p[Base] = (float *)base.cudaData;
    t.p[Output] = (float *)pre.cudaData;
    t.p[Output1] = (float *)post.cudaData;
    t.p[Output2] = (float *)comb.cudaData;
    t.p[Output3] = (float *)normalized.cudaData;
    t.p[Output4] = (float *)inverse.cudaData;
    Launch(t);
    return Finish();
}
extern "C" bool FastllmCudaV41ReferenceLinear(const Data &x, Data &w, Data &o, int groups, bool round) {
    if (groups <= 0 || !Valid(x) || w.multiDeviceData || w.dims.size() != 2 || w.dims[0] % groups != 0 ||
        (w.dataType != FLOAT32 && w.dataType != BFLOAT16) || x.dims.empty() ||
        x.dims.back() != w.dims[1] * groups)
        return false;
    w.ToDevice(DataDevice::CUDA);
    Data temp, raw;
    const Data &input = FloatInput(x, temp);
    auto dims = x.dims;
    dims.back() = w.dims[0];
    Prepare(raw, FLOAT32, dims);
    Task t;
    t.op = V41ReferenceOp::Linear;
    t.rows = LogicalCount(x) / x.dims.back();
    t.cols = x.dims.back();
    t.width = w.dims[1];
    t.out = w.dims[0];
    t.groups = groups;
    t.work = t.rows * t.out;
    t.dtype = w.dataType == BFLOAT16 ? 1 : 0;
    t.mode = round;
    t.p[Input] = (float *)input.cudaData;
    t.p[Weight] = (float *)w.cudaData;
    t.p[Output] = (float *)raw.cudaData;
    Launch(t);
    if (round)
        Store(raw, o, dims);
    else
        o.CopyFrom(raw);
    return Finish();
}
extern "C" bool FastllmCudaV41ReferenceAttention(const Data &q, const Data &chunk, const Data *ring,
                                                 const Data *compressed, const Data *idx, Data &sink,
                                                 int window, int start, float factor, Data &o) {
    if (!Valid(q) || q.dataType != BFLOAT16 || q.dims.size() != 4 || q.dims[0] != 1 || !Valid(chunk) ||
        q.dims[1] <= 0 || q.dims[2] <= 0 || q.dims[3] != 512 || window <= 0 || start < 0 ||
        chunk.dims != std::vector<int>({1, q.dims[1], 512}) ||
        (ring && (!Valid(*ring) || ring->dims != std::vector<int>({1, window, 512}))) ||
        (compressed && (!Valid(*compressed) || compressed->dims.size() != 3 || compressed->dims[0] != 1 ||
                        compressed->dims[2] != 512)) ||
        (idx &&
         (!compressed || idx->dataDevice != DataDevice::CUDA || !idx->cudaData || idx->dataType != INT32 ||
          idx->dims.size() != 3 || idx->dims[0] != 1 || idx->dims[1] != q.dims[1])))
        return false;
    sink.ToDevice(DataDevice::CUDA);
    Data qt, kt, ct, rt, keys, indices, raw;
    const Data &query = FloatInput(q, qt);
    const Data &kv = FloatInput(chunk, kt);
    int rows = q.dims[1], dim = q.dims[3], heads = q.dims[2];
    int slots = start == 0 ? std::min(rows, window) : window, offset = start == 0 ? rows : window + rows,
        width = idx && !idx->dims.empty() ? idx->dims.back() : 0;
    Prepare(keys, FLOAT32, {1, offset, dim});
    if (start) {
        if (!ring)
            return false;
        const Data &r = FloatInput(*ring, rt);
        cudaMemcpyAsync(keys.cudaData, r.cudaData, size_t(window) * dim * 4, cudaMemcpyDeviceToDevice);
    }
    cudaMemcpyAsync((float *)keys.cudaData + (start ? window * dim : 0), kv.cudaData, size_t(rows) * dim * 4,
                    cudaMemcpyDeviceToDevice);
    Prepare(indices, INT32, {1, rows, slots + width});
    MakeIndices<<<(rows * (slots + width) + 255) / 256, 256>>>((int *)indices.cudaData,
                                                               width ? (const int *)idx->cudaData : nullptr,
                                                               rows, window, start, slots, width, offset);
    Prepare(raw, FLOAT32, q.dims);
    Task t;
    t.op = V41ReferenceOp::SparseAttention;
    t.work = rows * heads;
    t.heads = heads;
    t.dim = dim;
    t.width = slots + width;
    t.offset = offset;
    t.factor = factor;
    t.p[Input] = (float *)query.cudaData;
    t.p[Keys] = (float *)keys.cudaData;
    t.p[Indices] = (float *)indices.cudaData;
    t.p[Sink] = (float *)sink.cudaData;
    t.p[Output] = (float *)raw.cudaData;
    if (width && compressed) {
        const Data &c = FloatInput(*compressed, ct);
        t.p[Values] = (float *)c.cudaData;
        t.visible = compressed->dims[1];
    }
    Launch(t);
    Store(raw, o, q.dims);
    return Finish();
}

namespace {
    struct EngramDot {
        const float *h, *k, *qw, *kw;
        __device__ float operator()(size_t j) const { return (h[j] * (qw[j] * kw[j])) * k[j]; }
    };
    __global__ void ReferenceEngram(float *hidden, const float *kv, const float *qw, const float *kw,
                                    const float *mask, int rows, int hc, int dim, float eps, float clamp) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= rows * hc)
            return;
        int token = i / hc, stream = i % hc;
        if (mask && mask[token] == 0)
            return;
        float *h = hidden + size_t(i) * dim;
        const float *k = kv + (size_t(token) * (hc + 1) + stream) * dim;
        const float *v = kv + (size_t(token) * (hc + 1) + hc) * dim;
        float h2 = Sum(dim, Load{h, true}), k2 = Sum(dim, Load{k, true});
        float dot = Sum(dim, EngramDot{h, k, qw + stream * dim, kw + stream * dim});
        float inv = (1.0f / sqrtf(h2 / dim + eps)) * (1.0f / sqrtf(k2 / dim + eps));
        float score = dot * inv * (1.0f / sqrtf(float(dim)));
        float root = copysignf(sqrtf(Max(fabsf(score), clamp)), score);
        float exponent = i < rows * hc / 32 * 32 ? ReductionExp(-root) : Exp(-root);
        float gate = 1.0f / (1.0f + exponent);
        for (int j = 0; j < dim; ++j)
            h[j] = BF(h[j] + gate * v[j]);
    }
} // namespace
extern "C" bool FastllmCudaV41ReferenceEngram(Data &hidden, const Data &kv, Data &qw, Data &kw,
                                              const Data *mask, float eps, float clamp) {
    if (!Valid(hidden) || hidden.dataType != BFLOAT16 || hidden.dims.size() != 4 || !Valid(kv) ||
        qw.dataType != FLOAT32 || kw.dataType != FLOAT32 ||
        (mask && mask->Count(0) > 0 && (mask->dataType != FLOAT32 || mask->dataDevice != DataDevice::CUDA)))
        return false;
    qw.ToDevice(DataDevice::CUDA);
    kw.ToDevice(DataDevice::CUDA);
    Data temp, kt;
    const Data &h = FloatInput(hidden, temp);
    const Data &k = FloatInput(kv, kt);
    int rows = hidden.dims[0] * hidden.dims[1], hc = hidden.dims[2], dim = hidden.dims[3];
    ReferenceEngram<<<(rows * hc + 127) / 128, 128>>>(
        (float *)h.cudaData, (float *)k.cudaData, (float *)qw.cudaData, (float *)kw.cudaData,
        mask && mask->Count(0) ? (float *)mask->cudaData : nullptr, rows, hc, dim, eps, clamp);
    Store(h, hidden, hidden.dims);
    return Finish();
}

namespace {
    __global__ void ReferenceQuantRows(Task t, int blocksPerRow) {
        size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= t.work)
            return;
        size_t row = i / blocksPerRow;
        t.p[Input] += row * t.dim;
        t.p[Output] += row * t.dim;
        Execute(t, i % blocksPerRow);
    }
} // namespace
extern "C" bool FastllmCudaV41ReferenceRotary(Data &x, int ropeDim, float theta, int start, int stride,
                                              bool inverse, int original, float factor, int betaFast,
                                              int betaSlow, int mode, int quantDim, int block) {
    if (!Valid(x) || x.dataType != BFLOAT16 || (x.dims.size() != 3 && x.dims.size() != 4))
        return false;
    Data temp;
    const Data &raw = FloatInput(x, temp);
    int dim = x.dims.back(), heads = x.dims.size() == 4 ? x.dims[2] : 1;
    Task t;
    t.op = V41ReferenceOp::Rotary;
    t.dim = dim;
    t.cols = dim * heads;
    t.ropeDim = ropeDim;
    t.theta = theta;
    t.start = start;
    t.stride = stride;
    t.inverse = inverse;
    t.original = original;
    t.factor = factor;
    t.mode = original > 0;
    auto correction = [&](int rotations) {
        return ropeDim * std::log(double(original) / (rotations * 2.0 * 3.14159265358979323846)) /
               (2.0 * std::log(double(theta)));
    };
    if (original > 0) {
        t.low = std::max(int(std::floor(correction(betaFast))), 0);
        t.high = std::min(int(std::ceil(correction(betaSlow))), ropeDim - 1);
    }
    t.p[Input] = t.p[Output] = (float *)raw.cudaData;
    t.work = LogicalCount(x) / dim * (ropeDim / 2);
    Launch(t);
    if (mode > 0) {
        t.op = V41ReferenceOp::Quantize;
        t.mode = mode == 1 ? 0 : mode == 3 ? 1 : 2;
        t.block = block;
        int blocks = quantDim / block;
        t.work = LogicalCount(x) / dim * blocks;
        ReferenceQuantRows<<<(t.work + 127) / 128, 128>>>(t, blocks);
    }
    Store(raw, x, x.dims);
    return Finish();
}
