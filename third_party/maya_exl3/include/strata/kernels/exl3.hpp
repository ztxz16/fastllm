// include/strata/kernels/exl3.hpp - EXL3 (exllamav3) quantized weights on the device: the decode GEMV, the routed
// experts, and the full reconstruction for the prompt path.
//
// The format (exllamav3, MIT, turboderp).  A weight with k inputs and n outputs (y = x W) is stored as
//   trellis  int16 [k/16][n/16][16 K]: each 16x16 tile is a tail-biting trellis of 256 K-bit steps read MSB-first
//            from its 8 K uint32 words (little-endian pairs of the int16s).  Step p's state is the 16-bit window that
//            ends at bit S(p) = (p + 1) K, modulo the tile's 256 K bits.  exllamav3 1.6 also writes the half-integer
//            K = KA + 0.5 (1.5 / 2.5 / 3.5: 24 / 40 / 56 int16 a tile, mul1 only): its steps alternate KA bits (even p)
//            and KA + 1 (odd p), so S(p) = floor((p + 1) k2 / 2) with k2 = 2 K - which is (p + 1) K for an integer K
//            too.  The engine carries every bitrate as k2 (2..16 even, or 3 / 5 / 7).  The codebook turns the state
//            into a value:
//              mul1 (cb 2):  fp16(1024 + bytesum(state * 0x83DCD12D)) * fp16(0x1eee) + fp16(0xc931), one FP16 FMA
//              mcg (cb 1):   x = state * 0xCBAC1FED, 3inst (cb 0): x = state * 89226354 + 64248484; then
//                            x = (x & 0x8fff8fff) ^ 0x3b603b60 and the value is the FP16 sum of x's two halves.
//            Step p sits at tile row (p/8 % 4) * 2 + (p & 1) + 8 (p/2 & 1), column p/32 + 8 (p/4 & 1) (the tensor-
//            core fragment order).
//   suh, svh FP16 [k], [n]: the input and output channel scales (signs and magnitudes).
// With H the 128-point Sylvester Hadamard over sqrt(128), applied to each 128 consecutive values:
//   y = H(H(x * suh) W') * svh,   W' the decoded k x n matrix.
//
// Activations enter as F32 or as the engine's q8_1 blocks; the kernels rotate them on the fly (each block rotates
// the slice it reads), so callers keep their existing buffers.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace strata::kernels::exl3 {

/// The weight type the engine's MvJob / WSlot carry for EXL3: their weight pointer is a (host) const Mat*.
constexpr int kTypeEXL3 = 200;

/// One EXL3 matrix, or a column range of one (a view: trellis/svh advanced to its first column, ld the row stride).
struct Mat {
    const uint32_t* trellis = nullptr;   // device; 4-byte aligned
    const uint16_t* suh = nullptr;       // device FP16 bits [k]
    const uint16_t* svh = nullptr;       // device FP16 bits [n]
    int k = 0, n = 0;
    int k2 = 0;                          // twice the bits per weight: 2..16 even (K 1..8), 3 / 5 / 7 (K 1.5 / 2.5 / 3.5)
    int cb = 2;                          // codebook: 0 3inst, 1 mcg, 2 mul1 (the only one with a half-integer K)
    int ld = 0;                          // tiles per 16-row slice in memory (the full matrix's n / 16); 0: n / 16
    // a GGUF's plain matrix instead (DeepSeek-V4.1's maya-ds4: fmt 1 BF16): raw = [n][ld] (row r = output r, ld
    // values a row; 0: k), the EXL3 fields unused - the engine sends it to its own kernels, never to these
    int fmt = 0;
    const void* raw = nullptr;
    int gg_type = 0;   // fmt 4 (maya-ds4's kFmtGG): the ggml block type
};

/// Trellis bytes of a k x n matrix at k2 / 2 bits: 16 k2 bytes a 16x16 tile.
inline size_t trellis_bytes_k2(int64_t k, int64_t n, int k2) {
    return (size_t) (k / 16) * (size_t) (n / 16) * 16u * (size_t) k2;
}
/// The columns [c0, c0 + nc) of m (c0 and nc multiples of 128: the output rotation's blocks stay whole).
Mat view(const Mat& m, int c0, int nc);
/// Whether the kernels take this bitrate and codebook: K 1..8 (k2 2..16 even) with any codebook, K 1.5 / 2.5 / 3.5
/// (k2 3 / 5 / 7) with mul1.
bool k2_supported(int k2, int cb);
/// Whether the kernels take this matrix (k, n multiples of 128, k2_supported).
bool supported(const Mat& m);

/// y[r] = alpha * (x W)[r] + (bias ? bias[r] : 0).  x: xf (F32) when given, else xq (q8_1 blocks of k values).
/// ncol > 1: as many activations (a verify window's tokens), activation c at xf + c xf_cs / xq + c xq_cs bytes, its
/// y at y + c y_cs - the weights read and decoded once for up to 8 of them, and each y bit for bit its own job's.
struct MvJob {
    const Mat* m = nullptr;
    const float* xf = nullptr;
    const void* xq = nullptr;
    float* y = nullptr;
    const float* bias = nullptr;
    float alpha = 1.0f;
    int ncol = 1;
    size_t xf_cs = 0, xq_cs = 0, y_cs = 0;
    // Fastllm extension: native 16-bit buffers, converted in the existing load/epilogue.
    // x16 overrides xf/xq, y16 overrides y. xf_cs/y_cs are element strides for either type.
    // Null keeps the upstream F32/q8_1 path. Accumulation and split scratch remain F32.
    const uint16_t* x16 = nullptr;
    uint16_t* y16 = nullptr;
    bool x_bf16 = false, y_bf16 = false;
};
constexpr int kMaxMvJobs = 8;
constexpr int kMaxMvCols = 16;   // a job's activations at most
bool mv(const MvJob* jobs, int n, cudaStream_t s);
/// The calling thread's launches of mv(), moe_gate_up() and moe_down() read `flag` (a device int) first and do no work
/// while it is non-zero (glmf::set_skip's, which sets it: a pipelined window's tokens past a wrong draft); nullptr (the
/// default): no check.  The flag changes only between launches (skip_gate, on the same stream).  (v1.1.8's kernels
/// before sm_80 do not read it.)
void set_skip(const int* flag);
/// How many pieces mv cuts m's inputs into on the current device (0: not covered) - one piece, and every column's
/// sum is the same whatever columns the job holds (a view's, or the whole matrix's).
int mv_splits(const Mat& m);
/// How many activations a lone job of m can take with its weights read once on the current device (the input split's
/// partials of them in the scratch; 0: not covered) - a caller with more cuts them in runs of this many.
int mv_cols(const Mat& m);

/// Allocates the per-stream scratch of mv / moe / win_experts on the current device (call once per stream before a
/// graph capture; the first use does it otherwise).
bool prepare(cudaStream_t s);

/// One routed expert's blob: three parts (gate, up, down) at byte offsets from the blob base, each holding the FP16
/// suh, the FP16 svh and the trellis of that matrix.
struct ExpertLayout {
    size_t suh[3] = {0, 0, 0}, svh[3] = {0, 0, 0}, trellis[3] = {0, 0, 0};
    int k2[3] = {0, 0, 0};   // each part's bitrate as twice its K (Mat::k2); gate and up alike
    int cb = 2;
    int fmt = 0;   // 2: a GGUF's MXFP4 expert (maya-ds4's own kernels): trellis[r] = matrix r's offset, [out][in / 32]
    int gg_type[3] = {0, 0, 0};   // fmt 2: each role's ggml type (0: MXFP4; maya-ds4's expert_type)
};
/// The routed experts of one token (the engine's device plan: plan_ptr[i] = the blob in VRAM, 0 = not on the device;
/// plan_w[i] its weight).  gate/up over x (n_embd F32) -> clamped swiglu -> h (F32 [k][n_ff], scratch the caller
/// owns), then down over h, the weighted sum in plan order, + cpu_part when *cpu_flag, + sh_out when given -> out.
/// A mixed-precision quant's layer (the experts of one layer at different K, gate / up / down of one expert
/// too): its experts in classes, class c's blobs in VRAM within [lo[c], hi[c]) with the layout lay[c].  n == 0: one
/// layout for the layer (the args' lay), as before.  The kernels pick each expert's class by its blob's address; a
/// class's K: 1-4 (K2 2, 4, 6, 8), mul1 (cb 2).
constexpr int kMaxClasses = 32;
struct MixedLayouts {
    int n = 0;
    unsigned long long lo[kMaxClasses] = {}, hi[kMaxClasses] = {};
    ExpertLayout lay[kMaxClasses];
};
bool mixed_supported(const ExpertLayout& lay);   // a class layout the mixed kernels take

struct MoeArgs {
    const unsigned long long* plan_ptr = nullptr;
    const float* plan_w = nullptr;
    const float* cpu_part = nullptr;
    const int* cpu_flag = nullptr;
    ExpertLayout lay;
    MixedLayouts mix;     // (n > 0: the experts' layouts by class - lay is not used)
    int k = 8, n_embd = 4096, n_ff = 2048;
    // Fastllm extension: plan_ptr points to three device base pointers (gate/up/down).
    // This avoids duplicating independently allocated expert weights.
    bool indirect = false;
    // Preserve BF16 boundaries of Fastllm's gate, up, activation and down tensors.
    bool round_bf16 = false;
    // Fastllm: write weighted per-route outputs for the shared hybrid reducer.
    bool route_output = false;
    // Fastllm: isolated scratch for parallel cache graph branches (0 is upstream).
    int scratch_lane = 0;
    float limit = 0.0f;   // 0: no clamp
    // Fastllm extension: optional native gate/up input (overrides the x argument).
    const uint16_t* x16 = nullptr;
    bool x_bf16 = false;
};
void moe_gate_up(const MoeArgs& a, const float* x, float* h, cudaStream_t s);
void moe_down(const MoeArgs& a, const float* h, const float* sh_out, float* out, cudaStream_t s);

/// A verify window's routed experts, each distinct expert read once for all the tokens it serves (after exllamav3's
/// fused MoE, which runs the slots that picked one expert as the rows of one tile): T tokens' plans [T][8] grouped by
/// expert (glmf::moe_win_group: group g's blob at grp_ptr[g], its entries e = t * 8 + i at grp_ent[grp_start[g] ..
/// grp_start[g + 1])), token t's F32 input at x + t x_stride, its output at out + t out_stride (plus the CPU lane's
/// part when cpu_flag[t], plus sh_out's row t) - every token's output is moe_gate_up + moe_down's for it alone, bit for
/// bit.  h [T * 8][n_ff] and part [T * 8][n_embd] are the caller's scratch.
struct WinArgs {
    const unsigned long long* grp_ptr = nullptr;
    const int *grp_start = nullptr, *grp_ent = nullptr, *n_groups = nullptr;
    const unsigned long long* plan_ptr = nullptr;   // [T][8]
    const float* plan_w = nullptr;                  // [T][8]
    const float* cpu_part = nullptr;                // [T][n_embd] (rows cpu_stride apart when given)
    size_t cpu_stride = 0;
    const int* cpu_flag = nullptr;                  // [T]
    const float* x = nullptr;
    size_t x_stride = 0;
    float* h = nullptr;
    float* part = nullptr;
    const float* sh_out = nullptr;                  // [T][n_embd] or null
    float* out = nullptr;
    size_t out_stride = 0;
    ExpertLayout lay;
    MixedLayouts mix;     // (n > 0: the experts' layouts by class - lay is not used)
    // Fastllm: native activation and BF16 boundaries, matching MoeArgs.
    const uint16_t* x16 = nullptr;
    bool x_bf16 = false, indirect = false, round_bf16 = false;
    // Weighted per-route output uses compact [T][k][n_embd] rows. Missing
    // routes can be preserved when separate launches fill disjoint routes.
    bool route_output = false, preserve_missing = false;
    int scratch_lane = 0;
    int T = 0, k = 8, n_embd = 4096, n_ff = 2048;
    float limit = 0.0f;
    // (given) the combine first waits on the device until *wait_ptr == wait_val (the host's answer to the CPU lane's
    // part arrives while the device computes the experts); cpu_flag may be null (no CPU part)
    const unsigned* wait_ptr = nullptr;
    unsigned wait_val = 0;
    // Fastllm: optional caller-owned scratch for whole-batch expert grouping.
    // Defaults retain the upstream fixed scratch and call semantics.
    float* up = nullptr;
    unsigned* group_counters = nullptr;
    int group_capacity = 0;
};
bool win_experts(const WinArgs& a, cudaStream_t s);
/// Whether the current device runs the several-activation kernels (win_experts; mv's ncol > 1): every device does
/// (they take ~10 KB of shared memory a block).
bool several_ok();

/// The prompt path's tensor-core GEMMs (sm_80 and later; mma_supported() says whether this device runs them).
bool mma_supported();
/// CUDA prompt-planning query: sm_80+ MMA or sm_7x WMMA; deliberately false on HIP.
/// This is separate from the HIP grouped-kernel dispatch (prompt_wmma), so querying it does not change planning.
bool prompt_tc();
/// Whether grouped expert kernels select WMMA on the current device, including HIP gfx11/gfx12.
/// On HIP, the imported default enables the matrix path on gfx11 only.
bool prompt_wmma();
/// A set of n_exp routed experts in VRAM, expert e's blob at wbase + e * stride (the layout in lay), owning the sorted
/// rows [bounds[e], bounds[e + 1]) (device, n_exp + 1 entries); a sorted row's LOCAL row is bounds[e] - row0 + r.
struct SetArgs {
    const uint8_t* wbase = nullptr;
    size_t stride = 0;
    ExpertLayout lay;
    MixedLayouts mix;     // (n > 0: expert e's blob at eptr[e] (device), its layout by class - wbase, stride, lay unused)
    const unsigned long long* eptr = nullptr;
    int n_exp = 0;
    const int* bounds = nullptr;
    int row0 = 0;
    int max_rows = 0;   // the most rows one expert of the set has
    int n_embd = 4096, n_ff = 2048;
    float limit = 0.0f;
    int skip_ge = 0;    // (ds4's low-bit GGUF experts) > 0: experts of at least this many rows are the caller's (FP16 GEMMs)
};
/// h[local row] = FP16 clamped swiglu of gate and up over x[row_idx[local row]] (F32 rows, ldx floats apart).
void set_gate_up(const SetArgs& a, const float* x, int ldx, const int* row_idx, uint16_t* h16, int ldh, cudaStream_t s);
/// y[local row] = down(h[local row]) (F32 rows, ldy floats apart).
void set_down(const SetArgs& a, const uint16_t* h16, int ldh, float* y, int ldy, cudaStream_t s);
/// Y[r] = beta Y[r] + (X[r] W) for r < rows: one matrix over FP16 rows (ldx), F32 output rows (ldy).
void gemm_rows(const Mat& m, const uint16_t* x16, int ldx, int rows, float* y, int ldy, float beta, cudaStream_t s);

/// The full weight in the engine's row layout: dst[r][c] = W[c][r] (n rows of k values), BF16 or F16 bits.
void reconstruct_rows(const Mat& m, uint16_t* dst, bool bf16, cudaStream_t s);
/// The decoded W' alone in that layout: dst[c][r] = W'[r][c], F16 bits (exactly the decoded values) - for a product
/// that rotates its activations and outputs instead of the weight (x W = H(H(x * suh) W') * svh): rows_rot_f16 before
/// the GEMM, rows_rot_out after it.  No Hadamard over the k x n weight: a plain decode, at the memory's speed.
void reconstruct_inner_rows(const Mat& m, uint16_t* dst, cudaStream_t s);
/// out[r][c] = FP16(H(x_r * suh)[c]) for r < nrows, c < ncols (ncols % 128 == 0), x_r = x[idx ? idx[r] : r] (F32, ldx
/// floats a row): an EXL3 matrix's rotated input rows, as the decode kernels rotate theirs.
void rows_rot_f16(const float* x, int ldx, const int* idx, int nrows, int ncols, const uint16_t* suh, uint16_t* out,
                  cudaStream_t s);
/// y[r][c] = H(y_r)[c] * svh[c] in place for r < nrows, c < ncols (each 128 values one block; ldy floats a row).
void rows_rot_out(float* y, int ldy, int nrows, int ncols, const uint16_t* svh, cudaStream_t s);
/// The decoded W' alone, row-major [k][n] F16 bits (tests).
void reconstruct_inner(const Mat& m, uint16_t* dst, cudaStream_t s);

/// out[r][c] = FP16(x[idx ? idx[r] : r][c]) for r < nrows, c < ncols (x: ldx floats a row): the prompt path's rows of
/// one expert, gathered for its GEMMs.
void rows_f16(const float* x, int ldx, const int* idx, int nrows, int ncols, uint16_t* out, cudaStream_t s);

/// Launch failures of this family since the process started (each is also printed).
int launch_errors();

}  // namespace strata::kernels::exl3
