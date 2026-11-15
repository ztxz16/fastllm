#include "quantized_kernels.h"
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace fastllm::minimax_h3_image::detail {
namespace {
using namespace cute;
namespace epi = cutlass::epilogue::threadblock;
using BF16 = cutlass::bfloat16_t;
constexpr auto roundNearest = cutlass::FloatRoundStyle::round_to_nearest;

// Per-row activation and per-column weight scaling inside the epilogue avoids
// writing/reading an INT32 matrix. Multiplication order matches the unfused
// path. The raw variant retains INT32 for TP reduction before scaling.
template <class Tile, class Warp, int Stages, class Output>
void run(void *out, const void *x, const void *w, const float *xs, const float *ws, int m, int n, int k,
         cudaStream_t stream) {
    using Map = epi::OutputTileThreadLayout<Tile, Warp, Output, 16 / sizeof(Output)>;
    using XScale = epi::VisitorColBroadcast<Map, float, Stride<_1, _0, _0>, false>;
    using WScale = epi::VisitorRowBroadcast<Map, float, Stride<_0, _1, _0>, false>;
    using Mul = epi::VisitorCompute<cutlass::multiplies, float, float, roundNearest>;
    using First = epi::Sm80EVT<Mul, epi::VisitorAccFetch, XScale>;
    using Product = epi::Sm80EVT<Mul, First, WScale>;
    using Store = epi::VisitorAuxStore<Map, Output, roundNearest, Stride<int64_t, _1, _0>>;
    using Value = std::conditional_t<std::is_same_v<Output,int32_t>, epi::VisitorAccFetch, Product>;
    using Visitor = epi::Sm80EVT<Store, Value>;
    using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        int8_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 16, int8_t,
        cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 16, Output, cutlass::layout::RowMajor,
        16 / sizeof(Output), int32_t, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, Tile, Warp,
        cutlass::gemm::GemmShape<16, 8, 32>, Visitor,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, Stages,
        cutlass::arch::OpMultiplyAddSaturate>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    typename XScale::Arguments xScale{xs};
    typename WScale::Arguments wScale{ws};
    typename First::Arguments first{{}, xScale, {}};
    typename Product::Arguments product{first, wScale, {}};
    typename Visitor::Arguments epilogue = [&] {
        typename Store::Arguments store{static_cast<Output *>(out), {int64_t(n), _1{}, _0{}}};
        if constexpr (std::is_same_v<Output,int32_t>) return typename Visitor::Arguments{{},store};
        else return typename Visitor::Arguments{product,store};
    }();
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                  {m, n, k},
                                  1,
                                  epilogue,
                                  x,
                                  w,
                                  nullptr,
                                  out,
                                  int64_t(m) * k,
                                  int64_t(n) * k,
                                  0,
                                  int64_t(m) * n,
                                  int64_t(k),
                                  int64_t(k),
                                  int64_t(n),
                                  int64_t(n)};
    auto check = [](cutlass::Status status) {
        if (status != cutlass::Status::kSuccess)
            throw std::runtime_error(std::string("image INT8 GEMM: ") +
                                     cutlass::cutlassGetStatusString(status));
    };
    Gemm gemm;
    check(gemm.can_implement(args));
    // This non-split GEMM requires no global scratch workspace.
    if (gemm.get_workspace_size(args) != 0)
        throw std::runtime_error("unexpected INT8 GEMM workspace requirement");
    check(gemm(args, nullptr, stream));
}
} // namespace
template<class Output>
void dispatch(void *out, const void *x, const void *w, const float *xs, const float *ws, int m, int n,
                    int k, cudaStream_t stream) {
    using cutlass::gemm::GemmShape;
    if (m <= 32)
        run<GemmShape<32, 64, 128>, GemmShape<16, 64, 64>, 5, Output>(out, x, w, xs, ws, m, n, k, stream);
    else if (m <= 128)
        run<GemmShape<64, 64, 128>, GemmShape<32, 64, 64>, 5, Output>(out, x, w, xs, ws, m, n, k, stream);
    else
        run<GemmShape<128, 128, 64>, GemmShape<64, 64, 64>, 5, Output>(out, x, w, xs, ws, m, n, k, stream);
}
void int8RawGemm(void *out,const void *x,const void *w,int m,int n,int k,cudaStream_t stream) {
    dispatch<int32_t>(out,x,w,nullptr,nullptr,m,n,k,stream);
}
void int8ScaledGemm(void *out,const void *x,const void *w,const float *xs,const float *ws,int m,int n,
                    int k,cudaStream_t stream,bool fp32Output) {
    if(fp32Output) dispatch<float>(out,x,w,xs,ws,m,n,k,stream);
    else dispatch<BF16>(out,x,w,xs,ws,m,n,k,stream);
}
} // namespace fastllm::minimax_h3_image::detail
