#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "fp8_scaled.h"
#include <stdexcept>
#include <string>

namespace fastllm::minimax_h3_image::detail {
namespace {
using namespace cute;
using namespace cutlass::epilogue::fusion;
using F8 = cutlass::float_e4m3_t;
using BF = cutlass::bfloat16_t;
template <int M, int N, int K, class O = BF> struct Config {
    using ElementOutput = O;
    using Tile = Shape<Int<M>, Int<N>, Int<K>>;
    using Cluster = Shape<_1, _1, _1>;
    using WScale = Sm90RowBroadcast<0, Tile, float, float, Stride<_0, _1, _0>>;
    using XScale = Sm90ColBroadcast<0, Tile, float, float, Stride<_1, _0, _0>>;
    using ScaledAccumulator =
        Sm90EVT<Sm90Compute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>,
                WScale, Sm90AccFetch>;
    using Output =
        Sm90EVT<Sm90Compute<cutlass::multiplies, O, float, cutlass::FloatRoundStyle::round_to_nearest>,
                XScale, ScaledAccumulator>;
    using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, Tile, Cluster,
        cutlass::epilogue::collective::EpilogueTileAuto, float, float, void, cutlass::layout::RowMajor, 8, O,
        cutlass::layout::RowMajor, 16 / sizeof(O), cutlass::epilogue::collective::EpilogueScheduleAuto,
        Output>::CollectiveOp;
    using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, F8, cutlass::layout::RowMajor, 16, F8,
        cutlass::layout::ColumnMajor, 16, float, Tile, Cluster,
        cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epilogue::SharedStorage)>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};
void check(cutlass::Status status) {
    if (status != cutlass::Status::kSuccess)
        throw std::runtime_error(std::string("image FP8 scaled GEMM: ") + cutlassGetStatusString(status));
}

template <class C>
void launch(const void *x, const void *w, const float *xs, const float *ws, void *y, int m, int n, int k,
            int device, int multiprocessors, void *workspace, size_t workspaceBytes, cudaStream_t stream) {
    using Kernel = typename C::Kernel;
    using Gemm = typename C::Gemm;
    auto sa = cutlass::make_cute_packed_stride(typename Kernel::StrideA{}, make_shape(m, k, 1));
    auto sb = cutlass::make_cute_packed_stride(typename Kernel::StrideB{}, make_shape(n, k, 1));
    auto outputStride = cutlass::make_cute_packed_stride(typename Kernel::StrideD{}, make_shape(m, n, 1));
    typename C::Output::Arguments scale{{xs}, {{ws}, {}, {}}, {}};
    cutlass::KernelHardwareInfo hardware;
    hardware.device_id = device;
    hardware.sm_count = multiprocessors;
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                  {m, n, k, 1},
                                  {static_cast<const F8 *>(x), sa, static_cast<const F8 *>(w), sb},
                                  {scale, nullptr, outputStride, static_cast<typename C::ElementOutput *>(y), outputStride},
                                  hardware};
    check(Gemm::can_implement(args));
    if (Gemm::get_workspace_size(args) > workspaceBytes)
        throw std::runtime_error("image FP8 scaled GEMM workspace overflow");
    Gemm gemm;
    check(gemm.initialize(args, workspace, stream));
    check(gemm.run(stream));
}
} // namespace

template<class Output>
void dispatch(const void *x, const void *w, const float *xs, const float *ws, void *y, int m, int n,
                    int k, int device, int multiprocessors, void *workspace, size_t workspaceBytes,
                    cudaStream_t stream) {
    // H3's wide QKV/FC1 projections benefit from a wider N tile. Narrow
    // projections use a shorter K tile where it improves SM120 throughput.
    // Small shapes retain the general kernel, including low-memory requests.
    if (m >= 1024 && n >= 16384)
        launch<Config<128, 256, 64, Output>>(x, w, xs, ws, y, m, n, k, device, multiprocessors, workspace,
                                     workspaceBytes, stream);
    else if (m >= 1024 && n <= 8192 && (m <= 3072 || k <= 8192))
        launch<Config<128, 128, 64, Output>>(x, w, xs, ws, y, m, n, k, device, multiprocessors, workspace,
                                     workspaceBytes, stream);
    else
        launch<Config<128, 128, 128, Output>>(x, w, xs, ws, y, m, n, k, device, multiprocessors, workspace,
                                      workspaceBytes, stream);
}
void fp8ScaledSm120(const void *x,const void *w,const float *xs,const float *ws,void *y,int m,int n,
                    int k,int device,int multiprocessors,void *workspace,size_t workspaceBytes,
                    cudaStream_t stream,bool fp32Output) {
    if(fp32Output) dispatch<float>(x,w,xs,ws,y,m,n,k,device,multiprocessors,workspace,workspaceBytes,stream);
    else dispatch<BF>(x,w,xs,ws,y,m,n,k,device,multiprocessors,workspace,workspaceBytes,stream);
}
} // namespace fastllm::minimax_h3_image::detail
