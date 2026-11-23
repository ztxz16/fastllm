#pragma once

// Modified for FastLLM fused draft execution. See the parent vendor NOTICE
// and UPSTREAM.json for adaptation details and original source hashes.

// One 32-output tile per CTA. Both N8 groups reuse the weight fragments;
// each query fragment serves two M tiles. Fold eight warp partials.
__device__ static void mlp_down_tile(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                             float* partials, float* output) {
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int g = lane >> 2, q = lane & 3;
  if (warp < 8) {
    constexpr int kTiles = kColsPerCtaD4 / 16;
    float accum[Chunks][kTiles][4] = {};
#pragma unroll
    for (int slot = 0; slot < 2; ++slot) {
      const int k = slot * 256 + warp * 32 + q * 8;
#pragma unroll
      for (int tile = 0; tile < kTiles; ++tile) {
        const auto* base = weight + tile * PackedMlpRow::Rows * kMlpInterR;
        const PackedMlpRow w0{base + g * PackedMlpRow::Page};
        const PackedMlpRow w1{base + (g + 8) * PackedMlpRow::Page};
        const uint4 a0 = *reinterpret_cast<const uint4*>(w0 + k);
        const uint4 a1 = *reinterpret_cast<const uint4*>(w1 + k);
#pragma unroll
        for (int chunk = 0; chunk < Chunks; ++chunk) {
          const uint4 b = *reinterpret_cast<const uint4*>(x + (chunk * 8 + g) * kMlpDownStride + k);
          naivert::attn_rows::mma_bf16_16816(accum[chunk][tile], a0.x,a1.x,a0.y,a1.y,b.x,b.y);
          naivert::attn_rows::mma_bf16_16816(accum[chunk][tile], a0.z,a1.z,a0.w,a1.w,b.z,b.w);
        }
      }
    }
#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
      float* mine = partials + (warp * kColsPerCtaD4 + tile * 16) * kMlpWindow;
#pragma unroll
      for (int chunk = 0; chunk < Chunks; ++chunk) {
        mine[g * kMlpWindow + chunk * 8 + 2*q] = accum[chunk][tile][0];
        mine[g * kMlpWindow + chunk * 8 + 2*q + 1] = accum[chunk][tile][1];
        mine[(g+8) * kMlpWindow + chunk * 8 + 2*q] = accum[chunk][tile][2];
        mine[(g+8) * kMlpWindow + chunk * 8 + 2*q + 1] = accum[chunk][tile][3];
      }
    }
  }
  __syncthreads();
  for (int item = tid; item < kMlpWindow * kColsPerCtaD4; item += kMlpThreads) {
    const int column = item / kMlpWindow, row = item % kMlpWindow;
    float sum = 0.0f;
#pragma unroll
    for (int w = 0; w < 8; ++w)
      sum += partials[(w * kColsPerCtaD4 + column) * kMlpWindow + row];
    output[row * kDownStageStride + column] = sum;
  }
  __syncthreads();
}

// Every GU CTA normalizes its own eight rows directly into shared memory.
// Each physical warp reproduces the eight original virtual-warp sums for
// one row. Keep per-thread FMA order and both XOR reduction trees unchanged.
__device__ static void mlp_normalize_local(
    const MlpArgs& args, const float* gamma, __nv_bfloat16* x,
    std::uint64_t* gamma_ready) {
  const int lane = threadIdx.x % 32, row = threadIdx.x / 32;
  uint4 staged[2][8];
  float sums[8] = {};
#pragma unroll
  for (int piece = 0; piece < 2; ++piece) {
#pragma unroll
    for (int virtual_warp = 0; virtual_warp < 8; ++virtual_warp) {
      const int k = (virtual_warp * 32 + lane) * 8 + piece * 2048;
      staged[piece][virtual_warp] = __ldg(reinterpret_cast<const uint4*>(
          args.hid_a + row * kMlpHidden + k));
    }
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
#pragma unroll
      for (int virtual_warp = 0; virtual_warp < 8; ++virtual_warp) {
        const auto* words = reinterpret_cast<const unsigned*>(&staged[piece][virtual_warp]);
        const float2 value = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(words + pair));
        sums[virtual_warp] = fmaf(value.x, value.x, sums[virtual_warp]);
        sums[virtual_warp] = fmaf(value.y, value.y, sums[virtual_warp]);
      }
    }
  }
#pragma unroll
  for (int virtual_warp = 0; virtual_warp < 8; ++virtual_warp) {
    sums[virtual_warp] = naivert::attn_rows::warp_sum_f32(sums[virtual_warp]);
    sums[virtual_warp] = __shfl_sync(0xffffffffu, sums[virtual_warp], 0);
  }
  float total = 0.0f;
#pragma unroll
  for (int virtual_warp = 0; virtual_warp < 8; ++virtual_warp)
    if (lane == virtual_warp) total = sums[virtual_warp];
  total = naivert::attn_rows::warp_sum_f32(total);
  float inv = lane == 0 ? rsqrtf(total / float(kMlpHidden) + args.rms_epsilon) : 0.0f;
  inv = __shfl_sync(0xffffffffu, inv, 0);
  unsigned ready = 0;
  if (lane == 0) ready = wait_weight(gamma_ready, 0u,
      naivert::MailboxWatchdog{args.watchdog_limit_ns, args.watchdog_flag});
  ready = __shfl_sync(0xffffffffu, ready, 0);
#pragma unroll
  for (int piece = 0; piece < 2; ++piece) {
#pragma unroll
    for (int virtual_warp = 0; virtual_warp < 8; ++virtual_warp) {
      const int k = (virtual_warp * 32 + lane) * 8 + piece * 2048;
      const int lo = naivert::attn_rows::PackedNormGamma::offset(k);
      const int hi = naivert::attn_rows::PackedNormGamma::offset(k + 4);
      const float4 g0 = ready ? *reinterpret_cast<const float4*>(gamma + lo) : make_float4(0,0,0,0);
      const float4 g1 = ready ? *reinterpret_cast<const float4*>(gamma + hi) : make_float4(0,0,0,0);
      const float scale[8] = {g0.x,g0.y,g0.z,g0.w,g1.x,g1.y,g1.z,g1.w};
      const auto* words = reinterpret_cast<const unsigned*>(&staged[piece][virtual_warp]);
      uint4 out;
      auto* output_words = reinterpret_cast<unsigned*>(&out);
#pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        const float2 value = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(words + pair));
        const __nv_bfloat162 packed = __floats2bfloat162_rn(
            float(__float2bfloat16_rn(value.x * inv)) * scale[2*pair], float(__float2bfloat16_rn(value.y * inv)) * scale[2*pair+1]);
        output_words[pair] = *reinterpret_cast<const unsigned*>(&packed);
      }
      *reinterpret_cast<uint4*>(x + mlp_input_index(row, k)) = out;
    }
  }
  asm volatile("bar.sync 1, 256;" ::: "memory");
}


// The producing CTA publishes through a device release; bypass stale L1
// copies when another CTA reuses this output buffer for normalized inputs.
__device__ static uint4 mlp_load_normalized(const __nv_bfloat16* p) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(p) : "memory");
  return v;
}

// The first eight warps normalize one row with the scalar FMA/reduction
// order. Helper warps remain free to prefetch weights while normalization runs.
__device__ static void mlp_normalize_row(const MlpArgs& args, int row,
    const float* gamma, std::uint64_t* ready, float* scratch) {
  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  uint4 staged[2];
  float sum = 0.f;
  if (tid < 256) {
#pragma unroll
    for (int piece = 0; piece < 2; ++piece) {
      staged[piece] = __ldg(reinterpret_cast<const uint4*>(args.hid_a + row * kMlpHidden + tid * 8 + piece * 2048));
      const unsigned* words = reinterpret_cast<const unsigned*>(&staged[piece]);
#pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        const float2 v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(words + pair));
        sum = fmaf(v.x,v.x,sum);sum = fmaf(v.y,v.y,sum);
      }
    }
    sum = naivert::attn_rows::warp_sum_f32(sum);
    if (lane == 0) scratch[warp] = sum;
  }
  asm volatile("bar.sync 1, 256;" ::: "memory");
  if (warp == 0) {
    float total = lane < 8 ? scratch[lane] : 0.f;
    total = naivert::attn_rows::warp_sum_f32(total);
    if (lane == 0) scratch[8] = rsqrtf(total / float(kMlpHidden) + args.rms_epsilon);
  }
  asm volatile("bar.sync 1, 256;" ::: "memory");
  if (tid < 256) {
    unsigned valid=0;
    if (lane == 0) valid=wait_weight(ready,0u,naivert::MailboxWatchdog{args.watchdog_limit_ns,args.watchdog_flag});
    valid=__shfl_sync(0xffffffffu,valid,0);
    const float inv=scratch[8];
#pragma unroll
    for (int piece = 0; piece < 2; ++piece) {
      const int k = tid * 8 + piece * 2048;
      const float4 g0 = valid ? *reinterpret_cast<const float4*>(gamma + naivert::attn_rows::PackedNormGamma::offset(k)) : make_float4(0,0,0,0);
      const float4 g1 = valid ? *reinterpret_cast<const float4*>(gamma + naivert::attn_rows::PackedNormGamma::offset(k+4)) : make_float4(0,0,0,0);
      const float scale[8]={g0.x,g0.y,g0.z,g0.w,g1.x,g1.y,g1.z,g1.w};
      uint4 result;
      const unsigned* in=reinterpret_cast<const unsigned*>(&staged[piece]);
      unsigned* out=reinterpret_cast<unsigned*>(&result);
#pragma unroll
      for (int pair=0;pair<4;++pair) {
        const float2 v=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(in+pair));
        const __nv_bfloat162 y=__floats2bfloat162_rn(float(__float2bfloat16_rn(v.x*inv))*scale[2*pair],float(__float2bfloat16_rn(v.y*inv))*scale[2*pair+1]);
        out[pair]=*reinterpret_cast<const unsigned*>(&y);
      }
      *reinterpret_cast<uint4*>(args.resid_out+row*kMlpHidden+k)=result;
    }
  }
  asm volatile("bar.sync 1, 256;" ::: "memory");
  // Low eight bits share the existing arrival word; arrivals use higher bits.
  // The post-SiLU grid join clears these bits only after every GU reader retires.
  if (tid == 0)
    cuda::atomic_ref<unsigned, cuda::thread_scope_device>(args.barrier[0]).fetch_or(1u << row, cuda::memory_order_release);
}

__device__ static bool mlp_wait_normalized(const MlpArgs& args) {
  cuda::atomic_ref<unsigned, cuda::thread_scope_device> ready(args.barrier[0]);
  const naivert::MailboxWatchdog watchdog{args.watchdog_limit_ns,args.watchdog_flag};
  const auto start=watchdog.limit_ns?naivert::mailbox_globaltimer_ns():0ull;
  unsigned spins=0;
  while ((ready.load(cuda::memory_order_relaxed) & kNormReadyMask) != kNormReadyMask) {
    if (watchdog.limit_ns && ((++spins & naivert::kMailboxWatchdogSpinMask)==0u) &&
        naivert::mailbox_globaltimer_ns()-start>watchdog.limit_ns) {
      naivert::mailbox_watchdog_trip(watchdog,0xD420u);return false;
    }
  }
  cuda::ptx::fence(cuda::ptx::sem_acquire,cuda::ptx::scope_gpu);
  return true;
}

// One helper owns each slot's DMA; eight consumer warps release each slot.
__device__ static void mlp_gate_pages(
    const MlpArgs& args, const __nv_bfloat16* x, __nv_bfloat16* weights,
    std::uint64_t* ready, std::uint64_t* empty, unsigned* failed,
    float* partials, float* gu, const MlpArgs* second) {
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const naivert::MailboxWatchdog watchdog{args.watchdog_limit_ns, args.watchdog_flag};
  if (tid == 256) {
    const int row = int(blockIdx.x) * PackedMlpRow::Rows;
    const auto* source = (row < kMlpInterR ? args.gate_w : args.up_w) +
        std::size_t(row % kMlpInterR) * kMlpHidden;
#pragma unroll
    for (int page = 0; page < kGuPages; ++page) {
      const int slot = page % kGuSlots;
      if (page >= kGuSlots &&
          !wait_weight(empty + slot, ((page / kGuSlots) - 1) & 1, watchdog))
        atomicOr(failed, 1u);
      if (*reinterpret_cast<volatile unsigned*>(failed) == 0u) {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        naivert::Barrier::expect_transaction(ready + slot, kGuPageBytes);
        WeightTransport::cp_async_bulk_evict_first(
            weights + slot * (kGuPageBytes / 2),
            source + page * (kGuPageBytes / 2), kGuPageBytes, ready + slot);
        naivert::Barrier::arrive(ready + slot);
      }
    }
  } else if (tid < 256) {
    const int g = lane >> 2, q = lane & 3;
    float accum[Chunks][4] = {};
    if constexpr (Chunks > 1) {
      if (lane == 0) {
#pragma unroll
        for(int chunk=0;chunk<Chunks;++chunk)
          if(!mlp_wait_normalized(chunk==0?args:second[chunk-1]))atomicOr(failed,1u);
      }
      __syncwarp();
    }
#pragma unroll
    for (int page = 0; page < kGuPages; ++page) {
      const int slot = page % kGuSlots;
      unsigned valid = 0u, failure = 0u;
      if (lane == 0) {
        valid = wait_weight(ready + slot, (page / kGuSlots) & 1, watchdog);
        if (!valid) atomicOr(failed, 1u);
        failure = *reinterpret_cast<volatile unsigned*>(failed);
      }
      valid = __shfl_sync(0xffffffffu, valid, 0);
      failure = __shfl_sync(0xffffffffu, failure, 0);
      if (valid && !failure) {
        const auto* pane = weights + slot * (kGuPageBytes / 2);
        const PackedMlpRow w0{pane + g * PackedMlpRow::Page};
        const PackedMlpRow w1{pane + (g + 8) * PackedMlpRow::Page};
#pragma unroll
        for (int k0 = warp * (kGuPageK / 8);
             k0 < (warp + 1) * (kGuPageK / 8); k0 += 32) {
          const int k = k0 + q * 8;
          const uint4 a0 = *reinterpret_cast<const uint4*>(w0 + k);
          const uint4 a1 = *reinterpret_cast<const uint4*>(w1 + k);
#pragma unroll
          for (int chunk = 0; chunk < Chunks; ++chunk) {
            uint4 b;
            if constexpr (Chunks == 1)
              b = *reinterpret_cast<const uint4*>(x + mlp_input_index(g, page * kGuPageK + k));
            else
              b = mlp_load_normalized((chunk == 0 ? args : second[chunk-1]).resid_out + g * kMlpHidden + page * kGuPageK + k);
            naivert::attn_rows::mma_bf16_16816(accum[chunk], a0.x, a1.x, a0.y, a1.y, b.x, b.y);
            naivert::attn_rows::mma_bf16_16816(accum[chunk], a0.z, a1.z, a0.w, a1.w, b.z, b.w);
          }
        }
      }
      __syncwarp();
      if (lane == 0) naivert::Barrier::arrive(empty + slot);
    }
    float* mine = partials + warp * 128 * Chunks;
#pragma unroll
    for (int chunk = 0; chunk < Chunks; ++chunk) {
      mine[g * kMlpWindow + chunk * 8 + 2 * q] = accum[chunk][0];
      mine[g * kMlpWindow + chunk * 8 + 2 * q + 1] = accum[chunk][1];
      mine[(g + 8) * kMlpWindow + chunk * 8 + 2 * q] = accum[chunk][2];
      mine[(g + 8) * kMlpWindow + chunk * 8 + 2 * q + 1] = accum[chunk][3];
    }
  }
  __syncthreads();
  for (int item=tid;item<128*Chunks;item+=kMlpThreads) {
    const int column = item / kMlpWindow, query = item % kMlpWindow;
    float sum = 0.0f;
#pragma unroll
    for (int s = 0; s < 8; ++s) sum += partials[s * 128 * Chunks + column * kMlpWindow + query];
    gu[query * 16 + column] = *failed ? 0.0f : __bfloat162float(__float2bfloat16_rn(sum));
  }
  __syncthreads();
}
