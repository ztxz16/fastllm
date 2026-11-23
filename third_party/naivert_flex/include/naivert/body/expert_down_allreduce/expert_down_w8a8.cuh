// Members of ExpertDownAllreduceBodyT. The existing helper, weight ring,
// SlotProtocol, the body's control flow and the stable descriptor remain the execution owner.

  // This exchange publishes the rank's whole weighted bf16 tile to every peer.
  static constexpr int kTileWords = kComputeOutputs / 2;
  static constexpr int kW8A8Passes = 4 / (kComputeOutputs / 32);   // experts per group = warps 0..3 / rows per warp
  static_assert(!kW8A8 || (kHidden / kComputeOutputs) * kTileWords * sizeof(unsigned long long)
                <= naivert::kMailboxPhaseStride);
  static_assert(!kW8A8 || (kComputeOutputs == 32 || kComputeOutputs == 64));

  static constexpr int kW8A8QuantOffset = 0x1000;
  static constexpr int kW8A8ScaleOffset = 0x1800;
  static constexpr int kW8A8MixtureOffset = 0x1840;
  static constexpr int kW8A8PeerPartialOffset = 0x2000;
  static constexpr int kW8A8ExpertPartialOffset = 0x2800;
  static constexpr int kW8A8MmaBarrierOffset = 0x3800;
  static constexpr int kW8A8TmemSlotOffset = 0x3810;
  static_assert(2 * 8 * 128 <= kW8A8QuantOffset);
  static_assert(!kW8A8 || kW8A8QuantOffset + kOutputSlots * kInter <= kW8A8ScaleOffset);
  static_assert(!kW8A8 || kW8A8ScaleOffset + kOutputSlots * 2 * 4 <= kW8A8MixtureOffset);
  static_assert(kW8A8MixtureOffset + 8 * 2 <= kW8A8PeerPartialOffset);
  static_assert(!kW8A8 || kW8A8PeerPartialOffset + kOutputSlots * 8 * 8 * 4 <= kW8A8ExpertPartialOffset);
  static_assert(!kW8A8 || kW8A8ExpertPartialOffset + kOutputSlots * 8 * 4 <= kW8A8MmaBarrierOffset);
  static_assert(!kW8A8 || kW8A8PeerPartialOffset + kW8A8Passes * kComputeOutputs * int(sizeof(float)) <= kW8A8ExpertPartialOffset);


  __device__ static float w8a8_scale(float amax) {
    return naivert::block128_activation_scale(amax);
  }

  __device__ static void quantize_w8a8(slot_ring::SlotRingState& st,
                                      const ExpertDownAllreduceArgs& args) {
    auto* quant = st.scratch_base + kW8A8QuantOffset;
    auto* scales = reinterpret_cast<float*>(st.scratch_base + kW8A8ScaleOffset);
    // One consumer warp owns an expert, one lane owns four values in K128.
    for (int group = 0; group < 2; ++group) {
      const int k = group * 128 + st.lane_id * 4;
      const int index = st.warp_id * kInter + k;
      const float4 value = *reinterpret_cast<const float4*>(args.up_gate + index);

      float amax = fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)),
                        fmaxf(fabsf(value.z), fabsf(value.w)));
      amax = naivert::block128_warp_amax(amax);
      const float scale = w8a8_scale(amax);
      const float inverse = naivert::block128_inverse_scale(scale);
      const __nv_fp8x4_e4m3 packed(make_float4(
          __fmul_rn(value.x, inverse), __fmul_rn(value.y, inverse),
          __fmul_rn(value.z, inverse), __fmul_rn(value.w, inverse)));
      *reinterpret_cast<unsigned*>(quant + index) = packed.__x;
      if (st.lane_id == 0) scales[st.warp_id * 2 + group] = scale;
    }
    naivert::named_barrier_sync(256u,1u);
  }

  __device__ static int w8a8_umma_offset(int row, int k) {
    return row * 128 + (((k / 16) ^ (row & 7)) * 16) + k % 16;
  }

  template <int WordsPerTile>
  struct W8A8MailboxLayout {
    __host__ __device__ static constexpr std::size_t tile_offset(int task, int word) {
      return std::size_t(task * WordsPerTile + word) * sizeof(unsigned long long);
    }
  };
  static_assert(!kW8A8 || kTiles * 64 * sizeof(unsigned long long) <= naivert::kMailboxPhaseStride);

  template <int WordsPerTile>
  __device__ static auto w8a8_mailbox(const ExpertDownAllreduceArgs& args, int slab) {
    const auto* domain = args.stable_domain;
    auto mailbox = naivert::TileMailbox<W8A8MailboxLayout<WordsPerTile>>::make(
        reinterpret_cast<const std::uint8_t* const*>(domain->peer_mailbox_base_ptrs),
        args.rank, step_gen(args),
        (std::size_t(domain->native_mailbox_slab) + slab) * naivert::kMailboxSlabStride);
    // Replays of the same layer must alternate banks even though its layer tag
    // is unchanged. A rank cannot lap two banks: completing one exchange needs
    // every peer to have entered that exchange. Layer-to-layer reuse is ordered
    // by the intervening attention collective in the existing model graph.
    if (domain->step_block_ptr != 0ull)
      mailbox.phase = unsigned(*reinterpret_cast<const std::uint64_t*>(domain->step_block_ptr)) & 1u;
    return mailbox;
  }

  __device__ static void combine_w8a8_m65_order(
      slot_ring::SlotRingState& st, const ExpertDownAllreduceArgs& args, int tile) {
    const auto* weighted = reinterpret_cast<const float*>(st.scratch_base + kW8A8PeerPartialOffset);
    auto* mixture = reinterpret_cast<__nv_bfloat16*>(st.scratch_base + kW8A8MixtureOffset);
    if (st.tid < kComputeOutputs) {
      float sum = weighted[st.tid];
#pragma unroll
      for (int pass = 1; pass < kW8A8Passes; ++pass) sum = __fadd_rn(sum, weighted[pass * kComputeOutputs + st.tid]);
      mixture[st.tid] = __float2bfloat16_rn(sum);
    }
    naivert::named_barrier_sync(256u,1u);
    const auto box = w8a8_mailbox<kTileWords>(args, 0);
    const auto* pairs = reinterpret_cast<const unsigned*>(mixture);
    if (st.tid < kTileWords) box.publish(tile, st.tid, pairs[st.tid]);
    if constexpr (kPublishOnly) {
      if (st.tid < kTileWords)
        reinterpret_cast<unsigned*>(args.exp_out)[tile*kTileWords+st.tid] = pairs[st.tid];
      naivert::named_barrier_sync(256u,1u);
      return;
    }

    if (st.tid < kTileWords) {
      // Issue the seven peer loads before inspecting any tag, so the
      // arrival round trip is paid once instead of seven times in sequence
      // (microbench: 2.9 -> 1.6 us per exchange).
      unsigned long long word[8];
      const unsigned long long* ptr[8];
#pragma unroll
      for (int source = 0; source < 8; ++source) {
        ptr[source] = source == args.rank ? nullptr
            : box.local_word(naivert::mailbox_source_slot(source, args.rank), tile, st.tid);
        word[source] = source == args.rank ? naivert::tagged_word(box.generation, pairs[st.tid])
            : naivert::load_tagged<false>(ptr[source]);
      }
      bool pending;
      do {
        pending = false;
#pragma unroll
        for (int source = 0; source < 8; ++source) {
          if (naivert::tagged_tag(word[source]) != box.generation) {
            word[source] = naivert::load_tagged<false>(ptr[source]);
            pending = true;
          }
        }
      } while (pending);
      float2 sum = make_float2(0.0f, 0.0f);
#pragma unroll
      for (int source = 0; source < 8; ++source) {
        const unsigned packed = naivert::tagged_payload(word[source]);
        const float2 value = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&packed));
        sum.x = __fadd_rn(sum.x, value.x); sum.y = __fadd_rn(sum.y, value.y);
      }
      const int hidden = tile * kComputeOutputs + st.tid * 2;
      const float2 residual = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(args.residual + hidden));
      auto* result = args.tp_result != nullptr ? args.tp_result : args.exp_out;
      *reinterpret_cast<__nv_bfloat162*>(result + hidden) = __floats2bfloat162_rn(
          __fadd_rn(sum.x, residual.x), __fadd_rn(sum.y, residual.y));
    }
    naivert::named_barrier_sync(256u,1u);
  }

  __device__ static void consume_output_tile_w8a8(
      slot_ring::SlotRingState& st, const ExpertDownAllreduceArgs& args, int output_tile) {
    using E4M3 = cutlass::float_e4m3_t;
    using Atom = cute::UMMA::Layout_K_SW128_Atom<E4M3>;
    // CUTLASS 4.1 exposes the same PTX wrapper without template parameters.
#if CUTLASS_MAJOR == 4 && CUTLASS_MINOR == 1
    using Mma = cute::SM100_MMA_F8F6F4_SS;
#else
    using Mma = cute::SM100_MMA_F8F6F4_SS<E4M3, E4M3, float, 128, 8,
        cute::UMMA::Major::K, cute::UMMA::Major::K>;
#endif
    const auto outer_tx = st.tx;
    auto* done = reinterpret_cast<std::uint64_t*>(st.scratch_base + kW8A8MmaBarrierOffset);
    auto* tmem_slot = reinterpret_cast<unsigned*>(st.scratch_base + kW8A8TmemSlotOffset);
    if (st.tid == 0) {
      naivert::Barrier::init(done, 1u);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (st.warp_id == 0 && !kKeepTmem) {
      cute::TMEM::Allocator1Sm allocator;
      allocator.allocate(32, tmem_slot);
      allocator.release_allocation_lock();
    }
    quantize_w8a8(st, args);
    const unsigned tmem = *tmem_slot;
    const auto* quant = st.scratch_base + kW8A8QuantOffset;
    const auto* activation_scales = reinterpret_cast<const float*>(st.scratch_base + kW8A8ScaleOffset);
    auto* b_image = st.scratch_base;  // Two 8x128 SW128 panels (2 KiB).
    const auto a_layout = cute::tile_to_shape(Atom{}, cute::Shape<cute::Int<128>, cute::Int<128>>{});
    const auto b_layout = cute::tile_to_shape(Atom{}, cute::Shape<cute::Int<8>, cute::Int<128>>{});
    const auto idesc = cute::UMMA::make_instr_desc<E4M3, E4M3, float, 128, 8,
        cute::UMMA::Major::K, cute::UMMA::Major::K>();
    const auto runtime_desc = cute::UMMA::make_runtime_instr_desc<>(idesc);
    // Keep all eight expert activations as the eight columns of B. Each
    // expert group computes M128xN8, and each expert epilogue
    // selects the matching column. B is published once and stays immutable
    // through all eight experts, avoiding eight broadcasts of a single row.
    for (int word = st.tid; word < 8 * 256 / 4; word += kConsumerThreads) {
      const int row = word / 64, k = word % 64 * 4;
      const unsigned value = *reinterpret_cast<const unsigned*>(quant + row * 256 + k);
      *reinterpret_cast<unsigned*>(b_image + (k / 128) * 8 * 128 +
                                   w8a8_umma_offset(row, k % 128)) = value;
    }
    // Per (pass, row) fp32 accumulator of route-weighted expert slices, owned by one lane throughout.
    auto* weighted = reinterpret_cast<float*>(st.scratch_base + kW8A8PeerPartialOffset);
    if constexpr (kParallelW8A8Epilogue) {
      for (int i = st.tid; i < kW8A8Passes * kComputeOutputs; i += kConsumerThreads) weighted[i] = 0.0f;
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    naivert::named_barrier_sync(256u,1u);
    // Only the four TMEM reader warps need the weight/MMA loop. They return
    // all256 logical slot credits after actual MMA completion:128×2 arrivals.
    // The other four consumer warps join them before the unchanged TP phase.
    if (st.tid < 128) {
    // Multirow: pass * 64 + row == tid, so no other thread reads this
    // partial before the final consumer join. Keep each RN update local.
    float weighted_accumulator = 0.0f;
    // Stack each expert group along M; K128 groups keep separate columns.
#pragma unroll (kParallelW8A8Epilogue ? 1 : 4)
    for (int expert_base = 0; expert_base < kOutputSlots; expert_base += kW8A8GroupExperts) {
      const int owned_pass = st.warp_id / (kComputeOutputs / 32);
      const int owned_row = (st.warp_id % (kComputeOutputs / 32)) * 32 + st.lane_id;
      const int owned_slot = expert_base + owned_pass;
      float early_scale0 = 0.0f, early_scale1 = 0.0f, early_route = 0.0f;
      if constexpr (!kParallelW8A8Epilogue) {
        const int owned_expert = logical_expert(args, owned_slot);
        early_scale0 = __fmul_rn(load_scale(args, owned_expert, output_tile * kComputeOutputs + owned_row, 0),
                                activation_scales[owned_slot * 2]);
        early_scale1 = __fmul_rn(load_scale(args, owned_expert, output_tile * kComputeOutputs + owned_row, 1),
                                activation_scales[owned_slot * 2 + 1]);
        early_route = route_weight(args, owned_slot);
        // Consume the operands here to keep their loads before the ring wait.
        // This compiler barrier emits no device instruction.
        asm volatile("" :: "f"(early_scale0), "f"(early_scale1), "f"(early_route) : "memory");
      }
      for (int pass = 0; pass < kW8A8GroupExperts; ++pass) {
        set_inner_ring_slot(st, expert_base + pass);
        Slot::consumer_wait(st);
      }
      if (st.warp_id == 0) {
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#pragma unroll
        for (int kg = 0; kg < 2; ++kg) {
          const auto a = cute::make_tensor(cute::make_smem_ptr(
              reinterpret_cast<E4M3*>(st.deep_base + kg * 16384)), a_layout);
          const auto b = cute::make_tensor(cute::make_smem_ptr(
              reinterpret_cast<E4M3*>(b_image + kg * 1024)), b_layout);
          const auto da = cute::UMMA::make_umma_desc<cute::UMMA::Major::K>(a);
          const auto db = cute::UMMA::make_umma_desc<cute::UMMA::Major::K>(b);
#pragma unroll
          for (int kk = 0; kk < 4; ++kk) {
            auto dak = da, dbk = db;
            dak.start_address_ = static_cast<unsigned short>(da.start_address_ + kk * 2);
            dbk.start_address_ = static_cast<unsigned short>(db.start_address_ + kk * 2);
            Mma::fma(static_cast<unsigned long long>(dak), static_cast<unsigned long long>(dbk),
                     tmem + kg * 8, unsigned(kk != 0), runtime_desc);
          }
        }
        cutlass::arch::umma_arrive(done);
      }
      while (!naivert::Barrier::try_wait(done, unsigned((expert_base / kW8A8GroupExperts) & 1))) {}
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      // The helper visits each expert once and has no outer refill. Only
      // multirow groups with a later reuse return their weight slots. Decode
      // preserves its existing releases and loop structure.
      if (kParallelW8A8Epilogue || expert_base + kW8A8GroupExperts < kOutputSlots) {
        for (int pass = 0; pass < kW8A8GroupExperts; ++pass) {
          set_inner_ring_slot(st, expert_base + pass);
          naivert::barrier_arrive_count(st.empty_barriers + st.tx.slot, 2u);
        }
      }
      if (st.warp_id < 4) {
        const int pass = st.warp_id / (kComputeOutputs / 32);
        const int row = (st.warp_id % (kComputeOutputs / 32)) * 32 + st.lane_id;
        const int expert_slot = expert_base + pass;
        const int expert = logical_expert(args, expert_slot);
        float result = 0.0f;
#pragma unroll
        for (int kg = 0; kg < 2; ++kg) {
          unsigned value;
          cute::SM100_TMEM_LOAD_32dp32b1x::copy(
              tmem + kg * 8 + expert_slot + (unsigned(st.warp_id * 32) << 16), value);
          cutlass::arch::fence_view_async_tmem_load();
          float scale;
          if constexpr (!kParallelW8A8Epilogue) scale = kg == 0 ? early_scale0 : early_scale1;
          else scale = __fmul_rn(load_scale(args, expert, output_tile * kComputeOutputs + row, kg),
                                 activation_scales[expert_slot * 2 + kg]);
          result = __fadd_rn(result, __fmul_rn(__uint_as_float(value), scale));
        }
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        // order -- weight by the route probability here, sum the experts in fp32 per rank.
        if constexpr (kParallelW8A8Epilogue) {
          weighted[pass * kComputeOutputs + row] = __fadd_rn(weighted[pass * kComputeOutputs + row],
                                                            __fmul_rn(result, kParallelW8A8Epilogue ? route_weight(args, expert_slot) : early_route));
        } else {
          weighted_accumulator = __fadd_rn(weighted_accumulator,
                                           __fmul_rn(result, early_route));
        }
      }
      naivert::named_barrier_sync(128u,2u);
    }
    // Each compute lane owns its partial until this final publication.
    if constexpr (!kParallelW8A8Epilogue) weighted[st.tid] = weighted_accumulator;
    }
    naivert::named_barrier_sync(256u,1u);
    combine_w8a8_m65_order(st, args, output_tile);
    if (st.warp_id == 0 && !kKeepTmem) {
      cute::TMEM::Allocator1Sm allocator;
      allocator.free(tmem, 32);
    }
    st.tx = outer_tx;
  }
