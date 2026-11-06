#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
log_dir=${W8A8_LOG_DIR:-"${repo_dir}/test/w8a8/logs/$(date -u +%Y%m%dT%H%M%SZ)"}
optest=${W8A8_OPTEST:-"${repo_dir}/optest"}
model=${W8A8_MODEL:-}
vllm_python=${W8A8_VLLM_PYTHON:-python}
nvcc=${W8A8_NVCC:-/usr/local/cuda/bin/nvcc}
attention_backend=${W8A8_ATTENTION_BACKEND:-auto}
kv_cache_layout=${W8A8_KV_CACHE_LAYOUT:-auto}
suite=${1:-ops}
mkdir -p "${log_dir}"
functional_summary_pending=0
attention_backend_args=(
    --flm-kv-cache-layout "${kv_cache_layout}"
    --flm-attention-backend "${attention_backend}"
)
attention_layout_matrix=(
    "continuous:auto"
    "paged:native_paged"
    "paged:flashinfer_paged"
)
if [[ "${W8A8_ATTENTION_BACKEND_STRICT:-0}" == 1 ]]; then
    attention_backend_args+=(--flm-attention-backend-strict)
fi

write_functional_summary() {
    "${vllm_python}" "${repo_dir}/test/w8a8/operator_functional_summary.py" \
        --log-dir "${log_dir}"
}

write_pending_functional_summary() {
    local status=$?
    trap - EXIT
    if [[ "${functional_summary_pending}" == 1 ]]; then
        write_functional_summary || \
            echo "WARNING: failed to write W8A8 functional summary" >&2
    fi
    exit "${status}"
}

trap write_pending_functional_summary EXIT

run_logged() {
    local name=$1
    shift
    local log_file="${log_dir}/${name}.log"
    {
        printf 'UTC: %s\n' "$(date -u +%FT%TZ)"
        printf 'PWD: %q\n' "${repo_dir}"
        printf 'COMMAND:'
        printf ' %q' "$@"
        printf '\n\n'
    } | tee "${log_file}"
    (cd "${repo_dir}" && "$@") 2>&1 | tee -a "${log_file}"
}

detect_arch() {
    local cc major minor
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed -n '1p')
    major=${cc%%.*}; minor=${cc#*.}
    printf '%d\n' "$((10#${major} * 10 + 10#${minor}))"
}

ops_scope() {
    case "${suite}" in
        ops-dense|ops-dense-*) printf 'dense\n' ;;
        ops-block128|ops-block128-*) printf 'block128\n' ;;
        ops-int8-symmetric|ops-int8-symmetric-*) printf 'int8-symmetric\n' ;;
        ops-int8-azp|ops-int8-azp-*) printf 'int8-azp\n' ;;
        ops-moe|ops-moe-*) printf 'moe\n' ;;
        *) printf 'all\n' ;;
    esac
}

run_build() {
    local arch build_arch
    arch=$(detect_arch)
    case "${arch}" in
        90) build_arch=90a ;;
        120) build_arch=120 ;;
        *)
            echo "unsupported build GPU: SM${arch}; expected SM90 or SM120" >&2
            exit 2
            ;;
    esac

    run_logged build bash install.sh -DUSE_CUDA=ON \
        -DCMAKE_CUDA_COMPILER="${nvcc}" -DCUDA_ARCH="${build_arch}" \
        -DUNIT_TEST=ON \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
}

run_standard_check() {
    local name=$1 m=$2 n=$3 k=$4 dtype=$5 bias=$6
    run_logged "${name}_${dtype}_bias${bias}" env \
        FASTLLM_CUDA_W8A8=1 FASTLLM_CUDA_W8A8_STRICT=1 \
        "${optest}" --op linear_fp8_w8a8 --device cuda:0 \
        --param batch="${m}" --param in="${k}" --param out="${n}" \
        --param weight_layout=perchannel --param input_type="${dtype}" \
        --param has_bias="${bias}" --param check=1 --warmup 0 --iters 1
}

run_sm120_block128_check() {
    local name=$1 m=$2 n=$3 k=$4 dtype=$5
    run_logged "${name}_${dtype}" env \
        FASTLLM_CUDA_W8A8=1 FASTLLM_CUDA_W8A8_STRICT=1 \
        "${optest}" --op linear_fp8_block128 --device cuda:0 \
        --param batch="${m}" --param in="${k}" --param out="${n}" \
        --param block=128 --param weight_layout=separate \
        --param input_type="${dtype}" --param has_bias=0 \
        --param check=1 --warmup 0 --iters 1
}

run_ops_functional_cases() {
    local arch scope
    arch=$(detect_arch)
    scope=$(ops_scope)

    if [[ "${arch}" == 90 ]]; then
        if [[ "${scope}" == all || "${scope}" == dense ]]; then
            # vLLM SM90标准FP8分派边界：小M走SwapAB，并按N=1280分支；
            # 64/128是SwapAB、Pingpong和默认Persistent配置的边界。
            while read -r m n; do
                run_standard_check "sm90_branch_m${m}_n${n}" \
                    "${m}" "${n}" 4096 bf16 1
            done <<'EOF'
1 1280
1 1288
16 4096
17 1280
17 1288
64 4096
65 4096
128 4096
129 4096
EOF

            for dtype in fp16 bf16; do
                for bias in 0 1; do
                    run_standard_check sm90_semantics \
                        17 4096 4096 "${dtype}" "${bias}"
                done
            done

            for bias in 0 1; do
                run_logged "sm90_tensorwise_weight_bias${bias}" env \
                    FASTLLM_CUDA_W8A8=1 FASTLLM_CUDA_W8A8_STRICT=1 \
                    "${optest}" --op linear_fp8_w8a8 --device cuda:0 \
                    --param batch=17 --param in=4096 --param out=4096 \
                    --param weight_layout=tensorwise --param input_type=bf16 \
                    --param has_bias="${bias}" --param check=1 \
                    --warmup 0 --iters 1
            done

            for check in 2 3 4 5 6; do
                run_logged "sm90_dense_lifecycle_${check}" env \
                    FASTLLM_CUDA_W8A8=1 FASTLLM_CUDA_W8A8_STRICT=1 \
                    "${optest}" --op linear_fp8_w8a8 --device cuda:0 \
                    --param batch=17 --param in=4096 --param out=4096 \
                    --param weight_layout=perchannel --param input_type=bf16 \
                    --param has_bias=1 --param check="${check}" \
                    --warmup 0 --iters 1
            done
        fi

        if [[ "${scope}" == all || "${scope}" == int8-azp ||
              "${scope}" == int8-symmetric ]]; then
            local -a activation_schemes=(symmetric)
            if [[ "${scope}" == all || "${scope}" == int8-azp ]]; then
                activation_schemes=(symmetric symmetric_static azp_dynamic azp_static)
            fi
            for activation_scheme in "${activation_schemes[@]}"; do
                for dtype in fp16 bf16; do
                    for bias in 0 1; do
                        run_logged "sm90_int8_w8a8_${activation_scheme}_${dtype}_bias${bias}" env \
                            FASTLLM_CUDA_W8A8=1 FASTLLM_CUDA_W8A8_STRICT=1 \
                            FASTLLM_CUDA_W8A8_TRACE=1 \
                            "${optest}" --op linear_int8_w8a8 --device cuda:0 \
                            --param batch=17 --param in=4096 --param out=4096 \
                            --param input_type="${dtype}" --param has_bias="${bias}" \
                            --param activation_scheme="${activation_scheme}" \
                            --param check=1 --warmup 0 --iters 1
                    done
                done
            done
        fi

        if [[ "${scope}" == all || "${scope}" == int8-azp ]]; then
            while read -r name m n; do
                run_logged "sm90_int8_w8a8_azp_${name}" env \
                    FASTLLM_CUDA_W8A8=1 FASTLLM_CUDA_W8A8_STRICT=1 \
                    FASTLLM_CUDA_W8A8_TRACE=1 \
                    "${optest}" --op linear_int8_w8a8 --device cuda:0 \
                    --param batch="${m}" --param in=128 --param out="${n}" \
                    --param input_type=bf16 --param has_bias=1 \
                    --param activation_scheme=azp_dynamic \
                    --param check=1 --warmup 0 --iters 1
            done <<'EOF'
branch_m1_n4096 1 4096
branch_m32_n4096 32 4096
branch_m33_n4096 33 4096
branch_m64_n4096 64 4096
branch_m65_n4096 65 4096
branch_m128_n4096 128 4096
branch_m129_n4096 129 4096
branch_m17_n8176 17 8176
branch_m17_n8192 17 8192
EOF
            run_logged sm90_int8_w8a8_reject_weight_zero_point env \
                FASTLLM_CUDA_W8A8=1 FASTLLM_CUDA_W8A8_STRICT=1 \
                FASTLLM_CUDA_W8A8_TRACE=1 \
                "${optest}" --op linear_int8_w8a8 --device cuda:0 \
                --param batch=17 --param in=4096 --param out=4096 \
                --param input_type=bf16 --param has_bias=1 \
                --param activation_scheme=azp_dynamic \
                --param check=2 --warmup 0 --iters 1
            run_logged sm90_int8_w8a8_reject_static_scale env \
                FASTLLM_CUDA_W8A8=1 FASTLLM_CUDA_W8A8_STRICT=1 \
                "${optest}" --op linear_int8_w8a8 --device cuda:0 \
                --param batch=17 --param in=4096 --param out=4096 \
                --param input_type=bf16 --param has_bias=1 \
                --param activation_scheme=azp_static \
                --param check=3 --warmup 0 --iters 1
            run_logged sm90_int8_w8a8_reject_symmetric_zero_point env \
                FASTLLM_CUDA_W8A8=1 FASTLLM_CUDA_W8A8_STRICT=1 \
                "${optest}" --op linear_int8_w8a8 --device cuda:0 \
                --param batch=17 --param in=4096 --param out=4096 \
                --param input_type=bf16 --param has_bias=1 \
                --param activation_scheme=symmetric \
                --param check=4 --warmup 0 --iters 1
        fi

        if [[ "${scope}" == all || "${scope}" == block128 ]]; then
            run_logged sm90_fp8_blockwise_function env \
                FASTLLM_CUDA_W8A8=1 FASTLLM_CUDA_W8A8_STRICT=1 \
                "${optest}" --op linear_fp8_block128 --device cuda:0 \
                --param batch=17 --param in=4096 --param out=4096 \
                --param block=128 --param weight_layout=separate \
                --param input_type=bf16 --param check=1 --warmup 0 --iters 1
        fi

        if [[ "${scope}" == all || "${scope}" == moe ]]; then
            while read -r case_name batch hidden inter input_type scale_layout shared_expert; do
                run_logged "${case_name}" env \
                    FASTLLM_CUDA_MOE_GROUPED_INDEXED=1 \
                    FASTLLM_CUDA_MOE_GROUPED_INDEXED_MIN_BATCH=1 \
                    FASTLLM_CUDA_MOE_CUTLASS_FP8_SM90=1 \
                    FASTLLM_CUDA_MOE_CUTLASS_FP8_SM90_STRICT=1 \
                    FASTLLM_CUDA_W8A8_TRACE=1 \
                    FASTLLM_CUDA_TRITON_MERGE_MOE=0 \
                    "${optest}" --op mergemoe_fp8 --device cuda:0 \
                    --param batch="${batch}" --param hidden="${hidden}" \
                    --param inter="${inter}" --param experts=16 --param topk=4 \
                    --param block=128 --param input_type="${input_type}" \
                    --param weight_type=fp8 --param scale_layout="${scale_layout}" \
                    --param shared_expert="${shared_expert}" \
                    --param path=check_fp8 --warmup 0 --iters 1
            done <<'EOF'
sm90_fp8_grouped_moe_m4_bf16 1 2048 768 bf16 perchannel 0
sm90_fp8_grouped_moe_m64_fp16 16 2048 768 fp16 perchannel 0
sm90_fp8_grouped_moe_n8192_bf16 32 2048 4096 bf16 tensorwise 0
sm90_fp8_grouped_moe_k8192_bf16 32 8192 768 bf16 perchannel 0
sm90_fp8_grouped_moe_default_shared_bf16 128 2048 768 bf16 perchannel 1
EOF

            while read -r case_name batch input_type shared_expert; do
                run_logged "${case_name}" env \
                    FASTLLM_CUDA_TRITON=1 \
                    FASTLLM_CUDA_TRITON_MERGE_MOE=1 \
                    FASTLLM_CUDA_TRITON_PYTHON="${vllm_python}" \
                    FASTLLM_CUDA_MOE_BLOCK128_SM90=1 \
                    FASTLLM_CUDA_MOE_BLOCK128_SM90_STRICT=1 \
                    FASTLLM_CUDA_W8A8_TRACE=1 \
                    "${optest}" --op mergemoe_fp8 --device cuda:0 \
                    --param batch="${batch}" --param hidden=2048 \
                    --param inter=768 --param experts=16 --param topk=4 \
                    --param block=128 --param input_type="${input_type}" \
                    --param weight_type=fp8 --param scale_layout=block128 \
                    --param shared_expert="${shared_expert}" \
                    --param path=check_block128 --warmup 0 --iters 1
            done <<'EOF'
sm90_block128_m4_bf16 1 bf16 0
sm90_block128_m16_fp16 4 fp16 0
sm90_block128_m64_bf16 16 bf16 0
sm90_block128_m256_bf16 64 bf16 0
sm90_block128_m512_shared_bf16 128 bf16 1
EOF

            while read -r case_name batch input_type; do
                run_logged "${case_name}" env \
                    FASTLLM_CUDA_TRITON=1 \
                    FASTLLM_CUDA_TRITON_MERGE_MOE=1 \
                    FASTLLM_CUDA_TRITON_PYTHON="${vllm_python}" \
                    FASTLLM_CUDA_MOE_BLOCK128_SM90=1 \
                    FASTLLM_CUDA_MOE_BLOCK128_SM90_STRICT=1 \
                    FASTLLM_CUDA_W8A8_TRACE=1 \
                    "${optest}" --op fusedmoe_fp8 --device cuda:0 \
                    --param batch="${batch}" --param hidden=2048 \
                    --param inter=768 --param experts=16 --param topk=4 \
                    --param block=128 --param input_type="${input_type}" \
                    --param scale_layout=block128 \
                    --param path=check_block128 --warmup 0 --iters 1
            done <<'EOF'
sm90_block128_fused_m4_bf16 1 bf16
sm90_block128_fused_m64_fp16 16 fp16
sm90_block128_fused_m256_bf16 64 bf16
EOF
        fi

        if [[ "${scope}" != int8-azp &&
              "${scope}" != int8-symmetric &&
              "${scope}" != moe ]]; then
            run_logged "sm90_vllm_official_functional_${scope}" \
                "${vllm_python}" test/w8a8/operator_functional_vllm.py \
                --scale-layout "${scope}"
        fi
    elif [[ "${arch}" == 120 ]]; then
        if [[ "${scope}" == all || "${scope}" == dense ]]; then
            # 16/32/256是SM120 CUTLASS tile选择的三个边界；
            # 两侧值用于确认每个条件分支都被执行。
        for m in 1 16 17 32 33 256 257; do
            run_standard_check "sm120_branch_m${m}" \
                "${m}" 4096 4096 bf16 1
        done
        # 输入类型和bias语义分支。
        for dtype in fp16 bf16; do
            for bias in 0 1; do
                run_standard_check sm120_semantics \
                    17 4096 4096 "${dtype}" "${bias}"
            done
        done
        # vLLM ScaledEpilogue同时支持B scale为per-channel或tensorwise。
        # 该case仍走正式Linear入口，覆盖融合epilogue的标量广播分支。
        for bias in 0 1; do
            run_logged "sm120_tensorwise_weight_bias${bias}" env \
                FASTLLM_CUDA_W8A8=1 FASTLLM_CUDA_W8A8_STRICT=1 \
                "${optest}" --op linear_fp8_w8a8 --device cuda:0 \
                --param batch=17 --param in=4096 --param out=4096 \
                --param weight_layout=tensorwise --param input_type=bf16 \
                --param has_bias="${bias}" --param check=1 --warmup 0 --iters 1
        done
        run_logged sm120_fixed_backend_lifecycle env \
            FASTLLM_CUDA_W8A8=1 FASTLLM_CUDA_W8A8_STRICT=1 \
            "${optest}" --op linear_fp8_w8a8 --device cuda:0 \
            --param batch=17 --param in=4096 --param out=4096 \
            --param weight_layout=perchannel --param input_type=bf16 \
            --param has_bias=1 --param check=2 --warmup 0 --iters 1
        # 后端生命周期故障测试使用正式CUTLASS入口，只通过测试开关注入
        # GEMM失败；不绕过正式状态机、缓存或同步边界。
        run_logged sm120_first_gemm_failure_rejected env \
            FASTLLM_CUDA_W8A8=1 \
            "${optest}" --op linear_fp8_w8a8 --device cuda:0 \
            --param batch=17 --param in=4096 --param out=4096 \
            --param weight_layout=perchannel --param input_type=bf16 \
            --param has_bias=1 --param check=3 --warmup 0 --iters 1
        run_logged sm120_fixed_cutlass_failure_no_fallback env \
            FASTLLM_CUDA_W8A8=1 FASTLLM_CUDA_W8A8_STRICT=1 \
            "${optest}" --op linear_fp8_w8a8 --device cuda:0 \
            --param batch=17 --param in=4096 --param out=4096 \
            --param weight_layout=perchannel --param input_type=bf16 \
            --param has_bias=1 --param check=4 --warmup 0 --iters 1
        run_logged sm120_destructor_clears_backend_state env \
            FASTLLM_CUDA_W8A8=1 FASTLLM_CUDA_W8A8_STRICT=1 \
            "${optest}" --op linear_fp8_w8a8 --device cuda:0 \
            --param batch=17 --param in=4096 --param out=4096 \
            --param weight_layout=perchannel --param input_type=bf16 \
            --param has_bias=1 --param check=5 --warmup 0 --iters 1
        run_logged sm120_gpu_migration_rebuilds_cache env \
            FASTLLM_CUDA_W8A8=1 FASTLLM_CUDA_W8A8_STRICT=1 \
            "${optest}" --op linear_fp8_w8a8 --device cuda:0 \
            --param batch=17 --param in=4096 --param out=4096 \
            --param weight_layout=perchannel --param input_type=bf16 \
            --param has_bias=1 --param check=6 --warmup 0 --iters 1
        # vLLM test_cutlass_scaled_mm.py中的全部对齐MNK形状。FastLLM只
        # 比较双方共同支持的per-token A scale和per-channel B scale语义。
        while read -r m n k; do
            run_standard_check "sm120_vllm_m${m}_n${n}_k${k}" \
                "${m}" "${n}" "${k}" bf16 0
        done <<'EOF'
1 256 128
1 16384 1024
1 24576 496
16 256 496
16 16384 128
16 24576 4096
32 8192 4096
32 16384 4096
33 1024 1024
33 8192 128
64 2048 496
64 16384 1024
100 8192 496
128 32768 4096
256 4096 4096
512 256 1024
512 8192 4096
512 16384 128
512 24576 128
EOF
        fi

        if [[ "${scope}" != dense ]]; then
            # vLLM SM120 Blockwise分派边界：M<=64或M%4!=0
            # 走SwapAB；其余M<=256走Pingpong；M>256走Cooperative。
        for m in 1 64 65 68 256 257; do
            run_sm120_block128_check \
                "sm120_fp8_blockwise_m${m}" "${m}" 4096 4096 bf16
        done

        # 与vLLM test_cutlass_fp8_blockwise_scale_gemm相同的MNK集合；
        # 仅保留K/N均可被128整除的合法Block128形状。
        while read -r m n k; do
            run_sm120_block128_check \
                "sm120_vllm_block_m${m}_n${n}_k${k}" \
                "${m}" "${n}" "${k}" bf16
        done <<'EOF'
1 256 128
1 16384 1024
16 16384 128
16 24576 4096
32 8192 4096
32 16384 4096
33 1024 1024
33 8192 128
64 16384 1024
128 32768 4096
256 4096 4096
512 256 1024
512 8192 4096
512 16384 128
512 24576 128
EOF
        # vLLM官方dtype case同时覆盖BF16和FP16输出。
        run_sm120_block128_check \
            sm120_vllm_block_output_dtype 512 512 512 fp16

        for check in 2 3 4 5 6; do
            run_logged "sm120_fp8_blockwise_lifecycle_${check}" env \
                FASTLLM_CUDA_W8A8=1 \
                "${optest}" --op linear_fp8_block128 --device cuda:0 \
                --param batch=68 --param in=4096 --param out=4096 \
                --param block=128 --param weight_layout=separate \
                --param input_type=bf16 --param has_bias=1 \
                --param check="${check}" --warmup 0 --iters 1
        done
        fi

        run_logged "sm120_vllm_official_functional_${scope}" \
            "${vllm_python}" test/w8a8/operator_functional_vllm.py \
            --scale-layout "${scope}"
    else
        echo "unsupported test GPU: SM${arch}; expected SM90 or SM120" >&2
        exit 2
    fi
}

run_ops_functional() {
    functional_summary_pending=1
    run_ops_functional_cases
    write_functional_summary
    functional_summary_pending=0
}

run_ops_performance() {
    local arch scope
    arch=$(detect_arch)
    scope=$(ops_scope)

    if [[ "${arch}" == 90 ]]; then
        if [[ "${scope}" == all || "${scope}" == dense ]]; then
            run_logged sm90_w8a8_perchannel_operator_compare \
                "${vllm_python}" test/w8a8/operator_performance_compare.py \
                --optest "${optest}" \
                --output-dir "${log_dir}/operator-compare/perchannel" \
                --scale-layout perchannel \
                --warmup 20 --iters 200 --outer-repeats 5
        fi

        if [[ "${scope}" == all || "${scope}" == int8-azp ]]; then
            run_logged sm90_int8_azp_operator_compare \
                "${vllm_python}" test/w8a8/operator_performance_compare.py \
                --optest "${optest}" \
                --output-dir "${log_dir}/operator-compare/int8-azp" \
                --quantization int8-azp --scale-layout perchannel \
                --warmup 20 --iters 200 --outer-repeats 5
        fi

        if [[ "${scope}" == all || "${scope}" == int8-symmetric ]]; then
            run_logged sm90_int8_symmetric_operator_compare \
                "${vllm_python}" test/w8a8/operator_performance_compare.py \
                --optest "${optest}" \
                --output-dir "${log_dir}/operator-compare/int8-symmetric" \
                --quantization int8-symmetric --scale-layout perchannel \
                --warmup 20 --iters 200 --outer-repeats 5
        fi

        for batch in 1 16 64 256 1024; do
            if [[ "${scope}" == all || "${scope}" == block128 ]]; then
                run_logged "sm90_fp8_blockwise_m${batch}_perf" env \
                    FASTLLM_CUDA_W8A8=1 FASTLLM_CUDA_W8A8_STRICT=1 \
                    "${optest}" --op linear_fp8_block128 --device cuda:0 \
                    --param batch="${batch}" --param in=4096 --param out=4096 \
                    --param block=128 --param weight_layout=separate \
                    --param input_type=bf16 --warmup 20 --iters 200
            fi
        done

        if [[ "${scope}" == all || "${scope}" == block128 ]]; then
            run_logged sm90_w8a8_block128_operator_compare \
                "${vllm_python}" test/w8a8/operator_performance_compare.py \
                --optest "${optest}" \
                --output-dir "${log_dir}/operator-compare/block128" \
                --scale-layout block128 \
                --warmup 20 --iters 200 --outer-repeats 5
        fi

        if [[ "${scope}" == all || "${scope}" == moe ]]; then
            run_logged sm90_fp8_grouped_moe_perf env \
                FASTLLM_CUDA_MOE_GROUPED_INDEXED=1 \
                FASTLLM_CUDA_MOE_GROUPED_INDEXED_MIN_BATCH=1 \
                FASTLLM_CUDA_MOE_CUTLASS_FP8_SM90=1 \
                FASTLLM_CUDA_MOE_CUTLASS_FP8_SM90_STRICT=1 \
                FASTLLM_CUDA_TRITON_MERGE_MOE=0 \
                "${optest}" --op mergemoe_fp8 --device cuda:0 \
                --param batch=256 --param hidden=2048 --param inter=768 \
                --param experts=16 --param topk=4 --param block=128 \
                --param input_type=bf16 --param weight_type=fp8 \
                --param scale_layout=perchannel \
                --param path=operator --warmup 20 --iters 200

            run_logged sm90_fp8_grouped_moe_fallback_perf env \
                FASTLLM_CUDA_MOE_GROUPED_INDEXED=1 \
                FASTLLM_CUDA_MOE_GROUPED_INDEXED_MIN_BATCH=1 \
                FASTLLM_CUDA_MOE_CUTLASS_FP8_SM90=0 \
                FASTLLM_CUDA_MOE_CUTLASS_FP8_SM90_STRICT=0 \
                FASTLLM_CUDA_TRITON_MERGE_MOE=0 \
                "${optest}" --op mergemoe_fp8 --device cuda:0 \
                --param batch=256 --param hidden=2048 --param inter=768 \
                --param experts=16 --param topk=4 --param block=128 \
                --param input_type=bf16 --param weight_type=fp8 \
                --param scale_layout=perchannel \
                --param path=operator --warmup 20 --iters 200

            for batch in 1 16 64 256; do
                run_logged "sm90_block128_moe_m$((batch * 4))_perf" env \
                    FASTLLM_CUDA_TRITON=1 \
                    FASTLLM_CUDA_TRITON_MERGE_MOE=1 \
                    FASTLLM_CUDA_TRITON_PYTHON="${vllm_python}" \
                    FASTLLM_CUDA_MOE_BLOCK128_SM90=1 \
                    FASTLLM_CUDA_MOE_BLOCK128_SM90_STRICT=1 \
                    "${optest}" --op mergemoe_fp8 --device cuda:0 \
                    --param batch="${batch}" --param hidden=2048 \
                    --param inter=768 --param experts=16 --param topk=4 \
                    --param block=128 --param input_type=bf16 \
                    --param weight_type=fp8 --param scale_layout=block128 \
                    --param path=operator --warmup 20 --iters 200
            done

            run_logged sm90_block128_moe_fallback_perf env \
                FASTLLM_CUDA_TRITON=1 \
                FASTLLM_CUDA_TRITON_MERGE_MOE=0 \
                FASTLLM_CUDA_MOE_BLOCK128_SM90=0 \
                FASTLLM_CUDA_MOE_BLOCK128_SM90_STRICT=0 \
                "${optest}" --op mergemoe_fp8 --device cuda:0 \
                --param batch=256 --param hidden=2048 --param inter=768 \
                --param experts=16 --param topk=4 --param block=128 \
                --param input_type=bf16 --param weight_type=fp8 \
                --param scale_layout=block128 \
                --param path=operator --warmup 20 --iters 200

            run_logged sm90_block128_fused_moe_perf env \
                FASTLLM_CUDA_TRITON=1 \
                FASTLLM_CUDA_TRITON_MERGE_MOE=1 \
                FASTLLM_CUDA_TRITON_PYTHON="${vllm_python}" \
                FASTLLM_CUDA_MOE_BLOCK128_SM90=1 \
                FASTLLM_CUDA_MOE_BLOCK128_SM90_STRICT=1 \
                "${optest}" --op fusedmoe_fp8 --device cuda:0 \
                --param batch=256 --param hidden=2048 --param inter=768 \
                --param experts=16 --param topk=4 --param block=128 \
                --param input_type=bf16 --param scale_layout=block128 \
                --param path=operator --warmup 20 --iters 200
        fi
    elif [[ "${arch}" == 120 ]]; then
        if [[ "${scope}" != block128 ]]; then
            run_logged sm120_w8a8_perchannel_operator_compare \
                "${vllm_python}" test/w8a8/operator_performance_compare.py \
                --optest "${optest}" \
                --output-dir "${log_dir}/operator-compare/perchannel" \
                --scale-layout perchannel \
                --warmup 20 --iters 200 --outer-repeats 5
        fi

        if [[ "${scope}" != dense ]]; then
            run_logged sm120_w8a8_block128_operator_compare \
                "${vllm_python}" test/w8a8/operator_performance_compare.py \
                --optest "${optest}" \
                --output-dir "${log_dir}/operator-compare/block128" \
                --scale-layout block128 \
                --warmup 20 --iters 200 --outer-repeats 5
        fi
    fi
}

require_model() {
    [[ -n "${model}" ]] || { echo "W8A8_MODEL is required" >&2; exit 2; }
}

run_forward() {
    require_model
    run_logged forward_check env FASTLLM_CUDA_W8A8=1 \
        FASTLLM_CUDA_W8A8_STRICT=1 FASTLLM_CUDA_W8A8_TRACE=1 \
        FASTLLM_SAMPLING_TOP1_TRACE=1 \
        W8A8_VLLM_PYTHON="${vllm_python}" \
        "${vllm_python}" test/w8a8/forward_check_vllm.py \
        --model "${model}" --tokens 8 --top-logprobs 10 \
        --vllm-python "${vllm_python}" \
        --max-model-len 512 --gpu-memory-utilization 0.90 \
        --flm-dtype auto --flm-atype bfloat16 --flm-device cuda \
        "${attention_backend_args[@]}" \
        --result-dir "${log_dir}/forward-results"
}

run_attention_layout_matrix() {
    local case_runner=$1 layout_backend layout backend status=0
    for layout_backend in "${attention_layout_matrix[@]}"; do
        layout=${layout_backend%%:*}
        backend=${layout_backend#*:}
        if ! "${case_runner}" "${layout}" "${backend}"; then
            status=1
        fi
    done
    return "${status}"
}

run_forward_attention_layout_case() {
    local layout=$1 backend=$2 strict_args=()
    if [[ "${backend}" != auto ]]; then
        strict_args+=(--flm-attention-backend-strict)
    fi

    run_logged "forward_single_request_${layout}_${backend}" \
        env FASTLLM_CUDA_W8A8=1 FASTLLM_CUDA_W8A8_STRICT=1 \
        FASTLLM_CUDA_W8A8_TRACE=1 FASTLLM_SAMPLING_TOP1_TRACE=1 \
        W8A8_VLLM_PYTHON="${vllm_python}" \
        "${vllm_python}" test/w8a8/forward_check_vllm.py \
        --model "${model}" --tokens 8 --top-logprobs 10 \
        --vllm-python "${vllm_python}" \
        --max-model-len 512 --gpu-memory-utilization 0.90 \
        --flm-dtype auto --flm-atype bfloat16 --flm-device cuda \
        --flm-kv-cache-layout "${layout}" \
        --flm-attention-backend "${backend}" \
        "${strict_args[@]}" \
        --result-dir "${log_dir}/forward-${layout}-${backend}"
}

run_attention_layout_forward() {
    require_model
    # forward_check只发起一个正式请求。Paged组合用于回归Llama单请求是否
    # 与多请求共用新版ForwardBatch，而不是静默落回旧Continuous路径。
    run_attention_layout_matrix run_forward_attention_layout_case
}

run_model_performance() {
    require_model
    run_logged model_performance_compare env FASTLLM_CUDA_W8A8=1 \
        FASTLLM_CUDA_W8A8_STRICT=1 \
        "${vllm_python}" test/w8a8/model_performance_compare.py \
        --model "${model}" --result-dir "${log_dir}/model-compare" \
        --vllm-python "${vllm_python}" --fastllm-python "${vllm_python}" \
        --flm-dtype auto --flm-atype bfloat16 --flm-device cuda \
        "${attention_backend_args[@]}" \
        --batch-sizes 1,2,4,8,16,32 --input-tokens 512 \
        --output-tokens 64 --warmup 1 --repeats 5 \
        --max-model-len 8192 --gpu-memory-utilization 0.90
}

run_model_performance_attention_layout_case() {
    local layout=$1 backend=$2 strict_args=()
    local result_root="${log_dir}/model-performance-attention-layouts"
    if [[ "${backend}" != auto ]]; then
        strict_args+=(--flm-attention-backend-strict)
    fi

    run_logged "model_performance_${layout}_${backend}" \
        env FASTLLM_CUDA_W8A8=1 FASTLLM_CUDA_W8A8_STRICT=1 \
        "${vllm_python}" test/w8a8/model_performance_compare.py \
        --model "${model}" \
        --result-dir "${result_root}/${layout}-${backend}" \
        --vllm-python "${vllm_python}" --fastllm-python "${vllm_python}" \
        --flm-dtype auto --flm-atype bfloat16 --flm-device cuda \
        --flm-kv-cache-layout "${layout}" \
        --flm-attention-backend "${backend}" \
        "${strict_args[@]}" \
        --batch-sizes 1,2,4,8,16,32 --input-tokens 512 \
        --output-tokens 64 --warmup 1 --repeats 5 \
        --max-model-len 8192 --gpu-memory-utilization 0.90
}

run_model_performance_attention_layouts() {
    require_model
    local result_root="${log_dir}/model-performance-attention-layouts" status=0
    run_attention_layout_matrix run_model_performance_attention_layout_case || status=$?
    "${vllm_python}" test/w8a8/attention_layout_matrix_report.py \
        --kind model-performance --result-dir "${result_root}"
    return "${status}"
}

run_decode_nsys_attention_layout_case() {
    local layout=$1 backend=$2 strict_args=() optional_args=()
    local result_root="${log_dir}/decode-nsys-attention-layouts"
    if [[ "${backend}" != auto ]]; then
        strict_args+=(--flm-attention-backend-strict)
    fi
    if [[ "${W8A8_NSYS_CPU_TRACE:-1}" == 1 ]]; then
        optional_args+=(--cpu-trace)
    fi
    if [[ "${W8A8_KEEP_NSYS_REPORTS:-0}" == 1 ]]; then
        optional_args+=(--keep-nsys-reports)
    fi

    run_logged "decode_nsys_${layout}_${backend}" \
        env FASTLLM_CUDA_W8A8=1 FASTLLM_CUDA_W8A8_STRICT=1 \
        VLLM_USE_FLASHINFER_SAMPLER=0 \
        "${vllm_python}" test/w8a8/decode_nsys_compare.py \
        --model "${model}" \
        --result-dir "${result_root}/${layout}-${backend}" \
        --backends fastllm,vllm --batch-sizes 1,2,4,8,16,32 \
        --prompt-tokens 512 --output-tokens 64 --warmup-output-tokens 8 \
        --prefix-cache-padding-tokens 128 \
        --vllm-python "${vllm_python}" --fastllm-python "${vllm_python}" \
        --flm-dtype auto --flm-atype bfloat16 --flm-device cuda \
        --flm-kv-cache-layout "${layout}" \
        --flm-attention-backend "${backend}" \
        "${strict_args[@]}" "${optional_args[@]}" \
        --max-model-len 8192 --gpu-memory-utilization 0.90 \
        --port 18082 --startup-timeout 1200 --request-timeout 3600 \
        --nsys nsys --top-kernels 20
}

run_decode_nsys_attention_layouts() {
    require_model
    local result_root="${log_dir}/decode-nsys-attention-layouts" status=0
    run_attention_layout_matrix run_decode_nsys_attention_layout_case || status=$?
    "${vllm_python}" test/w8a8/attention_layout_matrix_report.py \
        --kind decode-nsys --result-dir "${result_root}"
    return "${status}"
}

case "${suite}" in
    build) run_build ;;
    ops-dense-functional) run_ops_functional ;;
    ops-dense-performance) run_ops_performance ;;
    ops-dense) run_ops_functional; run_ops_performance ;;
    ops-block128-functional) run_ops_functional ;;
    ops-block128-performance) run_ops_performance ;;
    ops-block128) run_ops_functional; run_ops_performance ;;
    ops-int8-symmetric-functional) run_ops_functional ;;
    ops-int8-symmetric-performance) run_ops_performance ;;
    ops-int8-symmetric) run_ops_functional; run_ops_performance ;;
    ops-int8-azp-functional) run_ops_functional ;;
    ops-int8-azp-performance) run_ops_performance ;;
    ops-int8-azp) run_ops_functional; run_ops_performance ;;
    ops-moe-functional) run_ops_functional ;;
    ops-moe-performance) run_ops_performance ;;
    ops-moe) run_ops_functional; run_ops_performance ;;
    ops-functional) run_ops_functional ;;
    ops-performance) run_ops_performance ;;
    ops-compare) run_ops_performance ;;
    ops) run_ops_functional; run_ops_performance ;;
    forward) run_forward ;;
    forward-attention-layouts) run_attention_layout_forward ;;
    model-performance) run_model_performance ;;
    model-performance-attention-layouts) run_model_performance_attention_layouts ;;
    decode-nsys-attention-layouts) run_decode_nsys_attention_layouts ;;
    all) run_ops_functional; run_ops_performance; run_forward; run_model_performance ;;
    *)
        echo "usage: $0 {build|ops-dense-functional|ops-dense-performance|ops-dense|ops-block128-functional|ops-block128-performance|ops-block128|ops-int8-symmetric-functional|ops-int8-symmetric-performance|ops-int8-symmetric|ops-int8-azp-functional|ops-int8-azp-performance|ops-int8-azp|ops-moe-functional|ops-moe-performance|ops-moe|ops-functional|ops-performance|ops-compare|ops|forward|forward-attention-layouts|model-performance|model-performance-attention-layouts|decode-nsys-attention-layouts|all}" >&2
        exit 2
        ;;
esac

printf 'Logs: %s\n' "${log_dir}"
