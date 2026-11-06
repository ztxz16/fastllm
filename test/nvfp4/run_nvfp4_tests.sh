#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
log_dir=${NVFP4_LOG_DIR:-"${repo_dir}/test/nvfp4/logs/$(date -u +%Y%m%dT%H%M%SZ)"}
suite=${1:-ops}
model_path=${NVFP4_MODEL:-}
gpu_profile=${NVFP4_GPU_PROFILE:-auto}
cuda_arch=${NVFP4_CUDA_ARCH:-auto}
optest=${NVFP4_OPTEST:-"${repo_dir}/optest"}
nvcc_path=${NVFP4_NVCC:-$(command -v nvcc || true)}
attention_backend=${NVFP4_ATTENTION_BACKEND:-auto}
kv_cache_layout=${NVFP4_KV_CACHE_LAYOUT:-auto}
mkdir -p "${log_dir}"
attention_backend_args=(
    --flm-kv-cache-layout "${kv_cache_layout}"
    --flm-attention-backend "${attention_backend}"
)
if [[ "${NVFP4_ATTENTION_BACKEND_STRICT:-0}" == 1 ]]; then
    attention_backend_args+=(--flm-attention-backend-strict)
fi

resolve_gpu_profile() {
    local compute_cap major minor detected_arch
    compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed -n '1p')
    major=${compute_cap%%.*}
    minor=${compute_cap#*.}
    if [[ ! "${major}" =~ ^[0-9]+$ || ! "${minor}" =~ ^[0-9]+$ ]]; then
        echo "cannot detect GPU compute capability: ${compute_cap}" >&2
        exit 2
    fi
    detected_arch=$((10#${major} * 10 + 10#${minor}))
    if [[ "${gpu_profile}" == auto ]]; then
        if ((detected_arch >= 120 && detected_arch < 130)); then
            gpu_profile=sm120
        elif ((detected_arch >= 100 && detected_arch < 120)); then
            gpu_profile=sm100
        else
            gpu_profile=preblackwell
        fi
    fi
    if [[ "${cuda_arch}" == auto ]]; then
        cuda_arch=${detected_arch}
    fi
    printf 'GPU profile: %s; CUDA_ARCH: %s; compute capability: %s\n' \
        "${gpu_profile}" "${cuda_arch}" "${compute_cap}"
}

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

run_build() {
    local -a install_args
    if [[ -z "${nvcc_path}" ]]; then
        echo "nvcc not found; set NVFP4_NVCC=/path/to/nvcc" >&2
        exit 2
    fi
    run_logged environment bash -lc \
        'git rev-parse HEAD; git status --short --branch; nvidia-smi'
    run_logged nvcc_version "${nvcc_path}" --version
    install_args=(-DUSE_CUDA=ON -DUNIT_TEST=ON
        "-DCMAKE_CUDA_COMPILER=${nvcc_path}"
        "-DCUDA_ARCH=${cuda_arch}")
    run_logged install bash "${repo_dir}/install.sh" "${install_args[@]}"
}

run_ops_functional() {
    if [[ "${gpu_profile}" == sm100 || "${gpu_profile}" == sm120 ]]; then
        run_logged op_dense_marlin_w4a16_explicit env \
            FASTLLM_CUDA_NVFP4_TRACE=1 \
            "${optest}" --op linear_nvfp4 --device cuda:0 \
            --param batch=8 --param in=1024 --param out=1024 \
            --param bias=0 --param input_type=fp16 \
            --param quant_scheme=w4a16 \
            --warmup 0 --iters 1 --atol 0.05 --rtol 0.05
        run_logged op_dense_w4a4_marlin_fallback env \
            FASTLLM_CUDA_NVFP4_TRACE=1 \
            FASTLLM_CUDA_NVFP4_W4A4=0 \
            "${optest}" --op linear_nvfp4 --device cuda:0 \
            --param batch=8 --param in=1024 --param out=1024 \
            --param bias=0 --param input_type=fp16 \
            --param quant_scheme=w4a4 --param check_marlin_fallback=1 \
            --warmup 0 --iters 1 --atol 0.05 --rtol 0.05
        run_logged op_dense_w4a4_decode_padding_bias env \
            FASTLLM_CUDA_NVFP4_TRACE=1 \
            FASTLLM_CUDA_NVFP4_W4A4=1 \
            FASTLLM_CUDA_NVFP4_W4A4_STRICT=1 \
            "${optest}" --op linear_nvfp4 --device cuda:0 \
            --param batch=1 --param in=1008 --param out=1000 \
            --param bias=1 --param input_type=bf16 \
            --param check_release=1 --param check_fixed_backend=1 \
            --param check_fusion_separation=1 \
            --warmup 0 --iters 1 --atol 0.20 --rtol 0.20
        run_logged op_dense_w4a4_warmup_fallback env \
            FASTLLM_CUDA_NVFP4_TRACE=1 \
            FASTLLM_CUDA_NVFP4_W4A4=1 \
            "${optest}" --op linear_nvfp4 --device cuda:0 \
            --param batch=1 --param in=1008 --param out=1000 \
            --param bias=1 --param input_type=bf16 \
            --param check_warmup_fallback=1 \
            --warmup 0 --iters 1 --atol 0.20 --rtol 0.20
        run_logged op_dense_w4a4_aligned_fp16 env \
            FASTLLM_CUDA_NVFP4_TRACE=1 \
            FASTLLM_CUDA_NVFP4_W4A4=1 \
            FASTLLM_CUDA_NVFP4_W4A4_STRICT=1 \
            "${optest}" --op linear_nvfp4 --device cuda:0 \
            --param batch=128 --param in=1024 --param out=1024 \
            --param bias=0 --param input_type=fp16 \
            --warmup 0 --iters 1 --atol 0.20 --rtol 0.20
        run_logged op_dense_w4a4_padding_n_only env \
            FASTLLM_CUDA_NVFP4_TRACE=1 \
            FASTLLM_CUDA_NVFP4_W4A4=1 \
            FASTLLM_CUDA_NVFP4_W4A4_STRICT=1 \
            "${optest}" --op linear_nvfp4 --device cuda:0 \
            --param batch=128 --param in=1024 --param out=1000 \
            --param bias=0 --param input_type=bf16 \
            --warmup 0 --iters 1 --atol 0.20 --rtol 0.20
        run_logged op_dense_w4a4_padding_k_only env \
            FASTLLM_CUDA_NVFP4_TRACE=1 \
            FASTLLM_CUDA_NVFP4_W4A4=1 \
            FASTLLM_CUDA_NVFP4_W4A4_STRICT=1 \
            "${optest}" --op linear_nvfp4 --device cuda:0 \
            --param batch=128 --param in=1008 --param out=1024 \
            --param bias=0 --param input_type=bf16 \
            --warmup 0 --iters 1 --atol 0.20 --rtol 0.20
        if [[ $(nvidia-smi -L 2>/dev/null | wc -l) -ge 2 ]]; then
            run_logged op_dense_w4a4_device_move env \
                FASTLLM_CUDA_NVFP4_TRACE=1 \
                FASTLLM_CUDA_NVFP4_W4A4=1 \
                FASTLLM_CUDA_NVFP4_W4A4_STRICT=1 \
                "${optest}" --op linear_nvfp4 --device cuda:0 \
                --param batch=1 --param in=1024 --param out=1024 \
                --param bias=0 --param input_type=bf16 \
                --param check_release=1 --param check_device_move=1 \
                --warmup 0 --iters 1 --atol 0.20 --rtol 0.20
        fi
        local quant_k
        for quant_k in 4096 7168 14336; do
            run_logged "op_swiglu_fp4_quant_bf16_k${quant_k}" "${optest}" \
                --op nvfp4_swiglu_quant --device cuda:0 \
                --param rows=32 --param hidden="${quant_k}" --param input_type=bf16 \
                --warmup 0 --iters 1 --atol 0.20 --rtol 0.20
        done
        run_logged op_swiglu_fp4_quant_fp16 "${optest}" \
            --op nvfp4_swiglu_quant --device cuda:0 \
            --param rows=32 --param hidden=1024 --param input_type=fp16 \
            --warmup 0 --iters 1 --atol 0.20 --rtol 0.20
        local moe_m moe_topk moe_experts
        for moe_m in 1 2 64 224; do
            for moe_topk in 1 8; do
                for moe_experts in 40 64; do
                    run_logged "op_grouped_moe_w4a4_check_m${moe_m}_k${moe_topk}_e${moe_experts}" env \
                        FASTLLM_CUDA_NVFP4_TRACE=1 \
                        FASTLLM_CUDA_MOE_NVFP4_W4A4=1 \
                        FASTLLM_CUDA_MOE_NVFP4_W4A4_STRICT=1 \
                        "${optest}" --op mergemoe_fp8 --device cuda:0 \
                        --param weight_type=nvfp4 --param path=check_nvfp4 \
                        --param batch="${moe_m}" --param topk="${moe_topk}" \
                        --param experts="${moe_experts}" \
                        --param hidden=256 --param inter=256 --param input_type=bf16 \
                        --warmup 0 --iters 1
                done
            done
        done
        run_logged "op_grouped_moe_w4a4_check_geglu_m1_k1_e40" env \
            FASTLLM_CUDA_NVFP4_TRACE=1 \
            FASTLLM_CUDA_MOE_NVFP4_W4A4=1 \
            FASTLLM_CUDA_MOE_NVFP4_W4A4_STRICT=1 \
            "${optest}" --op mergemoe_fp8 --device cuda:0 \
            --param weight_type=nvfp4 --param path=check_nvfp4 \
            --param gate_type=geglu --param batch=1 --param topk=1 \
            --param experts=40 --param hidden=256 --param inter=256 \
            --param input_type=bf16 --warmup 0 --iters 1
        run_logged "op_grouped_moe_w4a4_check_shared_m2_k1_e40" env \
            FASTLLM_CUDA_NVFP4_TRACE=1 \
            FASTLLM_CUDA_MOE_NVFP4_W4A4=1 \
            FASTLLM_CUDA_MOE_NVFP4_W4A4_STRICT=1 \
            "${optest}" --op mergemoe_fp8 --device cuda:0 \
            --param weight_type=nvfp4 --param path=check_nvfp4 \
            --param shared_expert=1 --param batch=2 --param topk=1 \
            --param experts=40 --param hidden=256 --param inter=256 \
            --param input_type=bf16 --warmup 0 --iters 1
        run_logged "op_grouped_moe_w4a4_lifecycle_m2_k1_e40" env \
            FASTLLM_CUDA_NVFP4_TRACE=1 \
            FASTLLM_CUDA_MOE_NVFP4_W4A4=1 \
            FASTLLM_CUDA_MOE_NVFP4_W4A4_STRICT=0 \
            "${optest}" --op mergemoe_fp8 --device cuda:0 \
            --param weight_type=nvfp4 --param path=check_nvfp4_lifecycle \
            --param batch=2 --param topk=1 --param experts=40 \
            --param hidden=256 --param inter=256 --param input_type=bf16 \
            --warmup 0 --iters 1
    else
        run_logged op_dense_marlin_w4a16_check env \
            FASTLLM_CUDA_NVFP4_TRACE=1 \
            "${optest}" --op linear_nvfp4 --device cuda:0 \
            --param batch=8 --param in=1024 --param out=1024 \
            --param bias=0 --param input_type=fp16 \
            --param quant_scheme=w4a16 --warmup 0 --iters 1 \
            --atol 0.05 --rtol 0.05
        run_logged op_dense_marlin_w4a16_decode env \
            FASTLLM_CUDA_NVFP4_TRACE=1 \
            "${optest}" --op linear_nvfp4 --device cuda:0 \
            --param batch=1 --param in=1024 --param out=1024 \
            --param bias=0 --param input_type=fp16 \
            --param quant_scheme=w4a16 --warmup 0 --iters 1 \
            --atol 0.05 --rtol 0.05
    fi
}

run_ops_performance() {
    if [[ "${gpu_profile}" == sm100 || "${gpu_profile}" == sm120 ]]; then
        local dense_m quant_k moe_m moe_topk moe_experts
        # 覆盖激活scale的128行边界，以及SM120在M=256处的tile分派边界。
        for dense_m in 1 16 64 127 128 129 255 256 257 512 1024; do
            run_logged "op_dense_w4a4_perf_m${dense_m}" env \
                FASTLLM_CUDA_NVFP4_W4A4=1 \
                FASTLLM_CUDA_NVFP4_W4A4_STRICT=1 \
                "${optest}" --op linear_nvfp4 --device cuda:0 \
                --param batch="${dense_m}" --param in=4096 --param out=4096 \
                --param bias=0 --param input_type=bf16 --param check_release=1 \
                --param performance_only=1 \
                --warmup 5 --iters 20 --atol 0.20 --rtol 0.20
        done
        run_logged op_dense_w4a4_perf_fp16_m128 env \
            FASTLLM_CUDA_NVFP4_W4A4=1 \
            FASTLLM_CUDA_NVFP4_W4A4_STRICT=1 \
            "${optest}" --op linear_nvfp4 --device cuda:0 \
            --param batch=128 --param in=4096 --param out=4096 \
            --param bias=0 --param input_type=fp16 --param performance_only=1 \
            --warmup 5 --iters 20 --atol 0.20 --rtol 0.20
        local dense_shape
        for dense_shape in 4096:6144 4096:24576 12288:4096; do
            local dense_k=${dense_shape%%:*}
            local dense_n=${dense_shape#*:}
            run_logged "op_dense_w4a4_perf_m128_k${dense_k}_n${dense_n}" env \
                FASTLLM_CUDA_NVFP4_W4A4=1 \
                FASTLLM_CUDA_NVFP4_W4A4_STRICT=1 \
                "${optest}" --op linear_nvfp4 --device cuda:0 \
                --param batch=128 --param in="${dense_k}" --param out="${dense_n}" \
                --param bias=0 --param input_type=bf16 --param performance_only=1 \
                --warmup 5 --iters 20 --atol 0.20 --rtol 0.20
        done
        for quant_k in 4096 7168 14336; do
            run_logged "op_swiglu_fp4_quant_perf_k${quant_k}" "${optest}" \
                --op nvfp4_swiglu_quant --device cuda:0 \
                --param rows=32 --param hidden="${quant_k}" --param input_type=bf16 \
                --warmup 20 --iters 200 --atol 0.20 --rtol 0.20
        done
        for moe_m in 1 2 64 224; do
            for moe_topk in 1 8; do
                for moe_experts in 40 64; do
                    run_logged "op_grouped_moe_w4a4_perf_m${moe_m}_k${moe_topk}_e${moe_experts}" env \
                        FASTLLM_CUDA_MOE_NVFP4_W4A4=1 \
                        FASTLLM_CUDA_MOE_NVFP4_W4A4_STRICT=1 \
                        "${optest}" --op mergemoe_fp8 --device cuda:0 \
                        --param weight_type=nvfp4 --param path=operator \
                        --param batch="${moe_m}" --param topk="${moe_topk}" \
                        --param experts="${moe_experts}" \
                        --param hidden=256 --param inter=256 --param input_type=bf16 \
                        --warmup 20 --iters 200
                done
            done
        done
    else
        run_logged op_dense_marlin_w4a16_perf env \
            "${optest}" --op linear_nvfp4 --device cuda:0 \
            --param batch=8 --param in=1024 --param out=1024 \
            --param bias=0 --param input_type=fp16 \
            --param quant_scheme=w4a16 --warmup 20 --iters 200 \
            --atol 0.05 --rtol 0.05
    fi
    run_logged ops_performance_report python test/nvfp4/performance_report.py \
        --log-dir "${log_dir}" \
        --output-prefix "${log_dir}/ops-performance"
}

run_ops() {
    run_ops_functional
    run_ops_performance
}

run_ops_compare() {
    if [[ "${gpu_profile}" != sm100 && "${gpu_profile}" != sm120 ]]; then
        echo "NVFP4 W4A4 operator comparison requires SM100 or SM120." >&2
        exit 2
    fi
    # 保持原FastLLM算子性能测试不变；完成后由独立脚本分进程运行
    # 相同shape的vLLM算子，并汇总两边结果。
    run_ops_performance
    run_logged operator_performance_compare python \
        test/nvfp4/operator_performance_compare.py \
        --fastllm-log-dir "${log_dir}" \
        --output-prefix "${log_dir}/operator-performance-compare"
}

require_model() {
    if [[ -z "${model_path}" ]]; then
        echo "NVFP4_MODEL must point to a downloaded NVFP4 Hugging Face model." >&2
        exit 2
    fi
}

run_forward() {
    require_model
    run_logged forward_check_vllm env FASTLLM_CUDA_NVFP4_TRACE=1 \
        python test/nvfp4/forward_check_vllm.py --model "${model_path}" \
        --tokens 8 --flm-dtype auto --flm-atype bfloat16 \
        --flm-device cuda "${attention_backend_args[@]}" \
        --result-dir "${log_dir}/forward-vllm-results"
}

run_model() {
    require_model
    # 独立脚本只生成一次固定token ID；两个框架分时加载模型并直接消费
    # 同一组token，避免Chat Template差异和同时占用显存。
    run_logged model_performance_compare env -u FASTLLM_CUDA_NVFP4_TRACE \
        python test/nvfp4/model_performance_compare.py \
        --model "${model_path}" \
        --result-dir "${log_dir}/model-performance-results" \
        "${attention_backend_args[@]}" \
        --batch-sizes 1,2,4,8,16,32 --input-tokens 512 \
        --output-tokens 64 --warmup 1 --repeats 5
}

run_swiglu_versions() {
    local vllm_python=${NVFP4_VLLM_PYTHON:-python3}
    run_logged swiglu_versions_compare env \
        NVFP4_VLLM_PYTHON="${vllm_python}" \
        python3 test/benchmark/operator_compare/operator_benchmark.py \
        --config test/nvfp4/swiglu_versions_compare.json \
        --output-prefix "${log_dir}/swiglu-versions-compare"
}

resolve_gpu_profile

case "${suite}" in
    build) run_build ;;
    ops-functional) run_ops_functional ;;
    ops-performance) run_ops_performance ;;
    ops-compare|operator-performance) run_ops_compare ;;
    ops) run_ops ;;
    forward) run_forward ;;
    model-performance|model) run_model ;;
    swiglu-versions|swiglu-compare) run_swiglu_versions ;;
    all) run_build; run_ops_functional; run_ops_compare; run_forward; run_model ;;
    *) echo "usage: $0 {build|ops-functional|ops-performance|ops-compare|ops|forward|model-performance|swiglu-versions|all}; NVFP4_GPU_PROFILE=auto|sm100|sm120|preblackwell" >&2; exit 2 ;;
esac

printf 'Logs: %s\n' "${log_dir}"
