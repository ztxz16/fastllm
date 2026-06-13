#include "baseblock.h"
#include "fastllm.h"

#include <cstdlib>
#include <cstring>

namespace fastllm {
    namespace {
        bool EnvDefaultEnabled(const char *name) {
            const char *env = std::getenv(name);
            if (env == nullptr || env[0] == '\0') {
                return true;
            }
            return std::strcmp(env, "0") != 0 &&
                   std::strcmp(env, "false") != 0 && std::strcmp(env, "FALSE") != 0 &&
                   std::strcmp(env, "off") != 0 && std::strcmp(env, "OFF") != 0 &&
                   std::strcmp(env, "no") != 0 && std::strcmp(env, "NO") != 0;
        }

        int EnvInt(const char *name, int fallback) {
            const char *env = std::getenv(name);
            if (env == nullptr || env[0] == '\0') {
                return fallback;
            }
            char *end = nullptr;
            long value = std::strtol(env, &end, 10);
            if (end == env || value <= 0 || value > 4096) {
                return fallback;
            }
            return (int)value;
        }

        bool ShouldTrySwigluLinearAdd(Data *input, Data *gateUp, Data *down, Data *output) {
            if (!EnvDefaultEnabled("FASTLLM_CUDA_CUTLASS_LINEAR_FP8_SWIGLU_QUANT")) {
                return false;
            }
            if (input == nullptr || gateUp == nullptr || down == nullptr || output == nullptr ||
                input->dims.empty() || gateUp->dims.size() != 2 || down->dims.size() != 2 ||
                output->dims.empty() || input->dims.back() <= 0) {
                return false;
            }
            int tokens = (int)(input->Count(0) / input->dims.back());
            int minBatch = EnvInt("FASTLLM_CUDA_CUTLASS_LINEAR_FP8_MIN_BATCH", 7);
            if (tokens < minBatch) {
                return false;
            }
            int inter = down->dims[1];
            int hidden = output->dims.back();
            return (input->dataType == DataType::FLOAT16 || input->dataType == DataType::BFLOAT16) &&
                   output->dataType == input->dataType &&
                   down->dataType == DataType::FP8_E4M3 &&
                   down->blockM == 128 && down->blockK == 128 && !down->scales.empty() &&
                   gateUp->dims[0] == inter * 2 && down->dims[0] == hidden &&
                   (inter % 128) == 0 && (hidden % 128) == 0;
        }
    }

    /*
    gateUpResult = Linear(input, gateUp)
    swigluResult = Swiglu(gateUpResult)
    output += Linear(swigluResult, down)
    */
    void MLPBlock (
        Data *input, 
        Data *gateUp, Data *down, 
        Data *gateUpResult, 
        Data *swigluResult,
        Data *output
    ) {
        gateUp->tpPackType = TP_PACK_GATEUP;
        /* if (CanRunMLP()) {
            Data w3;
            Data mlpOutput;
            MLP(*input, *gateUp, *GetEmptyData(), *down, *GetEmptyData(), *gateUpResult, *swigluResult, w3, mlpOutput);
            AddTo(*output, mlpOutput);
        } else */ {
            if (ShouldTrySwigluLinearAdd(input, gateUp, down, output)) {
                Linear(*input, *gateUp, *GetEmptyData(), *gateUpResult);
                if (CanRunSwigluLinearAdd(*gateUpResult, *down, *GetEmptyData(), *output)) {
                    SwigluLinearAdd(*gateUpResult, *down, *GetEmptyData(), *swigluResult, *output);
                    return;
                }
                Swiglu(*gateUpResult, *swigluResult);
                LinearAddBlock(swigluResult, down, GetEmptyData(), gateUpResult, output);
                return;
            }
            LinearSwigluBlock(input, gateUp, GetEmptyData(), gateUpResult, swigluResult);
            LinearAddBlock(swigluResult, down, GetEmptyData(), gateUpResult, output);
        }
    }
}
