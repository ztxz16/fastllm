#pragma once

// HY-Nomas continuous-tensor inference. This API deliberately has no dependency
// on a Python tensor library or on the token sampling / LLM KV-cache APIs.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace fastllm { namespace nomas {

// An opt-in allocation pool for a serialized inference pipeline. A scope
// affects only allocations on its thread/device; other models keep the default.
class TensorPool {
public:
    explicit TensorPool(int device);
    class Scope {
    public:
        explicit Scope(const TensorPool &);
        ~Scope();
        Scope(const Scope &)=delete;
        Scope &operator=(const Scope &)=delete;
    private:
        std::shared_ptr<void> previous_;
    };
    // Call after synchronizing the inference stream.
    size_t reusableBytes() const;
    void trim();
private:
    std::shared_ptr<void> pool_;
};
enum class DType { BF16, FP16, FP32 };

struct Tensor {
    std::shared_ptr<void> storage;
    uint16_t *data = nullptr;
    int rows = 0, cols = 0;
    DType dtype = DType::BF16;
    Tensor() = default;
    Tensor(int rows, int cols, bool fp32 = false);
    Tensor(int rows, int cols, DType dtype);
    bool isFloat32() const { return dtype == DType::FP32; }
    size_t size() const { return size_t(rows) * cols; }
    size_t bytes() const { return size() * (isFloat32() ? 4 : 2); }
    Tensor rowSlice(int first, int count) const;
};

struct RopeTable { Tensor cos, sin; };

class Runtime {
public:
    explicit Runtime(int device);
    Runtime(int device, bool fp32);
    Runtime(int device, DType dtype);
    void activate();
    bool isFloat32() const { return dtype_ == DType::FP32; }
    DType dtype() const { return dtype_; }
    ~Runtime();
    Runtime(const Runtime &) = delete;
    Runtime &operator=(const Runtime &) = delete;
    Tensor upload(const float *values, int rows, int cols);
    // Same precision returns a shared view; conversion stays on the inference stream.
    Tensor cast(const Tensor &, bool fp32);
    std::vector<float> download(const Tensor &x);
    // Unpatchify selected PixelUMM frames and quantize on-device to THWC RGB8.
    std::vector<uint8_t> downloadPixelRGB(const Tensor &x, int width, int height,
                                          int firstFrame, int frames);
    // Official per-latent-frame CUDA normal noise, already patchified. Height
    // and width are output pixels; frame indices are in latent time units.
    Tensor pixelNoise(uint64_t seed, int firstFrame, int frames, int height, int width);
    Tensor cast(const Tensor &, DType dtype);
    Tensor linear(const Tensor &x, const Tensor &weight, const Tensor *bias = nullptr);
    // Output storage must not be shared with inputs. For 16-bit biased GEMM,
    // its address must retain the cuBLASLt plan's 256-byte alignment.
    void linearInto(const Tensor &x, const Tensor &weight, Tensor &out, const Tensor *bias = nullptr);
    Tensor norm(const Tensor &x, bool rms, const Tensor *weight = nullptr,
                const Tensor *bias = nullptr, float eps = 1e-6f, int width = 0);
    Tensor activation(const Tensor &x, bool gelu = false);
    Tensor add(const Tensor &x, const Tensor &y);
    Tensor multiply(const Tensor &x, const Tensor &y);
    Tensor siluMultiply(const Tensor &gate, const Tensor &up);
    Tensor bias(const Tensor &x, const Tensor &bias);
    Tensor mean(const Tensor &x);
    Tensor repeatRows(const Tensor &x, int copies);
    Tensor pixelQKRope(const Tensor &x, const Tensor &normT, const Tensor &normHW,
                       int frames, int height, int width);
    Tensor pixelCondition(const Tensor &x, int frames, int height, int width);
    Tensor modulate(const Tensor &x, const Tensor &mod, int shift, int scale);
    Tensor residual(const Tensor &x, const Tensor &y, const Tensor &mod, int gate);
    Tensor columns(const Tensor &x, int first, int count);
    Tensor embedding(const Tensor &weight, const std::vector<int> &tokens);
    Tensor textRope(const Tensor &x);
    Tensor textRope(const Tensor &x, float theta);
    Tensor concat(const std::vector<Tensor> &xs);
    Tensor rope(const Tensor &x, const float *cos, const float *sin, int headDim);
    RopeTable prepareRope(const float *cos, const float *sin, int rows, int headDim);
    Tensor rope(const Tensor &x, const RopeTable &table);
    // mask is row-major [query_tokens, key_tokens], 1=visible. Null means full
    // attention. Scratch memory is O(query_chunk * keys), never O(queries*keys).
    Tensor attention(const Tensor &q, const Tensor &k, const Tensor &v,
                     int heads, const uint8_t *mask = nullptr, int queryChunk = 128,
                     int kvHeads = 0, bool causal = false, int tile = 0,
                     int gridHeight = 0, int gridWidth = 0);
    // FP32 attention with an explicit score scale, including pre-scaled Q/K.
    Tensor attentionScaled(const Tensor &q, const Tensor &k, const Tensor &v,
                           int heads, float scoreScale, const uint8_t *mask = nullptr,
                           int queryChunk = 128, int kvHeads = 0, bool causal = false,
                           int tile = 0, int gridHeight = 0, int gridWidth = 0);
    void synchronize();
private:
    struct LtState;
    std::unique_ptr<LtState> lt_;
    void *blas_ = nullptr;
    int device_;
    DType dtype_;
};

class Weights {
public:
    Weights(Runtime &runtime, const std::string &path);
    ~Weights();
    const Tensor &get(const std::string &name);
    const Tensor *optional(const std::string &name);
    Tensor linear(const Tensor &x, const std::string &prefix, bool separateBias = false);
    // A supplied output cannot be combined with separateBias.
    Tensor linear(const Tensor &x, const std::string &prefix, bool separateBias, Tensor *out);
    void clear();
    // Optional lossless host cache for callers that swap model stages.
    void offload();
    // Optionally retain checkpoint BF16 tensors while streaming expanded FP32
    // layers. Use at most half the currently free memory, leaving reserveBytes.
    // clear() releases both representations; disabled unless explicitly enabled.
    void enablePackedCache(size_t reserveBytes);
    // Retain immutable CPU checkpoint bytes across clear(), preferring pinned
    // memory. Capped by one quarter of available host RAM; rejected pinning
    // uses ordinary memory, exhaustion uses files. Zero releases the cache.
    void enableHostCache(size_t maxBytes);
    // Release selected projections after precomputing their schedule-dependent outputs.
    // An explicitly enabled packed cache survives this operation.
    void erasePrefix(const std::string &prefix);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct BlockResult { Tensor image, text, key, value; };
struct DiTCache { std::vector<Tensor> key, value; };

class DiT {
public:
    DiT(Runtime &runtime, const std::string &checkpoint);
    BlockResult block(int layer, const Tensor &image, const Tensor &text,
                      const Tensor &time, const float *cos, const float *sin,
                      const Tensor *historyK = nullptr, const Tensor *historyV = nullptr,
                      const uint8_t *attentionMask = nullptr);
    BlockResult block(int layer, const Tensor &image, const Tensor &text,
                      const Tensor &time, const RopeTable &rope,
                      const Tensor *historyK = nullptr, const Tensor *historyV = nullptr,
                      const uint8_t *attentionMask = nullptr);
    Tensor forward(const Tensor &latent, const Tensor &textFeatures, float timestep,
                   int frames, int height, int width, int frameOffset = 0,
                   DiTCache *cache = nullptr, bool commit = false,
                   const Tensor *camera = nullptr);
    Tensor cameraEmbedding(const Tensor &extrinsics);
    Tensor cameraBlock(int layer, const Tensor &image, const Tensor &camera, int tokensPerFrame);
    void unload() { weights_.clear(); }
private:
    Tensor timeEmbedding(float timestep, const std::string &prefix);
    Tensor refineText(const Tensor &text, float timestep);
    Runtime &rt_;
    Weights weights_;
};

class PixelDecoder {
public:
    PixelDecoder(Runtime &runtime, const std::string &checkpoint);
    Tensor block(int layer, const Tensor &x, int frames, int height, int width, int tile = 8);
    // latent is [T*H*W,32]; noise and output are patchified [T*2H*2W,3072].
    Tensor forward(const Tensor &latent, const Tensor &noise, int frames, int height, int width);
    void unload() { weights_.clear(); }
private:
    Runtime &rt_;
    Weights weights_;
};

class QwenTextEncoder {
public:
    QwenTextEncoder(Runtime &runtime, const std::string &directory);
    Tensor encode(const std::vector<int> &tokens);
    void unload() { weights_.clear(); }
private:
    Runtime &rt_;
    Weights weights_;
};

std::string normalizeNFC(const std::string &text);

class QwenTokenizer {
public:
    explicit QwenTokenizer(const std::string &directory);
    std::vector<int> encode(const std::string &text) const;
    // Official video text-encoder template, including assistant prefix.
    std::vector<int> videoPrompt(const std::string &prompt, int &crop) const;
private:
    std::unordered_map<std::string,int> vocab_, ranks_;
    std::vector<std::pair<std::string,int>> special_;
    std::string bytes_[256];
    void plain(const std::string &text, std::vector<int> &out) const;
};

using Progress = int (*)(int stage, int current, int total, void *user);
// Retain the accepted scene text separately from the text used by a camera
// continuation. There are no structured per-action text tails in this API.
struct WorldState {
    Tensor features;
    std::string prompt;
    DiTCache cache;
    std::vector<float> decoderLatents;
    std::vector<int> segmentTokens;
    std::mt19937_64 generator;
    int width=0, height=0, steps=8, frameOffset=0, maxHistorySegments=6;
    uint64_t seed=0;
};
// Packed model input: first-frame-aware resampling, local OpenCV transforms,
// official 2.5x rotation gain and three repeated identity frames at the front.
// Actions: 0 stay, 1 forward, 2 backward, 3 left, 4 right,
// 5 look left, 6 look right, 7 look up, 8 look down.
std::vector<float> cameraTrajectory(int frames, int action);
class Pipeline {
public:
    Pipeline(const std::string &modelDirectory, const std::string &textDirectory, int device);
    std::vector<uint8_t> generate(const std::string &prompt, int width, int height,
                                  int frames, uint64_t seed, int steps,
                                  Progress progress = nullptr, void *user = nullptr);
    std::unique_ptr<WorldState> createWorld(const std::string &prompt, int width, int height,
                                           uint64_t seed, int steps);
    // Actions 0..8 apply camera conditioning; 9 follows text without a camera condition.
    // Unchanged text with camera control uses the official empty continuation
    // prompt after the first segment; new text is consumed only on commit.
    std::vector<uint8_t> stepWorld(WorldState &world, int action,
                                   Progress progress = nullptr, void *user = nullptr,
                                   const std::string *prompt = nullptr);
private:
    Tensor encodeWorldPrompt(const std::string &prompt);
    Runtime rt_;
    QwenTokenizer tokenizer_;
    QwenTextEncoder text_;
    DiT dit_;
    PixelDecoder decoder_;
    Tensor cameraContinuationFeatures_;
};

}} // namespace fastllm::nomas

extern "C" {
// C ABI uses host float32 arrays. FP32/BF16 conversion and inference are native.
// Errors never escape through ctypes as C++ exceptions.
void *fastllm_nomas_create(const char *checkpoint, int device);
void fastllm_nomas_destroy(void *handle);
const char *fastllm_nomas_error();
int fastllm_nomas_block(void *handle, int layer, int imageTokens, int textTokens,
                       const float *image, const float *text, const float *time,
                       const float *cos, const float *sin,
                       int historyTokens, const float *historyK, const float *historyV,
                       const uint8_t *mask, float *outImage, float *outText,
                       float *outKey, float *outValue);
int fastllm_nomas_dit(void *handle, int frames, int height, int width, int textTokens,
                     const float *latent, const float *features, float timestep,
                     int frameOffset, int commitCache, float *output);
void *fastllm_nomas_decoder_create(const char *checkpoint, int device);
void fastllm_nomas_decoder_destroy(void *handle);
int fastllm_nomas_decoder_block(void *handle, int layer, int frames, int height,
                               int width, int tile, const float *input, float *output);
int fastllm_nomas_decode(void *handle, int frames, int height, int width,
                        const float *latent, const float *noise, float *output);
void *fastllm_nomas_text_create(const char *directory, int device);
void fastllm_nomas_text_destroy(void *handle);
int fastllm_nomas_tokenize(void *handle, const char *text, int videoTemplate,
                         int *tokens, int capacity, int *crop);
int fastllm_nomas_encode(void *handle, const int *tokens, int count, float *features);
void *fastllm_nomas_pipeline_create(const char *modelDirectory, const char *textDirectory, int device);
void fastllm_nomas_pipeline_destroy(void *handle);
int fastllm_nomas_generate(void *handle, const char *prompt, int width, int height,
                          int frames, uint64_t seed, int steps, uint8_t *rgb,
                          size_t rgbCapacity, fastllm::nomas::Progress progress, void *user);
void *fastllm_nomas_world_create(void *pipeline, const char *prompt, int width, int height, uint64_t seed, int steps);
void fastllm_nomas_world_destroy(void *pipeline, void *world);
int fastllm_nomas_world_step(void *pipeline, void *world, int action, uint8_t *rgb, size_t rgbCapacity,
                            fastllm::nomas::Progress progress, void *user);
// An optional replacement prompt applies atomically with the next segment.
// Null retains the current prompt. Cancellation leaves all world state intact.
int fastllm_nomas_world_step_prompt(void *pipeline, void *world, const char *prompt, int action,
                                   uint8_t *rgb, size_t rgbCapacity,
                                   fastllm::nomas::Progress progress, void *user);
int fastllm_nomas_camera(void *handle, int layer, int frames, int height, int width,
                        const float *image, const float *extrinsics, float *output);
int fastllm_nomas_dit_camera(void *handle, int frames, int height, int width, int textTokens,
                            const float *latent, const float *features, float timestep,
                            int frameOffset, int commitCache, const float *extrinsics, float *output);
int fastllm_nomas_camera_trajectory(int frames, int action, float *output);
}
