#include "models/minimax_h3_image/model.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace fastllm::minimax_h3_image;
namespace {
std::vector<float> read(const std::string &path) {
    if (path.empty())
        return {};
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    auto n = f.tellg();
    if (!f || n <= 0 || n % 4)
        throw std::runtime_error("invalid float fixture: " + path);
    std::vector<float> out(size_t(n) / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char *>(out.data()), n);
    if (!f)
        throw std::runtime_error("truncated fixture: " + path);
    return out;
}
void write(const std::string &path, const void *data, size_t bytes) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(data), bytes);
    if (!f)
        throw std::runtime_error("cannot write " + path);
}
} // namespace
int main(int argc, char **argv) {
    try {
        Options options;
        std::string model, output = "image.png", textPath, noisePath, tracePath, vaePath, latentPath,
                           encodePath, vaeType = "bf16";
        int device = 0;
        std::vector<int> devices;
        std::string parallelMode = "auto";
        auto reduction = fastllm::nomas::RowReduction::Activation;
        double gpuMemoryLimit = 0, hostMemoryLimit = 0;
        bool skipDecode = false, traceBlocks = false;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            auto value = [&]() -> std::string {
                if (++i >= argc)
                    throw std::invalid_argument("missing value for " + arg);
                return argv[i];
            };
            if (arg == "--model")
                model = value();
            else if (arg == "--prompt")
                options.prompt = value();
            else if (arg == "--image")
                options.images.push_back(readImage(value()));
            else if (arg == "--min-generation-area")
                options.minGenerationArea = std::stoi(value());
            else if (arg == "--output")
                output = value();
            else if (arg == "--width")
                options.width = std::stoi(value());
            else if (arg == "--height")
                options.height = std::stoi(value());
            else if (arg == "--seed")
                options.seed = std::stoull(value());
            else if (arg == "--device")
                device = std::stoi(value());
            else if (arg == "--devices")
                devices = fastllm::nomas::parseParallelDevices(value());
            else if (arg == "--parallel-mode")
                parallelMode = value();
            else if (arg == "--tp-reduction") {
                auto mode = value();
                if (mode != "fp32" && mode != "activation")
                    throw std::invalid_argument("TP reduction must be fp32 or activation");
                reduction = mode == "fp32" ? fastllm::nomas::RowReduction::FP32
                                           : fastllm::nomas::RowReduction::Activation;
            } else if (arg == "--gpu-memory-limit")
                gpuMemoryLimit = fastllm::minimax_h3::parseGpuMemoryLimit(value());
            else if (arg == "--host-memory-limit")
                hostMemoryLimit = fastllm::minimax_h3::parseHostMemoryLimit(value());
            else if (arg == "--vae-precision")
                vaeType = value();
            else if (arg == "--text-in")
                textPath = value();
            else if (arg == "--noise-in")
                noisePath = value();
            else if (arg == "--trace")
                tracePath = value();
            else if (arg == "--trace-blocks")
                traceBlocks = true;
            else if (arg == "--skip-decode")
                skipDecode = true;
            else if (arg == "--vae-tile")
                vaePath = value();
            else if (arg == "--encode-image")
                encodePath = value();
            else if (arg == "--decode-latents")
                latentPath = value();
            else if (arg == "--help") {
                std::cout
                    << "minimax_h3_image_generate --model DIR --prompt TEXT --output image.png\n"
                       "  [--width 2048 --height 2048 --seed 42 --device 0 --vae-precision bf16|fp32]\n"
                       "  --model accepts a complete directory or FP8/INT8/NVFP4 safetensors file (auto-detected).\n"
                       "  [--gpu-memory-limit GiB --host-memory-limit GiB] (0: automatic)\n"
                       "  [--devices 0,1 --parallel-mode auto|tp|sp --tp-reduction fp32|activation]\n"
                       "Reference editing: --image FILE [--image FILE ...], up to 16 PNG/JPEG/WebP files.\n"
                       "  --min-generation-area 4194304 generates small outputs on a 2K canvas.\n"
                       "Diagnostics: --text-in F32 --noise-in F32 --trace DIR --trace-blocks --skip-decode\n"
                       "  --vae-tile F32 decodes HWC unnormalized latents; width/height are output pixels.\n"
                       "  --decode-latents F32 decodes normalized packed latents to PNG.\n"
                       "  --encode-image FILE writes sampled normalized CHW reference latents as FP32.\n";
                return 0;
            } else
                throw std::invalid_argument("unknown option: " + arg);
        }
        if (model.empty())
            throw std::invalid_argument("--model is required");
        if (!devices.empty())
            device = devices.front();
        const auto dtype = fastllm::minimax_h3::parseVideoPrecision(vaeType);
        if (dtype == DType::FP16)
            throw std::invalid_argument("image VAE supports bf16 or fp32");
        Runtime full(device, true);
        Trace trace;
        if (!tracePath.empty()) {
            std::filesystem::create_directories(tracePath);
            trace = [&](const std::string &name, const Tensor &tensor) {
                if (!traceBlocks && name.find(".block.") != std::string::npos)
                    return;
                auto data = full.download(tensor);
                write(tracePath + "/" + name + ".f32", data.data(), data.size() * 4);
                std::ofstream shape(tracePath + "/" + name + ".shape");
                shape << tensor.rows << ' ' << tensor.cols << '\n';
            };
        }
        auto progress = [](const std::string &stage, int i, int n) {
            std::cerr << stage << ' ' << i << '/' << n << '\n';
        };
        auto start = std::chrono::steady_clock::now();
        Image result;
        const bool diagnostic = !encodePath.empty() || !vaePath.empty() || !latentPath.empty();
        if (diagnostic)
            model = resolveModelSource(model).directory;
        if (diagnostic && devices.size() > 1)
            throw std::invalid_argument("standalone VAE diagnostics use one device");
        auto memory =
            diagnostic ? std::make_unique<MemoryPolicy>(full, gpuMemoryLimit, hostMemoryLimit) : nullptr;
        auto scope = memory && memory->pool
                         ? std::make_unique<fastllm::nomas::TensorPool::Scope>(*memory->pool)
                         : nullptr;
        if (!encodePath.empty()) {
            Runtime compute(device, dtype);
            VAE vae(compute, full, model);
            vae.setMemoryPolicy(*memory);
            std::mt19937 generator(uint32_t(options.seed));
            auto latent = vae.encode(readImage(encodePath), generator, progress, trace);
            write(output, latent.values.data(), latent.values.size() * 4);
        } else if (!vaePath.empty()) {
            Runtime compute(device, dtype);
            VAE vae(compute, full, model);
            vae.setMemoryPolicy(*memory);
            auto latent = read(vaePath);
            int h = options.height / 16, w = options.width / 16;
            if (options.height % 16 || options.width % 16 || h < 1 || w < 1 ||
                latent.size() != size_t(h) * w * 24)
                throw std::invalid_argument("VAE fixture size mismatch");
            auto pixels = full.download(vae.decodeTile(full.upload(latent.data(), h * w, 24), h, w, trace));
            write(output, pixels.data(), pixels.size() * 4);
        } else if (!latentPath.empty()) {
            Runtime compute(device, dtype);
            VAE vae(compute, full, model);
            vae.setMemoryPolicy(*memory);
            result = vae.decode(read(latentPath), options.height, options.width, progress);
            auto png = fastllm::qwen_image::encodePNG(result);
            write(output, png.data(), png.size());
        } else {
            auto text = read(textPath), noise = read(noisePath);
            Pipeline pipeline(model, device, dtype, gpuMemoryLimit, hostMemoryLimit, devices, reduction, parallelMode);
            result = pipeline.generate(options, progress, trace, textPath.empty() ? nullptr : &text,
                                       noisePath.empty() ? nullptr : &noise, !skipDecode);
            if (!skipDecode) {
                auto png = fastllm::qwen_image::encodePNG(result);
                write(output, png.data(), png.size());
            }
        }
        std::cerr << "Completed in "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
                  << " seconds\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
