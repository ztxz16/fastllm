#include "models/minimax_h3_image/model.h"
#include "json11.hpp"
#include "httplib.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <regex>
#include <thread>
#include <tuple>

using namespace fastllm::minimax_h3_image;
using Json = json11::Json;
namespace {
constexpr size_t historyLimit = 128;
int64_t now() { return std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()); }
void reply(httplib::Response &r, const Json &j, int status = 200) {
    r.status = status;
    r.set_content(j.dump(), "application/json");
}
void fail(httplib::Response &r, const std::string &message, int status) {
    reply(r, Json::object{{"error", Json::object{{"message", message}, {"code", status}}}}, status);
}
int integer(const Json &j, const std::string &key, int fallback, int low, int high) {
    if (j[key].is_null())
        return fallback;
    double v = j[key].number_value();
    if (!j[key].is_number() || !std::isfinite(v) || std::floor(v) != v || v < low || v > high)
        throw std::invalid_argument("invalid " + key);
    return int(v);
}
std::string read(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("file not found");
    return std::string((std::istreambuf_iterator<char>(file)), {});
}
std::string base64(const std::string &bytes) {
    const char *alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    for (size_t i = 0; i < bytes.size(); i += 3) {
        uint32_t n = uint8_t(bytes[i]) << 16;
        if (i + 1 < bytes.size())
            n |= uint8_t(bytes[i + 1]) << 8;
        if (i + 2 < bytes.size())
            n |= uint8_t(bytes[i + 2]);
        out += alphabet[n >> 18];
        out += alphabet[(n >> 12) & 63];
        out += i + 1 < bytes.size() ? alphabet[(n >> 6) & 63] : '=';
        out += i + 2 < bytes.size() ? alphabet[n & 63] : '=';
    }
    return out;
}
struct Job {
    Options options;
    Json::array references;
    std::string id, status = "queued", stage = "queued", failure, format = "url";
    int current = 0, total = 0, progress = 0;
    int64_t created = now();
    bool cancelled = false, done = false;
    std::mutex mutex;
    std::condition_variable changed;
};
// Caller holds the job lock, including while persisting completed results.
Json metadata(const Job &j) {
    return Json::object{
        {"id", j.id},
        {"object", "image"},
        {"model", "minimax-h3-image"},
        {"status", j.status},
        {"stage", j.stage},
        {"current", j.current},
        {"total", j.total},
        {"progress", j.progress},
        {"created_at", double(j.created)},
        {"prompt", j.options.prompt},
        {"size", std::to_string(j.options.width) + "x" + std::to_string(j.options.height)},
        {"seed", double(j.options.seed)},
        {"num_inference_steps", 6},
        {"mode", j.references.empty() ? "text-to-image" : "image-edit"},
        {"reference_count", int(j.references.size())},
        {"references", j.references},
        {"min_generation_area", j.options.minGenerationArea},
        {"canvas_size",
         std::to_string(j.options.canvas().first) + "x" + std::to_string(j.options.canvas().second)},
        {"url", j.status == "completed" ? Json("/v1/images/jobs/" + j.id + "/content") : Json()},
        {"error", j.failure.empty() ? Json() : Json(Json::object{{"message", j.failure}})}};
}
Json metadata(const std::shared_ptr<Job> &j) {
    std::lock_guard<std::mutex> lock(j->mutex);
    return metadata(*j);
}
class Service {
  public:
    Service(const std::string &model, std::string output, int device, DType vaeType, double gpuLimit,
            double hostLimit, const std::vector<int> &devices, fastllm::nomas::RowReduction reduction,
            const std::string &parallelMode)
        : output_(std::move(output)), gpuLimit_(gpuLimit), hostLimit_(hostLimit),
          parallelSize_(devices.empty() ? 1 : int(devices.size())) {
        std::filesystem::create_directories(output_);
        for (auto &entry : std::filesystem::directory_iterator(output_)) {
            auto name = entry.path().filename().string();
            if (!entry.is_regular_file() || entry.file_size() > 1024 * 1024 ||
                !std::regex_match(name, std::regex("image_[0-9]+_[0-9]+[.]json")))
                continue;
            try {
                std::string error;
                auto j = Json::parse(read(entry.path().string()), error);
                auto id = j["id"].string_value();
                if (!error.empty() || id + ".json" != name || j["status"].string_value() != "completed" ||
                    !std::filesystem::is_regular_file(output_ + "/" + id + ".png"))
                    continue;
                auto job = std::make_shared<Job>();
                job->options = options(j);
                if (j["min_generation_area"].is_null())
                    job->options.minGenerationArea = 1024;
                job->references = j["references"].array_items();
                job->id = id;
                job->created = j["created_at"].number_value();
                job->status = job->stage = "completed";
                job->progress = 100;
                job->done = true;
                jobs_[id] = job;
                if (jobs_.size() > historyLimit)
                    discardOldestCompleted();
            } catch (const std::exception &) { /* Ignore incomplete/stale result metadata. */
            }
        }
        worker_ = std::thread([this, model, device, vaeType, devices, reduction, parallelMode] {
            try {
                // Validate model configs and checkpoint indexes before advertising readiness.
                {
                    Runtime rt(device), full(device, true), vae(device, vaeType);
                    auto source = resolveModelSource(model);
                    Transformer transformer(rt, full, source.directory, {}, false, source.fp8Weights);
                    VAE decoder(vae, full, source.directory);
                    fastllm::nomas::QwenTokenizer tokenizer(source.directory + "/tokenizer");
                    if (!std::filesystem::is_regular_file(source.directory + "/text_encoder/config.json"))
                        throw std::runtime_error("missing text_encoder/config.json");
                }
                Pipeline pipeline(model, device, vaeType, gpuLimit_, hostLimit_, devices, reduction, parallelMode);
                fp8Enabled_ = pipeline.fp8();
                sequenceParallel_ = pipeline.sequenceParallel();
                streaming_ = pipeline.memoryPolicy().layerStreaming;
                hostCacheGiB_ = double(pipeline.memoryPolicy().host->limit()) / (size_t(1) << 30);
                ready_ = true;
                for (;;) {
                    std::shared_ptr<Job> job;
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        available_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
                        if (stopping_)
                            break;
                        job = queue_.front();
                        queue_.pop_front();
                    }
                    const std::string path = output_ + "/" + job->id;
                    try {
                        {
                            std::lock_guard<std::mutex> lock(job->mutex);
                            if (job->cancelled)
                                throw std::runtime_error("cancelled");
                            job->status = "in_progress";
                        }
                        auto image =
                            pipeline.generate(job->options, [&](const std::string &stage, int i, int n) {
                                std::lock_guard<std::mutex> lock(job->mutex);
                                if (job->cancelled || stopping_)
                                    throw std::runtime_error("cancelled");
                                job->stage = stage;
                                job->current = i;
                                job->total = n;
                                int start = stage == "text"        ? 0
                                            : stage == "reference" ? 15
                                            : stage == "prepare"   ? 25
                                            : stage == "denoise"   ? 35
                                                                   : 85;
                                int span = stage == "text"        ? 15
                                           : stage == "reference" ? 10
                                           : stage == "prepare"   ? 10
                                           : stage == "denoise"   ? 50
                                                                  : 14;
                                job->progress = start + (n ? span * i / n : 0);
                            });
                        auto png = fastllm::qwen_image::encodePNG(image);
                        std::lock_guard<std::mutex> lock(job->mutex);
                        if (job->cancelled || stopping_)
                            throw std::runtime_error("cancelled");
                        std::ofstream file(path + ".png.tmp", std::ios::binary);
                        file.write(reinterpret_cast<const char *>(png.data()), png.size());
                        file.close();
                        if (!file)
                            throw std::runtime_error("cannot save image");
                        std::filesystem::rename(path + ".png.tmp", path + ".png");
                        job->status = job->stage = "completed";
                        job->progress = 100;
                        std::ofstream sidecar(path + ".json.tmp");
                        sidecar << metadata(*job).dump();
                        sidecar.close();
                        if (!sidecar)
                            throw std::runtime_error("cannot save image metadata");
                        std::filesystem::rename(path + ".json.tmp", path + ".json");
                        job->done = true;
                    } catch (const std::exception &e) {
                        std::lock_guard<std::mutex> lock(job->mutex);
                        job->status = job->cancelled || stopping_ ? "cancelled" : "failed";
                        job->failure = e.what();
                        job->done = true;
                        for (auto suffix : {".png.tmp", ".json.tmp", ".png", ".json"}) {
                            std::error_code ignored;
                            std::filesystem::remove(path + suffix, ignored);
                        }
                    }
                    {
                        std::lock_guard<std::mutex> lock(job->mutex);
                        job->options.images.clear();
                    }
                    job->changed.notify_all();
                    --pending_;
                }
            } catch (const std::exception &e) {
                std::lock_guard<std::mutex> lock(mutex_);
                startupError_ = e.what();
                ready_ = false;
            }
        });
    }
    ~Service() {
        stopping_ = true;
        available_.notify_all();
        worker_.join();
    }
    bool ready() const { return ready_; }
    Json health() {
        std::lock_guard<std::mutex> lock(mutex_);
        return Json::object{{"status", ready_                  ? "ready"
                                       : startupError_.empty() ? "loading"
                                                               : "failed"},
                            {"error", startupError_.empty() ? Json() : Json(startupError_)},
                            {"dit_precision", fp8Enabled_ ? "fp8" : "bf16"},
                            {"gpu_memory_limit_gib", gpuLimit_},
                            {"host_memory_limit_gib", hostLimit_},
                            {"tensor_parallel_size", parallelSize_},
                            {"sequence_parallel_size", sequenceParallel_.load() ? parallelSize_ : 1},
                            {"dit_parallel_mode", sequenceParallel_.load() ? "sp" : "tp"},
                            {"gpu_memory_reserved", ready_.load() && gpuLimit_ > 0},
                            {"layer_streaming", streaming_.load()},
                            {"host_weight_cache_gib", hostCacheGiB_.load()}};
    }
    static Options options(const Json &j) {
        Options o;
        o.prompt = j["prompt"].string_value();
        auto size = j["size"].is_null() ? "2048x2048" : j["size"].string_value();
        std::smatch match;
        if (!std::regex_match(size, match, std::regex("([0-9]{2,4})x([0-9]{2,4})")))
            throw std::invalid_argument("size must be WIDTHxHEIGHT");
        o.width = std::stoi(match[1]);
        o.height = std::stoi(match[2]);
        o.minGenerationArea = integer(j, "min_generation_area", 2048 * 2048, 1024, 4096 * 4096);
        o.seed =
            integer(j, "seed", int(std::chrono::steady_clock::now().time_since_epoch().count() & 0x7fffffff),
                    0, 2147483647);
        o.validate();
        return o;
    }
    std::shared_ptr<Job> submit(const Json &j, std::vector<Image> images = {}) {
        if (!ready_)
            throw std::runtime_error("model is not ready");
        if (!j.is_object())
            throw std::invalid_argument("expected a JSON object");
        for (auto &kv : j.object_items())
            if (kv.first != "model" && kv.first != "prompt" && kv.first != "size" && kv.first != "seed" &&
                kv.first != "n" && kv.first != "num_inference_steps" && kv.first != "response_format" &&
                kv.first != "min_generation_area" && kv.first != "guidance_scale")
                throw std::invalid_argument("unsupported parameter: " + kv.first);
        if (!j["model"].is_null() && j["model"].string_value() != "minimax-h3-image")
            throw std::invalid_argument("unknown model");
        integer(j, "n", 1, 1, 1);
        integer(j, "num_inference_steps", 6, 6, 6);
        integer(j, "guidance_scale", 1, 1, 1);
        auto job = std::make_shared<Job>();
        job->options = options(j);
        job->options.images = std::move(images);
        job->options.validate();
        for (size_t i = 0; i < job->options.images.size(); ++i) {
            const auto &image = job->options.images[i];
            job->references.push_back(Json::object{{"label", "Picture " + std::to_string(i + 1)},
                                                   {"width", image.width},
                                                   {"height", image.height}});
        }
        if (!j["response_format"].is_null())
            job->format = j["response_format"].string_value();
        if (job->format != "url" && job->format != "b64_json")
            throw std::invalid_argument("response_format must be url or b64_json");
        std::lock_guard<std::mutex> lock(mutex_);
        if (pending_ >= 4)
            throw std::overflow_error("generation queue is full");
        if (jobs_.size() >= historyLimit)
            discardOldestCompleted();
        const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::system_clock::now().time_since_epoch())
                                .count();
        job->id = "image_" + std::to_string(micros) + "_" + std::to_string(++sequence_);
        jobs_[job->id] = job;
        queue_.push_back(job);
        ++pending_;
        available_.notify_one();
        return job;
    }
    std::shared_ptr<Job> find(const std::string &id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = jobs_.find(id);
        return found == jobs_.end() ? nullptr : found->second;
    }
    Json list() {
        std::lock_guard<std::mutex> lock(mutex_);
        Json::array result;
        for (auto &item : jobs_)
            result.push_back(metadata(item.second));
        return Json::object{{"data", result}};
    }

  private:
    // Called before starting the worker, or while holding the service mutex.
    // Result files stay on disk; only the bounded in-memory index is pruned.
    void discardOldestCompleted() {
        auto oldest = jobs_.end();
        for (auto it = jobs_.begin(); it != jobs_.end(); ++it) {
            std::lock_guard<std::mutex> jobLock(it->second->mutex);
            if (it->second->done &&
                (oldest == jobs_.end() || std::tie(it->second->created, it->second->id) <
                                              std::tie(oldest->second->created, oldest->second->id)))
                oldest = it;
        }
        if (oldest != jobs_.end())
            jobs_.erase(oldest);
    }
    std::string output_, startupError_;
    std::atomic<bool> ready_{false}, stopping_{false};
    const double gpuLimit_, hostLimit_;
    const int parallelSize_;
    std::atomic<bool> streaming_{false};
    std::atomic<double> hostCacheGiB_{0};
    std::atomic<int> pending_{0};
    std::atomic<bool> fp8Enabled_{false};
    uint64_t sequence_ = 0;
    std::mutex mutex_;
    std::atomic<bool> sequenceParallel_{false};
    std::condition_variable available_;
    std::thread worker_;
    std::deque<std::shared_ptr<Job>> queue_;
    std::unordered_map<std::string, std::shared_ptr<Job>> jobs_;
};
} // namespace
int main(int argc, char **argv) {
    try {
        std::string model, output = "./minimax-h3-image-output", host = "127.0.0.1", key,
                           demo = "tools/minimax_h3_image/demo.html", precision = "bf16";
        int port = 8009, device = 0;
        std::vector<int> devices;
        std::string parallelMode = "auto";
        auto reduction = fastllm::nomas::RowReduction::Activation;
        double gpuMemoryLimit = 0, hostMemoryLimit = 0;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--help") {
                std::cout
                    << "minimax_h3_image_server --model DIR [--host 127.0.0.1 --port 8009 --device 0 "
                       "--output DIR --demo FILE --api-key KEY --vae-precision bf16|fp32]\n"
                       "  --model accepts a complete directory or FP8 safetensors file (auto-detected).\n"
                       "  [--gpu-memory-limit GiB --host-memory-limit GiB] (0: automatic)\n"
                       "  [--devices 0,1 --parallel-mode auto|tp|sp --tp-reduction fp32|activation]\n";
                return 0;
            }
            if (++i >= argc)
                throw std::invalid_argument("missing value for " + arg);
            std::string value = argv[i];
            if (arg == "--model")
                model = value;
            else if (arg == "--output")
                output = value;
            else if (arg == "--host")
                host = value;
            else if (arg == "--port")
                port = std::stoi(value);
            else if (arg == "--device")
                device = std::stoi(value);
            else if (arg == "--devices")
                devices = fastllm::nomas::parseParallelDevices(value);
            else if (arg == "--parallel-mode")
                parallelMode = value;
            else if (arg == "--tp-reduction") {
                if (value != "fp32" && value != "activation")
                    throw std::invalid_argument("TP reduction must be fp32 or activation");
                reduction = value == "fp32" ? fastllm::nomas::RowReduction::FP32
                                            : fastllm::nomas::RowReduction::Activation;
            } else if (arg == "--api-key")
                key = value;
            else if (arg == "--demo")
                demo = value;
            else if (arg == "--gpu-memory-limit")
                gpuMemoryLimit = fastllm::minimax_h3::parseGpuMemoryLimit(value);
            else if (arg == "--host-memory-limit")
                hostMemoryLimit = fastllm::minimax_h3::parseHostMemoryLimit(value);
            else if (arg == "--vae-precision")
                precision = value;
            else
                throw std::invalid_argument("unknown option: " + arg);
        }
        if (!devices.empty())
            device = devices.front();
        if (model.empty() || port < 1 || port > 65535 || device < 0)
            throw std::invalid_argument("invalid model, port or device");
        if (precision != "bf16" && precision != "fp32")
            throw std::invalid_argument("VAE precision must be bf16 or fp32");
        Service service(model, output, device, precision == "fp32" ? DType::FP32 : DType::BF16,
                        gpuMemoryLimit, hostMemoryLimit, devices, reduction, parallelMode);
        httplib::Server server;
        server.new_task_queue = [] { return new httplib::ThreadPool(8); };
        server.set_payload_max_length(32 * 1024 * 1024);
        auto auth = [&](const httplib::Request &r, httplib::Response &s) {
            if (key.empty() || r.get_header_value("Authorization") == "Bearer " + key)
                return true;
            fail(s, "Unauthorized", 401);
            return false;
        };
        server.Get("/health",
                   [&](const auto &, auto &s) { reply(s, service.health(), service.ready() ? 200 : 503); });
        server.Get("/v1/models", [&](const auto &r, auto &s) {
            if (auth(r, s))
                reply(s, Json::object{{"object", "list"},
                                      {"data", Json::array{Json::object{{"id", "minimax-h3-image"},
                                                                        {"object", "model"},
                                                                        {"owned_by", "local"}}}}});
        });
        auto page = [&](const auto &, auto &s) {
            try {
                s.set_content(read(demo), "text/html; charset=utf-8");
            } catch (const std::exception &e) {
                fail(s, e.what(), 404);
            }
        };
        server.Get("/", page);
        server.Get("/demo", page);
        server.Get(R"(/assets/(image-workspace[.](?:js|css)))", [&](const auto &r, auto &s) {
            auto directory = std::filesystem::path(demo).parent_path();
            const std::string name = r.matches[1];
            auto asset = directory / name;
            if (!std::filesystem::is_regular_file(asset))
                asset = directory.parent_path() / "fastllm_pytools/ui_plugins/studio" / name;
            try {
                s.set_content(read(asset.string()), asset.extension() == ".css"
                                                        ? "text/css; charset=utf-8"
                                                        : "application/javascript; charset=utf-8");
            } catch (const std::exception &e) {
                fail(s, e.what(), 404);
            }
        });
        server.Get("/openapi.json", [&](const auto &, auto &s) {
            try {
                s.set_content(read((std::filesystem::path(demo).parent_path() / "openapi.json").string()),
                              "application/json");
            } catch (const std::exception &e) {
                fail(s, e.what(), 404);
            }
        });
        auto generate = [&](bool async, bool edit, const httplib::Request &r, httplib::Response &s) {
            if (!auth(r, s))
                return;
            try {
                std::string error;
                Json input;
                std::vector<Image> images;
                if (r.is_multipart_form_data()) {
                    if (r.files.count("image") && r.files.count("image[]"))
                        throw std::invalid_argument("use either image or image[] consistently to preserve "
                                                    "reference order");
                    Json::object fields;
                    size_t pixels = 0;
                    for (const auto &part : r.files) {
                        if (part.first == "image" || part.first == "image[]") {
                            if (images.size() >= 16)
                                throw std::invalid_argument("at most 16 reference images are supported");
                            images.push_back(decodeImage(part.second.content));
                            pixels += size_t(images.back().width) * images.back().height;
                            if (pixels > 64 * 1024 * 1024)
                                throw std::invalid_argument("combined references exceed 64 megapixels");
                        } else {
                            if (fields.count(part.first))
                                throw std::invalid_argument("duplicate parameter: " + part.first);
                            bool number = part.first == "seed" || part.first == "n" ||
                                          part.first == "num_inference_steps" ||
                                          part.first == "guidance_scale" ||
                                          part.first == "min_generation_area";
                            fields[part.first] =
                                number ? Json::parse(part.second.content, error) : Json(part.second.content);
                            if (!error.empty())
                                throw std::invalid_argument("invalid numeric parameter: " + part.first);
                        }
                    }
                    input = fields;
                } else {
                    input = Json::parse(r.body, error);
                    if (!error.empty())
                        throw std::invalid_argument("invalid JSON");
                }
                if (edit && images.empty())
                    throw std::invalid_argument("image editing requires multipart image or image[] files");
                if (!async && !edit && !images.empty())
                    throw std::invalid_argument("use /v1/images/edits for reference images");
                auto job = service.submit(input, std::move(images));
                if (async) {
                    reply(s, metadata(job), 202);
                    return;
                }
                std::unique_lock<std::mutex> lock(job->mutex);
                job->changed.wait(lock, [&] { return job->done; });
                if (job->status != "completed") {
                    fail(s, job->failure, 500);
                    return;
                }
                Json::object item;
                if (job->format == "b64_json")
                    item["b64_json"] = base64(read(output + "/" + job->id + ".png"));
                else
                    item["url"] = "/v1/images/jobs/" + job->id + "/content";
                reply(s, Json::object{{"created", double(job->created)},
                                      {"data", Json::array{item}},
                                      {"seed", double(job->options.seed)}});
            } catch (const std::invalid_argument &e) {
                fail(s, e.what(), 400);
            } catch (const std::overflow_error &e) {
                fail(s, e.what(), 429);
            } catch (const std::exception &e) {
                fail(s, e.what(), 503);
            }
        };
        server.Post("/v1/images/jobs", [&](const auto &r, auto &s) { generate(true, false, r, s); });
        server.Post("/v1/images/generations", [&](const auto &r, auto &s) { generate(false, false, r, s); });
        server.Post("/v1/images/edits", [&](const auto &r, auto &s) { generate(false, true, r, s); });
        server.Get("/v1/images/jobs", [&](const auto &r, auto &s) {
            if (auth(r, s))
                reply(s, service.list());
        });
        server.Get(R"(/v1/images/jobs/(image_[0-9]+_[0-9]+))", [&](const auto &r, auto &s) {
            if (!auth(r, s))
                return;
            auto job = service.find(r.matches[1]);
            if (job)
                reply(s, metadata(job));
            else
                fail(s, "unknown job", 404);
        });
        server.Post(R"(/v1/images/jobs/(image_[0-9]+_[0-9]+)/cancel)", [&](const auto &r, auto &s) {
            if (!auth(r, s))
                return;
            auto job = service.find(r.matches[1]);
            if (!job) {
                fail(s, "unknown job", 404);
                return;
            }
            {
                std::lock_guard<std::mutex> lock(job->mutex);
                if (!job->done) {
                    job->cancelled = true;
                    job->status = "cancelling";
                }
            }
            reply(s, metadata(job));
        });
        server.Get(R"(/v1/images/jobs/(image_[0-9]+_[0-9]+)/content)", [&](const auto &r, auto &s) {
            if (!auth(r, s))
                return;
            auto job = service.find(r.matches[1]);
            if (!job) {
                fail(s, "unknown job", 404);
                return;
            }
            {
                std::lock_guard<std::mutex> lock(job->mutex);
                if (job->status != "completed") {
                    fail(s, "image is not ready", 409);
                    return;
                }
            }
            try {
                s.set_content(read(output + "/" + job->id + ".png"), "image/png");
            } catch (const std::exception &e) {
                fail(s, e.what(), 404);
            }
        });
        std::cout << "MiniMax-H3-Image HTTP server: http://" << host << ':' << port << "/demo\n"
                  << std::flush;
        if (!server.listen(host, port))
            throw std::runtime_error("cannot listen on requested address");
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
