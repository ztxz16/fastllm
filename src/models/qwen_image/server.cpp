#include "models/qwen_image/runtime.h"
#include "json11.hpp"
#include "httplib.h"
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <regex>
#include <thread>

using namespace fastllm::qwen_image;
using Json = json11::Json;
namespace {
std::string base64(const std::vector<uint8_t> &v) {
    static const char *table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((v.size() + 2) / 3 * 4);
    for (size_t i = 0; i < v.size(); i += 3) {
        uint32_t n = uint32_t(v[i]) << 16;
        if (i + 1 < v.size())
            n |= uint32_t(v[i + 1]) << 8;
        if (i + 2 < v.size())
            n |= v[i + 2];
        out += table[n >> 18];
        out += table[n >> 12 & 63];
        out += i + 1 < v.size() ? table[n >> 6 & 63] : '=';
        out += i + 2 < v.size() ? table[n & 63] : '=';
    }
    return out;
}
int64_t now() { return std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()); }
void response(httplib::Response &r, const Json &j, int status = 200) {
    r.status = status;
    r.set_content(j.dump(), "application/json");
}
void error(httplib::Response &r, const std::string &message, int status) {
    response(r,
             Json::object{
                 {"error", Json::object{{"message", message},
                                        {"type", status == 400 ? "invalid_request_error" : "server_error"},
                                        {"code", status}}}},
             status);
}
int integer(const Json &j, const std::string &key, int fallback, int minimum, int maximum) {
    if (j[key].is_null())
        return fallback;
    double n = j[key].number_value();
    if (!j[key].is_number() || std::floor(n) != n || n < minimum || n > maximum)
        throw std::invalid_argument("invalid " + key);
    return int(n);
}
struct Job {
    GenerateOptions options;
    std::string id, format = "url", status = "queued", stage = "queued", failure;
    int step = 0, total = 0, failureCode = 500;
    int64_t created = now();
    bool cancelled = false, done = false;
    std::mutex mutex;
    std::condition_variable changed;
};
Json metadata(const std::shared_ptr<Job> &j) {
    std::lock_guard<std::mutex> lock(j->mutex);
    return Json::object{
        {"id", j->id},
        {"status", j->status},
        {"stage", j->stage},
        {"step", j->step},
        {"total_steps", j->total},
        {"created", double(j->created)},
        {"seed", double(j->options.seed)},
        {"error", j->failure.empty() ? Json() : Json(j->failure)},
        {"url", j->status == "completed" ? Json("/v1/images/files/" + j->id + ".png") : Json()}};
}
class Service {
  public:
    Service(std::string model, std::string output, int device) : output_(std::move(output)) {
        std::filesystem::create_directories(output_);
        worker_ = std::thread([this, model, device] {
            try {
                Pipeline pipeline(model, device);
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
                    try {
                        {
                            std::lock_guard<std::mutex> lock(job->mutex);
                            if (job->cancelled)
                                throw std::runtime_error("cancelled");
                            job->status = "in_progress";
                        }
                        auto result = pipeline.generate(job->options,
                                                        [&](const std::string &stage, int step, int total) {
                                                            std::lock_guard<std::mutex> lock(job->mutex);
                                                            if (job->cancelled)
                                                                throw std::runtime_error("cancelled");
                                                            job->stage = stage;
                                                            job->step = step;
                                                            job->total = total;
                                                        });
                        auto png = encodePNG(result);
                        std::ofstream f(output_ + "/" + job->id + ".png", std::ios::binary);
                        f.write(reinterpret_cast<const char *>(png.data()), png.size());
                        f.close();
                        if (!f)
                            throw std::runtime_error("cannot save output image");
                        std::lock_guard<std::mutex> lock(job->mutex);
                        job->options.images.clear();
                        job->status = "completed";
                        job->done = true;
                    } catch (const std::exception &e) {
                        std::lock_guard<std::mutex> lock(job->mutex);
                        job->options.images.clear();
                        job->status = job->cancelled ? "cancelled" : "failed";
                        job->failure = e.what();
                        job->failureCode = dynamic_cast<const std::invalid_argument *>(&e) ? 400 : 500;
                        job->done = true;
                    }
                    job->changed.notify_all();
                    --pending_;
                }
            } catch (const std::exception &e) {
                std::lock_guard<std::mutex> lock(mutex_);
                startupError_ = e.what();
            }
        });
    }
    ~Service() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        available_.notify_all();
        worker_.join();
    }
    bool ready() const { return ready_; }
    Json health() {
        std::lock_guard<std::mutex> lock(mutex_);
        return Json::object{{"status", ready_                  ? "ready"
                                       : startupError_.empty() ? "loading"
                                                               : "failed"},
                            {"error", startupError_.empty() ? Json() : Json(startupError_)}};
    }
    std::shared_ptr<Job> submit(const Json &j, std::vector<Image> images = {}) {
        if (!ready_)
            throw std::runtime_error("model is not ready");
        if (!j.is_object())
            throw std::invalid_argument("expected a JSON object");
        for (auto &kv : j.object_items())
            if (kv.first != "model" && kv.first != "prompt" && kv.first != "size" && kv.first != "n" &&
                kv.first != "response_format" && kv.first != "seed" && kv.first != "num_inference_steps" &&
                kv.first != "reference_resolution")
                throw std::invalid_argument("unsupported parameter: " + kv.first);
        if (!j["model"].is_null() && j["model"].string_value() != "qwen-image-2.1")
            throw std::invalid_argument("unknown model");
        auto job = std::make_shared<Job>();
        job->options.images = std::move(images);
        job->options.prompt = j["prompt"].string_value();
        integer(j, "n", 1, 1, 1);
        std::string size = j["size"].is_null() ? "1024x1024" : j["size"].string_value();
        std::smatch match;
        if (!std::regex_match(size, match, std::regex("([0-9]{2,4})x([0-9]{2,4})")))
            throw std::invalid_argument("size must be WIDTHxHEIGHT");
        job->options.width = std::stoi(match[1]);
        job->options.height = std::stoi(match[2]);
        job->options.referenceResolution = integer(j, "reference_resolution", 1024, 256, 1024);
        job->options.steps = integer(j, "num_inference_steps", 40, 2, 100);
        job->options.seed =
            integer(j, "seed", int(std::chrono::steady_clock::now().time_since_epoch().count() & 0x7fffffff),
                    0, 2147483647);
        job->options.validate();
        if (!j["response_format"].is_null())
            job->format = j["response_format"].string_value();
        if (job->format != "url" && job->format != "b64_json")
            throw std::invalid_argument("response_format must be url or b64_json");
        std::lock_guard<std::mutex> lock(mutex_);
        if (pending_ >= 4)
            throw std::overflow_error("generation queue is full");
        if (jobs_.size() >= 128) {
            auto oldest = jobs_.end();
            for (auto it = jobs_.begin(); it != jobs_.end(); ++it) {
                std::lock_guard<std::mutex> jobLock(it->second->mutex);
                if (it->second->done &&
                    (oldest == jobs_.end() || it->second->created < oldest->second->created))
                    oldest = it;
            }
            if (oldest != jobs_.end())
                jobs_.erase(oldest);
        }
        job->id = "image_" + std::to_string(now()) + "_" + std::to_string(++sequence_);
        jobs_[job->id] = job;
        queue_.push_back(job);
        ++pending_;
        available_.notify_one();
        return job;
    }
    std::shared_ptr<Job> find(const std::string &id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto i = jobs_.find(id);
        return i == jobs_.end() ? nullptr : i->second;
    }

  private:
    std::string output_;
    std::atomic<bool> ready_{false};
    std::atomic<int> pending_{0};
    uint64_t sequence_ = 0;
    bool stopping_ = false;
    std::string startupError_;
    std::mutex mutex_;
    std::condition_variable available_;
    std::thread worker_;
    std::deque<std::shared_ptr<Job>> queue_;
    std::unordered_map<std::string, std::shared_ptr<Job>> jobs_;
};
} // namespace
int main(int argc, char **argv) {
    try {
        std::string model, output = "./qwen-image-output", host = "127.0.0.1", key,
                           demo = "tools/qwen_image/demo.html";
        int port = 8008, device = 0;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (i + 1 >= argc)
                throw std::invalid_argument("missing value for " + arg);
            std::string value = argv[++i];
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
            else if (arg == "--api-key")
                key = value;
            else if (arg == "--demo")
                demo = value;
            else
                throw std::invalid_argument("unknown argument " + arg);
        }
        if (model.empty())
            throw std::invalid_argument("--model is required");
        Service service(model, output, device);
        httplib::Server server;
        server.new_task_queue = [] { return new httplib::ThreadPool(8); };
        server.set_payload_max_length(20 * 1024 * 1024);
        auto authenticate = [&](const httplib::Request &r, httplib::Response &s) {
            if (key.empty() || r.get_header_value("Authorization") == "Bearer " + key)
                return true;
            error(s, "Unauthorized", 401);
            return false;
        };
        server.Get("/health", [&](const auto &, auto &s) {
            response(s, service.health(), service.ready() ? 200 : 503);
        });
        server.Get("/v1/models", [&](const auto &r, auto &s) {
            if (authenticate(r, s))
                response(s, Json::object{{"object", "list"},
                                         {"data", Json::array{Json::object{{"id", "qwen-image-2.1"},
                                                                           {"object", "model"},
                                                                           {"owned_by", "local"}}}}});
        });
        auto page = [&](const auto &, auto &s) {
            std::ifstream f(demo);
            if (!f) {
                error(s, "demo file not found", 404);
                return;
            }
            s.set_content(std::string((std::istreambuf_iterator<char>(f)), {}), "text/html; charset=utf-8");
        };
        server.Get("/", page);
        server.Get("/demo", page);
        server.Get("/openapi.json", [&](const auto &, auto &s) {
            std::ifstream f(std::filesystem::path(demo).parent_path() / "openapi.json");
            if (!f) {
                error(s, "API specification not found", 404);
                return;
            }
            s.set_content(std::string((std::istreambuf_iterator<char>(f)), {}), "application/json");
        });
        auto generate = [&](bool async, bool edit, const httplib::Request &r, httplib::Response &s) {
            if (!authenticate(r, s))
                return;
            try {
                std::string err;
                Json request;
                std::vector<Image> images;
                if (r.is_multipart_form_data()) {
                    Json::object fields;
                    size_t pixels = 0;
                    for (auto &part : r.files) {
                        if (part.first == "image" || part.first == "image[]") {
                            if (images.size() >= 10)
                                throw std::invalid_argument("at most 10 reference images are supported");
                            images.push_back(decodePNG(part.second.content));
                            pixels += size_t(images.back().width) * images.back().height;
                            if (pixels > 16777216)
                                throw std::invalid_argument("combined reference images exceed 16 megapixels");
                        } else {
                            if (fields.count(part.first))
                                throw std::invalid_argument("duplicate parameter: " + part.first);
                            bool numeric = part.first == "seed" || part.first == "n" ||
                                           part.first == "num_inference_steps" ||
                                           part.first == "reference_resolution";
                            fields[part.first] =
                                numeric ? Json::parse(part.second.content, err) : Json(part.second.content);
                            if (!err.empty())
                                throw std::invalid_argument("invalid numeric parameter: " + part.first);
                        }
                    }
                    request = fields;
                } else {
                    request = Json::parse(r.body, err);
                    if (!err.empty())
                        throw std::invalid_argument("invalid JSON");
                }
                if (edit && images.empty())
                    throw std::invalid_argument("edits requires multipart image or image[] PNG files");
                if (!async && !edit && !images.empty())
                    throw std::invalid_argument("use /v1/images/edits for reference images");
                auto job = service.submit(request, std::move(images));
                if (async) {
                    response(s, metadata(job), 202);
                    return;
                }
                std::unique_lock<std::mutex> lock(job->mutex);
                job->changed.wait(lock, [&] { return job->done; });
                if (job->status != "completed") {
                    error(s, job->failure, job->failureCode);
                    return;
                }
                Json::object item;
                if (job->format == "b64_json") {
                    std::ifstream f(output + "/" + job->id + ".png", std::ios::binary);
                    if (!f)
                        throw std::runtime_error("cannot read result image");
                    std::vector<uint8_t> png((std::istreambuf_iterator<char>(f)), {});
                    item["b64_json"] = base64(png);
                } else
                    item["url"] =
                        "http://" + r.get_header_value("Host") + "/v1/images/files/" + job->id + ".png";
                response(s, Json::object{{"created", double(job->created)},
                                         {"data", Json::array{item}},
                                         {"seed", double(job->options.seed)}});
            } catch (const std::invalid_argument &e) {
                error(s, e.what(), 400);
            } catch (const std::overflow_error &e) {
                error(s, e.what(), 429);
            } catch (const std::exception &e) {
                error(s, e.what(), 503);
            }
        };
        server.Post("/v1/images/generations", [&](const auto &r, auto &s) { generate(false, false, r, s); });
        server.Post("/v1/images/edits", [&](const auto &r, auto &s) { generate(false, true, r, s); });
        server.Post("/v1/images/jobs", [&](const auto &r, auto &s) { generate(true, false, r, s); });
        server.Get(R"(/v1/images/jobs/(image_[0-9]+_[0-9]+))", [&](const auto &r, auto &s) {
            if (!authenticate(r, s))
                return;
            auto job = service.find(r.matches[1]);
            if (!job)
                error(s, "unknown job", 404);
            else
                response(s, metadata(job));
        });
        server.Post(R"(/v1/images/jobs/(image_[0-9]+_[0-9]+)/cancel)", [&](const auto &r, auto &s) {
            if (!authenticate(r, s))
                return;
            auto job = service.find(r.matches[1]);
            if (!job) {
                error(s, "unknown job", 404);
                return;
            }
            {
                std::lock_guard<std::mutex> lock(job->mutex);
                if (!job->done)
                    job->cancelled = true;
            }
            response(s, metadata(job));
        });
        server.Get(R"(/v1/images/files/(image_[0-9]+_[0-9]+)\.png)", [&](const auto &r, auto &s) {
            if (!authenticate(r, s))
                return;
            std::ifstream f(output + "/" + r.matches[1].str() + ".png", std::ios::binary);
            if (!f)
                error(s, "image not found", 404);
            else
                s.set_content(std::string((std::istreambuf_iterator<char>(f)), {}), "image/png");
        });
        std::cout << "Qwen-Image native HTTP server: http://" << host << ":" << port << "/demo\n"
                  << std::flush;
        if (!server.listen(host, port))
            throw std::runtime_error("cannot listen on requested address");
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
