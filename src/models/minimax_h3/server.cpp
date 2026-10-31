#include "models/minimax_h3/runtime.h"
#include "json11.hpp"
#include "httplib.h"
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <regex>
#include <thread>
#include <unistd.h>

using namespace fastllm::minimax_h3;
using Json = json11::Json;
namespace {
int64_t now() { return std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()); }
void reply(httplib::Response &r, const Json &j, int code = 200) {
    r.status = code;
    r.set_content(j.dump(), "application/json");
}
void fail(httplib::Response &r, const std::string &message, int code) {
    reply(
        r,
        Json::object{{"error", Json::object{{"message", message},
                                            {"type", code == 400 ? "invalid_request_error" : "server_error"},
                                            {"code", code}}}},
        code);
}
int integer(const Json &j, const std::string &key, int fallback, int minimum, int maximum) {
    if (j[key].is_null())
        return fallback;
    auto v = j[key].number_value();
    if (!j[key].is_number() || !std::isfinite(v) || std::floor(v) != v || v < minimum || v > maximum)
        throw std::invalid_argument("invalid " + key);
    return int(v);
}
struct Job {
    GenerateOptions options;
    std::string mode = "t2va";
    int referenceCount = 0;
    std::string id, status = "queued", stage = "queued", failure;
    int current = 0, total = 0, progress = 0;
    int64_t created = now(), completed = 0;
    bool cancelled = false, done = false;
    std::mutex mutex;
};
// Caller holds job.mutex while reading or persisting this snapshot.
Json metadata(const Job &job) {
    return Json::object{
        {"id", job.id},
        {"object", "video"},
        {"model", "minimax-h3"},
        {"mode", job.mode},
        {"reference_count", job.referenceCount},
        {"status", job.status},
        {"created_at", double(job.created)},
        {"completed_at", job.completed ? Json(double(job.completed)) : Json()},
        {"size", std::to_string(job.options.width) + "x" + std::to_string(job.options.height)},
        {"seconds", std::to_string(job.options.frames / 24.0)},
        {"frames", job.options.frames},
        {"fps", 24},
        {"seed", double(job.options.seed)},
        {"num_inference_steps", job.options.steps},
        {"prompt", job.options.prompt},
        {"progress", job.progress},
        {"stage", job.stage},
        {"current", job.current},
        {"total", job.total},
        {"error", job.failure.empty()
                      ? Json()
                      : Json(Json::object{{"message", job.failure}, {"code", "generation_failed"}})}};
}
Json metadata(const std::shared_ptr<Job> &job) {
    std::lock_guard<std::mutex> lock(job->mutex);
    return metadata(*job);
}
class Service {
  public:
    Service(std::string model, std::string output, int device, bool fp32, DType videoType, std::string ffmpeg)
        : output_(std::move(output)), fp32_(fp32), videoType_(videoType) {
        std::filesystem::create_directories(output_);
        referencesAvailable_ = std::filesystem::is_regular_file(model + "/transformer_ref/config.json");
        // Completed jobs have a small sidecar, so updating the inference service
        // preserves the IDs of videos already shown in the demo.
        for (const auto &entry : std::filesystem::directory_iterator(output_)) {
            auto name = entry.path().filename().string();
            if (jobs_.size() >= 1024 || !std::regex_match(name, std::regex("video_[0-9]+_[0-9]+[.]json")) ||
                !entry.is_regular_file() || entry.file_size() > 1024 * 1024)
                continue;
            std::ifstream file(entry.path());
            std::string bytes((std::istreambuf_iterator<char>(file)), {}), error;
            auto data = Json::parse(bytes, error);
            auto id = data["id"].string_value();
            if (!error.empty() || id + ".json" != name || data["status"].string_value() != "completed" ||
                !std::filesystem::is_regular_file(output_ + "/" + id + ".mp4"))
                continue;
            auto job = std::make_shared<Job>();
            job->id = id;
            job->status = job->stage = "completed";
            job->done = true;
            job->progress = 100;
            job->created = data["created_at"].number_value();
            job->completed = data["completed_at"].number_value();
            job->options.prompt = data["prompt"].string_value();
            job->options.frames = data["frames"].int_value();
            job->options.seed = data["seed"].number_value();
            if (data["num_inference_steps"].is_number())
                job->options.steps = data["num_inference_steps"].int_value();
            std::smatch match;
            auto size = data["size"].string_value();
            if (!std::regex_match(size, match, std::regex("([0-9]{2,4})x([0-9]{2,4})")))
                continue;
            job->options.width = std::stoi(match[1]);
            job->options.height = std::stoi(match[2]);
            if (data["mode"].is_string())
                job->mode = data["mode"].string_value();
            job->referenceCount = data["reference_count"].int_value();
            jobs_[id] = job;
        }
        worker_ = std::thread([this, model, device, fp32, videoType, ffmpeg] {
            try {
                Pipeline pipeline(model, device, fp32, videoType);
                ready_ = true;
                for (;;) {
                    std::shared_ptr<Job> job;
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        changed_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
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
                        pipeline.generateMP4(job->options, output_ + "/" + job->id + ".mp4", ffmpeg,
                                             [&](const std::string &stage, int current, int total) {
                                                 std::lock_guard<std::mutex> lock(job->mutex);
                                                 if (job->cancelled || stopping_)
                                                     throw std::runtime_error("cancelled");
                                                 job->stage = stage;
                                                 job->current = current;
                                                 job->total = total;
                                                 int first = stage == "text" || stage == "vision" ||
                                                                     stage.compare(0, 10, "condition_") == 0
                                                                 ? 0
                                                             : stage == "prepare"      ? 10
                                                             : stage == "denoise"      ? 15
                                                             : stage == "audio_decode" ? 85
                                                             : stage == "video_decode" ? 89
                                                                                       : 99;
                                                 int span = stage == "text"           ? 10
                                                            : stage == "prepare"      ? 5
                                                            : stage == "denoise"      ? 70
                                                            : stage == "audio_decode" ? 4
                                                            : stage == "video_decode" ? 10
                                                                                      : 0;
                                                 job->progress = first + (total ? span * current / total : 0);
                                             });
                        std::lock_guard<std::mutex> lock(job->mutex);
                        if (job->cancelled) {
                            std::filesystem::remove(output_ + "/" + job->id + ".mp4");
                            throw std::runtime_error("cancelled");
                        }
                        job->status = "completed";
                        job->completed = now();
                        job->progress = 100;
                        job->stage = "completed";
                        // Publish completion only after the sidecar is saved, so a
                        // concurrent DELETE cannot race with metadata creation.
                        try {
                            std::string path = output_ + "/" + job->id + ".json";
                            std::ofstream file(path + ".tmp");
                            file << metadata(*job).dump();
                            file.close();
                            if (!file)
                                throw std::runtime_error("cannot write video metadata");
                            std::filesystem::rename(path + ".tmp", path);
                        } catch (const std::exception &e) {
                            std::cerr << e.what() << "\n";
                        }
                        job->done = true;
                    } catch (const std::exception &e) {
                        std::lock_guard<std::mutex> lock(job->mutex);
                        job->status = job->cancelled ? "cancelled" : "failed";
                        job->failure = e.what();
                        job->done = true;
                    }
                    {
                        std::lock_guard<std::mutex> lock(job->mutex);
                        job->options.firstFrame.reset();
                        job->options.lastFrame.reset();
                        job->options.references.clear();
                    }
                    --pending_;
                }
            } catch (const std::exception &e) {
                std::lock_guard<std::mutex> lock(mutex_);
                startupError_ = e.what();
            }
        });
    }
    ~Service() {
        stopping_ = true;
        changed_.notify_all();
        worker_.join();
    }
    bool ready() const { return ready_; }
    Json health() {
        std::lock_guard<std::mutex> lock(mutex_);
        return Json::object{{"status", ready_                  ? "ready"
                                       : startupError_.empty() ? "loading"
                                                               : "failed"},
                            {"error", startupError_},
                            {"precision", fp32_ ? "fp32" : "bf16"},
                            {"vae_precision", precisionName(videoType_)},
                            {"conditioning", true},
                            {"reference_model_available", referencesAvailable_}};
    }
    std::shared_ptr<Job>
    submit(const Json &j,
           const std::vector<std::pair<std::string, const httplib::MultipartFormData *>> &uploads,
           const std::string &ffmpeg) {
        if (!ready_)
            throw std::runtime_error("model is not ready");
        if (pending_ >= 4)
            throw std::overflow_error("generation queue is full");
        if (!j.is_object())
            throw std::invalid_argument("expected a JSON object");
        for (auto &kv : j.object_items())
            if (kv.first != "model" && kv.first != "prompt" && kv.first != "size" && kv.first != "seconds" &&
                kv.first != "frames" && kv.first != "seed" && kv.first != "num_inference_steps" &&
                kv.first != "reference_image_short_edge" && kv.first != "reference_video_short_edge")
                throw std::invalid_argument("unsupported parameter: " + kv.first);
        if (!j["model"].is_null() && j["model"].string_value() != "minimax-h3")
            throw std::invalid_argument("unknown model");
        auto job = std::make_shared<Job>();
        job->options.prompt = j["prompt"].string_value();
        std::string size = j["size"].is_null() ? "768x448" : j["size"].string_value();
        std::smatch match;
        if (!std::regex_match(size, match, std::regex("([0-9]{2,4})x([0-9]{2,4})")))
            throw std::invalid_argument("size must be WIDTHxHEIGHT");
        job->options.width = std::stoi(match[1]);
        job->options.height = std::stoi(match[2]);
        if (!j["seconds"].is_null() && !j["frames"].is_null())
            throw std::invalid_argument("specify seconds or frames, not both");
        int frames = integer(j, "frames", 124, 120, 345);
        if (!j["seconds"].is_null()) {
            std::string seconds =
                j["seconds"].is_string() ? j["seconds"].string_value() : j["seconds"].dump();
            size_t consumed = 0;
            double duration = std::stod(seconds, &consumed);
            if (consumed != seconds.size() || !std::isfinite(duration) || duration < 5 || duration > 15)
                throw std::invalid_argument("seconds must be 5..15");
            frames = int(std::ceil(duration * 24));
        }
        while (frames % 17 != 5)
            ++frames;
        job->options.frames = frames;
        job->options.steps = integer(j, "num_inference_steps", 50, 2, 100);
        job->options.seed =
            integer(j, "seed", int(std::chrono::steady_clock::now().time_since_epoch().count() & 0x7fffffff),
                    0, 2147483647);
        job->options.referenceImageShortEdge = integer(j, "reference_image_short_edge", 2048, 64, 2048);
        job->options.referenceVideoShortEdge = integer(j, "reference_video_short_edge", 768, 64, 768);
        if (!uploads.empty()) {
            struct Temporary {
                std::string path;
                ~Temporary() {
                    if (!path.empty()) {
                        std::error_code ec;
                        std::filesystem::remove_all(path, ec);
                    }
                }
            } temp;
            std::string pattern =
                (std::filesystem::temp_directory_path() / "fastllm-h3-input-XXXXXX").string();
            std::vector<char> name(pattern.begin(), pattern.end());
            name.push_back(0);
            auto created = mkdtemp(name.data());
            if (!created)
                throw std::runtime_error("cannot create upload directory");
            temp.path = created;
            int index = 0;
            for (const auto &upload : uploads) {
                const auto &part = *upload.second;
                bool reference = upload.first == "reference";
                Reference::Kind kind = Reference::Kind::Image;
                if (reference) {
                    if (part.content_type.compare(0, 6, "video/") == 0)
                        kind = Reference::Kind::Video;
                    else if (part.content_type.compare(0, 6, "audio/") == 0)
                        kind = Reference::Kind::Audio;
                    else if (part.content_type.compare(0, 6, "image/") != 0)
                        throw std::invalid_argument(
                            "reference upload requires an image/*, video/* or audio/* content type");
                }
                if (part.content.empty() || part.content.size() > (size_t(128) << 20))
                    throw std::invalid_argument("each upload must contain 1..128 MiB");
                std::string path = temp.path + "/" + std::to_string(index++);
                std::ofstream file(path, std::ios::binary);
                file.write(part.content.data(), part.content.size());
                file.close();
                if (!file)
                    throw std::runtime_error("cannot save uploaded media");
                auto decoded = readReference(path, kind, frames, ffmpeg, "ffprobe",
                                             job->options.referenceVideoShortEdge);
                if (reference)
                    job->options.references.push_back(std::move(decoded));
                else {
                    auto &frame =
                        upload.first == "last_frame" ? job->options.lastFrame : job->options.firstFrame;
                    if (frame)
                        throw std::invalid_argument("duplicate first/last frame");
                    frame = std::move(decoded.frames[0]);
                }
            }
        }
        job->options.validate();
        if (!job->options.references.empty() && !referencesAvailable_)
            throw std::invalid_argument("Ref2VA requires transformer_ref weights in the model directory");
        job->mode = !job->options.references.empty()                    ? "ref2va"
                    : job->options.firstFrame && job->options.lastFrame ? "first_last"
                    : job->options.firstFrame                           ? "first_frame"
                    : job->options.lastFrame                            ? "last_frame"
                                                                        : "t2va";
        job->referenceCount = job->options.references.size();
        std::lock_guard<std::mutex> lock(mutex_);
        if (pending_ >= 4)
            throw std::overflow_error("generation queue is full");
        if (jobs_.size() >= 1024)
            throw std::overflow_error(
                "video history is full; delete completed videos before submitting more");
        do {
            job->id = "video_" + std::to_string(now()) + "_" + std::to_string(++sequence_);
        } while (jobs_.count(job->id));
        jobs_[job->id] = job;
        queue_.push_back(job);
        ++pending_;
        changed_.notify_one();
        return job;
    }
    std::shared_ptr<Job> find(const std::string &id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto i = jobs_.find(id);
        return i == jobs_.end() ? nullptr : i->second;
    }
    Json list() {
        std::lock_guard<std::mutex> lock(mutex_);
        Json::array data;
        for (auto &item : jobs_)
            data.push_back(metadata(item.second));
        return Json::object{{"object", "list"}, {"data", data}, {"has_more", false}};
    }
    void erase(const std::string &id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto i = jobs_.find(id);
        if (i == jobs_.end())
            return;
        auto job = i->second;
        std::lock_guard<std::mutex> jobLock(job->mutex);
        if (!job->done)
            throw std::logic_error("cancel the running video before deleting it");
        std::filesystem::remove(output_ + "/" + id + ".mp4");
        std::filesystem::remove(output_ + "/" + id + ".json");
        jobs_.erase(i);
    }

  private:
    std::string output_, startupError_;
    bool referencesAvailable_ = false;
    const bool fp32_;
    const DType videoType_;
    std::atomic<bool> ready_{false}, stopping_{false};
    std::atomic<int> pending_{0};
    uint64_t sequence_ = 0;
    std::mutex mutex_;
    std::condition_variable changed_;
    std::thread worker_;
    std::deque<std::shared_ptr<Job>> queue_;
    std::map<std::string, std::shared_ptr<Job>> jobs_;
};
} // namespace
int main(int argc, char **argv) {
    try {
        std::string model, output = "minimax-h3-output", host = "127.0.0.1", key,
                           demo = "tools/minimax_h3/demo.html", ffmpeg = "ffmpeg", vaePrecision;
        int port = 8009, device = 0;
        bool fp32 = false;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (i + 1 >= argc)
                throw std::invalid_argument("missing value for " + arg);
            std::string v = argv[++i];
            if (arg == "--model")
                model = v;
            else if (arg == "--output")
                output = v;
            else if (arg == "--host")
                host = v;
            else if (arg == "--port")
                port = std::stoi(v);
            else if (arg == "--device")
                device = std::stoi(v);
            else if (arg == "--api-key")
                key = v;
            else if (arg == "--demo")
                demo = v;
            else if (arg == "--ffmpeg")
                ffmpeg = v;
            else if (arg == "--precision") {
                if (v != "fp32" && v != "bf16")
                    throw std::invalid_argument("precision must be fp32 or bf16");
                fp32 = v == "fp32";
            } else if (arg == "--vae-precision")
                vaePrecision = v;
            else
                throw std::invalid_argument("unknown argument " + arg);
        }
        if (model.empty())
            throw std::invalid_argument("--model is required");
        auto videoType = parseVideoPrecision(vaePrecision, fp32);
        Service service(model, output, device, fp32, videoType, ffmpeg);
        httplib::Server server;
        server.new_task_queue = [] { return new httplib::ThreadPool(8); };
        server.set_payload_max_length(size_t(256) << 20);
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
                                      {"data", Json::array{Json::object{{"id", "minimax-h3"},
                                                                        {"object", "model"},
                                                                        {"owned_by", "local"}}}}});
        });
        auto page = [&](const auto &, auto &s) {
            std::ifstream f(demo);
            if (!f)
                fail(s, "demo not found", 404);
            else
                s.set_content(std::string((std::istreambuf_iterator<char>(f)), {}),
                              "text/html; charset=utf-8");
        };
        server.Get("/", page);
        server.Get("/demo", page);
        server.Get("/openapi.json", [&](const auto &, auto &s) {
            std::ifstream f(std::filesystem::path(demo).parent_path() / "openapi.json");
            if (!f)
                fail(s, "API specification not found", 404);
            else
                s.set_content(std::string((std::istreambuf_iterator<char>(f)), {}), "application/json");
        });
        server.Post("/v1/videos", [&](const auto &r, auto &s) {
            if (!auth(r, s))
                return;
            try {
                std::string error;
                Json j;
                std::vector<std::pair<std::string, const httplib::MultipartFormData *>> uploads;
                if (r.is_multipart_form_data()) {
                    Json::object fields;
                    for (auto &part : r.files) {
                        if (part.first == "input_reference" || part.first == "first_frame" ||
                            part.first == "last_frame" || part.first == "reference") {
                            uploads.push_back({part.first, &part.second});
                            if (uploads.size() > 12)
                                throw std::invalid_argument("too many media uploads");
                            continue;
                        }
                        if (part.second.content.size() > 32768)
                            throw std::invalid_argument("form field too large");
                        if (fields.count(part.first))
                            throw std::invalid_argument("duplicate parameter: " + part.first);
                        bool numeric = part.first == "seed" || part.first == "frames" ||
                                       part.first == "num_inference_steps" ||
                                       part.first == "reference_image_short_edge" ||
                                       part.first == "reference_video_short_edge";
                        fields[part.first] =
                            numeric ? Json::parse(part.second.content, error) : Json(part.second.content);
                        if (!error.empty())
                            throw std::invalid_argument("invalid numeric parameter: " + part.first);
                    }
                    j = fields;
                } else {
                    if (r.body.size() > 1024 * 1024)
                        throw std::invalid_argument("JSON request exceeds 1 MiB");
                    j = Json::parse(r.body, error);
                    if (!error.empty())
                        throw std::invalid_argument("invalid JSON");
                }
                reply(s, metadata(service.submit(j, uploads, ffmpeg)), 202);
            } catch (const std::invalid_argument &e) {
                fail(s, e.what(), 400);
            } catch (const std::out_of_range &e) {
                fail(s, e.what(), 400);
            } catch (const std::overflow_error &e) {
                fail(s, e.what(), 429);
            } catch (const std::exception &e) {
                fail(s, e.what(), 503);
            }
        });
        server.Get("/v1/videos", [&](const auto &r, auto &s) {
            if (auth(r, s))
                reply(s, service.list());
        });
        server.Get(R"(/v1/videos/(video_[0-9]+_[0-9]+))", [&](const auto &r, auto &s) {
            if (!auth(r, s))
                return;
            auto job = service.find(r.matches[1]);
            if (!job)
                fail(s, "unknown video", 404);
            else
                reply(s, metadata(job));
        });
        server.Post(R"(/v1/videos/(video_[0-9]+_[0-9]+)/cancel)", [&](const auto &r, auto &s) {
            if (!auth(r, s))
                return;
            auto job = service.find(r.matches[1]);
            if (!job) {
                fail(s, "unknown video", 404);
                return;
            }
            {
                std::lock_guard<std::mutex> lock(job->mutex);
                if (!job->done)
                    job->cancelled = true;
            }
            reply(s, metadata(job));
        });
        server.Delete(R"(/v1/videos/(video_[0-9]+_[0-9]+))", [&](const auto &r, auto &s) {
            if (!auth(r, s))
                return;
            auto id = r.matches[1].str();
            if (!service.find(id)) {
                fail(s, "unknown video", 404);
                return;
            }
            try {
                service.erase(id);
                reply(s, Json::object{{"id", id}, {"object", "video.deleted"}, {"deleted", true}});
            } catch (const std::exception &e) {
                fail(s, e.what(), 409);
            }
        });
        server.Get(R"(/v1/videos/(video_[0-9]+_[0-9]+)/content)", [&](const auto &r, auto &s) {
            if (!auth(r, s))
                return;
            auto job = service.find(r.matches[1]);
            if (!job) {
                fail(s, "unknown video", 404);
                return;
            }
            {
                std::lock_guard<std::mutex> lock(job->mutex);
                if (job->status != "completed") {
                    fail(s, "video is not completed", 409);
                    return;
                }
            }
            std::ifstream f(output + "/" + r.matches[1].str() + ".mp4", std::ios::binary);
            if (!f)
                fail(s, "video file not found", 404);
            else
                s.set_content(std::string((std::istreambuf_iterator<char>(f)), {}), "video/mp4");
        });
        std::cout << "MiniMax-H3 HTTP: http://" << host << ":" << port << "/demo\n" << std::flush;
        if (!server.listen(host, port))
            throw std::runtime_error("cannot listen on requested address");
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
