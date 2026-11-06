#include "models/basellm.h"
#include <chrono>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>

using namespace fastllm;

static void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

class StatisticsFixture : public basellm {
public:
    std::string MakeInput(const std::string &, int, const std::string &) override { return {}; }
    std::string MakeHistory(const std::string &, int, const std::string &,
                            const std::string &) override { return {}; }

    int Create(int input, int cached, int output, bool ending = true,
               ResponseContextError error = ResponseContextErrorNone) {
        std::lock_guard<std::mutex> lock(dictLocker);
        int handle = responseContextDict.CreateHandle();
        auto *context = responseContextDict.GetHandle(handle);
        context->inputTokens = input;
        context->cacheLen = cached;
        context->curTokens = output;
        context->isEnding = ending;
        context->error = error;
        for (int i = 0; i < output; ++i) context->resultTokenQueue.push(i + 10);
        return handle;
    }

    void Finish(int handle) {
        std::lock_guard<std::mutex> lock(dictLocker);
        responseContextDict.GetHandle(handle)->isEnding = true;
        dictCV.notify_all();
    }

    void RemoveAborted(int handle) {
        std::lock_guard<std::mutex> lock(dictLocker);
        Check(responseContextDict.GetHandle(handle)->isAbort, "request was not aborted");
        RemoveResponseContext(handle);
        dictCV.notify_all();
    }

    void ReplaceWaiting(int handle, bool notify = true) {
        std::lock_guard<std::mutex> lock(dictLocker);
        const auto generation = responseContextDict.GetHandle(handle)->generation;
        RemoveResponseContext(handle);
        // Force handle wraparound to exercise the generation guard even though
        // ordinary allocation avoids immediate reuse.
        responseContextDict.nextHandle = handle;
        Check(responseContextDict.CreateHandle() == handle, "waiting handle not reused");
        auto *context = responseContextDict.GetHandle(handle);
        Check(context->generation != generation, "request generation not advanced");
        context->resultTokenQueue.push(321);
        context->isEnding = true;
        if (notify) dictCV.notify_all();
    }
};

int main() {
    try {
        StatisticsFixture model;
        int tokens[64], statistics[3];
        // Exercise partial drains and start a new request after terminal fetch.
        for (int iteration = 0; iteration < 1000; ++iteration) {
            int output = iteration % 130;
            int handle = model.Create(10000, 8000, output);
            int received = 0, count;
            while ((count = model.FetchResponseTokensBatch(handle, tokens, 64, statistics)) > 0) {
                for (int i = 0; i < count; ++i)
                    Check(tokens[i] == received + i + 10, "token order changed");
                received += count;
                Check(statistics[0] == 8000 && statistics[1] == 2000 &&
                      statistics[2] == output, "incomplete batch statistics");
            }
            Check(count == -1 && received == output, "terminal result changed");
            int next = model.Create(17000, 0, 0);
            Check(next != handle, "terminal handle was immediately recycled");
            Check(model.FetchResponseTokensBatch(handle, tokens, 64) == -1,
                  "a stale fetch reached the next request");
            Check(statistics[0] == 8000 && statistics[1] == 2000 &&
                  statistics[2] == output, "terminal snapshot was overwritten");
            int cached, missed, generated;
            Check(model.GetResponseStatistics(next, cached, missed, generated) &&
                  cached == 0 && missed == 17000 && generated == 0,
                  "replacement request was corrupted");
            Check(model.FetchResponseTokensBatch(next, tokens, 64) == -1,
                  "legacy batch fetch changed");
        }
        int handle = model.Create(300000, 0, 0, true, ResponseContextErrorPromptTooLong);
        Check(model.FetchResponseTokensBatch(handle, tokens, 64, statistics) == -2,
              "prompt limit error changed");
        Check(statistics[0] == 0 && statistics[1] == 300000 && statistics[2] == 0,
              "missing error statistics");

        handle = model.Create(20, 30, 0);
        model.FetchResponseTokensBatch(handle, tokens, 64, statistics);
        Check(statistics[0] == 20 && statistics[1] == 0, "cache count not clamped");
        Check(model.FetchResponseTokensBatch(handle, tokens, 64, statistics) == -1 &&
              statistics[0] == -1 && statistics[1] == -1 && statistics[2] == -1,
              "missing context returned valid statistics");

        handle = model.Create(100, 80, 0, false);
        Check(model.FetchResponseTokensBatch(handle, tokens, 64, statistics) == 0,
              "keepalive timeout changed");
        Check(statistics[0] == 80 && statistics[1] == 20 && statistics[2] == 0,
              "keepalive lost the snapshot");
        auto waiting = std::async(std::launch::async, [&] {
            return model.FetchResponseTokensBatch(handle, tokens, 64, statistics);
        });
        model.Finish(handle);
        Check(waiting.get() == -1 && statistics[0] == 80, "terminal wake lost statistics");

        handle = model.Create(100, 80, 2, false);
        model.FetchResponseTokensBatch(handle, tokens, 64, statistics);
        model.AbortResponse(handle);
        Check(model.FetchResponseTokensBatch(handle, tokens, 64, statistics) == -1 &&
              statistics[0] == -1, "cancelled fetch must stop before scheduler removal");
        model.RemoveAborted(handle);
        Check(model.FetchResponseTokensBatch(handle, tokens, 64, statistics) == -1 &&
              statistics[0] == -1, "aborted context returned stale statistics");
        const int replacement = model.Create(30, 0, 1);
        Check(replacement != handle, "cancelled handle was immediately recycled");
        std::vector<float> logits;
        Check(model.FetchResponseTokens(handle) == -1 &&
              model.FetchResponseLogits(handle, logits) == -1,
              "stale single-token or logits fetch reached the replacement");
        Check(model.FetchResponseTokensBatch(replacement, tokens, 64) == 1 && tokens[0] == 10,
              "replacement token was consumed by a stale fetch");
        Check(model.FetchResponseTokensBatch(replacement, tokens, 64) == -1,
              "replacement request did not finish");
        // The scheduler and a cancelled request's outstanding token fetch
        // can both be asleep. Launch must wake both classes of waiter.
        StatisticsFixture wake;
        wake.mainLoop = new std::thread([] {});
        std::promise<void> ready[2];
        std::atomic<int> notified{0};
        std::thread waiters[2];
        for (int i = 0; i < 2; ++i) {
            waiters[i] = std::thread([&, i] {
                std::unique_lock<std::mutex> lock(wake.dictLocker);
                ready[i].set_value();
                if (wake.dictCV.wait_for(lock, std::chrono::milliseconds(500)) ==
                    std::cv_status::no_timeout)
                    ++notified;
            });
            ready[i].get_future().wait();
        }
        const int wakeHandle = wake.LaunchResponseTokens({1, 2, 3});
        for (auto &waiter : waiters)
            waiter.join();
        wake.RemoveResponseContext(wakeHandle);
        Check(notified == 2, "new requests must wake the scheduler as well as token consumers");
        for (int api = 0; api < 3; ++api) {
            handle = model.Create(100, 80, 0, false);
            auto fetch = [&] {
                std::vector<float> logits;
                if (api == 0) return model.FetchResponseTokens(handle);
                if (api == 1) return model.FetchResponseLogits(handle, logits);
                return model.FetchResponseTokensBatch(handle, tokens, 64, statistics);
            };
            auto waiter = std::async(std::launch::async, fetch);
            Check(waiter.wait_for(std::chrono::milliseconds(50)) == std::future_status::timeout,
                  "empty request did not block");
            model.ReplaceWaiting(handle);
            Check(waiter.get() == -1, "old consumer followed reused handle");
            Check(model.FetchResponseTokens(handle) == 321,
                  "old consumer drained replacement output");
            Check(model.FetchResponseTokens(handle) == -1, "replacement did not terminate");

            handle = model.Create(100, 80, 0, false);
            waiter = std::async(std::launch::async, fetch);
            Check(waiter.wait_for(std::chrono::milliseconds(50)) == std::future_status::timeout,
                  "terminal wake test did not block");
            model.Finish(handle);
            Check(waiter.get() == -1, "finished request did not wake consumer");
        }
        // Generation validation also applies to the keepalive timeout path.
        handle = model.Create(100, 80, 0, false);
        waiting = std::async(std::launch::async, [&] {
            return model.FetchResponseTokensBatch(handle, tokens, 64, statistics);
        });
        Check(waiting.wait_for(std::chrono::milliseconds(50)) == std::future_status::timeout,
              "keepalive reuse test did not block");
        model.ReplaceWaiting(handle, false);
        Check(waiting.get() == -1, "keepalive timeout hid a recycled request");
        Check(model.FetchResponseTokens(handle) == 321, "keepalive consumed replacement output");
        Check(model.FetchResponseTokens(handle) == -1, "keepalive replacement did not terminate");
        std::cout << "Response statistics tests passed (1000 request transitions, cancellation and forced handle reuse)\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
