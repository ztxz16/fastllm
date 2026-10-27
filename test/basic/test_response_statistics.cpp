#include "models/basellm.h"
#include <future>
#include <iostream>
#include <stdexcept>

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
};

int main() {
    try {
        StatisticsFixture model;
        int tokens[64], statistics[3];
        // Exercise partial drains and immediately recycle the terminal handle.
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
            Check(next == handle, "handle reuse was not exercised");
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
        model.RemoveAborted(handle);
        Check(model.FetchResponseTokensBatch(handle, tokens, 64, statistics) == -1 &&
              statistics[0] == -1, "aborted context returned stale statistics");
        std::cout << "Response statistics tests passed (1000 immediate handle reuses)\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
