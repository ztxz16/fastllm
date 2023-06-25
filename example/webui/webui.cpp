// Provide by Jacques CHEN (http://whchen.net/index.php/About.html)
// HTML file reference from ChatGLM-MNN （https://github.com/wangzhaode/ChatGLM-MNN)

#include "chatglm.h"
#include "httplib.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <thread>
#include <stdlib.h>
#include <string>
#include <mutex>

struct ChatSession {
    std::string history = "";
    std::string input = "";
    std::string output = "";
    int round = 0;
    int status = 0; // 0: 空闲 1: 结果生成好了 2: 已经写回了
};

std::map <std::string, ChatSession*> sessions;
std::mutex locker;

int main(int argc, char** argv) {
    fastllm::SetThreads(8);
    fastllm::ChatGLMModel model;
    model.LoadFromFile(argv[1]);

    httplib::Server svr;

    auto chat = [&](ChatSession *session, const std::string input) {
        if (input == "reset" || input == "stop") {
            session->history = "";
            session->round = 0;
            session->output = "<eop>\n";
            session->status = 2;
        } else {
            session->history += ("[Round " + std::to_string(session->round++) + "]\n问：" + input);
            auto prompt = session->round > 1 ? session->history : input;
            auto inputs = model.weight.tokenizer.Encode(prompt);
            std::vector<int> tokens;
            for (int i = 0; i < inputs.Count(0); i++) {
                tokens.push_back(((float *) inputs.cpuData)[i]);
            }
            int handleId = model.LaunchResponseTokens(tokens);
            std::vector<float> results;
            while (true) {
                auto result = model.FetchResponseTokens(handleId);
                if (result.first == false) {
                    break;
                } else {
                    results.clear();
                    results.push_back(result.second[0]);
                    session->output += model.weight.tokenizer.Decode(fastllm::Data (fastllm::DataType::FLOAT32, {(int)results.size()}, results));
                }
                if (session->status == 2) {
                    break;
                }
            }
            session->history += ("答：" + session->output + "\n");
            session->output += "<eop>\n";
            session->status = 2;
        }
    };

    svr.Post("/chat", [&](const httplib::Request &req, httplib::Response &res) {
        const std::string uuid = req.get_header_value("uuid");
        locker.lock();
        if (sessions.find(uuid) == sessions.end()) {
            sessions[uuid] = new ChatSession();
        }
        auto *session = sessions[uuid];
        locker.unlock();

        if (session->status != 0) {
            res.set_content(session->output, "text/plain");
            if (session->status == 2) {
                session->status = 0;
            }
        } else {
            session->output = "";
            session->status = 1;
            std::thread chat_thread(chat, session, req.body);
            chat_thread.detach();
        }
    });

    svr.set_mount_point("/", "../example/webui/web");
    std::cout << ">>> please open http://127.0.0.1:8081\n";
    svr.listen("0.0.0.0", 8081);

    return 0;
}