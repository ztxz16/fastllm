// Provide by Jacques CHEN (http://whchen.net/index.php/About.html)
// HTML file reference from ChatGLM-MNN （https://github.com/wangzhaode/ChatGLM-MNN)

#include "model.h"
#include "httplib.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <thread>
#include <stdlib.h>
#include <string>
#include <mutex>

struct WebConfig {
    std::string path = "chatglm-6b-int4.bin"; // 模型文件路径
    std::string webPath = "web"; // 网页文件路径
    int threads = 4; // 使用的线程数
    bool lowMemMode = false; // 是否使用低内存模式
    int port = 8081; // 端口号
};

void Usage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "[-h|--help]:                  显示帮助" << std::endl;
    std::cout << "<-p|--path> <args>:           模型文件的路径" << std::endl;
    std::cout << "<-w|--web> <args>:            网页文件的路径" << std::endl;
    std::cout << "<-t|--threads> <args>:        使用的线程数量" << std::endl;
    std::cout << "<-l|--low>:                   使用低内存模式" << std::endl;
    std::cout << "<--port> <args>:              网页端口号" << std::endl;
}

void ParseArgs(int argc, char **argv, WebConfig &config) {
    std::vector <std::string> sargv;
    for (int i = 0; i < argc; i++) {
        sargv.push_back(std::string(argv[i]));
    }
    for (int i = 1; i < argc; i++) {
        if (sargv[i] == "-h" || sargv[i] == "--help") {
            Usage();
            exit(0);
        } else if (sargv[i] == "-p" || sargv[i] == "--path") {
            config.path = sargv[++i];
        } else if (sargv[i] == "-t" || sargv[i] == "--threads") {
            config.threads = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "-l" || sargv[i] == "--low") {
            config.lowMemMode = true;
        } else if (sargv[i] == "-w" || sargv[i] == "--web") {
            config.webPath = sargv[++i];
        } else if (sargv[i] == "--port") {
            config.port = atoi(sargv[++i].c_str());
        } else {
            Usage();
            exit(-1);
        }
    }
}

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
    WebConfig config;
    ParseArgs(argc, argv, config);

    fastllm::SetThreads(config.threads);
    fastllm::SetLowMemMode(config.lowMemMode);
    auto model = fastllm::CreateLLMModelFromFile(config.path);

    httplib::Server svr;
    auto chat = [&](ChatSession *session, const std::string input) {
        if (input == "reset" || input == "stop") {
            session->history = "";
            session->round = 0;
            session->output = "<eop>\n";
            session->status = 2;
        } else {
            auto prompt = model->MakeInput(session->history, session->round, input);
            auto inputs = model->weight.tokenizer.Encode(prompt);

            std::vector<int> tokens;
            for (int i = 0; i < inputs.Count(0); i++) {
                tokens.push_back(((float *) inputs.cpuData)[i]);
            }

            int handleId = model->LaunchResponseTokens(tokens);
            std::vector<float> results;
            while (true) {
                int result = model->FetchResponseTokens(handleId);
                if (result == -1) {
                    break;
                } else {
                    results.clear();
                    results.push_back(result);
                    session->output += model->weight.tokenizer.Decode(fastllm::Data (fastllm::DataType::FLOAT32, {(int)results.size()}, results));
                }
                if (session->status == 2) {
                    break;
                }
            }
            session->history = model->MakeHistory(session->history, session->round++, input, session->output);
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

    svr.set_mount_point("/", config.webPath);
    std::cout << ">>> please open http://127.0.0.1:" + std::to_string(config.port) + "\n";
    svr.listen("0.0.0.0", config.port);

    return 0;
}
