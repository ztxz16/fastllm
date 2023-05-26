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

std::string GBKToUTF8(const std::string& strGBK);

int main(int argc, char** argv) {
    fastllm::SetThreads(8);
    fastllm::ChatGLMModel chatGlm;

    std::string type;
    std::cout << "Use chatglm-6b-v1.1-int4 or chatglm-6b-v1.1-int8 ? 8/4 (Default = 4) ";
    std::getline(std::cin, type);
    if (type == "8") {
        chatGlm.LoadFromFile("chatglm-6b-v1.1-int8.bin");
    } else if (type == "4" || type == "") {
        chatGlm.LoadFromFile("chatglm-6b-v1.1-int4.bin");
    }

    std::string history = "";
    int round = 0;
    static std::string ss = "";

    httplib::Server svr;
    std::atomic_bool waiting;
    waiting = false;
    std::string last_request = "";

    auto chat = [&](std::string input) {
        if (input == "reset" || input == "stop") {
            history = "";
            round = 0;
            ss = "<eop>\n";
        } else {
            history += ("[Round " + std::to_string(round++) + "]\n问：" + input);
            auto prompt = round > 1 ? history : input;

            waiting = true;
            std::string ret = chatGlm.Response(prompt, [](int index, const char* content) {
				if (index == -1) {
					printf("\n");
					ss = std::string(content) + "<eop>\n";
				}
				});
            waiting = false;

            history += ("答：" + ret + "\n");
        }
    };

    svr.Post("/chat", [&](const httplib::Request &req, httplib::Response &res) {
        if (req.body == last_request) {
            res.set_content(ss, "text/plain");
            return;
        }
        if (waiting) {
            res.set_content(ss, "text/plain");
        } else {
            ss = "";
            last_request = req.body;
            std::thread chat_thread(chat, last_request);
            chat_thread.detach();
        }
    });

    svr.set_mount_point("/", "web");
    std::cout << ">>> please open http://127.0.0.1:8081\n";
    svr.listen("0.0.0.0", 8081);

    return 0;
}