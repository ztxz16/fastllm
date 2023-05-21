#include <cstdio>
#include <cstring>
#include <iostream>

#include "moss.h"
#include "chatglm.h"

struct RunConfig {
    std::string model = "chatglm"; // 模型类型, chatglm或moss
    std::string path; // 模型文件路径
    int threads = 4; // 使用的线程数
};

void Usage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "[-h|--help]:                      显示帮助" << std::endl;
    std::cout << "<-m|--model> <args>:              模型类型，默认为chatglm, 可以设置为chatglm, moss" << std::endl;
    std::cout << "<-p|--path> <args>:               模型文件的路径" << std::endl;
    std::cout << "<-t|--threads> <args>:            使用的线程数量" << std::endl;
}

void ParseArgs(int argc, char **argv, RunConfig &config) {
	std::vector <std::string> sargv;
	for (int i = 0; i < argc; i++) {
		sargv.push_back(std::string(argv[i]));
	}
	for (int i = 1; i < argc; i++) {
		if (sargv[i] == "-h" || sargv[i] == "--help") {
			Usage();
			exit(0);
		} else if (sargv[i] == "-m" || sargv[i] == "--model") {
			config.model = sargv[++i];
		} else if (sargv[i] == "-p" || sargv[i] == "--path") {
			config.path = sargv[++i];
		} else if (sargv[i] == "-t" || sargv[i] == "--threads") {
			config.threads = atoi(sargv[++i].c_str());
		} else {
			Usage();
			exit(-1);
		}
	}
}

int main(int argc, char **argv) {
    RunConfig config;
    ParseArgs(argc, argv, config);
    fastllm::SetThreads(config.threads);

    if (config.model == "moss") {
        fastllm::MOSSModel moss;
        moss.LoadFromFile(config.path);

        while (true) {
            printf("用户: ");
            std::string input;
            std::getline(std::cin, input);
            if (input == "stop") {
                break;
            }
            std::string ret = moss.Response("You are an AI assistant whose name is MOSS. <|Human|>: " + input + "<eoh>");
        }
    } else if (config.model == "chatglm") {
        fastllm::ChatGLMModel chatGlm;
        chatGlm.LoadFromFile(config.path);

        int round = 0;
        std::string history;
        while (true) {
            printf("用户: ");
            std::string input;
            std::getline(std::cin, input);
            if (input == "stop") {
                break;
            }
            if (input == "reset") {
                history = "";
                round = 0;
                continue;
            }
            history += ("[Round " + std::to_string(round++) + "]\n问：" + input);
            auto prompt = round > 1 ? history : input;
            printf("ChatGLM: ");
            std::string ret = chatGlm.Response(prompt);
            history += ("\n答：" + ret + "\n");
        }

        //chatGlm.weight.SaveLowBitModel("/root/chatglm-6b-int4.bin", 4);
    } else {
        Usage();
        exit(-1);
    }

    return 0;
}
