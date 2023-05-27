#include <cstdio>
#include <cstring>
#include <iostream>
#include <getopt.h>

#include "factoryllm.h"

static factoryllm fllm;
static int modeltype = 0;
static char* modelpath = NULL;
static fastllm::basellm* chatGlm = fllm.createllm(LLM_TYPE_CHATGLM);
static fastllm::basellm* moss = fllm.createllm(LLM_TYPE_MOSS);
static int sRound = 0;
static std::string history;

struct RunConfig {
    int model = LLM_TYPE_CHATGLM; // 模型类型, LLM_TYPE_CHATGLM:chatglm, LLM_TYPE_MOSS:moss
    std::string path = "/sdcard/chatglm-6b-int4.bin"; // 模型文件路径
    int threads = 6; // 使用的线程数
};

static struct option long_options[] = {
        {"help",               no_argument,       nullptr, 'h'},
        {"model",              required_argument, nullptr, 'm'},
        {"path",               required_argument, nullptr, 'p'},
        {"threads",            required_argument, nullptr, 't'},
        {nullptr, 0,                              nullptr, 0},
};

void Usage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "[-h|--help]:                      显示帮助" << std::endl;
    std::cout << "<-m|--model> <args>:              模型类型，默认为chatglm, 可以设置为0, moss:1" << std::endl;
    std::cout << "<-p|--path> <args>:               模型文件的路径" << std::endl;
    std::cout << "<-t|--threads> <args>:            使用的线程数量" << std::endl;
}

void ParseArgs(int argc, char **argv, RunConfig &config) {
    int opt;
    int option_index = 0;
    const char *opt_string = "h:m:p:t:";

    while ((opt = getopt_long_only(argc, argv, opt_string, long_options, &option_index)) != -1) {
        switch (opt) {
            case 'h':
                Usage();
                exit (0);
            case 'm':
                config.model = atoi(argv[optind - 1]);
                break;
            case 'p':
                config.path = argv[optind - 1];
                break;
            case 't':
                config.threads = atoi(argv[optind - 1]);
                break;
            default:
                Usage();
                exit (-1);
        }
    }
}

int initLLMConf(int model,const char* modelPath,int threads) {
    fastllm::SetThreads(threads);
    modeltype = model;
//    printf("@@init llm:type:%d,path:%s\n",model,modelPath);
    if (modeltype == 0) {
        chatGlm->LoadFromFile(modelPath);
    }
    if (modeltype == 1) {
        moss->LoadFromFile(modelPath);
    }
    return 0;
}

int chat(const char* prompt) {
    std::string ret = "";
    //printf("@@init llm:type:%d,prompt:%s\n",modeltype,prompt);
    std::string input(prompt);
    if (modeltype == 0) {
        if (input == "reset") {
            history = "";
            sRound = 0;
            return 0;
        }
        history += ("[Round " + std::to_string(sRound++) + "]\n问：" + input);
        auto prompt = sRound > 1 ? history : input;
        ret = chatGlm->Response(prompt,[](int index,const char* content){
            
            if(index == 0) {
                printf("ChatGLM:");
            }
            printf("%s", content);
            if (index == -1) {
                printf("\n");
            }
            
        });
        history += ("\n答：" + ret + "\n");
    }

    if (modeltype == 1) {
        auto prompt = "You are an AI assistant whose name is MOSS. <|Human|>: " + input + "<eoh>";
        ret = moss->Response(prompt,[](int index,const char* content){
            if(index == 0) {
                printf("MOSS:");
            }
            printf("%s", content);
            if (index == -1) {
                printf("\n");
            }
        });
    }
    long len = ret.length();
    return len;
}

void uninitLLM()
{
    if (chatGlm)
    {
        delete chatGlm;
        chatGlm = NULL;
    }
    if (moss)
    {
        delete moss;
        moss = NULL;
    }
}


int main(int argc, char **argv) {
    RunConfig config;
    ParseArgs(argc, argv, config);
    
    initLLMConf(config.model, config.path.c_str(), config.threads);

    if (config.model == LLM_TYPE_MOSS) {

        while (true) {
            printf("用户: ");
            std::string input;
            std::getline(std::cin, input);
            if (input == "stop") {
                break;
            }
            chat(input.c_str());
        }
    } else if (config.model == LLM_TYPE_CHATGLM) {
        while (true) {
            printf("用户: ");
            std::string input;
            std::getline(std::cin, input);
            if (input == "stop") {
                break;
            }
            chat(input.c_str());
        }

    } else {
        Usage();
        exit(-1);
    }

    return 0;
}
