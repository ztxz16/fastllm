#include <cstdio>
#include <cstring>
#include <iostream>
#include <getopt.h>
#include "LLMChat.h"

#include "factoryllm.h"
//void(^ __nonnull RuntimeChat)(int index,const char* _Nonnull content) = NULL;//实时回调

static factoryllm fllm;
static int modeltype = 0;
static char* modelpath = NULL;
static fastllm::basellm* chatGlm = fllm.createllm(LLM_TYPE_CHATGLM);
static fastllm::basellm* moss = fllm.createllm(LLM_TYPE_MOSS);
static int sRound = 0;
static std::string history;
static RuntimeResult g_callback = NULL;

int initGptConf(int model,const char* modelPath,int threads) {
    fastllm::SetThreads(threads);
    modeltype = model;
    printf("@@init llm:type:%d,path:%s\n",model,modelPath);
    if (modeltype == 0) {
        chatGlm->LoadFromFile(modelPath);
    }
    if (modeltype == 1) {
        moss->LoadFromFile(modelPath);
    }
    return 0;
}

int chat(const char* prompt, RuntimeResult chatCallback) {
    std::string ret = "";
    g_callback = chatCallback;
    printf("@@init llm:type:%d,prompt:%s\n",modeltype,prompt);
    std::string input(prompt);
    if (modeltype == 0) {
        if (input == "reset") {
            history = "";
            sRound = 0;
            g_callback(0,"Done!");
            g_callback(-1,"");
            return 0;
        }
        history += ("[Round " + std::to_string(sRound++) + "]\n问：" + input);
        auto prompt = sRound > 1 ? history : input;
        ret = chatGlm->Response(prompt,[](int index,const char* content){
            g_callback(index,content);
        });
        history += ("\n答：" + ret + "\n");
    }

    if (modeltype == 1) {
        auto prompt = "You are an AI assistant whose name is MOSS. <|Human|>: " + input + "<eoh>";
        ret = moss->Response(prompt,[](int index,const char* content){
            g_callback(index,content);
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
