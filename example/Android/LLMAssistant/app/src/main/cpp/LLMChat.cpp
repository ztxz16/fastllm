#include <cstdio>
#include <cstring>
#include <iostream>
#include <getopt.h>
#include "LLMChat.h"

#include "model.h"
//void(^ __nonnull RuntimeChat)(int index,const char* _Nonnull content) = NULL;//实时回调

static int modeltype = 0;
static char* modelpath = NULL;
static std::unique_ptr<fastllm::basellm> chatGlm = NULL;
static int sRound = 0;
static std::string history;
static RuntimeResultMobile g_callback = NULL;

std::string initGptConf(const char* modelPath,int threads) {
    fastllm::SetThreads(threads);
    LOG_Debug("@@init llmpath:%s\n",modelPath);
    chatGlm = fastllm::CreateLLMModelFromFile(modelPath);
    if(chatGlm != NULL)
    {
        std::string modelName = chatGlm->model_type;
        LOG_Debug("@@model name:%s\n",modelName.c_str());
        return modelName;
    }
    LOG_Debug("@@CreateLLMModelFromFile failed.");
    return "";
}

int chat(const char* prompt, RuntimeResultMobile chatCallback) {
    std::string ret = "";
    g_callback = chatCallback;
    LOG_Debug("@@init llm:type:%d,prompt:%s\n",modeltype,prompt);
    std::string input(prompt);

    if (input == "reset") {
            history = "";
            sRound = 0;
            g_callback(0,"Done!");
            g_callback(-1,"");
            return 0;
    }

    ret = chatGlm->Response(chatGlm->MakeInput(history, sRound, input), [](int index, const char* content) {
             g_callback(index,content);
        });
    history = chatGlm->MakeHistory(history, sRound, input, ret);
    sRound++;

    long len = ret.length();
    return len;
}

void uninitLLM()
{
}
