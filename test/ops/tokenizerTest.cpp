//
// Created by TylunasLi on 9/9/24.
//

#include "fastllm.h"
#include "model.h"

#if defined(_WIN32) || defined(_WIN64)
#include <codecvt>

//GBK locale name in windows
const char* GBK_LOCALE_NAME = ".936";

std::string gbk_to_utf8(const std::string& str)
{
    std::wstring_convert<std::codecvt_byname<wchar_t, char, mbstate_t>> convert(new std::codecvt_byname<wchar_t, char, mbstate_t>(GBK_LOCALE_NAME));
    std::wstring tmp_wstr;
    tmp_wstr = convert.from_bytes(str);
    std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
    return conv.to_bytes(tmp_wstr);
}

std::string utf8_to_gbk(const std::string& str)
{
    std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
    std::wstring tmp_wstr;
    try {
        tmp_wstr = conv.from_bytes(str);
    } catch (const std::range_error& e) {
        return str;
    }
    std::wstring_convert<std::codecvt_byname<wchar_t, char, mbstate_t>> convert(new std::codecvt_byname<wchar_t, char, mbstate_t>(GBK_LOCALE_NAME));
    return convert.to_bytes(tmp_wstr);
}
#endif
static std::string modelType;


struct RunConfig {
    std::string path = "chatglm-6b-int4.bin"; // 模型文件路径
    std::string systemPrompt = "";
    std::string defaultResponse = "";
    std::set <std::string> eosToken;
};

void Usage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "[-h|--help]:                  显示帮助" << std::endl;
    std::cout << "<-p|--path> <args>:           模型文件的路径" << std::endl;
    std::cout << "<--system> <args>:            设置系统提示词(system prompt)" << std::endl;
    std::cout << "<--response> <args>:          设置默认回复" << std::endl;
    std::cout << "<--eos_token> <args>:         设置额外的EOS Token" << std::endl;
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
        } else if (sargv[i] == "-p" || sargv[i] == "--path") {
            config.path = sargv[++i];
        } else if (sargv[i] == "--system") {
            config.systemPrompt = sargv[++i];
        } else if (sargv[i] == "--response") {
            config.defaultResponse = sargv[++i];
        } else if (sargv[i] == "--eos_token") {
            config.eosToken.insert(sargv[++i]);
        } else {
            Usage();
            exit(-1);
        }
    }
}

int main(int argc, char **argv) {
    RunConfig config;
    fastllm::GenerationConfig generationConfig;
    ParseArgs(argc, argv, config);

    if (!fastllm::FileExists(config.path)) {
        printf("模型文件 %s 不存在！\n", config.path.c_str());
        exit(0);
    }
    bool isHFDir = fastllm::FileExists(config.path + "/config.json") || fastllm::FileExists(config.path + "config.json");
    auto model = !isHFDir ? fastllm::CreateLLMModelFromFile(config.path) : fastllm::CreateLLMTokenizerFromHF(config.path);
    for (auto &it : config.eosToken) {
        generationConfig.stop_token_ids.insert(model->weight.tokenizer.GetTokenId(it));
    }
    std::string systemConfig = config.systemPrompt;
    fastllm::ChatMessages *messages = systemConfig.empty() ? new fastllm::ChatMessages() : new fastllm::ChatMessages({{"system", systemConfig}});

    modelType = model->model_type;
    printf("欢迎使用 %s 模型. 输入内容对话，reset清空历史记录，stop退出程序.\n", modelType.c_str());

    while (true) {
        printf("用户: ");
        std::string input;
        std::getline(std::cin, input);
        if (input == "reset" || input.empty()) {
            if (systemConfig.empty())
                messages->erase(messages->begin(), messages->end());
            else
                messages->erase(std::next(messages->begin()), messages->end());
            continue;
        }
        if (input == "stop") {
            break;
        }
#if defined(_WIN32) || defined(_WIN64)
        input = gbk_to_utf8(input);
#endif
        messages->push_back(std::make_pair("user", input));
        std::string prompt = model->ApplyChatTemplate(*messages);
#if defined(_WIN32) || defined(_WIN64)
        printf("%s\n", utf8_to_gbk(prompt).c_str());
#else
        printf("%s\n", prompt.c_str());
#endif
        fastllm::Data inputTokenData = model->weight.tokenizer.Encode(prompt);
        std::vector<int> inputTokens;
        inputTokens.resize(1);
        for (int i = 0; i < inputTokenData.Count(0); i++) {
            inputTokens.push_back((int)((float *) inputTokenData.cpuData)[i]);
        }
        for (auto const &i: inputTokens)
            std::cout << i << " ";
        std::cout << std::endl;
        printf("tokens: %d\n", inputTokens.size());
        std::string response = config.defaultResponse.empty() ? u8"<think>\n\n</think>Hello, how can I assist you today ?" : config.defaultResponse; 
#if defined(_WIN32) || defined(_WIN64)
        printf("%s: %s\n", modelType.c_str(), utf8_to_gbk(response).c_str());
#else
        printf("%s: %s\n", modelType.c_str(), response.c_str());
#endif
        messages->push_back(std::make_pair("assistant", response));
    }
    delete messages;
    return 0;
}