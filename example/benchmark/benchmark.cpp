//
// Created by huangyuyang on 6/9/23.
//

#include "model.h"
#include "utils.h"
#include "fstream"

#if defined(_WIN32) or defined(_WIN64)
#include <codecvt>

//GBK locale name in windows
const char* GBK_LOCALE_NAME = ".936";

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

std::map <std::string, fastllm::DataType> dataTypeDict = {
    {"float32", fastllm::DataType::FLOAT32},
    {"half", fastllm::DataType::FLOAT16},
    {"float16", fastllm::DataType::FLOAT16},
    {"int8", fastllm::DataType::INT8},
    {"int4", fastllm::DataType::INT4_NOZERO},
    {"int4z", fastllm::DataType::INT4},
    {"int4g", fastllm::DataType::INT4_GROUP}
};

struct BenchmarkConfig {
    std::string path = "chatglm-6b-int4.bin"; // 模型文件路径
    int threads = 4; // 使用的线程数
    int limit = -1; // 输出token数限制，如果 < 0 则代表无限制
    int batch = -1; // batch数, -1时使用文件中的行数作为batch
    std::string file; // 输入文件
    std::string output; // 输出文件，如果不设定则输出到屏幕

    fastllm::DataType dtype = fastllm::DataType::FLOAT16;
    fastllm::DataType atype = fastllm::DataType::FLOAT32;
    int groupCnt = -1;
};

void Usage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "[-h|--help]:                  显示帮助" << std::endl;
    std::cout << "<-p|--path> <args>:           模型文件的路径" << std::endl;
    std::cout << "<-t|--threads> <args>:        使用的线程数量" << std::endl;
    std::cout << "<-l|--limit> <args>:          输出token数限制" << std::endl;
    std::cout << "<-b|--batch> <args>:          batch数"      << std::endl;
    std::cout << "<-f|--file> <args>:           输入文件，文件中每行一个prompt，如果行数不足batch则用之前的prompt补充"      << std::endl;
    std::cout << "<--dtype> <args>:             设置权重类型(读取hf文件时生效)" << std::endl;
    std::cout << "<--atype> <args>:             设置推理使用的数据类型（float32/float16）" << std::endl;
}

void ParseArgs(int argc, char **argv, BenchmarkConfig &config) {
    std::vector <std::string> sargv;
    for (int i = 0; i < argc; i++) {
        sargv.push_back(std::string(argv[i]));
    }
    for (int i = 1; i < argc; i++) {
        if (sargv[i] == "-h" || sargv[i] == "--help") {
            Usage();
            exit(0);
        }
        else if (sargv[i] == "-p" || sargv[i] == "--path") {
            config.path = sargv[++i];
        } else if (sargv[i] == "-t" || sargv[i] == "--threads") {
            config.threads = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "-l" || sargv[i] == "--limit") {
            config.limit = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "-b" || sargv[i] == "--batch") {
            config.batch = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "-f" || sargv[i] == "--file") {
            config.file = sargv[++i];
        } else if (sargv[i] == "-o" || sargv[i] == "--output") {
            config.output = sargv[++i];
        }  else if (sargv[i] == "--dtype") {
            std::string dtypeStr = sargv[++i];
            if (dtypeStr.size() > 5 && dtypeStr.substr(0, 5) == "int4g") {
                config.groupCnt = atoi(dtypeStr.substr(5).c_str());
                dtypeStr = dtypeStr.substr(0, 5);
            }
            fastllm::AssertInFastLLM(dataTypeDict.find(dtypeStr) != dataTypeDict.end(),
                                    "Unsupport data type: " + dtypeStr);
            config.dtype = dataTypeDict[dtypeStr];
        } else if (sargv[i] == "--atype") {
            std::string atypeStr = sargv[++i];
            fastllm::AssertInFastLLM(dataTypeDict.find(atypeStr) != dataTypeDict.end(),
                                    "Unsupport act type: " + atypeStr);
            config.atype = dataTypeDict[atypeStr];
        } else {
            Usage();
            exit(-1);
        }
    }
}

int main(int argc, char **argv) {
    BenchmarkConfig config;
    ParseArgs(argc, argv, config);
    fastllm::SetThreads(config.threads);

     if (!fastllm::FileExists(config.path)) {
        printf("模型文件 %s 不存在！\n", config.path.c_str());
        exit(0);
    }
    bool isHFDir = fastllm::FileExists(config.path + "/config.json") || fastllm::FileExists(config.path + "config.json");
    auto model = !isHFDir ? fastllm::CreateLLMModelFromFile(config.path) : fastllm::CreateLLMModelFromHF(config.path, config.dtype, config.groupCnt);
    if (config.atype != fastllm::DataType::FLOAT32) {
        model->SetDataType(config.atype);
    }

    fastllm::GenerationConfig generationConfig;
    generationConfig.output_token_limit = config.limit;

    fastllm::PrintInstructionInfo();
    std::vector <std::string> inputs;
    if (config.file != "") {
        std::ifstream finputs(config.file, std::ios::in);
        if (finputs.good()) {
            while (true) {
                std::string input = "";
                std::getline(finputs, input);
                if (input == "") {
                    break;
                } else {
                    inputs.push_back(input);
                }
            }
            finputs.close();
        }
    }
    if (inputs.empty()) {
        inputs.push_back("Hello!");
    }
    if (config.batch <= 0) {
        config.batch = inputs.size();
    }
    while (inputs.size() < config.batch) {
        inputs.push_back(inputs[rand() % inputs.size()]);
    }
    if (inputs.size() > config.batch && config.batch > 0) {
        inputs.resize(config.batch);
    }

    int promptTokenNum = 0;
    for (int i = 0; i < inputs.size(); i++) {
        fastllm::ChatMessages messages;
        messages.push_back({"user", inputs[i]});
        inputs[i] = model->ApplyChatTemplate(messages);
        promptTokenNum += model->weight.tokenizer.Encode(inputs[i]).Count(0);
    }

    std::vector <std::string> outputs;
    static int tokens = 0;
    auto st = std::chrono::system_clock::now();
    static auto promptTime = st;
    model->ResponseBatch(inputs, outputs, [](int index, std::vector<std::string> &contents) {
        if (index != -1) {
            if (index == 0) {
                promptTime = std::chrono::system_clock::now();
            } else {
                for (int i = 0; i < contents.size(); i++) {
                    tokens += (contents[i].size() > 0);
                }
            }
        }
    }, generationConfig);
    float promptSpend = fastllm::GetSpan(st, promptTime);
    float spend = fastllm::GetSpan(promptTime, std::chrono::system_clock::now());

    if (config.output != "") {
        FILE *fo = fopen(config.output.c_str(), "w");
        for (int i = 0; i < outputs.size(); i++) {
            fprintf(fo, "[ user: \"%s\", model: \"%s\"]\n", inputs[i].c_str(), outputs[i].c_str());
        }
        fclose(fo);
    } else {
        for (int i = 0; i < outputs.size(); i++) {
#if defined(_WIN32) or defined(_WIN64)
            printf("[ user: \"%s\", model: \"%s\"]\n", utf8_to_gbk(inputs[i]).c_str(), utf8_to_gbk(outputs[i]).c_str());
#else
            printf("[ user: \"%s\", model: \"%s\"]\n", inputs[i].c_str(), outputs[i].c_str());
#endif
        }
    }

    printf("batch: %d\n", (int)inputs.size());
    printf("prompt token number = %d\n", promptTokenNum);
    printf("prompt use %f s\n", promptSpend);
    printf("prompt speed = %f tokens / s\n", (float)promptTokenNum / promptSpend);
    printf("output %d tokens\nuse %f s\nspeed = %f tokens / s\n", tokens, spend, tokens / spend);
    return 0;
}