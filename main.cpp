#include "model.h"

std::map <std::string, fastllm::DataType> dataTypeDict = {
    {"float32", fastllm::DataType::FLOAT32},
    {"half", fastllm::DataType::FLOAT16},
    {"float16", fastllm::DataType::FLOAT16},
    {"int8", fastllm::DataType::INT8},
    {"int4", fastllm::DataType::INT4_NOZERO},
    {"int4z", fastllm::DataType::INT4},
    {"int4g", fastllm::DataType::INT4_GROUP}
};

struct RunConfig {
	std::string path = "chatglm-6b-int4.bin"; // 模型文件路径
    std::string systemPrompt = "";
    std::set <std::string> eosToken;
	int threads = 4; // 使用的线程数
	bool lowMemMode = false; // 是否使用低内存模式

    fastllm::DataType dtype = fastllm::DataType::FLOAT16;
    fastllm::DataType atype = fastllm::DataType::FLOAT32;
    int groupCnt = -1;
};

void Usage() {
	std::cout << "Usage:" << std::endl;
	std::cout << "[-h|--help]:                  显示帮助" << std::endl;
	std::cout << "<-p|--path> <args>:           模型文件的路径" << std::endl;
	std::cout << "<-t|--threads> <args>:        使用的线程数量" << std::endl;
	std::cout << "<-l|--low>:                   使用低内存模式" << std::endl;
    std::cout << "<--system> <args>:            设置系统提示词(system prompt)" << std::endl;
    std::cout << "<--eos_token> <args>:         设置eos token" << std::endl;
    std::cout << "<--dtype> <args>:             设置权重类型(读取hf文件时生效)" << std::endl;
    std::cout << "<--top_p> <args>:             采样参数top_p" << std::endl;
    std::cout << "<--top_k> <args>:             采样参数top_k" << std::endl;
    std::cout << "<--temperature> <args>:       采样参数温度，越高结果越不固定" << std::endl;
    std::cout << "<--repeat_penalty> <args>:    采样参数重复惩罚" << std::endl;
}

void ParseArgs(int argc, char **argv, RunConfig &config, fastllm::GenerationConfig &generationConfig) {
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
		} else if (sargv[i] == "-m" || sargv[i] == "--model") {
            i++;
        } else if (sargv[i] == "--top_p") {
            generationConfig.top_p = atof(sargv[++i].c_str());
        } else if (sargv[i] == "--top_k") {
            generationConfig.top_k = atof(sargv[++i].c_str());
        } else if (sargv[i] == "--temperature") {
            generationConfig.temperature = atof(sargv[++i].c_str());
        } else if (sargv[i] == "--repeat_penalty") {
            generationConfig.repeat_penalty = atof(sargv[++i].c_str());
        } else if (sargv[i] == "--system") {
            config.systemPrompt = sargv[++i];
        } else if (sargv[i] == "--eos_token") {
            config.eosToken.insert(sargv[++i]);
        } else if (sargv[i] == "--dtype") {
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
    RunConfig config;
    fastllm::GenerationConfig generationConfig;
	ParseArgs(argc, argv, config, generationConfig);

    fastllm::PrintInstructionInfo();
    fastllm::SetThreads(config.threads);
    fastllm::SetLowMemMode(config.lowMemMode);
    bool isHFDir = access((config.path + "/config.json").c_str(), R_OK) == 0 || access((config.path + "config.json").c_str(), R_OK) == 0;
    auto model = !isHFDir ? fastllm::CreateLLMModelFromFile(config.path) : fastllm::CreateLLMModelFromHF(config.path, config.dtype, config.groupCnt);
    if (config.atype != fastllm::DataType::FLOAT32) {
        model->SetDataType(config.atype);
    }
    model->SetSaveHistoryChat(true);    
    
    for (auto &it : config.eosToken) {
        generationConfig.stop_token_ids.insert(model->weight.tokenizer.GetTokenId(it));
    }
    std::string systemConfig = config.systemPrompt;
    fastllm::ChatMessages messages = {{"system", systemConfig}};

    static std::string modelType = model->model_type;
    printf("欢迎使用 %s 模型. 输入内容对话，reset清空历史记录，stop退出程序.\n", model->model_type.c_str());

    while (true) {
        printf("用户: ");
        std::string input;
        std::getline(std::cin, input);
        if (input == "reset") {
            messages = {{"system", config.systemPrompt}};
            continue;
        }
        if (input == "stop") {
            break;
        }
        messages.push_back(std::make_pair("user", input));
        std::string ret = model->Response(model->ApplyChatTemplate(messages), [](int index, const char* content) {
            if (index == 0) {
                printf("%s:%s", modelType.c_str(), content);
                fflush(stdout);
            }
            if (index > 0) {
                printf("%s", content);
                fflush(stdout);
            }
            if (index == -1) {
                printf("\n");
            }
        }, generationConfig);
        messages.push_back(std::make_pair("assistant", ret));
    }

	return 0;
}