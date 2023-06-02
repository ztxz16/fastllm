#include "factoryllm.h"

static factoryllm fllm;
static int modeltype = 0;
static char* modelpath = NULL;
static fastllm::basellm* chatGlm = fllm.createllm(LLM_TYPE_CHATGLM);
static fastllm::basellm* moss = fllm.createllm(LLM_TYPE_MOSS);
static fastllm::basellm* vicuna = fllm.createllm(LLM_TYPE_VICUNA);
static int sRound = 0;
static std::string history;

std::map <std::string, int> modelDict = {
        {"chatglm", 0}, {"moss", 1}, {"vicuna", 2}
};

struct RunConfig {
	int model = LLM_TYPE_CHATGLM; // 模型类型, 0 chatglm,1 moss,2 vicuna
	std::string path = "chatglm-6b-int4.bin"; // 模型文件路径
	int threads = 4; // 使用的线程数
	bool lowMemMode = false; // 是否使用低内存模式
};

void Usage() {
	std::cout << "Usage:" << std::endl;
	std::cout << "[-h|--help]:                  显示帮助" << std::endl;
	std::cout << "<-m|--model> <args>:          模型类型，默认为0, 可以设置为0(chatglm),1(moss),2(vicuna)" << std::endl;
	std::cout << "<-p|--path> <args>:           模型文件的路径" << std::endl;
	std::cout << "<-t|--threads> <args>:        使用的线程数量" << std::endl;
	std::cout << "<-l|--low> <args>:            使用低内存模式" << std::endl;
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
		}
		else if (sargv[i] == "-m" || sargv[i] == "--model") {
            if (modelDict.find(sargv[i + 1]) != modelDict.end()) {
                config.model = modelDict[sargv[++i]];
            } else {
                config.model = atoi(sargv[++i].c_str());
            }
		}
		else if (sargv[i] == "-p" || sargv[i] == "--path") {
			config.path = sargv[++i];
		}
		else if (sargv[i] == "-t" || sargv[i] == "--threads") {
			config.threads = atoi(sargv[++i].c_str());

		} else if (sargv[i] == "-l" || sargv[i] == "--low") {
			config.lowMemMode = true;
		} else {
			Usage();
			exit(-1);
		}
	}
}

int initLLMConf(int model,bool isLowMem, const char* modelPath, int threads) {
	fastllm::SetThreads(threads);
	fastllm::SetLowMemMode(isLowMem);
	modeltype = model;
	//printf("@@init llm:type:%d,path:%s\n", model, modelPath);
	if (modeltype == 0) {
		chatGlm->LoadFromFile(modelPath);
        chatGlm->WarmUp();
	}
	if (modeltype == 1) {
		moss->LoadFromFile(modelPath);
	}
    if (modeltype == 2) {
        vicuna->LoadFromFile(modelPath);
    }
	return 0;
}

int chat(const char* prompt) {
	std::string ret = "";
	//printf("@@init llm:type:%d,prompt:%s\n", modeltype, prompt);
	std::string input(prompt);
	if (modeltype == LLM_TYPE_CHATGLM) {
		if (input == "reset") {
			history = "";
			sRound = 0;
			return 0;
		}
		history += ("[Round " + std::to_string(sRound++) + "]\n问：" + input);
		auto prompt = sRound > 1 ? history : input;
		ret = chatGlm->Response((prompt), [](int index, const char* content) {
			if (index == 0) {
				printf("ChatGLM:%s", content);
			}
			if (index > 0) {
				printf("%s", content);
			}
			if (index == -1) {
				printf("\n");
			}
		});
		history += ("\n答：" + ret + "\n");
	}

	if (modeltype == LLM_TYPE_MOSS) {
		auto prompt = "You are an AI assistant whose name is MOSS. <|Human|>: " + (input) + "<eoh>";
		ret = moss->Response(prompt, [](int index, const char* content) {
			if (index == 0) {
				printf("MOSS:%s", content);
			}
			if (index > 0) {
				printf("%s", content);
			}
			if (index == -1) {
				printf("\n");
			}
		});
	}

    if (modeltype == LLM_TYPE_VICUNA) {
        if (history == "") {
            history = "ASSISTANT: A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. ";
        }

        auto prompt = history + "USER: " + input + " ASSISTANT: ";
        printf("prompt: %s\n", prompt.c_str());
        ret = vicuna->Response(prompt, [](int index, const char* content) {
            if (index == 0) {
                printf("VICUNA:%s", content);
            }
            if (index > 0) {
                printf("%s", content);
            }
            if (index == -1) {
                printf("\n");
            }
        });
        history += (ret + "</s>");
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
    if (vicuna) {
        delete vicuna;
        vicuna = NULL;
    }
}

int main(int argc, char **argv) {
	RunConfig config;
	ParseArgs(argc, argv, config);
	initLLMConf(config.model, config.lowMemMode, config.path.c_str(), config.threads);

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
	} else if (config.model == LLM_TYPE_VICUNA) {
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