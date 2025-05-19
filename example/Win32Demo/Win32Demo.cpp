#include "httplib.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <fstream>

#include "StringUtils.h"
#include "fastllm.h"
#include "models/basellm.h"
#include "model.h"
#include <shellapi.h>

std::map <std::string, fastllm::DataType> dataTypeDict = {
	{"float32", fastllm::DataType::FLOAT32},
	{"half", fastllm::DataType::FLOAT16},
	{"float16", fastllm::DataType::FLOAT16},
	{"int8", fastllm::DataType::INT8},
	{"int4", fastllm::DataType::INT4_NOZERO},
	{"int4z", fastllm::DataType::INT4},
	{"int4g", fastllm::DataType::INT4_GROUP}
};

enum RUN_TYPE {
	RUN_TYPE_CONSOLE = 0,
	RUN_TYPE_WEBUI = 1,
};

static int modeltype = 0;
static RUN_TYPE runType = RUN_TYPE_CONSOLE;
static std::unique_ptr<fastllm::basellm> model;
static fastllm::GenerationConfig* generationConfig;
static std::string modelType;
static fastllm::ChatMessages* messages;
static std::string currentContent = "";


struct RunConfig {
	std::string path = "chatglm-6b-int4.bin"; // 模型文件路径
	std::string systemPrompt = "";
	std::set <std::string> eosToken;
	int threads = 4; // 使用的线程数
	bool lowMemMode = false; // 是否使用低内存模式
	fastllm::DataType dtype = fastllm::DataType::FLOAT16;
	fastllm::DataType kvtype = fastllm::DataType::FLOAT32;
	int groupCnt = -1;
	bool webuiType = false; // false 控制台运行 true webui
};

void Usage() {
	std::cout << "Usage:" << std::endl;
	std::cout << "[-h|--help]:						显示帮助" << std::endl;
	std::cout << "<-p|--path> <args>:				模型文件的路径" << std::endl;
	std::cout << "<-t|--threads> <args>:			使用的线程数量" << std::endl;
	std::cout << "<-l|--low>:						使用低内存模式" << std::endl;
	std::cout << "<--system> <args>:				设置系统提示词(system prompt)" << std::endl;
	std::cout << "<--eos_token> <args>::			设置eos token" << std::endl;
	std::cout << "<--dtype> <args>:					设置权重类型(读取hf文件时生效)" << std::endl;
	std::cout << "<--kvtype> <args>:				设置推理使用的数据类型（float32/float16）" << std::endl;
	std::cout << "<--top_p> <args>:					采样参数top_p" << std::endl;
	std::cout << "<--top_k> <args>:					采样参数top_k" << std::endl;
	std::cout << "<--temperature> <args>:			采样参数温度，越高结果越不固定" << std::endl;
	std::cout << "<--repeat_penalty> <args>:		采样参数重复惩罚" << std::endl;
	std::cout << "<-w|--webui> <args>:				启用webui" << std::endl;
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
		} else if (sargv[i] == "--kvtype") {
			std::string atypeStr = sargv[++i];
			fastllm::AssertInFastLLM(dataTypeDict.find(atypeStr) != dataTypeDict.end(),
									"Unsupport act type: " + atypeStr);
			config.kvtype = dataTypeDict[atypeStr];
		} else if (sargv[i] == "-w" || sargv[i] == "--webui") {
			config.webuiType = true;
		} else {
			Usage();
			exit(-1);
		}
	}
}

int initLLMConf(RunConfig config) {
	fastllm::PrintInstructionInfo();
	fastllm::SetThreads(config.threads);
	fastllm::SetLowMemMode(config.lowMemMode);
	if (!fastllm::FileExists(config.path)) {
		printf("模型文件 %s 不存在！\n", config.path.c_str());
		exit(0);
	}
	bool isHFDir = fastllm::FileExists(config.path + "/config.json") || fastllm::FileExists(config.path + "config.json");
	model = isHFDir ? fastllm::CreateLLMModelFromHF(config.path, config.dtype, config.groupCnt) : fastllm::CreateLLMModelFromFile(config.path);
	if (config.kvtype != fastllm::DataType::FLOAT32) {
		model->SetDataType(config.kvtype);
	}
	model->SetSaveHistoryChat(true);
	for (auto &it : config.eosToken) {
		generationConfig->stop_token_ids.insert(model->weight.tokenizer.GetTokenId(it));
	}
	std::string systemConfig = config.systemPrompt;
	messages = systemConfig.empty() ? new fastllm::ChatMessages() : new fastllm::ChatMessages({{"system", systemConfig}});

	modelType = model->model_type;
	runType = config.webuiType ? RUN_TYPE_WEBUI : RUN_TYPE_CONSOLE;
	return 0;
}

int chatllm(const char* prompt, int type) {
	std::string ret = "";
	currentContent = "";
	std::string input(prompt);
	if (runType == RUN_TYPE_CONSOLE) {
		input = Gb2utf(input);
	}
	messages->push_back(std::make_pair("user", input));
	std::string strInput = model->ApplyChatTemplate(*messages);
	ret = model->Response(strInput, [](int index, const char* content) {
		if (runType == RUN_TYPE_WEBUI) {
			if (index > -1) {
				currentContent += content;
			} else {
				currentContent += "<eop>";
			}
		} else {
			// std::string result = utf2Gb(content);
			if (index == 0) {
				printf("%s: ", modelType.c_str());
				// printf("%s", result.c_str());
			}
			if (*content > 0 && *content < 127 || (strlen(content) % 3 == 0 && (*content > -32 && *content < -16))) {
				std::string result = utf2Gb(currentContent.c_str());
				currentContent = "";
				printf("%s", result.c_str());
			}
			// if (index > 0) {
				// printf("%s", result.c_str());
			// }
			if (index == -1) {
				std::string result = utf2Gb(currentContent.c_str());
				currentContent = "";
				printf("%s", result.c_str());
				printf("\n");
			} else {
				currentContent += content;
			}
		}

	}, *generationConfig);
	messages->push_back(std::make_pair("assistant", ret));
	return ret.length();
}

void runConslusion(RunConfig &config) {
	printf("欢迎使用 %s 模型. 输入内容对话，reset清空历史记录，stop退出程序.\n", modelType.c_str());
	while (true) {
		printf("用户: ");
		std::string input;
		std::getline(std::cin, input);
		if (input == "reset" || input.empty()) {
			auto begin = config.systemPrompt.empty() ? messages->begin() : std::next(messages->begin());
			messages->erase(begin, messages->end());
			continue;
		}
		if (input == "stop") {
			break;
		}
		chatllm(input.c_str(), RUN_TYPE_CONSOLE);
	}
}

void runWebUI(RunConfig &config)
{
	system("chcp 65001");

	httplib::Server svr;
	std::atomic_bool waiting;
	waiting = false;
	std::string last_request = "";

	auto chat = [&](std::string input) {
		chatllm(input.c_str(), RUN_TYPE_WEBUI);
	};

	svr.Post("/chat", [&](const httplib::Request& req, httplib::Response& res) {
		if (req.body == last_request)
		{
			res.set_content(currentContent, "text/plain");
			return;
		}
		if (waiting)
		{
			res.set_content(currentContent, "text/plain");
		}
		else
		{
			currentContent = "";
			last_request = req.body;
			std::thread chat_thread(chat, last_request);
			chat_thread.detach();
		}
	});

	svr.set_mount_point("/", "web");
	std::wstring url = L"http://localhost";
	std::cout << ">>>If the browser is not open, manually open the url: http://localhost\n";
	auto startExplorer = [&](std::wstring url) {
		Sleep(500);
		::ShellExecute(NULL, L"Open", url.c_str(), 0, 0, SW_SHOWNORMAL);
	};
	std::thread startExplorerThread(startExplorer, url);
	
	svr.listen("0.0.0.0", 80);
}

int main(int argc, char **argv) {
	RunConfig config;
	generationConfig = new fastllm::GenerationConfig();
	ParseArgs(argc, argv, config, *generationConfig);
	initLLMConf(config);

	if (!config.webuiType) {
		runConslusion(config);
	} else {
		runWebUI(config);
	}
	delete generationConfig;
	return 0;
}
