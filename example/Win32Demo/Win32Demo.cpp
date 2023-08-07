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

enum RUN_TYPE {
	RUN_TYPE_CONSOLE = 0,
	RUN_TYPE_WEBUI = 1,
};

static int modeltype = 0;
static RUN_TYPE runType = RUN_TYPE_CONSOLE;
static std::unique_ptr<fastllm::basellm> model;
static fastllm::GenerationConfig* generationConfig;
static int sRound = 0;
static std::string modelType;
static std::string history;
static std::string currentContent = "";


struct RunConfig {
	std::string path = "chatglm-6b-int4.bin"; // 模型文件路径
	int threads = 4; // 使用的线程数
	bool lowMemMode = false; // 是否使用低内存模式
	bool webuiType = false; // false 控制台运行 true webui
};

void Usage() {
	std::cout << "Usage:" << std::endl;
	std::cout << "[-h|--help]:						显示帮助" << std::endl;
	std::cout << "<-p|--path> <args>:				模型文件的路径" << std::endl;
	std::cout << "<-t|--threads> <args>:			使用的线程数量" << std::endl;
	std::cout << "<-l|--low>:						使用低内存模式" << std::endl;
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
	std::ifstream f(config.path.c_str());
	if (!f.good()) {
		printf("模型文件 %s 不存在！\n", config.path.c_str());
		exit(0);
	}
	model = fastllm::CreateLLMModelFromFile(config.path);
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
	std::string strInput = model->MakeInput(history, sRound, input);
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
			if (*content > 0 && *content < 127) {
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
	history = model->MakeHistory(history, sRound, input, ret);
	return ret.length();
}

void runConslusion() {
	printf("欢迎使用 %s 模型. 输入内容对话，reset清空历史记录，stop退出程序.\n", modelType.c_str());
	while (true) {
		printf("用户: ");
		std::string input;
		std::getline(std::cin, input);
		if (input == "reset") {
			history = "";
			sRound = 0;
			continue;
		}
		if (input == "stop") {
			break;
		}
		chatllm(input.c_str(), RUN_TYPE_CONSOLE);
	}
}

void runWebUI()
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
		runConslusion();
	} else {
		runWebUI();
	}
	delete generationConfig;
	return 0;
}
