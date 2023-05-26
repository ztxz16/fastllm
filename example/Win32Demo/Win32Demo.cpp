#include "httplib.h"
#include <cstdio>
#include <cstring>
#include <iostream>

#include "StringUtils.h"
#include "factoryllm.h"
#include <shellapi.h>

enum RUN_TYPE {
	RUN_TYPE_CONSOLE = 0,
	RUN_TYPE_WEBUI = 1,
};

static factoryllm fllm;
static int modeltype = 0;
static int runType = RUN_TYPE_CONSOLE;
static char* modelpath = NULL;
static fastllm::basellm* chatGlm = fllm.createllm(LLM_TYPE_CHATGLM);
static fastllm::basellm* moss = fllm.createllm(LLM_TYPE_MOSS);
static int sRound = 0;
static std::string history;
static std::string currentContent = "";


struct RunConfig {
	int model = LLM_TYPE_CHATGLM; // 模型类型, 0 chatglm,1 moss,2 alpaca 参考LLM_TYPE
	std::string path = "chatglm-6b-v1.1-int4.bin"; // 模型文件路径
	int threads = 4; // 使用的线程数
	bool lowMemMode = false; // 是否使用低内存模式
	bool webuiType = true;// false 控制台运行 true webui
};

void Usage() {
	std::cout << "Usage:" << std::endl;
	std::cout << "[-h|--help]:                      显示帮助" << std::endl;
	std::cout << "<-m|--model> <args>:              模型类型，默认为0, 可以设置为0(chatglm),1(moss)" << std::endl;
	std::cout << "<-p|--path> <args>:               模型文件的路径" << std::endl;
	std::cout << "<-t|--threads> <args>:            使用的线程数量" << std::endl;
	std::cout << "<-l|--low> <args>:				使用低内存模式" << std::endl;
	std::cout << "<-w|--webui> <args>:				启用webui" << std::endl;
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
			config.model = atoi(sargv[++i].c_str());
		}
		else if (sargv[i] == "-p" || sargv[i] == "--path") {
			config.path = sargv[++i];
		}
		else if (sargv[i] == "-t" || sargv[i] == "--threads") {
			config.threads = atoi(sargv[++i].c_str());
		}
		else if (sargv[i] == "-l" || sargv[i] == "--low") {
			config.lowMemMode = true;
		}
		else if (sargv[i] == "-w" || sargv[i] == "--webui") {
			config.webuiType = true;
		}
		else {
			Usage();
			exit(-1);
		}
	}
}

int initLLMConf(int model, bool isLowMem, const char* modelPath, int threads) {
	fastllm::SetThreads(threads);
	fastllm::SetLowMemMode(isLowMem);
	modeltype = model;
	//printf("@@init llm:type:%d,path:%s\n", model, modelPath);
	if (modeltype == 0) {
		chatGlm->LoadFromFile(modelPath);
	}
	if (modeltype == 1) {
		moss->LoadFromFile(modelPath);
	}
	return 0;
}

int chatllm(const char* prompt,int type) {
	std::string ret = "";
	runType = type;
	currentContent = "";
	//printf("@@init llm:type:%d,prompt:%s\n", modeltype, prompt);
	std::string input(prompt);
	if (modeltype == LLM_TYPE_CHATGLM) {
		if (input == "reset") 
		{
			history = "";
			sRound = 0;
			currentContent = "<eop>\n";
			return 0;
		}
		history += ("[Round " + std::to_string(sRound++) + "]\n问：" + input);
		auto prompt = sRound > 1 ? history : input;
		std::string strInput = prompt;
		if (runType == RUN_TYPE_CONSOLE) {
			strInput = Gb2utf(prompt);
		}
		ret = chatGlm->Response(strInput, [](int index, const char* content) {
			if (runType == RUN_TYPE_WEBUI)
			{
				if (index > -1) {
					currentContent += content;
				}
				else {
					currentContent += "<eop>";
				}
			}
			else
			{
				std::string result = utf2Gb(content);
				if (index == 0) {
					printf("ChatGLM:%s", result.c_str());
				}
				if (index > 0) {
					printf("%s", result.c_str());
				}
				if (index == -1) {
					printf("\n");
				}
			}

		});
		history += ("\n答：" + ret + "\n");
	}

	if (modeltype == LLM_TYPE_MOSS) {
		auto prompt = "You are an AI assistant whose name is MOSS. <|Human|>: " + (input) + "<eoh>";
		std::string strInput = prompt;
		if (runType == RUN_TYPE_CONSOLE) {
			strInput = Gb2utf(prompt);
		}
		ret = moss->Response(strInput, [](int index, const char* content) {
			if (runType == RUN_TYPE_WEBUI)
			{
				if (index > -1) {
					currentContent += content;
				}
				else {
					currentContent += "<eop>";
				}
			}
			else
			{
				std::string result = utf2Gb(content);
				if (index == 0) {
					printf("MOSS:%s", result.c_str());
				}
				if (index > 0) {
					printf("%s", result.c_str());
				}
				if (index == -1) {
					printf("\n");
				}
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

void runConslusion() {
	if (modeltype == LLM_TYPE_MOSS) {

		while (true) {
			printf("用户: ");
			std::string input;
			std::getline(std::cin, input);
			if (input == "stop") {
				break;
			}
			chatllm(input.c_str(), RUN_TYPE_CONSOLE);
		}
	}
	else if (modeltype == LLM_TYPE_CHATGLM) {
		while (true) {
			printf("用户: ");
			std::string input;
			std::getline(std::cin, input);
			if (input == "stop") {
				break;
			}
			chatllm(input.c_str(), RUN_TYPE_CONSOLE);
		}
	}
	else {
		Usage();
		exit(-1);
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
	ParseArgs(argc, argv, config);
	initLLMConf(config.model, config.lowMemMode, config.path.c_str(), config.threads);

	if (!config.webuiType) 
	{
		runConslusion();
	}
	else
	{
		runWebUI();
	}
	
	return 0;
}
