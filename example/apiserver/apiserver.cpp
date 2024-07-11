// Provide by Jacques CHEN (http://whchen.net/index.php/About.html)
// HTML file reference from ChatGLM-MNN （https://github.com/wangzhaode/ChatGLM-MNN)

#include <cstdio>
#include <cstring>
#include <iostream>
#include <thread>
#include <stdlib.h>
#include <string>
#include <mutex>

/*
 * Headers
 */

#ifdef _WIN32
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif //_CRT_SECURE_NO_WARNINGS

#ifndef _CRT_NONSTDC_NO_DEPRECATE
#define _CRT_NONSTDC_NO_DEPRECATE
#endif //_CRT_NONSTDC_NO_DEPRECATE

#if defined(_MSC_VER)
#if _MSC_VER < 1900
#error Sorry, Visual Studio versions prior to 2015 are not supported
#endif

#pragma comment(lib, "ws2_32.lib")

#ifdef _WIN64
using ssize_t = __int64;
#else
using ssize_t = long;
#endif
#endif // _MSC_VER

#ifndef S_ISREG
#define S_ISREG(m) (((m)&S_IFREG) == S_IFREG)
#endif // S_ISREG

#ifndef S_ISDIR
#define S_ISDIR(m) (((m)&S_IFDIR) == S_IFDIR)
#endif // S_ISDIR

#ifndef NOMINMAX
#define NOMINMAX
#endif // NOMINMAX

#include <io.h>
#include <winsock2.h>
#include <ws2tcpip.h>

#ifndef WSA_FLAG_NO_HANDLE_INHERIT
#define WSA_FLAG_NO_HANDLE_INHERIT 0x80
#endif

#ifndef strcasecmp
#define strcasecmp _stricmp
#endif // strcasecmp

using socket_t = SOCKET;
#ifdef CPPHTTPLIB_USE_POLL
#define poll(fds, nfds, timeout) WSAPoll(fds, nfds, timeout)
#endif

#else // not _WIN32

#include <arpa/inet.h>
#ifndef _AIX
#include <ifaddrs.h>
#endif
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#ifdef __linux__
#include <resolv.h>
#endif
#include <netinet/tcp.h>
#ifdef CPPHTTPLIB_USE_POLL
#include <poll.h>
#endif
#include <csignal>
#include <pthread.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

using socket_t = int;
#ifndef INVALID_SOCKET
#define INVALID_SOCKET (-1)
#endif
#endif //_WIN32

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cctype>
#include <climits>
#include <condition_variable>
#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include "model.h"

long long _GetCurrentTime() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::seconds>(duration).count();
}

std::string GenerateRandomID() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);

    std::stringstream ss;
    for (int i = 0; i < 36; ++i) {
        if (i == 8 || i == 13 || i == 18 || i == 23) {
            ss << '-';
        }
        ss << std::hex << dis(gen);
    }
    return ss.str();
}

std::map <std::string, fastllm::DataType> dataTypeDict = {
    {"float32", fastllm::DataType::FLOAT32},
    {"half", fastllm::DataType::FLOAT16},
    {"float16", fastllm::DataType::FLOAT16},
    {"int8", fastllm::DataType::INT8},
    {"int4", fastllm::DataType::INT4_NOZERO},
    {"int4z", fastllm::DataType::INT4},
    {"int4g", fastllm::DataType::INT4_GROUP}
};

struct APIConfig {
    std::string path = "chatglm-6b-int4.bin"; // 模型文件路径
    std::string modelName = "fastllm";

    int threads = 4; // 使用的线程数
    bool lowMemMode = false; // 是否使用低内存模式
    bool cudaEmbedding = false; // 是否使用cudaEmbedding
    int port = 8080; // 端口号
    int tokens = -1; // token容量限制
    int batch = 256; // batch数限制
    fastllm::DataType dtype = fastllm::DataType::FLOAT16;
    fastllm::DataType atype = fastllm::DataType::FLOAT32;
    int groupCnt = -1;

    std::map <std::string, int> devices;
};
APIConfig config;

void ToNext(char * &cur, const std::string &target, std::string &v) {
    v = "";
    while (*cur != 0) {
        bool stop = true;
        for (int i = 0; i < target.size(); i++) {
            if (cur[i] != target[i]) {
                stop = false;
                break;
            }
        }
        if (stop && target.size() > 0) {
            cur += target.size();
            break;
        } else {
            v += *(cur++);
        }
    }
}

struct HttpRequest {
    std::string method;
    std::string route;
    std::string type;
    std::unordered_map <std::string, std::string> headers;
    std::string body;

    void Init (char *buffer) {
        char *old = buffer;
        headers.clear();
        ToNext(buffer, " ", method);
        ToNext(buffer, " ", route);
        ToNext(buffer, "\r\n", type);
        while (true) {
            if (buffer[0] == 0 || ((long long)(buffer - old)) > 1024 * 1024) {
                break;
            }
            if (buffer[0] == '\r' && buffer[1] == '\n') {
                buffer += 2;
                ToNext(buffer, "", body);
                break;
            } else {
                std::string key;
                ToNext(buffer, ":", key);
                ToNext(buffer, "\r\n", headers[key]);
            }
        }
    }

    bool IsValid (char *buffer, int size) {
        char *old = buffer;
        headers.clear();
        ToNext(buffer, " ", method);
        ToNext(buffer, " ", route);
        ToNext(buffer, "\r\n", type);
        while (true) {
            if (buffer[0] == 0 || ((long long)(buffer - old)) > 1024 * 1024) {
                break;
            }
            if (buffer[0] == '\r' && buffer[1] == '\n') {
                if (headers.find("Content-Length") != headers.end()) {
                    if (size - ((long long)(buffer - old)) - 2 >= atoi(headers["Content-Length"].c_str())) {
                        return true;
                    } else {
                        return false;
                    }
                }
            } else {
                std::string key;
                ToNext(buffer, ":", key);
                ToNext(buffer, "\r\n", headers[key]);
            }
        }
        return false;
    }

    void Print() {
        for (auto &it : headers) {
            printf("%s: %s\n", it.first.c_str(), it.second.c_str());
        }
        printf("body: %s\n", body.c_str());
    }
} httpChecker;

struct WorkNode {
    int client;
    HttpRequest request;
    json11::Json config;
    std::string error;

    void Init(char *buffer, int client) {
        this->client = client;
        request.Init(buffer);
        config = json11::Json::parse(request.body, this->error);
    }
};

struct WorkQueue {
    std::unique_ptr<fastllm::basellm> model;
    int maxActivateQueryNumber = 256;
    int activateQueryNumber = 0;
    int totalQueryNumber = 0;
    std::mutex locker;
    std::condition_variable cv;
    std::queue <WorkNode*> q;
    std::thread *loop;

    void Push(char *buffer, int client) {
        locker.lock();
        q.push(new WorkNode());
        q.back()->Init(buffer, client);
        locker.unlock();

        cv.notify_all();
    }

    void Start() {
        loop = new std::thread ([] (WorkQueue *ts) {
            while (true) {
                std::unique_lock <std::mutex> lock(ts->locker);
                if (ts->activateQueryNumber >= ts->maxActivateQueryNumber) {
                    fastllm::MySleep(0);
                    continue;
                }
                if (ts->q.empty()) {
                    ts->cv.wait(lock);
                }

                while (ts->activateQueryNumber < ts->maxActivateQueryNumber && !ts->q.empty()) {
                    WorkNode *now = ts->q.front();
                    ts->q.pop();
                    ts->activateQueryNumber++;

                    ts->totalQueryNumber++;
                    printf("totalQueryNumber = %d\n", ts->totalQueryNumber);
//printf("activate = %d, q.size() = %d\n", ts->activateQueryNumber, (int) ts->q.size());

                    std::thread *t = new std::thread([](WorkQueue *ts, WorkNode *now) {
                        ts->Deal(now);
                        printf("Response client %d finish\n", now->client);
                        ts->locker.lock();
                        ts->activateQueryNumber--;
                        ts->locker.unlock();
                    }, ts, now);
                }
            }
        }, this);
    }

    void Deal(WorkNode *node) {
        auto *req = &node->request;
        if ((req->route == "/generate" || req->route == "/generate/") && req->method == "POST") {
            std::string message = "";
            message += "HTTP/1.1 200 OK\r\n";
            message += "Content-Type:application/json\r\n";
            message += "server:fastllm api server\r\n";
            message += "\r\n";

            if (node->error == "") {
                if (node->config["prompt"].is_null()) {
                    node->error = "prompt is empty!";
                }
            }
            if (node->error != "") {
                printf("error body = %s, prompt = %s, error = %s\n", node->request.body.c_str(), node->config["prompt"].string_value().c_str(), node->error.c_str());
                message += node->error;
                int ret = write(node->client, message.c_str(), message.length()); //返回error
                close(node->client);
                return;
            }

            std::string output = "";
            fastllm::ChatMessages messages;
            messages.push_back({"user", node->config["prompt"].string_value()});
            auto prompt = model->ApplyChatTemplate(messages);
            auto inputs = model->weight.tokenizer.Encode(prompt);
            std::vector<int> tokens;
            for (int i = 0; i < inputs.Count(0); i++) {
                tokens.push_back(((float *) inputs.cpuData)[i]);
            }
            fastllm::GenerationConfig config;
            config.output_token_limit = node->config["max_tokens"].is_null() ? 200 : node->config["max_tokens"].int_value();
            int handleId = model->LaunchResponseTokens(tokens, config);
            std::vector<float> results;
            while (true) {
                int result = model->FetchResponseTokens(handleId);
                if (result == -1) {
                    break;
                } else {
                    results.clear();
                    results.push_back(result);
                    output += model->weight.tokenizer.Decode(fastllm::Data (fastllm::DataType::FLOAT32, {(int)results.size()}, results));

                    std::string cur = (message + output);
                    int ret = write(node->client, cur.c_str(), cur.length()); //返回message
                }
            }

            message += output;
            int ret = write(node->client, message.c_str(), message.length()); //返回message

            close(node->client);
        } else if ((req->route == "/v1/chat/completions" || req->route == "/v1/chat/completions/") && req->method == "POST") {
            std::string message = "";
            message += "HTTP/1.1 200 OK\r\n";
            message += "Content-Type:application/json\r\n";
            message += "server:fastllm api server\r\n";
            message += "\r\n";

            fastllm::ChatMessages chatMessages;
            if (node->config["messages"].is_array()) {
                for (auto &it : node->config["messages"].array_items()) {
                    chatMessages.push_back({it["role"].string_value(), it["content"].string_value()});
                }
            } else if (node->config["prompt"].is_string()) {
                chatMessages.push_back({"user", node->config["prompt"].string_value()});
            } else {
                node->error = "no input.\n";
            }

            if (node->config["model"].string_value() != ::config.modelName) {
                node->error = "The model `" + node->config["model"].string_value() + "` does not exist.";
            }

            if (node->error != "") {                
                message += node->error;
                int ret = write(node->client, message.c_str(), message.length()); //返回error
                close(node->client);
                return;
            }

            auto prompt = model->ApplyChatTemplate(chatMessages);
            auto inputs = model->weight.tokenizer.Encode(prompt);
            std::vector<int> tokens;
            for (int i = 0; i < inputs.Count(0); i++) {
                tokens.push_back(((float *) inputs.cpuData)[i]);
            }

            fastllm::GenerationConfig config;
            config.output_token_limit = !node->config["max_tokens"].is_number() ? 256 : node->config["max_tokens"].int_value();
            if (node->config["frequency_penalty"].is_number()) {
                config.repeat_penalty = node->config["frequency_penalty"].number_value();
            }
            if (node->config["temperature"].is_number()) {
                config.temperature = node->config["temperature"].number_value();
            }
            if (node->config["top_p"].is_number()) {
                config.top_p = node->config["top_p"].number_value();
            }
            if (node->config["top_k"].is_number()) {
                config.top_k = node->config["top_k"].number_value();
            }

            std::string output = "";
            int handleId = model->LaunchResponseTokens(tokens, config);
            bool isStream = false;
            if (node->config["stream"].is_bool() && node->config["stream"].bool_value()) {
                isStream = true;
            }

            std::string curId = "fastllm-" + GenerateRandomID();
            auto createTime = _GetCurrentTime();

            if (isStream) {
                message = "";
                message += "HTTP/1.1 200 OK\r\n";
                message += "Content-Type:application/json\r\n";
                message += "server:fastllm api server\r\n";
                message += "Transfer-Encoding: chunked\r\n";
                message += "\r\n";
                int ret = write(node->client, message.c_str(), message.length()); //返回初始信息
            
                json11::Json startResult = json11::Json::object {
                    {"id", curId},
                    {"object", "chat.completion.chunk"},
                    {"created", createTime},
                    {"model", ::config.modelName},
                    {"choices", json11::Json::array {
                        json11::Json::object {
                            {"index", 0},
                            {"delta", json11::Json::object {
                                {"role", "assistant"}
                            }},
                            {"logprobs", nullptr},
                            {"finish_reason", nullptr},
                            {"stop_reason", nullptr}
                        }
                    }}
                };
                std::string cur = ("data: " + startResult.dump() + "\r\n");

                char chunk_header[50];
                sprintf(chunk_header, "%zx\r\n", cur.size());
                ret = write(node->client, chunk_header, strlen(chunk_header));
                ret = write(node->client, cur.data(), cur.size());
                ret = write(node->client, "\r\n", 2);

                int outputTokens = 0;
                std::vector<float> results;
                while (true) {
                    int result = model->FetchResponseTokens(handleId);
                    if (result == -1) {
                        json11::Json partResult = json11::Json::object {
                            {"id", curId},
                            {"object", "chat.completion.chunk"},
                            {"created", createTime},
                            {"model", ::config.modelName},
                            {"choices", json11::Json::array {
                                json11::Json::object {
                                    {"index", 0},
                                    {"delta", json11::Json::object {
                                        {"content", ""}
                                    }},
                                    {"logprobs", nullptr},
                                    {"finish_reason", nullptr},
                                    {"stop_reason", nullptr}
                                }
                            }},
                            {"usage", json11::Json::object {
                                {"prompt_tokens", (int)tokens.size()},
                                {"total_tokens", (int)tokens.size() + outputTokens},
                                {"completion_tokens", outputTokens}
                            }}
                        };

                        std::string cur = ("data: " + partResult.dump() + "\r\n");
                        sprintf(chunk_header, "%zx\r\n", cur.size());
                        ret = write(node->client, chunk_header, strlen(chunk_header));
                        ret = write(node->client, cur.data(), cur.size());
                        ret = write(node->client, "\r\n", 2);
                        break;
                    } else {
                        outputTokens++;
                        results.clear();
                        results.push_back(result);
                        std::string now = model->weight.tokenizer.Decode(fastllm::Data (fastllm::DataType::FLOAT32, {(int)results.size()}, results));
                        json11::Json partResult = json11::Json::object {
                            {"id", curId},
                            {"object", "chat.completion.chunk"},
                            {"created", createTime},
                            {"model", ::config.modelName},
                            {"choices", json11::Json::array {
                                json11::Json::object {
                                    {"index", 0},
                                    {"delta", json11::Json::object {
                                        {"content", now}
                                    }},
                                    {"logprobs", nullptr},
                                    {"finish_reason", nullptr},
                                    {"stop_reason", nullptr}
                                }
                            }}
                        };

                        std::string cur = ("data: " + partResult.dump() + "\r\n");
                        sprintf(chunk_header, "%zx\r\n", cur.size());
                        ret = write(node->client, chunk_header, strlen(chunk_header));
                        ret = write(node->client, cur.data(), cur.size());
                        ret = write(node->client, "\r\n", 2);
                    }
                }

                cur = ("data: [DONE]");
                sprintf(chunk_header, "%zx\r\n", cur.size());
                ret = write(node->client, chunk_header, strlen(chunk_header));
                ret = write(node->client, cur.data(), cur.size());
                ret = write(node->client, "\r\n", 2);

                ret = write(node->client, "0\r\n\r\n", 5);
                close(node->client);
            } else {
                int outputTokens = 0;
                std::vector<float> results;
                while (true) {
                    int result = model->FetchResponseTokens(handleId);
                    if (result == -1) {
                        break;
                    } else {
                        results.clear();
                        results.push_back(result);
                        output += model->weight.tokenizer.Decode(fastllm::Data (fastllm::DataType::FLOAT32, {(int)results.size()}, results));
                        outputTokens++;
                    }
                }

                json11::Json result = json11::Json::object {
                    {"id", curId},
                    {"object", "chat.completion"},
                    {"created", createTime},
                    {"model", ::config.modelName},
                    {"choices", json11::Json::array {
                        json11::Json::object {
                            {"index", 0},
                            {"message", json11::Json::object {
                                {"role", "assistant"},
                                {"content", output}
                            }},
                            {"logprobs", nullptr},
                            {"finish_reason", nullptr},
                            {"stop_reason", nullptr}
                        }
                    }},
                    {"usage", json11::Json::object {
                        {"prompt_tokens", (int)tokens.size()},
                        {"total_tokens", (int)tokens.size() + outputTokens},
                        {"completion_tokens", outputTokens}
                    }}
                };

                message += result.dump();
                int ret = write(node->client, message.c_str(), message.length()); //返回message
                close(node->client);
            }
            return;
        } else {
            close(node->client);
            return;
        }
    }
} workQueue;

void Usage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "[-h|--help]:                  显示帮助" << std::endl;
    std::cout << "<-p|--path> <args>:           模型文件的路径" << std::endl;
    std::cout << "<-t|--threads> <args>:        使用的线程数量" << std::endl;
    std::cout << "<-l|--low>:                   使用低内存模式" << std::endl;
    std::cout << "<--dtype> <args>:             设置权重类型(读取hf文件时生效)" << std::endl;
    std::cout << "<--atype> <args>:             设置推理使用的数据类型(float32/float16)" << std::endl;
    std::cout << "<--batch> <args>:             最大batch数" << std::endl;
    std::cout << "<--tokens> <args>:            最大tokens容量" << std::endl;
    std::cout << "<--model_name> <args>:        模型名(openai api中使用)" << std::endl;
    std::cout << "<--port> <args>:              网页端口号" << std::endl;
    std::cout << "<--cuda_embedding>:           使用cuda来执行embedding" << std::endl;
    std::cout << "<--device>:                   执行设备" << std::endl;
}

void ParseArgs(int argc, char **argv, APIConfig &config) {
    std::vector<std::string> sargv;
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
        } else if (sargv[i] == "--cuda_embedding"){
            config.cudaEmbedding = true;
        } else if (sargv[i] == "--port") {
            config.port = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "--dtype") {
            std::string dtypeStr = sargv[++i];
            if (dtypeStr.size() > 5 && dtypeStr.substr(0, 5) == "int4g") {
                config.groupCnt = atoi(dtypeStr.substr(5).c_str());
                dtypeStr = dtypeStr.substr(0, 5);
            }
            fastllm::AssertInFastLLM(dataTypeDict.find(dtypeStr) != dataTypeDict.end(),
                                    "Unsupport data type: " + dtypeStr);
            config.dtype = dataTypeDict[dtypeStr];
        } else if (sargv[i] == "--tokens") {
            config.tokens = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "--batch") {
            config.batch = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "--atype") {
            std::string atypeStr = sargv[++i];
            fastllm::AssertInFastLLM(dataTypeDict.find(atypeStr) != dataTypeDict.end(),
                                    "Unsupport act type: " + atypeStr);
            config.atype = dataTypeDict[atypeStr];
        } else if (sargv[i] == "--model_name") {
            config.modelName = sargv[++i];
        } else if (sargv[i] == "--device") {
            config.devices[sargv[++i]] = 1;
        } else {
            Usage();
            exit(-1);
        }
    }
}

char buff[1024 * 1024] = {0};
std::string url = "generate";
std::mutex locker;

int main(int argc, char** argv) {
    ParseArgs(argc, argv, config);

    if (config.devices.size() != 0) {
        fastllm::SetDeviceMap(config.devices);
    }
    fastllm::SetThreads(config.threads);
    fastllm::SetLowMemMode(config.lowMemMode);
    fastllm::SetCudaEmbedding(config.cudaEmbedding);
    if (!fastllm::FileExists(config.path)) {
        printf("模型文件 %s 不存在！\n", config.path.c_str());
        exit(0);
    }
    bool isHFDir = fastllm::FileExists(config.path + "/config.json") || fastllm::FileExists(config.path + "config.json");
    workQueue.model = isHFDir ? fastllm::CreateLLMModelFromHF(config.path, config.dtype, config.groupCnt)
        : fastllm::CreateLLMModelFromFile(config.path);
    workQueue.model->tokensLimit = config.tokens;
    workQueue.model->SetDataType(config.atype);
    workQueue.maxActivateQueryNumber = std::max(1, std::min(256, config.batch));
    workQueue.Start();

    int local_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (local_fd == -1) {
        std::cout << "socket error!" << std::endl;
        exit(-1);
    }
    std::cout << "socket ready!" << std::endl;

    struct sockaddr_in local_addr;
    local_addr.sin_family = AF_INET;
    local_addr.sin_port = htons(config.port);  //绑定端口
    local_addr.sin_addr.s_addr = INADDR_ANY; //绑定本机IP地址

    //3.bind()： 将一个网络地址与一个套接字绑定，此处将本地地址绑定到一个套接字上
    int res = bind(local_fd, (struct sockaddr *) &local_addr, sizeof(local_addr));
    if (res == -1) {
        std::cout << "bind error!" << std::endl;
        exit(-1);
    }
    std::cout << "bind ready!" << std::endl;
    listen(local_fd, 2000);    
    printf("start...\n");
    int queuePos = 0;
    while (true) { //循环接收客户端的请求
        //5.创建一个sockaddr_in结构体，用来存储客户机的地址
        struct sockaddr_in client_addr;
        socklen_t len = sizeof(client_addr);
        //6.accept()函数：阻塞运行，直到收到某一客户机的连接请求，并返回客户机的描述符
        int client = accept(local_fd, (struct sockaddr *) &client_addr, &len);
        if (client == -1) {
            exit(-1);
        }

        int size = 0;
        while (true) {
            int cur = read(client, buff + size, sizeof(buff) - size);
            size += cur;
            if (httpChecker.IsValid(buff, size)) {
                break;
            }
        }
        buff[size] = 0;

        while (workQueue.q.size() > workQueue.maxActivateQueryNumber) {
            fastllm::MySleep(0);
        }
        workQueue.Push(buff, client);
    }

    return 0;
}
