//
// Created by huangyuyang on 5/17/24.
//

#include "ui.h"
#include "model.h"
#include "utils.h"

int curRound = 0;
std::string history = "";

// RunConfig config;
fastllm::GenerationConfig generationConfig;

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: ./FastllmStudio_cli modelpath\n");
        return 0;
    }
    std::string modelPath = argv[1];

    fastllmui::HideCursor();
    fastllmui::ClearScreen();

    fastllmui::Menu mainMenu = fastllmui::Menu({"打开模型", "退出"});
    int ret = mainMenu.Show();
    if (ret == 0) {
        fastllmui::ClearScreen();
        char *buffer = new char[1005];
        fastllm::SetThreads(16);
        printf("正在载入模型 %s\n", modelPath.c_str());
        auto model = fastllm::CreateLLMModelFromFile(modelPath);
        model->SetSaveHistoryChat(true);

        fastllmui::ClearScreen();
        fastllmui::Menu openModelMenu = fastllmui::Menu({"对话", "设置", "退出"});
        int ret = openModelMenu.Show();
        if (ret == 0) {
            fastllmui::ClearScreen();
            static std::string modelType = model->model_type;
            printf("欢迎使用 %s 模型. 输入内容对话，reset清空历史记录，stop退出对话.\n", model->model_type.c_str());
            while (true) {
                printf("用户: ");
                std::string input;
                std::getline(std::cin, input);
                if (input == "reset") {
                    fastllmui::ClearScreen();
                    printf("欢迎使用 %s 模型. 输入内容对话，reset清空历史记录，stop退出对话.\n", model->model_type.c_str());
                    history = "";
                    curRound = 0;
                    continue;
                }
                if (input == "stop") {
                    break;
                }
                std::string ret = model->Response(model->MakeInput(history, curRound, input), [](int index, const char* content) {
                    static auto st = std::chrono::system_clock::now();
                    static int cnt = 0;
                    if (index == 0) {
                        st = std::chrono::system_clock::now();
                        cnt = 0;
                        printf("%s:%s", modelType.c_str(), content);
                        fflush(stdout);
                    }
                    if (index > 0) {
                        cnt++;
                        printf("%s", content);
                        fflush(stdout);
                    }
                    if (index == -1) {
                        printf("\n");
                        printf("tokens speed: %f tokens / s.\n", cnt / fastllm::GetSpan(st, std::chrono::system_clock::now()));
                    }
                }, generationConfig);
                history = model->MakeHistory(history, curRound, input, ret);
                curRound++;
            }
        }
    }
    if (ret == 1) {
        fastllmui::ClearScreen();
        fastllmui::ShowCursor();
        exit(0);
    }

    return 0;
}