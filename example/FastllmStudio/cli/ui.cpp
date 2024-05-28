//
// Created by huangyuyang on 5/17/24.
//

#include "ui.h"

namespace fastllmui {
    inline char getCh(){
        static char ch;
        int ret = system("stty -icanon -echo");
        ret = scanf("%c", &ch);
        ret = system("stty icanon echo");
        return ch;
    }

    void PrintNormalLine(const std::string &line) {
        printf("%s", line.c_str());
    }

    void PrintHighlightLine(const std::string &line) {
        printf("\e[1;31;40m %s \e[0m",  line.c_str());
    }

    void HideCursor() {
        printf("\033[?25l");
    }

    void ShowCursor() {
        printf("\033[?25h");
    }

    void ClearScreen() {
        printf("\033c");
    }

    void CursorUp() {
        printf("\033[F");
    }

    void CursorDown() {
        printf("\033[B");
    }

    void CursorClearLine() {
        printf("\033[1G");
        printf("\033[K");
    }

    int Menu::Show() {
        for (int i = 0; i < items.size(); i++) {
            if (i == curIndex) {
                PrintHighlightLine(items[i]);
                printf("\n");
            } else {
                PrintNormalLine(items[i]);
                printf("\n");
            }
        }

        for (int i = curIndex; i < items.size(); i++) {
            printf("\033[F");
        }

        std::string upString = {27, 91, 65};
        std::string downString = {27, 91, 66};
        std::string now = "";
        while (true) {
            char ch = getCh();
            if (ch == '\r' || ch == '\n') {
                return curIndex;
            } else {
                now += ch;
                if (now.size() >= 3 && now.substr(now.size() - 3) == downString) {
                    if (curIndex + 1 < items.size()) {
                        CursorClearLine();
                        PrintNormalLine(items[curIndex++]);
                        CursorDown();
                        CursorClearLine();
                        PrintHighlightLine(items[curIndex]);
                    }
                } else if (now.size() >= 3 && now.substr(now.size() - 3) == upString) {
                    if (curIndex - 1 >= 0) {
                        CursorClearLine();
                        PrintNormalLine(items[curIndex--]);
                        CursorUp();
                        CursorClearLine();
                        PrintHighlightLine(items[curIndex]);
                    }
                }
            }
        }
    }
} // namespace fastllmui
