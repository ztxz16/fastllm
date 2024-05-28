//
// Created by huangyuyang on 5/17/24.
//

#ifndef FASTLLMUI_H
#define FASTLLMUI_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

namespace fastllmui {
    void HideCursor();
    void ShowCursor();

    void ClearScreen();
    void CursorUp();
    void CursorDown();
    void CursorHome();
    void CursorClearLine();

    struct Menu {
        std::vector <std::string> items;
        int curIndex = 0;

        Menu (std::vector <std::string> items) :
            items(items) {}
        
        int Show();
    };
}

#endif