// A relocatable /MT launcher: only Windows system DLLs are required.
#include <windows.h>
#include <string>
#include <vector>
#include <cstdio>

static std::wstring quote(const std::wstring &value) {
    std::wstring result = L"\"";
    size_t slashes = 0;
    for (wchar_t ch : value) {
        if (ch == L'\\') { ++slashes; continue; }
        result.append(slashes * (ch == L'"' ? 2 : 1), L'\\');
        slashes = 0;
        if (ch == L'"') result += L'\\';
        result += ch;
    }
    result.append(slashes * 2, L'\\');
    return result + L'"';
}

static BOOL WINAPI control(DWORD event) {
    // The child shares the console and receives Ctrl+C itself. Keep waiting
    // so the launcher returns the child's exit code after graceful shutdown.
    return event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT;
}

int wmain(int argc, wchar_t **argv) {
    std::vector<wchar_t> module(32768);
    DWORD count = GetModuleFileNameW(nullptr, module.data(), DWORD(module.size()));
    if (!count || count == module.size()) return 1;
    std::wstring root(module.data(), count);
    root.resize(root.find_last_of(L"\\/"));
    std::wstring python = root + L"\\runtime\\python.exe";
    std::wstring command = quote(python) + L" -I -B -X utf8 -m ftllm.cli";
    if (argc == 1) command += L" launch";
    for (int i = 1; i < argc; ++i) command += L" " + quote(argv[i]);
    SetEnvironmentVariableW(L"PYTHONUTF8", L"1");
    SetEnvironmentVariableW(L"PYTHONIOENCODING", L"utf-8");
    SetEnvironmentVariableW(L"PYTHONUNBUFFERED", L"1");
    SetConsoleCtrlHandler(control, TRUE);
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    if (!CreateProcessW(python.c_str(), command.data(), nullptr, nullptr, TRUE,
                        0, nullptr, nullptr, &startup, &process)) {
        std::fwprintf(stderr, L"Cannot start bundled Python (Windows error %lu).\n"
                              L"Extract the complete ZIP before running ftllm.exe.\n", GetLastError());
        return 1;
    }
    CloseHandle(process.hThread);
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(process.hProcess, &code);
    CloseHandle(process.hProcess);
    return static_cast<int>(code);
}
