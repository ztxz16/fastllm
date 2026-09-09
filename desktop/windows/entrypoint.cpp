// Small /MT entrypoints keep Electron, Python and their DLLs in support/.
#include <windows.h>
#include <shellapi.h>
#include <cstdio>
#include <string>
#include <vector>

static std::wstring Quote(const std::wstring &value) {
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

static BOOL WINAPI ConsoleControl(DWORD event) {
    // Children share the console and receive Ctrl+C directly.
    return event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT;
}

static int Run(int argc, wchar_t **argv) {
    std::vector<wchar_t> module(32768);
    DWORD length = GetModuleFileNameW(nullptr, module.data(), DWORD(module.size()));
    if (!length || length == module.size()) return 1;
    std::wstring path(module.data(), length);
    size_t separator = path.find_last_of(L"\\/");
    std::wstring root = path.substr(0, separator);
    std::wstring name = path.substr(separator + 1);
    std::wstring support = root + L"\\support";
    bool desktop = _wcsicmp(name.c_str(), L"FastLLM-Launcher.exe") == 0;
    bool browser = _wcsicmp(name.c_str(), L"ftllm-launch-webui.exe") == 0;
    std::wstring executable = support + (desktop ? L"\\FastLLM-Launcher.exe" : L"\\ftllm.exe");
    std::wstring command = Quote(executable);
    if (browser) command += L" launch";
    DWORD consoleProcesses[2];
    bool terminal = !desktop && !browser && argc == 1 &&
                    GetConsoleProcessList(consoleProcesses, 2) == 1;
    if (terminal) {
        wchar_t system[MAX_PATH];
        if (!GetSystemDirectoryW(system, MAX_PATH)) return 1;
        executable = std::wstring(system) + L"\\WindowsPowerShell\\v1.0\\powershell.exe";
        command = Quote(executable) + L" -NoLogo -NoProfile -NoExit -ExecutionPolicy Bypass -File " +
                  Quote(support + L"\\env.ps1");
    } else {
        for (int i = 1; i < argc; ++i) command += L" " + Quote(argv[i]);
    }
    SetConsoleCtrlHandler(ConsoleControl, TRUE);
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    if (!CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr, TRUE,
                        0, nullptr, terminal ? root.c_str() : nullptr, &startup, &process)) {
        std::wstring error = L"Cannot start FastLLM (Windows error " + std::to_wstring(GetLastError()) +
            L").\nExtract the complete ZIP and keep the support folder beside the entrypoints.";
        if (desktop) MessageBoxW(nullptr, error.c_str(), L"FastLLM Launcher", MB_OK | MB_ICONERROR);
        else std::fwprintf(stderr, L"%ls\n", error.c_str());
        return 1;
    }
    CloseHandle(process.hThread);
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(process.hProcess, &code);
    CloseHandle(process.hProcess);
    return static_cast<int>(code);
}

#ifdef FTLLM_GUI_ENTRYPOINT
int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    int argc = 0;
    wchar_t **argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (!argv) return 1;
    int result = Run(argc, argv);
    LocalFree(argv);
    return result;
}
#else
int wmain(int argc, wchar_t **argv) { return Run(argc, argv); }
#endif
