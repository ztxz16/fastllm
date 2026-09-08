param([Parameter(Mandatory=$true)][int]$ProcessId)
$ErrorActionPreference = "Stop"
Add-Type @'
using System;
using System.Runtime.InteropServices;
using System.Text;
public static class DesktopSmokeWindow {
    public delegate bool WindowCallback(IntPtr window, IntPtr parameter);
    [DllImport("user32.dll")] public static extern bool EnumWindows(WindowCallback callback, IntPtr parameter);
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr window, out uint processId);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetWindowText(IntPtr window, StringBuilder text, int count);
    [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr window, uint message, IntPtr wparam, IntPtr lparam);
}
'@
$script:FoundDesktopWindow = $false
[DesktopSmokeWindow]::EnumWindows({
    param($window, $parameter)
    [uint32]$owner = 0
    [void][DesktopSmokeWindow]::GetWindowThreadProcessId($window, [ref]$owner)
    if ($owner -eq $ProcessId) {
        $title = New-Object Text.StringBuilder 256
        [void][DesktopSmokeWindow]::GetWindowText($window, $title, $title.Capacity)
        if ($title.ToString() -eq "FastLLM Launcher") {
            $script:FoundDesktopWindow = $true
            [void][DesktopSmokeWindow]::PostMessage($window, 0x0010, [IntPtr]::Zero, [IntPtr]::Zero)
        }
    }
    return $true
}, [IntPtr]::Zero) | Out-Null
if (-not $script:FoundDesktopWindow) { throw "No FastLLM Electron window owned by PID $ProcessId" }
