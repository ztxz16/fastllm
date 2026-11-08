"""User-only fixed-label localization; original facts and debug remain unchanged."""
import io
import json
import os
import pathlib
import subprocess
import sys
import unittest

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/"tools/fastllm_pytools"))
from terminal_colorizer import UserTerminalView, process_lines, strip_ansi, COLORS

READY="INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)\n"
ROOT = pathlib.Path(os.environ.get("FASTLLM_TEST_ROOT", os.path.join(os.path.expanduser("~"), ".cache", "fastllm-test")))
(ROOT / 'logs').mkdir(parents=True, exist_ok=True)
def event(category,name,fields="",rid="req-Image-Prefill-000000ab"):
    return ("[fastllm-event] seq=9 timestamp_monotonic_ns=1000000000 "
            "timestamp_utc=2026-10-20T11:14:19.168Z source=python category="+category+
            " request_id="+rid+" native_handle=7 event="+name+" "+fields+"\n")
def render(lines,mode,language):
    log,raw,tty,user=io.StringIO(),io.StringIO(),io.StringIO(),io.StringIO()
    process_lines(lines,log,tty,mode,raw,user,user_language=language)
    return log.getvalue(),raw.getvalue(),tty.getvalue(),user.getvalue()

class UserLanguageTest(unittest.TestCase):
    def test_default_english_and_explicit_english_are_identical(self):
        row=event("progress","request_status","phase=decode native_rate_state=fresh native_interval_tokens_s=68.570 native_output_tokens=475")
        self.assertEqual(UserTerminalView().line(row),UserTerminalView("en").line(row))
        self.assertIn("Native decode | 68.570 tokens/s | 475 output tokens",UserTerminalView().line(row))

    def test_chinese_phase_capacity_labels_copy_raw_values_and_id(self):
        rows=[READY,event("startup","request_received"),
              event("progress","request_status","phase=prefill native_rate_state=fresh native_interval_tokens_s=886.324 prefill_completed_tokens=1024 prefill_total_tokens=1280 l1_capacity_state=ok l1_used_tokens=87424 l1_available_tokens=731264 l1_total_tokens=818688 l2_capacity_state=ok l2_available_gib=16.000 l2_limit_gib=16.000 ssd_capacity_state=ready ssd_available_gib=244.576 ssd_limit_gib=256.000")]
        log,raw,tty,user=render(rows,"user","zh")
        self.assertEqual(log,"".join(rows));self.assertEqual(raw,log)
        self.assertEqual(user,strip_ansi(tty))
        for label in ("请求开始","预填充","缓存容量","L1 已用","剩余","总量","SSD 配额"):
            self.assertIn(label,user)
        for value in ("req-Image-Prefill-000000ab","1024/1280","886.324","87424","731264","818688","16.000","244.576","256.000"):
            self.assertIn(value,user)
        self.assertIn("Prefill",raw);self.assertIn("request_status",raw)
        self.assertIn(COLORS["prefill"],tty);self.assertNotIn(COLORS["progress"],tty);self.assertNotIn("\033[",log)

    def test_off_debug_and_loading_are_not_translated(self):
        loading="Original loading Image Prefill GPU line\n"
        status=event("progress","request_status","phase=decode native_rate_state=fresh native_interval_tokens_s=68.570")
        for mode in ("off","debug"):
            en=render([loading,READY,status],mode,"en")
            zh=render([loading,READY,status],mode,"zh")
            self.assertEqual(en,zh)
        log,raw,tty,user=render([loading,READY,status],"user","zh")
        self.assertTrue(user.startswith(loading+READY))
        self.assertIn("原生解码 | 68.570 tokens/s",user)
        self.assertNotIn("Native decode",user)
        self.assertEqual(raw,loading+READY+status)

    def test_translation_never_rewrites_error_payload_or_original_message(self):
        message="Native decode Image Prefill Failed at path /Image/Prefill/model"
        row=event("error","backend_error","message="+message)
        visible=UserTerminalView("zh").line(row)
        self.assertIn("错误: "+message,visible)
        self.assertIn("[000000ab]",visible)
        self.assertIn("2026-10-20T11:14:19.168Z",visible)

    def test_chinese_has_no_resource_window_or_callback_diagnostics(self):
        rows=[READY,
              event("progress","progress","callback_events=22 python_callback_tokens_s=999.000"),
              event("progress","request_status","phase=decode native_rate_state=fresh native_interval_tokens_s=68.570 native_output_tokens=475 native_interval_ms=1006.273 native_sample_age_ms=44.000 gpu_device_indices=0,1,2,3,4 gpu_util_max_pct=74 gpu_mem_used_mib=63875 proc_rss_mib=10134 host_mem_available_mib=49285 resource_sample_age_ms=831")]
        log,raw,tty,user=render(rows,"user","zh")
        self.assertIn("原生解码 | 68.570 tokens/s | 475 输出 tokens",user)
        for value in ("999.000","1006.273","44.000","0,1,2,3,4","63875","10134","49285","831"):
            self.assertNotIn(value,user);self.assertIn(value,raw)
        self.assertEqual(raw,log)

    def test_actual_cli_user_language_option_only_changes_user_stream(self):
        run=pathlib.Path(os.environ["FASTLLM_COMPACT_RUN"])/"language-cli"
        run.mkdir(exist_ok=False)
        module=pathlib.Path(__file__).resolve().parents[2]/"tools/fastllm_pytools/terminal_colorizer.py"
        data=READY+event("startup","request_received")+event("progress","request_status","phase=decode native_rate_state=fresh native_interval_tokens_s=68.570 native_output_tokens=475")
        results={}
        for language in ("en","zh"):
            paths={name:run/(language+"-"+name+".log") for name in ("server","raw","user","tty")}
            command=[sys.executable,"-B",str(module),"--mode","user","--user-language",language,
                     "--log",str(paths["server"]),"--raw-log",str(paths["raw"]),
                     "--user-log",str(paths["user"]),"--tty",str(paths["tty"])]
            with (ROOT/"logs/actions.jsonl").open("a") as f:
                f.write(json.dumps({"phase":"before","task":"user-language-cli-t0","command":command,
                                    "cwd":str(module.parent),"log_paths":{k:str(v) for k,v in paths.items()},
                                    "expected":"CPU-only stream formatting; no model or GPU"})+"\n")
                f.flush();os.fsync(f.fileno())
            process=subprocess.run(command,input=data,text=True,capture_output=True)
            with (ROOT/"logs/actions.jsonl").open("a") as f:
                f.write(json.dumps({"phase":"after","task":"user-language-cli-t0","command":command,
                                    "exit_code":process.returncode,"stderr":process.stderr,
                                    "resource_cleanup":"no runtime resources started"})+"\n")
                f.flush();os.fsync(f.fileno())
            self.assertEqual(process.returncode,0,process.stderr)
            self.assertEqual(paths["server"].read_text(),data)
            self.assertEqual(paths["raw"].read_text(),data)
            self.assertEqual(paths["user"].read_text(),strip_ansi(paths["tty"].read_text()))
            results[language]=paths["user"].read_text()
        self.assertIn("Native decode | 68.570 tokens/s",results["en"])
        self.assertIn("原生解码 | 68.570 tokens/s",results["zh"])

if __name__=="__main__":unittest.main()
