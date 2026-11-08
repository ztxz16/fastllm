#!/usr/bin/env python3
"""Human-readable terminal over the same unmodified raw events and colors."""
import ast,io,os,re,subprocess,sys,time,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools/fastllm_pytools"))
from terminal_colorizer import UserTerminalView,colorize_line,colorize_user_line,process_lines,strip_ansi
from request_progress import RequestProgressReporter
READY="INFO:     Uvicorn running on http://0.0.0.0:8080\n"
def event(category,name,fields="",source="native",rid="req-7"):
    return f"[fastllm-event] seq=1 timestamp_monotonic_ns=42 timestamp_utc=2026-10-17T00:00:00Z source={source} category={category} request={rid} request_id={rid} native_handle=7 event={name} {fields}\n"
def render(lines,mode,tty=True):
    log,raw,terminal=io.StringIO(),io.StringIO(),io.StringIO()
    process_lines(lines,log,terminal if tty else None,mode,raw)
    return log.getvalue(),raw.getvalue(),terminal.getvalue()

class HumanUserTest(unittest.TestCase):
    def test_all_loading_output_is_untouched_in_every_mode(self):
        lines=["Namespace()\n","[Vision] warmup\n","[DFlash] init\n",event("warning","loading"),READY]
        for mode in ("off","user","debug"):
            self.assertEqual(render(lines,mode),("".join(lines),)*3)
    def test_phrase_inside_event_is_not_a_ready_marker(self):
        lines=[event("warning","load","message=Uvicorn running on example"),"[Decode] loading\n",READY,"[Decode] inference\n"]
        self.assertEqual(render(lines,"user")[2],"".join(lines[:-1]))
    def test_full_logs_survive_and_only_user_terminal_is_readable(self):
        lines=[READY,event("decode","dflash","draft=8 accepted=3"),event("cache","cache_lookup"),event("cache","cache_result","source_tier=L1 effective_reused_tokens=128"),"\033[31mERROR: failure\033[0m\n"]
        for mode in ("off","user","debug"):
            log,raw,tty=render(lines,mode);self.assertEqual(log,strip_ansi("".join(lines)));self.assertEqual(log,raw)
            if mode=="user":
                plain=strip_ansi(tty).split(READY,1)[1]
                self.assertIn("L1 cache: hit 128 tokens",plain)
                for internal in ("[fastllm-event]","source=","category=","event=","draft=","cache_lookup"):
                    self.assertNotIn(internal,plain)
            else:self.assertEqual(strip_ansi(tty),log)
    def test_no_terminal_still_keeps_every_event(self):
        lines=[READY,event("scheduler","state","pending=2")]
        log,raw,tty=render(lines,"user",False);self.assertEqual(log,"".join(lines));self.assertEqual(raw,log);self.assertEqual(tty,"")
    def test_l1_l2_are_separate_and_never_summed(self):
        lines=[event("cache","cache_result","source_tier=L1 promoted_to=NONE effective_reused_tokens=128"),event("cache","cache_result","source_tier=L2 promoted_to=L1 effective_reused_tokens=256")]
        plain=strip_ansi(render([READY]+lines,"user")[2])
        self.assertIn("L1 cache: hit 128 tokens",plain);self.assertIn("L2 cache: hit 256 tokens -> L1",plain);self.assertNotIn("384",plain)
    def test_ssd_real_values_are_readable_without_rounding(self):
        line=event("cache","cache_result","source_tier=L3 promoted_to=L1 effective_reused_tokens=39936 lookup_ms=102.346 restore_ms=4340.834 read_mib_s=225.114 restore_tokens_s=3816.739")
        result=UserTerminalView().line(line)
        for text in ("SSD cache: hit 39936 tokens -> L1","read: 225.114 MiB/s","restore: 3816.739 tokens/s","import time: 4340.834 ms"):self.assertIn(text,result)
        self.assertNotIn("lookup_ms",result)
        missing=UserTerminalView().line(event("cache","cache_result","source_tier=L3 effective_reused_tokens=128"))
        self.assertNotIn("MiB/s",missing);self.assertNotIn("tokens/s",missing)
    def test_duplicate_cache_result_hidden_only_in_user_terminal(self):
        line=event("cache","cache_result","source_tier=L1 promoted_to=NONE source_tokens=128 effective_reused_tokens=128 lookup_ms=0.500")
        log,raw,tty=render([READY,line,line],"user")
        self.assertEqual(log,READY+line+line);self.assertEqual(raw,log)
        self.assertEqual(strip_ansi(tty).count("L1 cache: hit 128 tokens"),1)

    def test_zero_local_queries_are_visible_before_ssd_restore(self):
        lines=[READY,
               event("startup","request_received",source="python"),
               event("cache","cache_candidates","l2_query_status=ok l1_resident_tokens=0 l2_host_tokens=0 combined_local_tokens=0"),
               event("cache","cache_restore_begin","tier=L3(SSD) phase=read candidate_tokens=39936 ticket=1"),
               event("cache","cache_result","source_tier=L3 promoted_to=L1 effective_reused_tokens=39936 read_mib_s=222.196 restore_tokens_s=3782.248 restore_ms=4355.928")]
        log,raw,tty=render(lines,"user");plain=strip_ansi(tty)
        stages=["Request started","L1 prefix candidate: 0 tokens","L2 prefix candidate: 0 tokens","SSD cache: reading 39936 tokens","SSD cache: hit 39936 tokens -> L1","read: 222.196 MiB/s","restore: 3782.248 tokens/s"]
        positions=[plain.index(x) for x in stages];self.assertEqual(positions,sorted(positions))
        self.assertEqual(log,"".join(lines));self.assertEqual(raw,log)
        self.assertNotIn("ticket",plain);self.assertNotIn("category=",plain)

    def test_local_candidate_counts_do_not_claim_effective_reuse(self):
        lines=[READY,event("cache","cache_candidates","l2_query_status=ok l1_resident_tokens=128 l2_host_tokens=256 combined_local_tokens=384"),
               event("cache","cache_result","source_tier=L3 promoted_to=L1 effective_reused_tokens=1024")]
        plain=strip_ansi(render(lines,"user")[2])
        self.assertIn("L1 prefix candidate: 128 tokens",plain);self.assertIn("L2 prefix candidate: 256 tokens",plain)
        self.assertIn("SSD cache: hit 1024 tokens",plain);self.assertNotIn("L1 cache: hit 128",plain)
        self.assertNotIn("1408",plain)

    def test_l2_non_ok_state_is_not_synthetic_zero_or_miss(self):
        for status in ("busy","disabled","duplicate","incomplete","unknown"):
            v=UserTerminalView();line=v.line(event("cache","cache_candidates",f"l2_query_status={status} l1_resident_tokens=128 l2_host_tokens=0"))
            self.assertIn("L1 prefix candidate: 128 tokens",line);self.assertIn(f"L2 cache: {status}",line)
            self.assertNotIn("L2 cache: 0",line);self.assertNotIn("MISS",line)

    def test_missing_local_count_is_not_filled_with_zero(self):
        v=UserTerminalView();line=v.line(event("cache","cache_candidates","l2_query_status=ok"))
        self.assertNotIn("0 tokens",line);self.assertIn("L2 cache: unknown",line)

    def test_ssd_restore_visibility_is_immediate(self):
        lines=iter([READY,event("cache","cache_candidates","l2_query_status=ok l1_resident_tokens=0 l2_host_tokens=0"),
                    event("cache","cache_restore_begin","tier=L3(SSD) phase=read candidate_tokens=39936")])
        log,raw,tty=render(lines,"user");self.assertIn("SSD cache: reading 39936 tokens",strip_ansi(tty))
        self.assertNotIn("SSD cache: hit",strip_ansi(tty));self.assertEqual(log,raw)

    def test_uncached_prefill_and_native_speed_come_from_native_events(self):
        v=UserTerminalView()
        self.assertIn("Uncached: 2630 tokens",v.line(event("prefill","prefill_summary","missed_input_tokens=2630 prefill_tokens=2630")))
        self.assertIn("Prefill: 2630 tokens | 123.456 tokens/s | 21304.755 ms",v.line(event("prefill","prefill_speed","prefill_tokens=2630 prefill_tokens_s=123.456 elapsed_ms=21304.755")))
        self.assertIsNone(v.line(event("decode","native_decode","native_tokens_s=45.678 native_decode_tokens=31")))
        self.assertIn("First token: 5678.901 ms",v.line(event("output","native_first_token","ttft_ms=5678.901")))
    def test_callback_aliases_and_heartbeat_first_token_not_displayed(self):
        v=UserTerminalView();self.assertIsNone(v.line(event("progress","progress","decode_tokens_s=999 output_tokens_s=999",source="python")))
        self.assertIsNone(v.line(event("output","first_token","ttft_ms=5000",source="python")))
        done=v.line(event("output","request_finished","status=ok output_tokens=12 missed_input_tokens=3 elapsed_ms=500 request_total_average_tokens_s=24",source="python"))
        self.assertIn("Uncached: 3 tokens",done);self.assertIn("Finished: ok | 12 output tokens | 500 ms",done);self.assertNotIn("tokens/s",done)
    def test_phase_heartbeat_is_direct_raw_projection_for_python_and_native(self):
        lines=[READY,
               event("progress","phase_status","phase=image_preprocessing detail=loading_and_decoding_image phase_elapsed_ms=1000.123 idle_ms=1000.123",source="python",rid="image-request"),
               event("progress","phase_status","phase=prefill detail=forward_chunk phase_elapsed_ms=2000.456 idle_ms=999.321 completed_tokens=2048 total_tokens=42365",rid="text-request")]
        log,raw,tty=render(lines,"user");plain=strip_ansi(tty)
        self.assertEqual(log,"".join(lines));self.assertEqual(raw,log)
        self.assertIn("Status: image preprocessing | loading and decoding image | elapsed: 1000.123 ms | last update: 1000.123 ms ago",plain)
        self.assertIn("Status: prefill | forward chunk | elapsed: 2000.456 ms | last update: 999.321 ms ago | completed: 2048 tokens | total: 42365 tokens",plain)
        self.assertNotIn("tokens/s",plain)
        self.assertNotIn("completed:",plain.splitlines()[1])
        # Missing or explicitly unknown counts are retained as unknown; the
        # terminal never turns elapsed time into a progress percentage/rate.
        missing=UserTerminalView().line(event("progress","phase_status","phase=queued detail=waiting_for_scheduler idle_ms=3000.000"))
        self.assertNotIn("completed:",missing);self.assertNotIn("total:",missing)
        unknown=UserTerminalView().line(event("progress","phase_status","phase=ssd_read completed_tokens=unknown total_tokens=unknown"))
        self.assertIn("completed: unknown tokens",unknown)
        self.assertIn("total: unknown tokens",unknown)
    def test_unavailable_native_rate_is_unknown_not_zero(self):
        line=UserTerminalView().line(event("output","native_generation_summary","native_rate_available=0"))
        self.assertIn("Native decode: unknown",line);self.assertNotIn("0 tokens/s",line)
    def test_multimodal_labels_and_prefix_are_separate(self):
        v=UserTerminalView();prefix=v.line(event("cache","cache_result","source_tier=L1 effective_reused_tokens=39936"))
        v.line(event("vision","vision_encode_begin","images=1 video=0"))
        miss=v.line(event("vision","vision_cache_lookup_result","hit=0 lookup_ms=0.000"));encoded=v.line(event("vision","vision_encode_end","elapsed_ms=2634.685 features=1380 cache_misses=1"))
        self.assertIn("L1 cache: hit 39936 tokens",prefix);self.assertIn("Image cache: MISS",miss);self.assertIn("Image processing: 2634.685 ms",encoded)
        restore=v.line(event("vision","vision_cache_restore","tier=L1(GPU/RAM) bytes=1024"));self.assertNotIn("GPU",restore)
        v.line(event("vision","vision_encode_begin","images=1 video=1"));self.assertIn("Video cache: HIT",v.line(event("vision","vision_cache_lookup_result","hit=1")))
    def test_public_id_once_and_concurrent_requests_keep_short_ids(self):
        first="fastllm-request-alpha-0000000a";second="fastllm-request-beta-0000000b"
        lines=[event("startup","request_received",source="python",rid=first),
               event("startup","request_received",source="python",rid=second),
               event("cache","cache_result","source_tier=L1 effective_reused_tokens=128",rid=first),
               event("vision","vision_cache_lookup_result","hit=0",rid=second)]
        plain=strip_ansi(render([READY]+lines,"user")[2])
        self.assertEqual(plain.count("ID: "+first),1);self.assertEqual(plain.count("ID: "+second),1)
        self.assertEqual(plain.count(first),1);self.assertEqual(plain.count(second),1)
        self.assertRegex(plain,r"\["+first[-8:]+r"\] (?:\[[^\]]+\] )?L1 cache")
        self.assertRegex(plain,r"\["+second[-8:]+r"\] (?:\[[^\]]+\] )?Vision cache")

    def test_native_finish_stays_raw_and_tool_precedes_api_finish(self):
        lines=[READY,event("output","native_generation_finished","reason=eos output_tokens=12"),event("tool","tool_call","count=1",source="python"),event("output","request_finished","status=ok output_tokens=12 elapsed_ms=500",source="python")]
        log,raw,tty=render(lines,"user");plain=strip_ansi(tty);self.assertEqual(log,raw);self.assertIn("native_generation_finished",raw)
        self.assertEqual(plain.count("Finished"),1);self.assertLess(plain.index("Tool call"),plain.index("Finished"))
    def test_raw_field_order_and_numeric_values_are_never_rewritten(self):
        line=event("cache","cache_result","source_tier=L3 effective_reused_tokens=3072 read_mib_s=375.731 restore_tokens_s=3132.587")
        log,raw,tty=render([READY,line],"user");self.assertEqual(raw,READY+line);self.assertEqual(log,raw)
        for number in ("3072","375.731","3132.587"):self.assertIn(number,tty)
    def test_request_label_is_raw_id_suffix_with_no_user_counter(self):
        view=UserTerminalView();rid="fastllm-Qwen3.8-27B-FP8-3470492af4514d4d9ff55fa47e39a3c2"
        line=view.line(event("startup","request_received",source="python",rid=rid))
        self.assertTrue(line.startswith("["+rid[-8:]+"] "))
        self.assertIn("ID: "+rid,line)
        self.assertFalse(hasattr(view,"_next"))
        nextline=view.line(event("cache","cache_candidates","l2_query_status=ok l1_resident_tokens=0 l2_host_tokens=0",rid=rid))
        self.assertTrue(nextline.startswith("["+rid[-8:]+"] "))
        self.assertNotIn(rid,nextline)

    def test_user_view_has_no_independent_measurements_or_numeric_arithmetic(self):
        repo=Path(__file__).resolve().parents[2]
        module=ast.parse((repo/"tools/fastllm_pytools/terminal_colorizer.py").read_text())
        view=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=="UserTerminalView")
        forbidden_calls={"int","float","sum","max","min","round","monotonic","monotonic_ns","perf_counter","perf_counter_ns","time","tokenize"}
        for node in ast.walk(view):
            if isinstance(node,ast.AugAssign):
                string_rhs = isinstance(node.value,ast.JoinedStr) or (
                    isinstance(node.value,ast.Constant) and isinstance(node.value.value,str))
                self.assertTrue(isinstance(node.op,ast.Add) and string_rhs,
                                "user counters and numeric updates are forbidden; literal string formatting is allowed")
            if isinstance(node,ast.BinOp):
                self.assertNotIsInstance(node.op,(ast.Sub,ast.Mult,ast.Div,ast.FloorDiv,ast.Mod,ast.Pow),
                                         "user must copy raw values, not calculate measurements")
            if isinstance(node,ast.Call):
                name=node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else None
                self.assertNotIn(name,forbidden_calls,"user must not count, time, retokenize, or calculate metrics")

    def test_user_fields_are_strict_subset_of_actual_debug_raw_producers(self):
        from collections import defaultdict
        from terminal_colorizer import _USER_EVENT_FIELDS
        repo = Path(__file__).resolve().parents[2]
        schema = defaultdict(set)
        # Extract category/event and key=value format names from the actual
        # C++ emitter calls, including adjacent C++ string literals.
        emitter = re.compile(
            r'FastllmEvent\s*\(\s*"([^"\n]+)"\s*,\s*[^,;]+,\s*"([^"\n]+)"\s*'
            r'(?:,\s*((?:"(?:\\.|[^"\\])*"\s*)+))?', re.S)
        copied_identity_emitter = re.compile(
            r'FastllmEventWithPublicId\s*\(\s*"([^"\n]+)"\s*,\s*[^,;]+,\s*[^,;]+,\s*"([^"\n]+)"\s*'
            r'(?:,\s*((?:"(?:\\.|[^"\\])*"\s*)+))?', re.S)
        for name in ("src/models/basellm.cpp", "src/models/qwen3_5.cpp",
                     "src/models/qwen3_5_persistent.inc", "include/utils/request_phase_observer.h"):
            for producer in (emitter, copied_identity_emitter):
                for match in producer.finditer((repo / name).read_text()):
                    schema[(match[1], match[2])].update(
                        re.findall(r'\b([A-Za-z_]\w*)=', match[3] or ""))
        # Python literal event kwargs and method-local field dictionaries are
        # also facts exposed by the shared raw producer, not a user schema copy.
        tree = ast.parse((repo / "tools/fastllm_pytools/request_progress.py").read_text())
        for func in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
            local_fields = set()
            for node in ast.walk(func):
                if isinstance(node, ast.Dict):
                    local_fields.update(k.value for k in node.keys
                                        if isinstance(k, ast.Constant) and isinstance(k.value, str))
                if (isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
                        and isinstance(node.slice.value, str)):
                    local_fields.add(node.slice.value)
            for node in ast.walk(func):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "_event" and len(node.args) >= 2
                        and all(isinstance(x, ast.Constant) and isinstance(x.value, str)
                                for x in node.args[:2])):
                    key = tuple(n.value for n in node.args[:2])
                    schema[key].update(k.arg for k in node.keywords if k.arg)
                    if any(k.arg is None for k in node.keywords):
                        schema[key].update(local_fields)
        native_status=(repo/"include/utils/request_status.h").read_text()
        native_status+=(repo/"src/models/qwen3_5.cpp").read_text()
        status_schema=set(re.findall(r'(?:fields|snapshot.fields)\["([A-Za-z_]\w*)"\]',native_status))
        status_schema.update(re.findall(r'\{\s*"([A-Za-z_]\w*)"\s*,',native_status))
        for tier in ("l2","ssd"):
            for suffix in ("used_gib","available_gib","limit_gib"):
                status_schema.add(tier+"_"+suffix)
        for node in ast.walk(tree):
            if isinstance(node,ast.Subscript) and isinstance(node.slice,ast.Constant) and isinstance(node.slice.value,str):
                status_schema.add(node.slice.value)
            if isinstance(node,ast.Dict):
                status_schema.update(k.value for k in node.keys if isinstance(k,ast.Constant) and isinstance(k.value,str))
        schema[("progress","request_status")].update(status_schema)
        for key, fields in _USER_EVENT_FIELDS.items():
            with self.subTest(event=key):
                self.assertIn(key, schema, "user-only event has no raw/debug producer")
                self.assertTrue(fields <= schema[key],
                                (key, "user-only fields", fields - schema[key]))

    def test_untyped_vision_event_does_not_invent_image_or_video_kind(self):
        view=UserTerminalView()
        line=view.line(event("vision","vision_cache_lookup_result","hit=0"))
        self.assertIn("Vision cache: MISS",line)
        self.assertNotIn("Image",line);self.assertNotIn("Video",line)
        line=view.line(event("vision","vision_encode_begin","images=1"))
        self.assertIn("Vision processing started",line)

    def test_user_does_not_predeclare_unsupported_debug_fields(self):
        view=UserTerminalView()
        line=view.line(event("cache","cache_result","source_tier=L3 effective_reused_tokens=128 read_mib_s=22.123 promote_ms=777 read_ms=888"))
        self.assertIn("read: 22.123 MiB/s",line)
        self.assertNotIn("777",line);self.assertNotIn("888",line)
        tool=view.line(event("tool","tool_call","count=1 name=not_in_raw_producer",source="python"))
        self.assertIn("count: 1",tool);self.assertNotIn("not_in_raw_producer",tool)

    def test_protected_helpers_are_unchanged_and_cache_palette_is_pinned(self):
        path=Path(__file__).resolve().parents[2]/"tools/fastllm_pytools/terminal_colorizer.py"
        before=subprocess.run(["git","show","bf0bb92d:tools/fastllm_pytools/terminal_colorizer.py"],cwd=path.parents[2],text=True,capture_output=True,check=True).stdout
        def protected(code):
            result={}
            for node in ast.parse(code).body:
                name=node.name if isinstance(node,ast.FunctionDef) else next((t.id for t in node.targets if isinstance(t,ast.Name)),None) if isinstance(node,ast.Assign) else None
                if name in {"ANSI_RESET","_ansi256","colorize_line","category_for_line","_legacy_category"}:result[name]=ast.dump(node)
            return result
        self.assertEqual(protected(before),protected(path.read_text()))
        from terminal_colorizer import CACHE_COLORS
        self.assertEqual([CACHE_COLORS[t] for t in ("L1","L2","L3","MISS")],
                         ["\033[38;5;226m","\033[38;5;178m","\033[38;5;136m","\033[38;5;214m"])
        for tier in ("L1","L2","L3","MISS"):
            line=event("cache","cache_result",f"source_tier={tier} effective_reused_tokens=128");human=UserTerminalView().line(line)
            colored=colorize_user_line(line,human);self.assertEqual(strip_ansi(colored),human)
            self.assertEqual(re.match(r"(?:\x1b\[[0-9;]+m)+",colored)[0],re.match(r"(?:\x1b\[[0-9;]+m)+",colorize_line(line))[0])
    def test_errors_and_tracebacks_survive(self):
        lines=[READY,"WARNING: delayed\n","Traceback (most recent call last):\n",'  File "example.py", line 1\n','    raise ValueError("bad")\n',"ValueError: bad\n",event("scheduler","state")]
        self.assertEqual(strip_ansi(render(lines,"user")[2]),"".join(lines[:-1]))
    def test_python_user_debug_raw_schemas_match(self):
        from unittest.mock import patch
        streams=[]
        # Compare the two modes against the SAME factual snapshot. Independent
        # wall-clock sampling ages are expected to differ across executions.
        snapshot={"resource_sample_state":"fresh","resource_sample_age_ms":"123",
                  "gpu_sample_state":"scope_unknown","proc_rss_mib":"456"}
        with patch("request_progress.resource_snapshot",return_value=snapshot), patch(
                "request_progress.time.monotonic_ns",return_value=123456789):
            for mode in ("user","debug"):
                now=[0.0];events=[]
                r=RequestProgressReporter("req",emit=events.append,mode=mode,clock=lambda:now[0])
                r.cache_lookup();r.vision_lookup();now[0]=1.0;r.callback({"output_tokens":4})
                r.tool_call();now[0]=2.0;r.finish(output_tokens=4)
                streams.append([re.sub(r"(?:seq|timestamp_monotonic_ns|timestamp_utc)=\S+","",x) for x in events])
        self.assertEqual(*streams)

    def test_native_progress_is_projected_only_after_ready(self):
        line=event("progress","request_status","phase=decode native_rate_state=fresh native_interval_ms=1039.000 native_interval_tokens_s=41.47")
        log,raw,tty=render([line,READY,line],"user")
        self.assertEqual(log,line+READY+line);self.assertEqual(raw,log)
        before,after=strip_ansi(tty).split(READY,1)
        self.assertIn("event=request_status",before)
        self.assertIn("Native decode | 41.47 tokens/s",after)
        self.assertNotIn("native_interval_tokens_s",after)

    def test_progress_user_values_are_copied_from_the_same_raw_event(self):
        line=event("progress","request_status","phase=prefill native_rate_state=fresh native_interval_ms=1034.000 native_interval_tokens_s=675.83 prefill_completed_tokens=699 prefill_total_tokens=2000")
        body=strip_ansi(render([READY,line],"user")[2]).split(READY,1)[1].split("] ",1)[1]
        self.assertIn("Prefill | 699/2000 tokens completed | 675.83 tokens/s",body)
        for number in re.findall(r"\d+(?:\.\d+)?",body):
            self.assertIn(number,line,"user displayed a number absent from the same raw event")

    def test_progress_omits_absent_layers_instead_of_filling_zero(self):
        line=event("progress","request_status","phase=prefill native_rate_state=pending")
        body=strip_ansi(render([READY,line],"user")[2]).split(READY,1)[1]
        self.assertIn("Prefill | speed pending",body)
        self.assertIn("L1 capacity unknown",body)
        self.assertNotIn("0 tokens",body);self.assertNotIn("0.00",body)

    def test_python_callback_progress_is_labelled_as_a_separate_source(self):
        py=event("progress","progress","measurement=python_callback_interval elapsed_s=1.035 callback_events=12 python_callback_events_s=11.60 python_callback_tokens_s=18.37",source="python")
        native=event("decode","native_decode","native_tokens_s=99.99 native_decode_tokens=31")
        status=event("progress","request_status","phase=decode native_rate_state=fresh native_interval_tokens_s=41.47")
        log,raw,tty=render([READY,py,native,status],"user")
        self.assertEqual(log,READY+py+native+status);self.assertEqual(log,raw)
        body=strip_ansi(tty).split(READY,1)[1]
        self.assertEqual(len([row for row in body.strip().splitlines() if "Cache capacity |" not in row]),1)
        self.assertIn("Native decode | 41.47 tokens/s",body)
        for diagnostic in ("Callback","events/s","18.37","99.99"):self.assertNotIn(diagnostic,body)
    def test_user_view_is_persisted_separately_from_debug_logs(self):
        lines=[READY,event("scheduler","state","pending=2"),
               event("progress","request_status","phase=decode native_rate_state=fresh native_interval_ms=1000.000 native_interval_tokens_s=20.00")]
        log,raw,terminal,user=io.StringIO(),io.StringIO(),io.StringIO(),io.StringIO()
        process_lines(lines,log,terminal,"user",raw,user)
        self.assertEqual(log.getvalue(),"".join(lines));self.assertEqual(raw.getvalue(),log.getvalue())
        self.assertIn("category=scheduler",log.getvalue())
        self.assertNotIn("scheduler",user.getvalue())
        self.assertIn("Native decode",user.getvalue())
        self.assertEqual(user.getvalue(),strip_ansi(terminal.getvalue()))
    def test_user_line_copies_source_timestamp_and_preserves_logger_timestamp_in_raw(self):
        line=("2026-10-20 12:47:03,931 1062118 request_progress.py[line:116] INFO: "
              +event("progress","request_status","phase=decode native_rate_state=fresh native_interval_ms=1000.000 native_interval_tokens_s=20.00"))
        log,raw,tty=render([READY,line],"user")
        self.assertEqual(log,READY+line);self.assertEqual(raw,log)
        body=strip_ansi(tty).split(READY,1)[1]
        self.assertTrue(body.startswith("[req-7] [2026-10-17T00:00:00Z] "))
        self.assertIn("2026-10-20 12:47:03,931",raw)
    def test_native_line_uses_raw_timestamp_when_logger_prefix_is_absent(self):
        body=strip_ansi(render([READY,event("progress","request_status","phase=decode native_rate_state=fresh native_interval_ms=1000.000 native_interval_tokens_s=20.00")],"user")[2]).split(READY,1)[1]
        self.assertTrue(body.startswith("[req-7] [2026-10-17T00:00:00Z] "))
    def test_backend_log_records_debug_facts_while_terminal_shows_user_view(self):
        lines=[READY,event("scheduler","state","pending=2"),
               event("progress","request_status","phase=decode native_rate_state=fresh native_interval_tokens_s=20.00 proc_rss_mib=11 host_mem_available_mib=50565 resource_sample_state=fresh gpu_sample_state=scope_unknown")]
        log,raw,tty=render(lines,"user")
        self.assertEqual(log,"".join(lines));self.assertEqual(raw,log)
        self.assertIn("category=scheduler",log)
        plain=strip_ansi(tty);self.assertNotIn("category=scheduler",plain)
        self.assertIn("Native decode",plain);self.assertNotIn("Process RAM",plain)
    def test_progress_event_carries_raw_resource_snapshot(self):
        from unittest.mock import patch
        snapshot={"proc_rss_mib":"123","host_mem_available_mib":"456","cpu_load1":"7.89",
                  "resource_sample_state":"fresh","gpu_sample_state":"scope_unknown"}
        events=[];r=RequestProgressReporter("req-r",emit=events.append,mode="debug")
        with patch("request_progress.resource_snapshot",return_value=snapshot):
            r.maybe(force=True)
        raw=next(e for e in events if "event=request_status" in e)
        for key,value in snapshot.items():self.assertIn(key+"="+value,raw)
    def test_resource_values_are_projected_without_unit_invention(self):
        snapshot={"proc_rss_mib":"123","host_mem_available_mib":"456","cpu_load1":"7.89",
                  "gpu_util_max_pct":"77","gpu_mem_used_mib":"60001","gpu_device_indices":"0,1,2,3,4",
                  "resource_sample_state":"fresh","gpu_sample_state":"ok"}
        fields=" ".join(f"{key}={value}" for key,value in snapshot.items())
        line=event("progress","request_status","phase=decode native_rate_state=pending "+fields)
        body=strip_ansi(render([READY,line],"user")[2]).split(READY,1)[1]
        for diagnostic in ("Process RAM","Host available RAM","GPU max","GPU memory total","GPUs ","resource sample"):self.assertNotIn(diagnostic,body)
        self.assertNotIn("load1",body);self.assertNotIn("7.89",body)
    def test_idle_request_still_prints_one_progress_per_interval(self):
        events=[];r=RequestProgressReporter("req-idle",emit=events.append,mode="debug",interval_s=0.05)
        r.received(path="/");time.sleep(0.24);r.stop()
        diagnostics=[e for e in events if "event=progress " in e]
        statuses=[e for e in events if "event=request_status " in e]
        self.assertGreaterEqual(len(statuses),2)
        self.assertTrue(all("callback_events=0" in e for e in diagnostics))
        self.assertTrue(all("python_callback_events_s=0.00" in e for e in diagnostics))
        self.assertEqual(len(statuses),len(diagnostics))
    def test_off_mode_starts_no_pump_and_prints_progress(self):
        events=[];r=RequestProgressReporter("req-off",emit=events.append,mode="off",interval_s=0.05)
        r.received(path="/");time.sleep(0.16);r.stop()
        self.assertEqual([e for e in events if "category=progress" in e],[])
        self.assertIsNone(r._pump)
    def test_progress_debug_keeps_every_raw_field_in_order(self):
        line=event("progress","progress","phase=prefill elapsed_s=1.000 prefill_interval_tokens_s=100.00 committed_interval_tokens_s=0.00")
        log,raw,tty=render([READY,line],"debug")
        self.assertEqual(log,raw);self.assertEqual(strip_ansi(tty),log)
        self.assertIn("prefill_interval_tokens_s=100.00 committed_interval_tokens_s=0.00",tty)

    def test_every_event_in_one_second_is_printed_without_truncation(self):
        lines=[READY,
               event("progress","request_status","phase=prefill native_rate_state=fresh native_interval_tokens_s=100.00"),
               event("progress","request_status","phase=decode native_rate_state=fresh native_interval_tokens_s=20.00"),
               event("output","native_first_token","ttft_ms=12"),
               event("progress","request_status","phase=decode native_rate_state=fresh native_interval_tokens_s=21.00")]
        log,raw,tty=render(lines,"user")
        self.assertEqual(log,"".join(lines));self.assertEqual(raw,log)
        rows=[row for row in strip_ansi(tty).split(READY,1)[1].strip().splitlines() if "Cache capacity |" not in row]
        self.assertEqual(len(rows),4)
        for row,expected in zip(rows,("Prefill | 100.00 tokens/s","Native decode | 20.00 tokens/s","First token: 12 ms","Native decode | 21.00 tokens/s")):self.assertIn(expected,row)

class NativeRawT0Test(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("FASTLLM_TEST_OBSERVATION_PROBE"),"production event probe not provided")
    def test_actual_native_emitter_retains_user_debug_diagnostics(self):
        streams={}
        for mode in ("off","user","debug"):
            streams[mode]=subprocess.run([os.environ["FASTLLM_TEST_OBSERVATION_PROBE"]],env={**os.environ,"FASTLLM_OBSERVABILITY_MODE":mode},text=True,capture_output=True,check=True).stdout
        normalize=lambda x:re.sub(r"timestamp_(?:monotonic_ns|utc)=\S+","",x)
        self.assertEqual(normalize(streams["user"]),normalize(streams["debug"]));self.assertNotIn("[fastllm-event]",streams["off"])
        self.assertIn("event=backend_detail",streams["user"])
        log,raw,tty=render([READY]+streams["user"].splitlines(keepends=True),"user");self.assertEqual(log,raw);self.assertNotIn("backend_detail",tty);self.assertIn("L1 cache: hit 128 tokens",tty)
if __name__=="__main__":unittest.main()
