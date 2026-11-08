"""CPU-only contract checks for raw status, capacities and service GPU scope."""
import io
import os
import pathlib
import subprocess
import sys
import time
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools/fastllm_pytools"))
from request_progress import RequestProgressReporter, _ResourceSampler
from terminal_colorizer import UserTerminalView, process_lines, strip_ansi

READY = "INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)\n"

def event(category, name, fields, source="python"):
    return ("[fastllm-event] seq=7 timestamp_monotonic_ns=1000000000 "
            "timestamp_utc=2026-10-20T07:00:00Z source="+source+
            " category="+category+" request_id=req-capacity native_handle=7 event="+name+
            " "+fields+"\n")

class RequestStatusContractTest(unittest.TestCase):
    def test_producer_snapshot_is_copied_into_raw_then_user_without_recalculation(self):
        native={"phase":"prefill","prefill_completed_tokens":"2048","prefill_total_tokens":"8192",
                "native_interval_tokens_s":"1024.123","native_interval_ms":"1999.760",
                "native_rate_state":"fresh","l1_capacity_state":"ok",
                "l1_used_tokens":"256","l1_available_tokens":"768","l1_total_tokens":"1024",
                "l2_capacity_state":"ok","l2_available_gib":"14.000","l2_limit_gib":"16.000",
                "ssd_capacity_state":"ready","ssd_available_gib":"118.000","ssd_limit_gib":"128.000"}
        resources={"resource_sample_state":"fresh","proc_rss_mib":"4096",
                   "gpu_sample_state":"ok","gpu_device_indices":"0,2",
                   "gpu_util_max_pct":"77","gpu_mem_used_mib":"31000"}
        raw=[]
        reporter=RequestProgressReporter("req-capacity",native_handle=7,mode="debug",
                                         emit=raw.append,status_provider=lambda:native)
        with patch("request_progress.resource_snapshot",return_value=resources):
            reporter.maybe(force=True)
        row=next(line for line in raw if "event=request_status " in line)
        for key,value in {**native,**resources}.items():self.assertIn(key+"="+value,row)
        visible=UserTerminalView().line(row)
        for value in ("2048/8192","1024.123","256","768","1024","14.000","16.000","118.000","128.000"):
            self.assertIn(value,visible)
        self.assertNotIn("Callback",visible);self.assertNotIn("events/s",visible)
        for diagnostic in ("GPUs ","GPU max","GPU memory","Process RAM","window ","sample age","31000"):
            self.assertNotIn(diagnostic,visible)

    def test_unavailable_and_stale_samples_never_become_zero_or_current_speed(self):
        v=UserTerminalView()
        pending=v.line(event("progress","request_status","phase=prefill native_rate_state=pending l1_capacity_state=busy l2_capacity_state=disabled ssd_capacity_state=unavailable resource_sample_state=pending"))
        self.assertIn("speed pending",pending);self.assertIn("L1 capacity busy",pending)
        self.assertNotIn("0 tokens",pending);self.assertNotIn("0.000",pending)
        stale=v.line(event("progress","request_status","phase=decode native_rate_state=stale native_interval_tokens_s=999.888 native_sample_age_ms=4000.000 resource_sample_state=stale proc_rss_mib=123456"))
        self.assertIn("waiting for progress",stale);self.assertNotIn("4000.000",stale)
        self.assertNotIn("999.888",stale);self.assertNotIn("123456",stale)

    def test_capacity_labels_are_separate_from_prefix_matches_and_honor_quota(self):
        v=UserTerminalView()
        candidate=v.line(event("cache","cache_candidates","l1_resident_tokens=0 l2_host_tokens=0 l2_query_status=ok","native"))
        self.assertIn("L1 prefix candidate: 0 tokens",candidate)
        self.assertNotIn("L1 capacity",candidate);self.assertNotIn("L2 capacity",candidate)
        for limit,available in (("128.000","118.000"),("256.000","246.000"),("1014.000","1004.000")):
            line=v.line(event("progress","request_status","phase=preparing l1_capacity_state=ok l1_used_tokens=128 l1_available_tokens=896 l1_total_tokens=1024 ssd_capacity_state=ready ssd_available_gib="+available+" ssd_limit_gib="+limit))
            self.assertIn("L1 used 128 / free 896 / total 1024 tokens",line)
            self.assertIn("SSD quota free "+available+" / "+limit+" GiB",line)

    def test_only_one_decode_measurement_per_heartbeat_and_mean_only_at_end(self):
        rows=[READY,
              event("decode","native_decode","native_tokens_s=50.123","native"),
              event("progress","progress","measurement=python_callback_interval callback_events=22 python_callback_tokens_s=999.123"),
              event("progress","request_status","phase=decode native_rate_state=fresh native_interval_tokens_s=75.590 native_interval_ms=1019.000"),
              event("output","native_generation_summary","native_rate_available=1 native_average_tokens_s=74.443","native"),
              event("tool","tool_call","count=1"),
              event("output","request_finished","status=ok output_tokens=1720 elapsed_ms=192572.9")]
        log,raw,tty=io.StringIO(),io.StringIO(),io.StringIO()
        process_lines(rows,log,tty,"user",raw)
        self.assertEqual(raw.getvalue(),"".join(rows));self.assertEqual(log.getvalue(),raw.getvalue())
        visible=strip_ansi(tty.getvalue())
        self.assertEqual(visible.count("75.590 tokens/s"),1)
        self.assertIn("Native decode average: 74.443 tokens/s",visible)
        self.assertNotIn("50.123",visible);self.assertNotIn("999.123",visible)
        self.assertLess(visible.index("Tool call"),visible.index("Finished: ok"))

    def test_off_never_queries_native_or_starts_resource_sampling(self):
        with patch("request_progress.resource_snapshot") as sample:
            provider=lambda: self.fail("off must not query native")
            r=RequestProgressReporter("req-off",mode="off",status_provider=provider,emit=lambda _:self.fail("off emitted"))
            r.received(path="/");r.callback({"output_tokens":3});r.finish(output_tokens=3)
            self.assertIsNone(r._pump);sample.assert_not_called()

    def test_capacity_is_separate_and_not_repeated_on_each_heartbeat(self):
        view=UserTerminalView()
        initial="phase=decode native_rate_state=fresh native_interval_tokens_s=68.570 native_output_tokens=475 l1_capacity_state=ok l1_used_tokens=87936 l1_available_tokens=730752 l1_total_tokens=818688 l2_capacity_state=ok l2_available_gib=16.000 l2_limit_gib=16.000 ssd_capacity_state=ready ssd_available_gib=244.576 ssd_limit_gib=256.000"
        first=view.line(event("progress","request_status",initial))
        second=view.line(event("progress","request_status",initial))
        self.assertIn("Native decode | 68.570 tokens/s | 475 output tokens\n",first)
        self.assertIn("Cache capacity | L1 used 87936 / free 730752 / total 818688 tokens",first)
        self.assertEqual(len(second.strip().splitlines()),1)
        self.assertNotIn("Cache capacity",second)
        changed=view.line(event("progress","request_status",initial.replace("phase=decode","phase=finishing")))
        self.assertIn("Cache capacity",changed)
        busy=view.line(event("progress","request_status",initial.replace("l1_capacity_state=ok","l1_capacity_state=busy")))
        self.assertIn("L1 capacity busy",busy)

    def test_all_resource_and_timing_diagnostics_stay_raw_debug_only(self):
        status=event("progress","request_status","phase=decode native_rate_state=fresh native_interval_tokens_s=68.570 native_output_tokens=475 native_interval_ms=1006.273 native_sample_age_ms=44.000 proc_rss_mib=10134 host_mem_available_mib=49285 gpu_device_indices=0,1,2,3,4 gpu_util_max_pct=74 gpu_mem_used_mib=63875 resource_sample_state=fresh resource_sample_age_ms=831")
        for mode in ("user","debug"):
            log,raw,tty=io.StringIO(),io.StringIO(),io.StringIO()
            process_lines([READY,status],log,tty,mode,raw)
            self.assertEqual(log.getvalue(),READY+status);self.assertEqual(raw.getvalue(),READY+status)
            plain=strip_ansi(tty.getvalue()).split(READY,1)[1]
            for text in ("10134","49285","63875","1006.273","44.000","831","0,1,2,3,4"):
                if mode=="user":self.assertNotIn(text,plain)
                else:self.assertIn(text,plain)
            self.assertIn("68.570",plain)

class ResourceScopeTest(unittest.TestCase):
    ROWS=[["5","GPU-c","00000000:03:00.0","99","9000"],
          ["2","GPU-a","00000000:01:00.0","20","1000"],
          ["8","GPU-b","00000000:02:00.0","30","2000"]]

    def test_pci_ordered_cuda_ordinals_are_not_nvml_indices(self):
        with patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":"0,1","CUDA_DEVICE_ORDER":"PCI_BUS_ID"}):
            selected=_ResourceSampler._service_gpus(self.ROWS)
        self.assertEqual([row[0] for row in selected],["2","8"])

    def test_uuid_scope_and_unknown_order_fail_closed(self):
        with patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":"GPU-b","CUDA_DEVICE_ORDER":"FASTEST_FIRST"}):
            self.assertEqual(_ResourceSampler._service_gpus(self.ROWS),[self.ROWS[2]])
        with patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":"0","CUDA_DEVICE_ORDER":"FASTEST_FIRST"}):
            self.assertIsNone(_ResourceSampler._service_gpus(self.ROWS))
        with patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":"GPU-missing","CUDA_DEVICE_ORDER":"PCI_BUS_ID"}):
            self.assertIsNone(_ResourceSampler._service_gpus(self.ROWS))

    def test_sample_filters_other_gpu_and_keeps_max_and_total_distinct(self):
        sampler=_ResourceSampler()
        response=types.SimpleNamespace(returncode=0,stdout="\n".join(",".join(row) for row in self.ROWS))
        fake=types.SimpleNamespace(run=lambda *a,**k:response,SubprocessError=subprocess.SubprocessError)
        with patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":"0,1","CUDA_DEVICE_ORDER":"PCI_BUS_ID"}),patch("request_progress.subprocess",fake):
            sampler._sample()
        snap=sampler.snapshot()
        self.assertEqual(snap["gpu_device_indices"],"2,8")
        self.assertEqual(snap["gpu_util_max_pct"],"30")
        self.assertEqual(snap["gpu_mem_used_mib"],"3000")
        self.assertEqual(snap["gpu_sample_state"],"ok")
        sampler._sampled_at=time.monotonic()-4
        self.assertEqual(sampler.snapshot()["resource_sample_state"],"stale")

    def test_failed_gpu_query_does_not_reuse_previous_values(self):
        sampler=_ResourceSampler()
        sampler._values={"gpu_util_max_pct":"99","gpu_mem_used_mib":"123"}
        fake=types.SimpleNamespace(run=lambda *a,**k:types.SimpleNamespace(returncode=1,stdout=""),SubprocessError=subprocess.SubprocessError)
        with patch("request_progress.subprocess",fake):
            sampler._sample()
        snap=sampler.snapshot()
        self.assertEqual(snap["gpu_sample_state"],"unavailable")
        self.assertNotIn("gpu_util_max_pct",snap)

if __name__ == "__main__":
    unittest.main()
