"""Exercise real independent diagnostic header/lifecycle/C ABI with CPU mutexes."""
import datetime
import json
import os
import pathlib
import subprocess
import unittest

ROOT = pathlib.Path(os.environ.get("FASTLLM_TEST_ROOT", os.path.join(os.path.expanduser("~"), ".cache", "fastllm-test")))
(ROOT / 'logs').mkdir(parents=True, exist_ok=True)
REPO = pathlib.Path(__file__).resolve().parents[2]


def record(phase, **fields):
    with (ROOT / "logs/actions.jsonl").open("a") as output:
        output.write(json.dumps({
            "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "task": "debug-stall-native-t0", "phase": phase, **fields
        }) + "\n")
        output.flush()
        os.fsync(output.fileno())


class NativeRequestDiagnosticsT0(unittest.TestCase):
    def test_independent_read_real_wait_identity_cleanup_and_bounded_stacks(self):
        run = pathlib.Path(os.environ["FASTLLM_DIAGNOSTICS_ACCEPTANCE_RUN"])
        build = pathlib.Path(os.environ["FASTLLM_DIAGNOSTICS_ACCEPTANCE_BUILD"])
        bridge = (REPO / "tools/src/pytools.cpp").read_text()
        bridge = bridge[bridge.index("    DLL_EXPORT bool get_request_diagnostics_llm_model("):
                        bridge.index("    DLL_EXPORT bool get_request_status_llm_model(")]
        self.assertNotIn("models.GetModel", bridge)
        self.assertNotIn("ReadRequestStatus", bridge)
        body = r'''
#include <atomic>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include "utils/prefix_cache_logging.h"
#define DLL_EXPORT
''' + bridge + r'''
using namespace fastllm;
static RequestDiagnosticFields Read(int handle,const std::string &owner) {
    RequestDiagnosticFields fields;assert(ReadRequestDiagnostics(handle,owner,fields));return fields;
}
static std::string Rows(int handle,const std::string &owner) {
    return Read(handle,owner).at("native_diag_operations_json");
}
int main(int argc,char **) {
    if (argc>1) {
        assert(!RequestDiagnosticsEnabled());
        CreateRequestDiagnostics(900);RegisterRequestDiagnostics(900,"off");
        {RequestDiagnosticOperation op(900,"disabled","must_not_allocate_registry");}
        assert(RequestDiagnosticsRegistry().empty());
        assert(Read(900,"off").at("native_diag_state")=="disabled");
        std::puts("DIAGNOSTICS_OFF_OK");return 0;
    }
    BeginFastllmNativeRequest(7);RegisterFastllmPublicRequestId(7,"request-A");
    int model;
    BindRequestDiagnosticModel(7,&model);
    auto owners=RequestDiagnosticHandlesForModel(&model);assert(owners.size()==1 && owners[0]==7);
    const auto original=CaptureRequestDiagnosticToken(7);
    std::mutex realMutex;realMutex.lock();
    std::atomic<bool> entering{false},acquired{false};
    std::thread worker([&] {
        RequestDiagnosticOperation op(original,"real_mutex_lock_wait",
            RequestDiagnosticResource("cpu_test_mutex",&realMutex).c_str());
        entering=true;
        realMutex.lock();op.Finish();acquired=true;
        RequestDiagnosticOperation held(original,"real_mutex_held",
            RequestDiagnosticResource("cpu_test_mutex",&realMutex).c_str());
        realMutex.unlock();
    });
    auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
    while(!entering && std::chrono::steady_clock::now()<deadline)std::this_thread::yield();
    assert(entering && !acquired);
    const auto sampleStart=RequestDiagnosticNowNs();
    // Hold both original observability locks while invoking the actual ABI.
    // The independent getter must still complete and see the real wait.
    RequestStatusMutex().lock();FastllmPublicRequestIdsMutex().lock();
    char buffer[65536];
    assert(get_request_diagnostics_llm_model(12345,7,"request-A",buffer,sizeof(buffer)));
    FastllmPublicRequestIdsMutex().unlock();RequestStatusMutex().unlock();
    assert(RequestDiagnosticNowNs()-sampleStart<500000000ULL);
    assert(std::string(buffer).find("real_mutex_lock_wait")!=std::string::npos);
    assert(!get_request_diagnostics_llm_model(0,7,"request-A",buffer,4));
    assert(!get_request_diagnostics_llm_model(0,7,nullptr,buffer,sizeof(buffer)));
    auto running=Read(7,"request-A");
    assert(running.at("native_diag_progress_revision")=="0");
    std::printf("RUNNING_ROWS=%s\n",running.at("native_diag_operations_json").c_str());
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    std::printf("LATER_ROWS=%s\n",Rows(7,"request-A").c_str());
    realMutex.unlock();worker.join();assert(acquired);
    std::printf("RETURNED_ROWS=%s\n",Rows(7,"request-A").c_str());
    // Returning polling or lock scopes is not native model progress.
    assert(Read(7,"request-A").at("native_diag_progress_revision")=="0");
    RecordRequestDiagnosticProgress(7,RequestDiagnosticNowNs());
    assert(Read(7,"request-A").at("native_diag_progress_revision")=="1");
    assert(Read(7,"wrong-owner").at("native_diag_state")=="owner_mismatch");
    std::atomic<bool> registryLocked{false},releaseRegistry{false};
    std::thread lockOwner([&] {
        std::lock_guard<std::mutex> lock(RequestDiagnosticsMutex());
        registryLocked=true;while(!releaseRegistry)std::this_thread::yield();
    });
    deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
    while(!registryLocked && std::chrono::steady_clock::now()<deadline)std::this_thread::yield();
    assert(registryLocked);
    const auto busyStart=RequestDiagnosticNowNs();
    assert(get_request_diagnostics_llm_model(0,7,"request-A",buffer,sizeof(buffer)));
    assert(std::string(buffer).find("native_diag_state=busy")!=std::string::npos);
    assert(RequestDiagnosticNowNs()-busyStart<500000000ULL);
    releaseRegistry=true;lockOwner.join();
    try {RequestDiagnosticOperation op(7,"throws","cpu_test");throw std::runtime_error("expected");}
    catch (const std::runtime_error &) {}
    assert(Rows(7,"request-A").find("\"state\":\"unwound\"")!=std::string::npos);
    auto oldScope=std::make_unique<RequestDiagnosticOperation>(7,"old_generation_work","cpu_test");
    BeginFastllmNativeRequest(7);RegisterFastllmPublicRequestId(7,"request-B");
    UnregisterFastllmPublicRequestId(7,"request-A");oldScope.reset();
    {RequestDiagnosticOperation late(original,"late_background_old_generation","cpu_test");}
    auto current=Read(7,"request-B");
    assert(current.at("native_diag_active_count")=="0");
    assert(current.at("native_diag_completed_count")=="0");
    assert(current.at("native_diag_generation")!=std::to_string(original.generation));
    EndFastllmNativeRequest(7);UnregisterFastllmPublicRequestId(7,"request-B");
    assert(Read(7,"request-B").at("native_diag_state")=="missing");
    BeginFastllmNativeRequest(8);EndFastllmNativeRequest(8);
    RegisterFastllmPublicRequestId(8,"short-closed-before-bind");
    assert(Read(8,"short-closed-before-bind").at("native_diag_state")=="closed");
    UnregisterFastllmPublicRequestId(8,"short-closed-before-bind");
    assert(Read(8,"short-closed-before-bind").at("native_diag_state")=="missing");
    BeginFastllmNativeRequest(9);RegisterFastllmPublicRequestId(9,"cancelled");
    auto cancelling=std::make_unique<RequestDiagnosticOperation>(9,"cleanup_in_progress","cpu_test");
    UnregisterFastllmPublicRequestId(9,"cancelled");EndFastllmNativeRequest(9);
    assert(Read(9,"cancelled").at("native_diag_state")=="closed");
    assert(Read(9,"cancelled").at("native_diag_active_count")=="1");
    cancelling.reset();assert(Read(9,"cancelled").at("native_diag_state")=="missing");
    BeginFastllmNativeRequest(10);RegisterFastllmPublicRequestId(10,"bounded");
    std::vector<std::unique_ptr<RequestDiagnosticOperation>> active;
    for(int i=0;i<40;++i)active.emplace_back(new RequestDiagnosticOperation(
        10,("operation_"+std::to_string(i)).c_str(),"bounded_test"));
    auto bounded=Read(10,"bounded");
    assert(bounded.at("native_diag_active_count")=="32");
    assert(bounded.at("native_diag_dropped_active_count")=="8");
    assert(bounded.at("native_diag_operations_truncated")=="1");
    assert(bounded.at("native_diag_operations_json").size()<56001);
    assert(get_request_diagnostics_llm_model(0,10,"bounded",buffer,sizeof(buffer)));
    std::printf("BOUNDED_ROWS=%s\n",bounded.at("native_diag_operations_json").c_str());
    active.clear();EndFastllmNativeRequest(10);UnregisterFastllmPublicRequestId(10,"bounded");
    std::puts("REQUEST_DIAGNOSTICS_NATIVE_T0_OK");
}
'''
        record("before", command="prepare actual diagnostic header/lifecycle/ABI CPU probe",
               build_dir=str(build), log_path=str(run), gpu_mapping="none")
        build.mkdir(parents=True, exist_ok=False)
        run.mkdir(parents=True, exist_ok=True)
        (build / "probe.cpp").write_text(body)
        (build / "Makefile").write_text(
            "CXX := /usr/bin/c++\n"
            "CXXFLAGS := -std=c++17 -O2 -g1 -Wall -Wextra -Werror -pthread -I"
            + str(REPO / "include") + "\n"
            "all: diagnostics_probe\n"
            "diagnostics_probe: probe.cpp\n"
            "\t$(CXX) $(CXXFLAGS) $< -ldl -o $@\n")
        commands = [
            (["make", "-j36"], os.environ.copy(), "diagnostics-t0-compile.log"),
            ([str(build / "diagnostics_probe")],
             {**os.environ, "FASTLLM_OBSERVABILITY_MODE": "debug"}, "diagnostics-t0-run.log"),
            ([str(build / "diagnostics_probe"), "off"],
             {**os.environ, "FASTLLM_OBSERVABILITY_MODE": "off"}, "diagnostics-t0-off.log"),
            ([str(build / "diagnostics_probe"), "user"],
             {**os.environ, "FASTLLM_OBSERVABILITY_MODE": "user"}, "diagnostics-t0-user.log"),
        ]
        for command, env, name in commands:
            path = run / name
            record("before", command=command, cwd=str(build), build_dir=str(build),
                   log_path=str(path), gpu_mapping="none", expected="bounded CPU-only evidence; exit0")
            with path.open("xb") as output:
                result = subprocess.run(command, cwd=build, env=env, stdout=output,
                                        stderr=subprocess.STDOUT, timeout=20)
            record("after", command=command, exit_code=result.returncode,
                   log_path=str(path), resource_cleanup="CPU helper ended; no model or service")
            text = path.read_text()
            self.assertEqual(result.returncode, 0, text)
        text = (run / "diagnostics-t0-run.log").read_text()
        rows = {}
        for line in text.splitlines():
            for prefix in ("RUNNING_ROWS", "LATER_ROWS", "RETURNED_ROWS", "BOUNDED_ROWS"):
                if line.startswith(prefix + "="):
                    rows[prefix] = json.loads(line.split("=", 1)[1])
        first = next(x for x in rows["RUNNING_ROWS"] if x["operation"] == "real_mutex_lock_wait")
        later = next(x for x in rows["LATER_ROWS"] if x["operation"] == "real_mutex_lock_wait")
        returned = next(x for x in rows["RETURNED_ROWS"] if x["operation"] == "real_mutex_lock_wait")
        self.assertEqual(first["state"], "running")
        self.assertEqual(first["end_monotonic_ns"], 0)
        self.assertGreater(later["elapsed_ns"], first["elapsed_ns"])
        self.assertEqual(returned["state"], "returned")
        self.assertGreater(returned["end_monotonic_ns"], returned["start_monotonic_ns"])
        self.assertGreater(returned["end_unix_ns"], returned["start_unix_ns"])
        self.assertGreater(first["thread_id"], 0)
        self.assertEqual(first["stack_scope"], "first_operation_entry_same_label_not_current")
        self.assertLessEqual(len(first["entry_stack_frames"]), 8)
        self.assertGreater(len(first["entry_stack_frames"]), 0)
        self.assertGreater(first["entry_stack_capture_elapsed_ns"], 0)
        for frame in first["entry_stack_frames"]:
            self.assertTrue(frame["address"].startswith("0x"))
            if frame["module"]:
                self.assertEqual(int(frame["relative_address"], 16),
                                 int(frame["address"], 16) - int(frame["module_base"], 16))
        self.assertLessEqual(len(rows["BOUNDED_ROWS"]), 16)
        self.assertIn("REQUEST_DIAGNOSTICS_NATIVE_T0_OK", text)


if __name__ == "__main__":
    unittest.main()
