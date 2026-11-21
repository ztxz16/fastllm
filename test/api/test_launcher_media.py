"""Service isolation and generation workflows, using real local child servers."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
from pathlib import Path
import shlex
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from unittest.mock import patch
from types import SimpleNamespace

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
from fastllm_pytools.launcher import LauncherError, LauncherRuntime, create_launcher_app
from fastllm_pytools.launcher_services import LauncherServicePool
from fastllm_pytools.media_agent import GenerationBridge
from fastllm_pytools.service_backends import visible_gpu_environment
from fastllm_pytools.tui import DeployConfig, build_fastllm_env, build_fastllm_argv, config_from_dict, validate_config


WORKER = r'''
import argparse, base64, json, os, time
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
p=argparse.ArgumentParser()
p.add_argument('--port',type=int); p.add_argument('--host'); p.add_argument('--api-key',default='')
p.add_argument('--api_key',default=''); p.add_argument('--kind',default='video')
p.add_argument('--device')
args,_=p.parse_known_args(); jobs={}; started=time.monotonic()
if args.device is not None: args.kind='image'
class H(BaseHTTPRequestHandler):
 def log_message(self,*args): pass
 def answer(self,value,status=200,mime='application/json'):
  data=value if isinstance(value,bytes) else json.dumps(value).encode()
  self.send_response(status); self.send_header('Content-Type',mime); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
 def auth(self):
  if self.headers.get('Authorization')!='Bearer '+(args.api_key or 'test-service-key'):
   self.answer({'error':{'message':'unauthorized'}},401); return False
  return True
 def do_GET(self):
  if args.kind in ('duo','hy_nomas'):
   if self.path in ('/health', '/resolutions'): return self.answer({'ready':True})
   if self.path.endswith('/events'): return self.answer(b'data: {}\n\n',mime='text/event-stream')
   if args.kind=='hy_nomas' and self.path.endswith(('/content','?format=mp4')): return self.answer(b'fake-mp4',mime='video/mp4')
   if self.path.endswith('/avc'): return self.answer(b'frame',mime='application/octet-stream')
   return self.answer({'path':self.path})
  if self.path=='/health': return self.answer({'ready':time.monotonic()-started>float(os.environ.get('READY_DELAY','.4')), 'reference_model_available':True, 'dit_precision':'fp8', 'reference_dit_precision':'gguf', 'dit_parallel_mode':'sp', 'dit_sequence_parallel_size':2, 'dit_tensor_parallel_size':1},200 if time.monotonic()-started>float(os.environ.get('READY_DELAY','.4')) else 503)
  if not self.auth(): return
  if self.path=='/v1/models': return self.answer({'data':[{'id':'stub-'+args.kind}], 'visible':os.environ.get('CUDA_VISIBLE_DEVICES')})
  if self.path in ('/v1/videos','/v1/images/jobs'): return self.answer({'data':list(jobs.values())})
  if self.path.endswith('/content'):
   if args.kind=='image': return self.answer(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a5WQAAAAASUVORK5CYII='),mime='image/png')
   return self.answer(b'fake-mp4',mime='video/mp4')
  return self.answer(jobs[self.path.split('/')[-1]])
 def do_PATCH(self):
  self.answer({'data':self.rfile.read(int(self.headers.get('Content-Length','0'))).decode()})
 def do_DELETE(self): self.answer({'method':self.command})
 def do_POST(self):
  if args.kind in ('duo','hy_nomas'):
   return self.answer({'path':self.path,'size':len(self.rfile.read(int(self.headers.get('Content-Length','0'))))})
  if not self.auth(): return
  if self.path.endswith('/cancel'):
   job=jobs[self.path.split('/')[-2]]; job['status']='cancelled'; return self.answer(job)
  raw=self.rfile.read(int(self.headers.get('Content-Length','0')))
  if self.headers.get('Content-Type','').startswith('multipart/form-data'):
   message=BytesParser(policy=default).parsebytes(('Content-Type: '+self.headers['Content-Type']+'\r\n\r\n').encode()+raw)
   data={'references':[]}
   for part in message.iter_parts():
    name=part.get_param('name',header='content-disposition'); content=part.get_payload(decode=True)
    if name in ('image[]','reference'): data['references'].append({'name':part.get_filename(),'type':part.get_content_type(),'data':base64.b64encode(content).decode()})
    else: data[name]=content.decode()
  else: data=json.loads(raw)
  job=dict(data,id=args.kind+'_1_'+str(len(jobs)),status='queued' if data['prompt']=='hold' else 'completed',created_at=time.time(),progress=100,stage='done',current=6,total=6,error=None)
  jobs[job['id']]=job; return self.answer(job)
ThreadingHTTPServer((args.host,args.port),H).serve_forever()
'''


def port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return str(listener.getsockname()[1])


class ServiceTest(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name).resolve()
        self.worker = self.directory / "server.py"
        self.worker.write_text("#!" + sys.executable + "\n" + WORKER)
        self.worker.chmod(0o755)
        self.models = {family: self.directory / family for family in
                       ("llm", "minimax_h3", "minimax_h3_image", "hy_nomas", "duo", "qwen_image", "lingbot_world")}
        for root in self.models.values():
            root.mkdir()
        for family, pipeline in (("minimax_h3", "MiniMaxH3ModularPipeline"), ("minimax_h3_image", "MiniMaxH3ImagePipeline"),
                                 ("qwen_image", "QwenImage21Pipeline")):
            (self.models[family] / "model_index.json").write_text(json.dumps({"_class_name": pipeline}))
        for part in ("DiT/distilled", "VAE/PixelUMM_FT"):
            (self.models["hy_nomas"] / part).mkdir(parents=True)
        for name in ("realization", "intent", "intent_adapter", "codec", "speech"):
            (self.models["duo"] / name).mkdir()
        (self.models["duo"] / "realization/diffusion_pytorch_model.safetensors.index.json").write_text("{}")
        for name in ("transformers/model.safetensors.index.json", "google/umt5-xxl/tokenizer.json",
                     "native/t5.safetensors", "native/vae.safetensors"):
            asset = self.models["lingbot_world"] / name
            asset.parent.mkdir(exist_ok=True, parents=True)
            asset.touch()
        bin_dir = self.directory / "bin"
        bin_dir.mkdir()
        for name in ("minimax_h3_server", "minimax_h3_image_server", "duo_live", "libfastllm_nomas.so",
                     "qwen_image_server", "libfastllm_lingbot.so"):
            (bin_dir / name).symlink_to(self.worker)
        env = patch.dict(os.environ, {"PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
                                      "LD_LIBRARY_PATH": str(bin_dir)})
        env.start()
        self.addCleanup(env.stop)
        self.commands = []
        def popen(argv, **kwargs):
            self.commands.append((argv, kwargs["env"].copy()))
            from fastllm_pytools import cli
            from fastllm_pytools.model_service import identify_model
            self.assertEqual(argv[1:4], ["-m", "fastllm_pytools.cli", "server"])
            if identify_model(argv[4]).backend == "llm":
                argv = [str(self.worker), "--kind", "llm"] + argv[4:]
            else:
                with patch.dict(os.environ, kwargs["env"]), patch("os.execvpe") as execute:
                    cli.main(argv[3:])
                execute.assert_called_once()
                _, native, kwargs["env"] = execute.call_args.args
                argv = [str(self.worker), *native[1:]]
            return subprocess.Popen(argv, **kwargs)
        runtime = LauncherRuntime(str(self.directory / "profiles.json"), popen_factory=popen,
            webui_history_dir=str(self.directory / "history"),
            agent_workspace_root=str(self.directory), plugins_dir=str(self.directory / "plugins"))
        self.pool = LauncherServicePool(runtime)
        self.addCleanup(self.pool.close)
        self.client = TestClient(create_launcher_app(self.pool, "control-key"))
        self.headers = {"X-FTLLM-Launcher-Token": "control-key"}

    def profile(self, backend="llm", **changes):
        config = dict(name=backend, model=str(self.models[backend]), config_mode="auto",
            host="127.0.0.1", port=port(), api_key="test-service-key", visible_gpus="0",
            output_dir=str(self.directory / "output"))
        config.update(changes)
        return config

    def start(self, backend="llm", **changes):
        response = self.client.post("/api/runtime/start", headers=self.headers, json=self.profile(backend, **changes))
        self.assertEqual(response.status_code, 200, response.text)
        session = response.json()["sessionId"]
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            if self.pool.service(session).state()["ready"]:
                return session
            time.sleep(.05)
        self.fail(str(self.pool.state()) + str(self.pool.logs()))

    @unittest.skipUnless(hasattr(socket, "SO_REUSEPORT"), "requires SO_REUSEPORT")
    def test_interactive_port_reuse_after_stream_shutdown(self):
        from fastllm_pytools.launcher import _port_is_available
        for option in (socket.SO_REUSEADDR, socket.SO_REUSEPORT):
            with socket.socket() as server, socket.socket() as client:
                server.setsockopt(socket.SOL_SOCKET, option, 1)
                server.bind(("127.0.0.1", 0))
                address = server.getsockname()
                server.listen()
                client.connect(address)
                connection, _ = server.accept()
                self.assertFalse(_port_is_available(*address, interactive=True))
                self.assertFalse(_port_is_available(*address))
                connection.close()
                self.assertEqual(client.recv(1), b"")
            self.assertTrue(_port_is_available(*address, interactive=True))
            if option == socket.SO_REUSEADDR and os.name != "nt":
                self.assertTrue(_port_is_available(*address))

    def test_interactive_runtime_discovery(self):
        from fastllm_pytools import service_backends
        root = self.directory
        package = root / "tools/fastllm_pytools"
        package.mkdir(parents=True)
        duo = service_backends.BACKENDS["duo"]
        nomas = service_backends.BACKENDS["hy_nomas"]
        relative = Path("src/models/duo/live/duo_live")
        binary = root / "build-duo" / relative
        binary.parent.mkdir(parents=True)
        binary.symlink_to(self.worker)
        lib = root / "build-hy-nomas/libfastllm_nomas.so"
        lib.parent.mkdir()
        lib.touch()
        with patch.object(service_backends, "PACKAGE", package), patch.dict(os.environ, {"PATH": "", "LD_LIBRARY_PATH": ""}):
            self.assertEqual(duo.executable(), str(binary))
            self.assertEqual(nomas.library(), lib)
            packed = package / relative.name
            packed.symlink_to(self.worker)
            (package / lib.name).touch()
            self.assertEqual(duo.executable(), str(packed))
            self.assertEqual(nomas.library(), package / lib.name)

    def test_duo_tp_device_selection(self):
        from fastllm_pytools.cli import main
        mask = "7,6,5,4,3,2,1,0"
        base = ["server", str(self.models["duo"])]
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": mask}), patch("os.execvpe") as execute:
            for flags, profile, selected in [
                (["--tp", "8"], "eight", mask),
                (["--tp", "2"], "two", "7,6"),
                (["--tp", "cuda:3"], "single", "4"),
                (["--tp", "1,3"], "two", "6,4"),
                (["--tp", "2", "--devices", "3,1"], "two", "4,6"),
            ]:
                with self.subTest(flags=flags):
                    main(base + flags)
                    _, argv, env = execute.call_args.args
                    self.assertEqual(argv[-1], profile)
                    self.assertEqual(env["CUDA_VISIBLE_DEVICES"], selected)
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "5,3,1"}), \
                    patch("fastllm_pytools.util._cuda_device_count", return_value=8):
                main(base + ["--tp", "auto"])
                self.assertEqual(execute.call_args.args[2]["CUDA_VISIBLE_DEVICES"], "5,3")
            for flags in (["--tp", "3"], ["--tp", "8", "--devices", "0,1"], ["--tp", "0,8"],
                          ["--tp", "2,2"], ["--tp", "0,1", "--devices", "1,0"]):
                with self.subTest(flags=flags), self.assertRaises(SystemExit), patch("sys.stderr"):
                    main(base + flags)

    def test_nomas_prompt_enhancement_profile(self):
        from fastllm_pytools.service_backends import BACKENDS
        (self.directory / "tokenizer.json").touch()
        config = config_from_dict(self.profile("hy_nomas", api_key="", nomas_text_encoder=str(self.directory)))
        backend = BACKENDS["hy_nomas"]
        flag = "--prompt-enhancer-url"
        self.assertNotIn(flag, backend.argv(config))
        config.nomas_pe_url = "http://127.0.0.1:8008/v1/chat/completions"
        config.nomas_pe_prompts = str(self.directory)
        self.assertTrue(any("提示词文件" in error for error in validate_config(config)))
        for name in ("system_prompt", "first_turn"):
            (self.directory / f"online_user_video_rewriter_{name}.md").touch()
        self.assertEqual(validate_config(config), [])
        self.assertIn(flag, build_fastllm_argv(config))
        self.assertEqual(backend.argv(config)[-6:], [flag, config.nomas_pe_url, "--prompt-enhancer-model",
                                                     "Nomas-PE", "--prompt-enhancer-prompts", str(self.directory)])
        config.nomas_pe_model = "my-pe"
        self.assertIn("my-pe", backend.argv(config))

    def test_interactive_profiles_and_scoped_players(self):
        from fastllm_pytools.service_backends import BACKENDS
        (self.directory / "tokenizer.json").touch()
        for backend in ("hy_nomas", "duo"):
            options = dict(api_key="", nomas_text_encoder=str(self.directory))
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "5,4,3,2"}):
                config = config_from_dict(self.profile(backend, visible_gpus="3,2,1,0", **options))
                self.assertEqual(validate_config(config), [])
                self.assertEqual(build_fastllm_env(config)["CUDA_VISIBLE_DEVICES"], "2,3,4,5")
                self.assertEqual(build_fastllm_argv(config)[:3], ["ftllm", "server", config.model])
                argv = BACKENDS[backend].argv(config)
                if backend == "duo":
                    self.assertEqual(argv[-1], "four")
                    config.visible_gpus = "0,1"
                    self.assertEqual(validate_config(config), [])
                    self.assertEqual(BACKENDS[backend].argv(config)[-1], "two")
                    config.visible_gpus = "0,1,2"
                    self.assertTrue(validate_config(config))
                    config.visible_gpus = ""
                    self.assertEqual(BACKENDS[backend].argv(config)[-1], "single")
                else:
                    self.assertEqual(argv[-2:], ["--devices", "0,1,2,3"])
                    self.assertIn("bf16", argv)
            def stub(cfg):
                return [str(self.worker), "--host", "127.0.0.1", "--port", cfg.port, "--kind", backend]
            with patch.object(BACKENDS[backend], "argv", stub):
                session = self.start(backend, **options)
            path = "/demos/" + session + "/"
            self.assertEqual(self.client.get(path).status_code, 403)
            opened = self.client.post("/api/services/" + session + "/demo", headers=self.headers)
            self.assertEqual(opened.json(), {"url": path})
            page = self.client.get(path)
            self.assertEqual(page.status_code, 200)
            self.assertIn("text/html", page.headers["content-type"])
            self.assertEqual(self.client.get(path + "api/runtime").status_code, 404)
            self.assertEqual(self.pool.catalog(), [])
            if backend == "hy_nomas":
                sample = self.client.get(path + "nomas-cafe.webp")
                self.assertEqual(sample.headers["content-type"], "image/webp")
                self.assertEqual(sample.content[8:12], b"WEBP")
                world = path + "v1/worlds/world_" + "a" * 32
                self.assertEqual(self.client.patch(world, content=b"edit").json()["data"], "edit")
                self.assertEqual(self.client.delete(world, headers={"origin":"https://elsewhere.test"}).status_code, 403)
                self.assertEqual(self.client.delete(world).json()["method"], "DELETE")
                video = path + "v1/videos/video_" + "a" * 32
                for suffix in ("/content", "?format=mp4"):
                    self.assertEqual(self.client.get(video + suffix).headers["content-type"], "video/mp4")
                events = self.client.get(video + "/events")
                self.assertEqual(events.content, b"data: {}\n\n")
            else:
                self.assertIn("DuoAvcPlayer", self.client.get(path + "avc-player.js").text)
                self.assertEqual(self.client.post(path + "sessions?resolution=512x384").json()["path"], "/sessions?resolution=512x384")
                custom = self.client.post(path + "sessions/custom?resolution=256x192",
                    files={"reference_rgb": ("portrait.rgb", bytes(256 * 192 * 3), "application/octet-stream")},
                    data={"description": "A café attendant"})
                self.assertEqual(custom.json()["path"], "/sessions/custom?resolution=256x192")
                self.assertGreater(custom.json()["size"], 256 * 192 * 3)
                turn = path + "sessions/" + "b" * 32
                self.assertEqual(self.client.post(turn + "/turns", content=b"clip").json()["size"], 4)
                self.assertEqual(self.client.get(turn + "/turn-0/avc").content, b"frame")
            self.pool.stop(session)
            self.assertNotEqual(self.client.get(path).status_code, 200)

    def test_automatically_detected_qwen_and_lingbot_demos(self):
        qwen = self.start("qwen_image")
        with patch("fastllm_pytools.service_backends.LingBotBackend.argv",
                   lambda backend, cfg: [str(self.worker), "--kind", "hy_nomas", "--host", "127.0.0.1", "--port", cfg.port]):
            world = self.start("lingbot_world", api_key="")
        for session, title in ((qwen, "Qwen-Image"), (world, "LingBot")):
            response = self.client.post(f"/api/services/{session}/demo", headers=self.headers)
            self.assertEqual(response.status_code, 200, response.text)
            path = response.json()["url"]
            self.assertIn(title, self.client.get(path).text)
            self.assertEqual(self.client.get(path + "health").status_code, 200)
            self.assertEqual(self.client.get(path + "sessions").status_code, 404)
        response = self.client.post(f"/demos/{qwen}/v1/images/jobs", json={"prompt": "test"})
        self.assertEqual(response.status_code, 200, response.text)
        response = self.client.post(f"/demos/{world}/v1/worlds", json={"prompt": "test"})
        self.assertEqual(response.status_code, 200, response.text)

    def test_bundled_video_backend_is_found_without_path(self):
        from fastllm_pytools import service_backends
        binary = self.directory / "minimax_h3_server"
        binary.touch()
        binary.chmod(0o755)
        with patch.object(service_backends, "PACKAGE", self.directory), \
                patch.dict(os.environ, {"PATH": ""}):
            self.assertEqual(service_backends.H3Backend.executable(), str(binary))

    def test_image_gguf_path_reaches_native_service_without_precision_flags(self):
        model = self.directory / "image-Q4_K_M.gguf"
        def gguf_string(value):
            raw = value.encode()
            return struct.pack("<Q", len(raw)) + raw
        metadata = {"general.architecture": "wan", "config": json.dumps({"transformer": {"pdd_steps": 6, "pdd_sigmas": [1]}})}
        model.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(metadata)) + b"".join(
            gguf_string(key) + struct.pack("<I", 8) + gguf_string(value) for key, value in metadata.items()))
        session = self.start("minimax_h3_image", model=str(model))
        self.assertTrue(self.pool.service(session).state()["ready"])
        argv = self.commands[-1][0]
        self.assertEqual(argv[4], str(model))
        self.assertNotIn("--precision", argv)

    def test_quantized_video_manifest_and_parallel_mode(self):
        from fastllm_pytools import service_backends
        view = self.directory / "quantized-video"
        view.mkdir()
        manifest = view / "h3_quantization.json"
        manifest.write_text(json.dumps({"format_version": 1, "base_model": "../base",
                                        "pika": "../adapter", "fl2va": "../weights.safetensors"}))
        for model in (str(view), str(manifest)):
            config = config_from_dict(self.profile("minimax_h3", model=model, video_parallel_mode="tp", visible_gpus="4,5"))
            command = build_fastllm_argv(config)
            self.assertEqual(command[command.index("--parallel-mode") + 1], "tp")
            self.assertEqual(command[command.index("--devices") + 1], "0,1")
            self.assertEqual(service_backends.H3Backend().model_name(config), "pika-h3-flex")
            self.assertNotIn("--precision", command)
            self.assertEqual(validate_config(config), [])
        config.video_parallel_mode = "invalid"
        fields = []
        validate_config(config, fields)
        self.assertTrue(any("video_parallel_mode" in item["fields"] for item in fields))
        manifest.write_text("{}")
        self.assertEqual(service_backends.H3Backend().model_name(config), "minimax-h3")

    def test_quantized_image_parallel_mode(self):
        for mode in ("auto", "tp", "sp"):
            config = config_from_dict(self.profile("minimax_h3_image", video_parallel_mode=mode,
                                                   visible_gpus="6,7"))
            command = build_fastllm_argv(config)
            self.assertEqual(command[command.index("--devices") + 1], "0,1")
            if mode == "auto":
                self.assertNotIn("--parallel-mode", command)
            else:
                self.assertEqual(command[command.index("--parallel-mode") + 1], mode)
            self.assertEqual(validate_config(config), [])
        config.video_parallel_mode = "invalid"
        fields = []
        validate_config(config, fields)
        self.assertTrue(any("video_parallel_mode" in item["fields"] for item in fields))
        self.start("minimax_h3_image", video_parallel_mode="tp", visible_gpus="6,7")
        command, environment = self.commands[-1]
        self.assertEqual(command[command.index("--parallel-mode") + 1], "tp")
        self.assertEqual(command[command.index("--devices") + 1], "0,1")
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "6,7")

    def test_gpu_mask_is_composed_and_auto_configuration_preserves_it(self):
        self.assertEqual(visible_gpu_environment("1,0", {"CUDA_VISIBLE_DEVICES": "6,7"}), {"CUDA_VISIBLE_DEVICES": "7,6"})
        self.assertEqual(visible_gpu_environment("1", {"CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b"}), {"CUDA_VISIBLE_DEVICES": "GPU-b"})
        for selection, parent in [("2", "6,7"), ("0", "-1"), ("GPU-c", "GPU-a,GPU-b"),
                                  ("0,0", "6,7"), ("0,00", "6,7"), ("1", "6,-1,7"),
                                  ("1", "6,,7"), ("1", "6,6,7")]:
            with self.assertRaises(ValueError):
                visible_gpu_environment(selection, {"CUDA_VISIBLE_DEVICES": parent})
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "6,7"}):
            config = config_from_dict(self.profile(visible_gpus="1"))
            self.assertIn("--auto", build_fastllm_argv(config))
            self.assertEqual(build_fastllm_env(config)["CUDA_VISIBLE_DEVICES"], "7")
        self.assertEqual(DeployConfig().backend, "llm")
        with self.assertRaises(ValueError):
            visible_gpu_environment("0,00", {})
        with patch("fastllm_pytools.service_backends.subprocess.run", return_value=SimpleNamespace(stdout="6, GPU-a\n7, GPU-b\n")):
            self.assertEqual(visible_gpu_environment("GPU-b", {"CUDA_VISIBLE_DEVICES": "6,7"}), {"CUDA_VISIBLE_DEVICES": "7"})
            with self.assertRaises(ValueError):
                visible_gpu_environment("6,GPU-a", {})
        config = config_from_dict(self.profile("minimax_h3", visible_gpus="0,1"))
        command = build_fastllm_argv(config)
        self.assertEqual(command[command.index("--devices") + 1], "0,1")

    def test_image_backend_discovery_and_gpu_commands(self):
        from fastllm_pytools import service_backends
        binary = self.directory / "minimax_h3_image_server"
        binary.touch()
        binary.chmod(0o755)
        with patch.object(service_backends, "PACKAGE", self.directory), \
                patch.dict(os.environ, {"PATH": ""}):
            self.assertEqual(service_backends.H3ImageBackend.executable(), str(binary))
        config = config_from_dict(self.profile("minimax_h3_image", video_gpu_memory_limit="4", video_host_memory_limit="2", base_model="unused"))
        command = service_backends.H3ImageBackend().argv(config)
        self.assertEqual(command[command.index("--device") + 1], "0")
        self.assertNotIn("--devices", command)
        self.assertNotIn("--base-model", command)
        self.assertEqual(command[command.index("--gpu-memory-limit") + 1], "4")
        self.assertEqual(command[command.index("--host-memory-limit") + 1], "2")
        self.assertEqual(validate_config(config), [])
        dual = config_from_dict(self.profile("minimax_h3_image", visible_gpus="0,1"))
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4,5"}):
            self.assertEqual(validate_config(dual), [])
            self.assertEqual(build_fastllm_env(dual)["CUDA_VISIBLE_DEVICES"], "4,5")
        command = build_fastllm_argv(dual)
        self.assertEqual(command[command.index("--devices") + 1], "0,1")
        fields = []
        validate_config(config_from_dict(self.profile("minimax_h3_image", visible_gpus="0,1,2")), fields)
        self.assertTrue(any("visible_gpus" in item["fields"] for item in fields))

    def test_image_fp8_uses_existing_model_path(self):
        checkpoint = self.directory / "arbitrary.safetensors"
        header = json.dumps({"pdd_proj_out.0.weight": {"dtype": "F8_E4M3"}}).encode()
        checkpoint.write_bytes(struct.pack("<Q", len(header)) + header + b"0")
        config = config_from_dict(self.profile("minimax_h3_image", model=str(checkpoint)))
        self.assertEqual(validate_config(config), [])
        command = build_fastllm_argv(config)
        self.assertEqual(command[2], str(checkpoint))
        self.assertNotIn("--fp8-weights", command)

    def test_image_memory_budget_validation(self):
        for field, values in (("video_gpu_memory_limit", ("-1", "1", "1.0000000001", "nan", "inf", "1048577")),
                              ("video_host_memory_limit", ("-1", "1.99", "nan", "inf", "1048577"))):
            for value in values:
                fields = []
                validate_config(config_from_dict(self.profile("minimax_h3_image", **{field: value})), fields)
                self.assertTrue(any(field in error["fields"] for error in fields), (field, value))
        for gpu, host in (("", ""), ("0", "0"), ("4.5", "2"), ("64", "16")):
            self.assertEqual(validate_config(config_from_dict(self.profile("minimax_h3_image",
                video_gpu_memory_limit=gpu, video_host_memory_limit=host))), [])

    def test_image_api_contract_authenticated_proxy_and_cancellation(self):
        image = self.start("minimax_h3_image")
        info = self.client.get("/api/runtime/api-info", headers=self.headers).json()
        self.assertEqual(info["kind"], "image")
        self.assertEqual(info["request"]["model"], "minimax-h3-image")
        self.assertEqual(info["defaults"]["num_inference_steps"], 6)
        self.assertEqual(self.pool.catalog()[0]["capabilities"], ["image.generate", "image.edit"])
        self.assertTrue(info["edit"].endswith("/v1/images/edits"))
        self.assertEqual(info["references"]["max_count"], 16)
        path = "/api/media/" + image + "/images/jobs"
        self.assertEqual(self.client.get(path).status_code, 403)
        job = self.client.post(path, headers=self.headers, json={"prompt": "test image", "size": "2048x2048"}).json()
        self.assertEqual(self.client.get(path, headers=self.headers).json()["data"][0]["id"], job["id"])
        content = self.client.get(path + "/" + job["id"] + "/content", headers=self.headers)
        self.assertEqual(content.headers["content-type"], "image/png")
        self.assertTrue(content.content.startswith(b"\x89PNG\r\n\x1a\n"))
        held = self.client.post(path, headers=self.headers, json={"prompt": "hold"}).json()
        cancelled = self.client.post(path + "/" + held["id"] + "/cancel", headers=self.headers)
        self.assertEqual(cancelled.json()["status"], "cancelled")
        self.assertEqual(self.client.post("/api/media/" + image + "/videos", headers=self.headers, json={}).status_code, 400)
        self.assertEqual(self.client.get("/api/media/" + image + "/images/jobs/not-an-id/content", headers=self.headers).status_code, 400)

    def test_image_multipart_proxy_preserves_reference_order_and_bytes(self):
        import base64
        image = self.start("minimax_h3_image")
        files = [("image[]", ("first.jpg", b"jpeg fixture\x00\xff", "image/jpeg")),
                 ("image[]", ("second.webp", b"webp fixture\x00\x80", "image/webp"))]
        for route in ("images/jobs", "images/edits"):
            path = "/api/media/" + image + "/" + route
            response = self.client.post(path, headers=self.headers,
                data={"prompt": "Compose <Picture 1> and <Picture 2>", "size": "2048x2048"}, files=files)
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertEqual(result["prompt"], "Compose <Picture 1> and <Picture 2>")
            self.assertEqual([(r["name"], base64.b64decode(r["data"]), r["type"]) for r in result["references"]],
                             [item[1] for item in files])

    def test_services_run_and_stop_independently_with_separate_gpu_environments(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "6,7"}):
            chat = self.start(visible_gpus="0")
            video = self.start("minimax_h3", visible_gpus="1")
        self.assertEqual([item[1]["CUDA_VISIBLE_DEVICES"] for item in self.commands], ["6", "7"])
        self.assertNotIn("--startup-progress", self.commands[1][0])
        self.assertEqual(len(self.pool.states()), 2)
        self.assertNotIn("test-service-key", json.dumps(self.pool.state()) + json.dumps(self.pool.logs()))
        opened = self.client.post("/api/webui/open", headers=self.headers, json={"sessionId": chat})
        self.assertEqual(opened.status_code, 200, opened.text)
        self.assertEqual(self.pool.state()["sessionId"], video)
        self.client.post("/api/services/" + video + "/stop", headers=self.headers)
        self.assertTrue(self.pool.service(chat).state()["ready"])
        self.assertEqual(self.client.get("/webui/" + chat + "/api/config").status_code, 200)

    def test_video_memory_budget_is_validated_saved_and_passed_only_to_video(self):
        payload = self.profile("minimax_h3", video_gpu_memory_limit="4.5", video_host_memory_limit="2")
        config = config_from_dict(payload)
        command = build_fastllm_argv(config)
        self.assertEqual(command[command.index("--gpu-memory-limit") + 1], "4.5")
        self.assertEqual(command[command.index("--host-memory-limit") + 1], "2")
        saved = self.pool.save_profile(None, payload)
        self.assertEqual(saved["profile"]["video_host_memory_limit"], "2")
        self.assertEqual(saved["profile"]["video_gpu_memory_limit"], "4.5")
        for invalid in ("-1", "1", "1.0000000001", "nan", "inf", "4GB", "1048577"):
            fields = []
            validate_config(config_from_dict(dict(payload, video_gpu_memory_limit=invalid)), fields)
            self.assertTrue(any("video_gpu_memory_limit" in item["fields"] for item in fields), invalid)
        for valid in ("", "0", "4", "6.5"):
            self.assertEqual(validate_config(config_from_dict(dict(payload, video_gpu_memory_limit=valid))), [])
        for invalid in ("-1", "0.5", "1", "nan", "inf", "2GB", "1048577"):
            fields = []
            validate_config(config_from_dict(dict(payload, video_host_memory_limit=invalid)), fields)
            self.assertTrue(any("video_host_memory_limit" in item["fields"] for item in fields), invalid)
        for valid in ("", "0", "2", "2.5"):
            self.assertEqual(validate_config(config_from_dict(dict(payload, video_host_memory_limit=valid))), [])
        llm = build_fastllm_argv(config_from_dict(self.profile(video_gpu_memory_limit="4", video_host_memory_limit="2")))
        self.assertNotIn("--gpu-memory-limit", llm)
        self.assertNotIn("--host-memory-limit", llm)

    def test_video_api_contract_authentication_and_scoped_proxy(self):
        started = time.monotonic()
        video = self.start("minimax_h3", env_vars="READY_DELAY=1.5")
        self.assertGreaterEqual(time.monotonic() - started, 1.5)
        info = self.client.get("/api/runtime/api-info", headers=self.headers).json()
        self.assertEqual(info["kind"], "video")
        self.assertEqual(info["request"]["model"], "minimax-h3")
        self.assertIn("ref2va", info["modes"])
        self.assertEqual(info["references"]["field"], "reference")
        self.assertEqual(info["references"]["max_count"], 12)
        self.assertEqual(info["uploads"]["max_request_bytes"], 256 * 1024 * 1024)
        self.assertNotIn("test-service-key", json.dumps(self.pool.catalog()))
        path = "/api/media/" + video + "/videos"
        self.assertEqual(self.client.get(path).status_code, 403)
        created = self.client.post(path, headers=self.headers, json={"prompt": "hello", "size": "640x384"})
        self.assertEqual(created.status_code, 200, created.text)
        job = created.json()["id"]
        self.assertEqual(self.client.get(path + "/" + job, headers=self.headers).json()["status"], "completed")
        content = self.client.get(path + "/" + job + "/content", headers=self.headers)
        self.assertEqual(content.content, b"fake-mp4")
        self.assertEqual(content.headers["content-type"], "video/mp4")
        bad = self.client.get("/api/media/" + video + "/http://example.org/", headers=self.headers)
        self.assertEqual(bad.status_code, 400)
        self.assertEqual(self.client.post("/api/runtime/start", headers=self.headers, json=[]).status_code, 400)
        # Equivalent port spellings must not bypass the starting-service guard.
        active_port = self.pool.state()["endpoint"].rsplit(":", 1)[1]
        with self.assertRaises(LauncherError):
            self.pool.start(self.profile(port="0" + active_port))

    def test_video_mixed_references_preserve_order_bytes_and_health(self):
        import base64
        video = self.start("minimax_h3")
        path = "/api/media/" + video
        health = self.client.get(path + "/health", headers=self.headers)
        self.assertTrue(health.json()["reference_model_available"])
        self.assertEqual(self.client.get(path + "/health").status_code, 403)
        self.assertEqual(self.client.post(path + "/health", headers=self.headers).status_code, 400)
        refs = [("scene.png", b"\x00\xffimage-one", "image/png"),
                ("motion.mp4", b"\x00\xf0video", "video/mp4"),
                ("person.jpg", b"\xff\xd8image-two", "image/jpeg"),
                ("sound.wav", b"RIFF\x00audio", "audio/wav")]
        result = self.client.post(path + "/videos", headers=self.headers,
            data={"prompt": "Picture 2 moves like Video 1 in Picture 1, with Audio 1."},
            files=[("reference", item) for item in refs])
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual([(r["name"], base64.b64decode(r["data"]), r["type"])
                          for r in result.json()["references"]], refs)

    def test_media_upload_limits_cover_headers_streams_and_direct_calls(self):
        from fastllm_pytools.service_backends import media_body_limit
        video = self.start("minimax_h3")
        path = "/api/media/" + video + "/videos"
        self.assertEqual(media_body_limit(path), 256 * 1024 * 1024)
        self.assertEqual(media_body_limit("videos"), 256 * 1024 * 1024)
        self.assertEqual(media_body_limit("/api/media/id/images/edits"), 32 * 1024 * 1024)
        # The middleware accepts video lengths above the former 32 MiB cap.
        response = self.client.post(path, headers={**self.headers, "Content-Length": str(33 * 1024 * 1024)},
                                    json={"prompt": "hello"})
        self.assertEqual(response.status_code, 200, response.text)
        response = self.client.post(path, headers={**self.headers, "Content-Length": str(256 * 1024 * 1024 + 1)},
                                    content=b"{}")
        self.assertEqual(response.status_code, 413)
        # A missing or misleading length cannot bypass the streamed-body check.
        with patch("fastllm_pytools.launcher_services.media_body_limit", return_value=128):
            response = self.client.post(path, headers={**self.headers, "Content-Length": "0"},
                                        content=b"x" * 129)
            self.assertEqual(response.status_code, 413)
            with self.assertRaises(LauncherError):
                self.pool.media_request(video, "POST", "videos", b"x" * 129)

    def test_shared_history_excludes_concurrent_writes_but_stops_only_its_owner(self):
        first = self.start()
        second = self.start()
        a = self.pool.embedded_webui(first, "127.0.0.1").state.runtime
        b = self.pool.embedded_webui(second, "127.0.0.1").state.runtime
        conversation = a.store.create_conversation()
        self.assertEqual(a.store.database_path, b.store.database_path)
        control = a.begin_generation(conversation)
        self.assertTrue(b.is_generating(conversation))
        with self.assertRaises(RuntimeError):
            b.begin_generation(conversation)
        self.assertTrue(b.cancel_generation(conversation))
        self.assertTrue(control.cancelled)
        # Cancellation does not release the reservation until the writer exits.
        with self.assertRaises(RuntimeError):
            b.begin_generation(conversation)
        a._end_generation(conversation, control)
        replacement = b.begin_generation(conversation)
        a.close()
        self.assertFalse(replacement.cancelled)
        b._end_generation(conversation, replacement)
        self.assertFalse(b.is_generating(conversation))

    def test_media_request_rechecks_identity_after_service_slot_is_reused(self):
        session = self.start("minimax_h3")
        item = self.pool.service(session)
        def reused(_):
            with item._lock:
                item._state["sessionId"] = "replacement"
            return item
        with patch.object(self.pool, "service", side_effect=reused):
            with self.assertRaisesRegex(LauncherError, "服务已退出或更换"):
                self.pool.media_request(session, "POST", "videos", b'{"prompt":"wrong model"}')

    def test_stop_and_restart_are_serialized_and_closed_pool_rejects_start(self):
        session = self.start()
        item = self.pool.service(session)
        entered, release, starting = threading.Event(), threading.Event(), threading.Event()
        original_stop = item.stop
        def delayed_stop():
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test did not release stop")
            return original_stop()
        def restart():
            starting.set()
            return self.pool.start(self.profile())
        with ThreadPoolExecutor(max_workers=2) as executor, patch.object(item, "stop", side_effect=delayed_stop):
            stopping = executor.submit(self.pool.stop, session)
            self.assertTrue(entered.wait(3))
            restarting = executor.submit(restart)
            self.assertTrue(starting.wait(3))
            try:
                acquired = self.pool.lock.acquire(blocking=False)
                if acquired:
                    self.pool.lock.release()
                self.assertFalse(acquired)
            finally:
                release.set()
            stopping.result(timeout=5)
            replacement = restarting.result(timeout=5)
        self.assertNotEqual(replacement["sessionId"], session)
        self.pool.close()
        with self.assertRaises(LauncherError):
            self.pool.start(self.profile())

    def test_native_agent_installation_management_follows_active_owner(self):
        first = self.start()
        second = self.start()
        class Agent:
            _thread = None
            def __init__(self, phase):
                self.phase = phase
            def state(self):
                return {"phase": self.phase}
            def stop(self):
                self.phase = "stopped"
                return self.state()
            def manage(self, operation):
                self.phase = "upgrading"
                return self.state()
        a, b = Agent("running"), Agent("stopped")
        self.pool.service(first).codex = a
        self.pool.service(second).codex = b
        self.pool.plugins.manage_runtime("codex", "upgrade")
        self.assertEqual(a.phase, "stopped")
        self.pool.select(first)
        self.assertEqual(self.pool.plugins._runtimes["codex"]()["phase"], "upgrading")
        with self.assertRaises(LauncherError):
            self.pool.open_agent("codex", "127.0.0.1", "http://localhost")
        self.pool.plugins.manage_runtime("codex", "cancel")
        self.assertEqual(b.phase, "stopped")

    def test_agent_helper_saves_video_and_cancellation_only_affects_its_jobs(self):
        video = self.start("minimax_h3")
        downloads = []
        bridge = GenerationBridge(self.pool, lambda name, body: downloads.append((name, body)))
        self.addCleanup(bridge.close)
        command = [sys.executable, str(TOOLS / "fastllm_pytools/media_agent.py"), "--bridge", bridge.url, "--token=" + bridge.token]
        result = subprocess.run(command + ["generate", "--service", video, "--prompt", "hello", "--output", "videos/result.mp4", "--wait", "0"], cwd=self.directory, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.directory / "videos/result.mp4").read_bytes(), b"fake-mp4")
        self.assertEqual(downloads[0][1], b"fake-mp4")
        bad = subprocess.run(command + ["generate", "--service", video, "--prompt", "hello", "--output", "../escape.mp4"], cwd=self.directory, capture_output=True, text=True)
        self.assertNotEqual(bad.returncode, 0)
        pending = subprocess.run(command + ["generate", "--service", video, "--prompt", "hold", "--output", "videos/hold.mp4", "--wait", "0"], cwd=self.directory, capture_output=True, text=True)
        job_id = json.loads(pending.stdout)["id"]
        bridge.close(cancel=True)
        status = json.loads(self.pool.media_request(video, "GET", "videos/" + job_id)[2])
        self.assertEqual(status["status"], "cancelled")

    def test_pi_workspace_calls_video_and_returns_a_downloadable_attachment(self):
        chat = self.start()
        self.start("minimax_h3")
        app = self.pool.embedded_webui(chat, "127.0.0.1")
        runtime = app.state.runtime
        class Pi:
            def stream(agent, **kwargs):
                command_line = kwargs["system_prompt"].split("工具命令前缀：\n", 1)[1].split("\n", 1)[0]
                command = shlex.split(command_line)
                catalog = json.loads(subprocess.check_output(command + ["catalog"]))
                result = subprocess.run(command + ["generate", "--service", catalog[0]["id"], "--prompt", "hello", "--output", "videos/agent.mp4", "--wait", "0"], cwd=kwargs["working_directory"], capture_output=True, text=True)
                if result.returncode:
                    raise RuntimeError(result.stderr)
                yield {"type": "text_delta", "delta": "Video saved to videos/agent.mp4"}
        runtime.pi_agent = Pi()
        runtime.pi_agent_error = ""
        conversation_id = runtime.store.create_conversation()
        messages = [{"role": "user", "content": "Generate a video"}]
        events = [json.loads(line) for line in runtime.generate_workspace_agent(conversation_id, messages,
            {"agent_workspace": str(self.directory)}, "Video")]
        final = events[-1]["message"]
        self.assertNotIn("error", final, str(events))
        self.assertEqual(final["attachments"][0]["kind"], "video")
        self.client.post("/api/webui/open", headers=self.headers, json={"sessionId": chat})
        response = self.client.get("/webui/" + chat + final["attachments"][0]["url"])
        self.assertEqual(response.content, b"fake-mp4")

    def test_browser_profiles_service_selection_and_video_demo(self):
        try:
            from playwright.sync_api import sync_playwright, expect
        except ImportError:
            self.skipTest("Playwright is optional")
        import uvicorn
        chat = self.start()
        video = self.start("minimax_h3")
        other_video = self.start("minimax_h3")
        server = uvicorn.Server(uvicorn.Config(create_launcher_app(self.pool, "control-key"), host="127.0.0.1", port=int(port()), log_level="error"))
        thread = threading.Thread(target=server.run, daemon=True); thread.start()
        def stop_server():
            server.should_exit = True; thread.join(timeout=5)
        self.addCleanup(stop_server)
        deadline = time.monotonic() + 5
        while not server.started and time.monotonic() < deadline:
            time.sleep(.01)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.route("**/api/hardware*", lambda route: route.fulfill(json={"gpus": [
                {"index": "0", "name": "Test GPU A", "uuid": "GPU-a"},
                {"index": "1", "name": "Test GPU B", "uuid": "GPU-b"}]}))
            errors = []; page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.config.port}/?token=control-key")
            expect(page.locator("#config-path")).to_contain_text("profiles.json")
            page.locator("#language-select").select_option("en-US")
            page.locator("#new-profile").click()
            page.locator("#model-path").fill(str(self.models["minimax_h3"]))
            expect(page.locator("#model-backend")).to_contain_text("Pika")
            expect(page.locator('[data-field="server_binary"], [data-field="nomas_library"]')).to_have_count(0)
            expect(page.locator('[data-field="visible_gpus"]')).to_be_visible()
            expect(page.locator('[data-field="video_gpu_memory_limit"]')).to_be_visible()
            page.locator('[data-field="video_gpu_memory_limit"]').fill("3.75")
            self.assertTrue(page.locator('[data-field="video_gpu_memory_limit"]').evaluate("input => input.checkValidity()"))
            expect(page.locator('[data-field="video_host_memory_limit"]')).to_be_visible()
            page.locator('[data-field="video_host_memory_limit"]').fill("2.5")
            self.assertTrue(page.locator('[data-field="video_host_memory_limit"]').evaluate("input => input.checkValidity()"))
            page.locator("#model-path").fill(str(self.models["minimax_h3_image"]))
            expect(page.locator("#model-backend")).to_contain_text("H3 Image")
            expect(page.locator('[data-field="video_parallel_mode"]')).to_be_visible()
            page.locator('[data-field="video_parallel_mode"]').select_option("tp")
            expect(page.locator('[data-field="video_parallel_mode"]')).to_have_value("tp")
            expect(page.locator('[data-field="video_gpu_memory_limit"]')).to_be_visible()
            expect(page.locator('[data-field="video_host_memory_limit"]')).to_be_visible()
            expect(page.locator('[data-field="base_model"]')).not_to_be_visible()
            page.locator("#model-path").fill(str(self.models["minimax_h3"]))
            expect(page.locator("#model-backend")).to_contain_text("Pika")
            page.locator('#gpu-choices input[value="1"]').check()
            expect(page.locator('[data-field="visible_gpus"]')).to_have_value("1")
            expect(page.locator("#configuration-mode-settings")).not_to_be_visible()
            page.locator("#close-profile-editor").click()
            page.locator("#confirmation-confirm").click()
            page.locator("#service-selector").select_option(video)
            page.locator("#open-api-info").click()
            expect(page.locator("#api-contract")).to_have_value(re.compile(r"/v1/videos"))
            self.assertIn("/v1/videos", page.locator("#api-contract").input_value())
            page.locator("#api-info-close").click()
            page.locator("#video-service").select_option(other_video, force=True)
            page.locator("#open-endpoint").click()
            expect(page.locator("#video-service")).to_have_value(video)
            expect(page.locator("#video-studio")).to_be_visible()
            expect(page.locator("#video-runtime")).to_contain_text("FP8")
            expect(page.locator("#video-runtime")).to_contain_text("SP × 2")
            page.locator('#video-studio [data-mode="reference"]').click()
            expect(page.locator("#video-runtime")).to_contain_text("GGUF")
            page.locator('#video-studio [data-mode="text"]').click()
            page.locator("#video-prompt").fill("A cat walking")
            page.locator("#video-generate").click()
            expect(page.locator(".video-job")).to_have_count(1)
            expect(page.locator(".video-job p")).to_have_text("A cat walking")
            page.get_by_role("button", name="Preview and download").click()
            expect(page.locator("#video-player")).to_be_visible()
            expect(page.get_by_role("link", name="Download video")).to_be_visible()
            page.locator("#video-prompt").fill("A second video")
            page.locator("#video-generate").click()
            expect(page.locator(".video-job")).to_have_count(2)
            expect(page.locator(".video-job p").first).to_have_text("A second video")
            page.locator(".video-job").filter(has_text="A cat walking").click()
            page.locator("#language-select").select_option("zh-CN")
            expect(page.get_by_role("link", name="下载视频")).to_be_visible()
            expect(page.locator("#video-player")).to_be_visible()
            expect(page.locator("#config-path")).to_contain_text("profiles.json")
            page.locator("#language-select").select_option("en-US")
            page.locator("#studio-chat-tab").click()
            expect(page.locator("#webui-content")).to_be_visible()
            self.assertEqual(page.locator("#studio-chat-select").input_value(), chat)
            page.locator("#studio-video-tab").click()
            self.assertEqual(errors, [])
            browser.close()


class BridgeLifecycleTest(unittest.TestCase):
    def test_cancel_catches_jobs_created_after_bridge_closes(self):
        entered, release, cancelled = threading.Event(), threading.Event(), threading.Event()
        class Services:
            def media_request(self, service, method, path, body=None):
                if path.endswith("/cancel"):
                    cancelled.set()
                    return 200, "application/json", b'{}'
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("test did not release POST")
                return 200, "application/json", b'{"id":"late_job"}'
        bridge = GenerationBridge(Services())
        self.addCleanup(bridge.close)
        def create():
            request = urllib.request.Request(bridge.url + "/service/videos", data=b'{}',
                headers={"Authorization": "Bearer " + bridge.token})
            with urllib.request.urlopen(request, timeout=5) as response:
                return json.load(response)
        with ThreadPoolExecutor(max_workers=1) as executor:
            created = executor.submit(create)
            try:
                self.assertTrue(entered.wait(3))
                bridge.close(cancel=True)
            finally:
                release.set()
            self.assertEqual(created.result(timeout=5)["id"], "late_job")
        self.assertTrue(cancelled.is_set())
