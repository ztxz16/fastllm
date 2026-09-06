from datetime import datetime
import asyncio
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
import zipfile
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


TOOLS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "tools", "fastllm_pytools")
)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from web_agent import (
    SearchResult,
    WebAgent,
    extract_page_text,
    parse_bing_rss,
    parse_sogou_html,
    parse_so_html,
    validate_public_url,
)
from code_agent import CodeAgent, is_code_file
from data_agent import DataAgent
from knowledge_agent import KnowledgeAgent
from webui_history import ChatStore, attachment_kind, conversation_title
from webui_reasoning import split_reasoning
from webui_server import (
    GenerationControl,
    OpenAIModelClient,
    add_webui_args,
    create_app,
)
from pptx_generator import generate_presentation, normalize_deck_plan


class _Upload:
    def __init__(self, name, mime_type, data):
        self.name = name
        self.type = mime_type
        self._data = data

    def getvalue(self):
        return self._data


class WebUIHistoryTests(unittest.TestCase):
    def test_conversation_round_trip_and_upload_cleanup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChatStore(temp_dir, max_upload_mb=1)
            conversation_id = store.create_conversation()
            image_buffer = io.BytesIO()
            Image.new("RGB", (4, 3), (30, 60, 90)).save(image_buffer, format="PNG")
            attachment = store.save_upload(
                conversation_id,
                _Upload("sample.png", "image/png", image_buffer.getvalue()),
            )
            messages = [
                {"role": "user", "content": "描述图片", "attachments": [attachment]},
                {"role": "assistant", "content": "好的", "reasoning": "内部思考"},
                {"role": "user", "content": "继续", "attachments": []},
            ]
            settings = {
                "system_prompt": "简洁回答",
                "thinking_level": "高",
                "web_mode": "关闭",
                "agent_mode": "knowledge",
            }
            store.save_conversation(
                conversation_id, messages, settings=settings, title="图片会话"
            )

            loaded = store.load_conversation(conversation_id)
            self.assertEqual(loaded["messages"], messages)
            self.assertEqual(loaded["settings"], settings)

            store.delete_conversation(conversation_id)
            self.assertFalse(store.has_conversation(conversation_id))
            self.assertFalse(os.path.exists(os.path.dirname(attachment["path"])))

    def test_title_is_normalized_and_limited(self):
        self.assertEqual(conversation_title("  hello\n world  "), "hello world")
        self.assertTrue(conversation_title("x" * 80).endswith("…"))

    def test_upload_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChatStore(temp_dir, max_upload_mb=1)
            conversation_id = store.create_conversation()
            with self.assertRaises(ValueError):
                store.save_upload(
                    conversation_id,
                    _Upload("large.mp4", "video/mp4", b"x" * (1024 * 1024 + 1)),
                )

    def test_document_upload_and_code_artifact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChatStore(temp_dir, max_upload_mb=1)
            conversation_id = store.create_conversation()
            attachment = store.save_upload(
                conversation_id,
                _Upload("notes.md", "text/markdown", "重要结论".encode()),
            )
            self.assertEqual(attachment["kind"], "document")
            self.assertEqual(attachment_kind(".cpp", ""), "document")
            self.assertEqual(attachment_kind("", "", "Makefile"), "document")

            patch_path = store.new_code_patch_path(conversation_id)
            patch_path.write_text("--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-a\n+b\n")
            self.assertEqual(
                store.code_patch_path(conversation_id, patch_path.name),
                patch_path,
            )


class CodeAgentTests(unittest.TestCase):
    def test_cross_file_context_and_validated_patch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            app_path = Path(temp_dir) / "app.py"
            helper_path = Path(temp_dir) / "helper.cpp"
            app_path.write_text(
                "from helper import divide\nprint(divide(4, 2))\n",
                encoding="utf-8",
            )
            helper_path.write_text(
                "int divide(int value, int divisor) {\n"
                "    return value / divisor;\n}\n",
                encoding="utf-8",
            )
            agent = CodeAgent(max_context_chars=12000)
            sources, warnings = agent.load([
                {"name": "app.py", "path": str(app_path), "mime_type": "text/x-python"},
                {"name": "helper.cpp", "path": str(helper_path), "mime_type": "text/x-c++"},
            ])
            self.assertEqual(warnings, [])
            self.assertEqual(len(sources), 2)
            project = agent.project_context("检查除零问题", sources)
            self.assertEqual(project["files"], 2)
            self.assertIn("app.py", project["context"])
            self.assertIn("helper.cpp", project["context"])
            self.assertIn("return value / divisor", project["context"])
            self.assertEqual(
                {source["kind"] for source in project["sources"]}, {"code"})

            parsed = agent.extract_patch(
                "修复如下：\n```diff\n"
                "--- a/helper.cpp\n+++ b/helper.cpp\n"
                "@@ -1,3 +1,4 @@\n"
                " int divide(int value, int divisor) {\n"
                "+    if (divisor == 0) return 0;\n"
                "     return value / divisor;\n }\n```\n")
            self.assertEqual(parsed["files"], 1)
            self.assertEqual(parsed["additions"], 1)
            self.assertEqual(parsed["deletions"], 0)

            with self.assertRaises(ValueError):
                agent.extract_patch(
                    "```diff\n--- a/app.py\n+++ ../../escape.py\n"
                    "@@ -1 +1 @@\n-a\n+b\n```\n")
            with self.assertRaises(ValueError):
                agent.extract_patch(
                    "```diff\n--- a/app.py\n+++ b/app.py\n"
                    "@@ -1 +1 @@\n-a\n+b\n```\n"
                    "```diff\n--- a/other.py\n+++ b/other.py\n"
                    "@@ -1 +1 @@\n-a\n+b\n```\n")
            self.assertTrue(is_code_file("package-lock.json"))
            self.assertTrue(is_code_file("Makefile"))
            self.assertFalse(is_code_file("report.pdf"))


class KnowledgeAgentTests(unittest.TestCase):
    def test_cross_file_retrieval_and_locations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            alpha = os.path.join(temp_dir, "alpha.md")
            beta = os.path.join(temp_dir, "beta.csv")
            with open(alpha, "w", encoding="utf-8") as output:
                output.write("# 性能报告\nGPU 显存带宽利用率达到 82%。\n")
            with open(beta, "w", encoding="utf-8") as output:
                output.write("项目,数值\nCPU带宽利用率,43%\n")
            result = KnowledgeAgent(max_context_chars=8000).research(
                "对比 GPU 和 CPU 的带宽利用率",
                [
                    {"name": "alpha.md", "path": alpha},
                    {"name": "beta.csv", "path": beta},
                ],
            )
            self.assertIn("82%", result["context"])
            self.assertIn("43%", result["context"])
            self.assertEqual(
                {source["title"] for source in result["sources"]},
                {"alpha.md", "beta.csv"},
            )
            self.assertTrue(all(source["location"] for source in result["sources"]))

    def test_docx_and_xlsx_without_office_dependencies(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            docx_path = os.path.join(temp_dir, "brief.docx")
            with zipfile.ZipFile(docx_path, "w") as archive:
                archive.writestr("word/document.xml", """
                    <w:document xmlns:w="urn:test"><w:body>
                    <w:p><w:r><w:t>交付日期为九月十日</w:t></w:r></w:p>
                    </w:body></w:document>""")
            xlsx_path = os.path.join(temp_dir, "budget.xlsx")
            with zipfile.ZipFile(xlsx_path, "w") as archive:
                archive.writestr("xl/workbook.xml", """
                    <workbook xmlns:r="urn:rels"><sheets>
                    <sheet name="预算" r:id="rId1"/></sheets></workbook>""")
                archive.writestr("xl/_rels/workbook.xml.rels", """
                    <Relationships><Relationship Id="rId1"
                    Target="worksheets/sheet1.xml"/></Relationships>""")
                archive.writestr("xl/sharedStrings.xml", """
                    <sst><si><t>总预算</t></si></sst>""")
                archive.writestr("xl/worksheets/sheet1.xml", """
                    <worksheet><sheetData><row r="2">
                    <c r="A2" t="s"><v>0</v></c><c r="B2"><v>2680</v></c>
                    </row></sheetData></worksheet>""")
            result = KnowledgeAgent().research(
                "交付日期和总预算",
                [
                    {"name": "brief.docx", "path": docx_path},
                    {"name": "budget.xlsx", "path": xlsx_path},
                ],
            )
            self.assertIn("九月十日", result["context"])
            self.assertIn("A2: 总预算", result["context"])
            self.assertIn("工作表“预算”第 2-2 行", result["context"])


class DataAgentTests(unittest.TestCase):
    def test_validated_groupby_chart_and_excel_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = os.path.join(temp_dir, "sales.csv")
            with open(source, "w", encoding="utf-8") as output:
                output.write(
                    "地区,销售额,成本,备注\n华东,120,70,正常\n"
                    "华南,80,55,=2+2\n华东,100,60,正常\n")
            agent = DataAgent()
            datasets, warnings = agent.load([{
                "name": "sales.csv", "path": source,
            }])
            self.assertEqual(warnings, [])
            self.assertEqual(agent.profile(datasets)[0]["rows"], 3)
            raw_plan = {
                "title": "区域销售分析",
                "analyses": [{
                    "operation": "groupby", "dataset": "sales.csv",
                    "group_by": "地区", "value": "销售额",
                    "aggregation": "sum", "limit": 10, "chart": "bar",
                }, {
                    "operation": "python", "dataset": "sales.csv",
                    "code": "import os; os.system('false')",
                }],
            }
            plan = agent.normalize_plan(raw_plan, "分析销售", datasets)
            self.assertEqual(len(plan["analyses"]), 1)
            self.assertNotIn("code", plan["analyses"][0])
            report = agent.execute(plan, datasets, Path(temp_dir) / "output")
            rows = report["results"][0]["rows"]
            self.assertEqual(rows[0], {"地区": "华东", "sum_销售额": 220})
            artifacts = report["artifacts"]
            excel = next(item for item in artifacts
                         if item["kind"] == "analysis_report")
            chart = next(item for item in artifacts if item["kind"] == "chart")
            self.assertTrue(zipfile.is_zipfile(excel["path"]))
            with zipfile.ZipFile(excel["path"]) as workbook:
                worksheets = b"".join(
                    workbook.read(name) for name in workbook.namelist()
                    if name.startswith("xl/worksheets/sheet"))
                self.assertNotIn(b"<f>", worksheets)
            with open(chart["path"], "rb") as source_file:
                self.assertEqual(source_file.read(8), b"\x89PNG\r\n\x1a\n")

    def test_xlsx_and_json_loading(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as temp_dir:
            xlsx_path = os.path.join(temp_dir, "metrics.xlsx")
            json_path = os.path.join(temp_dir, "targets.json")
            with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
                pd.DataFrame({"月份": ["一月", "二月"], "数值": [10, 20]}).to_excel(
                    writer, sheet_name="月度", index=False)
            with open(json_path, "w", encoding="utf-8") as output:
                json.dump([{"部门": "甲", "目标": 30}], output, ensure_ascii=False)
            datasets, warnings = DataAgent().load([
                {"name": "metrics.xlsx", "path": xlsx_path},
                {"name": "targets.json", "path": json_path},
            ])
            self.assertEqual(warnings, [])
            self.assertEqual(
                [dataset.identifier for dataset in datasets],
                ["metrics.xlsx / 月度", "targets.json"],
            )
            self.assertEqual([len(dataset.frame) for dataset in datasets], [2, 1])


class WebAgentTests(unittest.TestCase):
    def test_rss_and_html_parsers(self):
        rss = b"""<?xml version="1.0"?><rss><channel>
        <item><title>First result</title><link>https://example.com/a</link>
        <description>Useful &amp; current</description></item>
        <item><title>Unsafe result</title><link>javascript:alert(1)</link>
        <description>Must not reach the UI</description></item>
        </channel></rss>"""
        results = parse_bing_rss(rss)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "First result")
        self.assertEqual(results[0].snippet, "Useful & current")
        page = "<html><style>hidden</style><h1>Title</h1><script>bad()</script><p>Body text</p></html>"
        text = extract_page_text(page)
        self.assertIn("Title", text)
        self.assertIn("Body text", text)
        self.assertNotIn("hidden", text)
        self.assertNotIn("bad", text)

        so_html = """
        <ul><li class="res-list">
          <h3 class="res-title"><a href="https://www.so.com/link"
            data-mdurl="https://example.com/game">China <em>wins</em></a></h3>
          <span class="res-list-summary">Final score <em>89-87</em></span>
        </li></ul>
        """
        so_results = parse_so_html(so_html)
        self.assertEqual(so_results, [SearchResult(
            "China wins", "https://example.com/game", "Final score 89-87")])

        sogou_html = """
        <div class="vrResult"><h3 class="vr-tit"><a class="resultLink"
          href="./tc?url=https%3A%2F%2Fexample.com%2Fofficial">
          Official <em>result</em></a></h3>
          <p>China won 89-87.</p></div>
        <div class="vrResult"><h3 class="vr-tit"><a class="resultLink"
          href="./tc?url=https%3A%2F%2Fexample.com%2Fsecond">
          Second source</a></h3><p>Same score.</p></div>
        """
        sogou_results = parse_sogou_html(sogou_html)
        self.assertEqual(sogou_results[0], SearchResult(
            "Official result", "https://example.com/official",
            "China won 89-87."))
        self.assertEqual(len(sogou_results), 2)

    def test_private_urls_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_public_url("file:///etc/passwd")
        with self.assertRaises(ValueError):
            validate_public_url("http://127.0.0.1/admin")

    def test_research_context_marks_sources(self):
        class FakeAgent(WebAgent):
            def search(self, query, limit=6):
                self.assert_query = query
                return [SearchResult("Example", "https://example.com", "Summary")]

            def read_page(self, url, limit=7000):
                return "Page body"

        result = FakeAgent().research("current topic", "深度浏览")
        self.assertIn("[1] Example", result["context"])
        self.assertIn("Page body", result["context"])
        self.assertEqual(result["sources"][0]["index"], 1)


class WebUIReasoningTests(unittest.TestCase):
    def test_think_and_kimi_reasoning_are_split(self):
        self.assertEqual(
            split_reasoning("<think>work</think>final answer"),
            ("work", "final answer"),
        )
        self.assertEqual(split_reasoning("<think>unfinished"), ("unfinished", ""))

        kimi = (
            "<|open|>think<|sep|>work<|close|>think<|sep|>"
            "<|open|>response<|sep|>answer<|close|>response<|sep|>")
        self.assertEqual(split_reasoning(kimi), ("work", "answer"))


class OpenAIModelClientTests(unittest.TestCase):
    def test_cancel_does_not_wait_for_a_blocked_response_close(self):
        class BlockingResource:
            fp = None

            def __init__(self):
                self.close_started = threading.Event()
                self.release = threading.Event()

            def close(self):
                self.close_started.set()
                self.release.wait(2)

        resource = BlockingResource()
        control = GenerationControl()
        control.bind_resource(resource)
        started = time.monotonic()
        control.cancel()
        elapsed = time.monotonic() - started
        try:
            self.assertLess(elapsed, 0.5)
            self.assertTrue(resource.close_started.wait(0.5))
            self.assertTrue(control.cancelled)
        finally:
            resource.release.set()

    def test_webui_parser_contains_only_runtime_options(self):
        import argparse

        parser = add_webui_args(argparse.ArgumentParser())
        args = parser.parse_args([
            "/models/demo",
            "--api-base", "http://127.0.0.1:8018/v1",
            "--code-max-context-chars", "32000",
            "--agent-runtime", "builtin",
            "--agent-workspace-root", "/srv/projects",
        ])
        self.assertEqual(args.model, "/models/demo")
        self.assertEqual(args.api_base, "http://127.0.0.1:8018/v1")
        self.assertEqual(args.code_max_context_chars, 32000)
        self.assertEqual(args.agent_runtime, "builtin")
        self.assertEqual(args.agent_workspace_root, "/srv/projects")
        self.assertFalse(args.allow_remote_workspace_agent)
        self.assertFalse(args.disable_workspace_agent)
        self.assertTrue(parser.parse_args(['--disable-workspace-agent']).disable_workspace_agent)
        self.assertEqual(
            parser.parse_args(["/models/demo"]).agent_runtime, "pi")
        self.assertEqual(parser.parse_args([
            "/models/demo", "--code-agent-runtime", "auto",
        ]).agent_runtime, "auto")
        self.assertFalse(hasattr(args, "device"))

    def test_model_discovery_streaming_and_qwen_thinking_mapping(self):
        requests = []

        class FakeResponse:
            def __init__(self, body=b"", lines=None):
                self.body = body
                self.lines = lines or []

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return self.body

            def __iter__(self):
                return iter(self.lines)

        def urlopen(request, timeout):
            requests.append((request, timeout))
            if request.full_url.endswith("/models"):
                return FakeResponse(json.dumps({
                    "data": [{"id": "Qwen3.8-27B-FP8"}],
                }).encode())
            payload = json.loads(request.data)
            if payload["stream"]:
                return FakeResponse(lines=[
                    b'data: {"choices":[{"delta":{"reasoning_content":"work"}}]}\n',
                    b'data: {"choices":[{"delta":{"content":"answer"}}]}\n',
                    b'data: [DONE]\n',
                ])
            return FakeResponse(json.dumps({
                "choices": [{"message": {
                    "content": "answer", "reasoning_content": "work",
                }}],
            }).encode())

        client = OpenAIModelClient(
            "http://127.0.0.1:8080", "", api_key="secret", timeout=9)
        with mock.patch("webui_server.urllib.request.urlopen", urlopen):
            self.assertEqual(client.wait_until_ready(1), "Qwen3.8-27B-FP8")
            self.assertEqual(client.complete(
                [{"role": "user", "content": "hello"}],
                max_tokens=64, thinking_level="高"), ("answer", "work"))
            self.assertEqual(list(client.stream(
                [{"role": "user", "content": "hello"}],
                max_tokens=-1, thinking_level="高", temperature=1,
                top_p=.8, top_k=1, repeat_penalty=1,
            )), [("", "work"), ("answer", "")])

        payload = json.loads(requests[-1][0].data)
        self.assertEqual(payload["model"], "Qwen3.8-27B-FP8")
        self.assertEqual(payload["max_tokens"], -1)
        self.assertEqual(payload["reasoning_effort"], "xhigh")
        self.assertTrue(payload["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(requests[-1][0].get_header("Authorization"),
                         "Bearer secret")


class WebUIServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_page_history_upload_and_streaming_api(self):
        import httpx

        class FakeAPIClient:
            model_name = "demo"
            base_url = "http://test-api/v1"

            def stream(self, messages, **kwargs):
                self.messages = messages
                self.stream_kwargs = kwargs
                return iter([("漂亮", ""), ("界面", "")])

            def complete(self, _messages, **_kwargs):
                return json.dumps({
                    "title": "FastLLM 测试演示",
                    "subtitle": "可编辑 PPTX",
                    "slides": [
                        {"type": "cover"},
                        {"type": "toc"},
                        {"type": "content", "layout": "cards",
                         "title": "核心价值", "bullets": ["高性能", "可移植"]},
                        {"type": "content", "layout": "timeline",
                         "title": "实施路径", "bullets": ["分析", "实现", "验证"]},
                        {"type": "summary", "title": "总结",
                         "bullets": ["正确性优先", "保持通用"]},
                    ],
                }, ensure_ascii=False), ""

        with tempfile.TemporaryDirectory() as temp_dir:
            args = SimpleNamespace(
                model="/models/demo", path="", title="FastLLM Test",
                history_dir=temp_dir, max_upload_mb=1,
                web_search_timeout=1.0, max_token=-1,
            )
            app = create_app(args)
            fake_api = FakeAPIClient()
            app.state.runtime.api_client = fake_api
            client = httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://testserver",
            )

            page = await client.get("/")
            self.assertEqual(page.status_code, 200)
            frontend = page.text
            for asset in ("template.html", "styles.css", "app.js"):
                response = await client.get("/assets/webui/" + asset)
                self.assertEqual(response.status_code, 200)
                frontend += response.text
            self.assertIn("FASTLLM INTELLIGENCE", frontend)
            self.assertIn("选择智能体", frontend)
            for agent in ("knowledge", "data", "ppt", "code"):
                self.assertIn(f'data-agent="{agent}"', frontend)
            for agent in ("chat", "web-fast", "web-deep"):
                self.assertNotIn(f'data-agent="{agent}"', frontend)
            self.assertIn('id="agentButton"', frontend)
            self.assertIn('id="webButton"', frontend)
            self.assertIn('id="stopButton"', frontend)
            tool_right_start = frontend.index('<div class="tool-right">')
            tool_right_end = frontend.index("</div>", tool_right_start)
            action_positions = [
                frontend.index(f'id="{button_id}"', tool_right_start)
                for button_id in ("attachButton", "agentButton", "sendButton")
            ]
            self.assertEqual(action_positions, sorted(action_positions))
            self.assertLess(action_positions[-1], tool_right_end)
            self.assertIn('id="newChat"', frontend)
            self.assertIn('id="newAgent"', frontend)
            self.assertIn('id="workspaceDialog"', frontend)
            self.assertLess(
                frontend.index('id="newChat"'),
                frontend.index('id="newAgent"'),
            )
            self.assertNotIn("state.generating", frontend)
            self.assertIn('id="languageButton"', frontend)
            self.assertIn('<span id="languageLabel">简体中文</span>', frontend)
            self.assertIn('id="conversationActionMenu"', frontend)
            self.assertIn('id="renameConversationAction"', frontend)
            self.assertIn('id="deleteConversationAction"', frontend)
            self.assertIn('[hidden] { display: none !important; }', frontend)
            self.assertNotIn('.status-line::before', frontend)
            self.assertIn('.tool-trace', frontend)
            self.assertIn('function renderToolTrace(', frontend)
            self.assertIn('["tool_start","tool_update","tool_end"]', frontend)
            self.assertIn(
                '$("#attachButton").disabled = running;', frontend)
            self.assertNotIn('id="dataButton"', frontend)
            self.assertNotIn('id="pptButton"', frontend)
            self.assertNotIn('id="openSettings"', frontend)
            self.assertNotIn('id="deleteChat"', frontend)
            self.assertNotIn('id="settingTitle"', frontend)
            self.assertIn('/assets/fastllm_icon.svg', frontend)
            icon = await client.get("/assets/fastllm_icon.svg")
            self.assertEqual(icon.status_code, 200)
            self.assertIn("image/svg+xml", icon.headers["content-type"])
            self.assertIn("FastLLM", icon.text)
            locales = await client.get("/assets/webui_locales.js")
            self.assertEqual(locales.status_code, 200)
            self.assertIn("application/javascript", locales.headers["content-type"])
            self.assertIn("FASTLLM_LOCALES", locales.text)
            self.assertIn('"zh-CN"', locales.text)
            self.assertIn('"en-US"', locales.text)
            self.assertIn('"locale.short": "简体中文"', locales.text)
            self.assertIn('"nav.new_agent": "新建Agent"', locales.text)
            self.assertIn('"tool.trace_done": "执行详情 · {count} 步"', locales.text)
            self.assertIn(
                '"status.prepare_workspace_files": "正在准备 {count} 个只读附件…"',
                locales.text,
            )
            self.assertEqual(
                (await client.get("/health")).json(), {"status": "ok"})
            config = (await client.get("/api/config")).json()
            self.assertEqual(config["max_token"], -1)
            self.assertTrue(config["agent_workspace_root"])
            self.assertTrue(config["workspace_agent_enabled"])
            self.assertIn(".py", config["code_extensions"])
            self.assertIn("makefile", config["code_filenames"])
            self.assertEqual(
                config["data_extensions"],
                [".csv", ".json", ".jsonl", ".tsv", ".xlsx"],
            )

            record = (await client.post("/api/conversations", json={})).json()
            conversation_id = record["id"]
            image_buffer = io.BytesIO()
            Image.new("RGB", (3, 2), (20, 40, 60)).save(
                image_buffer, format="PNG")
            attachment = (await client.post(
                f"/api/conversations/{conversation_id}/attachments",
                content=image_buffer.getvalue(),
                headers={"Content-Type": "image/png", "X-Filename": "sample.png"},
            )).json()
            media = await client.get(attachment["url"])
            self.assertEqual(media.status_code, 200)
            self.assertTrue(media.headers["content-type"].startswith("image/png"))

            response = await client.post(
                f"/api/conversations/{conversation_id}/chat",
                json={
                    "prompt": "你好",
                    "attachments": [{
                        "token": attachment["token"],
                        "name": attachment["name"],
                    }],
                },
            )
            events = [json.loads(line) for line in response.text.splitlines()]
            self.assertEqual(response.status_code, 200)
            self.assertTrue(any(
                event.get("message_key") == "status.prepare_model"
                for event in events
            ))
            self.assertEqual(events[-1]["type"], "done")
            self.assertEqual(events[-1]["message"]["content"], "漂亮界面")
            self.assertEqual(fake_api.stream_kwargs["max_tokens"], -1)
            image_part = fake_api.messages[0]["content"][0]
            self.assertEqual(image_part["type"], "image_url")
            self.assertTrue(image_part["image_url"]["url"].startswith(
                "data:image/png;base64,"))

            saved = (await client.get(
                f"/api/conversations/{conversation_id}")).json()
            self.assertEqual([message["role"] for message in saved["messages"]],
                             ["user", "assistant"])
            public_attachment = saved["messages"][0]["attachments"][0]
            self.assertNotIn("path", public_attachment)
            self.assertEqual(public_attachment["url"], attachment["url"])

            ppt_response = await client.post(
                f"/api/conversations/{conversation_id}/presentations",
                json={
                    "topic": "FastLLM 性能优化",
                    "audience": "GPU 工程师",
                    "slide_count": 5,
                    "style": "tech",
                    "web_mode": "关闭",
                },
            )
            ppt_events = [
                json.loads(line) for line in ppt_response.text.splitlines()]
            self.assertEqual(ppt_events[-1]["type"], "done")
            artifact = ppt_events[-1]["message"]["artifacts"][0]
            self.assertEqual(artifact["slides"], 5)
            self.assertNotIn("path", artifact)
            ppt_file = await client.get(artifact["url"])
            self.assertEqual(ppt_file.status_code, 200)
            self.assertTrue(ppt_file.content.startswith(b"PK"))
            await client.aclose()

    async def test_generations_are_per_conversation_and_can_be_cancelled(self):
        import httpx

        class ConcurrentAPIClient:
            model_name = "demo"
            base_url = "http://test-api/v1"

            def __init__(self):
                self.started = {
                    "first": threading.Event(),
                    "second": threading.Event(),
                }
                self.release = {
                    "first": threading.Event(),
                    "second": threading.Event(),
                }
                self.lock = threading.Lock()
                self.active = 0
                self.max_active = 0

            def stream(self, messages, control=None, **_kwargs):
                prompt = messages[-1]["content"]
                with self.lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                self.started[prompt].set()
                try:
                    yield (f"{prompt}-partial", "")
                    while not self.release[prompt].wait(0.02):
                        if control is not None:
                            control.check()
                    yield ("-done", "")
                finally:
                    with self.lock:
                        self.active -= 1

        with tempfile.TemporaryDirectory() as temp_dir:
            args = SimpleNamespace(
                model="/models/demo", path="", title="FastLLM Test",
                history_dir=temp_dir, max_upload_mb=1,
                agent_runtime="builtin", web_search_timeout=1.0,
                max_token=-1,
            )
            app = create_app(args)
            fake_api = ConcurrentAPIClient()
            app.state.runtime.api_client = fake_api
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://testserver",
            ) as client:
                first_id = (await client.post(
                    "/api/conversations", json={})).json()["id"]
                second_id = (await client.post(
                    "/api/conversations", json={})).json()["id"]

                first_request = asyncio.create_task(client.post(
                    f"/api/conversations/{first_id}/chat",
                    json={"prompt": "first"},
                ))
                self.assertTrue(await asyncio.to_thread(
                    fake_api.started["first"].wait, 2))
                second_request = asyncio.create_task(client.post(
                    f"/api/conversations/{second_id}/chat",
                    json={"prompt": "second"},
                ))
                self.assertTrue(await asyncio.to_thread(
                    fake_api.started["second"].wait, 2))

                self.assertEqual(fake_api.max_active, 2)
                running = (await client.get("/api/conversations")).json()
                self.assertEqual(
                    {item["id"] for item in running if item["generating"]},
                    {first_id, second_id},
                )
                duplicate = await client.post(
                    f"/api/conversations/{first_id}/chat",
                    json={"prompt": "duplicate"},
                )
                self.assertEqual(duplicate.status_code, 409)

                cancelled = await client.post(
                    f"/api/conversations/{first_id}/cancel", json={})
                self.assertEqual(cancelled.json(), {"cancelled": True})
                first_response = await asyncio.wait_for(first_request, 3)
                first_events = [
                    json.loads(line) for line in first_response.text.splitlines()]
                self.assertTrue(any(
                    event["type"] == "cancelled" for event in first_events))
                self.assertEqual(first_events[-1]["type"], "done")
                self.assertTrue(first_events[-1]["message"]["cancelled"])
                self.assertEqual(
                    first_events[-1]["message"]["content"], "first-partial")

                fake_api.release["second"].set()
                second_response = await asyncio.wait_for(second_request, 3)
                second_events = [
                    json.loads(line) for line in second_response.text.splitlines()]
                self.assertEqual(
                    second_events[-1]["message"]["content"],
                    "second-partial-done",
                )
                finished = (await client.get("/api/conversations")).json()
                self.assertFalse(any(item["generating"] for item in finished))

    async def test_workspace_agent_browses_scoped_directories_and_uses_pi(self):
        import httpx

        class FakeAPIClient:
            model_name = "demo"
            base_url = "http://test-api/v1"
            api_key = ""

            def complete(self, messages, **kwargs):
                self.image_messages = messages
                self.image_kwargs = kwargs
                return "图片主色为红色。", ""

        class FakePiAgent:
            def info(self):
                return {"available": True, "pi_version": "test"}

            def stream(
                self, prompt, files, system_prompt, thinking_level,
                working_directory, cancel_event=None,
            ):
                self.prompt = prompt
                self.files = files
                self.system_prompt = system_prompt
                self.thinking_level = thinking_level
                self.working_directory = working_directory
                self.cancel_event = cancel_event
                if "触发失败" in prompt:
                    yield {
                        "type": "tool_start",
                        "id": "call-before-error",
                        "name": "read",
                        "arguments": {"path": "missing.txt"},
                    }
                    raise RuntimeError("simulated Pi failure")
                yield {
                    "type": "tool_start",
                    "id": "call-workspace-test",
                    "name": "bash",
                    "arguments": {"command": "python -m pytest"},
                }
                yield {
                    "type": "tool_update",
                    "id": "call-workspace-test",
                    "name": "bash",
                    "result": "tests running",
                }
                yield {
                    "type": "tool_end",
                    "id": "call-workspace-test",
                    "name": "bash",
                    "result": "37 passed",
                    "is_error": False,
                }
                yield {"type": "text_delta", "text": "修改并验证完成。"}
                yield {"type": "done", "turns": 2}

        with tempfile.TemporaryDirectory() as temp_dir:
            container = Path(temp_dir)
            workspace_root = container / "projects"
            project = workspace_root / "demo"
            nested = project / "src"
            outside = container / "outside"
            nested.mkdir(parents=True)
            outside.mkdir()
            try:
                (workspace_root / "escape").symlink_to(outside, target_is_directory=True)
            except OSError:
                pass
            args = SimpleNamespace(
                model="/models/demo", path="", title="FastLLM Test",
                history_dir=str(container / "history"), max_upload_mb=1,
                agent_workspace_root=str(workspace_root),
                agent_runtime="builtin", web_search_timeout=1.0,
                max_token=-1,
            )
            app = create_app(args)
            fake_pi = FakePiAgent()
            fake_api = FakeAPIClient()
            app.state.runtime.api_client = fake_api
            app.state.runtime.agent_runtime = "pi"
            app.state.runtime.pi_agent_class = FakePiAgent
            app.state.runtime.pi_agent = fake_pi
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://testserver",
            ) as client:
                listing = (await client.get("/api/agent/directories")).json()
                self.assertEqual(listing["path"], str(workspace_root))
                self.assertEqual(
                    [item["name"] for item in listing["directories"]],
                    ["demo"],
                )
                child_listing = (await client.get(
                    "/api/agent/directories", params={"path": str(project)},
                )).json()
                self.assertEqual(child_listing["parent"], str(workspace_root))
                self.assertEqual(child_listing["directories"][0]["name"], "src")
                escaped = await client.get(
                    "/api/agent/directories", params={"path": str(outside)},
                )
                self.assertEqual(escaped.status_code, 400)

                record = (await client.post(
                    "/api/conversations",
                    json={"settings": {
                        "agent_mode": "workspace",
                        "agent_workspace": str(project),
                        "thinking_level": "关闭",
                    }},
                )).json()
                self.assertEqual(record["title"], "Agent · demo")
                self.assertEqual(
                    record["settings"]["agent_workspace"], str(project))
                rejected = await client.patch(
                    f"/api/conversations/{record['id']}",
                    json={"settings": {
                        "agent_mode": "workspace",
                        "agent_workspace": str(outside),
                    }},
                )
                self.assertEqual(rejected.status_code, 400)

                response = await client.post(
                    f"/api/conversations/{record['id']}/chat",
                    json={"prompt": "运行测试并修复失败项"},
                )
                events = [
                    json.loads(line) for line in response.text.splitlines()]
                tool_events = [event for event in events
                               if event["type"].startswith("tool_")]
                self.assertEqual(
                    [event["type"] for event in tool_events],
                    ["tool_start", "tool_update", "tool_end"],
                )
                self.assertEqual(
                    tool_events[0]["tool_call"]["arguments"]["command"],
                    "python -m pytest",
                )
                self.assertEqual(
                    tool_events[-1]["tool_call"]["result"], "37 passed")
                self.assertEqual(
                    tool_events[-1]["tool_call"]["status"], "done")
                self.assertEqual(events[-1]["message"]["content"], "修改并验证完成。")
                self.assertEqual(events[-1]["message"]["agent_runtime"], "pi")
                self.assertEqual(events[-1]["message"]["tool_calls"][0]["name"], "bash")
                self.assertEqual(
                    events[-1]["message"]["tool_calls"][0]["result"],
                    "37 passed",
                )
                self.assertEqual(fake_pi.working_directory, str(project))
                self.assertEqual(fake_pi.files, [])
                self.assertIn("read、grep、find", fake_pi.system_prompt)

                code_attachment = (await client.post(
                    f"/api/conversations/{record['id']}/attachments",
                    content=b"def attached_answer():\n    return 42\n",
                    headers={
                        "Content-Type": "text/x-python",
                        "X-Filename": "attached.py",
                    },
                )).json()
                brief_attachment = (await client.post(
                    f"/api/conversations/{record['id']}/attachments",
                    content="需求代号：晨星。必须保留兼容接口。".encode(),
                    headers={
                        "Content-Type": "text/markdown",
                        "X-Filename": "brief.md",
                    },
                )).json()
                image_attachment = (await client.post(
                    f"/api/conversations/{record['id']}/attachments",
                    content=b"small-image-fixture",
                    headers={
                        "Content-Type": "image/png",
                        "X-Filename": "reference.png",
                    },
                )).json()
                attached_response = await client.post(
                    f"/api/conversations/{record['id']}/chat",
                    json={
                        "prompt": "按附件实现晨星需求",
                        "attachments": [
                            {"token": item["token"], "name": item["name"]}
                            for item in (
                                code_attachment,
                                brief_attachment,
                                image_attachment,
                            )
                        ],
                    },
                )
                attached_events = [json.loads(line) for line
                                   in attached_response.text.splitlines()]
                attached_message = attached_events[-1]["message"]
                self.assertEqual(fake_pi.files[0]["name"], "attached.py")
                self.assertIn("return 42", fake_pi.files[0]["text"])
                self.assertIn("晨星", fake_pi.system_prompt)
                self.assertIn("只读源码附件", fake_pi.system_prompt)
                self.assertIn("图片主色为红色", fake_pi.system_prompt)
                image_part = fake_api.image_messages[-1]["content"][0]
                self.assertTrue(
                    image_part["image_url"]["url"].startswith(
                        "data:image/png;base64,"))
                self.assertEqual(
                    {source["kind"] for source in attached_message["sources"]},
                    {"code", "document"},
                )

                failed_response = await client.post(
                    f"/api/conversations/{record['id']}/chat",
                    json={"prompt": "触发失败", "attachments": []},
                )
                failed_events = [json.loads(line) for line
                                 in failed_response.text.splitlines()]
                failed_message = failed_events[-1]["message"]
                self.assertTrue(failed_message["error"])
                self.assertEqual(failed_message["agent_runtime"], "pi")
                self.assertEqual(
                    failed_message["tool_calls"][0]["status"], "error")

    async def test_code_agent_cross_file_stream_and_patch_download(self):
        import httpx

        class FakeAPIClient:
            model_name = "demo"
            base_url = "http://test-api/v1"

            def stream(self, messages, **_kwargs):
                self.messages = messages
                return iter([
                    ("已确认除零缺陷 [代码2:L1-L3]。\n\n", ""),
                    (
                        "```diff\n--- a/helper.cpp\n+++ b/helper.cpp\n"
                        "@@ -1,3 +1,4 @@\n"
                        " int divide(int value, int divisor) {\n"
                        "+    if (divisor == 0) return 0;\n"
                        "     return value / divisor;\n }\n```",
                        "",
                    ),
                ])

        with tempfile.TemporaryDirectory() as temp_dir:
            args = SimpleNamespace(
                model="/models/demo", path="", title="FastLLM Test",
                history_dir=temp_dir, max_upload_mb=1,
                code_max_context_chars=12000,
                agent_runtime="builtin",
                web_search_timeout=1.0, max_token=-1,
            )
            app = create_app(args)
            fake_api = FakeAPIClient()
            app.state.runtime.api_client = fake_api
            client = httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://testserver",
            )
            record = (await client.post("/api/conversations", json={})).json()
            conversation_id = record["id"]
            await client.patch(
                f"/api/conversations/{conversation_id}",
                json={"settings": {
                    "agent_mode": "code", "thinking_level": "关闭",
                }},
            )
            attachments = []
            for name, mime_type, content in (
                ("app.py", "text/x-python",
                 "from helper import divide\nprint(divide(4, 2))\n"),
                ("helper.cpp", "text/x-c++",
                 "int divide(int value, int divisor) {\n"
                 "    return value / divisor;\n}\n"),
            ):
                item = (await client.post(
                    f"/api/conversations/{conversation_id}/attachments",
                    content=content.encode(),
                    headers={"Content-Type": mime_type, "X-Filename": name},
                )).json()
                attachments.append({"token": item["token"], "name": item["name"]})

            response = await client.post(
                f"/api/conversations/{conversation_id}/chat",
                json={
                    "prompt": "修复 helper.cpp 的除零问题并给出补丁",
                    "attachments": attachments,
                },
            )
            events = [json.loads(line) for line in response.text.splitlines()]
            code_event = next(event for event in events if event["type"] == "code")
            self.assertEqual(code_event["files"], 2)
            self.assertEqual(
                {source["title"] for source in code_event["sources"]},
                {"app.py", "helper.cpp"},
            )
            self.assertIn("app.py", fake_api.messages[0]["content"])
            self.assertIn("return value / divisor", fake_api.messages[0]["content"])

            message = events[-1]["message"]
            self.assertEqual(events[-1]["type"], "done")
            self.assertEqual({source["kind"] for source in message["sources"]},
                             {"code"})
            self.assertTrue(all("path" not in source
                                for source in message["sources"]))
            artifact = message["artifacts"][0]
            self.assertEqual(artifact["kind"], "code_patch")
            self.assertEqual(artifact["files"], 1)
            self.assertEqual(artifact["additions"], 1)
            self.assertNotIn("path", artifact)
            patch_response = await client.get(artifact["url"])
            self.assertEqual(patch_response.status_code, 200)
            self.assertIn("text/x-diff", patch_response.headers["content-type"])
            self.assertIn(b"if (divisor == 0)", patch_response.content)
            await client.aclose()

    async def test_code_agent_can_stream_through_pi_bridge(self):
        import httpx

        class FakeAPIClient:
            model_name = "demo"
            base_url = "http://test-api/v1"
            api_key = ""

            def stream(self, *_args, **_kwargs):
                raise AssertionError("builtin model stream must not be used")

        class FakePiAgent:
            def info(self):
                return {"available": True, "pi_version": "test"}

            def stream(
                self, prompt, files, system_prompt, thinking_level,
                cancel_event=None,
            ):
                self.prompt = prompt
                self.files = files
                self.system_prompt = system_prompt
                self.thinking_level = thinking_level
                self.cancel_event = cancel_event
                yield {
                    "type": "tool_start",
                    "name": "read_project_file",
                    "arguments": {"file": 1},
                }
                yield {"type": "tool_end", "name": "read_project_file"}
                yield {
                    "type": "text_delta",
                    "text": "入口返回 42 [代码1:L1-L2]。",
                }
                yield {"type": "done", "turns": 2}

        with tempfile.TemporaryDirectory() as temp_dir:
            args = SimpleNamespace(
                model="/models/demo", path="", title="FastLLM Test",
                history_dir=temp_dir, max_upload_mb=1,
                code_max_context_chars=12000,
                agent_runtime="builtin",
                web_search_timeout=1.0, max_token=-1,
            )
            app = create_app(args)
            fake_pi = FakePiAgent()
            app.state.runtime.api_client = FakeAPIClient()
            app.state.runtime.agent_runtime = "pi"
            app.state.runtime.pi_agent_class = FakePiAgent
            app.state.runtime.pi_agent = fake_pi
            client = httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://testserver",
            )
            record = (await client.post("/api/conversations", json={})).json()
            conversation_id = record["id"]
            await client.patch(
                f"/api/conversations/{conversation_id}",
                json={"settings": {
                    "agent_mode": "code", "thinking_level": "关闭",
                }},
            )
            attachment = (await client.post(
                f"/api/conversations/{conversation_id}/attachments",
                content=b"def answer():\n    return 42\n",
                headers={"Content-Type": "text/x-python", "X-Filename": "app.py"},
            )).json()
            response = await client.post(
                f"/api/conversations/{conversation_id}/chat",
                json={
                    "prompt": "answer 返回什么？",
                    "attachments": [{
                        "token": attachment["token"],
                        "name": attachment["name"],
                    }],
                },
            )
            events = [json.loads(line) for line in response.text.splitlines()]
            message = events[-1]["message"]
            self.assertEqual(message["agent_runtime"], "pi")
            self.assertEqual(message["tool_calls"][0]["name"], "read_project_file")
            self.assertEqual(message["content"], "入口返回 42 [代码1:L1-L2]。")
            self.assertEqual(fake_pi.files[0]["name"], "app.py")
            self.assertIn("return 42", fake_pi.files[0]["text"])
            self.assertIn("list_project_files", fake_pi.system_prompt)
            await client.aclose()

    async def test_web_search_can_stream_through_pi_bridge(self):
        import httpx

        class FakeAPIClient:
            model_name = "demo"
            base_url = "http://test-api/v1"
            api_key = ""

            def stream(self, *_args, **_kwargs):
                raise AssertionError("builtin model stream must not be used")

        class FakePiAgent:
            def info(self):
                return {"available": True, "pi_version": "test"}

            def stream(
                self, prompt, files, system_prompt, thinking_level, web_backend,
                cancel_event=None,
            ):
                self.prompt = prompt
                self.files = files
                self.system_prompt = system_prompt
                self.web_backend = web_backend
                self.cancel_event = cancel_event
                yield {
                    "type": "tool_start",
                    "name": "web_search",
                    "arguments": {"query": "China Lebanon basketball 2026"},
                }
                sources = [{
                    "index": 1,
                    "title": "Official result",
                    "url": "https://example.com/game",
                    "snippet": "China won 89-87",
                }]
                yield {"type": "web_sources", "sources": sources}
                yield {
                    "type": "text_delta",
                    "text": "中国队以 89:87 获胜。[网页1]",
                }
                yield {"type": "done", "turns": 2, "web_sources": sources}

        with tempfile.TemporaryDirectory() as temp_dir:
            args = SimpleNamespace(
                model="/models/demo", path="", title="FastLLM Test",
                history_dir=temp_dir, max_upload_mb=1,
                agent_runtime="builtin",
                web_search_timeout=1.0, max_token=-1,
            )
            app = create_app(args)
            fake_pi = FakePiAgent()
            app.state.runtime.api_client = FakeAPIClient()
            app.state.runtime.agent_runtime = "pi"
            app.state.runtime.pi_agent_class = FakePiAgent
            app.state.runtime.pi_agent = fake_pi
            client = httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://testserver",
            )
            record = (await client.post("/api/conversations", json={})).json()
            conversation_id = record["id"]
            await client.patch(
                f"/api/conversations/{conversation_id}",
                json={"settings": {
                    "agent_mode": "chat",
                    "web_mode": "快速搜索",
                    "thinking_level": "关闭",
                }},
            )
            response = await client.post(
                f"/api/conversations/{conversation_id}/chat",
                json={"prompt": "中国和黎巴嫩最新篮球比赛谁赢了？"},
            )
            events = [json.loads(line) for line in response.text.splitlines()]
            message = events[-1]["message"]
            self.assertEqual(message["agent_runtime"], "pi")
            self.assertEqual(message["tool_calls"][0]["name"], "web_search")
            self.assertEqual(message["content"], "中国队以 89:87 获胜。[网页1]")
            self.assertEqual(message["sources"][0]["url"],
                             "https://example.com/game")
            self.assertIs(fake_pi.web_backend, app.state.runtime.web_agent)
            self.assertEqual(fake_pi.files, [])
            self.assertIn("web_search", fake_pi.system_prompt)
            self.assertIn(str(datetime.now().year), fake_pi.system_prompt)
            await client.aclose()

    async def test_document_knowledge_persists_across_requests(self):
        import httpx

        class FakeAPIClient:
            model_name = "demo"
            base_url = "http://test-api/v1"

            def stream(self, messages, **_kwargs):
                self.messages = messages
                return iter([("依据资料回答 [资料1]", "")])

        with tempfile.TemporaryDirectory() as temp_dir:
            args = SimpleNamespace(
                model="/models/demo", path="", title="FastLLM Test",
                history_dir=temp_dir, max_upload_mb=1,
                web_search_timeout=1.0, max_token=-1,
            )
            app = create_app(args)
            fake_api = FakeAPIClient()
            app.state.runtime.api_client = fake_api
            client = httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://testserver",
            )
            record = (await client.post("/api/conversations", json={})).json()
            conversation_id = record["id"]
            attachment = (await client.post(
                f"/api/conversations/{conversation_id}/attachments",
                content="项目代号：北斗。峰值带宽为 812 GB/s。".encode(),
                headers={"Content-Type": "text/markdown", "X-Filename": "report.md"},
            )).json()
            self.assertEqual(attachment["kind"], "document")

            first = await client.post(
                f"/api/conversations/{conversation_id}/chat",
                json={
                    "prompt": "请阅读这份报告",
                    "attachments": [{
                        "token": attachment["token"],
                        "name": attachment["name"],
                    }],
                },
            )
            first_events = [json.loads(line) for line in first.text.splitlines()]
            self.assertTrue(any(event["type"] == "knowledge"
                                for event in first_events))

            second = await client.post(
                f"/api/conversations/{conversation_id}/chat",
                json={"prompt": "峰值带宽是多少？", "attachments": []},
            )
            second_events = [json.loads(line) for line in second.text.splitlines()]
            knowledge = next(event for event in second_events
                             if event["type"] == "knowledge")
            self.assertEqual(knowledge["sources"][0]["title"], "report.md")
            self.assertIn("第 1-1 行", knowledge["sources"][0]["location"])
            done = second_events[-1]["message"]
            self.assertEqual(done["sources"][0]["kind"], "document")
            self.assertNotIn("path", done["sources"][0])
            self.assertTrue(done["sources"][0]["url"].startswith("/api/"))
            self.assertIn("812 GB/s", fake_api.messages[0]["content"])
            downloaded = await client.get(attachment["url"])
            self.assertEqual(downloaded.status_code, 200)
            await client.aclose()

    async def test_data_analysis_stream_and_artifacts(self):
        import httpx

        class FakeAPIClient:
            model_name = "demo"
            base_url = "http://test-api/v1"

            def complete(self, _messages, **_kwargs):
                return json.dumps({
                    "title": "区域销售分析",
                    "analyses": [{
                        "operation": "groupby",
                        "dataset": "sales.csv",
                        "group_by": "地区",
                        "value": "销售额",
                        "aggregation": "sum",
                        "limit": 10,
                        "chart": "bar",
                    }],
                }, ensure_ascii=False), ""

            def stream(self, _messages, **_kwargs):
                return iter([("华东销售额最高 [数据1]", "")])

        with tempfile.TemporaryDirectory() as temp_dir:
            args = SimpleNamespace(
                model="/models/demo", path="", title="FastLLM Test",
                history_dir=temp_dir, max_upload_mb=1, data_max_rows=1000,
                web_search_timeout=1.0, max_token=-1,
            )
            app = create_app(args)
            app.state.runtime.api_client = FakeAPIClient()
            client = httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://testserver",
            )
            record = (await client.post("/api/conversations", json={})).json()
            conversation_id = record["id"]
            attachment = (await client.post(
                f"/api/conversations/{conversation_id}/attachments",
                content="地区,销售额\n华东,120\n华南,80\n华东,100\n".encode(),
                headers={"Content-Type": "text/csv", "X-Filename": "sales.csv"},
            )).json()
            response = await client.post(
                f"/api/conversations/{conversation_id}/analyses",
                json={
                    "question": "按地区汇总销售额",
                    "attachments": [{
                        "token": attachment["token"],
                        "name": attachment["name"],
                    }],
                },
            )
            events = [json.loads(line) for line in response.text.splitlines()]
            self.assertTrue(any(event["type"] == "data" for event in events))
            plan = next(event for event in events if event["type"] == "data_plan")
            self.assertEqual(plan["analyses"][0]["operation"], "groupby")
            message = events[-1]["message"]
            self.assertEqual(message["content"], "华东销售额最高 [数据1]")
            self.assertEqual(message["sources"][0]["kind"], "data")
            report = next(item for item in message["artifacts"]
                          if item["kind"] == "analysis_report")
            chart = next(item for item in message["artifacts"]
                         if item["kind"] == "chart")
            excel_response = await client.get(report["url"])
            chart_response = await client.get(chart["url"])
            self.assertTrue(zipfile.is_zipfile(io.BytesIO(excel_response.content)))
            self.assertTrue(chart_response.content.startswith(b"\x89PNG"))
            self.assertNotIn("path", report)
            await client.aclose()


class PptxGeneratorTests(unittest.TestCase):
    def test_normalize_render_and_reopen(self):
        raw = {
            "title": "FastLLM 性能工程",
            "subtitle": "从算子到端到端推理",
            "slides": [
                {"type": "cover"},
                {"type": "toc"},
                {"type": "content", "layout": "cards",
                 "title": "性能全景", "bullets": ["建立基线", "采集波形"]},
                {"type": "summary", "title": "总结",
                 "bullets": ["正确性优先", "数据驱动优化"]},
            ],
        }
        sources = [{
            "index": 1,
            "title": "FastLLM",
            "url": "https://github.com/ztxz16/fastllm",
        }, {
            "index": 1,
            "kind": "document",
            "title": "性能报告.pdf",
            "location": "第 3 页",
            "url": "/api/conversations/example/attachments/report.pdf",
        }]
        plan = normalize_deck_plan(
            raw, "FastLLM 性能工程", "GPU 工程师", 6, sources)
        self.assertEqual(len(plan["slides"]), 6)
        self.assertEqual(plan["slides"][-2]["layout"], "references")
        with tempfile.TemporaryDirectory() as temp_dir:
            output = os.path.join(temp_dir, "presentation.pptx")
            report = generate_presentation(
                plan, output, style="premium", audience="GPU 工程师")
            self.assertEqual(report["slides"], 6)
            self.assertTrue(zipfile.is_zipfile(output))
            self.assertIn("FastLLM 性能工程", report["text_by_slide"][0])


if __name__ == "__main__":
    unittest.main()
