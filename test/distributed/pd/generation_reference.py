"""正式生成入口的离线基线；保存完整请求与响应，不以录制成功代替准确性通过。"""

import hashlib
import json
import threading
import urllib.request


def deployment(router, token, role):
    def get(url, path):
        query = urllib.request.Request(url.rstrip("/") + path,
            headers={"Authorization": "Bearer " + token})
        with urllib.request.urlopen(query, timeout=10) as response:
            return json.load(response)

    workers = get(router, "/health")["workers"]
    selected = [worker for worker in workers if worker["role"] == role]
    if not selected:
        raise ValueError(f"Router has no {role} service")
    configs, peers, evidence = [], {}, {}
    for worker in sorted(selected, key=lambda item: item["url"]):
        url = worker["url"].rstrip("/")
        value = get(url, "/internal/evidence")
        health = value["health"]
        if (not worker["healthy"] or not health["healthy"] or health["role"] != role or
                worker["boot"] != health["peer"]["boot"] or worker["model"] != health["model"]):
            raise ValueError(f"service identity/health changed: {url}")
        replicas = []
        for replica in value["replicas"]:
            states, ranks = replica["states"], replica["workers"]
            if not states or len(states) != len(ranks):
                raise ValueError(f"incomplete native rank evidence: {url}")
            config = dict(model_type=replica["model_type"], storage=replica["storage"],
                tp=len(ranks), ranks=[dict(graph=state["enabled"],
                    prefix_cache=state["prefix_cache"]["enabled"],
                    page_len=state["prefix_cache"].get("page_len"),
                    custom_allreduce=state["allreduce"]["enabled"]) for state in states])
            if "parallel" in replica:
                config["parallel"] = replica["parallel"]
            replicas.append(config)
        if not replicas:
            raise ValueError(f"no native replica evidence: {url}")
        configs.append(dict(model=health["model"], replicas=replicas))
        peers[url] = health["peer"]
        evidence[url] = value
    return dict(config=sorted(configs, key=lambda item: json.dumps(item, sort_keys=True)),
                peers=peers, evidence=evidence)


def canonical_request(payload):
    # 每轮测试使用独立会话；除不透明的会话编号之外，请求语义必须完全一致。
    return {key: value for key, value in payload.items() if key != "session_id"}


class ReferenceArchive:
    def __init__(self, args):
        self.args = args
        self.recording = args.record_reference
        self.lock = threading.Lock()
        self.records = {}
        self.used = set()
        self.router = args.reference if self.recording else args.router
        self.role = "full" if self.recording else "d"
        self.before = deployment(self.router, args.token, self.role)
        (args.output / "deployment.before.json").write_text(json.dumps(self.before, indent=2))

        if not self.recording:
            path = args.reference_record.resolve()
            raw = path.read_bytes()
            archive = json.loads(raw)
            report_path = path.parent / "summary.json"
            report_bytes = report_path.read_bytes()
            report = json.loads(report_bytes)
            if (archive.get("version") != 1 or archive.get("complete") is not True or
                    archive.get("summary_sha256") != hashlib.sha256(report_bytes).hexdigest() or
                    report.get("status") != "PASS" or len(report.get("cases", [])) != 15 or
                    any(case["status"] != "PASS" for case in report["cases"])):
                raise ValueError("reference archive is incomplete or its report is invalid")
            if archive["config"] != self.before["config"]:
                raise ValueError("reference and D model/TP/DP/Graph/Prefix/KV configuration differ")
            self.records = archive["records"]
            if len(self.records) != 21:
                raise ValueError("reference must contain all 21 requests in the 15 cases")
            self.archive_sha256 = hashlib.sha256(raw).hexdigest()
            (args.output / "reference.source.json").write_text(json.dumps(dict(
                path=str(path), sha256=self.archive_sha256, config=archive["config"]), indent=2))

    def key(self, name):
        prefix = self.args.profile + "_"
        if not name.startswith(prefix):
            raise ValueError("case name does not match profile")
        return name[len(prefix):]

    def obtain(self, name, payload, request):
        key = self.key(name)
        if self.recording:
            result = request(self.args.reference, self.args.token, payload)
            ids = result.get("token_ids")
            if (not isinstance(ids, list) or len(ids) != payload["max_new_tokens"] or
                    any(type(token) is not int or token < 0 for token in ids)):
                raise ValueError("reference token IDs or output length are invalid")
            with self.lock:
                if key in self.records:
                    raise ValueError(f"duplicate reference request: {key}")
                self.records[key] = dict(request=payload, response=result)
                (self.args.output / "reference.incomplete.json").write_text(json.dumps(
                    dict(version=1, complete=False, config=self.before["config"], records=self.records),
                    ensure_ascii=False, indent=2, allow_nan=False))
            return result

        with self.lock:
            if key in self.used or key not in self.records:
                raise ValueError(f"duplicate or missing archived request: {key}")
            entry = self.records[key]
            if canonical_request(entry["request"]) != canonical_request(payload):
                raise ValueError(f"request differs from archived reference: {key}")
            self.used.add(key)
        return entry["response"]

    def check_deployment(self, name):
        after = deployment(self.router, self.args.token, self.role)
        (self.args.output / (name + ".deployment.json")).write_text(json.dumps(after, indent=2))
        if after["config"] != self.before["config"] or after["peers"] != self.before["peers"]:
            raise ValueError("service restarted or changed configuration during this run")
        if name.endswith("_multi_turn"):
            count = len(self.records) if self.recording else len(self.used)
            if count != 21:
                raise ValueError(f"incomplete reference request coverage: {count}/21")

    def finalize(self, code):
        if code == 0 and self.recording:
            summary = (self.args.output / "summary.json").read_bytes()
            archive = dict(version=1, complete=True, config=self.before["config"],
                summary_sha256=hashlib.sha256(summary).hexdigest(),
                source_router=self.router, source_peers=self.before["peers"], records=self.records)
            with (self.args.output / "reference.json").open("x") as stream:
                json.dump(archive, stream, ensure_ascii=False, indent=2, allow_nan=False)
