import os
from PySide6.QtCore import QObject, Signal, Slot, Property, QThread


RECOMMENDED_MODELS = [
    {
        "repo_id": "Qwen/Qwen2-1.5B-Instruct",
        "name": "Qwen2-1.5B-Instruct",
        "description": "Qwen2 1.5B instruction-tuned model",
        "size": "~3GB",
    },
    {
        "repo_id": "Qwen/Qwen2-7B-Instruct",
        "name": "Qwen2-7B-Instruct",
        "description": "Qwen2 7B instruction-tuned model",
        "size": "~14GB",
    },
    {
        "repo_id": "THUDM/chatglm3-6b",
        "name": "ChatGLM3-6B",
        "description": "ChatGLM3 6B model by Tsinghua",
        "size": "~12GB",
    },
    {
        "repo_id": "meta-llama/Llama-2-7b-chat-hf",
        "name": "Llama-2-7B-Chat",
        "description": "Meta Llama 2 7B chat model",
        "size": "~14GB",
    },
    {
        "repo_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "name": "DeepSeek-R1-Distill-Qwen-1.5B",
        "description": "DeepSeek R1 distilled Qwen 1.5B",
        "size": "~3GB",
    },
]


class DownloadWorker(QThread):
    progress = Signal(str)
    finished = Signal(str, str)
    error = Signal(str, str)

    def __init__(self, repo_id: str, parent=None):
        super().__init__(parent)
        self._repo_id = repo_id

    def run(self):
        try:
            from app.utils.downloader import download_model
            self.progress.emit(self._repo_id)
            local_path = download_model(self._repo_id)
            self.finished.emit(self._repo_id, local_path)
        except Exception as e:
            self.error.emit(self._repo_id, str(e))


class ModelMarketViewModel(QObject):
    marketModelsChanged = Signal()
    downloadStarted = Signal(str, arguments=["repoId"])
    downloadFinished = Signal(str, str, arguments=["repoId", "localPath"])
    downloadError = Signal(str, str, arguments=["repoId", "error"])
    searchResultsChanged = Signal()

    def __init__(self, store=None, parent=None):
        super().__init__(parent)
        self._store = store
        self._recommended = list(RECOMMENDED_MODELS)
        self._search_results = []
        self._downloading = set()
        self._workers = []

    @Slot(result="QVariantList")
    def getRecommendedModels(self):
        result = []
        for m in self._recommended:
            item = dict(m)
            item["downloading"] = m["repo_id"] in self._downloading
            result.append(item)
        return result

    @Slot(result="QVariantList")
    def getSearchResults(self):
        result = []
        for m in self._search_results:
            item = dict(m)
            item["downloading"] = m["repo_id"] in self._downloading
            result.append(item)
        return result

    @Slot(str)
    def searchModels(self, query):
        import requests
        self._search_results = []
        try:
            hf_endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
            url = f"{hf_endpoint}/api/models?search={query}&limit=10"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            for item in resp.json():
                self._search_results.append({
                    "repo_id": item.get("id", ""),
                    "name": item.get("id", "").split("/")[-1],
                    "description": item.get("pipeline_tag", ""),
                    "size": "",
                })
        except Exception:
            pass
        self.searchResultsChanged.emit()

    @Slot(str, str)
    def downloadModel(self, repo_id, name):
        if repo_id in self._downloading:
            return
        self._downloading.add(repo_id)
        self.downloadStarted.emit(repo_id)
        self.marketModelsChanged.emit()

        worker = DownloadWorker(repo_id, self)
        worker.finished.connect(self._on_download_finished)
        worker.error.connect(self._on_download_error)
        self._workers.append(worker)
        worker.start()

    def _on_download_finished(self, repo_id, local_path):
        self._downloading.discard(repo_id)
        name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
        if self._store:
            self._store.add_model(name=name, path=local_path, source="market")
        self.downloadFinished.emit(repo_id, local_path)
        self.marketModelsChanged.emit()

    def _on_download_error(self, repo_id, error_msg):
        self._downloading.discard(repo_id)
        self.downloadError.emit(repo_id, error_msg)
        self.marketModelsChanged.emit()
