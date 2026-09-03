import json
import mimetypes
import os
import shutil
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    from .code_agent import is_code_file
    from .knowledge_agent import is_document
except ImportError:
    from code_agent import is_code_file
    from knowledge_agent import is_document


DEFAULT_SETTINGS = {
    "system_prompt": "",
    "thinking_level": "中",
    "web_mode": "关闭",
    "agent_mode": "chat",
}

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".gif", ".mkv", ".mov", ".mp4", ".webm"}


def media_kind(extension: str, mime_type: str) -> Optional[str]:
    if extension in IMAGE_EXTENSIONS or mime_type.startswith("image/"):
        return "image"
    if extension in VIDEO_EXTENSIONS or mime_type.startswith("video/"):
        return "video"
    return None


def attachment_kind(
    extension: str, mime_type: str, name: str = "",
) -> Optional[str]:
    kind = media_kind(extension, mime_type)
    if kind is not None:
        return kind
    if is_document(extension, mime_type) or is_code_file(
            name or f"source{extension}", mime_type):
        return "document"
    return None


def default_history_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".fastllm", "webui")


def conversation_title(text: str, limit: int = 28) -> str:
    title = " ".join(str(text or "").strip().split())
    if not title:
        return "新对话"
    return title if len(title) <= limit else title[:limit].rstrip() + "…"


class ChatStore:
    """Small SQLite-backed store used by the standalone WebUI."""

    def __init__(self, root_dir: Optional[str] = None, max_upload_mb: int = 512):
        self.root = Path(root_dir or default_history_dir()).expanduser().resolve()
        self.upload_root = self.root / "uploads"
        self.presentation_root = self.root / "presentations"
        self.analysis_root = self.root / "analyses"
        self.code_root = self.root / "code"
        for directory in (
            self.root,
            self.upload_root,
            self.presentation_root,
            self.analysis_root,
            self.code_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.database_path = self.root / "history.sqlite3"
        self.max_upload_bytes = max(1, int(max_upload_mb)) * 1024 * 1024
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path), timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    settings_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id)
                        ON DELETE CASCADE,
                    UNIQUE(conversation_id, position)
                );
                CREATE INDEX IF NOT EXISTS conversations_updated_idx
                    ON conversations(updated_at DESC);
                """
            )

    def create_conversation(
        self,
        title: str = "新对话",
        settings: Optional[Dict[str, Any]] = None,
    ) -> str:
        conversation_id = uuid.uuid4().hex
        now = time.time()
        normalized_settings = dict(DEFAULT_SETTINGS)
        normalized_settings.update(settings or {})
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO conversations "
                "(id, title, settings_json, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    conversation_id,
                    conversation_title(title),
                    json.dumps(normalized_settings, ensure_ascii=False),
                    now,
                    now,
                ),
            )
        return conversation_id

    def has_conversation(self, conversation_id: str) -> bool:
        if not conversation_id:
            return False
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
        return row is not None

    def list_conversations(self, limit: int = 200) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, title, created_at, updated_at FROM conversations "
                "ORDER BY updated_at DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def load_conversation(self, conversation_id: str) -> Dict[str, Any]:
        with self._connect() as connection:
            conversation = connection.execute(
                "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if conversation is None:
                raise KeyError(f"Unknown conversation: {conversation_id}")
            rows = connection.execute(
                "SELECT payload_json FROM messages "
                "WHERE conversation_id = ? ORDER BY position",
                (conversation_id,),
            ).fetchall()
        settings = dict(DEFAULT_SETTINGS)
        try:
            settings.update(json.loads(conversation["settings_json"]))
        except (TypeError, json.JSONDecodeError):
            pass
        return {
            "id": conversation["id"],
            "title": conversation["title"],
            "settings": settings,
            "created_at": conversation["created_at"],
            "updated_at": conversation["updated_at"],
            "messages": [json.loads(row["payload_json"]) for row in rows],
        }

    def save_conversation(
        self,
        conversation_id: str,
        messages: Sequence[Dict[str, Any]],
        settings: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None,
    ) -> None:
        now = time.time()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT title, settings_json FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown conversation: {conversation_id}")
            normalized_settings = dict(DEFAULT_SETTINGS)
            try:
                normalized_settings.update(json.loads(row["settings_json"]))
            except (TypeError, json.JSONDecodeError):
                pass
            normalized_settings.update(settings or {})
            connection.execute(
                "UPDATE conversations SET title = ?, settings_json = ?, "
                "updated_at = ? WHERE id = ?",
                (
                    conversation_title(title if title is not None else row["title"]),
                    json.dumps(normalized_settings, ensure_ascii=False),
                    now,
                    conversation_id,
                ),
            )
            connection.execute(
                "DELETE FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            )
            connection.executemany(
                "INSERT INTO messages "
                "(conversation_id, position, payload_json) VALUES (?, ?, ?)",
                [
                    (
                        conversation_id,
                        position,
                        json.dumps(message, ensure_ascii=False),
                    )
                    for position, message in enumerate(messages)
                ],
            )

    def delete_conversation(self, conversation_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM conversations WHERE id = ?", (conversation_id,)
            )
        for root in (
            self.upload_root,
            self.presentation_root,
            self.analysis_root,
            self.code_root,
        ):
            directory = self._conversation_storage_dir(
                root, conversation_id, create=False)
            if directory.exists():
                shutil.rmtree(directory)

    def save_upload(self, conversation_id: str, uploaded_file: Any) -> Dict[str, Any]:
        if not self.has_conversation(conversation_id):
            raise KeyError(f"Unknown conversation: {conversation_id}")
        data = uploaded_file.getvalue()
        if len(data) > self.max_upload_bytes:
            raise ValueError(
                f"文件 {uploaded_file.name!r} 超过 {self.max_upload_bytes // 1024 // 1024} MiB 限制"
            )
        original_name = os.path.basename(str(uploaded_file.name or "upload"))
        extension = Path(original_name).suffix.lower()
        mime_type = str(getattr(uploaded_file, "type", "") or "")
        kind = attachment_kind(extension, mime_type, original_name)
        if kind is None:
            raise ValueError(f"不支持的附件格式：{original_name}")
        if not extension:
            extension = mimetypes.guess_extension(mime_type) or (
                ".png" if kind == "image" else
                ".mp4" if kind == "video" else ".txt"
            )
        upload_dir = self._conversation_storage_dir(
            self.upload_root, conversation_id, create=True)
        media_path = upload_dir / f"{uuid.uuid4().hex}{extension}"
        with open(media_path, "wb") as media_file:
            media_file.write(data)
        return {
            "kind": kind,
            "name": original_name,
            "mime_type": mime_type,
            "size": len(data),
            "path": str(media_path),
        }

    @staticmethod
    def _conversation_storage_dir(
        root: Path, conversation_id: str, create: bool
    ) -> Path:
        if not conversation_id or any(
            character not in "0123456789abcdef" for character in conversation_id
        ):
            raise ValueError("Invalid conversation id")
        path = root / conversation_id
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def new_presentation_path(self, conversation_id: str) -> Path:
        if not self.has_conversation(conversation_id):
            raise KeyError(f"Unknown conversation: {conversation_id}")
        directory = self._conversation_storage_dir(
            self.presentation_root, conversation_id, create=True)
        return directory / f"{uuid.uuid4().hex}.pptx"

    def presentation_path(self, conversation_id: str, token: str) -> Path:
        return self._artifact_path(
            self.presentation_root,
            conversation_id,
            token,
            {".pptx"},
            "Presentation",
        )

    def analysis_directory(self, conversation_id: str) -> Path:
        if not self.has_conversation(conversation_id):
            raise KeyError(f"Unknown conversation: {conversation_id}")
        return self._conversation_storage_dir(
            self.analysis_root, conversation_id, create=True)

    def analysis_path(self, conversation_id: str, token: str) -> Path:
        return self._artifact_path(
            self.analysis_root,
            conversation_id,
            token,
            {".png", ".xlsx"},
            "Analysis artifact",
        )

    def new_code_patch_path(self, conversation_id: str) -> Path:
        if not self.has_conversation(conversation_id):
            raise KeyError(f"Unknown conversation: {conversation_id}")
        directory = self._conversation_storage_dir(
            self.code_root, conversation_id, create=True)
        return directory / f"{uuid.uuid4().hex}.patch"

    def code_patch_path(self, conversation_id: str, token: str) -> Path:
        return self._artifact_path(
            self.code_root,
            conversation_id,
            token,
            {".diff", ".patch"},
            "Code patch",
        )

    def _artifact_path(
        self,
        root: Path,
        conversation_id: str,
        token: str,
        extensions: set,
        label: str,
    ) -> Path:
        directory = self._conversation_storage_dir(
            root, conversation_id, create=False).resolve()
        raw_token = str(token or "")
        normalized = os.path.basename(raw_token)
        if (normalized != raw_token
                or Path(normalized).suffix.lower() not in extensions):
            raise ValueError(f"Invalid {label.lower()} token")
        path = (directory / normalized).resolve()
        try:
            path.relative_to(directory)
        except ValueError as error:
            raise ValueError(f"{label} path escapes its directory") from error
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {normalized}")
        return path
