import os
import json
from PySide6.QtCore import QStandardPaths


def get_data_dir() -> str:
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    data_dir = os.path.join(base, "FastllmStudio")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def get_models_json_path() -> str:
    return os.path.join(get_data_dir(), "models.json")


def get_settings_json_path() -> str:
    return os.path.join(get_data_dir(), "settings.json")


def load_json(path: str, default=None):
    if default is None:
        default = {}
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
