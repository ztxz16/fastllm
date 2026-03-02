from typing import List, Optional

from app.models.model_item import ModelItem, ModelLaunchConfig
from app.utils.config import get_models_json_path, load_json, save_json


class ModelStore:
    def __init__(self):
        self._models: List[ModelItem] = []
        self._load()

    def _load(self):
        data = load_json(get_models_json_path(), default=[])
        self._models = [ModelItem.from_dict(d) for d in data]

    def _save(self):
        data = [m.to_dict() for m in self._models]
        save_json(get_models_json_path(), data)

    def get_all(self) -> List[ModelItem]:
        return list(self._models)

    def get_by_id(self, model_id: str) -> Optional[ModelItem]:
        for m in self._models:
            if m.model_id == model_id:
                return m
        return None

    def add_model(self, name: str, path: str, source: str = "local") -> ModelItem:
        item = ModelItem(name=name, path=path, source=source)
        self._models.append(item)
        self._save()
        return item

    def remove_model(self, model_id: str) -> bool:
        before = len(self._models)
        self._models = [m for m in self._models if m.model_id != model_id]
        if len(self._models) < before:
            self._save()
            return True
        return False

    def update_launch_config(self, model_id: str, config: ModelLaunchConfig) -> bool:
        item = self.get_by_id(model_id)
        if item is None:
            return False
        item.launch_config = config
        self._save()
        return True

    def model_count(self) -> int:
        return len(self._models)
