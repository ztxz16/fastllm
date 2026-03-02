import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ModelLaunchConfig:
    device: str = ""
    threads: int = 4
    dtype: str = "auto"
    port: int = 0
    model_name: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelLaunchConfig":
        return cls(
            device=d.get("device", ""),
            threads=d.get("threads", 4),
            dtype=d.get("dtype", "auto"),
            port=d.get("port", 0),
            model_name=d.get("model_name", ""),
        )


@dataclass
class ModelItem:
    model_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    path: str = ""
    source: str = "local"
    launch_config: ModelLaunchConfig = field(default_factory=ModelLaunchConfig)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "path": self.path,
            "source": self.source,
            "launch_config": self.launch_config.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModelItem":
        lc = ModelLaunchConfig.from_dict(d.get("launch_config", {}))
        return cls(
            model_id=d.get("model_id", str(uuid.uuid4())),
            name=d.get("name", ""),
            path=d.get("path", ""),
            source=d.get("source", "local"),
            launch_config=lc,
        )
