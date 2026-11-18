"""Convert a released Duo sparse FP32 DCP to safetensors with the standard library."""

import argparse
import collections
import hashlib
import io
import json
import math
import os
import pickle
import struct
import sys
import tempfile
import zipfile
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Record:
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.fields = state


class ReleasedMetadata(pickle.Unpickler):
    """Interpret the storage descriptors as inert records."""

    def find_class(self, module, name):
        if module == "torch.distributed.checkpoint.metadata":
            if name in {
                "Metadata",
                "TensorStorageMetadata",
                "TensorProperties",
                "ChunkStorageMetadata",
                "MetadataIndex",
                "StorageMeta",
            }:
                return Record
            if name == "_MEM_FORMAT_ENCODING":
                return int
        if (module, name) == (
            "torch.distributed.checkpoint.filesystem",
            "_StorageInfo",
        ):
            return Record
        if (module, name) == ("torch.serialization", "_get_layout"):
            return str
        if (module, name) == ("torch", "Size"):
            return tuple
        if (module, name) == ("torch", "float32"):
            return "F32"
        raise ValueError(f"Unsupported DCP metadata field {module}.{name}")


class ReleasedTensor(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return lambda storage, offset, shape, strides, *_: (
                storage,
                offset,
                shape,
                strides,
            )
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if (module, name) == ("torch", "FloatStorage"):
            return "F32"
        raise ValueError(f"Unsupported DCP tensor field {module}.{name}")

    def persistent_load(self, reference):
        return reference


def read_tensor(root, info, shape):
    if Path(info.relative_path).name != info.relative_path:
        raise ValueError("Invalid DCP storage path")
    with (root / "dcp" / info.relative_path).open("rb") as shard:
        shard.seek(info.offset)
        payload = shard.read(info.length)
    with zipfile.ZipFile(io.BytesIO(payload)) as data:
        with data.open("archive/data.pkl") as obj:
            ref, offset, dims, strides = ReleasedTensor(obj).load()
        size = math.prod(shape)
        layout = tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))
        if (
            data.read("archive/byteorder") != b"little"
            or ref != ("storage", "F32", "0", "cpu", size)
            or offset != 0
            or dims != tuple(shape)
            or strides != layout
        ):
            raise ValueError("Expected a contiguous released FP32 tensor")
        raw = data.read("archive/data/0")
        if len(raw) != 4 * size:
            raise ValueError("DCP tensor size mismatch")
        return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if sys.byteorder != "little":
        raise RuntimeError("Duo import requires a little-endian host")
    root = args.model.resolve()
    if (root / "realization/complete.json").is_file():
        root /= "realization"
    manifest = json.loads((root / "complete.json").read_text())
    if manifest["format"] != "duo-release-dcp-v1" or len(manifest["models"]) != 1:
        raise ValueError("Expected a single-family DCP release; FP8 is already merged")
    family, model = next(iter(manifest["models"].items()))
    if model["mode"] != "lora" or model["role_embedding"]:
        raise ValueError(
            "Only sparse LoRA releases without role embeddings are supported"
        )
    # Verify the released immutable base and DCP identity before unpickling DCP metadata.
    for relative, identity in manifest["files"].items():
        path = root / relative
        if (
            not path.resolve().is_relative_to(root)
            or path.stat().st_size != identity["size"]
            or sha256(path) != identity["sha256"]
        ):
            raise ValueError(f"Checkpoint identity differs: {relative}")
    base = root / model["base"]["path"]
    for filename, field in (
        ("config.json", "config_sha256"),
        ("model.safetensors.index.json", "index_sha256"),
    ):
        if sha256(base / filename) != model["base"][field]:
            raise ValueError(f"Backbone identity differs: {filename}")
    with (root / "dcp/.metadata").open("rb") as f:
        directory = ReleasedMetadata(f).load()
    locations = {key.fqn: value for key, value in directory.storage_data.items()}
    prefix = f"models.{family}."
    metadata = directory.state_dict_metadata
    selected = {
        k.removeprefix(prefix): v for k, v in metadata.items() if k.startswith(prefix)
    }
    rank = model["lora"]["rank"]
    expected = {
        f"{name}.lora_{part}.default.weight"
        for name in model["lora"]["modules"]
        for part in ("A", "B")
    }
    expected.add("role_embeddings.weight")
    if set(selected) != expected:
        raise ValueError(f"Sparse tensor keys differ: {set(selected) ^ expected}")
    for name in model["lora"]["modules"]:
        a, b = (selected[f"{name}.lora_{part}.default.weight"] for part in ("A", "B"))
        if (
            len(a.size) != 2
            or len(b.size) != 2
            or a.size[0] != rank
            or b.size[1] != rank
        ):
            raise ValueError(f"Invalid LoRA shape: {name}")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing import: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".duo-import-", dir=output.parent) as temp:
        path = Path(temp)
        header, offset = {}, 0
        for name, tensor in sorted(selected.items()):
            shape = tuple(tensor.size)
            if (
                tensor.properties.fields[0] != "F32"
                or len(tensor.chunks) != 1
                or tuple(tensor.chunks[0].sizes) != shape
                or any(tensor.chunks[0].offsets)
            ):
                raise ValueError(f"Unsupported sparse DCP layout: {name}")
            end = offset + math.prod(shape) * 4
            header[name] = {
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [offset, end],
            }
            offset = end
        encoded = json.dumps(header, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        with (path / "adapter.safetensors").open("wb") as out:
            out.write(struct.pack("<Q", len(encoded)) + encoded)
            for name, desc in header.items():
                raw = read_tensor(root, locations[prefix + name], desc["shape"])
                values = memoryview(raw).cast("f")
                if not all(map(math.isfinite, values)):
                    raise ValueError(f"Nonfinite adapter: {name}")
                if name == "role_embeddings.weight" and any(values):
                    raise ValueError(
                        "Disabled role embeddings unexpectedly contain nonzero values"
                    )
                out.write(raw)
        report = {
            "format": "fastllm-duo-adapter-v1",
            "source_manifest": manifest,
            "source": str(root),
            "adapter_sha256": sha256(path / "adapter.safetensors"),
            "tensors": {
                k: {"shape": v["shape"], "dtype": "torch.float32"}
                for k, v in header.items()
            },
        }
        (path / "adapter.json").write_text(json.dumps(report, indent=2) + "\n")
        os.rename(path, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "tensors": len(header),
                "bytes": (output / "adapter.safetensors").stat().st_size,
            }
        )
    )


if __name__ == "__main__":
    main()
