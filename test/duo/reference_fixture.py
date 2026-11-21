"""Materialize the seeded inclusionAI/Duo reference cases using only the standard library."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile


def materialize(output, variant):
    archive = Path(__file__).with_name("reference-fixtures.zip")
    with ZipFile(archive) as source:
        files = json.loads(source.read("manifest.json"))[variant]
        for name, parts in files.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Invalid fixture path {name!r}")
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    for digest in parts:
                        data = source.read("blobs/" + digest)
                        if hashlib.sha256(data).hexdigest() != digest:
                            raise ValueError(f"Damaged reference data for {name!r}")
                        stream.write(data)
                    stream.close()
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument("--lora", dest="mode", action="store_const", const="lora")
    precision.add_argument("--fp8", dest="mode", action="store_const", const="fp8")
    parser.set_defaults(mode="bf16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--production-head", action="store_true", help="Use 128-wide attention heads")
    args = parser.parse_args()
    variant = args.mode
    if args.causal:
        variant += "-causal"
    if args.production_head:
        variant += "-production"
    materialize(args.output, variant)
    print(args.output)


if __name__ == "__main__":
    main()
