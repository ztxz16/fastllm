"""Extract packed weights for cudaGgufMmqRegression.cu using only the stdlib.

Usage: python3 test/ops/prepareGgufMmqFixtures.py model.gguf output_dir
Missing formats receive deterministic synthetic blocks with finite scales.
The C++ test decodes the packed weights with independent CPU GGUF routines.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import struct


# GGML type -> (name, elements per block, bytes per block).
FORMATS = {
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    29: ("IQ1_M", 256, 56),
}
METADATA_WIDTHS = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


@dataclass
class Case:
    type_id: int
    rows: int
    columns: int
    data: bytes
    source: str


def read_exact(stream, size):
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("Truncated GGUF file")
    return data


def read_number(stream, fmt):
    return struct.unpack("<" + fmt, read_exact(stream, struct.calcsize("<" + fmt)))[0]


def read_string(stream):
    return read_exact(stream, read_number(stream, "Q")).decode("utf-8")


def skip_metadata(stream, type_id):
    if type_id in METADATA_WIDTHS:
        stream.seek(METADATA_WIDTHS[type_id], 1)
    elif type_id == 8:  # String.
        stream.seek(read_number(stream, "Q"), 1)
    elif type_id == 9:  # Array.
        element_type = read_number(stream, "I")
        count = read_number(stream, "Q")
        if element_type in METADATA_WIDTHS:
            stream.seek(METADATA_WIDTHS[element_type] * count, 1)
        else:
            for _ in range(count):
                skip_metadata(stream, element_type)
    else:
        raise ValueError(f"Unknown GGUF metadata type: {type_id}")


def read_model_cases(path):
    cases = []
    seen = set()
    with path.open("rb") as stream:
        if read_exact(stream, 4) != b"GGUF" or read_number(stream, "I") not in (2, 3):
            raise ValueError("Expected a GGUF v2/v3 file")
        tensor_count = read_number(stream, "Q")
        metadata_count = read_number(stream, "Q")
        alignment = 32
        for _ in range(metadata_count):
            key = read_string(stream)
            type_id = read_number(stream, "I")
            if key == "general.alignment":
                if type_id != 4:
                    raise ValueError("GGUF alignment must be uint32")
                alignment = read_number(stream, "I")
            else:
                skip_metadata(stream, type_id)
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("GGUF alignment must be a positive power of two")

        tensors = []
        for _ in range(tensor_count):
            name = read_string(stream)
            dimensions = [read_number(stream, "Q") for _ in range(read_number(stream, "I"))]
            type_id = read_number(stream, "I")
            offset = read_number(stream, "Q")
            tensors.append((name, dimensions, type_id, offset))
        data_offset = (stream.tell() + alignment - 1) // alignment * alignment

        for name, dimensions, type_id, offset in tensors:
            if len(dimensions) != 2 or type_id not in FORMATS:
                continue
            columns, rows = dimensions
            # MMQ reads complete 256-element K tiles. Synthetic cases below
            # cover formats whose model tensors only have narrower dimensions.
            if rows <= 0 or columns <= 0 or columns % 256 or (type_id, columns) in seen:
                continue
            seen.add((type_id, columns))
            _, block_elements, block_bytes = FORMATS[type_id]
            row_bytes = columns // block_elements * block_bytes
            packed = []
            for index in range(33):
                source_row = index * (rows - 1) // 32
                stream.seek(data_offset + offset + source_row * row_bytes)
                packed.append(read_exact(stream, row_bytes))
            cases.append(Case(type_id, 33, columns, b"".join(packed), name))
    return cases


def synthetic_case(type_id, rng):
    name, block_elements, block_bytes = FORMATS[type_id]
    rows, columns = 33, 768
    data = bytearray(rng.randbytes(rows * columns // block_elements * block_bytes))
    for offset in range(0, len(data), block_bytes):
        if name == "IQ1_M":
            # Its FP16 base scale is split across four scale-word high nibbles.
            base_scale = struct.unpack("<H", struct.pack("<e", 0.002))[0]
            for index in range(4):
                position = offset + block_bytes - 8 + 2 * index
                scale_word = struct.unpack_from("<H", data, position)[0]
                scale_word = (scale_word & 0x0FFF) | (((base_scale >> (4 * index)) & 15) << 12)
                struct.pack_into("<H", data, position, scale_word)
        else:
            scale_offset = 0
            if name in ("Q3_K", "Q6_K"):
                scale_offset = block_bytes - 2
            elif name == "Q2_K":
                scale_offset = block_bytes - 4
            struct.pack_into("<e", data, offset + scale_offset, 0.002)
            if name in ("Q4_1", "Q5_1", "Q2_K", "Q4_K", "Q5_K"):
                struct.pack_into("<e", data, offset + scale_offset + 2, 0.001)
    return Case(type_id, rows, columns, bytes(data), "synthetic valid blocks")


def complete_cases(cases):
    rng = random.Random(7729)
    for type_id in FORMATS:
        if not any(case.type_id == type_id for case in cases):
            cases.append(synthetic_case(type_id, rng))

    for type_id, (_, block_elements, block_bytes) in FORMATS.items():
        source = next(case for case in cases if case.type_id == type_id)
        row_bytes = len(source.data) // source.rows
        for columns in (256, 768):
            if any(case.type_id == type_id and case.columns == columns for case in cases):
                continue
            repeats = (columns + source.columns - 1) // source.columns
            packed = b"".join(
                (source.data[row * row_bytes : (row + 1) * row_bytes] * repeats)[
                    : columns // block_elements * block_bytes
                ]
                for row in range(source.rows)
            )
            cases.append(Case(type_id, source.rows, columns, packed, source.source + " blocks"))

    # Both 33 and 129 output rows leave a partial output tile; 129 also
    # crosses the 128-row tile boundary on the tested CUDA architecture.
    for type_id in FORMATS:
        source = next(case for case in cases if case.type_id == type_id and case.columns == 256)
        row_bytes = len(source.data) // source.rows
        packed = b"".join(
            source.data[(row % source.rows) * row_bytes : (row % source.rows + 1) * row_bytes]
            for row in range(129)
        )
        cases.append(Case(type_id, 129, 256, packed, source.source + " repeated output rows"))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cases = complete_cases(read_model_cases(args.model))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "weights.bin").open("wb") as stream:
        stream.write(struct.pack("<I", len(cases)))
        for case in cases:
            stream.write(struct.pack("<4I", case.type_id, case.rows, case.columns, len(case.data)))
            stream.write(case.data)
    metadata = [
        dict(type=case.type_id, name=FORMATS[case.type_id][0], rows=case.rows,
             cols=case.columns, source=case.source)
        for case in cases
    ]
    (args.output / "weights.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Fixtures: {len(cases)}, formats: {len(FORMATS)}")


if __name__ == "__main__":
    main()
