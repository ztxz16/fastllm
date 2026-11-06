#!/usr/bin/env python3
"""生成可复用的Batch性能趋势PNG及Markdown引用。"""

from dataclasses import dataclass
import math
from pathlib import Path
import re
import struct
import zlib


BACKEND_COLORS = {
    "native": (68, 114, 196),
    "machete": (237, 125, 49),
    "fastllm": (68, 114, 196),
    "vllm": (237, 125, 49),
    "ratio": (112, 48, 160),
    "continuous-auto": (68, 114, 196),
    "paged-native_paged": (112, 173, 71),
    "paged-flashinfer_paged": (112, 48, 160),
}
GRID_COLOR = (218, 223, 230)
TEXT_COLOR = (45, 52, 64)


@dataclass(frozen=True)
class ChartSpec:
    """描述一个随Batch变化的后端对比子图。"""

    title: str
    field: str
    unit: str
    reference: float = None


def model_performance_chart_specs(workload):
    """返回与模型性能工作负载语义一致的趋势图指标。"""
    if workload == "prefill":
        return (
            ChartSpec("TTFT", "ttft_ms", "ms"),
            ChartSpec("PREFILL THROUGHPUT", "prefill_tok_s", "tok/s"),
            ChartSpec("E2EL", "e2el_ms", "ms"),
        )
    if workload == "decode":
        return (
            ChartSpec("TPOT", "tpot_ms", "ms"),
            ChartSpec("ITL", "itl_ms", "ms"),
            ChartSpec("OUTPUT THROUGHPUT", "output_tok_s", "tok/s"),
            ChartSpec("E2EL", "e2el_ms", "ms"),
        )
    raise ValueError(f"未知模型性能工作负载：{workload}")


# 报告标题只使用ASCII，内置5x7字模即可保证无第三方字体时仍能输出PNG。
_FONT = {
    " ": ("00000",) * 7,
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    "/": ("00001", "00010", "00100", "01000", "10000", "00000", "00000"),
    "(": ("00010", "00100", "01000", "01000", "01000", "00100", "00010"),
    ")": ("01000", "00100", "00010", "00010", "00010", "00100", "01000"),
    "%": ("11001", "11010", "00100", "01000", "10110", "00110", "00000"),
    ":": ("00000", "01100", "01100", "00000", "01100", "01100", "00000"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "J": ("00111", "00010", "00010", "00010", "10010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
}


class _Canvas:
    """提供报告折线图需要的最小RGB画布。"""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.pixels = bytearray((255, 255, 255)) * (width * height)

    def pixel(self, x, y, color):
        if 0 <= x < self.width and 0 <= y < self.height:
            offset = (y * self.width + x) * 3
            self.pixels[offset:offset + 3] = bytes(color)

    def line(self, x0, y0, x1, y1, color, width=1):
        dx, sx = abs(x1 - x0), 1 if x0 < x1 else -1
        dy, sy = -abs(y1 - y0), 1 if y0 < y1 else -1
        error = dx + dy
        while True:
            radius = max(0, width // 2)
            for py in range(y0 - radius, y0 + radius + 1):
                for px in range(x0 - radius, x0 + radius + 1):
                    self.pixel(px, py, color)
            if x0 == x1 and y0 == y1:
                break
            twice = 2 * error
            if twice >= dy:
                error += dy
                x0 += sx
            if twice <= dx:
                error += dx
                y0 += sy

    def circle(self, cx, cy, radius, color):
        for y in range(cy - radius, cy + radius + 1):
            for x in range(cx - radius, cx + radius + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    self.pixel(x, y, color)

    def text(self, x, y, value, color=TEXT_COLOR, scale=2):
        cursor = x
        for character in str(value).upper():
            glyph = _FONT.get(character, _FONT[" "])
            for row, pattern in enumerate(glyph):
                for column, enabled in enumerate(pattern):
                    if enabled == "1":
                        for py in range(scale):
                            for px in range(scale):
                                self.pixel(cursor + column * scale + px,
                                           y + row * scale + py, color)
            cursor += 6 * scale

    def save_png(self, path):
        raw = bytearray()
        stride = self.width * 3
        for row in range(self.height):
            raw.append(0)
            raw.extend(self.pixels[row * stride:(row + 1) * stride])

        def chunk(name, payload):
            return (struct.pack(">I", len(payload)) + name + payload +
                    struct.pack(">I", zlib.crc32(name + payload) & 0xffffffff))

        data = (b"\x89PNG\r\n\x1a\n" +
                chunk(b"IHDR", struct.pack(">IIBBBBB", self.width, self.height,
                                             8, 2, 0, 0, 0)) +
                chunk(b"IDAT", zlib.compress(bytes(raw), 9)) +
                chunk(b"IEND", b""))
        Path(path).write_bytes(data)


def _safe_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _format_tick(value):
    if abs(value) >= 1000:
        return f"{value / 1000:.1f}K"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _draw_panel(canvas, bounds, rows, spec, batch_field, backend_field):
    left, top, right, bottom = bounds
    plot_left, plot_top = left + 72, top + 42
    plot_right, plot_bottom = right - 24, bottom - 54
    canvas.text(left + 12, top + 10, f"{spec.title} ({spec.unit})", scale=2)

    batches = sorted({int(row[batch_field]) for row in rows
                      if row.get(batch_field) is not None})
    values = [_safe_float(row.get(spec.field)) for row in rows]
    values = [value for value in values if value is not None]
    reference = _safe_float(spec.reference)
    if reference is not None:
        values.append(reference)
    if not batches or not values:
        canvas.text(plot_left + 20, plot_top + 50, "NO DATA", scale=3)
        return

    upper = max(values)
    upper = upper * 1.12 if upper > 0 else 1.0
    canvas.line(plot_left, plot_top, plot_left, plot_bottom, TEXT_COLOR, 2)
    canvas.line(plot_left, plot_bottom, plot_right, plot_bottom, TEXT_COLOR, 2)
    for index in range(5):
        y = plot_bottom - round((plot_bottom - plot_top) * index / 4)
        value = upper * index / 4
        canvas.line(plot_left, y, plot_right, y, GRID_COLOR)
        canvas.text(left + 6, y - 7, _format_tick(value), scale=1)
    if reference is not None:
        reference_y = plot_bottom - round(
            (plot_bottom - plot_top) * reference / upper)
        canvas.line(plot_left, reference_y, plot_right, reference_y,
                    (112, 173, 71), 2)
        canvas.text(plot_right - 42, reference_y - 14,
                    _format_tick(reference), (112, 173, 71), 1)

    if len(batches) == 1:
        x_positions = {batches[0]: (plot_left + plot_right) // 2}
    else:
        x_positions = {
            batch: plot_left + round((plot_right - plot_left) * index /
                                     (len(batches) - 1))
            for index, batch in enumerate(batches)
        }
    for batch, x in x_positions.items():
        canvas.line(x, plot_bottom, x, plot_bottom + 5, TEXT_COLOR)
        label = str(batch)
        canvas.text(x - len(label) * 3, plot_bottom + 10, label, scale=1)
    canvas.text((plot_left + plot_right) // 2 - 15, bottom - 18, "BATCH", scale=1)

    present_backends = {
        str(row.get(backend_field, "")).lower() for row in rows
    }
    ordered_backends = [backend for backend in BACKEND_COLORS
                        if backend in present_backends]
    ordered_backends.extend(sorted(
        present_backends - set(ordered_backends)))
    for backend in ordered_backends:
        color = BACKEND_COLORS.get(backend, (128, 128, 128))
        points = []
        for row in rows:
            if str(row.get(backend_field, "")).lower() != backend:
                continue
            value = _safe_float(row.get(spec.field))
            batch = row.get(batch_field)
            if value is None or batch is None or int(batch) not in x_positions:
                continue
            x = x_positions[int(batch)]
            y = plot_bottom - round((plot_bottom - plot_top) * value / upper)
            points.append((x, y))
        points.sort()
        for first, second in zip(points, points[1:]):
            canvas.line(*first, *second, color, 3)
        for x, y in points:
            canvas.circle(x, y, 5, color)


def slugify(value):
    """把报告维度转换成稳定文件名。"""
    return re.sub(r"[^a-z0-9_-]+", "-", str(value).lower()).strip("-") or "all"


def write_dashboard(path, rows, specs, title, batch_field="batch",
                    backend_field="backend"):
    """把最多四个Batch趋势子图写入一张PNG。"""
    if not 1 <= len(specs) <= 4:
        raise ValueError("一张趋势图必须包含1到4个子图")
    canvas = _Canvas(1280, 820)
    canvas.text(36, 20, title, scale=3)
    present_backends = {
        str(row.get(backend_field, "")).lower() for row in rows
    }
    legend_backends = [backend for backend in BACKEND_COLORS
                       if backend in present_backends]
    compact_legend = len(legend_backends) > 2
    legend_x = 530 if compact_legend else 900
    legend_scale = 1 if compact_legend else 2
    for backend in legend_backends:
        color = BACKEND_COLORS[backend]
        label = ("FASTLLM/VLLM" if backend == "ratio" else
                 backend.upper().replace("_", "-"))
        canvas.line(legend_x, 31, legend_x + 50, 31, color, 4)
        canvas.text(legend_x + 60, 26 if compact_legend else 21,
                    label, scale=legend_scale)
        legend_x += 250 if compact_legend else 210

    bounds = (
        (20, 65, 640, 440), (640, 65, 1260, 440),
        (20, 440, 640, 810), (640, 440, 1260, 810),
    )
    for panel, spec in zip(bounds, specs):
        _draw_panel(canvas, panel, rows, spec, batch_field, backend_field)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save_png(path)
    return Path(path)


def markdown_images(paths, heading="Batch趋势图"):
    """生成使用相对路径引用趋势图片的Markdown段落。"""
    lines = ["", f"## {heading}", ""]
    for path in paths:
        path = Path(path)
        lines.extend((f"### {path.stem}", "", f"![{path.stem}](images/{path.name})", ""))
    return lines
