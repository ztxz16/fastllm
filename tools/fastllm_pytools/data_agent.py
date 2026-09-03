import json
import math
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DATA_EXTENSIONS = {".csv", ".tsv", ".json", ".jsonl", ".xlsx"}
ALLOWED_OPERATIONS = {"groupby", "trend", "correlation", "top"}
ALLOWED_AGGREGATIONS = {"sum", "mean", "median", "min", "max", "count"}


@dataclass
class Dataset:
    identifier: str
    name: str
    sheet: str
    path: str
    frame: Any


def is_dataset(path_or_name: str) -> bool:
    return Path(path_or_name).suffix.lower() in DATA_EXTENSIONS


def _json_object(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text,
                  flags=re.IGNORECASE)
    start = text.find("{")
    if start < 0:
        return {}
    depth = 0
    quoted = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                try:
                    result = json.loads(text[start:index + 1])
                    return result if isinstance(result, dict) else {}
                except json.JSONDecodeError:
                    return {}
    return {}


def _safe_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if bool(value != value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else None
    if hasattr(value, "item"):
        try:
            return _safe_value(value.item())
        except (TypeError, ValueError):
            pass
    text = str(value)
    return text if len(text) <= 1000 else text[:997] + "…"


def _records(frame: Any, limit: int = 30) -> List[Dict[str, Any]]:
    return [
        {str(key): _safe_value(value) for key, value in row.items()}
        for row in frame.head(limit).to_dict(orient="records")
    ]


def _unique_columns(columns: Iterable[Any]) -> List[str]:
    result: List[str] = []
    counts: Dict[str, int] = {}
    for raw in columns:
        name = (str(raw).strip() or "未命名列")[:120]
        count = counts.get(name, 0)
        counts[name] = count + 1
        result.append(name if count == 0 else f"{name}_{count + 1}")
    return result


class DataAgent:
    """Plans and executes a small, validated set of dataframe operations."""

    def __init__(self, max_rows: int = 200_000, max_columns: int = 256):
        self.max_rows = max(100, int(max_rows))
        self.max_columns = max(2, int(max_columns))

    def load(self, attachments: Iterable[Dict[str, Any]]) -> Tuple[List[Dataset], List[str]]:
        try:
            import pandas as pd
        except ImportError as error:
            raise RuntimeError(
                "数据分析功能需要 pandas，请安装 ftllm[webui]") from error

        datasets: List[Dataset] = []
        warnings: List[str] = []
        seen = set()
        for attachment in attachments:
            path = Path(str(attachment.get("path", ""))).resolve()
            if str(path) in seen or not is_dataset(path.name):
                continue
            seen.add(str(path))
            name = str(attachment.get("name", path.name))
            extension = path.suffix.lower()
            try:
                frames = self._read_frames(pd, path, extension)
            except Exception as error:
                warnings.append(f"{name}：{error}")
                continue
            for sheet, frame in frames:
                if frame.shape[1] > self.max_columns:
                    warnings.append(
                        f"{name}{' / ' + sheet if sheet else ''}：仅分析前 "
                        f"{self.max_columns} 列")
                    frame = frame.iloc[:, :self.max_columns]
                if len(frame) > self.max_rows:
                    warnings.append(
                        f"{name}{' / ' + sheet if sheet else ''}：仅分析前 "
                        f"{self.max_rows} 行")
                    frame = frame.iloc[:self.max_rows]
                frame = frame.copy()
                frame.columns = _unique_columns(frame.columns)
                frame = self._infer_numeric(pd, frame)
                if frame.empty and not len(frame.columns):
                    warnings.append(f"{name}：没有可分析的数据")
                    continue
                identifier = name if not sheet else f"{name} / {sheet}"
                datasets.append(Dataset(
                    identifier=identifier,
                    name=name,
                    sheet=sheet,
                    path=str(path),
                    frame=frame,
                ))
        return datasets, warnings

    def _read_frames(self, pd: Any, path: Path, extension: str):
        if extension in {".csv", ".tsv"}:
            separator = "\t" if extension == ".tsv" else ","
            last_error = None
            for encoding in ("utf-8-sig", "gb18030", "big5"):
                try:
                    frame = pd.read_csv(
                        path, sep=separator, encoding=encoding,
                        nrows=self.max_rows + 1, low_memory=False)
                    return [("", frame)]
                except UnicodeDecodeError as error:
                    last_error = error
            raise ValueError("无法识别文本编码") from last_error
        if extension in {".json", ".jsonl"}:
            lines = extension == ".jsonl"
            frame = pd.read_json(path, lines=lines)
            if isinstance(frame, pd.Series):
                frame = frame.to_frame()
            return [("", frame)]
        if extension == ".xlsx":
            try:
                book = pd.ExcelFile(path, engine="openpyxl")
            except ImportError as error:
                raise ValueError(
                    "读取 XLSX 需要 openpyxl，请安装 ftllm[webui]") from error
            frames = []
            with book:
                for sheet in book.sheet_names[:12]:
                    frame = pd.read_excel(
                        book, sheet_name=sheet, nrows=self.max_rows + 1)
                    if not frame.empty or len(frame.columns):
                        frames.append((str(sheet), frame))
            return frames
        raise ValueError(f"不支持的数据格式：{extension}")

    @staticmethod
    def _infer_numeric(pd: Any, frame: Any) -> Any:
        for column in frame.select_dtypes(include=["object"]).columns:
            values = frame[column]
            nonempty = values.dropna()
            if nonempty.empty:
                continue
            converted = pd.to_numeric(
                nonempty.astype(str).str.replace(",", "", regex=False),
                errors="coerce")
            if converted.notna().mean() >= 0.9:
                frame[column] = pd.to_numeric(
                    values.astype(str).str.replace(",", "", regex=False),
                    errors="coerce")
        return frame

    def profile(self, datasets: Sequence[Dataset]) -> List[Dict[str, Any]]:
        profiles = []
        for dataset in datasets:
            frame = dataset.frame
            numeric = list(frame.select_dtypes(include="number").columns)
            columns = []
            for column in frame.columns:
                series = frame[column]
                item = {
                    "name": str(column),
                    "dtype": str(series.dtype),
                    "missing": int(series.isna().sum()),
                    "unique": int(series.nunique(dropna=True)),
                }
                if column in numeric and series.notna().any():
                    item.update({
                        "min": _safe_value(series.min()),
                        "max": _safe_value(series.max()),
                        "mean": _safe_value(series.mean()),
                        "median": _safe_value(series.median()),
                    })
                else:
                    item["top_values"] = {
                        str(key)[:300]: int(value) for key, value in
                        series.astype("string").value_counts(dropna=True).head(5).items()
                    }
                columns.append(item)
            profiles.append({
                "dataset": dataset.identifier,
                "file": dataset.name,
                "sheet": dataset.sheet,
                "rows": int(len(frame)),
                "columns": columns,
                "sample": _records(frame, 5),
            })
        return profiles

    def planning_messages(
        self, question: str, profiles: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        schema = {
            "title": "分析标题",
            "analyses": [{
                "operation": "groupby|trend|correlation|top",
                "dataset": "必须原样复制数据集名称",
                "group_by": "groupby 分组列",
                "value": "groupby/trend 数值列",
                "aggregation": "sum|mean|median|min|max|count",
                "x": "trend 横轴列",
                "columns": ["correlation 数值列"],
                "sort_by": "top 排序列",
                "ascending": False,
                "limit": 10,
                "chart": "bar|line|none",
            }],
        }
        planning_profiles = [{
            **{key: value for key, value in profile.items()
               if key not in {"columns", "sample"}},
            "columns": profile["columns"][:80],
            "sample": profile["sample"][:3],
        } for profile in profiles[:12]]
        compact_profiles = json.dumps(
            planning_profiles, ensure_ascii=False, separators=(",", ":"))
        return [{
            "role": "system",
            "content": (
                "你是数据分析计划器。只输出一个 JSON 对象，不得输出代码、解释或 Markdown。"
                "只能选择给定 schema 中的操作；列名和数据集名必须原样复制。最多选择 4 个"
                "能直接回答问题的分析，禁止构造过滤表达式或任意 Python。"),
        }, {
            "role": "user",
            "content": (
                f"用户问题：{question}\nJSON schema："
                f"{json.dumps(schema, ensure_ascii=False)}\n数据概览：{compact_profiles}"),
        }]

    def normalize_plan(
        self, raw_plan: Any, question: str, datasets: Sequence[Dataset],
    ) -> Dict[str, Any]:
        parsed = _json_object(raw_plan)
        title = str(parsed.get("title") or question or "数据分析")[:80]
        incoming = parsed.get("analyses")
        incoming = incoming if isinstance(incoming, list) else []
        normalized = []
        for item in incoming[:8]:
            operation = self._normalize_operation(item, datasets)
            if operation is not None:
                normalized.append(operation)
            if len(normalized) >= 4:
                break
        if not normalized:
            normalized = self._fallback_operations(datasets)
        return {"title": title, "analyses": normalized}

    def _normalize_operation(
        self, item: Any, datasets: Sequence[Dataset],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None
        operation = str(item.get("operation", "")).lower()
        if operation not in ALLOWED_OPERATIONS:
            return None
        dataset = self._dataset(datasets, str(item.get("dataset", "")))
        if dataset is None:
            return None
        result: Dict[str, Any] = {
            "operation": operation,
            "dataset": dataset.identifier,
            "limit": min(50, max(3, int(item.get("limit", 10) or 10))),
        }
        columns = list(dataset.frame.columns)
        if operation == "groupby":
            group_by = self._column(columns, item.get("group_by"))
            aggregation = str(item.get("aggregation", "sum")).lower()
            value = self._column(columns, item.get("value"))
            if group_by is None or aggregation not in ALLOWED_AGGREGATIONS:
                return None
            if aggregation != "count" and value is None:
                return None
            result.update(group_by=group_by, value=value,
                          aggregation=aggregation,
                          chart="bar" if item.get("chart") != "none" else "none")
        elif operation == "trend":
            x = self._column(columns, item.get("x"))
            value = self._column(columns, item.get("value"))
            aggregation = str(item.get("aggregation", "sum")).lower()
            if x is None or value is None or aggregation not in ALLOWED_AGGREGATIONS:
                return None
            result.update(x=x, value=value, aggregation=aggregation,
                          chart="line" if item.get("chart") != "none" else "none")
        elif operation == "correlation":
            requested = item.get("columns")
            requested = requested if isinstance(requested, list) else []
            selected = [column for value in requested
                        if (column := self._column(columns, value)) is not None]
            result["columns"] = list(dict.fromkeys(selected))[:12]
            result["chart"] = "none"
        else:
            sort_by = self._column(columns, item.get("sort_by"))
            if sort_by is None:
                return None
            result.update(sort_by=sort_by,
                          ascending=bool(item.get("ascending", False)),
                          chart="none")
        return result

    @staticmethod
    def _dataset(
        datasets: Sequence[Dataset], name: str,
    ) -> Optional[Dataset]:
        for dataset in datasets:
            if dataset.identifier == name:
                return dataset
        lowered = name.casefold()
        matches = [dataset for dataset in datasets
                   if dataset.identifier.casefold() == lowered]
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _column(columns: Sequence[Any], name: Any) -> Optional[str]:
        text = str(name or "")
        if text in columns:
            return text
        matches = [str(column) for column in columns
                   if str(column).casefold() == text.casefold()]
        return matches[0] if len(matches) == 1 else None

    def _fallback_operations(self, datasets: Sequence[Dataset]) -> List[Dict[str, Any]]:
        operations = []
        for dataset in datasets[:2]:
            frame = dataset.frame
            numeric = [str(value) for value in
                       frame.select_dtypes(include="number").columns]
            categorical = [str(value) for value in frame.columns
                           if str(value) not in numeric]
            if categorical and numeric:
                operations.append({
                    "operation": "groupby", "dataset": dataset.identifier,
                    "group_by": categorical[0], "value": numeric[0],
                    "aggregation": "sum", "limit": 10, "chart": "bar",
                })
            if len(numeric) >= 2:
                operations.append({
                    "operation": "correlation", "dataset": dataset.identifier,
                    "columns": numeric[:8], "limit": 10, "chart": "none",
                })
        return operations[:4]

    def execute(
        self,
        plan: Dict[str, Any],
        datasets: Sequence[Dataset],
        output_directory: Path,
    ) -> Dict[str, Any]:
        output_directory.mkdir(parents=True, exist_ok=True)
        profiles = self.profile(datasets)
        results = []
        charts = []
        for index, operation in enumerate(plan.get("analyses", []), 1):
            dataset = self._dataset(datasets, operation["dataset"])
            if dataset is None:
                continue
            try:
                result = self._execute_operation(dataset, operation)
            except (KeyError, TypeError, ValueError) as error:
                result = {
                    "title": f"分析 {index}", "dataset": dataset.identifier,
                    "operation": operation["operation"], "columns": [],
                    "rows": [], "warning": str(error),
                }
            results.append(result)
            if operation.get("chart") in {"bar", "line"} and result["rows"]:
                path = output_directory / f"{uuid.uuid4().hex}.png"
                self._render_chart(result, path, operation["chart"])
                charts.append({
                    "kind": "chart",
                    "name": f"{self._safe_filename(result['title'])}.png",
                    "path": str(path), "size": path.stat().st_size,
                    "title": result["title"],
                })
        report_path = output_directory / f"{uuid.uuid4().hex}.xlsx"
        self._write_report(plan, datasets, profiles, results, charts, report_path)
        return {
            "title": plan["title"],
            "profiles": profiles,
            "results": results,
            "artifacts": [{
                "kind": "analysis_report",
                "name": f"{self._safe_filename(plan['title'])}.xlsx",
                "path": str(report_path),
                "size": report_path.stat().st_size,
                "datasets": len(datasets),
                "analyses": len(results),
            }] + charts,
        }

    def _execute_operation(self, dataset: Dataset, operation: Dict[str, Any]):
        import pandas as pd

        frame = dataset.frame
        kind = operation["operation"]
        if kind == "groupby":
            group = operation["group_by"]
            value = operation.get("value")
            aggregation = operation["aggregation"]
            if aggregation == "count":
                series = frame.groupby(group, dropna=False).size()
                value_name = "count"
            else:
                numeric = pd.to_numeric(frame[value], errors="coerce")
                series = frame.assign(__value=numeric).groupby(
                    group, dropna=False)["__value"].agg(aggregation)
                value_name = f"{aggregation}_{value}"
            output = series.sort_values(ascending=False).head(
                operation["limit"]).reset_index(name=value_name)
            title = f"按 {group} 汇总 {value or '记录数'}（{aggregation}）"
        elif kind == "trend":
            x, value = operation["x"], operation["value"]
            numeric = pd.to_numeric(frame[value], errors="coerce")
            working = frame.assign(__value=numeric).dropna(subset=[x, "__value"])
            output = working.groupby(x, dropna=False)["__value"].agg(
                operation["aggregation"]).reset_index(name=value)
            try:
                output = output.sort_values(x)
            except TypeError:
                output[x] = output[x].astype(str)
                output = output.sort_values(x)
            output = output.tail(min(80, operation["limit"] * 4))
            title = f"{value} 随 {x} 的变化"
        elif kind == "correlation":
            columns = operation.get("columns") or list(
                frame.select_dtypes(include="number").columns)[:12]
            if len(columns) < 2:
                raise ValueError("相关性分析至少需要两个数值列")
            numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
            output = numeric.corr().reset_index(names="column")
            title = "数值列相关性"
        else:
            sort_by = operation["sort_by"]
            output = frame.sort_values(
                sort_by, ascending=operation["ascending"], na_position="last"
            ).head(operation["limit"])
            title = f"按 {sort_by} 排序的前 {len(output)} 行"
        return {
            "title": title,
            "dataset": dataset.identifier,
            "operation": kind,
            "columns": [str(column) for column in output.columns],
            "rows": _records(output, 80),
        }

    def result_context(self, report: Dict[str, Any]) -> str:
        row_limit = 30
        while True:
            compact_results = []
            for result in report["results"]:
                columns = result["columns"][:40]
                compact_results.append({
                    **{key: value for key, value in result.items()
                       if key not in {"columns", "rows"}},
                    "columns": columns,
                    "rows": [{column: row.get(column) for column in columns}
                             for row in result["rows"][:row_limit]],
                })
            compact = {
                "title": report["title"],
                "data_profiles": [{
                    "dataset": profile["dataset"],
                    "rows": profile["rows"],
                    "columns": profile["columns"][:80],
                } for profile in report["profiles"][:12]],
                "executed_analyses": compact_results,
            }
            text = json.dumps(
                compact, ensure_ascii=False, separators=(",", ":"))
            if len(text) <= 30000 or row_limit <= 2:
                return text
            row_limit = max(2, row_limit // 2)

    def _write_report(
        self, plan, datasets, profiles, results, charts, path: Path,
    ) -> None:
        try:
            import pandas as pd
            __import__("xlsxwriter")  # Validate the optional writer dependency.
        except ImportError as error:
            raise RuntimeError(
                "导出 Excel 报告需要 pandas 和 XlsxWriter") from error

        used_names = set()
        with pd.ExcelWriter(
            path,
            engine="xlsxwriter",
            engine_kwargs={"options": {
                "strings_to_formulas": False,
                "strings_to_urls": False,
            }},
        ) as writer:
            workbook = writer.book
            title_format = workbook.add_format({
                "bold": True, "font_size": 16, "font_color": "#4056D9"})
            header_format = workbook.add_format({
                "bold": True, "font_color": "#FFFFFF", "bg_color": "#5267E8",
                "border": 0, "align": "left"})
            overview = workbook.add_worksheet("分析概览")
            writer.sheets["分析概览"] = overview
            overview.write("A1", plan["title"], title_format)
            overview.write_row("A3", ["数据集", "行数", "列数", "缺失单元格"],
                               header_format)
            for row, profile in enumerate(profiles, 3):
                missing = sum(column["missing"] for column in profile["columns"])
                overview.write_row(row, 0, [
                    profile["dataset"], profile["rows"],
                    len(profile["columns"]), missing])
            overview.set_column("A:A", 38)
            overview.set_column("B:D", 15)

            for index, result in enumerate(results, 1):
                sheet_name = self._sheet_name(f"分析{index}", used_names)
                result_frame = pd.DataFrame(result["rows"], columns=result["columns"])
                result_frame.to_excel(writer, sheet_name=sheet_name, index=False,
                                      startrow=2)
                sheet = writer.sheets[sheet_name]
                sheet.write("A1", result["title"], title_format)
                if result["columns"]:
                    sheet.set_row(2, None, header_format)
                sheet.freeze_panes(3, 0)
                sheet.set_column(0, max(0, len(result["columns"]) - 1), 18)

            for dataset in datasets:
                sheet_name = self._sheet_name(
                    f"数据-{dataset.identifier}", used_names)
                dataset.frame.head(5000).to_excel(
                    writer, sheet_name=sheet_name, index=False)
                sheet = writer.sheets[sheet_name]
                sheet.set_row(0, None, header_format)
                sheet.freeze_panes(1, 0)
                sheet.autofilter(0, 0, min(len(dataset.frame), 5000),
                                 max(0, len(dataset.frame.columns) - 1))
                sheet.set_column(0, max(0, len(dataset.frame.columns) - 1), 16)

            if charts:
                chart_sheet = workbook.add_worksheet("图表")
                writer.sheets["图表"] = chart_sheet
                row = 0
                for chart in charts:
                    chart_sheet.insert_image(row, 0, chart["path"], {
                        "x_scale": 0.62, "y_scale": 0.62})
                    row += 23

    @staticmethod
    def _sheet_name(raw: str, used: set) -> str:
        base = re.sub(r"[\\/*?:\[\]]", "-", raw).strip()[:31] or "Sheet"
        name = base
        counter = 2
        while name in used:
            suffix = f"-{counter}"
            name = base[:31 - len(suffix)] + suffix
            counter += 1
        used.add(name)
        return name

    @staticmethod
    def _safe_filename(raw: str) -> str:
        name = re.sub(r"[\\/:*?\"<>|]", "-", str(raw)).strip(" .")[:70]
        return name or "数据分析报告"

    def _render_chart(self, result: Dict[str, Any], path: Path, kind: str) -> None:
        from PIL import Image, ImageDraw, ImageFont

        rows = result["rows"][:20]
        columns = result["columns"]
        if len(columns) < 2:
            return
        labels = [str(row.get(columns[0], "")) for row in rows]
        values = []
        for row in rows:
            try:
                values.append(float(row.get(columns[-1], 0) or 0))
            except (TypeError, ValueError):
                values.append(0.0)
        image = Image.new("RGB", (1200, 675), "#F8F9FD")
        draw = ImageDraw.Draw(image)
        font = self._font(ImageFont, 25)
        small = self._font(ImageFont, 17)
        draw.rounded_rectangle((34, 30, 1166, 645), 24, fill="#FFFFFF",
                               outline="#E4E7F1", width=2)
        draw.text((70, 60), result["title"], fill="#242936", font=font)
        left, top, right, bottom = 110, 135, 1120, 565
        draw.line((left, bottom, right, bottom), fill="#CAD0DF", width=2)
        minimum, maximum = min(values or [0]), max(values or [1])
        low = min(0.0, minimum)
        span = max(1e-12, maximum - low)
        if kind == "line":
            points = []
            for index, value in enumerate(values):
                x = left + index * (right - left) / max(1, len(values) - 1)
                y = bottom - (value - low) / span * (bottom - top)
                points.append((x, y))
            if len(points) > 1:
                draw.line(points, fill="#5369E8", width=5, joint="curve")
            for x, y in points:
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="#7658DF")
        else:
            slot = (right - left) / max(1, len(values))
            for index, value in enumerate(values):
                x0 = left + index * slot + slot * 0.15
                x1 = left + (index + 1) * slot - slot * 0.15
                y = bottom - (value - low) / span * (bottom - top)
                draw.rounded_rectangle((x0, y, x1, bottom), 6,
                                       fill="#6074EC")
        label_step = max(1, math.ceil(len(labels) / 8))
        for index, label in enumerate(labels):
            if index % label_step:
                continue
            x = left + (index + 0.5) * (right - left) / max(1, len(labels))
            text = label[:12]
            box = draw.textbbox((0, 0), text, font=small)
            draw.text((x - (box[2] - box[0]) / 2, bottom + 17), text,
                      fill="#707789", font=small)
        draw.text((70, 610), f"数据集：{result['dataset']}",
                  fill="#9298A6", font=small)
        image.save(path, "PNG", optimize=True)

    @staticmethod
    def _font(image_font: Any, size: int):
        candidates = (
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        )
        for candidate in candidates:
            if Path(candidate).is_file():
                try:
                    return image_font.truetype(candidate, size)
                except OSError:
                    pass
        return image_font.load_default()
