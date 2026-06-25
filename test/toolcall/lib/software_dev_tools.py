import copy
from typing import Dict, List, Optional


def _function_tool(
    name: str,
    description: str,
    properties: Dict,
    required: Optional[List[str]] = None,
) -> Dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": copy.deepcopy(properties),
                "required": list(required or []),
            },
        },
    }


def read_file_tool() -> Dict:
    return _function_tool(
        "read_file",
        "Read a file or a line range from the workspace.",
        {
            "path": {"type": "string"},
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"},
        },
        ["path"],
    )


def search_code_tool() -> Dict:
    return _function_tool(
        "search_code",
        "Search source code by query and optional path or file pattern.",
        {
            "query": {"type": "string"},
            "path": {"type": "string"},
            "file_pattern": {"type": "string"},
        },
        ["query"],
    )


def run_tests_tool() -> Dict:
    return _function_tool(
        "run_tests",
        "Run a test command in a project directory.",
        {
            "command": {
                "type": "array",
                "items": {"type": "string"},
            },
            "cwd": {"type": "string"},
            "timeout_seconds": {"type": "integer"},
        },
        ["command"],
    )


def apply_patch_tool() -> Dict:
    return _function_tool(
        "apply_patch",
        "Apply a source patch to a file.",
        {
            "path": {"type": "string"},
            "patch": {"type": "string"},
        },
        ["path", "patch"],
    )


def git_status_tool() -> Dict:
    return _function_tool(
        "git_status",
        "Return repository status.",
        {},
        [],
    )


def git_diff_tool() -> Dict:
    return _function_tool(
        "git_diff",
        "Return git diff, optionally scoped to a path.",
        {"path": {"type": "string"}},
        [],
    )


def list_files_tool() -> Dict:
    return _function_tool(
        "list_files",
        "List files under a path.",
        {
            "path": {"type": "string"},
            "recursive": {"type": "boolean"},
        },
        ["path"],
    )


def software_dev_tools() -> List[Dict]:
    return [
        read_file_tool(),
        search_code_tool(),
        run_tests_tool(),
        apply_patch_tool(),
        git_status_tool(),
        git_diff_tool(),
        list_files_tool(),
    ]


def software_dev_tool_names() -> List[str]:
    return [tool["function"]["name"] for tool in software_dev_tools()]
