#!/usr/bin/env python3
import os
import re
import json
import argparse
import subprocess
import sys
from pathlib import Path
from fnmatch import translate
from typing import List, Tuple
import requests
from .util import get_fastllm_cache_path
from .util import make_download_parser

def find_metadata(repo_id) -> bool:
    hf_endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    url = f"{hf_endpoint}/api/models/{repo_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        metadata = response.json()
        return True
    except requests.HTTPError as e:
        return False

def search_model(repo_id):
    hf_endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    url = f"{hf_endpoint}/api/models?search={repo_id}&limit=3"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as e:
        return []
# ANSI颜色代码
COLORS = {
    "RED": "\033[0;31m",
    "GREEN": "\033[0;32m",
    "YELLOW": "\033[1;33m",
    "NC": "\033[0m"
}

def color_print(text: str, color: str) -> None:
    print(f"{COLORS[color]}{text}{COLORS['NC']}", flush=True)

class HFDArgs:
    def __init__(self):
        self.repo_id = ""
        self.include = []
        self.exclude = []
        self.hf_username = ""
        self.hf_token = ""
        self.tool = "aria2c"
        self.x = 4
        self.j = 5
        self.dataset = False
        self.local_dir = ""
        self.revision = "main"

class HFDDownloader:
    def __init__(self, args):
        self.args = args
        self.validate_arguments()
        self.setup_paths()
        self.hf_endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
        
        # 确保缓存目录存在
        (self.local_dir / ".hfd").mkdir(parents=True, exist_ok=True)
    
    def validate_arguments(self):
        for attr in ["x", "j"]:
            value = getattr(self.args, attr)
            if not (1 <= value <= 10):
                color_print(f"{attr} must be between 1-10", "RED")
                sys.exit(1)
        
    def setup_paths(self):
        # 处理本地目录路径
        if self.args.local_dir:
            self.local_dir = Path(self.args.local_dir).expanduser().resolve()
        else:
            repo_name = self.args.repo_id.split("/")[-1] if "/" in self.args.repo_id else self.args.repo_id
            self.local_dir = Path.cwd() / repo_name
        
        self.metadata_file = self.local_dir / ".hfd" / "repo_metadata.json"
        self.command_file = self.local_dir / ".hfd" / "last_command"
        self.fileslist_file = self.local_dir / ".hfd" / f"{self.args.tool}_urls.txt"

    def run(self):
        try:
            finish_flag = os.path.join(self.local_dir, "FASTLLM_FINISH_FLAG")
            if (os.path.exists(finish_flag)):
                return
            # 获取并处理元数据
            metadata = self.fetch_metadata()
            self.check_authentication(metadata)
            
            # 生成下载列表
            if self.should_regenerate_filelist():
                self.generate_filelist(metadata)
            
            # 执行下载
            self.execute_download()
            
            color_print("Download completed successfully.", "GREEN")
            with open(finish_flag, 'w') as file:
                file.write('finish')
        except KeyboardInterrupt:
            color_print("\nDownload interrupted. You can resume by re-running the command.", "YELLOW")
            sys.exit(1)
        except Exception as e:
            color_print(f"Error: {str(e)}", "RED")
            sys.exit(1)

    def fetch_metadata(self) -> dict:
        color_print("Fetching repository metadata...", "YELLOW")
        
        api_path = "datasets" if self.args.dataset else "models"
        url = f"{self.hf_endpoint}/api/{api_path}/{self.args.repo_id}"
        if self.args.revision != "main":
            url += f"/revision/{self.args.revision}"
            
        headers = {"Authorization": f"Bearer {self.args.hf_token}"} if self.args.hf_token else {}
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            metadata = response.json()
            
            # 保存元数据
            with open(self.metadata_file, "w") as f:
                json.dump(metadata, f)
                
            return metadata
        except requests.HTTPError as e:
            color_print(f"Failed to fetch metadata: {e}", "RED")
            sys.exit(1)

    def check_authentication(self, metadata: dict):
        if metadata.get("gated", False) and not (self.args.hf_username and self.args.hf_token):
            color_print("This repository requires authentication. Please provide --hf_username and --hf_token.", "RED")
            sys.exit(1)

    def should_regenerate_filelist(self) -> bool:
        current_command = self.get_current_command()
        
        try:
            with open(self.command_file) as f:
                previous_command = f.read()
            if current_command == previous_command and self.fileslist_file.exists():
                color_print("Using cached file list.", "GREEN")
                return False
        except FileNotFoundError:
            pass
        
        # 保存当前命令
        with open(self.command_file, "w") as f:
            f.write(current_command)
        return True
    
    def get_current_command(self) -> str:
        return " ".join(sys.argv[1:])

    def generate_filelist(self, metadata: dict):
        color_print("Generating download file list...", "YELLOW")
        
        include_regex = self.patterns_to_regex(self.args.include)
        exclude_regex = self.patterns_to_regex(self.args.exclude)
        
        files = []
        for item in metadata.get("siblings", []):
            rfilename = item.get("rfilename")
            if not rfilename:
                continue
            
            # 过滤文件
            if include_regex and not re.search(include_regex, rfilename):
                continue
            if exclude_regex and re.search(exclude_regex, rfilename):
                continue
            
            files.append(rfilename)
        
        # 生成下载列表
        with open(self.fileslist_file, "w") as f:
            for file in files:
                url = f"{self.hf_endpoint}/{self.args.repo_id}/resolve/{self.args.revision}/{file}"
                if self.args.tool == "aria2c":
                    lines = [
                        url,
                        f" dir={os.path.dirname(file)}",
                        f" out={os.path.basename(file)}"
                    ]
                    if self.args.hf_token:
                        lines.append(f" header=Authorization: Bearer {self.args.hf_token}")
                    f.write("\n".join(lines) + "\n\n")
                else:
                    f.write(url + "\n")

    @staticmethod
    def patterns_to_regex(patterns: List[str]) -> str:
        if not patterns:
            return ""
        regexes = [translate(p).replace(r"\Z(?ms)", r"$") for p in patterns]
        return "|".join(regexes)

    def execute_download(self):
        color_print(f"Starting download with {self.args.tool}...", "YELLOW")
        
        os.chdir(self.local_dir)
        if self.args.tool == "aria2c":
            cmd = [
                "aria2c",
                "--check-certificate=false",
                "--console-log-level=error",
                "--file-allocation=none",
                "-x", str(self.args.x),
                "-j", str(self.args.j),
                "-s", str(self.args.x),
                "-k", "1M",
                "-c",
                "-i", str(self.fileslist_file),
                "--save-session", str(self.fileslist_file)
            ]
        else:
            cmd = [
                "wget",
                "-x",
                "-nH",
                f"--cut-dirs={5 if self.args.dataset else 4}",
                "--continue",
                "--input-file", str(self.fileslist_file)
            ]
            if self.args.hf_token:
                cmd.append(f"--header=Authorization: Bearer {self.args.hf_token}")
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            color_print(f"Download failed: {e}", "RED")
            sys.exit(1)

class HFDNormalDownloader(HFDDownloader):
    def __init__(self,
                 repo_id: str,
                 include = [],
                 exclude = [],
                 hf_username = "",
                 hf_token = "",
                 tool = "aria2c",
                 x = 4,
                 j = 5,
                 dataset = False,
                 local_dir = "",
                 revision = "main"):
        self.args = HFDArgs()
        self.args.include = include
        self.args.exclude = exclude
        self.args.hf_username = hf_username
        self.args.hf_token = hf_token
        self.args.tool = tool
        self.args.x = x
        self.args.j = j
        self.args.dataset = dataset
        self.args.repo_id = repo_id
        self.args.revision = revision
        self.args.local_dir = local_dir
        print("Model dir:", self.args.local_dir)
        self.validate_arguments()
        self.setup_paths()
        self.hf_endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
        # 确保缓存目录存在
        (self.local_dir / ".hfd").mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    try:
        downloader = HFDDownloader()
        #model_name = "Qwen/Qwen2-1.5B-Instruct"
        #downloader = HFDNormalDownloader(model_name, local_dir = get_fastllm_cache_path(model_name))
        downloader.run()
    except KeyboardInterrupt:
        color_print("\nDownload interrupted.", "YELLOW")
        sys.exit(1)