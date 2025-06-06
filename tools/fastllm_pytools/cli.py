import argparse
from .util import make_normal_parser
import readline
import os

def save_defaults_to_json(parser, filename):
    # 获取所有参数的默认值
    defaults = {}

    # 添加位置参数的默认值
    for action in parser._actions:
        if not action.option_strings and action.dest != 'help':
            defaults[action.dest] = action.default if action.default is not None else ""
    if defaults["model"] == '':
        defaults["model"] = "Qwen/Qwen2-0.5B-Instruct"
    for action in parser._actions:
        # 跳过位置参数（没有option_strings的）
        if not action.option_strings:
            continue
            
        # 获取参数名（选择最长的选项名，去掉前面的--或-）
        name = max(action.option_strings, key=len).lstrip('-')
        if (name == 'low' or name == 'path'):
            continue
        if (name == 'device'):
            action.default = ""
        # 处理store_true类型的参数
        if action.const is True:
            defaults[name] = False  # store_true参数的默认值是False
        else:
            defaults[name] = action.default
    defaults["FASTLLM_USE_NUMA"] = "OFF"
    defaults["FASTLLM_NUMA_THREADS"] = 27

    # 将字典转换为JSON并保存到文件
    import json
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(defaults, f, indent = 4, ensure_ascii = False)
    
    print("Create config to -> \"" + filename + "\"")

def args_parser():
    parser = argparse.ArgumentParser(description = "fastllm")
    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # 创建共享的解析器
    shared_parser = make_normal_parser("fastllm", add_help = False)

    # 下载解析器
    from ftllm.download import make_download_parser
    download_parser = make_download_parser(add_help = False)

    # 打开ui界面
    ui_parser_ = subparsers.add_parser('ui', parents = [shared_parser], help = 'ui模式')

    # 创建chat子命令（使用共享解析器）
    chat_parser_ = subparsers.add_parser('chat', parents = [shared_parser], help = '聊天模式')

    # 创建run子命令（使用相同的共享解析器）
    run_parser_ = subparsers.add_parser('run', parents = [shared_parser], help = '运行模式')

    download_parser_ = subparsers.add_parser('download', parents = [download_parser], help = '下载模型')

    # 创建webui子命令（独立的解析器）
    webui_parser_ = subparsers.add_parser('webui', parents = [shared_parser], help='Web UI')
    webui_parser_.add_argument('--port', type = int, default = 1616, help = '端口号')
    webui_parser_.add_argument("--max_token", type = int, default = 4096, help = "输出最大token数")
    webui_parser_.add_argument("--think", type = str, default = "false", help = "if <think> lost")

    server_parser = shared_parser
    from ftllm.util import add_server_args
    add_server_args(server_parser)
    server_parser_ = subparsers.add_parser('serve', parents = [server_parser], help = 'api模式')
    serve_parser_ = subparsers.add_parser('server', parents = [server_parser], help = 'api模式')

    config_parser_ = subparsers.add_parser('config', help = '创建配置文件')
    config_parser_.add_argument('file', nargs='?', help = '配置文件的路径')

    export_parser_ = subparsers.add_parser('export', parents = [shared_parser], help = '创建配置文件')
    export_parser_.add_argument('-o', '--output', type = str, required = True, help = '导出路径')

    parser.add_argument('-v', '--version', action='store_true', help='输出版本号并退出')

    return parser

def main():
    args = args_parser().parse_args()
    if (args.version):
        from . import __version__
        print("ftllm version: " + __version__)
        return
    # 根据不同的子命令执行不同的操作
    if args.command == 'ui':
        from .ui import FastllmStartUI
        FastllmStartUI()
    if args.command == 'config':
        file = args.file
        if not(file) or file == '':
            file = "config.json"
        if os.path.exists(file):
            choice = input("File '" + file + "' exists，replace it? (Y/N): ").strip().upper()
            if choice == 'Y':
                pass
            else:
                return
        save_defaults_to_json(make_normal_parser("fastllm", add_help = False), file)
    elif args.command in ('chat', 'run'):
        from ftllm.chat import fastllm_chat
        fastllm_chat(args)
    elif args.command == "download":
        from ftllm.download import HFDDownloader
        HFDDownloader(args).run()
    elif args.command == 'webui':
        current_path = os.path.dirname(os.path.abspath(__file__))
        web_demo_path = os.path.join(current_path, 'web_demo.py')
        
        args_dict = vars(args)
        args_dict.pop('command', None)
        port = args_dict.pop('port', 1616)
        model = args_dict.pop('model', '')
        # Convert remaining arguments to command line format
        args_list = []
        if (model != ''):
            args_list.append(model)
        for key, value in args_dict.items():
            if value is not None:
                if isinstance(value, bool) and key not in ["think"]:
                    if value:
                        args_list.append(f"--{key}")
                elif key in ["moe_device"] and value != '':
                    args_list.append(f"--{key} \"{value}\"")
                elif value != '':
                    args_list.append(f"--{key} {value}")
        
        # Build the command
        cmd = f"streamlit run --server.port {port} {web_demo_path} -- {' '.join(args_list)}"
        print(f"Running: {cmd}")
        os.system(cmd)
        return
    elif args.command in ('server', 'serve'):
        from ftllm.server import fastllm_server
        fastllm_server(args)
    elif args.command == 'export':
        from ftllm import llm
        if (args.path == '' or args.path is None):
            args.path = args.model
        llm.export_llm_model_fromhf(path = args.path, dtype = args.dtype, moe_dtype = args.moe_dtype, lora = args.lora, output = args.output, dtype_config = args.dtype_config)
    else:
        print("Invalid command: ", args.command)
        exit(0)

if __name__ == "__main__":
    main()