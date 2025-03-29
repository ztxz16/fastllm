import argparse
from .util import make_normal_parser
import readline
import os

def args_parser():
    parser = argparse.ArgumentParser(description = "fastllm")
    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # 创建共享的解析器
    shared_parser = make_normal_parser("fastllm", add_help = False)

    # 下载解析器
    from ftllm.download import make_download_parser
    download_parser = make_download_parser(add_help = False)

    # 创建chat子命令（使用共享解析器）
    chat_parser_ = subparsers.add_parser('chat', parents = [shared_parser], help = '聊天模式')

    # 创建run子命令（使用相同的共享解析器）
    run_parser_ = subparsers.add_parser('run', parents = [shared_parser], help = '运行模式')

    download_parser_ = subparsers.add_parser('download', parents = [download_parser], help = '下载模型')

    # 创建webui子命令（独立的解析器）
    webui_parser_ = subparsers.add_parser('webui', parents = [shared_parser], help='Web UI')
    webui_parser_.add_argument('--port', type = int, default = 1616, help = '端口号')

    server_parser = shared_parser
    from ftllm.util import add_server_args
    add_server_args(server_parser)
    server_parser_ = subparsers.add_parser('serve', parents = [server_parser], help = 'api模式')
    serve_parser_ = subparsers.add_parser('server', parents = [server_parser], help = 'api模式')

    return parser

def main():
    args = args_parser().parse_args()
    # 根据不同的子命令执行不同的操作
    if args.command in ('chat', 'run'):
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
                if isinstance(value, bool):
                    if value:
                        args_list.append(f"--{key}")
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
    else:
        print("Invalid command: ", args.command)
        exit(0)

if __name__ == "__main__":
    main()