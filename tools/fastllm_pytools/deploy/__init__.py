"""Single-host replica deployment, independent of the ordinary serve command."""
import json


def add_deploy_args(parser):
    parser.add_argument('action', choices=('render', 'start', 'status', 'stop'))
    parser.add_argument('--config', required=True, help='Deployment JSON path')


def main(args):
    from .config import load_config, load_target, public_plan
    config = (load_config if args.action in ('render', 'start') else load_target)(args.config)
    if args.action == 'render':
        print(json.dumps(public_plan(config), ensure_ascii=False, indent=2))
        return 0
    from .supervisor import start, status, stop
    result = {'start': start, 'status': status, 'stop': stop}[args.action](config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
