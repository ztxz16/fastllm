"""Single-host replica deployment, independent of the ordinary serve command."""
import json


def add_deploy_args(parser):
    parser.add_argument('action', choices=('render', 'start', 'status', 'stop'))
    parser.add_argument('--config', required=True, help='Deployment JSON path')
    parser.add_argument('--resolve-topology', action='store_true',
                        help='With render, resolve this Linux host GPU/NUMA topology without starting workers')


def main(args):
    from .config import load_config, load_target, public_plan
    if args.resolve_topology and args.action != 'render':
        raise ValueError('--resolve-topology is only valid with render; start always resolves topology')
    config = (load_config if args.action in ('render', 'start') else load_target)(args.config)
    if args.action == 'render':
        if args.resolve_topology:
            from .numa import prepare_numa
            config = prepare_numa(config)
        print(json.dumps(public_plan(config), ensure_ascii=False, indent=2))
        return 0
    from .supervisor import start, status, stop
    result = {'start': start, 'status': status, 'stop': stop}[args.action](config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
