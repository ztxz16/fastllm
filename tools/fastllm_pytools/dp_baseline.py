"""已验收的四卡V100 Qwen3服务配置；统一经正式CLI启动，不编译或下载。"""

import argparse
import json
import math
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PROFILES = {'dp4_tp1': (4, 1), 'dp2_tp2': (2, 2)}


def model_flags(model, generate, worker, output, timeout, tp):
    """/** 构造同TP的正式模型参数；DP数和监听参数由服务入口补充。 */"""
    flags = [str(Path(model).resolve()), '--tp-backend', 'process', '--tp', str(tp),
             '--tp-binary', str(Path(generate).resolve()), '--tp-worker', str(Path(worker).resolve()),
             '--tp-output', str(Path(output).resolve()), '--tp-timeout', str(timeout),
             '--tp-request-timeout', str(timeout), '--tp-request-log', 'on',
             '--prefix-cache', 'false', '--cuda-graph']
    if tp == 2:
        flags.append('--custom-allreduce')
    return flags


def server_flags(profile, model, generate, worker, output, timeout, host='127.0.0.1', port=8080):
    """/** 固定轮转路由与2×DP等待队列；建议客户端并发等于DP，不把队列容量当作并发上限。 */"""
    dp, tp = PROFILES[profile]
    return model_flags(model, generate, worker, output, timeout, tp) + [
        '--dp', str(dp), '--dp-routing', 'round_robin', '--tp-queue-size', str(2*dp),
        '--host', host, '--port', str(port), '--model_name', 'qwen3']


def main(argv=None):
    """/** 打印可复核命令或交给正式CLI；缺失模型/原生产物直接失败，不回退到其它后端。 */"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=PROFILES, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--generate', type=Path, default=ROOT / 'build-process-tp-cu126/fastllm_tp_generate')
    parser.add_argument('--worker', type=Path, default=ROOT / 'build-process-tp-cu126/fastllm_tp_worker')
    parser.add_argument('--output', type=Path, default=ROOT / 'process-tp-results/baseline')
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--print-command', action='store_true', help='只输出配置和命令JSON，不枚举CUDA、不加载模型')
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout < 5 or not 0 <= args.port <= 65535 or not args.host:
        parser.error('requires timeout>=5, nonempty host and port in 0..65535')

    flags = server_flags(args.profile, args.model, args.generate, args.worker, args.output,
                         args.timeout, args.host, args.port)
    dp, tp = PROFILES[args.profile]
    if args.print_command:
        print(json.dumps(dict(profile=args.profile, dp=dp, tp=tp, recommended_client_concurrency=dp,
            queue_size=2*dp, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            command=[sys.executable, '-m', 'fastllm_pytools.cli', 'serve', *flags]), ensure_ascii=False, indent=2))
        return 0

    if not args.model.is_dir():
        parser.error('--model must be an existing local model directory')
    for name, path in (('generate', args.generate), ('worker', args.worker)):
        if not path.is_file() or not os.access(path, os.X_OK):
            parser.error(f'missing executable {name}: {path}; use the previously validated native build')
    print(f'PROFILE: {args.profile}; recommended client concurrency={dp}; queue size={2*dp}', flush=True)
    from . import cli
    previous = sys.argv
    try:
        sys.argv = ['ftllm', 'serve', *flags]
        return cli.main() or 0
    finally:
        sys.argv = previous


if __name__ == '__main__':
    raise SystemExit(main())
