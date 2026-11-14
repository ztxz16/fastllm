#!/usr/bin/env python3
"""生成独立的稠密小模型及本地tokenizer，调用既有正式DP/TP验收，不需下载其他模型。"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid

from run_process_tp import ROOT, case_process, wait_case
from run_qwen3_prefill import make_model


def main():
    """/** 使用既有Dense公共入口回归四种配置；任何子验收失败即非零退出。 */"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    args = parser.parse_args()
    directory = ROOT / 'test/distributed/results' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_dense_regression_' + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True)
    model = directory / 'model'
    config, _ = make_model(model, {})
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    vocab = {'[UNK]': 0, '[EOS]': config['vocab_size'] - 1}
    vocab.update({f'w{i}': i for i in range(1, config['vocab_size']-1)})
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]', eos_token='[EOS]')
    fast.chat_template = "{% for message in messages %}w1 {{ message['content'] }} w2 {% endfor %}w3"
    fast.save_pretrained(model)
    for dp, tp, graph in ((2, 1, False), (4, 1, True), (2, 2, False), (2, 2, True)):
        command = [sys.executable, str(ROOT / 'test/distributed/run_qwen3_dp.py'), '--model', str(model),
                   '--generate', str(args.build / 'fastllm_tp_generate'), '--worker', str(args.build / 'fastllm_tp_worker'),
                   '--build-cache', str(args.build / 'CMakeCache.txt'), '--dp', str(dp), '--tp', str(tp)]
        if graph:
            command += ['--cuda-graph', '--prefix-cache']
        with (directory / f'dp{dp}_tp{tp}_graph{int(graph)}.log').open('wb') as log, case_process(command, stdout=log, stderr=log) as process:
            if wait_case(process, 1800):
                raise RuntimeError(f'Dense regression failed; inspect {directory}')
    print(f'PASS: dense DP/TP regression logs: {directory}', flush=True)


if __name__ == '__main__':
    main()
