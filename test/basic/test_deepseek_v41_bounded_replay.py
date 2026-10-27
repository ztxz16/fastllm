#!/usr/bin/env python3
"""Unpack the existing micro fixture and run the CPU bounded-prefill regression.

python test/basic/test_deepseek_v41_bounded_replay.py --binary BUILD/deepseekV41BoundedReplayRegression
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

from test_deepseek_v41_cpu_fixture import unpack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='v41-bounded-') as directory:
        unpack(str(Path(__file__).with_name('deepseek_v41_fixture.npz')), directory)
        env = dict(os.environ, FASTLLM_SKIP_WARMUP='1', FASTLLM_DSV41_DISABLE_FAKE_QUANT='1',
                   FASTLLM_DSV41_DECODER_SWA_BOUNDED_REPLAY='1')
        subprocess.run([args.binary, directory], env=env, check=True)


if __name__ == '__main__':
    main()
