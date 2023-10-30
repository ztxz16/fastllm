import sys
import logging
import sys
import struct
import numpy as np
import argparse

from .utils import convert
from .utils.converter import QuantType

def parse_args():
    # -p 模型路径或hf路径
    # -o --out_path 导出路径
    # -q 量化位数
    parser = argparse.ArgumentParser(description='build fastllm libs')
    parser.add_argument('-o', dest='export_path', default=None,
                    help='output export path')
    parser.add_argument('-p', dest='model_path', type=str, default='',
                    help='the model path or huggingface path, such as: -p THUDM/chatglm-6b')
    parser.add_argument('--lora', dest='lora_path', default='',
                    help='lora model path')
    parser.add_argument('-m', dest='model', default='chatglm6B',
                    help='model name with(alpaca, baichuan7B, chatglm6B, moss)')
    parser.add_argument('-q', dest='q_bit', type=int,
                    help='model quantization bit')
    args = parser.parse_args()
    return args


def main(args=None):
    if not args: args = parse_args()

    quant_type_to_qbit = {
        QuantType.FP32: 32,
        QuantType.FP16: 16,
        QuantType.INT8: 8,
        QuantType.INT4: 4,
    }
    qbit_to_quant_type = {v: k for k, v in quant_type_to_qbit.items()}
    q_type = qbit_to_quant_type[args.q_bit]
    convert(args.model_path, args.export_path, q_type=q_type)

if __name__ == "__main__":
    args = parse_args()
    main(args)