import argparse
from ftllm import llm
from .util import make_normal_parser
from .util import make_normal_llm_model
import readline

def args_parser():
    parser = make_normal_parser('fastllm_export')
    parser.add_argument('-o', '--output', type = str, required = True, help = '导出路径')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = args_parser()
    llm.export_llm_model_fromhf(path = args.path, dtype = args.dtype, moe_dtype = args.moe_dtype, lora = args.lora, output = args.output, dtype_config = args.dtype_config)