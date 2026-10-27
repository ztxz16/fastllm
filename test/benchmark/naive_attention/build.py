"""Build only the standalone attention operators; never relink the model server.

python test/benchmark/naive_attention/build.py --arch sm_90
python test/benchmark/naive_attention/validate.py
python test/benchmark/naive_attention/bench.py --full
"""

import argparse
from pathlib import Path
import subprocess

source = Path(__file__).resolve().parent
root = source.parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--arch", default="sm_90")
parser.add_argument("--nvcc", default="/usr/local/cuda/bin/nvcc")
args = parser.parse_args()
build = source / "build"
build.mkdir(parents=True, exist_ok=True)

# Compile the current production SWA kernel as the baseline without maintaining
# another copy in the repository. DSA baseline templates come from its header.
code = (root / "src/devices/cuda/models/naive-n05-kernels.cu").read_text()
start = code.index("constexpr int kSwaQkDim")
end = code.index("// Dispatch by workload shape", start)
(build / "old_swa.cuh").write_text(
    "#include <type_traits>\n"
    "using BF16=__nv_bfloat16;\nconstexpr int kSwaWindow=128;\n"
    "struct FastllmNaiveSWARow {const void *key,*value;};\n"
    "__device__ inline float RoundBF16(float x){return naive_dsa_decode::Round(x);}\n"
    "__device__ inline float WarpSum(float x){return naive_dsa_decode::WarpSum(x);}\n"
    + code[start:end]
)
command = [
    args.nvcc,
    "-O3",
    "-std=c++17",
    "-arch=" + args.arch,
    "-shared",
    "-Xcompiler=-fPIC",
    "-I" + str(root / "include"),
    "-I" + str(root / "src/devices/cuda/models"),
    "-I" + str(build),
    str(source / "operators.cu"),
    "-lcuda",
    "-o",
    str(build / "bulk.so"),
]
subprocess.run(command, check=True)
print(build / "bulk.so")
