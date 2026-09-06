"""Optional SM90 FlashInfer GDN compiler, loaded only by a /compile request.

CuTe's C export includes the cubin and the host code that encodes TMA
descriptors. The cached shared library executes without Python or PyTorch in
the inference process. FlashInfer and CuTe DSL remain compiler dependencies.
"""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver
from cutlass.cute.export import aot_config
from flashinfer.gdn_kernels.delta_rule_dsl import delta_rule_cp_sm90 as cp


class FlashInferGdnSm90:
    def __init__(self, key_heads, value_heads):
        self.kh, self.vh = key_heads, value_heads
        self.t = cp.CPDeltaRuleTPrecomputeSm90(cutlass.Float16)
        self.mn = cp.CPDeltaRuleMNPrecomputeSm90(cutlass.Float16)
        self.fix = cp.CPDeltaRuleFixupHmmaSm90(True)
        self.o = cp.CPDeltaRulePrefillSm90(cutlass.Float16, needs_initial_state=True)

    @cute.jit
    def __call__(
        self, q: cute.Pointer, k: cute.Pointer, v: cute.Pointer,
        alpha: cute.Pointer, beta: cute.Pointer, initial: cute.Pointer,
        state: cute.Pointer, out: cute.Pointer, t: cute.Pointer,
        transfer: cute.Pointer, local: cute.Pointer, fixed: cute.Pointer,
        maps: cute.Pointer, seq: cute.Pointer,
        tokens: cutlass.Int32, cp_len: cutlass.Int32,
        sms: cutlass.Int32, stream: driver.CUstream,
    ):
        kh, vh = self.kh, self.vh
        nt = (tokens + 63) // 64
        nc = (tokens + cp_len - 1) // cp_len
        tq = cute.make_tensor(q, cute.make_layout(
            (tokens, 128, kh), stride=(kh * 128, 1, 128)))
        tk = cute.make_tensor(k, cute.make_layout(
            (128, tokens, kh), stride=(1, kh * 128, 128)))
        # V points into the combined Q/K/V convolution output, without a copy.
        tv = cute.make_tensor(v, cute.make_layout(
            (128, tokens, vh), stride=(1, (2 * kh + vh) * 128, 128)))
        to = cute.make_tensor(out, cute.make_layout(
            (128, tokens, vh), stride=(1, vh * 128, 128)))
        tt = cute.make_tensor(t, cute.make_layout(
            (64, 64, vh, nt), stride=(64, 1, 4096, vh * 4096)))
        tm = cute.make_tensor(transfer, cute.make_layout(
            (128, 128, vh, nc), stride=(128, 1, 16384, vh * 16384)))
        tn = cute.make_tensor(local, cute.make_layout(
            (128, 128, vh, nc), stride=(128, 1, 16384, vh * 16384)))
        ta = cute.make_tensor(alpha, cute.make_layout(tokens * vh))
        tb = cute.make_tensor(beta, cute.make_layout(tokens * vh))
        ti = cute.make_tensor(initial, cute.make_layout(vh * 16384))
        ts = cute.make_tensor(state, cute.make_layout(vh * 16384))
        tf = cute.make_tensor(fixed, cute.make_layout(nc * vh * 16384))
        tseq = cute.make_tensor(seq, cute.make_layout(2))
        self.t(tk, tb, cute.make_tensor(t, cute.make_layout(nt * vh * 4096)),
               tseq, cutlass.Int32(kh), cutlass.Int32(vh), nt, nt,
               cutlass.Int32(1), stream)
        self.mn(tk, tv, tt, ta,
                cute.make_tensor(transfer, cute.make_layout(nc * vh * 16384)),
                cute.make_tensor(local, cute.make_layout(nc * vh * 16384)),
                tseq, cp_len, cutlass.Int32(kh), cutlass.Int32(vh),
                cutlass.Int32(vh), nc, nc, cutlass.Int32(1), stream)
        self.fix(tm, tn, ti, tf, tseq, cp_len, nc, cutlass.Int32(1),
                 cutlass.Int32(vh), stream)
        self.o(tq, tk, tv, tt, to, ta, ts, tf, ti,
               cute.make_tensor(maps, cute.make_layout(sms * 128)), tseq,
               cutlass.Float32(128 ** -0.5), cutlass.Int32(kh),
               cutlass.Int32(kh), cutlass.Int32(vh), cutlass.Int32(vh),
               cp_len, nc, nc, cutlass.Int32(1), stream)


_BRIDGE = r'''
#include "gdn.h"
#include <mutex>
#include <set>
static gdn_Kernel_Module_t module;
extern "C" int fastllm_flashinfer_gdn_init(int device) {
    static std::mutex mutex;
    static bool initialized = false;
    static std::set<int> devices;
    std::lock_guard<std::mutex> guard(mutex);
    if (devices.count(device)) return 0;
    cudaLibrary_t *library = &module.module;
    cudaError_t ret = cudaSuccess;
    if (!initialized) {
        void *args[] = {&library, &ret};
        _mlir_gdn_cuda_init(args);
        if (ret != cudaSuccess) return ret;
        initialized = true;
    }
    void *args[] = {&library, &device, &ret};
    _mlir_gdn_cuda_load_to_device(args);
    if (ret == cudaSuccess) devices.insert(device);
    return ret;
}
extern "C" int fastllm_flashinfer_gdn_launch(
        void **p, int tokens, int cp, int sms, void *stream) {
    return cute_dsl_gdn_wrapper(&module,
        p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7],
        p[8], p[9], p[10], p[11], p[12], p[13],
        tokens, cp, sms, (cudaStream_t)stream);
}
'''


def compile_gdn(payload):
    kh, vh = int(payload['key_heads']), int(payload['value_heads'])
    if (int(payload['arch']) != 90 or payload['dtype'] != 'fp16'
            or kh <= 0 or vh < 32 or vh > 128 or vh % kh):
        raise ValueError('FlashInfer GDN requires SM90, FP16, and compatible heads')
    cache = Path(payload['cache_dir']).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    name = f'flashinfer_gdn_v1_fp16_sm90_k{kh}_v{vh}'
    versions = {name: importlib.metadata.version(name) for name in
                ('flashinfer-python', 'nvidia-cutlass-dsl')}
    libdir = str(Path(aot_config.get_libdir()).resolve())
    digest = hashlib.sha256(Path(__file__).read_bytes()
                            + Path(cp.__file__).read_bytes()
                            + json.dumps(versions, sort_keys=True).encode()
                            + libdir.encode()).hexdigest()[:16]
    dest = cache / f'{name}_{digest}'
    library = dest / 'gdn.so'
    if not library.exists():
        with tempfile.TemporaryDirectory(prefix=name + '_', dir=cache) as tmp:
            work = Path(tmp)
            types = ([cutlass.Float16] * 3 + [cutlass.Float32] * 4
                     + [cutlass.Float16] * 2 + [cutlass.Float32] * 3
                     + [cutlass.Int8, cutlass.Int64])
            args = [cute.runtime.make_ptr(
                dt, 0, cute.AddressSpace.gmem, assumed_align=128) for dt in types]
            args += [cutlass.Int32(1024), cutlass.Int32(512),
                     cutlass.Int32(114), driver.CUstream(0)]
            compiled = cute.compile(FlashInferGdnSm90(kh, vh), *args,
                options=f'--gpu-arch=sm_90a --keep-cubin --dump-dir={work}')
            compiled.export_to_c(str(work), 'gdn')
            (work / 'gdn.cubin').write_bytes(compiled.artifacts.CUBIN)
            (work / 'bridge.cpp').write_text(_BRIDGE)
            nvcc = shutil.which('nvcc')
            cuda = Path(os.environ.get('CUDA_HOME') or
                        (str(Path(nvcc).resolve().parent.parent) if nvcc else '/usr/local/cuda'))
            command = ['g++', '-shared', '-fPIC', '-O2', '-std=c++17', '-pthread',
                       f'-I{cuda / "include"}', str(work / 'bridge.cpp'),
                       str(work / 'gdn.o'), f'-L{libdir}',
                       *shlex.split(aot_config.get_libs()), f'-Wl,-rpath,{libdir}',
                       '-Wl,-z,defs', '-Wl,-Bsymbolic', '-o', str(work / 'gdn.so')]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError('CuTe C export link failed: ' + result.stderr[-4000:])
            dest.mkdir(exist_ok=True)
            # The shared library embeds the cubin and TMA host code. Retain
            # the standalone cubin for inspection, but discard build files.
            for filename in ('gdn.cubin', 'gdn.so'):
                os.replace(work / filename, dest / filename)
    meta = dict(ok=True, op='flashinfer_gdn', abi=1, arch=90, dtype='fp16',
                key_heads=kh, value_heads=vh, library=str(library),
                cubin=str(dest / 'gdn.cubin'), versions=versions)
    # Separate compiler processes may publish the same specialization.
    with tempfile.TemporaryDirectory(prefix=name + '_index_', dir=cache) as tmp:
        temporary = Path(tmp) / 'index.json'
        temporary.write_text(json.dumps(meta, sort_keys=True))
        temporary.replace(cache / f'{name}.json')
    return meta
