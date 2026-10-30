"""Compare native camera conditioning and normalized motion with official code."""
import argparse
import ast
import contextlib
import ctypes as C
import json
import math
import os
import sys
import tempfile
from types import SimpleNamespace
from pathlib import Path
import typing

import numpy as np
import torch
from torch.nn import functional as F
from safetensors import safe_open
from compare_block import official_symbols, metrics
import compare_models as common


def camera_symbols(root):
    env=official_symbols(root)
    env.update(vars(typing))
    env.update(contextlib=contextlib,get_sequence_parallel_state=lambda:False,
               _MAGI_ATTENTION_AVAILABLE=True,_segmented_attn_range_cache={})
    pose=root/'hymm/models/dit/modules/posemb_layers.py'
    common.selected(env,pose,{n.name for n in ast.parse(pose.read_text()).body if isinstance(n,ast.FunctionDef)})
    file=root/'hymm/models/camera/action_module.py'
    common.selected(env,file,{n.name for n in ast.parse(file.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))})
    def ranges(q,k,v,q_ranges,k_ranges,**kwargs):
        mask=torch.zeros(q.shape[0],k.shape[0],device=q.device,dtype=torch.bool)
        for (a,b),(c,d) in zip(q_ranges.tolist(),k_ranges.tolist()): mask[a:b,c:d]=True
        out=F.scaled_dot_product_attention(q.transpose(0,1)[None],k.transpose(0,1)[None],v.transpose(0,1)[None],attn_mask=mask)
        return out[0].transpose(0,1),None
    env.update(range_attention=ranges,flex_camera_range_attention=ranges,
               flash_attn_func=lambda q,k,v,**kw:F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)).transpose(1,2))
    return env


def preprocess(weights,extrinsics):
    def linear(x,name):
        return F.linear(x,weights.get_tensor(name+'.weight').cuda().to(common.DTYPE),weights.get_tensor(name+'.bias').cuda().to(common.DTYPE))
    x=linear(extrinsics.reshape(1,-1,64).to(common.DTYPE),'extrinsics_embedding_camctrl')
    return x+linear(F.silu(linear(x,'c2ws_hidden_states_layer1')),'c2ws_hidden_states_layer2')


def official_camera_input(root, frames, action):
    """Execute upstream keyboard -> prepare_camera_inputs -> model packing.

    Uses complete upstream function bodies, including canonical resampling,
    per-segment normalization, rotation gain, first-frame padding and grouping.
    Identity projections expose the exact pre-linear packed camera input.
    """
    from scipy.spatial.transform import Rotation, Slerp
    sys.path.insert(0, str(root)) if str(root) not in sys.path else None
    env = dict(np=np, torch=torch, F=F, math=math, os=os, json=json,
               R=Rotation, Slerp=Slerp,
               logger=SimpleNamespace(info=lambda *a:None, warning=lambda *a:None))
    common.selected(env, root/'reactor/hy_nomas_camera.py', {'plan_keyboard'})
    common.selected(env, root/'hymm/utils/camera_utils.py',
                    {'canonical_camera_frame_count','pixel_to_camera_boundaries'})
    common.selected(env, root/'hymm/pipeline/bidirectional.py',
                    {'_resample_camera_poses','_resample_demo_camera_segments','prepare_camera_inputs'})
    keys = ((),('w',),('s',),('a',),('d',),('left',),('right',),('up',),('down',))[action]
    poses, intrinsics = env['plan_keyboard'](keys, frames)
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory)/'camera.npz'
        np.savez(source, c2ws=poses, intrinsics=intrinsics)
        camera = env['prepare_camera_inputs'](
            str(source),frames,torch.device('cpu'),torch.float32,
            normalize_relative_to_first_frame=True,normalize_framewise=True,
            normalize_displacement=True,normalize_displacement_per_video=True,
            segment_boundaries_pixel=[0,frames],translation_scale=1.0,rotation_scale=2.5)
    path = root/'hymm/models/dit/nomas_cpt.py'
    model = next(n for n in ast.parse(path.read_text()).body
                 if isinstance(n,ast.ClassDef) and n.name=='CPTHYNomasDiffusionTransformer')
    method = next(n for n in model.body if isinstance(n,ast.FunctionDef) and n.name=='_preprocess_camera')
    tree = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),method],type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),env)
    model = SimpleNamespace(camera_injection_mode='extrinsics_attn',
        extrinsics_embedding_camctrl=lambda x:x,
        c2ws_hidden_states_layer1=lambda x:x,
        c2ws_hidden_states_layer2=torch.zeros_like)
    packed = env['_preprocess_camera'](model,camera['norm_extrinsics'],camera['norm_intrinsics'],
                                      camera['action_onehot'],64,64,torch.float32,torch.device('cpu'))
    return packed.numpy().reshape(frames,4,4)


@torch.no_grad()
def main(args):
    torch.set_num_threads(2);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    common.DTYPE=torch.float32
    lib=C.CDLL(str(args.library));fp=C.POINTER(C.c_float)
    lib.fastllm_nomas_error.restype=C.c_char_p
    lib.fastllm_nomas_create.argtypes=[C.c_char_p,C.c_int];lib.fastllm_nomas_create.restype=C.c_void_p
    lib.fastllm_nomas_destroy.argtypes=[C.c_void_p]
    lib.fastllm_nomas_camera.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_int,fp,fp,fp]
    lib.fastllm_nomas_camera_trajectory.argtypes=[C.c_int,C.c_int,fp]
    ptr=lambda a:a.ctypes.data_as(fp)
    report={'precision':'fp32','camera_chunk_size':1,
            'scope':'official keyboard, complete camera preparation, model padding/packing, camera projection',
            'cases':[]}
    for frames in (4,8,12,20,40):
        for action in range(9):
            expected=official_camera_input(args.official,frames,action)
            actual=np.empty((frames,4,4),np.float32)
            assert lib.fastllm_nomas_camera_trajectory(frames,action,ptr(actual))==0
            result=dict(name=f'camera_input_{frames}_{action}',**metrics(actual,expected))
            result['passed']=result['max_abs']<1e-6
            report['cases'].append(result)
    if args.geometry_only:
        report['passed']=all(c['passed'] for c in report['cases'])
        args.report.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'passed':report['passed'],'cases':len(report['cases']),
                          'max_abs':max(c['max_abs'] for c in report['cases'])}),flush=True)
        if not report['passed']:raise SystemExit('Camera preprocessing mismatch')
        return
    checkpoint=args.weights/'DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors'
    handle=lib.fastllm_nomas_create(str(checkpoint).encode(),0)
    if not handle:raise RuntimeError(lib.fastllm_nomas_error().decode())
    env=camera_symbols(args.official)
    try:
        with safe_open(str(checkpoint),framework='pt',device='cpu') as weights:
            for layer,t,h,w in [(0,1,2,2),(27,3,2,3),(53,5,4,4)]:
                raw=np.empty((t*4,4,4),np.float32)
                assert lib.fastllm_nomas_camera_trajectory(t*4,6,ptr(raw))==0
                cam=preprocess(weights,torch.from_numpy(raw).cuda())
                module=env['ActionModule'](keyboard_dim_in=64,per_chunk_camera_attn=True,camera_chunk_size=1)
                module=common.load(module,weights,f'action_modules.{layer}.')
                x=torch.randn(1,t*h*w,2048,device='cuda')
                expected=module(x,t,h,w,cam)
                actual=np.empty((t*h*w,2048),np.float32);inp=x.cpu().numpy()
                rc=lib.fastllm_nomas_camera(handle,layer,t,h,w,ptr(inp),ptr(raw),ptr(actual))
                if rc:raise RuntimeError(lib.fastllm_nomas_error().decode())
                result=dict(name=f'camera_block_{layer}',**metrics(actual,expected.cpu().numpy()))
                result['passed']=result['relative_l2']<.001 and result['cosine']>.999999
                report['cases'].append(result);print(json.dumps(result),flush=True)
                del module
    finally:lib.fastllm_nomas_destroy(handle)
    report['passed']=all(c['passed'] for c in report['cases'])
    args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    if not report['passed']:raise SystemExit('Camera precision failed')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('official','weights','library','report'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--geometry-only',action='store_true')
    main(p.parse_args())
