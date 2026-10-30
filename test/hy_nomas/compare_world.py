"""Replay two camera-conditioned world segments with official modules.

Identical initial/commit/decoder noise is read from capture_world.py's trace.
Reference history is built from reference outputs, never native KV tensors.
This covers the bounded plain-prompt session path, not the official UI's prompt
enhancer, action-text parser, or stable-prefix streaming decoder.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F
from safetensors import safe_open
from compare_block import official_symbols, metrics
from compare_camera import camera_symbols, preprocess, official_camera_input
import compare_models as common


@torch.no_grad()
def main(args):
    dtype = torch.float32 if args.precision == 'fp32' else torch.bfloat16
    common.DTYPE = dtype
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    t, h, w = 5, args.height//32, args.width//32
    read = lambda segment, name: np.fromfile(args.trace / f'step_{segment}_{name}.f32', dtype=np.float32)
    tensor = lambda a: torch.from_numpy(a).cuda()
    report = dict(precision=args.precision, grid=[t,h,w], steps=8,
                  relative_l2_limit=args.l2_limit,
                  scope='two segments, reference-derived KV history, plain prompt, explicit SDPA backend',
                  second_action=args.second_action, cases=[])

    def record(name, actual, expected):
        if isinstance(expected, torch.Tensor): expected = expected.float().cpu().numpy()
        result = dict(name=name, **metrics(actual, expected))
        result['passed'] = result['finite'] and result['relative_l2'] <= args.l2_limit
        report['cases'].append(result)
        args.report.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(result), flush=True)

    env = official_symbols(args.official)
    env.update(math=math, linear_dtype=lambda m:m.weight.dtype)
    base = args.official/'hymm/models'
    common.selected(env, base/'dit/modules/embed_layers.py', {'timestep_embedding','TimestepEmbedder','TextProjection'})
    common.selected(env, base/'text_encoder/token_refiner.py', {'IndividualTokenRefinerBlock','IndividualTokenRefiner','SingleTokenRefiner'})
    common.selected(env, base/'dit/modules/mlp_layers.py', {'FinalLayer'})
    histories = {}
    layer_index, committing = 0, False

    def attention(q,k,v,**kwargs):
        backend=torch.nn.attention.SDPBackend.MATH if dtype==torch.float32 else torch.nn.attention.SDPBackend.FLASH_ATTENTION
        with torch.nn.attention.sdpa_kernel(backend):
            return F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)).transpose(1,2).flatten(2)

    def joint(qs,ks,vs,**kwargs):
        n = qs[0].shape[1]
        q,k,v = [torch.cat(xs,1) for xs in (qs,ks,vs)]
        if layer_index in histories:
            hk,hv = histories[layer_index]
            k,v = torch.cat([hk,k],1),torch.cat([hv,v],1)
        result = attention(q,k,v)
        if committing: histories[layer_index] = (k,v)
        return result[:,:n],result[:,n:]

    env.update(attention=attention,causal_attention_with_cache_mp=joint)
    cam_env = camera_symbols(args.official)
    checkpoint=args.weights/'DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors'
    with safe_open(str(checkpoint),framework='pt',device='cpu') as weights:
        refiner=common.load(env['SingleTokenRefiner'](3584,2048,16,2,device='meta',dtype=dtype),weights,'txt_in.')
        time_in=common.load(env['TimestepEmbedder'](2048,torch.nn.SiLU,device='meta',dtype=dtype),weights,'time_in.')
        blocks=[common.load(env['MMDoubleStreamBlock'](2048,16,4,qkv_bias=True,device='meta',dtype=dtype),weights,f'double_blocks.{i}.') for i in range(54)]
        actions=[common.load(cam_env['ActionModule'](keyboard_dim_in=64,per_chunk_camera_attn=True,camera_chunk_size=1),weights,f'action_modules.{i}.') for i in range(54)]
        final=common.load(env['FinalLayer'](2048,[1,1,1],32,torch.nn.SiLU,device='meta',dtype=dtype),weights,'final_layer.')
        image_weight=weights.get_tensor('img_in.proj.weight').flatten(1).cuda().to(dtype)
        image_bias=weights.get_tensor('img_in.proj.bias').cuda().to(dtype)
        reference_cameras=[official_camera_input(args.official,20,action) if action!=9 else None
                           for action in (0,args.second_action)]
        for s,poses in enumerate(reference_cameras):
            if poses is not None:record(f'segment_{s}_camera_input',read(s,'camera'),poses)
            else:assert read(s,'camera').size==0, 'text-only segment must not carry a camera condition'
        cameras=[preprocess(weights,tensor(poses)) if poses is not None else None for poses in reference_cameras]
    reference_features=[np.load(args.features),np.load(args.second_features or args.features)]
    sigmas=torch.linspace(1,0,9);sigmas=5*sigmas/(1+4*sigmas)
    finals=[]
    for segment in range(2):
        features=tensor(reference_features[segment])[None].to(dtype); nt=features.shape[1]
        record(f'segment_{segment}_features',read(segment,'features'),reference_features[segment])
        latent=tensor(read(segment,'latent_0').reshape(1,t*h*w,32))
        pos=torch.meshgrid(torch.arange(segment*t,(segment+1)*t,device='cuda'),torch.arange(h,device='cuda'),torch.arange(w,device='cuda'),indexing='ij')
        freq=torch.cat([p.flatten()[:,None]*256**(-torch.arange(0,d,2,device='cuda').float()/d) for p,d in zip(pos,[16,56,56])],-1).repeat_interleave(2,-1)

        def forward(value,timestep):
            nonlocal layer_index
            timestep=torch.tensor([timestep],device='cuda',dtype=torch.float32)
            txt=refiner(features,timestep,torch.ones(1,nt,device='cuda',dtype=torch.bool))
            vec=time_in(timestep);x=F.linear(value.to(dtype),image_weight,image_bias)
            for layer_index,(block,action) in enumerate(zip(blocks,actions)):
                x,txt=block.forward_with_cache(x,txt,vec,freqs_cis=(freq.cos(),freq.sin()),kv_cache={})
                if cameras[segment] is not None:x=action(x,t,h,w,cameras[segment])
            return final(x,vec).float()

        committing=False
        for step in range(8):
            latent=latent+forward(latent,float(sigmas[step]*1000))*(sigmas[step+1]-sigmas[step]).cuda()
            record(f'segment_{segment}_euler_{step+1}',read(segment,f'latent_{step+1}'),latent)
        finals.append(latent)
        cache_noise=tensor(read(segment,'cache_noise').reshape(1,t*h*w,32))
        cache_input=(1-np.float32(.206897))*latent+np.float32(.206897)*cache_noise
        record(f'segment_{segment}_cache_input',read(segment,'cache_input'),cache_input)
        committing=True
        forward(cache_input,float(np.float32(.206897)*np.float32(1000)))
        print(f'committed segment {segment}: {histories[0][0].shape[1]} history tokens',flush=True)
    del blocks,actions,refiner,time_in,final,image_weight,image_bias,histories,cache_noise,cache_input
    torch.cuda.empty_cache()
    file=args.official/'hymm/models/vae/pixelumm_decoder/model.py'
    spec=importlib.util.spec_from_file_location('world_pixel_reference',file)
    pixel=importlib.util.module_from_spec(spec);sys.modules[spec.name]=pixel;spec.loader.exec_module(pixel)
    with safe_open(str(args.weights/'VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors'),framework='pt',device='cpu') as weights:
        with torch.device('meta'):decoder=pixel.DCAEVPixelUMMDecoder(local_attention_tile=8)
        decoder=common.load(decoder,weights,'decoder.')
    report['rgb']=[]
    for segment in range(2):
        context=torch.cat(finals[:segment+1],1);decode_t=(segment+1)*t
        record(f'segment_{segment}_decoder_context',read(segment,'decoder_latents'),context)
        noise=pixel.unpatchify(tensor(read(segment,'pixel_noise').reshape(1,-1,3072)),(decode_t,h*2,w*2))
        expected=decoder(context.reshape(1,decode_t,h,w,32).permute(0,4,1,2,3).to(dtype),noise)
        record(f'segment_{segment}_decoded_pixels',read(segment,'pixels'),pixel.patchify(expected))
        rgb=((expected[0,:,-20:].permute(1,2,3,0).float()*.5+.5).clamp(0,1)*255).round().byte().cpu().numpy()
        actual=np.fromfile(args.trace/f'step_{segment}_served.rgb',dtype=np.uint8).reshape(rgb.shape)
        diff=np.abs(actual.astype(np.int16)-rgb.astype(np.int16))
        rgb_metrics=metrics(actual,rgb)
        rgb_pass=bool(diff.max()<=1 and diff.mean()<.01) if dtype==torch.float32 else rgb_metrics['finite'] and rgb_metrics['relative_l2']<=args.l2_limit
        report['rgb'].append(dict(segment=segment,max_channel_error=int(diff.max()),mean_channel_error=float(diff.mean()),relative_l2=rgb_metrics['relative_l2'],passed=rgb_pass))
    report['passed']=all(c['passed'] for c in report['cases']+report['rgb'])
    args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    if not report['passed']:raise SystemExit('World comparison failed')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('official','weights','trace','features','report'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--second-features',type=Path,help='Reference text features for a live prompt edit in segment two')
    p.add_argument('--second-action',type=int,choices=(1,9),default=1,help='1 = forward camera; 9 = text only')
    p.add_argument('--width',type=int,default=128);p.add_argument('--height',type=int,default=128)
    p.add_argument('--precision',choices=('fp32','bf16'),default=os.environ.get('FASTLLM_NOMAS_PRECISION','bf16'))
    p.add_argument('--l2-limit',type=float)
    args=p.parse_args()
    if args.l2_limit is None:args.l2_limit=.001 if args.precision=='fp32' else .02
    main(args)
