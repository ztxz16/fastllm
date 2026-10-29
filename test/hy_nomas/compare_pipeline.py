"""Replay an actual native pipeline trace using official FP32 modules.

Run native generation with FASTLLM_NOMAS_PIPELINE_TRACE pointing to an existing
private directory. Compare the same initial latent and PixelUMM noise, avoiding
any assumption that C++ and torch RNG seeds produce identical samples. Reference
text features must be from compare_text.py with the same prompt. This validates
one stateless, camera-disabled clip, not the official world-session pipeline.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F
from safetensors import safe_open
from compare_block import official_symbols, metrics
import compare_models as common


@torch.no_grad()
def main(args):
    common.DTYPE = torch.float32
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    t, h, w = args.frames//4, args.height//32, args.width//32
    read = lambda name: np.fromfile(args.trace / (name+'.f32'), dtype=np.float32)
    tensor = lambda a: torch.from_numpy(a).to('cuda')
    report = {'precision': 'fp32', 'grid': [t,h,w], 'steps': args.steps,
              'scope': 'camera-disabled stateless clip; identical noise; SDPA math reference', 'cases': []}

    def record(name, actual, expected):
        if isinstance(expected, torch.Tensor):
            expected = expected.float().cpu().numpy()
        result = dict(name=name, **metrics(actual, expected))
        result['passed'] = result['finite'] and result['relative_l2'] <= .001 and result['cosine'] >= .999999
        report['cases'].append(result)
        args.report.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(result), flush=True)

    reference_features = np.load(args.features)
    record('text_features', read('features'), reference_features)
    features = tensor(reference_features)[None]
    nt = features.shape[1]
    latent = tensor(read('latent_0').reshape(1,t*h*w,32))
    env = official_symbols(args.official)
    env.update(math=math, linear_dtype=lambda m: m.weight.dtype)
    base = args.official/'hymm/models'
    common.selected(env, base/'dit/modules/embed_layers.py', {'timestep_embedding','TimestepEmbedder','TextProjection'})
    common.selected(env, base/'text_encoder/token_refiner.py', {'IndividualTokenRefinerBlock','IndividualTokenRefiner','SingleTokenRefiner'})
    common.selected(env, base/'dit/modules/mlp_layers.py', {'FinalLayer'})

    def attention(q,k,v,**kwargs):
        return F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)).transpose(1,2).flatten(2)

    def joint(q,k,v,**kwargs):
        n=q[0].shape[1]
        a=attention(*[torch.cat(xs,1) for xs in (q,k,v)])
        return a[:,:n],a[:,n:]

    env.update(attention=attention,causal_attention_with_cache_mp=joint)
    with safe_open(str(args.weights/'DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors'),framework='pt',device='cpu') as weights:
        refiner=common.load(env['SingleTokenRefiner'](3584,2048,16,2,device='meta',dtype=torch.float32),weights,'txt_in.')
        time_in=common.load(env['TimestepEmbedder'](2048,torch.nn.SiLU,device='meta',dtype=torch.float32),weights,'time_in.')
        blocks=[common.load(env['MMDoubleStreamBlock'](2048,16,4,qkv_bias=True,device='meta',dtype=torch.float32),weights,f'double_blocks.{i}.') for i in range(54)]
        final=common.load(env['FinalLayer'](2048,[1,1,1],32,torch.nn.SiLU,device='meta',dtype=torch.float32),weights,'final_layer.')
        image_weight=weights.get_tensor('img_in.proj.weight').flatten(1).cuda().float()
        image_bias=weights.get_tensor('img_in.proj.bias').cuda().float()
    pos=torch.meshgrid(*[torch.arange(n,device='cuda') for n in (t,h,w)],indexing='ij')
    freq=torch.cat([p.flatten()[:,None]*256**(-torch.arange(0,d,2,device='cuda').float()/d) for p,d in zip(pos,[16,56,56])],-1).repeat_interleave(2,-1)
    # EulerSolver.set_timesteps / sd3_time_shift (inference preset: shift=5).
    sigmas=torch.linspace(1,0,args.steps+1)
    sigmas=5*sigmas/(1+4*sigmas)
    for step in range(args.steps):
        timestep=(sigmas[step:step+1]*1000).cuda()
        txt=refiner(features,timestep,torch.ones(1,nt,device='cuda',dtype=torch.bool))
        vec=time_in(timestep)
        x=F.linear(latent,image_weight,image_bias)
        for block in blocks:
            x,txt=block.forward_with_cache(x,txt,vec,freqs_cis=(freq.cos(),freq.sin()),kv_cache={})
        velocity=final(x,vec)
        latent=latent+velocity*(sigmas[step+1]-sigmas[step]).cuda()
        record(f'euler_step_{step+1}',read(f'latent_{step+1}'),latent)
    del blocks,block,refiner,time_in,final,image_weight,image_bias,velocity,x,txt,vec
    torch.cuda.empty_cache()
    file=args.official/'hymm/models/vae/pixelumm_decoder/model.py'
    spec=importlib.util.spec_from_file_location('pipeline_pixel_reference',file)
    pixel=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=pixel
    spec.loader.exec_module(pixel)
    with safe_open(str(args.weights/'VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors'),framework='pt',device='cpu') as weights:
        with torch.device('meta'):
            decoder=pixel.DCAEVPixelUMMDecoder(local_attention_tile=8)
        decoder=common.load(decoder,weights,'decoder.')
    noise=pixel.unpatchify(tensor(read('pixel_noise').reshape(1,-1,3072)),(t,h*2,w*2))
    expected=decoder(latent.reshape(1,t,h,w,32).permute(0,4,1,2,3),noise)
    record('decoded_pixels',read('pixels'),pixel.patchify(expected))
    expected_rgb=((expected[0].permute(1,2,3,0).float()*.5+.5).clamp(0,1)*255).round().byte().cpu().numpy()
    # Independently validate native unpatchification and quantization against
    # the RGB bytes actually served over HTTP (saved by the acceptance runner).
    if args.rgb:
        actual=np.fromfile(args.rgb,dtype=np.uint8).reshape(expected_rgb.shape)
        diff=np.abs(actual.astype(np.int16)-expected_rgb.astype(np.int16))
        report['rgb']={'max_channel_error':int(diff.max()),'mean_channel_error':float(diff.mean()),
                       'identical_channels_fraction':float((diff==0).mean()),'passed':bool(diff.max()<=1 and diff.mean()<.01)}
    report['passed']=all(c['passed'] for c in report['cases']) and report.get('rgb',{}).get('passed',True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if not report['passed']:
        raise SystemExit('Pipeline comparison failed')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('official','weights','trace','features','report'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--rgb',type=Path)
    p.add_argument('--width',type=int,default=128)
    p.add_argument('--height',type=int,default=128)
    p.add_argument('--frames',type=int,default=16)
    p.add_argument('--steps',type=int,default=8)
    main(p.parse_args())
