"""Prepare authentic I2V conditions with official T5, VAE and camera code."""
import argparse
import gc
import importlib
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--official', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--example', default='01')
    p.add_argument('--height', type=int, default=96)
    p.add_argument('--width', type=int, default=160)
    p.add_argument('--segments', type=int, default=4)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    for name, path in [('wan', a.official/'wan'), ('wan.modules', a.official/'wan/modules'), ('wan.utils', a.official/'wan/utils')]:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        sys.modules[name] = pkg
    example = a.official/'examples'/a.example
    prompt = (example/'prompt.txt').read_text().strip()
    t5 = importlib.import_module('wan.modules.t5')
    model = t5.umt5_xxl(encoder_only=True, return_tokenizer=False, dtype=torch.bfloat16, device='meta')
    weights = torch.load(a.model/'models_t5_umt5-xxl-enc-bf16.pth', map_location='cpu', weights_only=True, mmap=True)
    model.load_state_dict(weights, strict=True, assign=True)
    del weights
    model = model.eval().cuda()
    tokenizer = t5.HuggingfaceTokenizer(str(a.model/'google/umt5-xxl'), seq_len=512, clean='whitespace')
    ids, mask = tokenizer(prompt, return_mask=True, add_special_tokens=True)
    count = int(mask.sum())
    text = model(ids.cuda(), mask.cuda())[0, :count].float().cpu().numpy()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print('T5 ready', count, flush=True)
    latent_frames = a.segments*3
    frames = 1+(latent_frames-1)*4
    image = np.array(Image.open(example/'image.jpg').convert('RGB'), dtype=np.float32)
    image = (torch.from_numpy(image).permute(2, 0, 1)/255-.5)/.5
    image = torch.nn.functional.interpolate(image[None], (a.height, a.width), mode='bicubic')[0]
    pixels = torch.zeros(3, frames, a.height, a.width, device='cuda')
    pixels[:, 0] = image.cuda()
    vae_module = importlib.import_module('wan.modules.vae2_1')
    vae = vae_module.Wan2_1_VAE(vae_pth=str(a.model/'Wan2.1_VAE.pth'))
    encoded = vae.model.encode(pixels[None], vae.scale)[0]
    msk = torch.zeros(4, latent_frames, a.height//8, a.width//8, device='cuda')
    msk[:, 0] = 1
    condition = torch.cat([msk, encoded]).permute(1, 2, 3, 0).contiguous().cpu().numpy()
    del vae, pixels, encoded, msk
    gc.collect()
    torch.cuda.empty_cache()
    print('VAE ready', condition.shape, flush=True)
    camera_module = importlib.import_module('wan.utils.cam_utils')
    poses = np.load(example/'poses.npy')[:frames]
    poses = camera_module.interpolate_camera_poses(np.arange(len(poses)), poses[:, :3, :3],
        poses[:, :3, 3], np.linspace(0, len(poses)-1, latent_frames))
    poses = camera_module.compute_relative_poses(poses, framewise=True)
    intrinsics = camera_module.get_Ks_transformed(torch.from_numpy(np.load(example/'intrinsics.npy')).float(),
        480, 832, a.height, a.width, a.height, a.width)
    camera = camera_module.get_plucker_embeddings(poses.cuda(), intrinsics[:1].repeat(latent_frames, 1).cuda(), a.height, a.width)
    camera = camera.reshape(latent_frames, a.height//8, 8, a.width//8, 8, 6).permute(0, 1, 3, 5, 2, 4)
    camera = camera.reshape(latent_frames, a.height//8, a.width//8, 384).bfloat16().float().cpu().numpy()
    np.savez(a.output, text=text, tokens=ids[0, :count].numpy(), condition=condition, camera=camera,
             prompt=prompt, first_image=image.permute(1, 2, 0).numpy(), poses=poses.numpy(), intrinsics=intrinsics[0].numpy())
    print('saved', a.output, flush=True)


if __name__ == '__main__':
    main()
