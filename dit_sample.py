# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from dit.diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from dit.download import find_model
# from models.dit_models_condition import DiT_models
from model.dit_models import DiT_models
import argparse

import os
import torch.nn as nn
from einops import rearrange
import random

from model.meshAE import MeshAutoencoder, DiagonalGaussianDistribution
from data.dataset import ShapeNetCore_obj
from meshgpt_pytorch.meshgpt_pytorch import undiscretize
from dit_train import save_as_obj

def p_sample_loop_with_cfg(model, shape, noise, cfg_scale, model_kwargs, device):
    """
    A loop to sample using CFG.
    - model: the diffusion model.
    - shape: the shape of the generated sample.
    - noise: initial noise tensor (z).
    - cfg_scale: guidance scale for CFG.
    - model_kwargs: additional arguments such as `y` (text condition).
    - device: the device to use (CPU/GPU).
    """
    samples = []
    x = noise
    for t in range(shape[0]):  # Assuming `shape[0]` is the number of time steps
        # Create conditional and unconditional inputs
        cond_kwargs = model_kwargs
        uncond_kwargs = model_kwargs.copy()
        uncond_kwargs['y'] = torch.zeros_like(model_kwargs['y']).to(device)  # Set y to zeros for uncond

        # Combine conditional and unconditional inputs
        combined_kwargs = {
            'y': torch.cat([uncond_kwargs['y'], cond_kwargs['y']], dim=0)
        }
        
        # Forward with CFG
        x = model.forward_with_cfg(x, t, **combined_kwargs, cfg_scale=cfg_scale)
        samples.append(x)
    return samples

def create_mask(B, N, bpt_latent_size, device='cuda'):
    """
    生成形状为 (B, N) 的 mask，每个样本的前 bpt_latent_size[i] 个位置为 1，其余为 0。
    
    参数:
        B: batch size
        N: 序列长度（x.shape[1]）
        bpt_latent_size: 需要保留的位置数（可以是 int、torch.Tensor、numpy.ndarray 或 list）
        device: 设备（如 'cuda' 或 'cpu'）
    """
    # 初始化全为 0 的 mask
    mask = torch.zeros((B, N), device=device)
    
    # 转换输入为 torch.Tensor（如果还不是）
    if not isinstance(bpt_latent_size, (int, torch.Tensor)):
        if isinstance(bpt_latent_size, (np.ndarray, list)):
            bpt_latent_size = torch.tensor(bpt_latent_size, device=device)
        else:
            raise ValueError("bpt_latent_size 必须是 int、torch.Tensor、numpy.ndarray 或 list")
    
    # 处理标量情况
    if isinstance(bpt_latent_size, int):
        mask[:, :bpt_latent_size] = 1
    # 处理张量情况
    else:
        # 确保张量在正确的设备上
        bpt_latent_size = bpt_latent_size.to(device)
        # 向量化计算
        indices = torch.arange(N, device=device).expand(B, N)  # (B, N)
        mask = torch.where(indices < bpt_latent_size.unsqueeze(1), 1.0, mask)
    
    return mask

# import matplotlib.pyplot as plt
def main(args):
    # Setup PyTorch:
    # torch.manual_seed(123)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

   
    autoencoder = MeshAutoencoder(
        num_discrete_coors = 128,
        encoder_dim = 768,
        encoder_depth = 12,
        patch_size = 1,
        dropout = 0.,
        bin_smooth_blur_sigma = 0.4,
        quant_face = False,
        decoder_fine_dim = 256,
        graph_layers = 1,

        latent_dim = 8,
        pos_embed = False,

        decoder_depth = 18,
        decoder_dim = 384,
    ).to('cuda')

    
    checkpoint = torch.load('/root/github_code/pivotmesh/checkpoints/AE-shapenet_10000obj_under800_8channel_12en_768_18de_384_1e-4kl_1GCN_womask_ordered/mesh-autoencoder.ckpt.43.pt')
    autoencoder.load_state_dict(checkpoint['model'])
    autoencoder.requires_grad = False
    autoencoder.eval()
    

    # Load model:
    # gt_feature = torch.from_numpy(np.load("feature.npy")) / 50

    latent_size = 800
    model = DiT_models[args.model](
        input_tokens=latent_size,
        in_channels=8,
    ).to(device)

    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt
    #state_dict = find_model(ckpt_path)
    
    checkpoint = torch.load(ckpt_path, weights_only=False)
    #print(checkpoint)
    model.load_state_dict(checkpoint['model'])
    
    #model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    
    bs = 10
    z = torch.randn(bs, 8, latent_size, device=device)

    pos_indices = torch.arange(latent_size, device=device)  # [0, 1, ..., N-1]
    grid = torch.stack([
        pos_indices,
        torch.zeros_like(pos_indices)
    ], dim=0).unsqueeze(0).repeat(latent_size, 1, 1)
    
    # 10objs:[34, 85,  9, 65, 17, 35,  4, 60,  8, 21]
    # seq_len = np.array([196, 156, 329, 331, 252, 315, 430, 527, 293, 422])
    seq_len = torch.randint(low=200, high=300, size=(bs,)) #latent_size
    seq_len = torch.tensor(seq_len, dtype=torch.int).to(device=device)

    mask = torch.ones(bs, latent_size).to(device=device)
    # mask = create_mask(bs, latent_size, seq_len)
    
    # short_ann_val = 'A table with legs.'
    model_kwargs = dict(mask=mask.long(), grid=grid.long(), seq_len=seq_len)

    samples = diffusion.p_sample_loop(
        model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
 
    samples = samples.permute(0, 2, 1)
    face_mask_patch = rearrange(mask, 'b (np p) -> b np p', p=autoencoder.patch_size).float().mean(dim=-1).bool()
    x_logits = autoencoder.decode(samples, face_mask_patch)

    pred_face_coords = autoencoder.to_coor_logits(x_logits)
    recon_faces = undiscretize(
        pred_face_coords.argmax(dim = -1),
        num_discrete = autoencoder.num_discrete_coors,
        continuous_range = autoencoder.coor_continuous_range,
    )
    recon_faces = rearrange(recon_faces, 'b nf (nv c) -> b nf nv c', nv = 3)
    face_mask = rearrange(mask, 'b nf -> b nf 1 1').bool()
    recon_faces = recon_faces.masked_fill(~face_mask, float('nan'))
    face_mask = rearrange(face_mask, 'b nf 1 1 -> b nf')
    print(recon_faces.shape)
    
    for idx in range(recon_faces.shape[0]):
        output_filename = os.path.join("dit_test/10000obj", f'{args.step}steps', f'recon_{idx}-th.obj')
        save_as_obj(recon_faces, output_filename, idx=idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-MIN/3")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default='C:/Users/Administrator/Desktop/abcde/results/013-DiT-MIN-2/checkpoints/0200000.pt',
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--step", type=int, default=100, required=True,)
    args = parser.parse_args()
    main(args)
