# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import yaml
from einops import rearrange, repeat, reduce, pack

from model.dit_models import DiT_models
from dit.diffusion import create_diffusion

from torch.nn.utils.rnn import pad_sequence
from functools import partial
import datetime
import gc
from model.meshAE import MeshAutoencoder, DiagonalGaussianDistribution
from data.dataset import ShapeNetCore_obj
from meshgpt_pytorch.meshgpt_pytorch import undiscretize
#################################################################################
#                             Training Helper Functions                         #
#################################################################################
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def first(it):
    return it[0]

def custom_collate(data, pad_id = -1):
    is_dict = isinstance(first(data), dict)

    if is_dict:
        keys = first(data).keys()
        data = [d.values() for d in data]

    output = []

    for datum in zip(*data):
        if torch.is_tensor(first(datum)):
            datum = pad_sequence(datum, batch_first = True, padding_value = pad_id)
            output.append(datum)
        else:
            datum = list(datum)
            output.append(datum)

    output = tuple(output)

    if is_dict:
        output = dict(zip(keys, output))

    return output

def save_as_obj(faces: torch.Tensor, filename: str, idx: int):
    """将重建的面片保存为OBJ文件"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    faces = faces[idx]  # [nf, 3, 3]
    valid_mask = ~torch.isnan(faces).any(dim=-1).any(dim=-1)
    faces = faces[valid_mask]  # [valid_nf, 3, 3]
    
    vertices = faces.reshape(-1, 3)  # [nf*3, 3]
    unique_verts, indices = torch.unique(vertices, dim=0, return_inverse=True)
    face_indices = indices.reshape(-1, 3) + 1
    
    with open(filename, 'w') as f:
        for v in unique_verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in face_indices:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=1800),) #
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    print(rank)
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # objectName = 'table_under100kb' # 5000tables
    objectName = args.obj_name
    
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"/root/github_code/pivotmesh/{args.results_dir}/{objectName}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        show_dir = f'/root/github_code/pivotmesh/results/{objectName}-{model_string_name}'
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(show_dir, exist_ok=True)
        logger = create_logger(show_dir)
        logger.info(f"Experiment directory created at {show_dir}")
    else:
        logger = create_logger(None)

    # Load config
    # with open("configs/AE.yaml", "r") as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)

    # Create model:
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
    ).to(device)

    
    checkpoint = torch.load('/root/github_code/pivotmesh/checkpoints/AE-shapenet_10000obj_under800_8channel_12en_768_18de_384_1e-4kl_1GCN_womask_ordered/mesh-autoencoder.ckpt.43.pt')
    autoencoder.load_state_dict(checkpoint['model'])
    autoencoder.requires_grad = False
    autoencoder.eval()

    # autoencoder = DDP(autoencoder, device_ids=[device])
    # Setup data:
   
    latent_size = 800 #######################
    model = DiT_models[args.model](
        input_tokens=latent_size,
        in_channels=8,
    )
    if args.from_pretrained and rank == 0:
        checkpoint = torch.load(args.from_pretrained, weights_only=False)
        model.load_state_dict(checkpoint['model'])

    model = DDP(model.to(device), device_ids=[device])

    dist.barrier()
    
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    if args.from_pretrained and rank == 0:
        opt.load_state_dict(checkpoint['opt'])

    dist.barrier()


    #############################################################
    # load data
    #############################################################    
    dataset = ShapeNetCore_obj(
        data_dir=args.data_path,
        return_pivot=False,
        augment=False,
        quant_bit=7,
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=True,
        # sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
        collate_fn = partial(custom_collate, pad_id = args.pad_id),
    )
    logger.info(f"Dataset contains {len(dataset):,} datas ({args.data_path})")

    # Variables for monitoring/logging purposes:
    autoencoder.eval()
    train_steps = 650000
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        # sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")

        for data in loader:
            with torch.no_grad():
                vertices = data['vertices'].to(device)
                faces = data['faces'].to(device)
                
                encoded_feature, _ = autoencoder.encode(vertices=vertices, faces=faces, return_face_coordinates=True,)
                B, N, C = encoded_feature.shape
                x, _ = autoencoder.reparameterize(encoded_feature)

                face_mask = reduce(faces != autoencoder.pad_id, 'b nf c -> b nf', 'all')
                face_num = face_mask.long().sum(dim=-1)
                # print(face_num)
                # exit()
                pos_indices = torch.arange(N, device=device)  # [0, 1, ..., N-1]
                grid = pos_indices.unsqueeze(0).repeat(B, 1)  # (B, N)

                x = x.to(device).permute(0, 2, 1)

                face_mask_patch = rearrange(face_mask, 'b (np p) -> b np p', p=autoencoder.patch_size).float().mean(dim=-1).bool()
                #############
                # x = x.permute(0, 2, 1)
                # x_logits = autoencoder.decode(x, face_mask_patch)
                # pred_face_coords = autoencoder.to_coor_logits(x_logits)

                # recon_faces = undiscretize(
                #     pred_face_coords.argmax(dim = -1),
                #     num_discrete = autoencoder.num_discrete_coors,
                #     continuous_range = autoencoder.coor_continuous_range,
                # )
                # recon_faces = rearrange(recon_faces, 'b nf (nv c) -> b nf nv c', nv = 3)
                # face_mask = rearrange(face_mask, 'b nf -> b nf 1 1')
                # recon_faces = recon_faces.masked_fill(~face_mask, float('nan'))
                # face_mask = rearrange(face_mask, 'b nf 1 1 -> b nf')
                # print(recon_faces.shape)
                
                # for idx in range(recon_faces.shape[0]):
                #     output_filename = os.path.join("dit_test", f'recon_{idx}-th.obj')
                #     save_as_obj(recon_faces, output_filename, idx=idx)

                # exit()
                ###############

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(mask=face_mask, grid=grid.long(), seq_len=face_num)
            
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            # #update_ema(ema, model.module)
            
            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

                dist.barrier()    

                if rank == 0:        
                    model.eval()  # important! This disables randomized embedding dropout
                    with torch.no_grad():
                        n = 10
                    
                        #####################################################
                        ####### test random
                        z = torch.randn(n, 8, latent_size, device=device)
                        mask = torch.ones(n, latent_size).to(device=device)

                        face_num = torch.randint(low=50, high=latent_size, size=(n,))
                        face_num = torch.tensor(face_num, dtype=torch.int).to(device=device)

                        face_mask_patch = rearrange(mask, 'b (np p) -> b np p', p=autoencoder.patch_size).float().mean(dim=-1).bool()

                        pos_indices = torch.arange(latent_size, device=device)  # [0, 1, ..., N-1]
                        grid = torch.stack([
                            pos_indices,
                            torch.zeros_like(pos_indices)
                        ], dim=0).unsqueeze(0).repeat(latent_size, 1, 1)

                        ####### test on train
                        # vertices = data['vertices'].to(device)
                        # faces = data['faces'].to(device)
                        # B, N, C = faces.shape
                        # z = torch.randn(B, 8, N, device=device)

                        # mask = reduce(faces != autoencoder.pad_id, 'b nf c -> b nf', 'all')
                        # face_num = mask.long().sum(dim=-1)
                        
                        # pos_indices = torch.arange(N, device=device)  # [0, 1, ..., N-1]
                        # grid = pos_indices.unsqueeze(0).repeat(B, 1)  # (B, N)

                        ######## model_kwargs
                        # short_ann_val = 'A table with legs.'
                        model_kwargs = dict(mask=mask, grid=grid.long(), seq_len=face_num)
                        #####################################################
                    
                        samples = diffusion.p_sample_loop(
                            model.module.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device,
                        )

                        samples = samples.permute(0, 2, 1)

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

                        for idx in range(recon_faces.shape[0]):
                            output_filename = os.path.join(f"{experiment_dir}", f'{train_steps}steps',f'recon_{idx}-th.obj')
                            save_as_obj(recon_faces, output_filename, idx=idx)
                
                dist.barrier()

        gc.collect()

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=False, default=None)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-MIN/3")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5000000)
    parser.add_argument("--global-batch-size", type=int, default=64 * 4) # 6 is gpu num
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--obj-name", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=False, default='')
    parser.add_argument("--from-pretrained", type=str, required=False, default=None)
    parser.add_argument("--pad-id", type=float, required=False, default=-1.)
    args = parser.parse_args()
    main(args)
