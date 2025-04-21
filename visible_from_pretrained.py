import os
import torch
import yaml
from model.meshAE import MeshAutoencoder
from data.dataset import ShapeNetCore_obj  # 根据实际数据集调整
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import re
from pathlib import Path
import pickle


# 1. 加载预训练权重
pretrained_path = "/root/github_code/pivotmesh/checkpoints/AE-shapenet_14400*10_8channel_12en_768_18de_384_1e-4kl_1GCN_mask/mesh-autoencoder.ckpt.33.pt"  # 替换为实际路径
checkpoint = torch.load(pretrained_path, map_location='cpu')

# 加载配置文件
with open("configs/AE.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# config = pickle.loads(checkpoint['config'])
# print(config)
# exit()

# 2. 初始化模型
autoencoder = MeshAutoencoder(
    num_discrete_coors=config['num_discrete_coors'],
    encoder_dim=config['encoder_dim'],
    encoder_depth=config['encoder_depth'],
    patch_size=config['patch_size'],
    dropout=config['dropout'],
    bin_smooth_blur_sigma=config['bin_smooth_blur_sigma'],
    quant_face=config['quant_face'],
    decoder_fine_dim=config['decoder_fine_dim'],
    graph_layers=config['graph_layers'],

    latent_dim = config['latent_dim'],
    pos_embed = config['pos_embed'],

    decoder_depth = config['decoder_depth'],
    decoder_dim = config['decoder_dim'],
    kl_regularization = config['kl_regularization'],
)


if os.path.exists(pretrained_path):
    autoencoder.load_state_dict(checkpoint['model'])
    print(f"Loaded pretrained model from {pretrained_path}")
else:
    raise FileNotFoundError(f"Pretrained model not found at {pretrained_path}")

# 3. 准备数据集
dataset_path =f"/root/github_code/TripoSF/mesh" # _10_augment
# dataset_path =f"/root/mesh_compress/data/train/train_mesh_data_under_25kb_100"
dataset = ShapeNetCore_obj(
    dataset_path,  # 验证集路径
    return_pivot=False,
    augment=False,  # 推理时关闭数据增强
    quant_bit=config['quant_bit']
)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # batch_size=1便于逐个可视化

# 4. 定义可视化保存函数
def save_as_obj(faces: torch.Tensor, filename: str):
    """将重建的面片保存为OBJ文件"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    faces = faces[0]  # [nf, 3, 3]
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

def plot_3d_mesh(vertices, faces):
    """简易3D可视化"""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制面片
    ax.plot_trisurf(
        vertices[:, 0], vertices[:, 1], vertices[:, 2],
        triangles=faces, alpha=0.8, edgecolor='k'
    )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# 5. 推理循环
autoencoder.eval()
output_dir = "visualization_results"
os.makedirs(output_dir, exist_ok=True)

with torch.no_grad():
    for idx, batch in enumerate(dataloader):
        # 前向推理
        vertices = batch['vertices']  # [1, nv, 3]
        faces = batch['faces']        # [1, nf, 3]
        
        # 获取重建结果
        # print(faces)
        recon_face, _ = autoencoder(vertices=vertices, faces=faces, return_recon_faces=True)  # 假设模型返回重建面片
        # print(recon_face)
        
        # output_filename = f'{output_dir}/reconstructed_{idx}.obj'
        # Shorten the weight name (keep AE prefix and key parameters)
        weight_short = re.sub(r'AE-shapenet_(\d+)\*(\d+)_(\d+)channel_(\d+)depth', 
                            r'AE-\1x\2_c\3_d\4', 
                            Path(pretrained_path).parent.name)

        # Shorten the dataset name (keep ShapeNet prefix and key parameters)
        dataset_short = re.sub(r'ShapeNetCorev2_obj_(\d+)faces_(\d+)_(\d+)', 
                            r'SN_\1f_\2_\3', 
                            Path(dataset_path).name)

        # Combine them
        output_filename = os.path.join("visualization_results", f"{weight_short}_{dataset_short}", f'recon_{idx}-th.obj')
        os.makedirs(output_dir, exist_ok=True)
        
        save_as_obj(recon_face, output_filename)
        print(f'save {idx}-th obj!')
        
        
print("Inference and visualization completed!")