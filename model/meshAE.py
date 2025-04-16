from functools import partial
from math import pi
import torch
from torch import nn, Tensor
from torch.nn import Module
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast
from torchtyping import TensorType
from pytorch_custom_utils import save_load
from beartype import beartype
from beartype.typing import Tuple, Optional
from einops import rearrange, repeat, reduce, pack
from einops.layers.torch import Rearrange
from x_transformers import Encoder
from x_transformers.x_transformers import AbsolutePositionalEmbedding
from meshgpt_pytorch.data import derive_face_edges_from_faces
from meshgpt_pytorch.meshgpt_pytorch import (
    discretize, get_derived_face_features,
    scatter_mean, default, pad_at_dim, exists, gaussian_blur_1d,
    undiscretize, first,
)
from torch_geometric.nn.conv import SAGEConv
import torch.nn.functional as F

import numpy as np

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False, mask=None):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        self.mask = mask
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        # x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        noise = torch.randn(self.mean.shape).to(device=self.parameters.device)
        x = self.mean + self.std * noise
        if self.mask is not None:
            x = x * self.mask.unsqueeze(-1)  # broadcast over latent dim
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    # def kl(self, other=None):
    #     if self.deterministic:
    #         return torch.tensor([0.], device=self.parameters.device)
    #     if other is None:
    #         kl = 0.5 * torch.sum(
    #             self.mean.pow(2) + self.var - 1.0 - self.logvar,
    #             dim=-1  # per latent dim
    #         )
    #     else:
    #         kl = 0.5 * torch.sum(
    #             (self.mean - other.mean).pow(2) / other.var +
    #             self.var / other.var - 1.0 -
    #             self.logvar + other.logvar,
    #             dim=-1
    #         )

    #     if self.mask is not None:
    #         kl = kl * self.mask  # shape: [B, N]
    #     return kl.sum(dim=-1)  # sum over N


    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    # def nll(self, sample, dims=[1, 2]):
    #     if self.deterministic:
    #         return torch.tensor([0.], device=self.parameters.device)

    #     logtwopi = np.log(2.0 * np.pi)
    #     nll = 0.5 * (logtwopi + self.logvar + (sample - self.mean).pow(2) / self.var)
    #     nll = nll.sum(dim=-1)  # sum over latent dim

    #     if self.mask is not None:
    #         nll = nll * self.mask
    #     return nll.sum(dim=-1)  # sum over N


    def mode(self):
        return self.mean

    # def mode(self):
    #     if self.mask is not None:
    #         return self.mean * self.mask.unsqueeze(-1)  # mask shape [B, N] -> [B, N, 1]
    #     return self.mean

@save_load()
class MeshAutoencoder(Module):
    @beartype
    def __init__(
        self,
        encoder_depth = 12,
        decoder_depth = 18,
        encoder_heads = 8,
        decoder_heads = 8,
        encoder_dim = 768,
        decoder_dim = 384,
        decoder_fine_dim = 192,
        num_discrete_coors = 128,
        coor_continuous_range: Tuple[float, float] = (-1., 1.),
        dim_coor_embed = 64,
        num_discrete_area = 128,
        dim_area_embed = 16,
        num_discrete_normals = 128,
        dim_normal_embed = 64,
        num_discrete_angle = 128,
        dim_angle_embed = 16,
        
        kl_loss_weight = 1e-4,
        bin_smooth_blur_sigma = 0.4,  # they blur the one hot discretized coordinate positions
        pad_id = -1,
        checkpoint_quantizer = False,
        patch_size = 1,
        dropout = 0.,
        quant_face = False,
        graph_layers = 1,

        latent_dim = 8,
        kl_regularization = True,
        pos_embed=True,
    ):
        super().__init__()

        # main face coordinate embedding

        self.num_discrete_coors = num_discrete_coors
        self.coor_continuous_range = coor_continuous_range

        self.discretize_face_coords = partial(discretize, num_discrete = num_discrete_coors, continuous_range = coor_continuous_range)
        self.coor_embed = nn.Embedding(num_discrete_coors, dim_coor_embed)

        # derived feature embedding

        self.discretize_angle = partial(discretize, num_discrete = num_discrete_angle, continuous_range = (0., pi))
        self.angle_embed = nn.Embedding(num_discrete_angle, dim_angle_embed)

        lo, hi = coor_continuous_range
        self.discretize_area = partial(discretize, num_discrete = num_discrete_area, continuous_range = (0., (hi - lo) ** 2))
        self.area_embed = nn.Embedding(num_discrete_area, dim_area_embed)

        self.discretize_normals = partial(discretize, num_discrete = num_discrete_normals, continuous_range = coor_continuous_range)
        self.normal_embed = nn.Embedding(num_discrete_normals, dim_normal_embed)

        # initial dimension

        init_dim = dim_coor_embed * 9 + dim_angle_embed * 3 + dim_normal_embed * 3 + dim_area_embed
        
        # patchify related
        
        self.patch_size = patch_size
        self.project_in = nn.Linear(init_dim * patch_size, encoder_dim)
        
        # graph init
        self.graph_layers = graph_layers
        if self.graph_layers > 0:
            sageconv_kwargs = dict(
                normalize = True,
                project = True,
            )
            self.init_sage_conv = SAGEConv(encoder_dim, encoder_dim, **sageconv_kwargs)
            self.init_encoder_act_and_norm = nn.Sequential(
                nn.SiLU(),
                nn.LayerNorm(encoder_dim)
            )
            self.graph_encoders = nn.ModuleList([])

        for _ in range(graph_layers - 1):
            sage_conv = SAGEConv(
                encoder_dim,
                encoder_dim,
                **sageconv_kwargs
            )
            self.graph_encoders.append(sage_conv)
        
        # transformer encoder
        # NOTE: currently no positional embedding
        
        self.encoder = Encoder(
            dim = encoder_dim,
            depth = encoder_depth,
            heads = encoder_heads,
            attn_flash = True,
            attn_dropout = dropout,
            ff_dropout = dropout,
        )


        self.sim_quant = quant_face
    
        self.checkpoint_quantizer = checkpoint_quantizer # whether to memory checkpoint the quantizer

        self.pad_id = pad_id # for variable lengthed faces, padding quantized ids will be set to this value

        # decoder
        
        self.init_decoder = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim),
            nn.SiLU(),
            nn.LayerNorm(decoder_dim),
        )

        self.decoder_coarse = Encoder(
            dim = decoder_dim,
            depth = decoder_depth // 2,
            heads = decoder_heads,
            attn_flash = True,
            attn_dropout = dropout,
            ff_dropout = dropout,
        )
        
        self.coarse_to_fine = nn.Linear(decoder_dim, decoder_fine_dim * 3)

        self.decoder_fine = Encoder(
            dim = decoder_fine_dim,
            depth = decoder_depth // 2,
            heads = decoder_heads,
            attn_flash = True,
            attn_dropout = dropout,
            ff_dropout = dropout,
        )

        self.to_coor_logits = nn.Sequential(
            nn.Linear(decoder_fine_dim, patch_size * num_discrete_coors * 3),
            Rearrange('b (nf nv) (v c) -> b nf (nv v) c', nv = 3, v = 3)
            # Rearrange('b np (p v c) -> b (np p) v c', v = 9, p = patch_size)
        )

        # loss related

        self.kl_loss_weight = kl_loss_weight
        self.bin_smooth_blur_sigma = bin_smooth_blur_sigma

        self.to_latent = nn.Sequential(
            nn.Linear(encoder_dim, 2 * latent_dim),  # 输出mean和logvar
            Rearrange('b n (d k) -> b n d k', k=2)    # [B, N, D, 2]
        )
        self.latent_dim = latent_dim
        self.kl_regularization = kl_regularization

        self.latent_proj = nn.Linear(latent_dim, decoder_dim)

        self.pos_embed = pos_embed

    def reparameterize(self, x, mask=None):
        if mask is not None:
            params = self.to_latent(x) * mask.unsqueeze(-1).unsqueeze(-1)  # [B, N, D, 2]
        else:
            params = self.to_latent(x)


        mean, logvar = params.unbind(dim=-1)  # 拆分为mean和logvar
        
        if self.kl_regularization:
            posterior = DiagonalGaussianDistribution(
                torch.cat([mean, logvar], dim=-1),
                mask=mask  # shape [B, N]
            )
            latent = posterior.sample()
            kl = posterior.kl()
        else:
            latent = mean
            kl = None
        
        return latent, kl

    @beartype
    def encode(
        self,
        *,
        vertices:         TensorType['b', 'nv', 3, float],
        faces:            TensorType['b', 'nf', 3, int],
        # face_edges:       TensorType['b', 'e', 2, int],
        # face_mask:        TensorType['b', 'nf', bool],
        # face_edges_mask:  TensorType['b', 'e', bool],
        face_edges: Optional[TensorType['b', 'e', 2, int]] = None,
        face_mask: Optional[TensorType['b', 'nf', bool]] = None,
        face_edges_mask: Optional[TensorType['b', 'e', bool]] = None,
        return_face_coordinates = False
    ):
        """
        einops:
        b - batch
        nf - number of faces
        nv - number of vertices (3)
        c - coordinates (3)
        d - embed dim
        """
        if face_edges is None:
            face_edges = derive_face_edges_from_faces(faces, pad_id=self.pad_id)
        if face_mask is None:
            face_mask = reduce(faces != self.pad_id, 'b nf c -> b nf', 'all')
        if face_edges_mask is None:
            face_edges_mask = reduce(face_edges != self.pad_id, 'b e ij -> b e', 'all')
        
        batch, num_vertices, num_coors, device = *vertices.shape, vertices.device
        _, num_faces, _ = faces.shape

        face_without_pad = faces.masked_fill(~rearrange(face_mask, 'b nf -> b nf 1'), 0)

        faces_vertices = repeat(face_without_pad, 'b nf nv -> b nf nv c', c = num_coors)
        vertices = repeat(vertices, 'b nv c -> b nf nv c', nf = num_faces)

        # continuous face coords

        face_coords = vertices.gather(-2, faces_vertices)

        # compute derived features and embed

        derived_features = get_derived_face_features(face_coords)

        discrete_angle = self.discretize_angle(derived_features['angles'])
        angle_embed = self.angle_embed(discrete_angle)

        discrete_area = self.discretize_area(derived_features['area'])
        area_embed = self.area_embed(discrete_area)

        discrete_normal = self.discretize_normals(derived_features['normals'])
        normal_embed = self.normal_embed(discrete_normal)

        # discretize vertices for face coordinate embedding

        discrete_face_coords = self.discretize_face_coords(face_coords)
        discrete_face_coords = rearrange(discrete_face_coords, 'b nf nv c -> b nf (nv c)') # 9 coordinates per face

        face_coor_embed = self.coor_embed(discrete_face_coords)
        face_coor_embed = rearrange(face_coor_embed, 'b nf c d -> b nf (c d)')

        # combine all features and project into model dimension

        face_embed, _ = pack([face_coor_embed, angle_embed, area_embed, normal_embed], 'b nf *')
        
        # patchify faces: [b nf d] -> [b nf//patch_size d]
        
        face_embed = rearrange(face_embed, 'b (num_patch patch_size) d -> b num_patch (patch_size d)', 
                               patch_size = self.patch_size)
        face_embed = self.project_in(face_embed)
        
        face_mask = rearrange(face_mask, 'b (num_patch patch_size) -> b num_patch patch_size', 
                              patch_size = self.patch_size).float().mean(dim=-1).bool()
        
        # face_embed = face_embed * face_mask.unsqueeze(-1)

        # init graph 
        
        if self.graph_layers > 0:
            orig_face_embed_shape = face_embed.shape[:2]
            face_embed = face_embed[face_mask]
            
            face_index_offsets = reduce(face_mask.long(), 'b nf -> b', 'sum')
            face_index_offsets = F.pad(face_index_offsets.cumsum(dim = 0), (1, -1), value = 0)
            face_index_offsets = rearrange(face_index_offsets, 'b -> b 1 1')
            
            face_edges = face_edges + face_index_offsets
            face_edges = face_edges[face_edges_mask]
            face_edges = rearrange(face_edges, 'be ij -> ij be')
            
            
            face_embed = self.init_sage_conv(face_embed, face_edges)
            face_embed = self.init_encoder_act_and_norm(face_embed)

            for conv in self.graph_encoders:
                face_embed = conv(face_embed, face_edges)
            
            shape = (*orig_face_embed_shape, face_embed.shape[-1])
            face_embed = face_embed.new_zeros(shape).masked_scatter(rearrange(face_mask, '... -> ... 1'), face_embed)  
        
        # encode face embeddings
        if self.pos_embed:
            seq_len = face_embed.shape[1]  # Get sequence length from input
            pos_emb = AbsolutePositionalEmbedding(
                dim=face_embed.shape[-1],     # Feature dimension
                max_seq_len=seq_len           # Must specify maximum sequence length
            ).to(face_embed.device)

            # Add positional embeddings to face_embed
            face_embed = face_embed + pos_emb(face_embed)  # shape remains (batch_size, seq_len, dim)
            face_embed = self.encoder(face_embed, mask=face_mask) * face_mask.unsqueeze(-1) 
        else:
            face_embed = self.encoder(face_embed, mask=face_mask)
            # face_embed = face_embed * face_mask.unsqueeze(-1) 
            
        
        if not return_face_coordinates:
            return face_embed

        return face_embed, discrete_face_coords

    @beartype
    def decode(
        self,
        quantized: TensorType['b', 'n', 'd', float],
        face_mask:  TensorType['b', 'n', bool]
    ):
        # if self.kl_regularization:
        quantized = self.latent_proj(quantized)

        conv_face_mask = rearrange(face_mask, 'b n -> b n 1')
        vertice_mask = repeat(face_mask, 'b nf -> b (nf nv)', nv = 3)
        x = quantized
        x = x.masked_fill(~conv_face_mask, 0.)

        x = self.init_decoder(x)
        # x = self.decoder(x)
        x = self.decoder_coarse(x, mask = face_mask)
        x = self.coarse_to_fine(x)
        x = rearrange(x, 'b nf (nv d) -> b (nf nv) d', nv=3)
        x = self.decoder_fine(x, mask = vertice_mask) # * vertice_mask.unsqueeze(-1)
        return x

    @beartype
    def forward(
        self,
        *,
        vertices:       TensorType['b', 'nv', 3, float],
        faces:          TensorType['b', 'nf', 3, int],
        face_edges:     Optional[TensorType['b', 'e', 2, int]] = None,
        pivot_mask:     Optional[TensorType['b', 'nv', bool]] = None,
        return_loss_breakdown = False,
        return_recon_faces = False,
        only_return_recon_faces = False,
    ):
        if not exists(face_edges):
            face_edges = derive_face_edges_from_faces(faces, pad_id = self.pad_id)

        num_faces, num_face_edges, device = faces.shape[1], face_edges.shape[1], faces.device

        face_mask = reduce(faces != self.pad_id, 'b nf c -> b nf', 'all')
        face_mask_patch = rearrange(face_mask, 'b (np p) -> b np p', p=self.patch_size).float().mean(dim=-1).bool()
        face_edges_mask = reduce(face_edges != self.pad_id, 'b e ij -> b e', 'all')

        encoded, face_coordinates = self.encode(
            vertices = vertices,
            faces = faces,
            face_edges = face_edges,
            face_edges_mask = face_edges_mask,
            face_mask = face_mask,
            return_face_coordinates = True,
        )
        
        # quantized = encoded
        # gaussian = DiagonalGaussianDistribution(quantized)
        # kl_loss = gaussian.kl().mean()

        # print('encoded', encoded.shape)

        
        latent, kl_loss = self.reparameterize(encoded, mask=None) 
        
        # latent = encoded

        # print('latent', latent.shape)
        # make sure the right data type    
        latent = latent.to(encoded.dtype)
        
        decode = self.decode(
            latent,
            face_mask = face_mask_patch
        )
        pred_face_coords = self.to_coor_logits(decode)

        # compute reconstructed faces if needed

        if return_recon_faces or only_return_recon_faces:

            recon_faces = undiscretize(
                pred_face_coords.argmax(dim = -1),
                num_discrete = self.num_discrete_coors,
                continuous_range = self.coor_continuous_range,
            )

            recon_faces = rearrange(recon_faces, 'b nf (nv c) -> b nf nv c', nv = 3)
            face_mask = rearrange(face_mask, 'b nf -> b nf 1 1')
            recon_faces = recon_faces.masked_fill(~face_mask, float('nan'))
            face_mask = rearrange(face_mask, 'b nf 1 1 -> b nf')

        if only_return_recon_faces:
            return recon_faces, pred_face_coords, face_coordinates, face_mask

        # prepare for recon loss

        pred_face_coords = rearrange(pred_face_coords, 'b ... c -> b c (...)')
        face_coordinates = rearrange(face_coordinates, 'b ... -> b 1 (...)')

        # reconstruction loss on discretized coordinates on each face
        # they also smooth (blur) the one hot positions, localized label smoothing basically

        with autocast(enabled = False):
            pred_log_prob = pred_face_coords.log_softmax(dim = 1)

            target_one_hot = torch.zeros_like(pred_log_prob).scatter(1, face_coordinates, 1.)

            if self.bin_smooth_blur_sigma >= 0.:
                target_one_hot = gaussian_blur_1d(target_one_hot, sigma = self.bin_smooth_blur_sigma)

            # cross entropy with localized smoothing

            recon_losses = (-target_one_hot * pred_log_prob).sum(dim = 1)

            face_mask = repeat(face_mask, 'b nf -> b (nf r)', r = 9)
            recon_loss = recon_losses[face_mask].mean()

        # calculate total loss
        total_loss = recon_loss + kl_loss.mean() * self.kl_loss_weight if self.kl_regularization else recon_loss
        
        # calculate loss breakdown if needed

        loss_breakdown = (recon_loss, kl_loss) if self.kl_regularization else (recon_loss, torch.tensor(0.0, device=recon_loss.device))

        # some return logic

        if not return_loss_breakdown:
            if not return_recon_faces:
                return total_loss

            return recon_faces, total_loss

        if not return_recon_faces:
            return total_loss, loss_breakdown

        return recon_faces, total_loss, loss_breakdown
    
    
    
if __name__ == '__main__':
    # init model
    meshAE = MeshAutoencoder()
    
    # mock input
    vertices = torch.randn(4, 99, 3)
    faces = torch.arange(99).reshape(33, 3)
    faces = repeat(faces, 'f c -> b f c', b=4)
    
    meshAE(vertices=vertices, faces=faces)