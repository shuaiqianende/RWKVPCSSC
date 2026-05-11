from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional
from knn_cuda import KNN
from pointnet2_ops.pointnet2_utils import (
    furthest_point_sample,
    gather_operation,
    grouping_operation,
)
from torch import nn, einsum
import math
import spconv.pytorch as spconv
import random

from .serialization import Point
from src.models.components.kernels.utils.gpu_neigbors import knn
from src.models.components.kpnext_blocks import KPConvD, KPConvX, KPNextBlock, _KPNextBlock, KPNextResidualBlock
from src.models.components.prwkv import Block as RWKVBlock
from src.models.components.prwkv import DropPath as DropPath

def check_tensor(tensor, name="tensor"):
    if torch.isnan(tensor).any():
        raise ValueError(f"{name} contains NaN values with shape {tuple(tensor.shape)}.")
    if torch.isinf(tensor).any():
        raise ValueError(f"{name} contains Inf values with shape {tuple(tensor.shape)}.")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} is not contiguous.")


def sample_and_group_knn(xyz, points, npoint, k, use_xyz=True, idx=None):
    """
    Args:
        xyz: Tensor, (B, 3, N)
        points: Tensor, (B, f, N)
        npoint: int
        k: int
        use_xyz: boolean
        idx: Tensor, (B, npoint, nsample)

    Returns:
        new_xyz: Tensor, (B, 3, npoint)
        new_points: Tensor, (B, 3 | f+3 | f, npoint, nsample)
        idx_local: Tensor, (B, npoint, nsample)
        grouped_xyz: Tensor, (B, 3, npoint, nsample)

    """
    xyz_flipped = xyz.permute(0, 2, 1).contiguous()  # (B, N, 3)
    if xyz.shape[-1] == npoint:
        new_xyz = xyz
    else:
        new_xyz = gather_operation(
            xyz, furthest_point_sample(xyz_flipped, npoint)
        )  # (B, 3, npoint)
    if idx is None:
        _, idx = KNN(k, transpose_mode=True)(
            xyz_flipped, new_xyz.permute(0, 2, 1).contiguous()
        )
        idx = idx.int()
        # idx = query_knn(k, xyz_flipped, new_xyz.permute(0, 2, 1).contiguous())
    grouped_xyz = grouping_operation(xyz, idx)  # (B, 3, npoint, nsample)
    grouped_xyz -= new_xyz.unsqueeze(3).repeat(1, 1, 1, k)

    if points is not None:
        grouped_points = grouping_operation(points, idx)  # (B, f, npoint, nsample)
        if use_xyz:
            new_points = torch.cat([grouped_xyz, grouped_points], 1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):
    """
    Args:
        xyz: Tensor, (B, 3, nsample)
        points: Tensor, (B, f, nsample)
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, 1)
        new_points: Tensor, (B, f|f+3|3, 1, nsample)
        idx: Tensor, (B, 1, nsample)
        grouped_xyz: Tensor, (B, 3, 1, nsample)
    """
    b, _, nsample = xyz.shape
    device = xyz.device
    # new_xyz = torch.mean(xyz, dim=1)
    new_xyz = torch.zeros((1, 3, 1), dtype=torch.float, device=device).repeat(b, 1, 1)
    grouped_xyz = xyz.reshape((b, 3, 1, nsample))
    idx = torch.arange(nsample, device=device).reshape(1, 1, nsample).repeat(b, 1, 1)
    if points is not None:
        if use_xyz:
            new_points = torch.cat([xyz, points], 1)
        else:
            new_points = points
        new_points = new_points.unsqueeze(2)
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        _, idx = KNN(k=k, transpose_mode=True)(
            x.transpose(1, 2), x.transpose(1, 2)
        )  # (batch_size, num_points, k)
    device = torch.device("cuda")

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = (
        idx + idx_base
    )  # batch_size * num_points * k + range(0, batch_size)*num_points

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()

    feature = x.view(batch_size * num_points, -1)[
        idx, :
    ]  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature

class MlpRes(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        super(MlpRes, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (B, out_dim, n)
        """
        shortcut = self.conv_shortcut(x)
        out = self.conv_2(torch.relu(self.conv_1(x))) + shortcut
        return out


class MlpConv(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MlpConv, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=(1, 1),
        stride=(1, 1),
        if_bn=True,
        activation_fn: Optional[Callable] = torch.relu,
    ):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, x):
        out = self.conv(x)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out


class PointNetSaModuleKNN(nn.Module):
    def __init__(
        self,
        npoint,
        nsample,
        in_channel,
        mlp,
        if_bn=True,
        group_all=False,
        use_xyz=True,
        if_idx=False,
    ):
        """
        Args:
            npoint: int, number of points to sample
            nsample: int, number of points in each local region
            in_channel: int, input channel of features(points)
            mlp: list of int,
        """
        super(PointNetSaModuleKNN, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp = mlp
        self.group_all = group_all
        self.use_xyz = use_xyz
        self.if_idx = if_idx
        if use_xyz:
            in_channel += 3

        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp[:-1]:
            self.mlp_conv.append(Conv2d(last_channel, out_channel, if_bn=if_bn))
            last_channel = out_channel
        self.mlp_conv.append(
            Conv2d(last_channel, mlp[-1], if_bn=False, activation_fn=None)
        )
        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    def forward(self, xyz, points, idx=None):
        """
        Args:
            xyz: Tensor, (B, 3, N)
            points: Tensor, (B, f, N)
            idx: Tensor, (B, npoint, nsample)

        Returns:
            new_xyz: Tensor, (B, 3, npoint)
            new_points: Tensor, (B, mlp[-1], npoint)
        """
        if self.group_all:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(
                xyz, points, self.use_xyz
            )
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_knn(
                xyz, points, self.npoint, self.nsample, self.use_xyz, idx=idx
            )

        new_points = self.mlp_conv(new_points)
        new_points = torch.max(new_points, 3)[0]

        if self.if_idx:
            return new_xyz, new_points, idx
        else:
            return new_xyz, new_points

class RWKVFormer(nn.Module):
    def __init__(
        self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4
    ):
        super(RWKVFormer, self).__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(dim, dim, 1)
        self.conv_query = nn.Conv1d(dim, dim, 1)
        self.conv_value = nn.Conv1d(dim, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1),
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1),
        )

        self.linear_start = nn.Conv1d(in_channel, dim, 1)
        self.linear_end = nn.Conv1d(dim, in_channel, 1)
        self.rwkv_block = RWKVBlock(n_embd=64, n_layer=8, layer_id=0, init_mode='fancy', drop_path=0)

        self.query_knn = KNN(k=n_knn, transpose_mode=True)

    def forward(self, x, pos):
        """feed forward of transformer
        Args:
            x: Tensor of features, (B, in_channel, n)
            pos: Tensor of positions, (B, 3, n)

        Returns:
            y: Tensor of features with attention, (B, in_channel, n)
        """

        identity = x

        x = self.linear_start(x)
        b, dim, n = x.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous()
        # idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped)
        _, idx_knn = self.query_knn(pos_flipped, pos_flipped)
        idx_knn = idx_knn.int()
        key = self.conv_key(x)
        value = self.conv_value(x)
        query = self.conv_query(x)

        key = grouping_operation(key, idx_knn)  # b, dim, n, n_knn
        qk_rel = query.reshape((b, -1, n, 1)) - key

        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(
            pos, idx_knn
        )  # b, 3, n, n_knn
        pos_embedding = self.pos_mlp(pos_rel)  # b, dim, n, n_knn

        attention = self.attn_mlp(qk_rel + pos_embedding)
        attention = torch.softmax(attention, -1)

        value = self.rwkv_block(value.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        value = grouping_operation(value, idx_knn) - value.reshape(b, -1, n, 1) + pos_embedding  # b, dim, n, n_knn

        agg = einsum("b c i j, b c i j -> b c i", attention, value)  # b, dim, n
        y = self.linear_end(agg)

        return y + identity


class Transformer(nn.Module):
    def __init__(
        self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4
    ):
        super(Transformer, self).__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(dim, dim, 1)
        self.conv_query = nn.Conv1d(dim, dim, 1)
        self.conv_value = nn.Conv1d(dim, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1),
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1),
        )

        self.linear_start = nn.Conv1d(in_channel, dim, 1)
        self.linear_end = nn.Conv1d(dim, in_channel, 1)

        self.query_knn = KNN(k=n_knn, transpose_mode=True)

    def forward(self, x, pos):
        """feed forward of transformer
        Args:
            x: Tensor of features, (B, in_channel, n)
            pos: Tensor of positions, (B, 3, n)

        Returns:
            y: Tensor of features with attention, (B, in_channel, n)
        """

        identity = x

        x = self.linear_start(x)
        b, dim, n = x.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous()
        # idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped)
        _, idx_knn = self.query_knn(pos_flipped, pos_flipped)
        idx_knn = idx_knn.int()
        key = self.conv_key(x)
        value = self.conv_value(x)
        query = self.conv_query(x)

        key = grouping_operation(key, idx_knn)  # b, dim, n, n_knn
        qk_rel = query.reshape((b, -1, n, 1)) - key

        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(
            pos, idx_knn
        )  # b, 3, n, n_knn
        pos_embedding = self.pos_mlp(pos_rel)  # b, dim, n, n_knn

        attention = self.attn_mlp(qk_rel + pos_embedding)
        attention = torch.softmax(attention, -1)

        value = value.reshape((b, -1, n, 1)) + pos_embedding

        agg = einsum("b c i j, b c i j -> b c i", attention, value)  # b, dim, n
        y = self.linear_end(agg)

        return y + identity


class RWKV_FeatureExtractor(nn.Module):
    def __init__(self, feat_channel=128, out_dim=1024, L=1):
        """Encoder that encodes information of partial point cloud"""
        super(RWKV_FeatureExtractor, self).__init__()
        self.feat_channel = feat_channel
        self.Length = L
        rwkv_downs = []
        for i in range(L):
            rwkv_downs.append(RWKVBlock(n_embd=128, n_layer=L if L > 1 else 8, layer_id=i, init_mode='fancy'))
        self.rwkv_downs = nn.ModuleList(rwkv_downs)
        self.sa_module_1 = PointNetSaModuleKNN(
            1024, 16, 128, [128], group_all=False, if_bn=False, if_idx=True,
        )
        self.sa_module_2 = PointNetSaModuleKNN(
            128, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True
        )
        self.sa_module_3 = PointNetSaModuleKNN(
            None, None, 256, [512, out_dim], group_all=True, if_bn=False
        )

    def forward(self, point_cloud, points):
        points = points.transpose(1, 2).contiguous()
        for i in range(self.Length):
            points = self.rwkv_downs[i](points, patch_resolution=None)
        points = points.permute(0, 2, 1).contiguous()
        l1_xyz, l1_points, idx1 = self.sa_module_1(
            point_cloud, points
        )  # (B, 3, 512), (B, 128, 512)
        l2_xyz, l2_points, idx2 = self.sa_module_2(
            l1_xyz, l1_points
        )  # (B, 3, 128), (B, 256, 512)
        l3_xyz, l3_points = self.sa_module_3(
            l2_xyz, l2_points
        )  # (B, 3, 1), (B, out_dim, 1)
        return l3_points


class SEG_HEAD(nn.Module):
    def __init__(self, i, up_factor=1, cls=16, hidden_dim=128, k=16, L=1, num=None, out_dim=1024, kp_radius=0.1):
        super(SEG_HEAD, self).__init__()
        self.i = i
        self.k = k
        self.out_dim = out_dim
        self.up_factor = up_factor
        self.mlp_delta = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, cls, kernel_size=1),
        )
        self.Length = L
        # self.subconv = CPE(in_dim=128, hidden_dim=64, out_dim=128, kernel_size=3, grid_size=grid_size)
        self.radius = kp_radius
        self.KP_D = KPNextBlock(128, 128, (1, 14, 28), self.radius, self.radius, 0)
        self.KP_D_2 = KPNextBlock(128, 128, (1, 14, 28), self.radius, self.radius, 0)

        rwkv_downs = []
        for i in range(self.Length):
            rwkv_downs.append(RWKVBlock(n_embd=hidden_dim, n_layer=8, layer_id=i, init_mode='fancy'))
            # rwkv_downs.append(RWKVBlock(n_embd=128, n_layer=L if L > 1 else 8, layer_id=i, init_mode='fancy'))
        self.rwkv_downs = nn.ModuleList(rwkv_downs)


    def forward(self, px, px2=None, last_label=None):
        p, x = px  # (n, 3), (n, c), (b)
        b, c, n = x.shape

        device = x.device
        offset = torch.arange(b, device=device).view(-1, 1) * n
        offset = offset.repeat(1, n).view(-1)
        xyz = p.transpose(1, 2).contiguous()
        points = x.transpose(1, 2).contiguous()
        idx = knn(xyz, xyz, self.k, distance_limit=self.radius**2)
        _xyz = xyz.view(-1, 3).contiguous()
        _points = points.view(-1, c).contiguous()
        _idx = idx.view(-1, self.k).contiguous()
        # print('_idx', _idx, self.i)
        _idx = _idx + offset.view(-1, 1)

        _points = self.KP_D(_xyz, _xyz, _points, _idx) + _points
        _points = self.KP_D_2(_xyz, _xyz, _points, _idx) + _points
        _points = _points.view(b, n, -1)
        points = _points + points

        for i in range(self.Length):
            points = self.rwkv_downs[i](points, patch_resolution=None)
        points = points.transpose(1, 2).contiguous()

        label = self.mlp_delta(points) # (B, c, N_prev * up_factor)
        return points, label, None


class SeedGenerator(nn.Module):
    def __init__(self, dim_feat=512, hidden_dim=128, up_factor=2, radius=1.0, num_p0=4096, cls=16, kp_radius=128, grid_size=0.01):
        super(SeedGenerator, self).__init__()
        self.rwkvpd = RWKV_PD(dim_feat=dim_feat, up_factor=up_factor, hidden_dim=hidden_dim, num=num_p0, radius=radius, id=0)
        self.seg = SEG_HEAD(i=-1, up_factor=up_factor, cls=cls, hidden_dim=hidden_dim, num=num_p0, kp_radius=kp_radius)
        self.num_p0 = num_p0
        self.grid_size = grid_size
        self.dim = hidden_dim
    def forward(self, xyz, feat, global_feat):
        pcd_coarse, k_prev_corse, _ = self.rwkvpd(
            xyz, global_feat, feat
        )  # (B, 3, N_prev * up_factor), (B, 128, N_prev * up_factor), (B, 512, 1)

        pcd_coarse = torch.cat([pcd_coarse, xyz], dim=2)
        k_prev_corse = torch.cat([k_prev_corse, feat], dim=2)
        pcd_and_feat = torch.cat([pcd_coarse, k_prev_corse], dim=1)
        pcd_and_feat_down = gather_operation(
            pcd_and_feat,
            furthest_point_sample(pcd_coarse.transpose(1, 2).contiguous(), self.num_p0),
        )  # (B, 3 + hidden_feat, num_pc)
        pcd_coarse = pcd_and_feat_down[:, :3, :].contiguous()
        k_prev_corse = pcd_and_feat_down[:, 3:(self.dim+3), :].contiguous()
        # k_prev = pcd_and_feat_down[:, (self.dim+3):, :].contiguous()

        pcd_coarse_tran = pcd_coarse.transpose(1, 2).contiguous()
        k_prev_corse_tran = k_prev_corse.transpose(1, 2).contiguous()
        order = serialization(pcd_coarse_tran, grid_size=self.grid_size)
        bs, n_p, _ = pcd_coarse_tran.size()
        pcd_coarse_tran = pcd_coarse_tran.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
        k_prev_corse_tran = k_prev_corse_tran.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
        pcd_coarse = pcd_coarse_tran.transpose(1, 2).contiguous()
        k_prev_corse = k_prev_corse_tran.transpose(1, 2).contiguous()

        k_prev, pcd_label, step_feat = self.seg(
            [pcd_coarse, k_prev_corse],
        )
        return pcd_coarse, k_prev_corse, pcd_label, k_prev, None

class CPE(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=64, out_dim=128, kernel_size=3, grid_size=128, use_res=False):
        super(CPE, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.mlp_1 = nn.Linear(in_dim, hidden_dim)
        self.cpe1 = spconv.SubMConv3d(hidden_dim, hidden_dim, kernel_size=kernel_size, bias=True, indice_key=None)
        self.cpe2 = spconv.SubMConv3d(hidden_dim, hidden_dim, kernel_size=kernel_size, bias=True, indice_key=None)
        # self.cpe3 = spconv.SubMConv3d(hidden_dim, hidden_dim, kernel_size=kernel_size, bias=True, indice_key=None)
        self.mlp_2 = nn.Linear(hidden_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.ReLU()
        self.mlp_3 = nn.Linear(out_dim, out_dim)
        self.grid_size = grid_size
        self.use_res = use_res

    def forward(self, p, x):
        b, c, n = x.shape
        xyz = p.transpose(1, 2).contiguous()
        points = x.transpose(1, 2).contiguous()
        hidden_points = self.mlp_1(points)
        normalized_p = (xyz.clone().detach().requires_grad_(True) + 1) / 2
        # normalized_p = (torch.tensor(xyz).clone().detach().requires_grad_(True) + 1) / 2
        indices = (normalized_p * (self.grid_size - 1)).long()  # Convert coordinates to integer voxel indices.
        indices_batch = indices.view(-1, 3).contiguous()  # [b * n, 3]
        batch = torch.arange(0, b).repeat_interleave(n).view(-1, 1).to(p)
        indices_batch = torch.clamp(indices_batch, min=0, max=self.grid_size - 1)
        features_batch = hidden_points.view(-1, self.hidden_dim).contiguous()  # [b * n, 128]
        sparse_tensor = spconv.SparseConvTensor(
            features=features_batch,
            indices=torch.cat([batch.int(), indices_batch.int()], dim=1).contiguous(),
            spatial_shape=(self.grid_size, self.grid_size, self.grid_size),
            batch_size=b,
        )

        # sparse_tensor1 = self.cpe1(sparse_tensor)
        # sparse_tensor2 = self.cpe2(sparse_tensor1) + sparse_tensor

        sparse_tensor = self.cpe1(sparse_tensor) + sparse_tensor
        sparse_tensor = self.cpe2(sparse_tensor) + sparse_tensor
        # sparse_tensor = self.cpe3(sparse_tensor) + sparse_tensor
        out = sparse_tensor.features.view(b, n, self.hidden_dim).contiguous()
        out = self.mlp_3(self.norm(self.act(self.mlp_2(out)))+points)
        # out = self.norm(self.mlp_2(out))
        return out

class RWKV_SEG(nn.Module):
    def __init__(self, i, up_factor=1, cls=16, hidden_dim=128, k=16, L=1, num=None, out_dim=1024, kp_radius=0.1):
        super(RWKV_SEG, self).__init__()
        self.i = i
        self.k = k
        self.out_dim = out_dim
        self.up_factor = up_factor
        self.mlp_delta = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, cls, kernel_size=1),
        )
        self.Length = L
        self.radius = kp_radius
        self.KP_D = KPNextBlock(128, 128, (1, 14, 28), self.radius, self.radius, 0)
        self.KP_D_2 = KPNextBlock(128, 128, (1, 14, 28), self.radius, self.radius, 0)

        rwkv_downs = []
        for i in range(self.Length):
            rwkv_downs.append(RWKVBlock(n_embd=hidden_dim, n_layer=8, layer_id=i, init_mode='fancy'))
        self.rwkv_downs = nn.ModuleList(rwkv_downs)

    def forward(self, px, px2=None, last_label=None):
        p, x = px  # (n, 3), (n, c), (b)
        b, c, n = x.shape

        device = x.device
        offset = torch.arange(b, device=device).view(-1, 1) * n
        offset = offset.repeat(1, n).view(-1)
        xyz = p.transpose(1, 2).contiguous()
        points = x.transpose(1, 2).contiguous()
        idx = knn(xyz, xyz, self.k, distance_limit=self.radius**2)
        _xyz = xyz.view(-1, 3).contiguous()
        _points = points.view(-1, c).contiguous()
        _idx = idx.view(-1, self.k).contiguous()
        _idx = _idx + offset.view(-1, 1)
        _points = self.KP_D(_xyz, _xyz, _points, _idx) + _points
        _points = self.KP_D_2(_xyz, _xyz, _points, _idx) + _points
        _points = _points.view(b, n, -1)
        points = _points + points
        for i in range(self.Length):
            points = self.rwkv_downs[i](points, patch_resolution=None)
        points = points.transpose(1, 2).contiguous()
        label = self.mlp_delta(points) # (B, c, N_prev * up_factor)
        return points, label, None


class Group(nn.Module):  # FPS + KNN
    def __init__(self, channel=3, num_group=2048, group_size=16):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)
        self.channel = channel

    def forward(self, pcd):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = pcd.shape  # B N C
        if self.channel != 3:
            xyz = pcd[:, :, :3].contiguous()
            label = pcd[:, :, 3:].contiguous()
        else:
            xyz = pcd

        # fps the centers out
        if num_points == self.num_group:
            center = xyz  # B G C
        else:
            center = gather_operation(
                xyz.transpose(1, 2).contiguous(), furthest_point_sample(xyz, self.num_group),
            ).transpose(1, 2).contiguous()  # (B, C, num_pc)
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center)  # B G M : get M idx for every center
        assert idx.size(1) == self.num_group  # G center
        assert idx.size(2) == self.group_size  # M knn group
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)
        if self.channel != 3:
            label_neighborhood = label.view(batch_size * num_points, -1)[idx, :]
            label_neighborhood = label_neighborhood.view(
                batch_size, self.num_group, self.group_size, self.channel
            ).contiguous()
            neighborhood = torch.cat([neighborhood, label_neighborhood], dim=3)
        return neighborhood, center

class First_Encoder(nn.Module):   ## Embedding module
    def __init__(self, in_channel=3, encoder_channel=128):
        super().__init__()
        self.in_channel = in_channel
        if in_channel != 3:
            self.in_channel += 3
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(self.in_channel, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N in_channel
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, self.in_channel)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 64 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 64 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 128 n
        feature = self.second_conv(feature) # BG 128 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 128
        return feature_global.reshape(bs, g, self.encoder_channel)  # B G 128



class Get_Kprev_RWKV(nn.Module):
    def __init__(self, n, feat_channel=3, k=16, out_dim=1024, L=4, grid_size=0.01):
        """Encoder that encodes information of partial point cloud"""
        super(Get_Kprev_RWKV, self).__init__()
        self.Length = L
        self.grid_size = grid_size
        self.group_divider = Group(channel=feat_channel, num_group=n, group_size=k)
        self.encoder = First_Encoder(in_channel=3, encoder_channel=128)
        rwkv_downs = []
        # rwkv_uppers = []
        for i in range(self.Length):
            rwkv_downs.append(RWKVBlock(n_embd=128, n_layer=L, layer_id=i, init_mode='fancy'))
        self.rwkv_downs = nn.ModuleList(rwkv_downs)
        self.sa_module_0 = PointNetSaModuleKNN(
            2048, 8, 128, [128], group_all=False, if_bn=False, if_idx=True,
        )
        self.sa_module_1 = PointNetSaModuleKNN(
            512, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True,
        )
        self.sa_module_2 = PointNetSaModuleKNN(
            128, 16, 256, [256, 512], group_all=False, if_bn=False, if_idx=True
        )
        self.sa_module_3 = PointNetSaModuleKNN(
            None, None, 512, [512, out_dim], group_all=True, if_bn=False
        )

    def forward(self, pcd):
        xyz = pcd
        neighborhood, center = self.group_divider(xyz.permute(0, 2, 1).contiguous()) # B G K 3
        points = self.encoder(neighborhood)  # B G C
        # pos = self.pos_embed(center)
        order = serialization(center, grid_size=self.grid_size)
        bs, n_p, _ = center.size()
        center = center.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
        points = points.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
        f_list = []
        for i in range(self.Length):
            points = self.rwkv_downs[i](points, patch_resolution=None)
            f_list.append(points)
        center = center.permute(0, 2, 1).contiguous()
        points = points.permute(0, 2, 1).contiguous()
        l0_xyz, l0_points, idx0 = self.sa_module_0(center, points)  # (B, 3, 512), (B, 128, 512)
        l1_xyz, l1_points, idx1 = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)
        l2_xyz, l2_points, idx2 = self.sa_module_2(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 512)
        _, global_feat = self.sa_module_3(l2_xyz, l2_points)  # (B, 3, 1), (B, out_dim, 1)
        return center, points, global_feat


def serialization(pos, order="random", grid_size=0.02):
    bs, n_p, _ = pos.size()
    if order == "random":
        options = ["z", "z-trans", "hilbert", "hilbert-trans"]
        order = random.choice(options)

    if not isinstance(order, list):
        order = [order]

    scaled_coord = pos / grid_size
    grid_coord = torch.floor(scaled_coord).to(torch.int64)
    min_coord = grid_coord.min(dim=1, keepdim=True)[0]
    grid_coord = grid_coord - min_coord

    batch_idx = torch.arange(0, pos.shape[0], 1.0).unsqueeze(1).repeat(1, pos.shape[1]).to(torch.int64).to(pos.device)

    point_dict = {'batch': batch_idx.flatten(), 'grid_coord': grid_coord.flatten(0, 1), }
    point_dict = Point(**point_dict)
    point_dict.serialization(order=order)

    order = point_dict.serialized_order
    return order

def point_shift(pcd_coarse, k_curr, grid_size):
    """
    Reorder point and feature tensors using serialized voxel order.

    Args:
        pcd_coarse (torch.Tensor): Input point cloud tensor with shape
            (batch_size, n_points, n_features).
        k_curr (torch.Tensor): Input feature tensor with shape
            (batch_size, n_points, n_features).
        grid_size (int): Grid size used for serialization.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Reordered point and feature tensors.
    """
    pcd_coarse_tran = pcd_coarse.transpose(1, 2).contiguous()
    k_curr_tran = k_curr.transpose(1, 2).contiguous()
    order = serialization(pcd_coarse_tran, grid_size=grid_size)
    bs, n_p, _ = pcd_coarse_tran.size()
    pcd_coarse_tran = pcd_coarse_tran.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
    k_curr_tran = k_curr_tran.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
    return pcd_coarse_tran.transpose(1, 2).contiguous(), k_curr_tran.transpose(1, 2).contiguous()



class RWKV_ATTN(nn.Module):
    def __init__(
        self,
        in_channel,
        pos_channel,
        dim=256,
        n_knn=16,
        pos_hidden_dim=64,
        attn_hidden_multiplier=4,
        drop_path=0.3,
        L=2,
    ):
        super(RWKV_ATTN, self).__init__()
        self.mlp_v = MlpRes(
            in_dim=in_channel * 2, hidden_dim=in_channel, out_dim=in_channel
        )
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(in_channel, dim, 1)
        self.conv_query = nn.Conv1d(in_channel, dim, 1)
        self.conv_value = nn.Conv1d(in_channel, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(pos_channel, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1),
        )
        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1),
        )
        self.conv_end = nn.Conv1d(dim, in_channel, 1)
        self.query_knn = KNN(k=self.n_knn, transpose_mode=True)
        rwkv_downs = []
        self.Length = L
        for i in range(L):
            rwkv_downs.append(RWKVBlock(n_embd=dim, n_layer=L if L > 1 else 8, layer_id=0, init_mode='fancy'))
        self.rwkv_downs = nn.ModuleList(rwkv_downs)
        self.attn1 = nn.Linear(dim, dim, bias=False)
        self.attn2 = nn.Linear(dim, dim, bias=False)

    def forward(self, pos, key, query):
        """
        Args:
            pos: (B, 3, N)
            key: (B, in_channel, N)
            query: (B, in_channel, N)

        Returns:
            Tensor: (B, in_channel, N), shape context feature
        """
        value = self.mlp_v(torch.cat([key, query], 1))
        identity = value
        value = self.conv_value(value)
        key = self.conv_key(key)
        query = self.conv_query(query)

        b, dim, n = value.shape
        pos_flipped = pos.permute(0, 2, 1).contiguous()

        _, idx_knn = self.query_knn(pos_flipped, pos_flipped)
        idx_knn = idx_knn.int()

        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(
            pos, idx_knn
        )  # b, 3, n, n_knn
        pos_embedding = self.pos_mlp(pos_rel)

        points = value.transpose(1, 2).contiguous()
        attn = self.rwkv_downs[0](points, patch_resolution=None)
        vvv = self.rwkv_downs[1](points, patch_resolution=None)
        attn = torch.sigmoid(self.attn1(attn))
        vvv = attn * vvv
        vvv = self.attn2(vvv)
        vvv = vvv.transpose(1, 2).contiguous()
        value = grouping_operation(vvv, idx_knn) - vvv.reshape(b, -1, n, 1) + pos_embedding

        key = grouping_operation(key, idx_knn)  # b, dim, n, n_knn
        qk_rel = query.reshape((b, -1, n, 1)) - key
        attention = self.attn_mlp(qk_rel + pos_embedding)  # b, dim, n, n_knn
        attention = torch.softmax(attention, -1)
        agg = einsum("b c i j, b c i j -> b c i", attention, value)  # b, dim, n
        y = self.conv_end(agg)

        return y + identity


class RWKV_PD(nn.Module):
    def __init__(self, dim_feat=512, hidden_dim=128, up_factor=2, num=None, radius=1.0, id=0, grid_size=0.01):
        """Snowflake Point Deconvolution"""
        super(RWKV_PD, self).__init__()
        self.id = id
        self.grid_size = grid_size
        self.up_factor = up_factor
        self.radius = radius
        self.up_sampler = nn.Upsample(scale_factor=up_factor)
        self.mlp_1 = MlpConv(in_channel=3, layer_dims=[hidden_dim//2, hidden_dim])
        self.mlp_2 = MlpConv(in_channel=hidden_dim * 2 + dim_feat, layer_dims=[hidden_dim * 2, hidden_dim])
        self.rwkvattn = RWKV_ATTN(in_channel=hidden_dim, pos_channel=3, dim=64, drop_path=0.5)
        self.mlp_ps = MlpConv(in_channel=hidden_dim, layer_dims=[hidden_dim//2, hidden_dim//4])
        self.ps = nn.ConvTranspose1d(
            hidden_dim//4, hidden_dim, up_factor, up_factor, bias=False
        )  # point-wise splitting
        self.mlp_delta_feature = MlpRes(in_dim=hidden_dim * 2, hidden_dim=hidden_dim, out_dim=hidden_dim)
        self.mlp_delta = MlpConv(in_channel=hidden_dim, layer_dims=[hidden_dim//2, 3])

    def forward(self, pcd_prev, feat_global, k_prev=None):
        b, _, n_prev = pcd_prev.shape  # (B, 3, N_prev)
        feat_1 = self.mlp_1(pcd_prev)  # (B, 128, N_prev)
        feat_1 = torch.cat(
            [
                feat_1,
                torch.max(feat_1, 2, keepdim=True)[0].repeat((1, 1, feat_1.size(2))),
                feat_global.repeat(1, 1, feat_1.size(2)),
            ],
            1,
        )  # (B, 128*2 + 512, N_prev)
        query = self.mlp_2(feat_1)  # (B, 128, N_prev)

        hidden = self.rwkvattn(
            pcd_prev, k_prev if k_prev is not None else query, query
        )  # (B, 128, N_prev) with relative position embedding applied internally
        feat_child = self.mlp_ps(hidden)  # (B, 32, N_prev)
        feat_child = self.ps(feat_child)  # (B, 128, N_prev * up_factor)
        hidden_up = self.up_sampler(hidden)  # (B, 128, N_prev * up_factor)
        k_curr = self.mlp_delta_feature(
            torch.cat([feat_child, hidden_up], 1)
        )  # (B, 128, N_prev * up_factor)
        pcd_child = self.up_sampler(pcd_prev)  # (B, 3, N_prev * up_factor)
        delta = torch.tanh(self.mlp_delta(torch.relu(k_curr)))
        if self.radius != 1:
            delta = delta * self.radius
        pcd_coarse = pcd_child + delta  # (B, 3, N_prev * up_factor)

        if self.id != 0:
            pcd_coarse, k_curr = point_shift(pcd_coarse, k_curr, self.grid_size)
            return (
                pcd_coarse,
                k_curr,
                None,
            )  # (B, 3, N_prev * up_factor), (B, 128, N_prev * up_factor), (B, 512, 1)
        return (
            pcd_coarse,
            k_curr,
            None,
        )  # (B, 3, N_prev * up_factor), (B, 128, N_prev * up_factor), (B, 512, 1)


class PCSSC(nn.Module):
    def __init__(
        self,
        class_num,
        dim_feat=1024,
        num_p0=1024,
        num_pin=4096,
        radius=1,
        up_factors=(2, 2, 2),
        hidden_dim=256,
        kp_radius=None,
        grid_size=None,
    ):
        """
        Args:
            dim_feat: int, dimension of global feature
            num_p0: int
            radius: searching radius
            up_factors: list of int
        """
        super(PCSSC, self).__init__()
        self.cls = class_num
        current_num = []
        current_points = num_p0
        for factor in up_factors:
            current_points *= factor
            current_num.append(current_points)
        self.cls = class_num
        self.num_p0 = num_p0

        self.up_factors = list(up_factors)

        uppers = []
        for i, factor in enumerate(self.up_factors):
            uppers.append(RWKV_PD(dim_feat=dim_feat, up_factor=factor, hidden_dim=hidden_dim,
                                  num=current_num[i], radius=radius, id=i+1, grid_size=grid_size[i+1]))
        self.uppers = nn.ModuleList(uppers)

        seg_uppers = []
        for i, factor in enumerate(self.up_factors):
            seg_uppers.append(RWKV_SEG(i=i, up_factor=factor, cls=self.cls, hidden_dim=hidden_dim, num=current_num[i], kp_radius=grid_size[i+1]))
        self.seg_uppers = nn.ModuleList(seg_uppers)
        self.get_feat = Get_Kprev_RWKV(num_pin, 3, out_dim=dim_feat, grid_size=grid_size[0])
        self.seed_genarate = SeedGenerator(dim_feat=dim_feat, up_factor=1, radius=radius, hidden_dim=hidden_dim,
                                           num_p0=num_p0, cls=class_num, kp_radius=kp_radius[0], grid_size=grid_size[1])
        self.feat_extract = RWKV_FeatureExtractor(128, dim_feat)


    def forward(self, point_cloud):
        """
        Args:
            point_cloud: (B, N, 3)
        """
        point_cloud = point_cloud.permute(0, 2, 1).contiguous()  # (B, 3, N)
        arr_pcd = []
        raw_xyz, raw_xyz_feat, step_feat = self.get_feat(point_cloud)
        pcd_coarse, k_prev_coarse, pcd_label, k_prev, _ = self.seed_genarate(raw_xyz, raw_xyz_feat, step_feat)
        pcd = torch.cat(
            [pcd_coarse, pcd_label], dim=1
        )  # (B, 3 + 12, N_prev * up_factor)
        arr_pcd.append(pcd.permute(0, 2, 1).contiguous())
        for i, upper in enumerate(self.uppers):
            step_feat = self.feat_extract(pcd_coarse, k_prev_coarse)  # (B, 512, 1)
            pcd_coarse, k_prev_coarse, _ = upper(
                pcd_coarse, step_feat, k_prev=k_prev_coarse
            )
            k_prev, pcd_label, _ = self.seg_uppers[i](
                [pcd_coarse, k_prev_coarse]
            )
            pcd = torch.cat(
                [pcd_coarse, pcd_label], dim=1
            )  # (B, 3 + 12, N_prev * up_factor)
            arr_pcd.append(pcd.permute(0, 2, 1).contiguous())
        return arr_pcd
