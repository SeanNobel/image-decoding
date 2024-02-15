import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from functools import partial
from typing import Optional, Union, Callable, List, Tuple
from termcolor import cprint


class SpatialAttention(nn.Module):
    def __init__(
        self, loc: torch.Tensor, D1: int, K: int, d_drop: float, flat: bool = True
    ):
        super().__init__()

        self.flat = flat

        # TODO: Check if those two are identical.
        x, y = loc.T
        if self.flat:  # Implementation version 1
            self.z_re = nn.Parameter(torch.Tensor(D1, K, K))
            self.z_im = nn.Parameter(torch.Tensor(D1, K, K))
            nn.init.kaiming_uniform_(self.z_re, a=np.sqrt(5))
            nn.init.kaiming_uniform_(self.z_im, a=np.sqrt(5))

            k_arange = torch.arange(K)
            rad1 = torch.einsum("k,c->kc", k_arange, x)
            rad2 = torch.einsum("l,c->lc", k_arange, y)
            rad = rad1.unsqueeze(1) + rad2.unsqueeze(0)
            self.register_buffer("cos", torch.cos(2 * torch.pi * rad))
            self.register_buffer("sin", torch.sin(2 * torch.pi * rad))

        else:  # Implementation version 2
            # make a complex-valued parameter, reshape k,l into one dimension
            self.z = nn.Parameter(torch.rand(size=(D1, K**2), dtype=torch.cfloat))

            # vectorize of k's and l's
            a = []
            for k in range(K):
                for l in range(K):
                    a.append((k, l))
            a = torch.tensor(a)
            k, l = a[:, 0], a[:, 1]
            # NOTE: pre-compute the values of cos and sin (they depend on k, l, x and y which repeat)
            phi = 2 * torch.pi * (torch.einsum("k,x->kx", k, x) + torch.einsum("l,y->ly", l, y))  # fmt: skip
            self.register_buffer("cos", torch.cos(phi))
            self.register_buffer("sin", torch.sin(phi))

        self.spatial_dropout = SpatialDropout(loc, d_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            X ( b, c, t ): _description_

        Returns:
            _type_: _description_
        """
        # NOTE: drop some channels within a d_drop of the sampled channel
        X = self.spatial_dropout(X)  # ( b, c, t )

        if self.flat:
            real = torch.einsum("dkl,klc->dc", self.z_re, self.cos)
            imag = torch.einsum("dkl,klc->dc", self.z_im, self.sin)
            # ( D1, c )
        else:
            real = torch.einsum("jm, me -> je", self.z.real, self.cos)
            imag = torch.einsum("jm, me -> je", self.z.imag, self.sin)

        # NOTE: to get the softmax spatial attention weights over input electrodes,
        # we don't compute exp, etc (as in the eq. 5), we take softmax instead:
        a = F.softmax(real + imag, dim=-1)  # ( D1, c )

        # NOTE: each output is a diff weighted sum over each input channel
        return torch.einsum("oi,bit->bot", a, X)


class SpatialDropout(nn.Module):
    """Using same drop center for all samples in batch"""

    def __init__(self, loc, d_drop):
        super().__init__()
        self.loc = loc  # ( num_channels, 2 )
        self.d_drop = d_drop
        self.num_channels = loc.shape[0]

    def forward(self, X):  # ( B, num_channels, seq_len )
        assert X.shape[1] == self.num_channels

        if self.training:
            drop_center = self.loc[np.random.randint(self.num_channels)]  # ( 2, )
            distances = (self.loc - drop_center).norm(dim=-1)  # ( num_channels, )
            mask = torch.where(distances < self.d_drop, 0.0, 1.0).to(device=X.device)
            # ( num_channels, )
            X = torch.einsum("c,bct->bct", mask, X)
            # cprint(1 - torch.count_nonzero(X) / torch.numel(X), "yellow")

        return X


class SubjectBlock(nn.Module):
    def __init__(
        self,
        num_subjects: int,
        loc: np.ndarray,
        D1: int,
        K: int,
        d_drop: float,
        num_channels: int,
        spatial_attention: bool = True,
    ):
        super().__init__()

        self.num_subjects = num_subjects

        if spatial_attention:
            self.spatial_attention = SpatialAttention(loc, D1, K, d_drop)
        else:
            cprint("Not using spatial attention.", "yellow")
            self.spatial_attention = None

        self.conv = nn.Conv1d(
            in_channels=D1 if spatial_attention else num_channels,
            out_channels=D1,
            kernel_size=1,
            stride=1,
        )
        self.subject_layer = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=D1,
                    out_channels=D1,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                )
                for _ in range(self.num_subjects)
            ]
        )

    def forward(
        self, X: torch.Tensor, subject_idxs: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if self.spatial_attention is not None:
            X = self.spatial_attention(X)  # ( B, 270, 256 )

        X = self.conv(X)  # ( B, 270, 256 )

        if subject_idxs is not None:
            X = torch.cat(
                [
                    self.subject_layer[i](x.unsqueeze(dim=0))
                    for i, x in zip(subject_idxs, X)
                ]
            )  # ( B, 270, 256 )

        else:
            cprint("Unknown subject.", "yellow")

            X = torch.stack(
                [self.subject_layer[i](X) for i in range(self.num_subjects)]
            ).mean(dim=0)

        return X


class ConvBlock(nn.Module):
    def __init__(
        self,
        k: int,
        D1: int,
        D2: int,
        ksize: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()

        self.k = k
        in_channels = D1 if k == 0 else D2

        self.conv0 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=D2,
            kernel_size=ksize,
            padding="same",
            dilation=2 ** ((2 * self.k) % 5),
        )
        self.batchnorm0 = nn.BatchNorm1d(num_features=D2)
        self.conv1 = nn.Conv1d(
            in_channels=D2,
            out_channels=D2,
            kernel_size=ksize,
            padding="same",
            dilation=2 ** ((2 * self.k + 1) % 5),
        )
        self.batchnorm1 = nn.BatchNorm1d(num_features=D2)
        self.conv2 = nn.Conv1d(
            in_channels=D2,
            out_channels=2 * D2,
            kernel_size=ksize,
            padding="same",
            dilation=2,  # NOTE: The text doesn't say this, but the picture shows dilation=2
        )
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.k == 0:
            X = self.conv0(X)
        else:
            X = self.conv0(X) + X  # skip connection

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        X = self.conv2(X)
        X = F.glu(X, dim=-2)

        return self.dropout(X)


class TemporalAggregation(nn.Module):
    def __init__(
        self,
        temporal_dim: int,
        embed_dim: int,
        temporal_agg: str = "affine",
        multiplier: int = 1,
    ) -> None:
        super().__init__()

        """Modified from: https://ai.meta.com/static-resource/image-decoding"""
        # self.layers = nn.Sequential()

        # NOTE: conv_final corresponds to linear projection in the paper as long as the kernel size and stride are 1
        # self.layers.add_module("linear_projection",nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim * expand * multiplier, kernel_size=1))

        if temporal_agg == "affine":
            self.layers = nn.Linear(temporal_dim, multiplier)
        elif temporal_agg == "pool":
            self.layers = nn.AdaptiveAvgPool1d(1)
        else:
            raise NotImplementedError()

        # NOTE: MLP projectors are provided for CLIP and MSE
        # self.layers.add_module("mlp_projector",nn.Sequential(nn.Flatten(), nn.Linear(embed_dim * expand * multiplier, embed_dim * multiplier), nn.GELU()))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.layers(X)  # ( b, F * multiplier )


class BrainEncoder(nn.Module):
    def __init__(self, args, subjects: Union[int, List[str]]) -> None:
        super().__init__()

        D1, D2, D3, K, F = args.D1, args.D2, args.D3, args.K, args.F
        temporal_dim = int(args.seq_len * args.brain_sfreq)
        num_clip_tokens = args.num_clip_tokens
        num_subjects: int = subjects if isinstance(subjects, int) else len(subjects)
        num_channels: int = args.num_channels
        spatial_attention: bool = args.spatial_attention
        num_blocks: int = args.num_blocks
        conv_block_ksize: int = args.conv_block_ksize
        temporal_agg: str = args.temporal_agg
        p_drop: float = args.p_drop
        d_drop: float = args.d_drop
        final_ksize: int = args.final_ksize
        final_stride: int = args.final_stride

        self.ignore_subjects = args.ignore_subjects

        loc = self._ch_locations_2d(args.montage_path)

        num_subjects = num_subjects if not self.ignore_subjects else 1
        self.subject_block = SubjectBlock(
            num_subjects, loc, D1, K, d_drop, num_channels, spatial_attention
        )

        self.blocks = nn.Sequential()

        for k in range(num_blocks):
            self.blocks.add_module(
                f"block{k}", ConvBlock(k, D1, D2, conv_block_ksize, p_drop)
            )

        self.conv_final = nn.Conv1d(
            in_channels=D2,
            out_channels=D3,
            kernel_size=final_ksize,
            stride=final_stride,
        )

        # temporal_dim = conv_output_size(
        #     init_temporal_dim,
        #     ksize=final_ksize,
        #     stride=final_stride,
        #     repetition=3 if temporal_agg == "original" else 1,
        #     downsample=sum(downsample),
        # )

        self.temporal_aggregation = TemporalAggregation(
            temporal_dim, D3, temporal_agg, multiplier=num_clip_tokens
        )

        self.clip_head = nn.Sequential(nn.LayerNorm([D3, num_clip_tokens]), nn.GELU(), nn.Conv1d(D3, F, 1))  # fmt: skip
        self.mse_head = nn.Sequential(nn.LayerNorm([D3, num_clip_tokens]), nn.GELU(), nn.Conv1d(D3, F, 1))  # fmt: skip

    @staticmethod
    def _ch_locations_2d(montage_path: str) -> torch.Tensor:
        loc = np.load(montage_path)

        # Min-max normalization
        loc = (loc - loc.min(axis=0)) / (loc.max(axis=0) - loc.min(axis=0))

        # Scale down to keep a margin of 0.1 on each side
        loc = loc * 0.8 + 0.1

        return torch.from_numpy(loc.astype(np.float32))

    def forward(self, X: torch.Tensor, subject_idxs: torch.Tensor) -> torch.Tensor:
        X = self.subject_block(
            X, subject_idxs if not self.ignore_subjects else torch.zeros_like(subject_idxs)  # fmt: skip
        )

        X = self.blocks(X)

        X = F.gelu(self.conv_final(X))

        X = self.temporal_aggregation(X)

        Z_clip = self.clip_head(X)
        Z_mse = self.mse_head(X)

        return {"Z_clip": Z_clip, "Z_mse": Z_mse}

    def encode(
        self,
        X: torch.Tensor,
        subject_idxs: Optional[torch.Tensor],
        return_mse: bool = True,
        normalize: bool = True,
        stats: Optional[Tuple[float]] = None,
        device=None,
    ) -> torch.Tensor:
        if device is not None:
            orig_device = X.device
            X, subject_idxs = X.to(device), subject_idxs.to(device)

        single = X.dim == 2

        if single:
            X = X.unsqueeze(0)

            if subject_idxs is not None:
                subject_idxs = subject_idxs.unsqueeze(0)

        Z = self(X, subject_idxs)
        Z = Z[1] if return_mse else Z[0]

        if normalize:
            Z /= Z.norm(dim=-1, keepdim=True)

        if stats is not None:
            # Inverse normalization
            Z = (Z - Z.mean()) / Z.std()
            mean, std = stats
            Z = Z * std + mean

        if device is not None:
            Z = Z.to(orig_device)

        if single:
            Z = Z.squeeze(0)

        return Z
