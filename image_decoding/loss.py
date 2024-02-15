import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm
from termcolor import cprint
from typing import Optional


class CLIPLoss(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.compute_similarity = nn.CosineSimilarity(dim=-1)

        self.cross_entropy = nn.CrossEntropyLoss(reduction=args.reduction)

        # Temperature (scaler)
        self.temp = nn.Parameter(torch.tensor([float(args.clip_temp_init)]))
        self.temp_min = args.clip_temp_min
        self.temp_max = args.clip_temp_max
        if not args.clip_temp_learn:
            self.temp.requires_grad = False

    def forward(self, x, y, fast=True, return_logits=False):
        batch_size = x.size(0)
        assert batch_size > 1, "Batch size must be greater than 1."
        targets = torch.arange(batch_size, requires_grad=False).long().to(device=x.device)  # fmt: skip

        if not fast:
            # less efficient way
            x_ = rearrange(x, "b f t -> 1 b (f t)")
            y_ = rearrange(y, "b f t -> b 1 (f t)")
            logits = self.compute_similarity(x_, y_)  # s

        else:
            # fast way
            x = x.reshape(batch_size, -1)
            y = y.reshape(batch_size, -1)

            # NOTE: scale the embeddings to unit norm
            x = x / x.norm(dim=-1, keepdim=True)
            y = y / y.norm(dim=-1, keepdim=True)

            # get dot products
            logits = torch.matmul(x, y.T)  # ( b, b )

        # FIXME: Probably exp is not needed, but keeping it for consistency.
        logits *= torch.exp(self.temp)

        # NOTE: as in https://arxiv.org/abs/2103.00020
        loss = (self.cross_entropy(logits, targets) + self.cross_entropy(logits.t(), targets)) / 2  # fmt: skip

        if return_logits:
            return logits, loss
        else:
            return loss

    def clamp_params(self):
        if not (self.temp_min is None and self.temp_max is None):
            self.temp.data.clamp_(min=self.temp_min, max=self.temp_max)
