import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from termcolor import cprint
from typing import List
import gc


def calc_similarity(
    Z: torch.Tensor, Y: torch.Tensor, sequential: bool, pbar: bool = True
) -> torch.Tensor:
    batch_size, _size = len(Z), len(Y)

    Z = Z.contiguous().view(batch_size, -1)
    Y = Y.contiguous().view(_size, -1)

    # NOTE: avoid CUDA out of memory.
    if sequential:
        Z = Z / Z.norm(dim=-1, keepdim=True)
        Y = Y / Y.norm(dim=-1, keepdim=True)

        similarity = torch.empty(batch_size, _size).to(device=Z.device)

        if pbar:
            pbar = tqdm(total=batch_size, desc="Similarity matrix of test size")

        for i in range(batch_size):
            similarity[i] = Z[i] @ Y.T

            if pbar:
                pbar.update(1)
    else:
        Z = rearrange(Z, "b f -> b 1 f")
        Y = rearrange(Y, "b f -> 1 b f")
        similarity = F.cosine_similarity(Y, Z, dim=-1)

    torch.cuda.empty_cache()

    return similarity


def top_k_accuracy(k: int, similarity: torch.Tensor, labels: torch.Tensor):
    """_summary_
    Args:
        k (int): _description_
        similarity ( b, 2400 ): _description_
        labels ( b, ): _description_
    Returns:
        _type_: _description_
    """
    return np.mean(
        [
            label in row
            for row, label in zip(
                torch.topk(similarity, k, dim=1, largest=True)[1], labels
            )
        ]
    )


class DiagonalClassifier(nn.Module):
    def __init__(self, topk: List[int] = [1, 10]):
        super().__init__()

        self.topk = topk

        # NOTE: Do we need to adjust the accuracies for the dataset size?
        self.factor = 1  # self.batch_size / 241

    @torch.no_grad()
    def forward(
        self, Z: torch.Tensor, Y: torch.Tensor, sequential=False, return_pred=False
    ) -> torch.Tensor:
        batch_size = Z.size(0)

        diags = torch.arange(batch_size).to(device=Z.device)

        similarity = calc_similarity(Z, Y, sequential)  # ( b, b )

        topk_accs = np.array([top_k_accuracy(k, similarity, diags) for k in self.topk])

        if return_pred:
            cprint(similarity.argmax(axis=1).shape, "cyan")
            cprint(Y.shape, "cyan")

            return topk_accs, similarity.argmax(axis=1).cpu()
        else:
            return topk_accs, similarity


class LabelClassifier(nn.Module):
    @torch.no_grad()
    def __init__(self, dataset, topk: List[int] = [1, 5], device="cuda"):
        super().__init__()

        self.topk = topk

        test_y_idxs = dataset.y_idxs[dataset.test_idxs].numpy()
        # ( 9600, )
        # NOTE: torch.unique has no return_index option
        test_y_idxs, arg_unique = np.unique(test_y_idxs, return_index=True)
        # ( 2400, )
        self.test_y_idxs = torch.from_numpy(test_y_idxs)

        self.Y = torch.index_select(dataset.Y, 0, dataset.test_idxs)
        self.Y = torch.index_select(self.Y, 0, torch.from_numpy(arg_unique))
        # ( 2400, F )

        self.Y = self.Y.to(device)
        self.test_y_idxs = self.test_y_idxs.to(device)

    @torch.no_grad()
    def forward(
        self, Z: torch.Tensor, y_idxs: torch.Tensor, sequential: bool = False
    ) -> np.ndarray:
        """_summary_

        Args:
            Z ( b=9600, F ): _description_
            y_idxs ( b=9600, ): _description_

        Returns:
            torch.Tensor: _description_
        """
        similarity = calc_similarity(Z, self.Y, sequential)  # ( 2400, b )

        labels = y_idxs == self.test_y_idxs.unsqueeze(1)  # ( 2400, b )
        assert torch.all(labels.sum(dim=0) == 1)
        labels = labels.to(int).argmax(dim=0)  # ( b, )

        topk_accs = np.array([top_k_accuracy(k, similarity, labels) for k in self.topk])

        return topk_accs
