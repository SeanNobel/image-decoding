import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from typing import Union, Optional, Callable

def sequential_apply(
    X: Union[torch.Tensor, np.ndarray],
    # NOTE: nn.Module is a hint for general DNNs. Callable is a hint for CLIP encoder
    fn: Union[transforms.Compose, nn.Module, Callable],
    batch_size: int,
    device: Optional[str] = None,
    subject_idxs: Optional[torch.Tensor] = None,
    desc: str = "",
    reduction: str = "mean",
) -> torch.Tensor:
    """Avoid CPU / CUDA out of memory.
    Args:
        X (torch.Tensor): _description_
        model (Union[transforms.Compose, VisionEncoder]): _description_
        batch_size (int): _description_
        subject_idxs (Optional[torch.Tensor], optional): _description_. Defaults to None.
    Returns:
        torch.Tensor: _description_
    """
    # NOTE: This is for torchvision transforms, which doesn't accept a batch of samples.
    # A bit of messy implementation.
    if isinstance(fn, transforms.Compose) and isinstance(X, np.ndarray):
        # NOTE: np.split needs number of subarrays, while torch.split needs the size of chunks.
        return torch.cat(
            [
                fn(Image.fromarray(_X.squeeze())).unsqueeze(0)
                for _X in np.split(X, X.shape[0])
            ]
        )

    orig_device = X.device

    if device is None:
        device = orig_device

    # NOTE: sequential_apply doesn't do sequential application if batch_size == X.shape[0].
    if batch_size == X.shape[0]:
        # assert isinstance(X, torch.Tensor) and isinstance(model, nn.Module)
        if subject_idxs is None:
            return fn(X.to(device)).to(orig_device)
        else:
            return fn(X.to(device), subject_idxs.to(device)).to(orig_device)

    if subject_idxs is None:
        output = [
            fn(_X.to(device)) for _X in tqdm(torch.split(X, batch_size), desc=desc)
        ]
    else:
        output = [
            fn(_X.to(device), _subject_idxs.to(device))
            for _X, _subject_idxs in tqdm(
                zip(
                    torch.split(X, batch_size),
                    torch.split(subject_idxs, batch_size),
                ),
                desc=desc,
            )
        ]

    if isinstance(output[0], torch.Tensor):
        return torch.cat(output).to(orig_device)

    elif isinstance(output[0], dict):
        stacked_dict = {}

        for key in output[0].keys():
            _output = [_dict[key] for _dict in output]

            if _output[0].ndim == 0:
                _output = torch.stack(_output)

                if reduction == "mean":
                    _output = _output.mean()
                elif reduction == "sum":
                    _output = _output.sum()

                stacked_dict.update({key: _output})
            else:
                stacked_dict.update({key: torch.cat(_output)})

        return stacked_dict
    else:
        raise ValueError(f"Unknown output type: {type(output[0])}")