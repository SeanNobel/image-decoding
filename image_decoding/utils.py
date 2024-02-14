import os, sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
from typing import Union, Optional, Callable, Dict
from termcolor import cprint

matplotlib.use("Agg")


class Models:
    """
    This class is implemented so that vision encoder could also betrained
    but it is not used for this project.
    """

    def __init__(
        self,
        brain_encoder: nn.Module,
        vision_encoder: Optional[nn.Module] = None,
        loss_func: Optional[nn.Module] = None,
    ):
        self.brain_encoder = brain_encoder
        self.vision_encoder = vision_encoder
        self.loss_func = loss_func

        self.brain_encoder_params = self._clone_params_list(self.brain_encoder)
        if self.vision_encoder is not None:
            self.vision_encoder_params = self._clone_params_list(self.vision_encoder)

    def get_params(self):
        params = list(self.brain_encoder.parameters()) + list(self.loss_func.parameters())  # fmt: skip

        if self.vision_encoder is not None:
            params += list(self.vision_encoder.parameters())

        return params

    @staticmethod
    def _clone_params_list(model: nn.Module) -> Dict[str, torch.Tensor]:
        return {name: params.clone().cpu() for name, params in model.named_parameters()}

    @staticmethod
    def _get_non_updated_layers(
        new_params: Dict[str, torch.Tensor],
        prev_params: Dict[str, torch.Tensor],
    ) -> list:
        return [
            key
            for key in new_params.keys()
            if torch.equal(prev_params[key], new_params[key]) and new_params[key].requires_grad  # fmt: skip
        ]

    def params_updated(self, show_non_updated: bool = True) -> bool:
        updated = True

        new_params = self._clone_params_list(self.brain_encoder)
        non_updated_layers = self._get_non_updated_layers(
            new_params, self.brain_encoder_params
        )
        if len(non_updated_layers) > 0:
            if show_non_updated:
                cprint(f"Following layers in brain encoder are not updated: {non_updated_layers}","red")  # fmt: skip

            updated = False
        self.brain_encoder_params = new_params

        if self.vision_encoder is not None:
            new_params = self._clone_params_list(self.vision_encoder)
            non_updated_layers = self._get_non_updated_layers(
                new_params, self.vision_encoder_params
            )
            if len(non_updated_layers) > 0:
                if show_non_updated:
                    cprint(f"Following layers in vision encoder are not updated: {non_updated_layers}","red")  # fmt: skip

                updated = False
            self.vision_encoder_params = new_params

        return updated

    def train(self) -> None:
        self.brain_encoder.train()
        if self.vision_encoder is not None:
            self.vision_encoder.train()
        self.loss_func.train()

    def eval(self) -> None:
        self.brain_encoder.eval()
        if self.vision_encoder is not None:
            self.vision_encoder.eval()
        self.loss_func.eval()

    def save(self, run_dir: str, best: bool = False) -> None:
        torch.save(
            self.brain_encoder.state_dict(),
            os.path.join(run_dir, f"brain_encoder_{'best' if best else 'last'}.pt"),
        )

        if self.vision_encoder is not None:
            torch.save(
                self.vision_encoder.state_dict(),
                os.path.join(
                    run_dir, f"vision_encoder_{'best' if best else 'last'}.pt"
                ),
            )


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


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_latents_2d(
    latents: np.ndarray,
    classes: np.ndarray,
    save_path: str,
    cmap: str = "gist_rainbow",
) -> None:
    """
    latents ( samples, dim )
    classes ( samples, )
    """
    classes = classes.astype(float) / classes.max()

    fig, axs = plt.subplots(ncols=5, figsize=(20, 5), tight_layout=True)
    fig.suptitle(f"Originally {latents.shape[1]} dimensions")

    latents_reduced = PCA(n_components=2).fit_transform(latents)
    axs[0].scatter(*latents_reduced.T, c=classes, cmap=cmap)
    axs[0].set_title("PCA")

    for perplexity, ax in zip([2, 10, 40, 100], axs[1:]):
        latents_reduced = TSNE(n_components=2, perplexity=perplexity).fit_transform(latents)  # fmt: skip

        ax.scatter(*latents_reduced.T, c=classes, cmap=cmap)
        ax.set_title(f"t-SNE (perplexity={perplexity})")

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    fig.savefig(save_path)
