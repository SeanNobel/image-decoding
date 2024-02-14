"""
Here I use the preprocessed data in Hebart et al., 2023. It looks that scaling and baseline correction
is performed but clamping is not, which is different from the Meta paper. I resample the data from 200Hz
to 120Hz as in the Meta paper.
"""

import os, sys
import numpy as np
import mne
import torch
from PIL import Image
from sklearn.preprocessing import RobustScaler
from functools import partial
from termcolor import cprint
from natsort import natsorted
from tqdm import tqdm
from typing import Tuple, List
import hydra
from omegaconf import DictConfig

from transformers import AutoProcessor, CLIPVisionModel
import clip

from image_decoding.utils import sequential_apply


def scale_clamp(
    X: np.ndarray,
    clamp_lim: float = 5.0,
    clamp: bool = True,
    scale_transposed: bool = True,
) -> np.ndarray:
    """
    Args:
        X: ( channels, timesteps )
    Returns:
        X: ( channels, timesteps )
    """
    X = RobustScaler().fit_transform(X.T if scale_transposed else X)

    if scale_transposed:
        X = X.T

    if clamp:
        X = X.clip(min=-clamp_lim, max=clamp_lim)

    return X


@torch.no_grad()
def encode_images(y_list: List[str], preprocess, clip_model, device) -> torch.Tensor:
    """Encodes images with either OpenAI or Huggingface pretrained CLIP. https://huggingface.co/openai/clip-vit-large-patch14"""
    if isinstance(clip_model, CLIPVisionModel):
        last_hidden_states = []

        for y in tqdm(y_list, desc="Preprocessing & encoding images"):
            model_input = preprocess(images=Image.open(y), return_tensors="pt")

            model_output = clip_model(**model_input.to(device))

            last_hidden_states.append(model_output.last_hidden_state.cpu())

        return torch.cat(last_hidden_states, dim=0)
    else:
        model_input = torch.stack(
            [
                preprocess(Image.open(y).convert("RGB"))
                for y in tqdm(y_list, desc="Preprocessing images")
            ]
        )

        return sequential_apply(
            model_input,
            clip_model.encode_image,
            batch_size=32,
            device=device,
            desc="Encoding images",
        ).float()


@hydra.main(version_base=None, config_path="./", config_name="config")
def run(args: DictConfig) -> None:
    meg_paths = [
        os.path.join(args.meg_dir, f"preprocessed_P{i+1}-epo.fif") for i in range(4)
    ]
    sample_attrs_paths = [
        os.path.join(args.thingsmeg_dir, f"sourcedata/sample_attributes_P{i+1}.csv")
        for i in range(4)
    ]

    save_dir = os.path.join(args.save_dir, "preproc")
    os.makedirs(save_dir, exist_ok=True)

    for subject_id, (meg_path, sample_attrs_path) in enumerate(zip(meg_paths, sample_attrs_paths)):  # fmt: skip
        cprint(f"==== Processing subject {subject_id+1} ====", "cyan")

        sample_attrs = np.loadtxt(
            sample_attrs_path, dtype=str, delimiter=",", skiprows=1
        )

        device = f"cuda:{args.cuda_id}"

        # -----------------
        #        MEG
        # -----------------
        if not args.skip_meg:

            epochs = mne.read_epochs(meg_path)

            cprint(f"> Resampling epochs to {args.brain_sfreq}Hz...", "cyan")
            epochs.resample(args.brain_sfreq, n_jobs=8)

            cprint(f"> Scale and clamping epochs to ±{args.clamp_lim}...", "cyan")
            epochs.apply_function(
                partial(scale_clamp, scale_transposed=False, clamp_lim=args.clamp_lim),
                n_jobs=8,
            )

            cprint("> Baseline correction...", "cyan")
            epochs.apply_baseline((None, 0))

            X = torch.from_numpy(epochs.get_data()).to(torch.float32)
            # ( 27048, 271, segment_len )

            cprint(f"MEG P{subject_id+1}: {X.shape}", "cyan")

            torch.save(X, os.path.join(save_dir, f"MEG_P{subject_id+1}.pt"))

        # -----------------
        #      Images
        # -----------------
        if not args.skip_images:
            if args.vision_model.startswith("ViT-"):
                clip_model, preprocess = clip.load(args.vision_model)
                clip_model = clip_model.eval().to(device)

            elif args.vision_model.startswith("openai/"):
                clip_model = CLIPVisionModel.from_pretrained(args.vision_model).to(device)  # fmt: skip
                preprocess = AutoProcessor.from_pretrained(args.vision_model)
            else:
                raise ValueError(f"Unknown pretrained CLIP type: {args.vision_model}")  # fmt: skip

            y_list = []
            for path in sample_attrs[:, 8]:
                if "images_meg" in path:
                    y_list.append(
                        os.path.join(args.images_dir, "/".join(path.split("/")[1:]))
                    )
                elif "images_test_meg" in path:
                    y_list.append(
                        os.path.join(
                            args.images_dir,
                            "_".join(os.path.basename(path).split("_")[:-1]),
                            os.path.basename(path),
                        )
                    )
                elif "images_catch_meg" in path:
                    y_list.append(os.path.join(args.images_dir, "black.jpg"))
                else:
                    raise ValueError(f"Unknown image path type: {path}")

            np.savetxt(
                os.path.join(save_dir, f"Images_P{subject_id+1}.txt"),
                y_list,
                fmt="%s",
                delimiter="\n",
            )

            Y = encode_images(y_list, preprocess, clip_model, device)

            cprint(f"Images P{subject_id+1}: {Y.shape}", "cyan")

            torch.save(Y, os.path.join(save_dir, f"Images_P{subject_id+1}.pt"))


if __name__ == "__main__":
    run()
