import os, sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from time import time
from tqdm import tqdm
from termcolor import cprint
from typing import Union, Optional
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

import clip
from transformers import AutoProcessor, CLIPVisionModel

from image_decoding.dataset import ThingsMEGCLIPDataset
from image_decoding.brain_encoder import BrainEncoder
from image_decoding.classifiers import DiagonalClassifier, LabelClassifier
from image_decoding.loss import CLIPLoss
from image_decoding.utils import (
    Models,
    sequential_apply,
    count_parameters,
    plot_latents_2d,
)


def train():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_name = args.train_name

    if sweep:
        wandb.init(config=None)

        run_name += "_" + "".join(
            [
                f"{k}-{v:.3f}_" if isinstance(v, float) else f"{k}-{v}_"
                for k, v in wandb.config.items()
            ]
        )

        wandb.run.name = run_name
        args.__dict__.update(wandb.config)
        cprint(wandb.config, "cyan")
        wandb.config.update(args.__dict__)

    run_dir = os.path.join("runs", args.dataset.lower(), run_name)
    os.makedirs(run_dir, exist_ok=True)

    device = f"cuda:{args.cuda_id}"

    # -----------------------
    #       Dataloader
    # -----------------------
    dataset = ThingsMEGCLIPDataset(args)
    train_set = torch.utils.data.Subset(dataset, dataset.train_idxs)
    test_set = torch.utils.data.Subset(dataset, dataset.test_idxs)

    loader_args = {
        "collate_fn": None,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": False,
    }
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True, **loader_args
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=len(test_set), shuffle=False, **loader_args
    )

    # ---------------
    #      Loss
    # ---------------
    loss_func = CLIPLoss(args).to(device)

    # ---------------------
    #        Model
    # ---------------------
    subjects = dataset.subject_names if hasattr(dataset, "subject_names") else dataset.num_subjects  # fmt: skip

    brain_encoder = BrainEncoder(args, subjects=subjects).to(device)

    models = Models(brain_encoder, None, loss_func)

    if sweep:
        wandb.config.update({"brain_encoder_params": count_parameters(brain_encoder)})

    # ---------------------
    #      Classifiers
    # ---------------------
    train_classifier = DiagonalClassifier(args.acc_topks)
    test_classifier = LabelClassifier(dataset, args.acc_topks, device)

    # ---------------------
    #      Optimizers
    # ---------------------
    optimizer = torch.optim.Adam(models.get_params(), lr=args.lr)

    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )
    elif args.lr_scheduler == "multistep":
        mlstns = [int(m * args.epochs) for m in args.lr_multistep_mlstns]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=mlstns, gamma=args.lr_step_gamma
        )
    else:
        cprint("Using no scheduler.", "yellow")
        scheduler = None

    # -----------------------
    #     Strat training
    # -----------------------
    max_test_acc = 0.0
    no_best_counter = 0

    for epoch in range(args.epochs):
        train_clip_losses = []
        train_mse_losses = []
        test_clip_losses = []
        test_mse_losses = []
        train_topk_accs = []
        test_topk_accs = []

        # For plotting latents
        train_Y_list = []
        train_Z_list = []
        train_categories_list = []

        # -----------------------
        #       Train step
        # -----------------------
        models.train()
        for batch in tqdm(train_loader, desc="Train"):
            X, Y, subject_idxs, y_idxs, classes, high_categories = *batch, *[None] * (6 - len(batch))  # fmt: skip
            X, Y = X.to(device), Y.to(device)

            ret_dict = brain_encoder(X, subject_idxs)

            Z, Z_mse = ret_dict["Z_clip"], ret_dict["Z_mse"]

            clip_loss = loss_func(Y, Z)

            mse_loss = F.mse_loss(
                rearrange(Y, "b d t -> b (d t)"),
                rearrange(Z_mse, "b d t -> b (d t)"),
                reduction=args.reduction,
            )

            loss = args.lambd * clip_loss + (1 - args.lambd) * mse_loss

            with torch.no_grad():
                if isinstance(train_classifier, DiagonalClassifier):
                    topk_accs, _ = train_classifier(Z, Y)
                elif isinstance(train_classifier, LabelClassifier):
                    topk_accs = train_classifier(Z, y_idxs.to(device))
                else:
                    raise NotImplementedError

            train_clip_losses.append(clip_loss.item())
            train_mse_losses.append(mse_loss.item())
            train_topk_accs.append(topk_accs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.plot_latents:
                train_Y_list.append(Y.detach().cpu().numpy())
                train_Z_list.append(Z.detach().cpu().numpy())

                if high_categories is not None:
                    train_categories_list.append(high_categories.numpy())
                elif classes is not None:
                    train_categories_list.append(classes.numpy())
                else:
                    raise ValueError("plot_latents is True but no classes are given.")

        loss_func.clamp_params()

        _ = models.params_updated()

        # -----------------------
        #       Test step
        # -----------------------
        models.eval()
        for batch in tqdm(test_loader, desc="Test"):
            X, Y, subject_idxs, y_idxs, classes, high_categories = *batch, *[None] * (6 - len(batch))  # fmt: skip
            X, Y = X.to(device), Y.to(device)

            with torch.no_grad():
                # NOTE: sequential_apply doesn't do sequential application if batch_size == X.shape[0].
                Z = sequential_apply(
                    X,
                    brain_encoder,
                    args.batch_size,
                    subject_idxs=subject_idxs,
                    desc="BrainEncoder",
                    reduction=args.reduction,
                )

                ret_dict = Z

                Z, Z_mse = ret_dict["Z_clip"], ret_dict["Z_mse"]

                clip_loss = loss_func(Y, Z)

                mse_loss = F.mse_loss(
                    rearrange(Y, "b d t -> b (d t)"),
                    rearrange(Z_mse, "b d t -> b (d t)"),
                    reduction=args.reduction,
                )

                if isinstance(test_classifier, DiagonalClassifier):
                    topk_accs, _ = test_classifier(
                        Z, Y, sequential=args.test_with_whole
                    )
                elif isinstance(test_classifier, LabelClassifier):
                    topk_accs = test_classifier(
                        Z, y_idxs.to(device), sequential=args.test_with_whole
                    )
                else:
                    raise NotImplementedError

            test_clip_losses.append(clip_loss.item())
            test_mse_losses.append(mse_loss.item())
            test_topk_accs.append(topk_accs)

        train_topk_accs = np.stack(train_topk_accs)
        test_topk_accs = np.stack(test_topk_accs)

        print(
            f"Epoch {epoch}/{args.epochs} | ",
            f"avg train CLIP loss: {np.mean(train_clip_losses):.3f} | ",
            f"avg test CLIP loss: {np.mean(test_clip_losses):.3f} | ",
            f"lr: {optimizer.param_groups[0]['lr']:.5f}",
        )

        if sweep:
            performance_now = {
                "epoch": epoch,
                "train_clip_loss": np.mean(train_clip_losses),
                "train_mse_loss": np.mean(train_mse_losses),
                "test_clip_loss": np.mean(test_clip_losses),
                "test_mse_loss": np.mean(test_mse_losses),
                "lrate": optimizer.param_groups[0]["lr"],
                "temp": loss_func.temp.item(),
            }

            performance_now.update(
                {
                    f"train_top{k}_acc": np.mean(train_topk_accs[:, i])
                    for i, k in enumerate(args.acc_topks)
                }
            )
            performance_now.update(
                {
                    f"test_top{k}_acc": np.mean(test_topk_accs[:, i])
                    for i, k in enumerate(args.acc_topks)
                }
            )

            wandb.log(performance_now)

        if scheduler is not None:
            scheduler.step()

        models.save(run_dir)

        # NOTE: This is mean over multiple ks.
        if np.mean(test_topk_accs) > max_test_acc:
            cprint(f"New best. Saving models to {run_dir}", color="cyan")
            models.save(run_dir, best=True)

            max_test_acc = np.mean(test_topk_accs)
            no_best_counter = 0
        else:
            no_best_counter += 1

        if len(train_categories_list) > 0:
            if epoch == 0:
                plot_latents_2d(
                    np.concatenate(train_Y_list),
                    np.concatenate(train_categories_list),
                    save_path=os.path.join(run_dir, f"plots/image_latents/epoch{epoch}.png"),  # fmt: skip
                )
            if epoch % 50 == 0:
                plot_latents_2d(
                    np.concatenate(train_Z_list),
                    np.concatenate(train_categories_list),
                    save_path=os.path.join(run_dir, f"plots/ecog_latents/epoch{epoch}.png"),  # fmt: skip
                )

        if no_best_counter > args.patience:
            cprint(f"Early stopping at epoch {epoch}", color="cyan")
            break


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def run(_args: DictConfig) -> None:
    global args, sweep

    # NOTE: Using default.yaml only for specifying the experiment settings yaml.
    args = OmegaConf.load(os.path.join("configs", _args.config_path))

    sweep = _args.sweep

    if sweep:
        sweep_config = OmegaConf.to_container(
            args.sweep_config, resolve=True, throw_on_missing=True
        )

        sweep_id = wandb.sweep(sweep_config, project=args.project_name)

        wandb.agent(sweep_id, train, count=args.sweep_count)
    else:
        train()


if __name__ == "__main__":
    run()
