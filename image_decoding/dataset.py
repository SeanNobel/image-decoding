import os, sys
import torch
from torchvision.transforms.functional import to_tensor
import numpy as np
from PIL import Image
import cv2
from termcolor import cprint
from tqdm import tqdm
from typing import Tuple, List
import gc

from nd.utils.eval_utils import get_run_dir


class ThingsMEGCLIPDataset(torch.utils.data.Dataset):
    def __init__(self, args) -> None:
        super().__init__()

        self.num_subjects = 4
        self.large_test_set = args.large_test_set

        # NOTE: Some categories
        high_categories = np.loadtxt(
            os.path.join(
                args.things_dir, "27 higher-level categories/category_mat_manual.tsv"
            ),
            dtype=int,
            delimiter="\t",
            skiprows=1,
        )  # ( 1854, 27 )

        preproc_dir = os.path.join(args.preprocessed_data_dir, args.preproc_name)

        sample_attrs_paths = [
            os.path.join(args.thingsmeg_dir, f"sourcedata/sample_attributes_P{i+1}.csv")
            for i in range(self.num_subjects)
        ]

        X_list = []
        Y_list = []
        subject_idxs_list = []
        categories_list = []
        y_idxs_list = []
        train_idxs_list = []
        test_idxs_list = []
        for subject_id, sample_attrs_path in enumerate(sample_attrs_paths):
            # MEG
            X_list.append(
                torch.load(os.path.join(preproc_dir, f"MEG_P{subject_id+1}.pt"))
            )
            # ( 27048, 271, segment_len )

            # Images (or Texts)
            if args.align_to == "vision":
                Y = torch.load(os.path.join(preproc_dir, f"Images_P{subject_id+1}.pt"), map_location="cpu")  # fmt: skip
            elif args.align_to == "text":
                Y = torch.load(os.path.join(preproc_dir, f"Texts_P{subject_id+1}.pt"), map_location="cpu")  # fmt: skip
            else:
                raise ValueError(f"Invalid align_to: {args.align_to}")

            if Y.ndim == 2:
                assert args.num_clip_tokens == 1, "num_clip_tokens > 1 is specified, but the embessings don't have temporal dimension."  # fmt: skip
                assert not args.align_tokens == "all", "align_tokens is specified as 'all', but the embessings don't have temporal dimension."  # fmt: skip

                Y = Y.unsqueeze(1)
            else:
                if args.align_tokens == "mean":
                    assert args.num_clip_tokens == 1
                    Y = Y.mean(dim=1, keepdim=True)
                elif args.align_tokens == "cls":
                    assert args.num_clip_tokens == 1
                    Y = Y[:, :1]
                else:
                    assert args.align_tokens == "all"

            Y_list.append(Y.clone())
            del Y; gc.collect()  # fmt: skip

            # Indexes
            sample_attrs = np.loadtxt(
                sample_attrs_path, dtype=str, delimiter=",", skiprows=1
            )  # ( 27048, 18 )

            categories_list.append(torch.from_numpy(sample_attrs[:, 2].astype(int)))
            y_idxs_list.append(torch.from_numpy(sample_attrs[:, 1].astype(int)))

            subject_idxs_list.append(
                torch.ones(len(sample_attrs), dtype=int) * subject_id
            )

            # Split
            train_idxs, test_idxs = self.make_split(
                sample_attrs, large_test_set=self.large_test_set
            )
            idx_offset = len(sample_attrs) * subject_id
            train_idxs_list.append(train_idxs + idx_offset)
            test_idxs_list.append(test_idxs + idx_offset)

        self.X = torch.cat(X_list, dim=0)
        self.Y = torch.cat(Y_list, dim=0)
        self.subject_idxs = torch.cat(subject_idxs_list, dim=0)

        self.categories = torch.cat(categories_list) - 1
        assert torch.equal(self.categories.unique(), torch.arange(self.categories.max() + 1))  # fmt: skip
        self.num_categories = len(self.categories.unique())

        self.high_categories = self.to_high_categories(self.categories, high_categories)
        self.num_high_categories = self.high_categories.max() + 1

        self.y_idxs = torch.cat(y_idxs_list) - 1
        assert torch.equal(self.y_idxs.unique(), torch.arange(self.y_idxs.max() + 1))

        self.train_idxs = torch.cat(train_idxs_list, dim=0)
        self.test_idxs = torch.cat(test_idxs_list, dim=0)

        cprint(f"X: {self.X.shape} | Y: {self.Y.shape} | subject_idxs: {self.subject_idxs.shape} | train_idxs: {self.train_idxs.shape} | test_idxs: {self.test_idxs.shape}", "cyan")  # fmt: skip

        # self.subject_names = [f"s0{i+1}" for i in range(4)]

        if args.chance:
            self.X = self.X[torch.randperm(len(self.X))]

        del X_list, Y_list, categories_list, y_idxs_list, subject_idxs_list, train_idxs_list, test_idxs_list  # fmt: skip
        gc.collect()

    def __len__(self) -> int:
        return len(self.Y)

    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.subject_idxs[i], self.y_idxs[i], self.categories[i], self.high_categories[i]  # fmt: skip

    @staticmethod
    def make_split(
        sample_attrs: np.ndarray,
        large_test_set: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sample_attrs ( 27048, 18 ): Elements are strs.
            refined (bool): If True, use splits by Meta, modified from Hebart et al., 2023.
        Returns:
            train_trial_idxs ( 22248, ): _description_
            test_trial_idxs ( 2400, ): _description_
        """
        trial_types = sample_attrs[:, 0]  # ( 27048, )

        if not large_test_set:
            # Small test set
            train_trial_idxs = np.where(trial_types == "exp")[0]  # ( 22248, )
            test_trial_idxs = np.where(trial_types == "test")[0]  # ( 2400, )

            assert len(train_trial_idxs) == 22248 and len(test_trial_idxs) == 2400
        else:
            category_idxs = sample_attrs[:, 2].astype(int)  # ( 27048, )

            test_trial_idxs = np.where(trial_types == "test")[0]  # ( 2400, )
            test_category_idxs = np.unique(np.take(category_idxs, test_trial_idxs))
            # ( 200, )

            test_trial_idxs = np.where(
                np.logical_and(
                    np.isin(category_idxs, test_category_idxs),
                    np.logical_not(trial_types == "test"),
                )
            )[0]
            # ( 2400, )

            train_trial_idxs = np.where(
                np.logical_and(
                    trial_types == "exp",
                    np.logical_not(np.isin(category_idxs, test_category_idxs)),
                )
            )[0]
            # ( 19848, )

            assert len(train_trial_idxs) == 19848 and len(test_trial_idxs) == 2400

        return torch.from_numpy(train_trial_idxs), torch.from_numpy(test_trial_idxs)

    def to_high_categories(
        self, categories: torch.Tensor, high_categories: np.ndarray
    ) -> torch.Tensor:
        """_summary_
        Args:
            categories ( 27048 * 4, ): Elements are integers of [0, 1854].
                End value 1854 is for catch trials.
            high_categories ( 1854, 27 ): Each row is a zero to three -hot vector.
        Returns:
            high_categories ( 27048 * 4, ): Elements are integers of [0, 27].
        """
        # Set categories that are not in any of higher categories as "uncategorized".
        unc = np.where(high_categories.sum(axis=1) == 0)[0]
        # This takes the first higher-category for categories that are in multiple higher-categories.
        high_categories = np.argmax(high_categories, axis=1)  # ( 1854, )

        # Higher-category 27 for "uncategorized" and catch trials.
        high_categories[unc] = high_categories.max() + 1  # ( 1854, )
        high_categories = np.append(high_categories, high_categories.max())  # ( 1855, )

        return torch.from_numpy(high_categories)[categories]


class ThingsMEGDecoderDataset(torch.utils.data.Dataset):
    def __init__(self, args) -> None:
        super().__init__()

        self.image_size = args.image_sizes[-1]

        preproc_dir = os.path.join(args.preprocessed_data_dir, args.preproc_name)
        embeds_dir = os.path.join(args.clip_embeds_dir, *get_run_dir(args).split("/")[2:])  # fmt: skip

        sample_attrs_paths = [
            os.path.join(args.thingsmeg_dir, f"sourcedata/sample_attributes_P{i+1}.csv")
            for i in range(4)
        ]

        Y_path_list = []
        train_idxs_list = []
        test_idxs_list = []
        for subject_id, sample_attrs_path in enumerate(sample_attrs_paths):
            # Image paths
            Y_path_list.append(
                np.loadtxt(
                    os.path.join(preproc_dir, f"Images_P{subject_id+1}.txt"), dtype=str
                )
            )

            sample_attrs = np.loadtxt(
                sample_attrs_path, dtype=str, delimiter=",", skiprows=1
            )
            train_idxs, test_idxs = ThingsMEGCLIPDataset.make_split(
                sample_attrs, large_test_set=args.large_test_set
            )
            idx_offset = len(sample_attrs) * subject_id
            train_idxs_list.append(train_idxs + idx_offset)
            test_idxs_list.append(test_idxs + idx_offset)

        self.Y_path = np.concatenate(Y_path_list)  # ( 27048, )

        self.train_idxs = torch.cat(train_idxs_list, dim=0)
        self.test_idxs = torch.cat(test_idxs_list, dim=0)

        # MEG embeddings
        Z = torch.load(os.path.join(embeds_dir, "brain_mse_embeds.pt"))
        Y_embeds = torch.load(os.path.join(embeds_dir, "vision_embeds.pt"))
        self.Z = self._load_postproc_embeds(Z, Y_embeds)

        del Z, Y_embeds, Y_path_list, train_idxs_list, test_idxs_list
        gc.collect()

    def __len__(self) -> int:
        return len(self.Z)

    def __getitem__(self, i):
        # _Y = cv2.resize(cv2.imread(self.Y_path[i]), (self.image_size, self.image_size))
        # _Y = torch.from_numpy(_Y).to(torch.float32).permute(2, 0, 1) / 255.0

        Y = Image.open(self.Y_path[i]).resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )

        return self.Z[i], to_tensor(Y)

    def _load_postproc_embeds(self, Z, Y_embeds) -> torch.Tensor:
        """_summary_
        Args:
            Z ( 108192, F ): _description_
            Y_embeds ( 108192, F ): _description_
        Returns:
            Z ( 108192, F ): z-score normalized and inverse z-score normalized Z.
        """
        Y_embeds = Y_embeds[self.train_idxs]
        mean, std = Y_embeds.mean(dim=0), Y_embeds.std(dim=0)

        # z-score normalize each feature across predictions
        Z = (Z - Z.mean(dim=0)) / Z.std(dim=0)

        # Inverse z-score normalize each feature across predictions
        return Z * std + mean