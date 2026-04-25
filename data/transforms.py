# Copyright (c) Ruopeng Gao. All Rights Reserved.

import copy
import math
import torch
import einops
import random
from torchvision.transforms import v2
import torchvision.transforms as T
from math import floor
from PIL import Image
from triton.language import dtype

from utils.box_ops import box_xywh_to_xyxy, box_xyxy_to_cxcywh
from .util import is_legal


class MultiIdentity:
    def __init__(self):
        return

    def __call__(self, images, annotations, metas):
        return images, annotations, metas


class MultiCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, annotations, metas):
        for transform in self.transforms:
            images, annotations, metas = transform(images, annotations, metas)
        return images, annotations, metas



class MultiStack:
    """
    Stack a sequence of images into a single tensor, (T, C, H, W).
    The result tensor is more suitable for multi-image processing.
    """
    def __init__(self):
        return

    def __call__(self, images, annotations, metas):
        if isinstance(images, list):
            if isinstance(images[0], torch.Tensor):
                images = torch.stack(images, dim=0)
        return images, annotations, metas


class MultiBoxXYWHtoXYXY:
    def __init__(self):
        return

    def __call__(self, images, annotations, metas):
        for _ in range(len(annotations)):
            annotations[_]["bbox"] = box_xywh_to_xyxy(annotations[_]["bbox"])
        return images, annotations, metas


class MultiBoxXYXYtoCXCYWH:
    def __init__(self):
        return

    def __call__(self, images, annotations, metas):
        for _ in range(len(annotations)):
            annotations[_]["bbox"] = box_xyxy_to_cxcywh(annotations[_]["bbox"])
        return images, annotations, metas


class MultiRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, images, annotations, metas):
        # Here, the boxes in annotations are in the format of (x1, y1, x2, y2).
        if torch.rand(1).item() < self.p:
            if isinstance(images, torch.Tensor):
                images = v2.functional.horizontal_flip_image(images)
            elif isinstance(images, list):
                assert isinstance(images[0], Image.Image)
                images = [v2.functional.hflip(_) for _ in images]
            else:
                raise NotImplementedError(f"The input image type {type(images)} is not supported.")
            h, w = get_image_hw(images)
            for annotation in annotations:
                annotation["bbox"] = (
                    annotation["bbox"][:, [2, 1, 0, 3]]
                    * torch.as_tensor([-1, 1, -1, 1])
                    + torch.as_tensor([w, 0, w, 0])
                )
        return images, annotations, metas






class MultiToTensor:
    def __call__(self, images, annotations, metas):
        if isinstance(images, list):
            images = [v2.functional.to_tensor(im) for im in images]
        return images, annotations, metas




class MultiNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, annotations, metas):
        # images = images.to(torch.float32).div(255)
        # images = v2.functional.normalize(images, mean=self.mean, std=self.std)
        h, w = images.shape[-2:]
        for annotation in annotations:
            annotation["bbox"] = annotation["bbox"] / torch.tensor([w, h, w, h])
        return images.contiguous(), annotations, metas


class MultiNormalizeBoundingBoxes:
    def __init__(self):
        return

    def __call__(self, images, annotations, metas):
        # Only normalize the bounding boxes,
        # the images will be normalized in the training loop (on cuda).
        h, w = images.shape[-2:]
        for annotation in annotations:
            annotation["bbox"] = annotation["bbox"] / torch.tensor([w, h, w, h])
        return images.contiguous(), annotations, metas


# For MOTIP only, biding the ID label:

class GenerateIDLabels:
    def __init__(self, num_id_vocabulary: int, aug_num_groups: int, num_training_ids: int):
        self.num_id_vocabulary = num_id_vocabulary
        self.aug_num_groups = aug_num_groups
        self.num_training_ids = num_training_ids

    def __call__(self, images, annotations, metas):
        _T = len(images)
        _G = self.aug_num_groups
        # Collect all IDs:
        ids_set = set()
        for annotation in annotations:
            ids_set.update(set(annotation["id"].tolist()))
        _N = len(ids_set)

        # ID anns consist of the following parts:
        # (1): a (_G, _T, _N) tensor, representing the ID labels of each object in each frame.
        # (2): a (_G, _T, _N) tensor, representing the corresponding index of each object in detection annotations.
        # (3): a (_G, _T, _N) tensor, representing the mask of ID labels in each frame.
        # (4): a (_G, _T, _N) tensor, representing the time index of each object.

        ids_list = list(ids_set)
        id_to_idx = {ids_list[_]: _ for _ in range(_N)}     # the idx in the final ID labels
        base_id_masks = torch.ones((_T, _N), dtype=torch.bool)
        base_ann_idxs = - torch.ones((_T, _N), dtype=torch.int64)
        # These "base" ID anns are used to generate the final ID anns, do not directly use them.
        for t in range(_T):
            annotation = annotations[t]
            for i in range(len(annotation["id"])):
                _id = annotation["id"][i].item()
                _ann_idx = i
                _n = id_to_idx[_id]
                # generate the corresponding ID ann:
                base_id_masks[t, _n] = False
                base_ann_idxs[t, _n] = _ann_idx

        # Generate the final ID anns
        # If the number of IDs is larger than `num_id_vocabulary`, we need to randomly select a subset of IDs.
        # Also, if the number of IDs is larger than `num_training_ids`, we need to randomly select a subset of IDs.
        if _N > self.num_id_vocabulary or _N > self.num_training_ids:
            _random_select_idxs = torch.randperm(_N)[:self.num_training_ids if _N > self.num_training_ids else self.num_id_vocabulary]
            base_id_masks = base_id_masks[:, _random_select_idxs]
            base_ann_idxs = base_ann_idxs[:, _random_select_idxs]
            _N = self.num_training_ids if _N > self.num_training_ids else self.num_id_vocabulary
            pass
        # Normal processing:
        id_labels = torch.zeros((_G, _T, _N), dtype=torch.int64)
        id_masks = torch.ones((_G, _T, _N), dtype=torch.bool)
        ann_idxs = - torch.ones((_G, _T, _N), dtype=torch.int64)
        for group in range(_G):
            _random_id_labels = torch.randperm(self.num_id_vocabulary)[:_N]
            _random_id_labels = _random_id_labels[None, ...].repeat(_T, 1)
            # _random_id_labels[base_id_masks] = -1
            id_labels[group] = _random_id_labels.clone()
            id_masks[group] = base_id_masks.clone()
            ann_idxs[group] = base_ann_idxs.clone()
        # Generate the time indexes:
        times = torch.arange(_T, dtype=torch.int64)[None, :, None].repeat(_G, 1, _N)
        # Check the shapes:
        assert id_labels.shape == id_masks.shape == ann_idxs.shape == times.shape

        # Split the ID anns into each frame:
        id_labels_list = torch.split(id_labels, split_size_or_sections=1, dim=1)    # each item is in (_G, 1, _N)
        id_masks_list = torch.split(id_masks, split_size_or_sections=1, dim=1)      # each item is in (_G, 1, _N)
        ann_idxs_list = torch.split(ann_idxs, split_size_or_sections=1, dim=1)      # each item is in (_G, 1, _N)
        times_list = torch.split(times, split_size_or_sections=1, dim=1)            # each item is in (_G, 1, _N)

        # Update the annotations (put the ID anns into the annotations):
        for t in range(_T):
            annotations[t]["id_labels"] = id_labels_list[t]
            annotations[t]["id_masks"] = id_masks_list[t]
            annotations[t]["ann_idxs"] = ann_idxs_list[t]
            annotations[t]["times"] = times_list[t]
        pass
        return images, annotations, metas


class TurnIntoTrajectoryAndUnknown:
    def __init__(
            self,
            num_id_vocabulary: int,
            aug_trajectory_occlusion_prob: float,
            aug_trajectory_switch_prob: float,
    ):
        self.num_id_vocabulary = num_id_vocabulary
        self.aug_trajectory_occlusion_prob = aug_trajectory_occlusion_prob
        self.aug_trajectory_switch_prob = aug_trajectory_switch_prob
        return

    def __call__(self, images, annotations, metas):
        id_labels = torch.cat([annotation["id_labels"] for annotation in annotations], dim=1)
        id_masks = torch.cat([annotation["id_masks"] for annotation in annotations], dim=1)
        ann_idxs = torch.cat([annotation["ann_idxs"] for annotation in annotations], dim=1)
        times = torch.cat([annotation["times"] for annotation in annotations], dim=1)
        _G, _T, _N = id_labels.shape
        # Del these fields from the annotations:
        for t in range(_T):
            del annotations[t]["id_labels"]
            del annotations[t]["id_masks"]
            del annotations[t]["ann_idxs"]
            del annotations[t]["times"]

        # Copy the ID anns to "trajectory_" and "unknown_":
        trajectory_id_labels = id_labels.clone()
        trajectory_id_masks = id_masks.clone()
        trajectory_ann_idxs = ann_idxs.clone()
        trajectory_times = times.clone()
        unknown_id_labels = id_labels.clone()
        unknown_id_masks = id_masks.clone()
        unknown_ann_idxs = ann_idxs.clone()
        unknown_times = times.clone()

        if self.aug_trajectory_occlusion_prob > 0.0:
            # Make trajectory occlusion:
            # 1. Turn the shape into (_G * _N, _T):
            trajectory_id_masks = einops.rearrange(trajectory_id_masks, "G T N -> (G N) T")
            unknown_id_masks = einops.rearrange(unknown_id_masks, "G T N -> (G N) T")
            # 2. Generate the occlusion mask:
            trajectory_occlusion_masks = torch.zeros_like(trajectory_id_masks, dtype=torch.bool)
            unknown_occlusion_masks = torch.zeros_like(unknown_id_masks, dtype=torch.bool)
            for i in range(_G * _N):
                if random.random() < self.aug_trajectory_occlusion_prob:
                    begin_idx = random.randint(0, _T - 1)
                    _max_T = _T - 1 - begin_idx
                    end_idx = begin_idx + math.ceil(_max_T * random.random())
                    trajectory_occlusion_masks[i, begin_idx:end_idx] = True
                    unknown_occlusion_masks[i, begin_idx:end_idx] = True
            # Currently, we do not check the legality of the occlusion mask.
            # However, we did it in the previous version.
            # 3. Apply the occlusion mask:
            trajectory_id_masks = trajectory_id_masks | trajectory_occlusion_masks
            unknown_id_masks = unknown_id_masks | unknown_occlusion_masks
            # 4. Turn the shape back:
            trajectory_id_masks = einops.rearrange(trajectory_id_masks, "(G N) T -> G T N", G=_G, N=_N)
            unknown_id_masks = einops.rearrange(unknown_id_masks, "(G N) T -> G T N", G=_G, N=_N)

        if self.aug_trajectory_switch_prob > 0.0:
            # Make trajectory switch:
            # 1. Turn the shape into (_G * _T, _N):
            trajectory_id_labels = einops.rearrange(trajectory_id_labels, "G T N -> (G T) N")
            trajectory_id_masks = einops.rearrange(trajectory_id_masks, "G T N -> (G T) N")
            trajectory_ann_idxs = einops.rearrange(trajectory_ann_idxs, "G T N -> (G T) N")
            # 2. Switch for each frame:
            #    (switching the ID labels is the same as switching the ann_idxs and masks)
            for g_t in range(_G * _T):
                switch_p = torch.ones((_N, )) * self.aug_trajectory_switch_prob
                switch_map = torch.bernoulli(switch_p)
                switch_idxs = torch.nonzero(switch_map)[:, 0]
                if len(switch_idxs) > 1:    # make sure can be switched
                    shuffled_switch_idxs = switch_idxs[torch.randperm(len(switch_idxs))]
                    # Do switch:
                    trajectory_ann_idxs[g_t, switch_idxs] = trajectory_ann_idxs[g_t, shuffled_switch_idxs]
                    trajectory_id_masks[g_t, switch_idxs] = trajectory_id_masks[g_t, shuffled_switch_idxs]
                    pass
                pass
            # 3. Turn the shape back:
            trajectory_id_labels = einops.rearrange(trajectory_id_labels, "(G T) N -> G T N", G=_G, T=_T)
            trajectory_id_masks = einops.rearrange(trajectory_id_masks, "(G T) N -> G T N", G=_G, T=_T)
            trajectory_ann_idxs = einops.rearrange(trajectory_ann_idxs, "(G T) N -> G T N", G=_G, T=_T)
            pass

        # Check all ID labels are legal:
        assert torch.all(trajectory_id_labels >= 0)
        assert torch.all(unknown_id_labels >= 0)

        # Add "newborn" ID label to unknown ID labels for supervision:
        # 1. Turn the shape into (_G * _N, _T):
        trajectory_id_labels = einops.rearrange(trajectory_id_labels, "G T N -> (G N) T")
        trajectory_id_masks = einops.rearrange(trajectory_id_masks, "G T N -> (G N) T")
        unknown_id_labels = einops.rearrange(unknown_id_labels, "G T N -> (G N) T")
        unknown_id_masks = einops.rearrange(unknown_id_masks, "G T N -> (G N) T")
        # 2. Calculate the already_born masks:
        already_born_masks = torch.cumsum(~trajectory_id_masks, dim=1)
        already_born_masks = already_born_masks > 0
        # 3. Generate the newborn ID labels:
        newborn_id_label_masks = ~ torch.cat(
            [
                torch.zeros((_G * _N, 1), dtype=torch.bool),
                already_born_masks[:, :-1]
            ],
            dim=-1
        )
        unknown_id_labels[newborn_id_label_masks] = self.num_id_vocabulary
        # 4. Turn the shape back:
        trajectory_id_labels = einops.rearrange(trajectory_id_labels, "(G N) T -> G T N", G=_G, N=_N)
        trajectory_id_masks = einops.rearrange(trajectory_id_masks, "(G N) T -> G T N", G=_G, N=_N)
        unknown_id_labels = einops.rearrange(unknown_id_labels, "(G N) T -> G T N", G=_G, N=_N)
        unknown_id_masks = einops.rearrange(unknown_id_masks, "(G N) T -> G T N", G=_G, N=_N)

        # Update the annotations:
        for t in range(_T):
            annotations[t]["trajectory_id_labels"] = trajectory_id_labels[:, t:t+1, :]
            annotations[t]["trajectory_id_masks"] = trajectory_id_masks[:, t:t+1, :]
            annotations[t]["trajectory_ann_idxs"] = trajectory_ann_idxs[:, t:t+1, :]
            annotations[t]["trajectory_times"] = trajectory_times[:, t:t+1, :]
            annotations[t]["unknown_id_labels"] = unknown_id_labels[:, t:t+1, :]
            annotations[t]["unknown_id_masks"] = unknown_id_masks[:, t:t+1, :]
            annotations[t]["unknown_ann_idxs"] = unknown_ann_idxs[:, t:t+1, :]
            annotations[t]["unknown_times"] = unknown_times[:, t:t+1, :]

        return images, annotations, metas

#change
def build_transforms(config):
    return MultiCompose([
        MultiBoxXYWHtoXYXY(),

        # ❌ Remove MultiSimulate
        # ❌ Remove RandomResize / RandomCrop
        # ❌ Remove ColorJitter
        # ❌ Remove RandomPhotometricDistort
        # ❌ Remove RandomHorizontalFlip (optional)
        
        # ✔ Just resize to the RF-DETR input:
        MultiRandomHorizontalFlip(p=0.5),
        MultiCompose([
            lambda images, annotations, metas: (
                [v2.functional.resize(im, [512, 512]) for im in images],
                annotations,
                metas
            )
        ]),

        MultiBoxXYXYtoCXCYWH(),
        
        MultiToTensor(),
        MultiStack(),

        # ✔ For RF-DETR: normalize boxes only here,
        # and do image normalization inside training loop
        MultiNormalizeBoundingBoxes(),

        # ✔ Keep: ID label generation
        GenerateIDLabels(
            num_id_vocabulary=config["NUM_ID_VOCABULARY"],
            aug_num_groups=config["AUG_NUM_GROUPS"],
            num_training_ids=config.get("NUM_TRAINING_IDS", config["NUM_ID_VOCABULARY"]),
        ),

        TurnIntoTrajectoryAndUnknown(
            num_id_vocabulary=config["NUM_ID_VOCABULARY"],
            aug_trajectory_occlusion_prob=config["AUG_TRAJECTORY_OCCLUSION_PROB"],
            aug_trajectory_switch_prob=config["AUG_TRAJECTORY_SWITCH_PROB"],
        ),
    ])

def get_image_hw(image: torch.Tensor | list | Image.Image):
    if isinstance(image, torch.Tensor):
        return image.shape[-2], image.shape[-1]
    elif isinstance(image, list):
        return get_image_hw(image[0])
    elif isinstance(image, Image.Image):
        return image.height, image.width
    else:
        raise NotImplementedError("The input image type is not supported.")


