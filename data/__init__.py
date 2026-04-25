# Copyright (c) Ruopeng Gao. All Rights Reserved.

from torch.utils.data import DataLoader

from .joint_dataset import JointDataset
from .transforms import build_transforms
from .util import collate_fn

"""
def build_dataset(config: dict):
    return JointDataset(
        data_root=config["DATA_ROOT"],
        datasets=config["DATASETS"],
        splits=config["DATASET_SPLITS"],
        transforms=build_transforms(config),
        size_divisibility=config.get("SIZE_DIVISIBILITY", 0),
    )
"""
def build_dataset(config: dict):
    dataset = JointDataset(
        data_root=config["DATA_ROOT"],
        datasets=config["DATASETS"],
        splits=config["DATASET_SPLITS"],
        transforms=build_transforms(config),
        size_divisibility=config.get("SIZE_DIVISIBILITY", 0),
    )

    # ------------------------------------------
    # MINI DEBUG DATASET (SAFE FOR JointDataset)
    # ------------------------------------------
    max_seq = config.get("DATASET_MAX_SEQUENCES", None)
    if max_seq is not None:
        for dset in config["DATASETS"]:
            for split in config["DATASET_SPLITS"]:

                # fetch original sequence names
                seq_dict = dataset.sequence_infos[dset][split]
                seq_names = list(seq_dict.keys())

                print(f"[DEBUG] Reducing {dset}.{split} from {len(seq_names)} → {max_seq} sequences")

                # keep only first 'max_seq' sequences
                keep = seq_names[:max_seq]

                # slice sequence_infos
                dataset.sequence_infos[dset][split] = {
                    k: seq_dict[k] for k in keep
                }

                # slice image_paths
                dataset.image_paths[dset][split] = {
                    k: dataset.image_paths[dset][split][k] for k in keep
                }

                # slice annotations
                dataset.annotations[dset][split] = {
                    k: dataset.annotations[dset][split][k] for k in keep
                }

                # slice legality masks
                dataset.ann_is_legals[dset][split] = {
                    k: dataset.ann_is_legals[dset][split][k] for k in keep
                }

        print(f"[DEBUG] Mini dataset applied successfully.")

    return dataset




def build_dataloader(dataset, batch_size: int, shuffle: bool, num_workers: int):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
