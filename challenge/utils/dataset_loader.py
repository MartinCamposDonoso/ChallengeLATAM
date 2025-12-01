"""
Dataset Loading Functions

This module contains functions to load datasets in supervision format.
"""

from pathlib import Path
import supervision as sv
from typing import Tuple


def get_ds_from_supervision(
    dataset_path: str
) -> Tuple[sv.DetectionDataset, sv.DetectionDataset, sv.DetectionDataset]:
    """
    Loads train, valid, and test datasets in supervision format.

    Args:
        dataset_path: Root path of the dataset

    Returns:
        Tuple containing (train_dataset, valid_dataset, test_dataset)
    """
    base_path = Path(dataset_path)

    # Load training set
    print(f"Loading Training set from {base_path / 'train'}...")
    ds_train = sv.DetectionDataset.from_yolo(
        images_directory_path=str(base_path / "train" / "images"),
        annotations_directory_path=str(base_path / "train" / "labels"),
        data_yaml_path=str(base_path / "data.yaml")
    )
    print(f"Training loaded: {len(ds_train)} images.\n")

    # Load validation set
    print(f"Loading Validation set from {base_path / 'valid'}...")
    ds_valid = sv.DetectionDataset.from_yolo(
        images_directory_path=str(base_path / "valid" / "images"),
        annotations_directory_path=str(base_path / "valid" / "labels"),
        data_yaml_path=str(base_path / "data.yaml")
    )
    print(f"Validation loaded: {len(ds_valid)} images.\n")

    # Load test set
    print(f"Loading Test set from {base_path / 'test'}...")
    ds_test = sv.DetectionDataset.from_yolo(
        images_directory_path=str(base_path / "test" / "images"),
        annotations_directory_path=str(base_path / "test" / "labels"),
        data_yaml_path=str(base_path / "data.yaml")
    )
    print(f"Test loaded: {len(ds_test)} images.\n")

    print("Data loading completed successfully.\n")

    return ds_train, ds_valid, ds_test
