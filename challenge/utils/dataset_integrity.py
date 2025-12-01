"""
Dataset Integrity Checking Functions

This module contains functions to verify the integrity of YOLO format datasets.
"""

from pathlib import Path
from typing import Dict, Any, List, Set
import supervision as sv


def check_integrity(dataset_path: str) -> None:
    """
    Verifies that each label file has its corresponding image file.

    Args:
        dataset_path: Root path of the dataset
    """
    print(f"\nStarting integrity check on: {dataset_path}\n")

    splits = ['train', 'valid', 'test']
    all_healthy = True

    for split in splits:
        images_path = Path(dataset_path) / split / "images"
        labels_path = Path(dataset_path) / split / "labels"

        if not images_path.exists() or not labels_path.exists():
            print(f"⚠️  {split.upper()}: Missing folders")
            all_healthy = False
            continue

        image_files = {f.stem for f in images_path.glob("*")
                      if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}}
        label_files = {f.stem for f in labels_path.glob("*.txt")}

        if image_files == label_files:
            print(f"✅ {split.upper()}: Perfect. {len(label_files)} labels have their corresponding image.")
        else:
            missing_images = label_files - image_files
            missing_labels = image_files - label_files
            all_healthy = False

            if missing_images:
                print(f"❌ {split.upper()}: {len(missing_images)} labels without images")
            if missing_labels:
                print(f"❌ {split.upper()}: {len(missing_labels)} images without labels")

    print("\n" + "-" * 50)
    if all_healthy:
        print("RESULT: The dataset is HEALTHY. No missing files.")
    else:
        print("RESULT: Issues found. Please review the dataset.")


def check_class_consistency(
    ds_train: sv.DetectionDataset,
    ds_valid: sv.DetectionDataset,
    ds_test: sv.DetectionDataset
) -> None:
    """
    Verifies that all three datasets have the same classes.

    Args:
        ds_train: Training dataset
        ds_valid: Validation dataset
        ds_test: Test dataset
    """
    print("--- Verifying Class Consistency ---")

    train_classes = set(ds_train.classes)
    valid_classes = set(ds_valid.classes)
    test_classes = set(ds_test.classes)

    if train_classes == valid_classes == test_classes:
        print("The classes match exactly across all three datasets.")
        print(f"   Total classes: **{len(train_classes)}**")
        print(f"   Class List: {sorted(train_classes)}")
    else:
        print("⚠️ WARNING: Classes do not match across datasets")
        print(f"   Train only: {train_classes - valid_classes - test_classes}")
        print(f"   Valid only: {valid_classes - train_classes - test_classes}")
        print(f"   Test only: {test_classes - train_classes - valid_classes}")

    print("-----------------------------------\n")


def check_missing_labels(dataset_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Checks how many images DO NOT have their corresponding label file.

    Args:
        dataset_path: The root path of the dataset

    Returns:
        A dictionary containing the results for each split
    """
    splits: List[str] = ['train', 'valid', 'test']
    valid_img_ext: Set[str] = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    print("🔍 Checking images for missing label files...")
    print("=" * 80)

    total_missing: int = 0
    results: Dict[str, Dict[str, Any]] = {}

    for split in splits:
        images_path: Path = Path(dataset_path) / split / "images"
        labels_path: Path = Path(dataset_path) / split / "labels"

        if not images_path.exists():
            print(f"\n⚠️  {split.upper()}: Images folder not found")
            continue

        if not labels_path.exists():
            print(f"\n⚠️  {split.upper()}: Labels folder not found")
            continue

        image_stems: Set[str] = {f.stem for f in images_path.glob("*")
                                 if f.suffix.lower() in valid_img_ext}
        label_stems: Set[str] = {f.stem for f in labels_path.glob("*.txt")}
        missing_labels: Set[str] = image_stems - label_stems

        results[split] = {
            'total_images': len(image_stems),
            'total_labels': len(label_stems),
            'missing': len(missing_labels),
            'missing_files': sorted(missing_labels)
        }

        total_missing += len(missing_labels)

        print(f"\n{split.upper()}:")
        print(f"  Total Images:              {len(image_stems):,}")
        print(f"  Total .txt Labels:         {len(label_stems):,}")
        print(f"  Images WITHOUT Label:      {len(missing_labels):,}")

        if len(missing_labels) > 0:
            print(f"  ❌ CRITICAL: {len(missing_labels)} images are missing a label file")
            print(f"\n  Examples of unlabeled images (first 5):")
            for i, missing in enumerate(list(missing_labels)[:5]):
                for ext in valid_img_ext:
                    candidate: Path = images_path / f"{missing}{ext}"
                    if candidate.exists():
                        print(f"    {i+1}. {candidate}")
                        break

            if len(missing_labels) > 5:
                print(f"    ... and {len(missing_labels) - 5} more.")
        else:
            print(f"  ✅ All images have their corresponding label file")

    print("\n" + "=" * 80)
    print("GENERAL SUMMARY:")
    print("=" * 80)

    for split in splits:
        if split in results:
            r = results[split]
            status: str = "✅ OK" if r['missing'] == 0 else f"❌ {r['missing']} unlabeled"
            print(f"{split.upper():<10} | Images: {r['total_images']:>6,} | "
                  f"Labels: {r['total_labels']:>6,} | {status}")

    print("=" * 80)

    if total_missing > 0:
        print(f"\n⚠️  TOTAL IMAGES WITHOUT LABEL: {total_missing}")
        print("\nRECOMMENDATION:")
        print("  1. Verify if these images should have annotations")
        print("  2. If they are background images (no objects), create empty .txt files")
        print("  3. If annotations are missing, you need to add them or delete the images")
    else:
        print(f"\n✅ PERFECT: All images have their corresponding label file")

    return results
