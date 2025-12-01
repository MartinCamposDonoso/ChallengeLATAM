"""
Data Leakage Detection Functions

This module contains functions to detect data leakage between train/valid/test splits.
"""

import matplotlib.pyplot as plt
import cv2
import os
import hashlib
import pandas as pd
from collections import defaultdict
from glob import glob
import numpy as np
from typing import Dict, List, Tuple, Optional


# Colors for the report (HEX Format)
COLORS = {
    'train': '#1f77b4',  # Blue
    'valid': '#ff7f0e',  # Orange
    'test':  '#d62728'   # Red
}


def get_file_hash_safe(filepath: str) -> Optional[str]:
    """
    Calculates hash while handling long Windows paths.

    Args:
        filepath: Path to the file

    Returns:
        MD5 hash of the file or None if error
    """
    abs_path = os.path.abspath(filepath)
    if os.name == 'nt' and not abs_path.startswith('\\\\?\\'):
        abs_path = '\\\\?\\' + abs_path

    hasher = hashlib.md5()
    try:
        with open(abs_path, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
    except Exception:
        return None


def load_image_safe(filepath: str) -> Optional[np.ndarray]:
    """
    Loads image with OpenCV while handling long paths.

    Args:
        filepath: Path to the image

    Returns:
        Image in RGB format or None if error
    """
    abs_path = os.path.abspath(filepath)
    if os.name == 'nt' and not abs_path.startswith('\\\\?\\'):
        abs_path = '\\\\?\\' + abs_path

    try:
        with open(abs_path, "rb") as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                return None
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        return None


def add_border(img: np.ndarray, color_hex: str, thickness: int = 10) -> np.ndarray:
    """
    Adds a colored border to the image to identify the split.

    Args:
        img: Input image
        color_hex: Border color in hex format
        thickness: Border thickness in pixels

    Returns:
        Image with border
    """
    h = color_hex.lstrip('#')
    rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    return cv2.copyMakeBorder(
        img, thickness, thickness, thickness, thickness,
        cv2.BORDER_CONSTANT, value=rgb
    )


def generate_audit_report(
    dataset_path: str,
    output_dir: str = "leakage_report"
) -> Optional[pd.DataFrame]:
    """
    Generates a complete audit report for data leakage detection.

    Args:
        dataset_path: Root path of the dataset
        output_dir: Folder where evidence will be saved

    Returns:
        DataFrame with leakage summary or None if no leakage found
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    hashes = defaultdict(list)
    splits = ['train', 'valid', 'test']

    print("1. Scanning dataset...")
    for split in splits:
        files = glob(os.path.join(dataset_path, split, "images", "*"))
        for f in files:
            h = get_file_hash_safe(f)
            if h:
                hashes[h].append((split, f))

    # Filter for leakages
    leakages = []
    for h, occ in hashes.items():
        if len(occ) > 1:
            unique_splits = set(x[0] for x in occ)
            if len(unique_splits) > 1:
                leakages.append((h, occ))

    print(f"2. Found {len(leakages)} leakage cases.")
    print(f"3. Generating evidence in folder '{output_dir}'...")

    report_data = []

    for i, (file_hash, group) in enumerate(leakages):
        num_imgs = len(group)
        fig, axes = plt.subplots(1, num_imgs, figsize=(6 * num_imgs, 6))
        if num_imgs == 1:
            axes = [axes]

        splits_involved = " vs ".join(sorted(list(set(g[0].upper() for g in group))))
        fig.suptitle(
            f"EVIDENCE #{i+1}: {splits_involved}\nMD5 Hash: {file_hash}",
            fontsize=16, weight='bold', y=0.95
        )

        row_data = {
            "Case_ID": i+1,
            "Hash": file_hash,
            "Splits_Involved": splits_involved
        }

        for idx, (split, filepath) in enumerate(group):
            img = load_image_safe(filepath)
            filename = os.path.basename(filepath)

            row_data[f"File_{idx+1}_Split"] = split
            row_data[f"File_{idx+1}_Name"] = filename

            if img is not None:
                img_border = add_border(img, COLORS[split], thickness=20)

                axes[idx].imshow(img_border)

                title_obj = axes[idx].set_title(
                    f"{split.upper()}",
                    fontsize=14, weight='bold', color='white', pad=10
                )
                title_obj.set_backgroundcolor(COLORS[split])

                axes[idx].set_xlabel(f"\n{filename}", fontsize=9, style='italic')
            else:
                axes[idx].text(0.5, 0.5, "Image Error", ha='center')

            axes[idx].set_xticks([])
            axes[idx].set_yticks([])
            for spine in axes[idx].spines.values():
                spine.set_visible(False)

        plt.tight_layout()

        save_path = os.path.join(output_dir, f"leakage_case_{i+1:03d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        report_data.append(row_data)

    if report_data:
        df = pd.DataFrame(report_data)
        csv_path = os.path.join(output_dir, "leakage_summary.csv")
        df.to_csv(csv_path, index=False)
        print("Report successfully generated.")
        print(f"  - Images: {output_dir}/*.png")
        print(f"  - Excel/CSV: {csv_path}")

        return df
    else:
        print("Dataset is clean. No report to generate.")
        return None
