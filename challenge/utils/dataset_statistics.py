"""
Dataset Statistics and Reporting Functions

This module contains functions to generate statistical reports from datasets.
"""

from collections import Counter
import pandas as pd
import supervision as sv
from typing import Dict, Tuple


def get_image_counts(dataset: sv.DetectionDataset) -> Counter:
    """
    Obtains the count of images per class.

    Args:
        dataset: Detection dataset

    Returns:
        Counter with class names and their image counts
    """
    image_counts = Counter()
    for detections in dataset.annotations.values():
        unique_class_ids = set(detections.class_id)
        for class_id in unique_class_ids:
            class_name = dataset.classes[class_id]
            image_counts[class_name] += 1
    return image_counts


def generate_comparison_report(
    ds_train: sv.DetectionDataset,
    ds_valid: sv.DetectionDataset,
    ds_test: sv.DetectionDataset
):
    """
    Generates a comparison report of class distribution across all splits.

    Args:
        ds_train: Training dataset
        ds_valid: Validation dataset
        ds_test: Test dataset

    Returns:
        Dictionary with styled DataFrames for the report
    """
    datasets = {
        'Train': ds_train,
        'Valid': ds_valid,
        'Test': ds_test
    }

    # Collect statistics for each dataset
    stats_data = []

    for split_name, dataset in datasets.items():
        class_counts = Counter()

        # Count detections per class
        for detections in dataset.annotations.values():
            for class_id in detections.class_id:
                class_name = dataset.classes[class_id]
                class_counts[class_name] += 1

        # Add to stats
        for class_name, count in class_counts.items():
            stats_data.append({
                'Split': split_name,
                'Class': class_name,
                'Count': count
            })

    # Create DataFrame
    df = pd.DataFrame(stats_data)

    # Pivot table for better visualization
    pivot_df = df.pivot(index='Class', columns='Split', values='Count').fillna(0).astype(int)

    # Add total column
    pivot_df['Total'] = pivot_df.sum(axis=1)

    # Sort by total descending
    pivot_df = pivot_df.sort_values('Total', ascending=False)

    # Calculate percentages
    percent_df = pivot_df.copy()
    for col in ['Train', 'Valid', 'Test']:
        if col in percent_df.columns:
            total = percent_df[col].sum()
            percent_df[f'{col} %'] = (percent_df[col] / total * 100).round(2)

    # Style the DataFrames
    styled_count = pivot_df.style.background_gradient(cmap='YlOrRd', subset=['Train', 'Valid', 'Test'])
    styled_percent = percent_df[[col for col in percent_df.columns if '%' in col]].style.background_gradient(
        cmap='Blues'
    )

    print("--- Dataset Statistics Generated Successfully ---\n")

    return {
        'Count Distribution': styled_count,
        'Percentage Distribution': styled_percent
    }
