"""
Distribution Visualization Functions

This module contains functions to visualize dataset distributions.
"""

from collections import Counter
import matplotlib.pyplot as plt
import supervision as sv
from typing import Dict


def plot_dataset_sizes_and_percentages(
    ds_train: sv.DetectionDataset,
    ds_valid: sv.DetectionDataset,
    ds_test: sv.DetectionDataset
) -> None:
    """
    Plots the distribution of images across train, valid, and test sets.

    Args:
        ds_train: Training dataset
        ds_valid: Validation dataset
        ds_test: Test dataset
    """
    sizes = [len(ds_train), len(ds_valid), len(ds_test)]
    labels = ['Train', 'Valid', 'Test']
    colors = ['#1f77b4', '#ff7f0e', '#d62728']

    total = sum(sizes)
    percentages = [f"{(size/total*100):.1f}%" for size in sizes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart
    bars = ax1.bar(labels, sizes, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Number of Images', fontsize=12)
    ax1.set_title('Dataset Size Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(sizes) * 1.1)

    for bar, size, pct in zip(bars, sizes, percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{size:,}\n({pct})',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Pie chart
    wedges, texts, autotexts = ax2.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 12, 'weight': 'bold'}
    )

    for autotext in autotexts:
        autotext.set_color('white')

    ax2.set_title(f'Dataset Distribution\nTotal: {total:,} images',
                  fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()


def get_image_counts(dataset: sv.DetectionDataset) -> Counter:
    """
    Obtains the count of images per class.

    Args:
        dataset: Detection dataset

    Returns:
        Counter with class names and their counts
    """
    image_counts = Counter()
    for detections in dataset.annotations.values():
        unique_class_ids = set(detections.class_id)
        for class_id in unique_class_ids:
            class_name = dataset.classes[class_id]
            image_counts[class_name] += 1
    return image_counts


def plot_pie_chart_with_legend(
    dataset: sv.DetectionDataset,
    split_name: str,
    colors_dict: Dict[str, tuple]
) -> None:
    """
    Creates a pie chart with detailed legend for class distribution.

    Args:
        dataset: Detection dataset
        split_name: Name of the split (Train/Valid/Test)
        colors_dict: Dictionary mapping class names to colors
    """
    image_counts = get_image_counts(dataset)
    total_images = len(dataset)

    # Sort data by count (descending)
    data_sorted = sorted(image_counts.items(), key=lambda x: x[1], reverse=True)
    classes = [item[0] for item in data_sorted]
    values = [item[1] for item in data_sorted]
    percentages = [(v / sum(values) * 100) for v in values]

    colors = [colors_dict[cls] for cls in classes]

    # Create figure
    fig, (ax_pie, ax_legend) = plt.subplots(
        1, 2,
        figsize=(16, 8),
        gridspec_kw={'width_ratios': [2, 1]}
    )

    # Custom function to show only percentages >10%
    def autopct_format(pct):
        return f'{pct:.1f}%' if pct > 10 else ''

    # Create pie chart
    wedges, texts, autotexts = ax_pie.pie(
        values,
        labels=None,
        autopct=autopct_format,
        startangle=90,
        colors=colors,
        textprops={'fontsize': 10, 'weight': 'bold'},
        pctdistance=0.85
    )

    # Improve readability of percentages
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_weight('bold')

    ax_pie.set_title(
        f'{split_name} Set - Class Distribution\nTotal Images: {total_images:,}',
        fontsize=16, fontweight='bold', pad=20
    )

    # Create detailed legend
    ax_legend.axis('off')

    legend_title = f"Class Breakdown ({len(classes)} classes)"
    ax_legend.text(0.05, 0.95, legend_title, fontsize=14, fontweight='bold',
                   transform=ax_legend.transAxes, verticalalignment='top')

    y_position = 0.90
    y_step = 0.85 / len(classes) if len(classes) > 0 else 0.05

    for cls, val, pct, color in zip(classes, values, percentages, colors):
        # Color box
        ax_legend.add_patch(
            plt.Rectangle((0.05, y_position - 0.015), 0.03, 0.03,
                         facecolor=color, edgecolor='black', linewidth=1,
                         transform=ax_legend.transAxes)
        )

        # Class name
        class_text = f"{cls[:25]}" if len(cls) <= 25 else f"{cls[:22]}..."
        ax_legend.text(0.10, y_position, class_text, fontsize=9,
                      transform=ax_legend.transAxes, verticalalignment='center')

        # Count and percentage
        stats_text = f"{val:,} ({pct:.1f}%)"
        ax_legend.text(0.70, y_position, stats_text, fontsize=9, fontweight='bold',
                      transform=ax_legend.transAxes, verticalalignment='center',
                      horizontalalignment='right')

        y_position -= y_step

    plt.tight_layout()
    plt.show()
