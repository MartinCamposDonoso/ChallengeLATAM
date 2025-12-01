"""
Utils 2 Package - Dataset Exploration Utilities

This package contains modular utilities for dataset exploration and analysis.
"""

# Dataset Integrity
from .dataset_integrity import (
    check_integrity,
    check_class_consistency,
    check_missing_labels
)

# Dataset Loading
from .dataset_loader import get_ds_from_supervision

# Dataset Statistics
from .dataset_statistics import (
    get_image_counts,
    generate_comparison_report
)

# Visualization Distributions
from .visualization_distributions import (
    plot_dataset_sizes_and_percentages,
    plot_pie_chart_with_legend
)

# Area Analysis
from .area_analysis import (
    plot_area_distribution,
    calculate_area_ranges_json,
    visualize_area_grid_per_class,
    visualize_area_grid_5x5_optimized
)

# Leakage Detection
from .leakage_detection import (
    get_file_hash_safe,
    load_image_safe,
    add_border,
    generate_audit_report
)

# Embedding Utils
from .embedding_utils import (
    image_to_data_uri,
    pil_image_to_data_uri,
    display_projections
)

__all__ = [
    # Dataset Integrity
    'check_integrity',
    'check_class_consistency',
    'check_missing_labels',

    # Dataset Loading
    'get_ds_from_supervision',

    # Dataset Statistics
    'get_image_counts',
    'generate_comparison_report',

    # Visualization Distributions
    'plot_dataset_sizes_and_percentages',
    'plot_pie_chart_with_legend',

    # Area Analysis
    'plot_area_distribution',
    'calculate_area_ranges_json',
    'visualize_area_grid_per_class',
    'visualize_area_grid_5x5_optimized',

    # Leakage Detection
    'get_file_hash_safe',
    'load_image_safe',
    'add_border',
    'generate_audit_report',

    # Embedding Utils
    'image_to_data_uri',
    'pil_image_to_data_uri',
    'display_projections',
]
