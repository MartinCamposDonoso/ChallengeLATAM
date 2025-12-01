# Utils 2 - Dataset Exploration Utilities

This package contains modular utilities extracted from `01_dataset_exploration.ipynb` for clean, reusable, and maintainable code.

## 📁 Package Structure

```
utils_2/
├── __init__.py                      # Package initialization and exports
├── dataset_integrity.py             # Dataset integrity verification functions
├── dataset_loader.py                # Dataset loading utilities
├── dataset_statistics.py            # Statistical analysis and reporting
├── visualization_distributions.py   # Distribution visualization functions
├── area_analysis.py                 # Detection area analysis and visualization
├── leakage_detection.py            # Data leakage detection between splits
├── embedding_utils.py              # Embedding extraction and visualization
└── README.md                        # This file
```

## 📦 Modules Overview

### 1. `dataset_integrity.py`
Functions to verify dataset integrity and consistency.

**Functions:**
- `check_integrity(dataset_path)` - Verify image-label correspondence
- `check_class_consistency(ds_train, ds_valid, ds_test)` - Check class consistency across splits
- `check_missing_labels(dataset_path)` - Find images without label files

**Example:**
```python
from utils_2 import check_integrity, check_class_consistency

check_integrity("../data")
check_class_consistency(ds_train, ds_valid, ds_test)
```

### 2. `dataset_loader.py`
Functions to load datasets in supervision format.

**Functions:**
- `get_ds_from_supervision(dataset_path)` - Load train/valid/test datasets

**Example:**
```python
from utils_2 import get_ds_from_supervision

ds_train, ds_valid, ds_test = get_ds_from_supervision("../data")
```

### 3. `dataset_statistics.py`
Statistical analysis and report generation.

**Functions:**
- `get_image_counts(dataset)` - Count images per class
- `generate_comparison_report(ds_train, ds_valid, ds_test)` - Generate comparison statistics

**Example:**
```python
from utils_2 import generate_comparison_report

stats_report = generate_comparison_report(ds_train, ds_valid, ds_test)
for table_name, styled_df in stats_report.items():
    display(styled_df)
```

### 4. `visualization_distributions.py`
Visualization functions for dataset distributions.

**Functions:**
- `plot_dataset_sizes_and_percentages(ds_train, ds_valid, ds_test)` - Plot dataset size distribution
- `plot_pie_chart_with_legend(dataset, split_name, colors_dict)` - Create pie chart with legend

**Example:**
```python
from utils_2 import plot_dataset_sizes_and_percentages, plot_pie_chart_with_legend

plot_dataset_sizes_and_percentages(ds_train, ds_valid, ds_test)

# Create color palette
all_classes = sorted(set(ds_train.classes))
colors_dict = {cls: plt.cm.tab20c(i) for i, cls in enumerate(all_classes)}

plot_pie_chart_with_legend(ds_train, "Training", colors_dict)
```

### 5. `area_analysis.py`
Analysis and visualization of detection areas.

**Functions:**
- `plot_area_distribution(train_ds, valid_ds, test_ds, log_scale=False)` - Plot area boxplots
- `calculate_area_ranges_json(df, filename)` - Calculate and save area ranges to JSON
- `visualize_area_grid_per_class(...)` - Visualize area grid per class (2x5)
- `visualize_area_grid_5x5_optimized(...)` - Optimized 5x5 grid visualization

**Example:**
```python
from utils_2 import plot_area_distribution, calculate_area_ranges_json

df_detections = plot_area_distribution(ds_train, ds_valid, ds_test)
json_ranges = calculate_area_ranges_json(df_detections, "area_ranges.json")
```

### 6. `leakage_detection.py`
Detect data leakage between train/valid/test splits.

**Functions:**
- `get_file_hash_safe(filepath)` - Calculate file hash safely
- `load_image_safe(filepath)` - Load image handling long paths
- `add_border(img, color_hex, thickness)` - Add colored border to images
- `generate_audit_report(dataset_path, output_dir)` - Generate full leakage report

**Example:**
```python
from utils_2 import generate_audit_report

df_report = generate_audit_report("../data", output_dir="leakage_report")
if df_report is not None:
    display(df_report)
```

### 7. `embedding_utils.py`
Utilities for embedding extraction and interactive visualization.

**Functions:**
- `image_to_data_uri(image_path)` - Convert image to base64 URI
- `pil_image_to_data_uri(image)` - Convert PIL image to base64 URI
- `display_projections(labels, projections, images, ...)` - Interactive 3D scatter plot

**Example:**
```python
from utils_2 import display_projections

display_projections(clusters, projections, crops_pil)
```

## 🚀 Usage

### Quick Start

```python
# Import all utilities
from utils_2 import (
    check_integrity,
    get_ds_from_supervision,
    plot_dataset_sizes_and_percentages,
    generate_audit_report
)

# Load and verify dataset
DATASET_PATH = "../data"
check_integrity(DATASET_PATH)

ds_train, ds_valid, ds_test = get_ds_from_supervision(DATASET_PATH)

# Visualize distributions
plot_dataset_sizes_and_percentages(ds_train, ds_valid, ds_test)

# Check for data leakage
df_report = generate_audit_report(DATASET_PATH)
```

### Using in Notebooks

The refactored notebook `02_dataset_exploration_refactored.ipynb` demonstrates the complete usage of all utilities.

## 📋 Requirements

- Python 3.8+
- supervision
- pandas
- numpy
- matplotlib
- seaborn
- opencv-python (cv2)
- torch
- transformers
- plotly
- scikit-learn
- umap-learn
- more-itertools
- tqdm
- Pillow

## 🎯 Benefits of This Structure

1. **Modularity**: Each module focuses on a specific aspect of dataset exploration
2. **Reusability**: Functions can be imported and used across different notebooks
3. **Maintainability**: Changes to functions only need to be made in one place
4. **Testability**: Each function can be tested independently
5. **Documentation**: Clear docstrings and organized structure
6. **Clean Notebooks**: Notebooks focus on analysis flow, not implementation details

## 📝 Notes

- All functions include comprehensive docstrings
- Error handling is implemented for robust operation
- Functions support both Windows and Unix paths
- Visualization functions use consistent color schemes
- Optimized versions available for computationally intensive operations

## 🔄 Migration from Original Notebook

The original notebook `01_dataset_exploration.ipynb` has been refactored into:
- **Modular Python files**: All functions extracted to `utils_2/`
- **Clean notebook**: `02_dataset_exploration_refactored.ipynb` uses only function calls

This approach separates concerns and makes the codebase more professional and maintainable.
