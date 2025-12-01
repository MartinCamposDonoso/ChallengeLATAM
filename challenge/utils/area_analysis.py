"""
Detection Area Analysis Functions

This module contains functions to analyze and visualize detection areas.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import supervision as sv
import json
import numpy as np
import os
import cv2
from typing import Union, Dict, List, Tuple


def plot_area_distribution(
    train_ds: sv.DetectionDataset,
    valid_ds: sv.DetectionDataset,
    test_ds: sv.DetectionDataset,
    log_scale: bool = False
) -> Union[pd.DataFrame, None]:
    """
    Generates a box plot comparing the distribution of detection area (in pixels)
    by class and by split (Train/Valid/Test).

    Args:
        train_ds: The training dataset
        valid_ds: The validation dataset
        test_ds: The test dataset
        log_scale: If True, uses a logarithmic scale for the Y-axis

    Returns:
        A summary table of median areas per class and split, or None if no detections
    """
    data: List[Dict[str, any]] = []
    datasets: Dict[str, sv.DetectionDataset] = {
        "Train": train_ds,
        "Valid": valid_ds,
        "Test": test_ds
    }

    print("Collecting area data...")

    for split_name, dataset in datasets.items():
        for detections in dataset.annotations.values():
            areas = detections.area
            class_ids = detections.class_id

            for area, class_id in zip(areas, class_ids):
                if class_id < len(dataset.classes):
                    class_name = dataset.classes[class_id]
                else:
                    class_name = "Unknown"

                data.append({
                    "Area (px)": area,
                    "Class": class_name,
                    "Split": split_name
                })

    df = pd.DataFrame(data)

    if df.empty:
        print("⚠️ No detections found to plot.")
        return None

    print(f"Total of {len(df)} detections found.")
    plt.figure(figsize=(16, 8))
    sns.set_style("whitegrid")

    ax = sns.boxplot(
        data=df,
        x="Class",
        y="Area (px)",
        hue="Split",
        palette={"Train": "#1f77b4", "Valid": "#ff7f0e", "Test": "#d62728"},
        linewidth=1.5,
        fliersize=2
    )

    plt.title("Distribution of Detection Area by Class and Dataset Split",
              fontsize=16, pad=20)
    plt.xlabel("Class", fontsize=12)
    plt.xticks(rotation=45, ha='right')

    if log_scale:
        ax.set_yscale("log")
        plt.ylabel("Area in Pixels (Logarithmic Scale)", fontsize=12)
        plt.title("Distribution of Area (Log) by Class and Dataset Split", fontsize=16)
    else:
        plt.ylabel("Area in Pixels²", fontsize=12)

    plt.tight_layout()
    plt.show()

    print("\nSummary of Median Areas (px):")
    summary = df.groupby(['Class', 'Split'])['Area (px)'].median().unstack()
    print(summary)
    return df


def calculate_area_ranges_json(
    df: pd.DataFrame,
    filename: str = "area_ranges.json"
) -> str:
    """
    Calculates detection area ranges by class and saves result in JSON format.

    Args:
        df: DataFrame with area data
        filename: Output JSON filename

    Returns:
        JSON string with area ranges
    """
    if df.empty:
        return json.dumps({"error": "DataFrame is empty. No data to calculate."})

    results = {}

    grouped = df.groupby('Class')['Area (px)']
    q1 = grouped.quantile(0.25)
    q3 = grouped.quantile(0.75)
    iqr = q3 - q1
    ltl = q1 - 1.5 * iqr
    utl = q3 + 1.5 * iqr
    max_area = grouped.max()

    for class_name in df['Class'].unique():
        _ltl = float(max(0.0, ltl.loc[class_name]))
        _q1 = float(q1.loc[class_name])
        _q3 = float(q3.loc[class_name])
        _utl = float(utl.loc[class_name])
        _max_area = float(max_area.loc[class_name])

        results[class_name] = {
            "Range_0_Lower_Outlier": {
                "inicio": 0.0,
                "fin": _ltl,
                "description": "Area from 0 up to the Theoretical Lower Limit (LTL)"
            },
            "Range_1_Lower_Whisker_to_Q1": {
                "inicio": _ltl,
                "fin": _q1,
                "description": "From LTL to the first quartile (Q1)"
            },
            "Range_2_Q1_to_Q3_Box": {
                "inicio": _q1,
                "fin": _q3,
                "description": "The box body: from Q1 to Q3"
            },
            "Range_3_Q3_to_Upper_Whisker": {
                "inicio": _q3,
                "fin": _utl,
                "description": "From Q3 to the Theoretical Upper Limit (UTL)"
            },
            "Range_4_Upper_Outlier_to_Max": {
                "inicio": _utl,
                "fin": _max_area,
                "description": "From UTL to the actual maximum area"
            }
        }

    json_output = json.dumps(results, indent=4)

    try:
        with open(filename, 'w') as f:
            f.write(json_output)
        print(f"\n✅ Area ranges successfully saved to '{filename}'")
    except IOError:
        print(f"\n❌ Error saving file to '{filename}'. Printing to console instead.")

    return json_output


def visualize_area_grid_per_class(
    train_ds: sv.DetectionDataset,
    valid_ds: sv.DetectionDataset,
    test_ds: sv.DetectionDataset,
    images_base_path: str,
    json_path: str = "deteccion_area_ranges.json",
    samples_per_range: int = 2
) -> None:
    """
    Generates a separate grid (2 rows x 5 columns) for each class showing examples
    from the 5 area ranges.

    Args:
        train_ds: Training dataset
        valid_ds: Validation dataset
        test_ds: Test dataset
        images_base_path: Base path to images
        json_path: Path to JSON with area ranges
        samples_per_range: Number of examples per range
    """
    try:
        with open(json_path, 'r') as f:
            area_ranges_json = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Archivo JSON no encontrado en la ruta: {json_path}")
        return

    dataset = train_ds
    box_annotator = sv.BoxAnnotator()
    class_names = list(area_ranges_json.keys())

    for class_name in class_names:
        all_annotated_images: List[Tuple[str, np.ndarray]] = []
        ranges = area_ranges_json[class_name]

        print(f"\n{'='*56}")
        print(f"--- 🖼️ Procesando CLASE: {class_name} ({len(ranges)} Rangos) ---")
        print(f"{'='*56}")

        for range_key, range_data in ranges.items():
            min_area = range_data['inicio']
            max_area = range_data['fin']

            if min_area >= max_area:
                continue

            print(f"  > Buscando muestras para Rango: {range_key} "
                  f"({min_area:.2f} a {max_area:.2f} px²)")

            image_names = list(dataset.annotations.keys())
            image_sample_candidates = np.random.choice(
                image_names,
                min(len(image_names), 150),
                replace=False
            )

            samples_found_for_this_slot = 0

            for image_name in image_sample_candidates:
                if samples_found_for_this_slot >= samples_per_range:
                    break

                detections = dataset.annotations[image_name]

                try:
                    target_class_id = dataset.classes.index(class_name)
                except ValueError:
                    continue

                class_mask = detections.class_id == target_class_id
                area_mask = (detections.area > min_area) & (detections.area <= max_area)
                combined_mask = class_mask & area_mask

                filtered_detections = detections[combined_mask]

                if len(filtered_detections) > 0:
                    full_image_path = os.path.join(images_base_path, image_name)
                    image_bgr = cv2.imread(full_image_path)

                    if image_bgr is None:
                        continue

                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    annotated_image = box_annotator.annotate(
                        scene=image_rgb,
                        detections=filtered_detections
                    )

                    title = f"{range_key.replace('_', ' ')}\n({len(filtered_detections)} Dets)"
                    all_annotated_images.append((title, annotated_image))

                    samples_found_for_this_slot += 1

        if not all_annotated_images:
            print(f"⚠️ No se encontraron ejemplos para la clase {class_name} en ningún rango.")
            continue

        titles, images = zip(*all_annotated_images)

        max_slots = 10
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()

        fig.suptitle(
            f"Distribución de Detecciones por Área - Clase: {class_name}",
            fontsize=16
        )

        for i in range(max_slots):
            ax = axes[i]
            if i < len(all_annotated_images):
                ax.imshow(images[i])
                ax.set_title(titles[i], fontsize=10, wrap=True)
                ax.axis('off')
            else:
                ax.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


def visualize_area_grid_5x5_optimized(
    train_ds: sv.DetectionDataset,
    json_path: str = "deteccion_area_ranges.json",
    samples_per_range: int = 5,
    max_iterations: int = None
) -> None:
    """
    Optimized version: Uses dataset iterator and early sampling.
    Stops when enough samples are found for each range.

    Args:
        train_ds: Training dataset
        json_path: Path to JSON with area ranges
        samples_per_range: Number of samples per range
        max_iterations: Maximum iterations (None = all dataset)
    """
    try:
        with open(json_path, 'r') as f:
            area_ranges_json = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Archivo JSON no encontrado en la ruta: {json_path}")
        return

    label_annotator = sv.LabelAnnotator()
    box_annotator = sv.BoxAnnotator(thickness=8)
    class_names = list(area_ranges_json.keys())

    samples_found: Dict[str, Dict[str, List[Tuple[str, np.ndarray]]]] = {
        cls: {rng: [] for rng in area_ranges_json[cls].keys()} for cls in class_names
    }

    def all_ranges_complete(cls):
        return all(
            len(samples_found[cls][rng]) >= samples_per_range
            for rng in area_ranges_json[cls].keys()
        )

    def all_complete():
        return all(all_ranges_complete(cls) for cls in class_names)

    print("--- ⏱️ Buscando muestras en el dataset... ---")

    processed = 0
    for image_path, image_rgb, detections in train_ds:

        if detections is None or len(detections) == 0:
            continue

        for class_name in class_names:

            if all_ranges_complete(class_name):
                continue

            try:
                target_class_id = train_ds.classes.index(class_name)
            except ValueError:
                continue

            class_mask = detections.class_id == target_class_id
            if not np.any(class_mask):
                continue

            class_detections = detections[class_mask]
            areas = class_detections.area

            for range_key, range_data in area_ranges_json[class_name].items():

                if len(samples_found[class_name][range_key]) >= samples_per_range:
                    continue

                min_area = range_data['inicio']
                max_area = range_data['fin']

                area_mask = (areas > min_area) & (areas <= max_area)

                if np.any(area_mask):
                    range_detections = class_detections[area_mask]

                    annotated_image = box_annotator.annotate(
                        scene=image_rgb.copy(),
                        detections=range_detections
                    )
                    labels = [
                        train_ds.classes[class_id]
                        for class_id in range_detections.class_id
                    ]
                    annotated_image = label_annotator.annotate(
                        scene=annotated_image,
                        detections=range_detections,
                        labels=labels
                    )
                    title = f"{range_key.replace('_', ' ')}\n({len(range_detections)} Dets)"

                    samples_found[class_name][range_key].append((title, annotated_image))

        processed += 1

        if processed % 1000 == 0:
            print(f"   Procesadas {processed} imágenes...")

        if all_complete():
            print(f"✅ ¡Todas las muestras encontradas! Procesadas {processed} imágenes.")
            break

        if max_iterations and processed >= max_iterations:
            print(f"⚠️ Alcanzado límite de {max_iterations} imágenes.")
            break

    print(f"--- ✅ Búsqueda finalizada. Total procesadas: {processed} imágenes ---")

    for class_name in class_names:

        all_samples_for_class: List[Tuple[str, np.ndarray]] = []

        for range_key in area_ranges_json[class_name].keys():

            available = samples_found[class_name][range_key]

            selected = available[:samples_per_range]

            while len(selected) < samples_per_range:
                selected.append((range_key.replace('_', ' '), None))

            all_samples_for_class.extend(selected)

        if all(img is None for title, img in all_samples_for_class):
            print(f"⚠️ No se encontraron ejemplos para la clase '{class_name}'.")
            continue

        titles, images = zip(*all_samples_for_class)

        rows, cols = 5, 5
        fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
        axes = axes.flatten()

        fig.suptitle(
            f"Distribución de Detecciones por Área - Clase: {class_name}\n"
            f"(5 Rangos x {samples_per_range} Ejemplos)",
            fontsize=18, y=1.0
        )

        for i in range(rows * cols):
            ax = axes[i]
            title, image = titles[i], images[i]

            if image is not None:
                ax.imshow(image)
                if i % cols == 0:
                    ax.set_title(title, fontsize=10, loc='left', wrap=True)
                else:
                    ax.set_title(f"Ejemplo {i % cols + 1}", fontsize=8)
            else:
                ax.text(0.5, 0.5, 'SIN DETECCIONES', ha='center', va='center', fontsize=10)
                if i % cols == 0:
                    ax.set_title(title, fontsize=10, loc='left', wrap=True)

            ax.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show()
