"""
Embedding and Visualization Utilities

This module contains functions for embeddings extraction and interactive visualization.
"""

import base64
import json
import os
from io import BytesIO
from typing import List, Dict
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from IPython.core.display import display, HTML, Javascript


def image_to_data_uri(image_path: str) -> str:
    """
    Converts an image from path to a base64 Data URI.

    Args:
        image_path: Path to the image file

    Returns:
        Data URI string or empty string if error
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        return "data:image/jpeg;base64," + encoded_image
    except FileNotFoundError:
        print(f"Error: Archivo no encontrado en la ruta {image_path}")
        return ""


def pil_image_to_data_uri(image: Image.Image) -> str:
    """
    Converts a PIL Image to a base64 Data URI.

    Args:
        image: PIL Image object

    Returns:
        Data URI string
    """
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=80)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"


def display_projections(
    labels: np.ndarray,
    projections: np.ndarray,
    images: List[Image.Image],
    show_legend: bool = False,
    show_markers_with_text: bool = True
) -> None:
    """
    Displays interactive 3D scatter plot with image preview on click.

    Args:
        labels: Array of labels for each point
        projections: 3D projections (N x 3 array)
        images: List of PIL Images
        show_legend: Whether to show legend
        show_markers_with_text: Whether to show text on markers
    """
    PLOTLY_DIV_ID = 'plotly-scatter-plot-3d'

    # Prepare Data URIs
    image_data_uris = {
        f"image_{i}": pil_image_to_data_uri(image)
        for i, image in enumerate(images)
    }
    image_ids = np.array([f"image_{i}" for i in range(len(images))])

    unique_labels = np.unique(labels)
    traces = []

    for unique_label in unique_labels:
        mask = labels == unique_label
        customdata_masked = image_ids[mask]
        trace = go.Scatter3d(
            x=projections[mask][:, 0],
            y=projections[mask][:, 1],
            z=projections[mask][:, 2],
            mode='markers+text' if show_markers_with_text else 'markers',
            text=labels[mask],
            customdata=customdata_masked,
            name=str(unique_label),
            marker=dict(size=8),
            hovertemplate="<b>class: %{text}</b><br>image ID: %{customdata}<extra></extra>"
        )
        traces.append(trace)

    # Calculate shared range for cube appearance
    all_axes = projections
    min_val = np.min(all_axes)
    max_val = np.max(all_axes)
    padding = (max_val - min_val) * 0.05
    axis_range = [min_val - padding, max_val + padding]

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', range=axis_range),
            yaxis=dict(title='Y', range=axis_range),
            zaxis=dict(title='Z', range=axis_range),
            aspectmode='cube'
        ),
        width=1000,
        height=1000,
        showlegend=show_legend
    )

    fig.show(config={'displayModeBar': False})

    # Image display HTML
    image_display_html = """
    <div id="image-container" style="
        position: absolute;
        top: 0;
        right: 0;
        width: 250px;
        height: 250px;
        padding: 5px;
        border: 1px solid #ccc;
        background-color: white;
        z-index: 1000;
        box-sizing: border-box;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
    ">
        <img id="image-display" src="" alt="Selected image" style="width: 100%; height: 100%; object-fit: contain; display: none;" />
        <p id="placeholder-text">Click en un punto para ver la imagen</p>
    </div>
    """

    uris_json = json.dumps(image_data_uris)

    javascript_code = f"""
    <script>
        function displayImage(imageId) {{
            var imageElement = document.getElementById('image-display-{PLOTLY_DIV_ID}');
            var placeholderText = document.getElementById('placeholder-text-{PLOTLY_DIV_ID}');
            var imageDataURIs = {uris_json};

            if (imageElement && imageDataURIs[imageId]) {{
                imageElement.src = imageDataURIs[imageId];
                imageElement.style.display = 'block';
                placeholderText.style.display = 'none';
            }}
        }}

        function attachPlotlyListener() {{
            var plotDiv = document.querySelector('.js-plotly-plot');

            if (plotDiv) {{
                var imageContainer = document.getElementById('image-container-{PLOTLY_DIV_ID}');
                if(imageContainer) {{
                    plotDiv.style.position = 'relative';
                    plotDiv.appendChild(imageContainer);
                }}

                plotDiv.on('plotly_click', function(data) {{
                    if (data.points.length > 0) {{
                        var customdata = data.points[0].customdata;
                        displayImage(customdata);
                    }}
                }});
                return true;
            }}
            return false;
        }}

        document.getElementById('image-container').id = 'image-container-{PLOTLY_DIV_ID}';
        document.getElementById('image-display').id = 'image-display-{PLOTLY_DIV_ID}';
        document.getElementById('placeholder-text').id = 'placeholder-text-{PLOTLY_DIV_ID}';

        var intervalId = setInterval(function() {{
            if (attachPlotlyListener()) {{
                clearInterval(intervalId);
            }}
        }}, 500);
    </script>
    """

    display(HTML(image_display_html))
    display(Javascript(javascript_code))
