import numpy as np
import networkx as nx
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from image_stitching import models
from core.enums import ConnectivityMatrixMode


def plot_connectivity_matrix(matrix: np.ndarray, image_names: list[str]):
    n_images = matrix.shape[0]
    fig, ax = plt.subplots(figsize=(10, 8))  # width, height in inches
    cax = ax.imshow(matrix)

    #set mode
    if np.issubdtype(matrix.dtype, np.integer):
        mode = ConnectivityMatrixMode.COUNT
    else:
        mode = ConnectivityMatrixMode.RATIO
    
    # Add colorbar
    plt.colorbar(cax)

    # Axis labels

    labels = [f"{image_names[i]}" for i in range(n_images)]
    ax.set_xticks(range(n_images))
    ax.set_yticks(range(n_images))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Rotate x labels
    plt.xticks(rotation=90)

    # Annotate each cell with value
    for i in range(n_images):
        for j in range(n_images):
            display_value = ""
            if mode == ConnectivityMatrixMode.RATIO:
                display_value = f"{matrix[i, j]:.2f}"
            else:
                display_value = f"{int(matrix[i, j])}"
            ax.text(j, i, display_value, ha="center", va="center")

    plt.xlabel("Target Image")
    plt.ylabel("Reference Image")

    if mode == ConnectivityMatrixMode.RATIO:
        title = "Match Percentage"
    else:
        title = "Match Count"

    plt.title(title)

    plt.tight_layout()
    plt.show()


def plot_panorama_graph(graph: models.PanoramaGraph):   
    pos = nx.spring_layout(graph.graph)
    # Draw graph
    nx.draw(graph.graph, pos, with_labels=True)
    plt.show()


def plot_2_image_stitch(path_ref: str, path_base: str, H: np.ndarray):
    img_ref = cv2.imread(path_ref, cv2.IMREAD_COLOR)
    img_base = cv2.imread(path_base, cv2.IMREAD_COLOR)

    if img_ref is None or img_base is None:
        raise ValueError("Failed to load one or both images")

    h_ref, w_ref = img_ref.shape[:2]
    h_base, w_base = img_base.shape[:2]

    ref_corners = np.array([[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]],
                            dtype=np.float32).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(ref_corners, H).reshape(-1, 2)
    base_corners = np.array([[0, 0], [w_base, 0], [w_base, h_base], [0, h_base]],
                             dtype=np.float32).reshape(-1, 2)

    all_corners = np.vstack([warped_corners, base_corners])
    x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)

    T = np.array([[1, 0, -x_min],
                  [0, 1, -y_min],
                  [0, 0, 1]], dtype=np.float64)

    canvas_h = y_max - y_min
    canvas_w = x_max - x_min
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    base_offset_x = int(-x_min)
    base_offset_y = int(-y_min)
    canvas[base_offset_y:base_offset_y + h_base, base_offset_x:base_offset_x + w_base] = img_base

    H_total = T @ H
    warped_ref = cv2.warpPerspective(img_ref, H_total.astype(np.float32),
                                     (canvas_w, canvas_h))
    mask = warped_ref > 0
    canvas[mask] = warped_ref[mask]

    fig, ax = plt.subplots(figsize=(12, 8))
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    ax.imshow(canvas_rgb)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_matches_per_feature(matching_df: pd.DataFrame):
    match_counts = matching_df['Matched_ImgIds'].apply(len)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(match_counts, bins=20, edgecolor='black')
    ax.set_xlabel('Number of Matches per Feature')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Matches per Feature')

    plt.tight_layout()
    plt.show()