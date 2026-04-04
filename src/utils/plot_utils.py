from enum import Enum
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from image_stitching import models


class ConnectivityMatrixMode(Enum):
    RATIO = 0
    COUNT = 1


def plot_connectivity_matrix(matrix: np.ndarray, mode=ConnectivityMatrixMode.RATIO):
    n_images = matrix.shape[0]
    fig, ax = plt.subplots(figsize=(10, 8))  # width, height in inches
    cax = ax.imshow(matrix)

    # Add colorbar
    plt.colorbar(cax)

    # Axis labels

    labels = [f"{i}" for i in range(n_images)]
    ax.set_xticks(range(n_images))
    ax.set_yticks(range(n_images))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Rotate x labels
    plt.xticks(rotation=45)

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