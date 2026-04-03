import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from image_stitching import models


def plot_connectivity_matrix(match_ratio: np.ndarray):
    n_images = match_ratio.shape[0]
    fig, ax = plt.subplots()
    cax = ax.imshow(match_ratio)

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
            display_value = f"{int(match_ratio[i, j]*100)}%"
            ax.text(j, i, display_value, ha="center", va="center")

    plt.xlabel("Reference Image")
    plt.ylabel("Target Image")
    plt.title("Match Percentage")

    plt.tight_layout()
    plt.show()


def plot_panorama_graph(graph: models.PanoramaGraph):   
    pos = nx.spring_layout(graph.graph)
    # Draw graph
    nx.draw(graph.graph, pos, with_labels=True)
    plt.show()