import networkx as nx
import numpy as np
import logging
from dataclasses import dataclass, field

from core.enums import ValidConnection, HomographyReference


logger = logging.getLogger("image_stitching")


@dataclass
class ImageConnection:
    n_inliers: int = 0
    n_overlap: int = 0
    homography: np.ndarray = field(default_factory=lambda: np.identity(3, dtype=np.float32))
    reference: int = 0
    target: int = 0
    inlier_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))

    @property
    def weight(self) -> float:
        if self.n_overlap == 0:
            return 1.0
        return 1.0 - (self.n_inliers / self.n_overlap)

class PanoGraph():

    def __init__(self) -> None:
        self._graph = nx.Graph()
        self._base_node = 0

    @property
    def graph(self) -> nx.Graph:
        return self._graph
   
    @property
    def base_node(self) -> int:
        return self._base_node
    
    def add_connection(self, ic: ImageConnection) -> None:
        img_id1 = ic.reference
        img_id2 = ic.target

        if self._graph.has_edge(img_id1, img_id2):
            existing_weight = self._graph[img_id1][img_id2].get("weight", float("inf"))
            if ic.weight >= existing_weight:
                return
        self._graph.add_edge(img_id1, img_id2, weight=ic.weight, data=ic)

    def set_base_node(self):
        closeness = nx.closeness_centrality(self._graph)
        self._base_node = max(closeness, key=closeness.get)

    def post_process(self):
        self._get_minimum_spanning_tree()
        self._keep_largest_connected_group()

    def _get_minimum_spanning_tree(self):
        self._graph = nx.minimum_spanning_tree(self._graph)
        self.set_base_node()

    def _keep_largest_connected_group(self):
        largest_nodes = max(nx.connected_components(self.graph), key=len)
        nodes_to_remove = set(self.graph.nodes) - largest_nodes
        self.graph.remove_nodes_from(nodes_to_remove)
        self.set_base_node()