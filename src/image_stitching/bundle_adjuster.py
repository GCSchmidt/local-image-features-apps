import numpy as np
import logging
from dataclasses import dataclass, field
from scipy.optimize import least_squares

from image_stitching.panograph import ImageConnection, PanoGraph
from classes.feature_image import FeatureImage


logger = logging.getLogger("image_stitching")


@dataclass
class BundleResult:
    homographies: dict[int, np.ndarray] = field(default_factory=dict)
    n_iterations: int = 0
    final_cost: float = 0.0
    added_order: list[int] = field(default_factory=list)


def _affine_to_matrix(a: float, b: float, c: float, d: float, tx: float, ty: float) -> np.ndarray:
    return np.array([
        [a, b, tx],
        [c, d, ty],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _matrix_to_affine(H: np.ndarray) -> tuple[float, float, float, float, float, float]:
    return (float(H[0, 0]), float(H[0, 1]),
            float(H[1, 0]), float(H[1, 1]),
            float(H[0, 2]), float(H[1, 2]))


class BundleAdjuster:

    def __init__(self, graph: PanoGraph, images: list[FeatureImage]) -> None:
        self._graph = graph
        self._images = images

    def optimize(self) -> BundleResult:
        base_id = self._graph.base_node
        n_nodes = len(self._graph.graph.nodes)

        if n_nodes <= 1:
            return BundleResult(
                homographies={base_id: np.eye(3, dtype=np.float64)},
                n_iterations=0,
                final_cost=0.0,
                added_order=[base_id],
            )

        logger.debug(f"Bundle adjustment starting from base image: {base_id}")

        added_ids, homographies, added_order = self._accumulate_transforms()

        if len(added_ids) <= 1:
            return BundleResult(
                homographies=homographies,
                n_iterations=0,
                final_cost=0.0,
                added_order=added_order,
            )

        aff_params = self._optimize_affine(added_ids, homographies)

        for img_id, params in aff_params.items():
            homographies[img_id] = _affine_to_matrix(*params)

        cost = self._compute_total_cost(added_ids, homographies)

        result = BundleResult(
            homographies=homographies,
            n_iterations=1,
            final_cost=cost,
            added_order=added_order,
        )

        logger.debug(f"Bundle adjustment complete. Final cost: {cost:.4f}")
        return result

    def _accumulate_transforms(self) -> tuple[set[int], dict[int, np.ndarray], list[int]]:
        base_id = self._graph.base_node
        added_ids: set[int] = {base_id}
        homographies: dict[int, np.ndarray] = {base_id: np.eye(3, dtype=np.float64)}
        added_order: list[int] = [base_id]

        while len(added_ids) < len(self._graph.graph.nodes):
            next_id = self._find_next_image(added_ids)
            if next_id is None:
                logger.warning("No more connected images to add")
                break

            H_init = self._initialize_homography(next_id, added_ids, homographies)
            added_ids.add(next_id)
            homographies[next_id] = H_init
            added_order.append(next_id)
            logger.debug(f"Added image {next_id} to bundle (total: {len(added_ids)})")

        return added_ids, homographies, added_order

    def _find_next_image(self, added_ids: set[int]) -> int | None:
        best_id = None
        best_inliers = 0

        for node in self._graph.graph.nodes:
            if node in added_ids:
                continue
            for neighbor in self._graph.graph.neighbors(node):
                if neighbor not in added_ids:
                    continue
                ic: ImageConnection = self._graph.graph[node][neighbor]["data"]
                if ic.n_inliers > best_inliers:
                    best_inliers = ic.n_inliers
                    best_id = node

        return best_id

    def _initialize_homography(
        self,
        new_id: int,
        added_ids: set[int],
        homographies: dict[int, np.ndarray],
    ) -> np.ndarray:
        best_inliers = 0
        best_H = np.eye(3, dtype=np.float64)

        for neighbor in self._graph.graph.neighbors(new_id):
            if neighbor not in added_ids:
                continue
            ic: ImageConnection = self._graph.graph[new_id][neighbor]["data"]
            if ic.n_inliers > best_inliers:
                best_inliers = ic.n_inliers
                H_rel = ic.homography.astype(np.float64)
                H_neighbor = homographies[neighbor]
                if ic.reference == new_id:
                    best_H = H_neighbor @ np.linalg.inv(H_rel)
                else:
                    best_H = H_neighbor @ H_rel

        return best_H

    def _optimize_affine(
        self,
        added_ids: set[int],
        homographies: dict[int, np.ndarray],
    ) -> dict[int, tuple[float, float, float, float, float, float]]:
        base_id = self._graph.base_node
        sorted_ids = sorted(added_ids - {base_id})
        id_to_idx = {img_id: i for i, img_id in enumerate(sorted_ids)}
        n_images = len(sorted_ids)
        n_params = n_images * 6

        x0 = np.empty(n_params)
        for img_id in sorted_ids:
            a, b, c, d, tx, ty = _matrix_to_affine(homographies[img_id])
            idx = id_to_idx[img_id]
            x0[idx * 6 : (idx + 1) * 6] = [a, b, c, d, tx, ty]

        edge_matches = self._cache_edge_matches(added_ids)

        def residuals(x):
            res_list = []
            for u, v, attr in self._graph.graph.edges(data=True):
                if u not in added_ids or v not in added_ids:
                    continue
                ic: ImageConnection = attr["data"]
                if ic.n_inliers == 0:
                    continue

                matched_pairs = edge_matches.get((u, v))
                if matched_pairs is None or len(matched_pairs) == 0:
                    continue

                H1 = self._get_affine_matrix(ic.reference, base_id, sorted_ids, id_to_idx, x)
                H2 = self._get_affine_matrix(ic.target, base_id, sorted_ids, id_to_idx, x)

                mask_len = min(len(ic.inlier_mask), len(matched_pairs))
                for i in range(mask_len):
                    if not ic.inlier_mask[i]:
                        continue
                    kp_id1, kp_id2 = matched_pairs[i]
                    pt1 = np.array([self._images[ic.reference].kps[kp_id1].pt[0],
                                    self._images[ic.reference].kps[kp_id1].pt[1], 1.0])
                    pt2 = np.array([self._images[ic.target].kps[kp_id2].pt[0],
                                    self._images[ic.target].kps[kp_id2].pt[1], 1.0])

                    proj1 = H1 @ pt1
                    proj2 = H2 @ pt2
                    proj1 = proj1[:2] / proj1[2]
                    proj2 = proj2[:2] / proj2[2]

                    res_list.append(proj1[0] - proj2[0])
                    res_list.append(proj1[1] - proj2[1])

            reg_weight = 0.1
            for img_id in sorted_ids:
                idx = id_to_idx[img_id]
                a, b, c, d = x[idx * 6 : idx * 6 + 4]
                # Regularize: encourage similarity-like behavior (a ≈ d, b ≈ -c)
                res_list.append(reg_weight * (a - d))
                res_list.append(reg_weight * (b + c))

            return np.array(res_list, dtype=np.float64)

        result = least_squares(residuals, x0, method="lm", max_nfev=200)

        aff_params: dict[int, tuple[float, float, float, float, float, float]] = {}
        for img_id in sorted_ids:
            idx = id_to_idx[img_id]
            p = result.x[idx * 6 : (idx + 1) * 6]
            aff_params[img_id] = (p[0], p[1], p[2], p[3], p[4], p[5])
        aff_params[base_id] = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

        return aff_params

    def _get_affine_matrix(
        self,
        img_id: int,
        base_id: int,
        sorted_ids: list[int],
        id_to_idx: dict[int, int],
        x: np.ndarray,
    ) -> np.ndarray:
        if img_id == base_id:
            return np.eye(3, dtype=np.float64)
        idx = id_to_idx[img_id]
        a, b, c, d, tx, ty = x[idx * 6 : (idx + 1) * 6]
        return _affine_to_matrix(a, b, c, d, tx, ty)

    def _cache_edge_matches(self, added_ids: set[int]) -> dict[tuple[int, int], list[tuple[int, int]]]:
        from image_stitching import pipeline
        cache: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for u, v, attr in self._graph.graph.edges(data=True):
            if u not in added_ids or v not in added_ids:
                continue
            ic: ImageConnection = attr["data"]
            if ic.n_inliers == 0:
                continue
            img1 = self._images[ic.reference]
            img2 = self._images[ic.target]
            if img1.kps is None or img2.kps is None:
                continue
            pairs = pipeline.get_good_matches(img1, img2)
            cache[(u, v)] = pairs
            if (u, v) != (ic.reference, ic.target):
                cache[(v, u)] = [(b, a) for a, b in pairs]
        return cache

    def _compute_total_cost(
        self,
        added_ids: set[int],
        homographies: dict[int, np.ndarray],
    ) -> float:
        total_sq_error = 0.0
        n_correspondences = 0

        for u, v, attr in self._graph.graph.edges(data=True):
            if u not in added_ids or v not in added_ids:
                continue
            ic: ImageConnection = attr["data"]
            if ic.n_inliers == 0:
                continue

            img1 = self._images[ic.reference]
            img2 = self._images[ic.target]
            if img1.kps is None or img2.kps is None:
                continue

            from image_stitching import pipeline
            good_matches = pipeline.get_good_matches(img1, img2)
            mask_len = min(len(ic.inlier_mask), len(good_matches))

            H1 = homographies[ic.reference]
            H2 = homographies[ic.target]

            for i in range(mask_len):
                if not ic.inlier_mask[i]:
                    continue
                kp_id1, kp_id2 = good_matches[i]
                pt1 = np.array([img1.kps[kp_id1].pt[0], img1.kps[kp_id1].pt[1], 1.0])
                pt2 = np.array([img2.kps[kp_id2].pt[0], img2.kps[kp_id2].pt[1], 1.0])

                proj1 = H1 @ pt1
                proj2 = H2 @ pt2
                proj1 = proj1[:2] / proj1[2]
                proj2 = proj2[:2] / proj2[2]

                total_sq_error += np.sum((proj1 - proj2) ** 2)
                n_correspondences += 1

        if n_correspondences == 0:
            return 0.0
        return np.sqrt(total_sq_error / n_correspondences)
