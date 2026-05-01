import numpy as np
import logging
from dataclasses import dataclass, field
from scipy.optimize import least_squares

import image_stitching.pipeline as pipeline
from image_stitching.panograph import ImageConnection, PanoGraph
from classes.feature_image import FeatureImage


logger = logging.getLogger("image_stitching")


@dataclass
class BundleResult:
    homographies: dict[int, np.ndarray] = field(default_factory=dict)
    n_iterations: int = 0
    final_cost: float = 0.0
    added_order: list[int] = field(default_factory=list)


class BundleAdjuster:

    def __init__(self, graph: PanoGraph, images: list[FeatureImage]) -> None:
        self._graph = graph
        self._images = images

    def optimize(self) -> BundleResult:
        base_id = self._graph.base_node
        added_ids: set[int] = {base_id}
        homographies: dict[int, np.ndarray] = {base_id: np.eye(3, dtype=np.float64)}
        added_order: list[int] = [base_id]

        logger.debug(f"Bundle adjustment starting from base image: {base_id}")

        while len(added_ids) < len(self._graph.graph.nodes):
            next_id = self._find_next_image(added_ids)
            if next_id is None:
                logger.warning("No more connected images to add, stopping bundle adjustment early")
                break

            H_init = self._initialize_homography(next_id, added_ids, homographies)
            added_ids.add(next_id)
            homographies[next_id] = H_init
            added_order.append(next_id)

            logger.debug(f"Added image {next_id} to bundle (total: {len(added_ids)})")

            self._optimize_all(added_ids, homographies)

        result = BundleResult(
            homographies=homographies,
            n_iterations=len(added_order) - 1,
            final_cost=self._compute_total_cost(added_ids, homographies),
            added_order=added_order,
        )

        logger.debug(f"Bundle adjustment complete. Iterations: {result.n_iterations}, Final cost: {result.final_cost:.4f}")
        return result

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
            if ic.n_inliers < best_inliers:
                continue
            best_inliers = ic.n_inliers
            H_rel = ic.homography.astype(np.float64)
            H_neighbor = homographies[neighbor]
            if ic.reference == new_id:
                best_H = H_neighbor @ np.linalg.inv(H_rel)
            else:
                best_H = H_neighbor @ H_rel

        return best_H

    def _optimize_all(
        self,
        added_ids: set[int],
        homographies: dict[int, np.ndarray],
    ) -> None:
        base_id = self._graph.base_node
        sorted_ids = sorted(added_ids - {base_id})
        id_to_param_idx = {img_id: i for i, img_id in enumerate(sorted_ids)}
        n_images = len(sorted_ids)
        n_params = n_images * 8

        if n_params == 0:
            return

        x0 = np.empty(n_params)
        for img_id in sorted_ids:
            H = homographies[img_id]
            idx = id_to_param_idx[img_id]
            x0[idx * 8 : (idx + 1) * 8] = [
                H[0, 0], H[0, 1], H[0, 2],
                H[1, 0], H[1, 1], H[1, 2],
                H[2, 0], H[2, 1],
            ]

        def residuals(x):
            current_homographies = {base_id: np.eye(3, dtype=np.float64)}
            for img_id in sorted_ids:
                idx = id_to_param_idx[img_id]
                p = x[idx * 8 : (idx + 1) * 8]
                H = np.array([
                    [p[0], p[1], p[2]],
                    [p[3], p[4], p[5]],
                    [p[6], p[7], 1.0],
                ], dtype=np.float64)
                current_homographies[img_id] = H

            res_list = []
            for u, v, attr in self._graph.graph.edges(data=True):
                if u not in added_ids or v not in added_ids:
                    continue
                ic: ImageConnection = attr["data"]
                if len(ic.inlier_mask) == 0:
                    continue

                img1 = self._images[ic.reference]
                img2 = self._images[ic.target]

                if img1.kps is None or img2.kps is None:
                    continue

                H1 = current_homographies[ic.reference]
                H2 = current_homographies[ic.target]

                good_matches = pipeline.get_good_matches(img1, img2)
                if len(good_matches) == 0:
                    continue

                n_matches = len(good_matches)
                mask_len = min(len(ic.inlier_mask), n_matches)

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

                    res_list.append(proj1[0] - proj2[0])
                    res_list.append(proj1[1] - proj2[1])

            reg_weight = 0.05
            proj_weight = 5.0
            for img_id in sorted_ids:
                idx = id_to_param_idx[img_id]
                p = x[idx * 8 : (idx + 1) * 8]
                p0 = x0[idx * 8 : (idx + 1) * 8]

                res_list.extend(reg_weight * (p[:6] - p0[:6]))
                res_list.extend(proj_weight * p[6:8])

            return np.array(res_list) if res_list else np.zeros(1)

        lower_bounds = np.full(n_params, -np.inf)
        upper_bounds = np.full(n_params, np.inf)
        for img_id in sorted_ids:
            idx = id_to_param_idx[img_id]
            base = idx * 8
            lower_bounds[base + 0] = 0.3
            upper_bounds[base + 0] = 3.0
            lower_bounds[base + 1] = -1.0
            upper_bounds[base + 1] = 1.0
            lower_bounds[base + 2] = -3000
            upper_bounds[base + 2] = 3000
            lower_bounds[base + 3] = -1.0
            upper_bounds[base + 3] = 1.0
            lower_bounds[base + 4] = 0.3
            upper_bounds[base + 4] = 3.0
            lower_bounds[base + 5] = -3000
            upper_bounds[base + 5] = 3000
            lower_bounds[base + 6] = -0.002
            upper_bounds[base + 6] = 0.002
            lower_bounds[base + 7] = -0.002
            upper_bounds[base + 7] = 0.002

        result = least_squares(residuals, x0, bounds=(lower_bounds, upper_bounds), method='trf', max_nfev=100)

        for img_id in sorted_ids:
            idx = id_to_param_idx[img_id]
            p = result.x[idx * 8 : (idx + 1) * 8]
            homographies[img_id] = np.array([
                [p[0], p[1], p[2]],
                [p[3], p[4], p[5]],
                [p[6], p[7], 1.0],
            ], dtype=np.float64)

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
            if len(ic.inlier_mask) == 0:
                continue

            img1 = self._images[ic.reference]
            img2 = self._images[ic.target]
            if img1.kps is None or img2.kps is None:
                continue

            good_matches = pipeline.get_good_matches(img1, img2)
            n_matches = len(good_matches)
            mask_len = min(len(ic.inlier_mask), n_matches)

            H1 = homographies[ic.reference]
            H2 = homographies[ic.target]

            for i in range(mask_len):
                if not ic.inlier_mask[i]:
                    continue
                kp_id1, kp_id2 = good_matches[i]
                pt1 = np.array([img1.kps[kp_id1].pt[0], img1.kps[kp_id1].pt[1], 1.0])
                pt2 = np.array([img2.kps[kp_id2].pt[0], img2.kps[kp_id2].pt[1], 1.0])

                proj1 = (H1 @ pt1)
                proj2 = (H2 @ pt2)
                proj1 = proj1[:2] / proj1[2]
                proj2 = proj2[:2] / proj2[2]

                total_sq_error += np.sum((proj1 - proj2) ** 2)
                n_correspondences += 1

        if n_correspondences == 0:
            return 0.0
        return np.sqrt(total_sq_error / n_correspondences)
