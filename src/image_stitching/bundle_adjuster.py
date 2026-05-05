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
    focal_length: float = 0.0


def _rodrigues(angle_axis: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(angle_axis)
    if theta < 1e-10:
        return np.eye(3, dtype=np.float64)

    r = angle_axis / theta
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    K = np.array([
        [0.0, -r[2], r[1]],
        [r[2], 0.0, -r[0]],
        [-r[1], r[0], 0.0],
    ], dtype=np.float64)

    return np.eye(3, dtype=np.float64) + sin_t * K + (1.0 - cos_t) * (K @ K)


def _rotation_to_angle_axis(R: np.ndarray) -> np.ndarray:
    trace = np.trace(R)
    cos_t = (trace - 1.0) / 2.0
    cos_t = np.clip(cos_t, -1.0, 1.0)
    theta = np.arccos(cos_t)

    if theta < 1e-10:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)

    rx = R[2, 1] - R[1, 2]
    ry = R[0, 2] - R[2, 0]
    rz = R[1, 0] - R[0, 1]
    r = np.array([rx, ry, rz], dtype=np.float64)
    norm = np.linalg.norm(r)
    if norm < 1e-10:
        if cos_t > 0:
            return np.array([0.0, 0.0, 0.0], dtype=np.float64)
        angle = np.pi
        if R[0, 0] > 0:
            r = np.array([1, 0, 0], dtype=np.float64)
        elif R[1, 1] > 0:
            r = np.array([0, 1, 0], dtype=np.float64)
        elif R[2, 2] > 0:
            r = np.array([0, 0, 1], dtype=np.float64)
        else:
            r = np.array([1, 1, 1], dtype=np.float64) / np.sqrt(3)
        return r * angle

    r = r / norm
    return r * theta


def _closest_rotation(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def _params_to_homography(rx: float, ry: float, rz: float, tx: float, ty: float, f: float, cx: float, cy: float) -> np.ndarray:
    K = np.array([
        [f, 0.0, cx],
        [0.0, f, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    K_inv = np.linalg.inv(K)

    angle_axis = np.array([rx, ry, rz], dtype=np.float64)
    R = _rodrigues(angle_axis)

    T = np.array([
        [1.0, 0.0, tx],
        [0.0, 1.0, ty],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    return T @ K @ R @ K_inv


def _decompose_homography(H: np.ndarray, cx: float, cy: float, f_hint: float = 500.0) -> tuple[float, float, float, float, float, float]:
    H = H / H[2, 2]

    K_guess = np.array([
        [f_hint, 0.0, cx],
        [0.0, f_hint, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    K_inv_guess = np.array([
        [1.0 / f_hint, 0.0, -cx / f_hint],
        [0.0, 1.0 / f_hint, -cy / f_hint],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    M = K_inv_guess @ H @ K_guess

    R_est = _closest_rotation(M)

    angle_axis = _rotation_to_angle_axis(R_est)

    residual = np.linalg.inv(R_est) @ M
    tx = residual[0, 2]
    ty = residual[1, 2]

    return float(angle_axis[0]), float(angle_axis[1]), float(angle_axis[2]), tx, ty, f_hint


class BundleAdjuster:

    def __init__(self, graph: PanoGraph, images: list[FeatureImage]) -> None:
        self._graph = graph
        self._images = images
        self._centers: dict[int, tuple[float, float]] = {}
        for img_id, img in enumerate(images):
            h, w = img.shape
            self._centers[img_id] = (w / 2.0, h / 2.0)

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

        added_ids, camera_params, added_order = self._accumulate_transforms()

        if len(added_ids) <= 1:
            homographies = {img_id: _params_to_homography(
                p.rx, p.ry, p.rz, p.tx, p.ty, 1000.0, *self._centers[img_id]
            ) for img_id, p in camera_params.items()}
            return BundleResult(
                homographies=homographies,
                n_iterations=0,
                final_cost=0.0,
                added_order=added_order,
            )

        global_focal, cam_params = self._optimize_camera_params(added_ids, camera_params)

        homographies: dict[int, np.ndarray] = {}
        for img_id in added_ids:
            p = cam_params[img_id]
            homographies[img_id] = _params_to_homography(p.rx, p.ry, p.rz, p.tx, p.ty, global_focal, *self._centers[img_id])

        cost = self._compute_total_cost(added_ids, homographies)

        result = BundleResult(
            homographies=homographies,
            n_iterations=1,
            final_cost=cost,
            added_order=added_order,
            focal_length=global_focal,
        )

        logger.debug(f"Bundle adjustment complete. Final cost: {cost:.4f}, focal: {global_focal:.1f}")
        return result

    def _accumulate_transforms(self) -> tuple[set[int], dict[int, "CameraParam"], list[int]]:
        base_id = self._graph.base_node
        added_ids: set[int] = {base_id}
        added_order: list[int] = [base_id]

        focal_estimates_per_image: list[float] = []
        for img_id in self._graph.graph.nodes:
            h, w = self._images[img_id].shape
            focal_estimates_per_image.append(max(w, h))

        default_focal = float(np.median(focal_estimates_per_image)) if focal_estimates_per_image else 500.0

        camera_params: dict[int, CameraParam] = {
            base_id: CameraParam(rx=0.0, ry=0.0, rz=0.0, tx=0.0, ty=0.0, f=default_focal)
        }

        while len(added_ids) < len(self._graph.graph.nodes):
            next_id, best_neighbor = self._find_next_image(added_ids)
            if next_id is None:
                logger.warning("No more connected images to add")
                break

            self._initialize_camera_params(next_id, best_neighbor, camera_params)
            added_ids.add(next_id)
            added_order.append(next_id)
            logger.debug(f"Added image {next_id} to bundle (total: {len(added_ids)})")

        return added_ids, camera_params, added_order

    def _find_next_image(self, added_ids: set[int]) -> tuple[int | None, int | None]:
        best_id = None
        best_neighbor = None
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
                    best_neighbor = neighbor

        return best_id, best_neighbor

    def _initialize_camera_params(
        self,
        new_id: int,
        neighbor_id: int,
        camera_params: dict[int, "CameraParam"],
    ) -> None:
        neighbor_params = camera_params[neighbor_id]
        ic: ImageConnection = self._graph.graph[new_id][neighbor_id]["data"]

        H_neighbor = _params_to_homography(
            neighbor_params.rx, neighbor_params.ry, neighbor_params.rz,
            neighbor_params.tx, neighbor_params.ty,
            neighbor_params.f, *self._centers[neighbor_id],
        )

        if ic.reference == new_id:
            H_pair = ic.homography.astype(np.float64)
            H_new = H_neighbor @ H_pair
        elif ic.reference == neighbor_id:
            H_pair = ic.homography.astype(np.float64)
            H_new = H_neighbor @ np.linalg.inv(H_pair)
        else:
            camera_params[new_id] = CameraParam(
                rx=neighbor_params.rx, ry=neighbor_params.ry, rz=neighbor_params.rz,
                tx=neighbor_params.tx, ty=neighbor_params.ty, f=neighbor_params.f,
            )
            return

        cx_new, cy_new = self._centers[new_id]
        rx, ry, rz, tx, ty, f_est = _decompose_homography(H_new, cx_new, cy_new, f_hint=neighbor_params.f)

        camera_params[new_id] = CameraParam(
            rx=rx, ry=ry, rz=rz,
            tx=tx, ty=ty,
            f=neighbor_params.f,
        )

    def _optimize_camera_params(
        self,
        added_ids: set[int],
        camera_params: dict[int, "CameraParam"],
    ) -> tuple[float, dict[int, "CameraParam"]]:
        base_id = self._graph.base_node
        sorted_ids = sorted(added_ids - {base_id})
        id_to_idx = {img_id: i for i, img_id in enumerate(sorted_ids)}
        n_images = len(sorted_ids)

        default_focal = camera_params[base_id].f
        focal_estimates = [camera_params[img_id].f for img_id in sorted_ids]
        if focal_estimates:
            default_focal = float(np.median(focal_estimates))

        x0 = np.empty(n_images * 5 + 1)
        for i, img_id in enumerate(sorted_ids):
            p = camera_params[img_id]
            x0[i * 5] = p.rx
            x0[i * 5 + 1] = p.ry
            x0[i * 5 + 2] = p.rz
            x0[i * 5 + 3] = p.tx
            x0[i * 5 + 4] = p.ty
        x0[n_images * 5] = default_focal

        edge_matches = self._cache_edge_matches(added_ids)

        f_min = default_focal * 0.3
        f_max = default_focal * 3.0
        lower_bounds = np.full(n_images * 5 + 1, -np.inf)
        upper_bounds = np.full(n_images * 5 + 1, np.inf)
        lower_bounds[n_images * 5] = f_min
        upper_bounds[n_images * 5] = f_max

        def residuals(x):
            res_list = []
            f = x[n_images * 5]

            for u, v, attr in self._graph.graph.edges(data=True):
                if u not in added_ids or v not in added_ids:
                    continue
                ic: ImageConnection = attr["data"]
                if ic.n_inliers == 0:
                    continue

                cached = edge_matches.get((u, v))
                if cached is None:
                    continue

                ref_kps, tgt_kps = cached

                H_u = self._get_homography(ic.reference, sorted_ids, id_to_idx, x)
                H_v = self._get_homography(ic.target, sorted_ids, id_to_idx, x)

                proj_ref = (H_u @ ref_kps.T).T
                proj_tgt = (H_v @ tgt_kps.T).T
                proj_ref = proj_ref[:, :2] / proj_ref[:, 2:3]
                proj_tgt = proj_tgt[:, :2] / proj_tgt[:, 2:3]

                diff = proj_ref - proj_tgt
                for j in range(len(diff)):
                    res_list.append(diff[j, 0])
                    res_list.append(diff[j, 1])

            for i, img_id in enumerate(sorted_ids):
                idx = i * 5
                rot_weight = 0.1
                tx_weight = 0.05
                res_list.append(rot_weight * (x[idx] - x0[idx]))
                res_list.append(rot_weight * (x[idx + 1] - x0[idx + 1]))
                res_list.append(rot_weight * (x[idx + 2] - x0[idx + 2]))
                res_list.append(tx_weight * (x[idx + 3] - x0[idx + 3]))
                res_list.append(tx_weight * (x[idx + 4] - x0[idx + 4]))

            focal_prior_weight = 1.0
            res_list.append(focal_prior_weight * (f - default_focal))

            return np.array(res_list, dtype=np.float64).reshape(-1) if len(res_list) > 1 else np.array(res_list, dtype=np.float64)

        result = least_squares(residuals, x0, method="trf", bounds=(lower_bounds, upper_bounds), max_nfev=500)

        f_opt = result.x[n_images * 5]
        cam_params: dict[int, CameraParam] = {}
        for img_id in sorted_ids:
            idx = id_to_idx[img_id]
            cam_params[img_id] = CameraParam(
                rx=result.x[idx * 5],
                ry=result.x[idx * 5 + 1],
                rz=result.x[idx * 5 + 2],
                tx=result.x[idx * 5 + 3],
                ty=result.x[idx * 5 + 4],
                f=f_opt,
            )
        cam_params[base_id] = CameraParam(rx=0.0, ry=0.0, rz=0.0, tx=0.0, ty=0.0, f=f_opt)

        return f_opt, cam_params

    def _get_homography(
        self,
        img_id: int,
        sorted_ids: list[int],
        id_to_idx: dict[int, int],
        x: np.ndarray,
    ) -> np.ndarray:
        base_id = self._graph.base_node
        if img_id == base_id:
            cx, cy = self._centers[base_id]
            f = x[len(sorted_ids) * 5]
            return _params_to_homography(0.0, 0.0, 0.0, 0.0, 0.0, f, cx, cy)

        idx = id_to_idx[img_id]
        rx = x[idx * 5]
        ry = x[idx * 5 + 1]
        rz = x[idx * 5 + 2]
        tx = x[idx * 5 + 3]
        ty = x[idx * 5 + 4]
        f = x[len(sorted_ids) * 5]
        cx, cy = self._centers[img_id]

        return _params_to_homography(rx, ry, rz, tx, ty, f, cx, cy)

    def _cache_edge_matches(self, added_ids: set[int]) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
        from image_stitching import pipeline
        cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
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
            if len(pairs) == 0:
                continue

            inlier_indices = np.where(ic.inlier_mask)[0]
            valid_indices = inlier_indices[inlier_indices < len(pairs)]
            if len(valid_indices) == 0:
                continue

            ref_kps = np.empty((len(valid_indices), 3), dtype=np.float64)
            tgt_kps = np.empty((len(valid_indices), 3), dtype=np.float64)
            for j, idx in enumerate(valid_indices):
                kp_id1, kp_id2 = pairs[idx]
                ref_kps[j, 0] = img1.kps[kp_id1].pt[0]
                ref_kps[j, 1] = img1.kps[kp_id1].pt[1]
                ref_kps[j, 2] = 1.0
                tgt_kps[j, 0] = img2.kps[kp_id2].pt[0]
                tgt_kps[j, 1] = img2.kps[kp_id2].pt[1]
                tgt_kps[j, 2] = 1.0

            cache[(u, v)] = (ref_kps, tgt_kps)
            cache[(v, u)] = (tgt_kps, ref_kps)
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

            H1 = homographies[ic.reference]
            H2 = homographies[ic.target]

            inlier_indices = np.where(ic.inlier_mask)[0]
            for idx in inlier_indices:
                if idx >= len(good_matches):
                    continue
                kp_id1, kp_id2 = good_matches[idx]
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


@dataclass
class CameraParam:
    rx: float
    ry: float
    rz: float
    tx: float
    ty: float
    f: float
