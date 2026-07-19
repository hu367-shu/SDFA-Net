"""
RANSAC-Based Local Road Surface Plane Fitting.

Used for:
    - Generating ground-truth Δz (manhole cover elevation relative to road)
    - Inference-time disease severity classification

The algorithm:
    1. Given a manhole cover center, extract an annular region of road points
       (r_inner to r_outer) around it.
    2. Use RANSAC to robustly fit a plane z = a*x + b*y + c to the road points.
    3. Compute the average elevation of manhole cover points (within cover_radius).
    4. Δz = z̄_cover - z_plane(center).
    5. Classify: Δz < -15mm → sunken, Δz > +15mm → raised, else good.

Paper threshold: 15 mm = 0.015 m.
"""

import numpy as np
from typing import Tuple, Optional, List


def fit_plane_ransac(
    points: np.ndarray,
    threshold: float = 0.02,
    max_iterations: int = 500,
    min_inliers: int = 20,
) -> Optional[Tuple[float, float, float]]:
    """
    Fit a plane z = a*x + b*y + c using RANSAC.

    Args:
        points: [N, 3] xyz point cloud (road surface points).
        threshold: Inlier distance threshold in meters.
        max_iterations: Max RANSAC iterations.
        min_inliers: Minimum number of inliers required.

    Returns:
        (a, b, c): Plane coefficients z = a*x + b*y + c, or None if fitting fails.
    """
    if points.shape[0] < 3:
        return None

    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    best_plane = None
    best_inlier_count = 0

    for _ in range(max_iterations):
        # Sample 3 random points
        idx = np.random.choice(points.shape[0], 3, replace=False)
        sample = points[idx]

        # Fit plane through 3 points
        p1, p2, p3 = sample[0], sample[1], sample[2]
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)

        if np.linalg.norm(normal) < 1e-10:
            continue

        a, b, c_normal = normal
        # z = (-a/c)*x + (-b/c)*y + d
        if abs(c_normal) < 1e-10:
            continue

        a_plane = -a / c_normal
        b_plane = -b / c_normal
        c_plane = - (a_plane * p1[0] + b_plane * p1[1] - p1[2])
        # Actually: z = a_plane*x + b_plane*y + c_plane
        # Verify: p1[2] should equal a_plane*p1[0] + b_plane*p1[1] + c_plane
        c_plane = p1[2] - a_plane * p1[0] - b_plane * p1[1]

        # Compute residuals
        z_pred = a_plane * X + b_plane * Y + c_plane
        residuals = np.abs(Z - z_pred)
        inlier_count = np.sum(residuals < threshold)

        if inlier_count > best_inlier_count and inlier_count >= min_inliers:
            best_inlier_count = inlier_count
            best_plane = (a_plane, b_plane, c_plane)

    if best_plane is None:
        return None

    # Refit using all inliers
    a, b, c = best_plane
    z_pred = a * X + b * Y + c
    residuals = np.abs(Z - z_pred)
    inlier_mask = residuals < threshold

    if inlier_mask.sum() < min_inliers:
        return best_plane

    # Least squares refit on inliers
    X_in = X[inlier_mask]
    Y_in = Y[inlier_mask]
    Z_in = Z[inlier_mask]

    A = np.column_stack([X_in, Y_in, np.ones_like(X_in)])
    try:
        params, _, _, _ = np.linalg.lstsq(A, Z_in, rcond=None)
        a_refit, b_refit, c_refit = params[0], params[1], params[2]
        return (float(a_refit), float(b_refit), float(c_refit))
    except np.linalg.LinAlgError:
        return best_plane


def compute_delta_z_ransac(
    points: np.ndarray,
    center_xy: Tuple[float, float],
    r_inner: float = 0.45,
    r_outer: float = 1.0,
    cover_radius: float = 0.35,
    ransac_threshold: float = 0.02,
) -> Optional[float]:
    """
    Compute the relative elevation offset Δz of a manhole cover
    relative to the local road surface plane.

    Args:
        points: [N, 3] aggregated LiDAR point cloud in LiDAR frame.
        center_xy: (x, y) center of the manhole cover in meters.
        r_inner: Inner radius of the road annulus (m). Points inside this
                 are considered part of the cover and excluded.
        r_outer: Outer radius of the road annulus (m).
        cover_radius: Radius of the manhole cover for elevation averaging (m).
        ransac_threshold: RANSAC inlier distance threshold (m).

    Returns:
        delta_z: Relative elevation in meters (positive = raised,
                 negative = sunken), or None if insufficient data.
    """
    x_c, y_c = center_xy

    # Compute distances from center
    dist = np.sqrt((points[:, 0] - x_c) ** 2 + (points[:, 1] - y_c) ** 2)

    # ── Road points: annular region ────────────────────────
    road_mask = (dist > r_inner) & (dist < r_outer)
    road_points = points[road_mask]

    if road_points.shape[0] < 10:
        return None

    # ── Manhole cover points ───────────────────────────────
    cover_mask = dist < cover_radius
    cover_points = points[cover_mask]

    if cover_points.shape[0] < 3:
        return None

    # ── Fit road plane ─────────────────────────────────────
    plane = fit_plane_ransac(road_points, threshold=ransac_threshold)

    if plane is None:
        # Fallback: use simple mean of road points
        z_plane = float(np.median(road_points[:, 2]))
    else:
        a, b, c = plane
        z_plane = a * x_c + b * y_c + c

    # ── Average cover elevation ────────────────────────────
    z_cover = float(np.mean(cover_points[:, 2]))

    delta_z = z_cover - z_plane

    return delta_z


def classify_condition(
    delta_z: Optional[float],
    cls_scores: Optional[dict] = None,
    threshold: float = 0.015,
) -> str:
    """
    Classify manhole cover condition from elevation offset and/or class scores.

    Classification logic (per paper):
        - missing: class score for 'missing' > 0.5 (requires visual features)
        - broken: class score for 'broken' > 0.5
        - sunken: Δz < -15 mm
        - raised: Δz > +15 mm
        - good: otherwise

    Args:
        delta_z: Relative elevation offset in meters (from RANSAC).
        cls_scores: Dict of class name → confidence score.
        threshold: Δz threshold in meters (default 0.015 = 15 mm).

    Returns:
        condition: One of 'good', 'sunken', 'raised', 'broken', 'missing'.
    """
    if cls_scores is not None:
        if cls_scores.get("missing", 0.0) > 0.5:
            return "missing"
        if cls_scores.get("broken", 0.0) > 0.5:
            return "broken"

    if delta_z is None:
        return "good"

    if delta_z < -threshold:
        return "sunken"
    elif delta_z > threshold:
        return "raised"
    else:
        return "good"


# ── Batch RANSAC computation ────────────────────────────────────

def compute_delta_z_batch(
    points_list: List[np.ndarray],
    centers: List[Tuple[float, float]],
    **kwargs,
) -> List[Optional[float]]:
    """
    Compute Δz for multiple manhole covers.

    Args:
        points_list: List of point cloud arrays [N_i, 3].
        centers: List of (cx, cy) tuples.
        **kwargs: Passed to compute_delta_z_ransac.

    Returns:
        delta_zs: List of Δz values (or None for each).
    """
    return [
        compute_delta_z_ransac(pts, ctr, **kwargs)
        for pts, ctr in zip(points_list, centers)
    ]
