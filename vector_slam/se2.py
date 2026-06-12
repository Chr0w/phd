"""SE2 transform utilities for the vector SLAM sandbox."""

import math
from typing import Tuple

import numpy as np

Point = Tuple[float, float]

EXCLUDED_ANGLE_THRESHOLD_DEG = 10.0
SCAN_ANGLE_NOISE_STD_DEG = 0.1


def transform_point(point: Point, tx: float, ty: float, theta_deg: float) -> Point:
    """Apply SE2 transform: p' = R(theta) @ p + [tx, ty]."""
    theta_rad = math.radians(theta_deg)
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)
    x, y = point
    return (cos_t * x - sin_t * y + tx, sin_t * x + cos_t * y + ty)


def transform_line(
    p1: Point, p2: Point, tx: float, ty: float, theta_deg: float
) -> Tuple[Point, Point]:
    """Transform both endpoints of a line segment."""
    return (
        transform_point(p1, tx, ty, theta_deg),
        transform_point(p2, tx, ty, theta_deg),
    )


def line_midpoint(p1: Point, p2: Point) -> Point:
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def rotate_point_about(center: Point, point: Point, delta_deg: float) -> Point:
    """Rotate point about center by delta_deg (counter-clockwise)."""
    cx, cy = center
    x, y = point[0] - cx, point[1] - cy
    theta = math.radians(delta_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return (cos_t * x - sin_t * y + cx, sin_t * x + cos_t * y + cy)


def add_scan_line_noise(
    p1: Point, p2: Point, rng: np.random.Generator
) -> Tuple[Point, Point]:
    """Apply Gaussian angular noise (std = SCAN_ANGLE_NOISE_STD_DEG) about midpoint."""
    noise_deg = float(rng.normal(0.0, SCAN_ANGLE_NOISE_STD_DEG))
    mid = line_midpoint(p1, p2)
    return (
        rotate_point_about(mid, p1, noise_deg),
        rotate_point_about(mid, p2, noise_deg),
    )


def acute_angle_between_vectors(vx1: float, vy1: float, vx2: float, vy2: float) -> float:
    """Acute angle in degrees between two 2D direction vectors."""
    len1 = math.hypot(vx1, vy1)
    len2 = math.hypot(vx2, vy2)
    if len1 < 1e-12 or len2 < 1e-12:
        return 0.0
    cos_signed = max(-1.0, min(1.0, (vx1 * vx2 + vy1 * vy2) / (len1 * len2)))
    angle = math.degrees(math.acos(cos_signed))
    return min(angle, 180.0 - angle)


def line_direction(p1: Point, p2: Point) -> Tuple[float, float]:
    return (p2[0] - p1[0], p2[1] - p1[1])


def pair_angle_diff_deg(static_p1: Point, static_p2: Point, scan_p1: Point, scan_p2: Point) -> float:
    """Acute angular difference between static line and scan line directions."""
    vx1, vy1 = line_direction(static_p1, static_p2)
    vx2, vy2 = line_direction(scan_p1, scan_p2)
    return acute_angle_between_vectors(vx1, vy1, vx2, vy2)


def relative_angle_deg(scan_p1: Point, scan_p2: Point, ep_x: float, ep_y: float) -> float:
    """Acute angle (deg) between scan line direction and vector midpoint→Ep."""
    mx, my = line_midpoint(scan_p1, scan_p2)
    vx_ep = ep_x - mx
    vy_ep = ep_y - my
    vx_line, vy_line = line_direction(scan_p1, scan_p2)
    return acute_angle_between_vectors(vx_line, vy_line, vx_ep, vy_ep)
