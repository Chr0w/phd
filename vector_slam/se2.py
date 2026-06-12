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


def rotate_line_about(
    p1: Point, p2: Point, center: Point, delta_deg: float
) -> Tuple[Point, Point]:
    """Rotate both endpoints of a line about center."""
    return (
        rotate_point_about(center, p1, delta_deg),
        rotate_point_about(center, p2, delta_deg),
    )


def add_scan_line_noise(
    p1: Point,
    p2: Point,
    rng: np.random.Generator,
    noise_std_deg: float = SCAN_ANGLE_NOISE_STD_DEG,
) -> Tuple[Point, Point]:
    """Apply Gaussian angular noise about midpoint."""
    noise_deg = float(rng.normal(0.0, noise_std_deg))
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


def _signed_direction_diff_deg(vx1: float, vy1: float, vx2: float, vy2: float) -> float:
    """Signed CCW angle (deg) from direction (vx2, vy2) to (vx1, vy1)."""
    a1 = math.atan2(vy1, vx1)
    a2 = math.atan2(vy2, vx2)
    diff = a1 - a2
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff < -math.pi:
        diff += 2 * math.pi
    return math.degrees(diff)


def pair_signed_angle_diff_deg(
    static_p1: Point, static_p2: Point, scan_p1: Point, scan_p2: Point
) -> float:
    """Signed angle (deg) from scan line direction to static line direction."""
    vx1, vy1 = line_direction(static_p1, static_p2)
    vx2, vy2 = line_direction(scan_p1, scan_p2)
    return _signed_direction_diff_deg(vx1, vy1, vx2, vy2)


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


def line_unit_normal(p1: Point, p2: Point) -> Tuple[float, float]:
    """Unit normal to the line (90° CCW from p1→p2 direction)."""
    dx, dy = line_direction(p1, p2)
    length = math.hypot(dx, dy)
    if length < 1e-12:
        return (0.0, 0.0)
    return (-dy / length, dx / length)


def signed_distance_to_line(point: Point, line_p1: Point, line_p2: Point) -> float:
    """Signed distance from point to the infinite line through line_p1–line_p2."""
    nx, ny = line_unit_normal(line_p1, line_p2)
    vx = point[0] - line_p1[0]
    vy = point[1] - line_p1[1]
    return vx * nx + vy * ny


def foot_of_perpendicular(point: Point, line_p1: Point, line_p2: Point) -> Point:
    """Foot of the perpendicular from point onto the infinite line."""
    d = signed_distance_to_line(point, line_p1, line_p2)
    nx, ny = line_unit_normal(line_p1, line_p2)
    return (point[0] - d * nx, point[1] - d * ny)


def signed_distance_between_lines(
    line_a_p1: Point, line_a_p2: Point, line_b_p1: Point, line_b_p2: Point
) -> float:
    """Signed distance from line A to line B along A's unit normal."""
    nx, ny = line_unit_normal(line_a_p1, line_a_p2)
    vx = line_b_p1[0] - line_a_p1[0]
    vy = line_b_p1[1] - line_a_p1[1]
    return vx * nx + vy * ny


def intersect_lines(
    line_a_p1: Point, line_a_p2: Point, line_b_p1: Point, line_b_p2: Point
) -> Point | None:
    """Intersection of two infinite lines, or None if parallel."""
    x1, y1 = line_a_p1
    x2, y2 = line_a_p2
    x3, y3 = line_b_p1
    x4, y4 = line_b_p2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def intersection_angle_weight_deg(angle_deg: float) -> float:
    """Weight from line intersection angle: 90° → 1, 0° → 0."""
    acute = min(angle_deg, 180.0 - angle_deg)
    return math.sin(math.radians(acute))
