"""Flask app for the SE2 Vector SLAM sandbox simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import math
import numpy as np
from flask import Flask, jsonify, render_template, request

from gmm import fit_angle_gmm
from persistence import clear_snapshot, load_snapshot, save_snapshot
from se2 import (
    EXCLUDED_ANGLE_THRESHOLD_DEG,
    SCAN_ANGLE_NOISE_STD_DEG,
    add_scan_line_noise,
    acute_angle_between_vectors,
    foot_of_perpendicular,
    intersect_lines,
    intersection_angle_weight_deg,
    line_direction,
    line_midpoint,
    line_unit_normal,
    pair_signed_angle_diff_deg,
    relative_angle_deg,
    rotate_line_about,
    signed_distance_between_lines,
    transform_line,
)

app = Flask(__name__)

TRUE_POSE = (0.0, 0.0, 0.0)
RANDOM_POSE_XY_STD_M = 0.2
RANDOM_POSE_YAW_MAX_DEG = 10.0
RANDOM_LINE_HALF_EXTENT_M = 12.0
RANDOM_LINE_MIN_LEN_M = 2.0
RANDOM_LINE_MAX_LEN_M = 10.0
RANDOM_LINE_MIN_ORIGIN_DIST_M = 1.0
BATCH_SIM_MAX_TRIALS = 500


@dataclass
class Line:
    id: int
    p1: tuple[float, float]
    p2: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        mid = line_midpoint(self.p1, self.p2)
        return {
            "id": self.id,
            "p1": list(self.p1),
            "p2": list(self.p2),
            "midpoint": list(mid),
        }


@dataclass
class GameState:
    static_lines: list[Line] = field(default_factory=list)
    estimated_pose: tuple[float, float, float] | None = None
    scan_lines: list[Line] = field(default_factory=list)
    min_intersection_angle_deg: float = 5.0
    scan_angle_noise_std_deg: float = SCAN_ANGLE_NOISE_STD_DEG
    _next_id: int = 1
    _rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def recompute_scan_lines(self) -> None:
        self.scan_lines = []
        if self.estimated_pose is None:
            return
        tx, ty, theta = self.estimated_pose
        for line in self.static_lines:
            p1, p2 = transform_line(line.p1, line.p2, tx, ty, theta)
            p1, p2 = add_scan_line_noise(
                p1, p2, self._rng, self.scan_angle_noise_std_deg
            )
            self.scan_lines.append(Line(id=line.id, p1=p1, p2=p2))

    def add_static_line(self, p1: tuple[float, float], p2: tuple[float, float]) -> Line:
        line = Line(id=self._next_id, p1=p1, p2=p2)
        self._next_id += 1
        self.static_lines.append(line)
        self.recompute_scan_lines()
        return line

    def update_static_line(
        self,
        line_id: int,
        p1: tuple[float, float] | None = None,
        p2: tuple[float, float] | None = None,
    ) -> Line | None:
        for line in self.static_lines:
            if line.id == line_id:
                if p1 is not None:
                    line.p1 = p1
                if p2 is not None:
                    line.p2 = p2
                self.recompute_scan_lines()
                return line
        return None

    def set_estimated_pose(self, x: float, y: float, theta: float) -> None:
        self.estimated_pose = (x, y, theta)
        self.recompute_scan_lines()

    def randomize_estimated_pose(self) -> None:
        x = float(self._rng.normal(0.0, RANDOM_POSE_XY_STD_M))
        y = float(self._rng.normal(0.0, RANDOM_POSE_XY_STD_M))
        theta = float(
            self._rng.uniform(-RANDOM_POSE_YAW_MAX_DEG, RANDOM_POSE_YAW_MAX_DEG)
        )
        self.set_estimated_pose(x, y, theta)

    def _clear_scene(self) -> None:
        self.static_lines = []
        self.estimated_pose = None
        self.scan_lines = []
        self._next_id = 1

    def _add_random_static_line(self) -> Line:
        normal_angle = float(self._rng.uniform(0.0, 2.0 * math.pi))
        origin_dist = float(
            self._rng.uniform(RANDOM_LINE_MIN_ORIGIN_DIST_M, RANDOM_LINE_HALF_EXTENT_M)
        )
        foot_x = math.cos(normal_angle) * origin_dist
        foot_y = math.sin(normal_angle) * origin_dist
        line_angle = normal_angle + math.pi / 2.0
        half_len = float(
            self._rng.uniform(RANDOM_LINE_MIN_LEN_M, RANDOM_LINE_MAX_LEN_M) / 2.0
        )
        dx = math.cos(line_angle) * half_len
        dy = math.sin(line_angle) * half_len
        line = Line(
            id=self._next_id,
            p1=(foot_x - dx, foot_y - dy),
            p2=(foot_x + dx, foot_y + dy),
        )
        self._next_id += 1
        self.static_lines.append(line)
        return line

    def add_random_static_lines(self, count: int) -> None:
        for _ in range(max(0, count)):
            self._add_random_static_line()

    def run_single_trial(self) -> dict[str, Any]:
        self._clear_scene()
        line_count = int(self._rng.integers(1, 6))
        self.add_random_static_lines(line_count)
        self.randomize_estimated_pose()
        snapshot = self.to_dict()
        return _extract_trial_errors(snapshot, line_count)

    def reset(self) -> None:
        self.static_lines = []
        self.estimated_pose = None
        self.scan_lines = []
        self.min_intersection_angle_deg = 5.0
        self.scan_angle_noise_std_deg = SCAN_ANGLE_NOISE_STD_DEG
        self._next_id = 1
        self._rng = np.random.default_rng()

    def to_snapshot(self) -> dict[str, Any]:
        est = None
        if self.estimated_pose is not None:
            x, y, theta = self.estimated_pose
            est = {"x": x, "y": y, "theta": theta}
        return {
            "version": 1,
            "next_id": self._next_id,
            "static_lines": [
                {"id": line.id, "p1": list(line.p1), "p2": list(line.p2)}
                for line in self.static_lines
            ],
            "scan_lines": [
                {"id": line.id, "p1": list(line.p1), "p2": list(line.p2)}
                for line in self.scan_lines
            ],
            "estimated_pose": est,
            "rng_state": self._rng.bit_generator.state,
            "min_intersection_angle_deg": self.min_intersection_angle_deg,
            "scan_angle_noise_std_deg": self.scan_angle_noise_std_deg,
        }

    @classmethod
    def from_snapshot(cls, data: dict[str, Any]) -> GameState:
        state = cls()
        state._next_id = int(data.get("next_id", 1))
        for item in data.get("static_lines", []):
            state.static_lines.append(
                Line(
                    id=int(item["id"]),
                    p1=(float(item["p1"][0]), float(item["p1"][1])),
                    p2=(float(item["p2"][0]), float(item["p2"][1])),
                )
            )
        for item in data.get("scan_lines", []):
            state.scan_lines.append(
                Line(
                    id=int(item["id"]),
                    p1=(float(item["p1"][0]), float(item["p1"][1])),
                    p2=(float(item["p2"][0]), float(item["p2"][1])),
                )
            )
        est = data.get("estimated_pose")
        if est is not None:
            state.estimated_pose = (float(est["x"]), float(est["y"]), float(est["theta"]))
        state.min_intersection_angle_deg = float(
            data.get("min_intersection_angle_deg", 5.0)
        )
        state.scan_angle_noise_std_deg = float(
            data.get("scan_angle_noise_std_deg", SCAN_ANGLE_NOISE_STD_DEG)
        )
        rng_state = data.get("rng_state")
        if rng_state is not None:
            rng = np.random.default_rng()
            rng.bit_generator.state = rng_state
            state._rng = rng
        return state

    def _build_line_pairs(self) -> list[dict[str, Any]]:
        pairs: list[dict[str, Any]] = []
        static_by_id = {line.id: line for line in self.static_lines}
        ep = self.estimated_pose

        for index, scan in enumerate(self.scan_lines, start=1):
            static = static_by_id.get(scan.id)
            scan_mid = line_midpoint(scan.p1, scan.p2)
            static_mid = line_midpoint(static.p1, static.p2) if static else None

            pair: dict[str, Any] = {
                "id": scan.id,
                "index": index,
                "static_line": static.to_dict() if static else None,
                "scan_line": scan.to_dict(),
                "static_midpoint": list(static_mid) if static_mid else None,
                "scan_midpoint": list(scan_mid),
                "midpoint": list(scan_mid),
                "relative_angle_deg": None,
                "excluded": False,
            }
            if ep is not None:
                ep_x, ep_y, _ = ep
                angle = relative_angle_deg(scan.p1, scan.p2, ep_x, ep_y)
                pair["relative_angle_deg"] = round(angle, 2)
                pair["excluded"] = angle < EXCLUDED_ANGLE_THRESHOLD_DEG
            pairs.append(pair)
        return pairs

    def _build_pair_angle_diffs(self, pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        diffs: list[dict[str, Any]] = []
        static_by_id = {line.id: line for line in self.static_lines}
        scan_by_id = {line.id: line for line in self.scan_lines}

        for pair in pairs:
            static = static_by_id.get(pair["id"])
            scan = scan_by_id.get(pair["id"])
            if static is None or scan is None:
                continue
            angle = pair_signed_angle_diff_deg(static.p1, static.p2, scan.p1, scan.p2)
            diffs.append(
                {
                    "id": pair["id"],
                    "index": pair["index"],
                    "angle_deg": round(angle, 4),
                }
            )
        return diffs

    def _build_aligned_scan_lines(self, alpha_peak: float | None) -> list[dict[str, Any]]:
        if self.estimated_pose is None or alpha_peak is None:
            return []
        ep_x, ep_y, _ = self.estimated_pose
        center = (ep_x, ep_y)
        aligned: list[dict[str, Any]] = []
        for line in self.scan_lines:
            p1, p2 = rotate_line_about(line.p1, line.p2, center, alpha_peak)
            mid = line_midpoint(p1, p2)
            aligned.append(
                {
                    "id": line.id,
                    "p1": list(p1),
                    "p2": list(p2),
                    "midpoint": list(mid),
                }
            )
        return aligned

    def _build_correction_vectors(
        self,
        pairs: list[dict[str, Any]],
        aligned_scan_lines: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if self.estimated_pose is None or not aligned_scan_lines:
            return []

        ep_x, ep_y, _ = self.estimated_pose
        sensor = (ep_x, ep_y)
        static_by_id = {line.id: line for line in self.static_lines}
        aligned_by_id = {item["id"]: item for item in aligned_scan_lines}
        corrections: list[dict[str, Any]] = []

        for pair in pairs:
            line_id = pair["id"]
            static = static_by_id.get(line_id)
            aligned = aligned_by_id.get(line_id)
            if static is None or aligned is None:
                continue

            a_p1 = (float(aligned["p1"][0]), float(aligned["p1"][1]))
            a_p2 = (float(aligned["p2"][0]), float(aligned["p2"][1]))
            s_p1 = static.p1
            s_p2 = static.p2

            foot = foot_of_perpendicular(sensor, a_p1, a_p2)
            nx, ny = line_unit_normal(a_p1, a_p2)
            distance = signed_distance_between_lines(a_p1, a_p2, s_p1, s_p2)
            end = (ep_x + distance * nx, ep_y + distance * ny)

            dx, dy = line_direction(a_p1, a_p2)
            line_len = math.hypot(dx, dy)
            if line_len < 1e-12:
                half_len = 0.0
                tx, ty = 1.0, 0.0
            else:
                half_len = line_len / 2
                tx, ty = dx / line_len, dy / line_len
            prob_p1 = (end[0] - half_len * tx, end[1] - half_len * ty)
            prob_p2 = (end[0] + half_len * tx, end[1] + half_len * ty)

            corrections.append(
                {
                    "id": line_id,
                    "index": pair["index"],
                    "excluded": pair["excluded"],
                    "direction_line": {
                        "p1": list(sensor),
                        "p2": list(foot),
                    },
                    "correction_distance": round(distance, 4),
                    "correction_vector": {
                        "start": list(sensor),
                        "end": list(end),
                    },
                    "probability_line": {
                        "p1": list(prob_p1),
                        "p2": list(prob_p2),
                    },
                }
            )
        return corrections

    def _build_probability_intersections(
        self,
        corrections: list[dict[str, Any]],
        min_angle_deg: float,
    ) -> tuple[list[dict[str, Any]], dict[str, float] | None]:
        active = [c for c in corrections if not c["excluded"]]
        intersections: list[dict[str, Any]] = []
        nr = 0

        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                ci = active[i]
                cj = active[j]
                pl_i = ci["probability_line"]
                pl_j = cj["probability_line"]
                ip1 = (float(pl_i["p1"][0]), float(pl_i["p1"][1]))
                ip2 = (float(pl_i["p2"][0]), float(pl_i["p2"][1]))
                jp1 = (float(pl_j["p1"][0]), float(pl_j["p1"][1]))
                jp2 = (float(pl_j["p2"][0]), float(pl_j["p2"][1]))

                point = intersect_lines(ip1, ip2, jp1, jp2)
                if point is None:
                    continue

                vix, viy = line_direction(ip1, ip2)
                vjx, vjy = line_direction(jp1, jp2)
                angle_deg = acute_angle_between_vectors(vix, viy, vjx, vjy)
                weight = intersection_angle_weight_deg(angle_deg)
                nr += 1

                intersections.append(
                    {
                        "nr": nr,
                        "id_a": ci["id"],
                        "id_b": cj["id"],
                        "index_a": ci["index"],
                        "index_b": cj["index"],
                        "point": [round(point[0], 4), round(point[1], 4)],
                        "angle_deg": round(angle_deg, 2),
                        "weight": round(weight, 4),
                        "excluded": angle_deg < min_angle_deg,
                    }
                )

        included = [item for item in intersections if not item["excluded"]]
        weighted_position = None

        if len(active) == 1:
            end = active[0]["correction_vector"]["end"]
            weighted_position = {
                "x": round(float(end[0]), 4),
                "y": round(float(end[1]), 4),
            }
        else:
            total_weight = sum(item["weight"] for item in included)
            if total_weight > 1e-12:
                wx = sum(item["point"][0] * item["weight"] for item in included)
                wy = sum(item["point"][1] * item["weight"] for item in included)
                weighted_position = {
                    "x": round(wx / total_weight, 4),
                    "y": round(wy / total_weight, 4),
                }

        return intersections, weighted_position

    def to_dict(self) -> dict[str, Any]:
        est = None
        if self.estimated_pose is not None:
            x, y, theta = self.estimated_pose
            est = {"x": x, "y": y, "theta": theta}

        line_pairs = self._build_line_pairs()
        excluded_count = sum(1 for p in line_pairs if p["excluded"])
        pair_angle_diffs = self._build_pair_angle_diffs(line_pairs)
        angle_values = [d["angle_deg"] for d in pair_angle_diffs]
        gmm = fit_angle_gmm(angle_values)
        alpha_peak = gmm["peak"]["x"] if gmm and gmm.get("peak") else None
        aligned_scan_lines = self._build_aligned_scan_lines(alpha_peak)
        correction_vectors = self._build_correction_vectors(line_pairs, aligned_scan_lines)
        probability_intersections, weighted_position = self._build_probability_intersections(
            correction_vectors, self.min_intersection_angle_deg
        )
        included_intersections = sum(
            1 for item in probability_intersections if not item["excluded"]
        )

        return {
            "true_pose": {"x": TRUE_POSE[0], "y": TRUE_POSE[1], "theta": TRUE_POSE[2]},
            "estimated_pose": est,
            "static_lines": [line.to_dict() for line in self.static_lines],
            "scan_lines": [line.to_dict() for line in self.scan_lines],
            "aligned_scan_lines": aligned_scan_lines,
            "correction_vectors": correction_vectors,
            "probability_intersections": probability_intersections,
            "weighted_position": weighted_position,
            "min_intersection_angle_deg": self.min_intersection_angle_deg,
            "scan_angle_noise_std_deg": self.scan_angle_noise_std_deg,
            "alpha_peak_deg": alpha_peak,
            "line_pairs": line_pairs,
            "pair_angle_diffs": pair_angle_diffs,
            "angle_gmm": gmm,
            "excluded_angle_threshold_deg": EXCLUDED_ANGLE_THRESHOLD_DEG,
            "counts": {
                "static": len(self.static_lines),
                "scan": len(self.scan_lines),
                "excluded": excluded_count,
                "intersections": len(probability_intersections),
                "intersections_included": included_intersections,
            },
        }


def _wrap_angle_deg(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def _extract_trial_errors(snapshot: dict[str, Any], line_count: int) -> dict[str, Any]:
    true_x, true_y, true_theta = TRUE_POSE
    pose_error: float | None = None
    yaw_error: float | None = None

    weighted = snapshot.get("weighted_position")
    if weighted is not None:
        pose_error = math.hypot(weighted["x"] - true_x, weighted["y"] - true_y)

    estimated = snapshot.get("estimated_pose")
    alpha_peak = snapshot.get("alpha_peak_deg")
    if estimated is not None and alpha_peak is not None:
        corrected_theta = _wrap_angle_deg(estimated["theta"] + alpha_peak)
        yaw_error = abs(corrected_theta - true_theta)
    elif estimated is not None:
        yaw_error = abs(_wrap_angle_deg(estimated["theta"]) - true_theta)

    return {
        "line_count": line_count,
        "pose_error_m": round(pose_error, 6) if pose_error is not None else None,
        "yaw_error_deg": round(yaw_error, 6) if yaw_error is not None else None,
    }


def _compute_batch_statistics(values: list[float | None]) -> dict[str, Any] | None:
    data = [float(v) for v in values if v is not None]
    if not data:
        return None

    arr = np.array(data, dtype=float)
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    rms = float(np.sqrt(np.mean(arr * arr)))

    n_bins = int(max(min(len(data) // 2, 20), 5))
    counts, edges = np.histogram(arr, bins=n_bins)
    mode: float | None = None
    if counts.sum() > 0:
        idx = int(np.argmax(counts))
        mode = float((edges[idx] + edges[idx + 1]) / 2.0)

    return {
        "mean": round(mean, 6),
        "median": round(median, 6),
        "mode": round(mode, 6) if mode is not None else None,
        "rms": round(rms, 6),
        "count": len(data),
    }


def _load_persisted_state() -> GameState:
    snapshot = load_snapshot()
    if snapshot is not None:
        return GameState.from_snapshot(snapshot)
    return GameState()


def _persist_state() -> None:
    save_snapshot(state.to_snapshot())


state = _load_persisted_state()


def _apply_settings(data: dict[str, Any]) -> bool:
    """Apply optional settings fields. Returns True if noise std changed."""
    noise_changed = False
    if "min_intersection_angle_deg" in data:
        state.min_intersection_angle_deg = max(
            0.0, float(data["min_intersection_angle_deg"])
        )
    if "scan_angle_noise_std_deg" in data:
        new_noise = max(0.0, float(data["scan_angle_noise_std_deg"]))
        if new_noise != state.scan_angle_noise_std_deg:
            state.scan_angle_noise_std_deg = new_noise
            noise_changed = True
    return noise_changed


def _parse_point(data: dict[str, Any], key: str) -> tuple[float, float] | None:
    val = data.get(key)
    if val is None or not isinstance(val, (list, tuple)) or len(val) != 2:
        return None
    return (float(val[0]), float(val[1]))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state", methods=["GET"])
def get_state():
    return jsonify(state.to_dict())


@app.route("/api/static_line", methods=["POST"])
def create_static_line():
    data = request.get_json(silent=True) or {}
    p1 = _parse_point(data, "p1")
    p2 = _parse_point(data, "p2")
    if p1 is None or p2 is None:
        return jsonify({"error": "p1 and p2 required as [x, y]"}), 400
    state.add_static_line(p1, p2)
    _persist_state()
    return jsonify(state.to_dict())


@app.route("/api/static_line/<int:line_id>", methods=["PATCH"])
def update_static_line(line_id: int):
    data = request.get_json(silent=True) or {}
    p1 = _parse_point(data, "p1") if "p1" in data else None
    p2 = _parse_point(data, "p2") if "p2" in data else None
    if p1 is None and p2 is None:
        return jsonify({"error": "p1 and/or p2 required"}), 400
    line = state.update_static_line(line_id, p1, p2)
    if line is None:
        return jsonify({"error": "line not found"}), 404
    _persist_state()
    return jsonify(state.to_dict())


@app.route("/api/move_sensor", methods=["POST"])
def move_sensor():
    data = request.get_json(silent=True) or {}
    has_pose = all(k in data for k in ("x", "y", "theta"))
    has_settings = (
        "min_intersection_angle_deg" in data or "scan_angle_noise_std_deg" in data
    )
    if not has_pose and not has_settings:
        return jsonify({"error": "pose and/or settings required"}), 400

    noise_changed = _apply_settings(data)
    if has_pose:
        try:
            x = float(data["x"])
            y = float(data["y"])
            theta = float(data["theta"])
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": "x, y, and theta required as numbers"}), 400
        state.estimated_pose = (x, y, theta)
        state.recompute_scan_lines()
    elif noise_changed:
        state.recompute_scan_lines()

    _persist_state()
    return jsonify(state.to_dict())


@app.route("/api/random_pose", methods=["POST"])
def random_pose():
    state.randomize_estimated_pose()
    _persist_state()
    return jsonify(state.to_dict())


@app.route("/api/settings", methods=["POST"])
def update_settings():
    data = request.get_json(silent=True) or {}
    noise_changed = _apply_settings(data)
    if noise_changed:
        state.recompute_scan_lines()
    _persist_state()
    return jsonify(state.to_dict())


@app.route("/api/batch_simulate", methods=["POST"])
def batch_simulate():
    data = request.get_json(silent=True) or {}
    try:
        n = int(data.get("n", 50))
    except (TypeError, ValueError):
        return jsonify({"error": "n must be an integer"}), 400
    n = max(1, min(n, BATCH_SIM_MAX_TRIALS))

    trials: list[dict[str, Any]] = []
    for _ in range(n):
        trials.append(state.run_single_trial())

    pose_errors = [t["pose_error_m"] for t in trials]
    yaw_errors = [t["yaw_error_deg"] for t in trials]

    _persist_state()
    response = state.to_dict()
    response["batch_results"] = {
        "n": n,
        "trials": trials,
        "pose_errors_m": pose_errors,
        "yaw_errors_deg": yaw_errors,
        "pose_statistics": _compute_batch_statistics(pose_errors),
        "yaw_statistics": _compute_batch_statistics(yaw_errors),
    }
    return jsonify(response)


@app.route("/api/reset", methods=["POST"])
def reset():
    state.reset()
    clear_snapshot()
    return jsonify(state.to_dict())


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
