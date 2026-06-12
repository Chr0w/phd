"""Flask app for the SE2 Vector SLAM sandbox simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template, request

from gmm import fit_angle_gmm
from persistence import clear_snapshot, load_snapshot, save_snapshot
from se2 import (
    EXCLUDED_ANGLE_THRESHOLD_DEG,
    add_scan_line_noise,
    line_midpoint,
    pair_signed_angle_diff_deg,
    relative_angle_deg,
    rotate_line_about,
    transform_line,
)

app = Flask(__name__)

TRUE_POSE = (0.0, 0.0, 0.0)


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
    _next_id: int = 1
    _rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def recompute_scan_lines(self) -> None:
        self.scan_lines = []
        if self.estimated_pose is None:
            return
        tx, ty, theta = self.estimated_pose
        for line in self.static_lines:
            p1, p2 = transform_line(line.p1, line.p2, tx, ty, theta)
            p1, p2 = add_scan_line_noise(p1, p2, self._rng)
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

    def reset(self) -> None:
        self.static_lines = []
        self.estimated_pose = None
        self.scan_lines = []
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

        return {
            "true_pose": {"x": TRUE_POSE[0], "y": TRUE_POSE[1], "theta": TRUE_POSE[2]},
            "estimated_pose": est,
            "static_lines": [line.to_dict() for line in self.static_lines],
            "scan_lines": [line.to_dict() for line in self.scan_lines],
            "aligned_scan_lines": aligned_scan_lines,
            "alpha_peak_deg": alpha_peak,
            "line_pairs": line_pairs,
            "pair_angle_diffs": pair_angle_diffs,
            "angle_gmm": gmm,
            "excluded_angle_threshold_deg": EXCLUDED_ANGLE_THRESHOLD_DEG,
            "counts": {
                "static": len(self.static_lines),
                "scan": len(self.scan_lines),
                "excluded": excluded_count,
            },
        }


def _load_persisted_state() -> GameState:
    snapshot = load_snapshot()
    if snapshot is not None:
        return GameState.from_snapshot(snapshot)
    return GameState()


def _persist_state() -> None:
    save_snapshot(state.to_snapshot())


state = _load_persisted_state()


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
    try:
        x = float(data["x"])
        y = float(data["y"])
        theta = float(data["theta"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "x, y, and theta required as numbers"}), 400
    state.set_estimated_pose(x, y, theta)
    _persist_state()
    return jsonify(state.to_dict())


@app.route("/api/reset", methods=["POST"])
def reset():
    state.reset()
    clear_snapshot()
    return jsonify(state.to_dict())


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
