import math
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import isaac_sim_utils as isu
import robot_utils
from pxr import Gf, UsdGeom, Sdf


# Match SegMap robot motion so agents travel the same way.
_AGENT_TOP_SPEED_MPS = 1.0
_LINEAR_ACCEL_MPS2 = 3.0
_ANGULAR_ACCEL_RADPS2 = 1.2
_MAX_ANGULAR_VELOCITY_RADPS = 2.5
_YAW_ALIGN_GAIN = 2.0
_WAYPOINT_ARRIVAL_M = 0.3
_OVERLAP_MARGIN_M = 0.1
_DYNAMIC_AGENTS_ROOT = "/map/dynamic_agents"
_DYNAMIC_AGENTS_INSTANCER = f"{_DYNAMIC_AGENTS_ROOT}/instancer"
_DEFAULT_AGENT_RADIUS_M = 0.5
_ROBOT_RADIUS_M = 0.5
_COLLISION_CHECK_SUBSAMPLE = 8


@dataclass(frozen=True)
class LayoutDevelopmentModeConfig:
    name: str
    runtime_minutes: float
    storage_utilization_start: float
    storage_utilization_target: float
    event_period_seconds: float
    event_random_actions: int
    dynamic_agents: int = 0


MODE_REGISTRY: dict[str, LayoutDevelopmentModeConfig] = {
    "no_changes": LayoutDevelopmentModeConfig(
        name="no_changes",
        runtime_minutes=20,
        storage_utilization_start=0.0,
        storage_utilization_target=0.0,
        event_period_seconds=0,
        event_random_actions=0,
    ),
    "fill_up": LayoutDevelopmentModeConfig(
        name="fill_up",
        runtime_minutes=15,
        storage_utilization_start=0.0,
        storage_utilization_target=0.8,
        event_period_seconds=10,
        event_random_actions=5,
    ),
    "overnight_changes": LayoutDevelopmentModeConfig(
        name="overnight_changes",
        runtime_minutes=5,
        storage_utilization_start=0.5,
        storage_utilization_target=0.5,
        event_period_seconds=0,
        event_random_actions=0,
    ),
    "random_changes_only": LayoutDevelopmentModeConfig(
        name="random_changes_only",
        runtime_minutes=15,
        storage_utilization_start=0.4,
        storage_utilization_target=0.4,
        event_period_seconds=5,
        event_random_actions=20,
    ),
    "test_reach": LayoutDevelopmentModeConfig(
        name="test_reach",
        runtime_minutes=30,
        storage_utilization_start=0.75,
        storage_utilization_target=0.75,
        event_period_seconds=0,
        event_random_actions=0,
    ),
    "test_dynamics": LayoutDevelopmentModeConfig(
        name="test_dynamics",
        runtime_minutes=15,
        storage_utilization_start=0.0,
        storage_utilization_target=0.0,
        event_period_seconds=0,
        event_random_actions=0,
        dynamic_agents=10,
    ),
}


def get_mode_config(mode_name: str) -> LayoutDevelopmentModeConfig:
    if mode_name not in MODE_REGISTRY:
        raise ValueError(
            f"Unknown layout development mode '{mode_name}'. "
            f"Available modes: {', '.join(sorted(MODE_REGISTRY))}"
        )
    return MODE_REGISTRY[mode_name]


def compute_actions_per_event(total_bins: int, config: LayoutDevelopmentModeConfig) -> float:
    if config.event_period_seconds <= 0 or total_bins <= 0:
        return 0.0
    num_events = (config.runtime_minutes * 60) / config.event_period_seconds
    net_bin_delta = (config.storage_utilization_target - config.storage_utilization_start) * total_bins
    return net_bin_delta / num_events


def compute_storage_utilization(occupied: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return max(0.0, min(1.0, float(occupied) / float(total)))


@dataclass(frozen=True)
class SimProgress:
    total_test_time_minutes: float
    percentage_complete: float
    minutes_passed: float
    minutes_left: float
    estimated_real_minutes_to_completion: float


class BinAssetManager:
    """Manages PointInstancer assets per (section_id, size) group based on bin occupancy."""

    def __init__(self, stage, user: str, seed_nr: int) -> None:
        self._stage = stage
        self._user = user
        self._seed_nr = seed_nr
        self._asset_pools: dict[str, list[str]] = {}
        self._bin_catalog: dict[str, dict] = {}
        self._occupied: set[str] = set()
        self._instances: dict[str, dict] = {}
        self._group_keys: dict[tuple[str, str], list[str]] = {}
        self._group_bin_index: dict[tuple[str, str], dict[str, int]] = {}
        self._group_proto_map: dict[tuple[str, str], dict[str, int]] = {}
        self._slot_templates: dict[str, dict] = {}
        self._groups_ready: set[tuple[str, str]] = set()
        self._pending_updates: deque[tuple[str, str, str, str]] = deque()
        self._section_order: list[str] = []
        self._section_rank: dict[str, int] = {}

    def initialize(
        self,
        layout: dict,
        origin_xy: list[float] | tuple[float, float] | None = None,
    ) -> None:
        self._asset_pools = {}
        for size in robot_utils.BIN_ASSET_SIZES:
            assets = robot_utils.list_bin_assets(self._user, size)
            if assets:
                self._asset_pools[size] = assets

        self._bin_catalog = {}
        self._group_keys = {}
        for bin_data in robot_utils.all_layout_bins(layout):
            key = bin_data["key"]
            self._bin_catalog[key] = bin_data
            group = (bin_data["section_id"], bin_data["size"])
            self._group_keys.setdefault(group, []).append(key)

        for group_keys in self._group_keys.values():
            group_keys.sort()

        self._section_order = robot_utils.layout_section_ids_by_distance(layout, origin_xy)
        self._section_rank = {
            section_id: index for index, section_id in enumerate(self._section_order)
        }

        self._group_bin_index = {
            group: {bin_key: index for index, bin_key in enumerate(keys)}
            for group, keys in self._group_keys.items()
        }
        self._slot_templates = {}
        self._groups_ready = set()
        self._pending_updates.clear()

    def _bin_spawn_sort_key(self, bin_key: str) -> tuple[int, str, int]:
        bin_data = self._bin_catalog[bin_key]
        section_rank = self._section_rank.get(bin_data["section_id"], 10**9)
        return (section_rank, bin_data["size"], bin_data["number"])

    @property
    def total_bins(self) -> int:
        return len(self._bin_catalog)

    @property
    def occupied_bins(self) -> set[str]:
        return set(self._occupied)

    def storage_utilization(self) -> float:
        return compute_storage_utilization(len(self._occupied), self.total_bins)

    def _rng_for_bin(self, bin_key: str) -> random.Random:
        bin_data = self._bin_catalog[bin_key]
        salt = bin_data["number"]
        salt += sum(ord(c) for c in bin_data["section_id"]) * 10000
        salt += sum(ord(c) for c in bin_data["size"]) * 100
        return random.Random(self._seed_nr + salt)

    def _slot_template_for_bin(self, bin_key: str) -> dict:
        if bin_key in self._slot_templates:
            return self._slot_templates[bin_key]

        bin_data = self._bin_catalog[bin_key]
        assets = self._asset_pools.get(bin_data["size"], [])
        template = {
            "asset_path": "",
            "x": bin_data["center_x"],
            "y": bin_data["center_y"],
            "z": 0.0,
            "yaw_degrees": 0.0,
        }
        if assets:
            rng = self._rng_for_bin(bin_key)
            template["asset_path"] = rng.choice(assets)
            template["yaw_degrees"] = rng.uniform(0.0, 360.0)

        self._slot_templates[bin_key] = template
        return template

    def _queue_visibility_update(self, op: str, bin_key: str) -> None:
        bin_data = self._bin_catalog[bin_key]
        self._pending_updates = deque(
            item for item in self._pending_updates if item[3] != bin_key
        )
        self._pending_updates.append((op, bin_data["section_id"], bin_data["size"], bin_key))

    def _ensure_group_ready(self, section_id: str, size: str) -> None:
        group = (section_id, size)
        if group in self._groups_ready:
            return
        self._create_group_shell(section_id, size)
        self._groups_ready.add(group)

    def _create_group_shell(self, section_id: str, size: str) -> None:
        group = (section_id, size)
        bin_keys = self._group_keys.get(group, [])
        if not bin_keys:
            return

        pool_assets = self._asset_pools.get(size, [])
        if not pool_assets:
            return

        instances = [self._slot_template_for_bin(bin_key) for bin_key in bin_keys]
        if not any(instance["asset_path"] for instance in instances):
            return

        UsdGeom.Xform.Define(self._stage, f"/map/assets/{section_id}")
        instancer_path = self._instancer_path(section_id, size)
        self._group_proto_map[group] = isu.create_point_instancer_shell(
            self._stage,
            instancer_path,
            instances,
            pool_assets,
        )

    def has_pending_usd_work(self) -> bool:
        return bool(self._pending_updates)

    def flush_one_update(self) -> bool:
        if not self._pending_updates:
            return False

        op, section_id, size, bin_key = self._pending_updates.popleft()
        self._ensure_group_ready(section_id, size)
        group = (section_id, size)
        slot_index = self._group_bin_index.get(group, {}).get(bin_key)
        if slot_index is None:
            return True

        instancer_path = self._instancer_path(section_id, size)
        if op == "show":
            instance = self._instances[bin_key]
            proto_index = self._group_proto_map[group][instance["asset_path"]]
            isu.show_point_instancer_slot(
                self._stage, instancer_path, slot_index, proto_index
            )
        else:
            isu.hide_point_instancer_slot(self._stage, instancer_path, slot_index)
        return True

    def _sync_group_visibility(self, section_id: str, size: str) -> None:
        group = (section_id, size)
        bin_keys = self._group_keys.get(group, [])
        if not bin_keys:
            return

        self._ensure_group_ready(section_id, size)
        proto_map = self._group_proto_map.get(group, {})
        proto_indices = []
        invisible = []
        for index, bin_key in enumerate(bin_keys):
            if bin_key in self._occupied:
                asset_path = self._instances[bin_key]["asset_path"]
                proto_indices.append(proto_map[asset_path])
            else:
                proto_indices.append(isu.EMPTY_PROTO_INDEX)
                invisible.append(index)

        instancer_path = self._instancer_path(section_id, size)
        isu.set_point_instancer_proto_indices(self._stage, instancer_path, proto_indices)
        isu.set_point_instancer_invisible_ids(self._stage, instancer_path, invisible)

    def apply_all_visibility(self) -> None:
        groups_to_sync: set[tuple[str, str]] = set()
        for bin_key in self._occupied:
            bin_data = self._bin_catalog[bin_key]
            groups_to_sync.add((bin_data["section_id"], bin_data["size"]))

        for section_id, size in groups_to_sync:
            self._sync_group_visibility(section_id, size)

    def _instancer_path(self, section_id: str, size: str) -> str:
        return f"/map/assets/{section_id}/{size}/instancer"

    def flush_pending_updates(self) -> None:
        while self._pending_updates:
            self.flush_one_update()

    def clear_all(self) -> None:
        occupied_before = list(self._occupied)
        self._occupied = set()
        self._instances = {}
        self._pending_updates.clear()
        for bin_key in occupied_before:
            self._queue_visibility_update("hide", bin_key)

    def spawn_all(self) -> int:
        self.clear_all()
        return self.set_occupied_bins(set(self._bin_catalog.keys()))

    def spawn_section(self, section_id: str) -> int:
        section_keys = {
            key
            for key, bin_data in self._bin_catalog.items()
            if bin_data["section_id"] == section_id
        }
        for bin_key in section_keys:
            if bin_key in self._occupied:
                self.remove_bin(bin_key, flush=False)
        spawned = 0
        for bin_key in sorted(section_keys, key=self._bin_spawn_sort_key):
            if self._add_bin_internal(bin_key):
                spawned += 1
                self._queue_visibility_update("show", bin_key)
        return spawned

    def set_occupied_bins(self, bin_keys: set[str]) -> int:
        self._occupied = set()
        self._instances = {}
        self._pending_updates.clear()
        spawned = 0

        for bin_key in sorted(bin_keys, key=self._bin_spawn_sort_key):
            if bin_key not in self._bin_catalog:
                continue
            if self._add_bin_internal(bin_key):
                spawned += 1
                self._queue_visibility_update("show", bin_key)

        return spawned

    def _add_bin_internal(self, bin_key: str) -> bool:
        if bin_key in self._occupied or bin_key not in self._bin_catalog:
            return False

        template = self._slot_template_for_bin(bin_key)
        if not template["asset_path"]:
            return False

        self._instances[bin_key] = dict(template)
        self._occupied.add(bin_key)
        return True

    def add_bin(self, bin_key: str, *, flush: bool = False) -> bool:
        if not self._add_bin_internal(bin_key):
            return False
        bin_data = self._bin_catalog[bin_key]
        if flush:
            self._ensure_group_ready(bin_data["section_id"], bin_data["size"])
            self._sync_group_visibility(bin_data["section_id"], bin_data["size"])
        else:
            self._queue_visibility_update("show", bin_key)
        return True

    def remove_bin(self, bin_key: str, *, flush: bool = False) -> bool:
        if bin_key not in self._occupied:
            return False
        bin_data = self._bin_catalog[bin_key]
        self._occupied.discard(bin_key)
        self._instances.pop(bin_key, None)
        if flush:
            self._sync_group_visibility(bin_data["section_id"], bin_data["size"])
        else:
            self._queue_visibility_update("hide", bin_key)
        return True

    def empty_bins(self) -> list[str]:
        return [key for key in self._bin_catalog if key not in self._occupied]

    def occupied_bin_list(self) -> list[str]:
        return list(self._occupied)


@dataclass
class _DynamicAgentState:
    slot: int
    asset_path: str
    solid_proto: int
    ghost_proto: int
    current_waypoint_id: str
    path: list[str]
    path_index: int
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    radius: float = _DEFAULT_AGENT_RADIUS_M
    previous_speed: float = 0.0
    previous_angular_velocity: float = 0.0
    collision_enabled: bool = True


class DynamicAgentsController:
    """
    Drive many dynamic agents via a single PointInstancer.

    Shared USD prototypes give visual realism; convex-hull colliders on the
    solid prototypes give lidar returns close to the mesh shape. Per-frame
    cost is two array writes (positions + orientations) regardless of count.
    Pass-through swaps an instance to a collision-free ghost prototype of the
    same asset so visuals stay while agents/robot can overlap.
    """

    def __init__(
        self,
        stage,
        layout: dict,
        agent_count: int,
        user: str,
        seed_nr: int,
    ) -> None:
        self._stage = stage
        self._agent_count = int(agent_count)
        self._user = user
        self._rng = random.Random(seed_nr + 7919)
        self._positions = robot_utils.layout_waypoint_positions(layout)
        self._graph = robot_utils.build_waypoint_graph(layout)
        self._agents: list[_DynamicAgentState] = []
        self._pair_pass_through: dict[tuple[int, int], int] = {}
        self._spawned = False
        self._tick = 0
        self._instancer_path = _DYNAMIC_AGENTS_INSTANCER
        self._instancer = None
        self._positions_attr = None
        self._orientations_attr = None
        self._proto_indices_attr = None
        self._gf_positions: list = []
        self._gf_orientations: list = []
        self._proto_indices: list[int] = []
        self._asset_radius: dict[str, float] = {}
        self._validate_spawn_requirements()

    @property
    def agent_count(self) -> int:
        return len(self._agents)

    def _validate_spawn_requirements(self) -> None:
        if self._agent_count <= 0:
            return
        waypoint_count = len(self._positions)
        if waypoint_count < self._agent_count:
            raise RuntimeError(
                f"Need at least {self._agent_count} waypoints to spawn "
                f"{self._agent_count} dynamic agents, found {waypoint_count}"
            )
        assets = robot_utils.list_dynamic_agent_assets(self._user)
        if not assets:
            raise RuntimeError(
                f"No dynamic agent USD assets found in {robot_utils.dynamic_agent_assets_dir(self._user)}"
            )

    def spawn(self) -> None:
        if self._spawned or self._agent_count <= 0:
            return

        self._validate_spawn_requirements()
        waypoint_ids = list(self._positions.keys())
        assets = robot_utils.list_dynamic_agent_assets(self._user)

        spawn_ids = waypoint_ids[:]
        self._rng.shuffle(spawn_ids)
        spawn_ids = spawn_ids[: self._agent_count]

        UsdGeom.Xform.Define(self._stage, _DYNAMIC_AGENTS_ROOT)
        solid_proto, ghost_proto = self._create_instancer_prototypes(assets)

        self._agents = []
        self._gf_positions = []
        self._gf_orientations = []
        self._proto_indices = []

        for index, waypoint_id in enumerate(spawn_ids):
            asset_path = self._rng.choice(assets)
            x, y = self._positions[waypoint_id]
            agent = _DynamicAgentState(
                slot=index,
                asset_path=asset_path,
                solid_proto=solid_proto[asset_path],
                ghost_proto=ghost_proto[asset_path],
                current_waypoint_id=waypoint_id,
                path=[waypoint_id],
                path_index=0,
                x=float(x),
                y=float(y),
                yaw=0.0,
                radius=self._asset_radius.get(asset_path, _DEFAULT_AGENT_RADIUS_M),
            )
            self._assign_new_path(agent)
            self._agents.append(agent)
            self._gf_positions.append(Gf.Vec3f(agent.x, agent.y, 0.0))
            self._gf_orientations.append(isu._yaw_degrees_to_quat_h(0.0))
            self._proto_indices.append(agent.solid_proto)

        self._positions_attr.Set(self._gf_positions)
        self._orientations_attr.Set(self._gf_orientations)
        self._proto_indices_attr.Set(self._proto_indices)

        self._spawned = True
        print(
            f"Spawned {len(self._agents)} dynamic agent(s) via PointInstancer at waypoints: "
            + ", ".join(spawn_ids)
        )

    def _create_instancer_prototypes(
        self, assets: list[str]
    ) -> tuple[dict[str, int], dict[str, int]]:
        """
        Build shared solid (lidar collision) + ghost (pass-through) prototypes.

        Prototype indices are dense from 0. For each asset:
          even index = solid (convex-hull colliders)
          odd index  = ghost (same mesh, no collision)
        """
        prototypes_root = f"{self._instancer_path}/Prototypes"
        UsdGeom.Xform.Define(self._stage, prototypes_root)

        solid_proto: dict[str, int] = {}
        ghost_proto: dict[str, int] = {}
        proto_paths: list[str] = []
        next_index = 0

        for asset_path in assets:
            solid_path = f"{prototypes_root}/proto_{next_index:02d}"
            isu.add_reference_to_stage(usd_path=asset_path, prim_path=solid_path)
            isu._strip_rigid_bodies_recursive(self._stage, solid_path)
            isu.apply_convex_hull_colliders(self._stage, solid_path)
            solid_proto[asset_path] = next_index
            proto_paths.append(solid_path)
            self._asset_radius[asset_path] = self._measure_xy_radius(solid_path)
            next_index += 1

            ghost_path = f"{prototypes_root}/proto_{next_index:02d}"
            isu.add_reference_to_stage(usd_path=asset_path, prim_path=ghost_path)
            isu._strip_rigid_bodies_recursive(self._stage, ghost_path)
            isu.strip_collisions_recursive(self._stage, ghost_path)
            ghost_proto[asset_path] = next_index
            proto_paths.append(ghost_path)
            next_index += 1

        self._instancer = UsdGeom.PointInstancer.Define(self._stage, self._instancer_path)
        self._instancer.CreatePrototypesRel().SetTargets(
            [Sdf.Path(path) for path in proto_paths]
        )
        self._positions_attr = self._instancer.CreatePositionsAttr([])
        self._orientations_attr = self._instancer.CreateOrientationsAttr([])
        self._proto_indices_attr = self._instancer.CreateProtoIndicesAttr([])
        return solid_proto, ghost_proto

    def _measure_xy_radius(self, prim_path: str) -> float:
        try:
            obb = isu._prim_obb_components(self._stage, prim_path)
            if obb is None:
                return _DEFAULT_AGENT_RADIUS_M
            _center, _axes, half = obb
            radius = float(math.hypot(half[0], half[1]))
            if radius < 1e-3:
                return _DEFAULT_AGENT_RADIUS_M
            return min(radius, 1.5)
        except Exception:
            return _DEFAULT_AGENT_RADIUS_M

    def update(
        self,
        step_size: float,
        robot_xy: tuple[float, float] | None = None,
        robot_radius: float = _ROBOT_RADIUS_M,
        robot_xy_fn=None,
    ) -> None:
        if not self._spawned or not self._agents:
            return

        self._tick += 1
        for agent in self._agents:
            self._step_agent_motion(agent, step_size)

        # One batched USD write for all agents.
        with Sdf.ChangeBlock():
            for agent in self._agents:
                self._gf_positions[agent.slot] = Gf.Vec3f(agent.x, agent.y, 0.0)
                self._gf_orientations[agent.slot] = isu._yaw_degrees_to_quat_h(
                    math.degrees(agent.yaw)
                )
            self._positions_attr.Set(self._gf_positions)
            self._orientations_attr.Set(self._gf_orientations)

        if self._tick % _COLLISION_CHECK_SUBSAMPLE == 0:
            if robot_xy is None and robot_xy_fn is not None:
                robot_xy = robot_xy_fn()
            self._update_pass_through_collisions(robot_xy, robot_radius)

    def _assign_new_path(self, agent: _DynamicAgentState) -> None:
        candidates = [
            waypoint_id
            for waypoint_id in self._positions
            if waypoint_id != agent.current_waypoint_id
        ]
        self._rng.shuffle(candidates)

        for target_id in candidates:
            path = robot_utils.shortest_waypoint_path(
                self._graph, agent.current_waypoint_id, target_id
            )
            if path and len(path) >= 2:
                agent.path = path
                agent.path_index = 1
                return

        agent.path = [agent.current_waypoint_id]
        agent.path_index = 0

    def _current_target_xy(self, agent: _DynamicAgentState) -> tuple[float, float] | None:
        if agent.path_index >= len(agent.path):
            return None
        return self._positions[agent.path[agent.path_index]]

    def _step_agent_motion(self, agent: _DynamicAgentState, step_size: float) -> None:
        target = self._current_target_xy(agent)
        if target is None:
            self._assign_new_path(agent)
            target = self._current_target_xy(agent)
            if target is None:
                return

        target_x, target_y = target
        dx = target_x - agent.x
        dy = target_y - agent.y
        target_yaw = math.atan2(dy, dx)
        current_yaw = agent.yaw

        yaw_error = target_yaw - current_yaw
        while yaw_error > math.pi:
            yaw_error -= 2.0 * math.pi
        while yaw_error < -math.pi:
            yaw_error += 2.0 * math.pi

        desired_angular = max(
            -_MAX_ANGULAR_VELOCITY_RADPS,
            min(_MAX_ANGULAR_VELOCITY_RADPS, _YAW_ALIGN_GAIN * yaw_error),
        )
        angular_delta = desired_angular - agent.previous_angular_velocity
        max_angular_delta = _ANGULAR_ACCEL_RADPS2 * step_size
        if angular_delta > max_angular_delta:
            angular_velocity = agent.previous_angular_velocity + max_angular_delta
        elif angular_delta < -max_angular_delta:
            angular_velocity = agent.previous_angular_velocity - max_angular_delta
        else:
            angular_velocity = desired_angular

        target_speed = _AGENT_TOP_SPEED_MPS * max(0.0, math.cos(yaw_error))
        distance = math.hypot(dx, dy)
        if distance < 0.5:
            target_speed = max(target_speed * distance / 2.5, 0.5)

        max_speed_delta = _LINEAR_ACCEL_MPS2 * step_size
        speed_delta = target_speed - agent.previous_speed
        if speed_delta > max_speed_delta:
            speed = agent.previous_speed + max_speed_delta
        elif speed_delta < -max_speed_delta:
            speed = agent.previous_speed - max_speed_delta
        else:
            speed = target_speed

        agent.x += math.cos(current_yaw) * speed * step_size
        agent.y += math.sin(current_yaw) * speed * step_size
        agent.yaw = current_yaw + angular_velocity * step_size
        agent.previous_speed = speed
        agent.previous_angular_velocity = angular_velocity

        arrived_distance = math.hypot(target_x - agent.x, target_y - agent.y)
        if arrived_distance < _WAYPOINT_ARRIVAL_M:
            agent.current_waypoint_id = agent.path[agent.path_index]
            agent.path_index += 1
            agent.previous_speed = 0.0
            agent.previous_angular_velocity = 0.0
            if agent.path_index >= len(agent.path):
                self._assign_new_path(agent)

    @staticmethod
    def _circles_within_overlap_margin(
        ax: float,
        ay: float,
        ar: float,
        bx: float,
        by: float,
        br: float,
        margin_m: float = _OVERLAP_MARGIN_M,
    ) -> bool:
        limit = ar + br + margin_m
        dx = ax - bx
        dy = ay - by
        return (dx * dx + dy * dy) <= (limit * limit)

    def _update_pass_through_collisions(
        self,
        robot_xy: tuple[float, float] | None,
        robot_radius: float,
    ) -> None:
        should_disable = [False] * len(self._agents)
        active_pairs: set[tuple[int, int]] = set()

        for i in range(len(self._agents)):
            a = self._agents[i]
            for j in range(i + 1, len(self._agents)):
                b = self._agents[j]
                if not self._circles_within_overlap_margin(
                    a.x, a.y, a.radius, b.x, b.y, b.radius
                ):
                    continue
                pair = (i, j)
                active_pairs.add(pair)
                if pair not in self._pair_pass_through:
                    self._pair_pass_through[pair] = self._rng.choice([i, j])
                should_disable[self._pair_pass_through[pair]] = True

        for pair in list(self._pair_pass_through):
            if pair not in active_pairs:
                del self._pair_pass_through[pair]

        if robot_xy is not None:
            rx, ry = robot_xy
            for index, agent in enumerate(self._agents):
                if self._circles_within_overlap_margin(
                    agent.x, agent.y, agent.radius, rx, ry, robot_radius
                ):
                    should_disable[index] = True

        proto_changed = False
        for index, agent in enumerate(self._agents):
            enabled = not should_disable[index]
            if agent.collision_enabled == enabled:
                continue
            agent.collision_enabled = enabled
            self._proto_indices[agent.slot] = (
                agent.solid_proto if enabled else agent.ghost_proto
            )
            proto_changed = True

        if proto_changed:
            with Sdf.ChangeBlock():
                self._proto_indices_attr.Set(self._proto_indices)


class LayoutDevelopmentController:
    def __init__(
        self,
        stage,
        layout: dict,
        config: LayoutDevelopmentModeConfig,
        user: str,
        seed_nr: int,
        origin_xy: list[float] | tuple[float, float] | None = None,
    ) -> None:
        self._config = config
        self._seed_nr = seed_nr
        self._rng = random.Random(seed_nr)
        self._asset_manager = BinAssetManager(stage, user, seed_nr)
        self._asset_manager.initialize(layout, origin_xy=origin_xy)

        self._actions_per_event = compute_actions_per_event(
            self._asset_manager.total_bins, config
        )
        self._tracking_carry = 0.0
        self._start_sim_time: Optional[float] = None
        self._last_event_time: Optional[float] = None
        self._started = False
        self._finished = False
        self._wall_start_time: Optional[float] = None
        self._dynamic_agents: Optional[DynamicAgentsController] = None
        if config.dynamic_agents > 0:
            self._dynamic_agents = DynamicAgentsController(
                stage,
                layout,
                config.dynamic_agents,
                user,
                seed_nr,
            )

    @property
    def asset_manager(self) -> BinAssetManager:
        return self._asset_manager

    @property
    def config(self) -> LayoutDevelopmentModeConfig:
        return self._config

    @property
    def dynamic_agents(self) -> Optional[DynamicAgentsController]:
        return self._dynamic_agents

    @property
    def storage_utilization(self) -> float:
        return self._asset_manager.storage_utilization()

    @property
    def total_bins(self) -> int:
        return self._asset_manager.total_bins

    @property
    def occupied_count(self) -> int:
        return len(self._asset_manager.occupied_bins)

    def start(self, sim_time: float) -> None:
        if self._started:
            return

        # Validate agents before arming the run so a spawn failure does not
        # leave the mission callback half-started / disabled.
        if self._dynamic_agents is not None:
            self._dynamic_agents._validate_spawn_requirements()

        self._started = True
        self._start_sim_time = sim_time
        self._last_event_time = sim_time
        self._wall_start_time = time.monotonic()

        initial_count = self._initial_occupied_count()
        all_keys = list(self._asset_manager._bin_catalog.keys())
        self._rng.shuffle(all_keys)
        initial_keys = set(all_keys[:initial_count])
        spawned = self._asset_manager.set_occupied_bins(initial_keys)

        print(
            f"Layout development mode '{self._config.name}' started: "
            f"{spawned}/{self.total_bins} bins occupied "
            f"(SU={self.storage_utilization:.3f}, target={self._config.storage_utilization_target})"
        )

        if self._dynamic_agents is not None:
            self._dynamic_agents.spawn()

    def _initial_occupied_count(self) -> int:
        return int(round(self._config.storage_utilization_start * self.total_bins))

    def is_finished(self, sim_time: float) -> bool:
        if not self._started or self._start_sim_time is None:
            return False
        elapsed_minutes = (sim_time - self._start_sim_time) / 60.0
        return elapsed_minutes >= self._config.runtime_minutes

    def test_progress(self, sim_time: float) -> Optional[SimProgress]:
        if not self._started or self._start_sim_time is None:
            return None

        total_minutes = float(self._config.runtime_minutes)
        minutes_passed = max(0.0, (sim_time - self._start_sim_time) / 60.0)
        minutes_left = max(0.0, total_minutes - minutes_passed)
        if total_minutes > 0.0:
            percentage_complete = min(100.0, (minutes_passed / total_minutes) * 100.0)
        else:
            percentage_complete = 100.0

        estimated_real_minutes_to_completion = minutes_left
        if self._wall_start_time is not None and percentage_complete > 0.0:
            real_minutes_passed = (time.monotonic() - self._wall_start_time) / 60.0
            estimated_total_real_minutes = real_minutes_passed / (percentage_complete / 100.0)
            estimated_real_minutes_to_completion = max(
                0.0, estimated_total_real_minutes - real_minutes_passed
            )

        return SimProgress(
            total_test_time_minutes=total_minutes,
            percentage_complete=percentage_complete,
            minutes_passed=minutes_passed,
            minutes_left=minutes_left,
            estimated_real_minutes_to_completion=estimated_real_minutes_to_completion,
        )

    def update(
        self,
        sim_time: float,
        step_size: float | None = None,
        robot_xy: tuple[float, float] | None = None,
        robot_xy_fn=None,
    ) -> None:
        if not self._started or self._finished:
            return

        if self.is_finished(sim_time):
            self._finished = True
            print(
                f"Layout development mode '{self._config.name}' finished after "
                f"{self._config.runtime_minutes} min (SU={self.storage_utilization:.3f})"
            )
            return

        if self._dynamic_agents is not None and step_size is not None:
            self._dynamic_agents.update(
                step_size, robot_xy=robot_xy, robot_xy_fn=robot_xy_fn
            )

        if self._config.event_period_seconds <= 0:
            return
        if self._last_event_time is None:
            self._last_event_time = sim_time
            return

        if sim_time - self._last_event_time < self._config.event_period_seconds:
            return

        self._last_event_time = sim_time
        self._run_event()

    def _run_event(self) -> None:
        tracking_count = self._next_tracking_action_count()
        net_delta = self._config.storage_utilization_target - self._config.storage_utilization_start

        if net_delta >= 0:
            for _ in range(tracking_count):
                self._try_add_random_bin()
        else:
            for _ in range(tracking_count):
                self._try_remove_random_bin()

        for _ in range(self._config.event_random_actions):
            if self._rng.random() < 0.5:
                self._try_add_random_bin(max_attempts=5)
            else:
                self._try_remove_random_bin(max_attempts=5)

    def _next_tracking_action_count(self) -> int:
        self._tracking_carry += self._actions_per_event
        count = int(self._tracking_carry)
        self._tracking_carry -= count
        return count

    def _try_add_random_bin(self, max_attempts: int = 1) -> bool:
        empty = self._asset_manager.empty_bins()
        if not empty:
            return False
        for _ in range(max_attempts):
            bin_key = self._rng.choice(empty)
            if self._asset_manager.add_bin(bin_key):
                return True
            empty = self._asset_manager.empty_bins()
            if not empty:
                break
        return False

    def _try_remove_random_bin(self, max_attempts: int = 1) -> bool:
        occupied = self._asset_manager.occupied_bin_list()
        if not occupied:
            return False
        for _ in range(max_attempts):
            bin_key = self._rng.choice(occupied)
            if self._asset_manager.remove_bin(bin_key):
                return True
            occupied = self._asset_manager.occupied_bin_list()
            if not occupied:
                break
        return False

    def clear_all_objects(self) -> None:
        self._asset_manager.clear_all()

    def spawn_all_objects(self) -> int:
        return self._asset_manager.spawn_all()
