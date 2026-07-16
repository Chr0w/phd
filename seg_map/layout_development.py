import random
import time
from dataclasses import dataclass
from typing import Optional

import isaac_sim_utils as isu
import robot_utils
from pxr import Usd, UsdGeom, Sdf


@dataclass(frozen=True)
class LayoutDevelopmentModeConfig:
    name: str
    runtime_minutes: float
    storage_utilization_start: float
    storage_utilization_target: float
    event_period_seconds: float
    event_random_actions: int
    static_fill_fraction: Optional[float] = None


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
        runtime_minutes=20,
        storage_utilization_start=0.0,
        storage_utilization_target=0.7,
        event_period_seconds=20,
        event_random_actions=10,
    ),
    "overnight_changes": LayoutDevelopmentModeConfig(
        name="overnight_changes",
        runtime_minutes=20,
        storage_utilization_start=0.5,
        storage_utilization_target=0.5,
        event_period_seconds=0,
        event_random_actions=0,
        static_fill_fraction=0.5,
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

    def initialize(self, layout: dict) -> None:
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

    def _instancer_path(self, section_id: str, size: str) -> str:
        return f"/map/assets/{section_id}/{size}/instancer"

    def _rebuild_group(self, section_id: str, size: str) -> None:
        group = (section_id, size)
        instancer_path = self._instancer_path(section_id, size)
        prim = self._stage.GetPrimAtPath(Sdf.Path(instancer_path))
        if prim and prim.IsValid():
            self._stage.RemovePrim(Sdf.Path(instancer_path))

        instances = []
        for bin_key in self._group_keys.get(group, []):
            if bin_key not in self._occupied:
                continue
            instance = self._instances[bin_key]
            instances.append(
                {
                    "asset_path": instance["asset_path"],
                    "x": instance["x"],
                    "y": instance["y"],
                    "z": instance.get("z", 0.0),
                    "yaw_degrees": instance["yaw_degrees"],
                }
            )

        if not instances:
            return

        UsdGeom.Xform.Define(self._stage, f"/map/assets/{section_id}")
        isu.spawn_point_instancer(self._stage, instancer_path, instances)

    def _rebuild_all_groups(self) -> None:
        for section_id, size in self._group_keys:
            self._rebuild_group(section_id, size)

    def clear_all(self) -> None:
        self._occupied = set()
        self._instances = {}
        for section_id, size in self._group_keys:
            instancer_path = self._instancer_path(section_id, size)
            prim = self._stage.GetPrimAtPath(Sdf.Path(instancer_path))
            if prim and prim.IsValid():
                self._stage.RemovePrim(Sdf.Path(instancer_path))

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
                self.remove_bin(bin_key)
        spawned = 0
        for bin_key in sorted(section_keys):
            if self._add_bin_internal(bin_key):
                spawned += 1
        for size in robot_utils.BIN_ASSET_SIZES:
            if any(self._bin_catalog[key]["size"] == size for key in section_keys):
                self._rebuild_group(section_id, size)
        return spawned

    def set_occupied_bins(self, bin_keys: set[str]) -> int:
        self._occupied = set()
        self._instances = {}
        spawned = 0

        for bin_key in sorted(bin_keys):
            if bin_key not in self._bin_catalog:
                continue
            if self._add_bin_internal(bin_key):
                spawned += 1

        self._rebuild_all_groups()
        return spawned

    def _add_bin_internal(self, bin_key: str) -> bool:
        if bin_key in self._occupied or bin_key not in self._bin_catalog:
            return False

        bin_data = self._bin_catalog[bin_key]
        assets = self._asset_pools.get(bin_data["size"], [])
        if not assets:
            return False

        rng = self._rng_for_bin(bin_key)
        self._instances[bin_key] = {
            "asset_path": rng.choice(assets),
            "x": bin_data["center_x"],
            "y": bin_data["center_y"],
            "z": 0.0,
            "yaw_degrees": rng.uniform(0.0, 360.0),
        }
        self._occupied.add(bin_key)
        return True

    def add_bin(self, bin_key: str) -> bool:
        if not self._add_bin_internal(bin_key):
            return False
        bin_data = self._bin_catalog[bin_key]
        self._rebuild_group(bin_data["section_id"], bin_data["size"])
        return True

    def remove_bin(self, bin_key: str) -> bool:
        if bin_key not in self._occupied:
            return False
        bin_data = self._bin_catalog[bin_key]
        self._occupied.discard(bin_key)
        self._instances.pop(bin_key, None)
        self._rebuild_group(bin_data["section_id"], bin_data["size"])
        return True

    def empty_bins(self) -> list[str]:
        return [key for key in self._bin_catalog if key not in self._occupied]

    def occupied_bin_list(self) -> list[str]:
        return list(self._occupied)


class LayoutDevelopmentController:
    def __init__(
        self,
        stage,
        layout: dict,
        config: LayoutDevelopmentModeConfig,
        user: str,
        seed_nr: int,
    ) -> None:
        self._config = config
        self._seed_nr = seed_nr
        self._rng = random.Random(seed_nr)
        self._asset_manager = BinAssetManager(stage, user, seed_nr)
        self._asset_manager.initialize(layout)

        self._actions_per_event = compute_actions_per_event(
            self._asset_manager.total_bins, config
        )
        self._tracking_carry = 0.0
        self._start_sim_time: Optional[float] = None
        self._last_event_time: Optional[float] = None
        self._started = False
        self._finished = False
        self._wall_start_time: Optional[float] = None

    @property
    def asset_manager(self) -> BinAssetManager:
        return self._asset_manager

    @property
    def config(self) -> LayoutDevelopmentModeConfig:
        return self._config

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

    def _initial_occupied_count(self) -> int:
        if self._config.static_fill_fraction is not None:
            fraction = self._config.static_fill_fraction
        else:
            fraction = self._config.storage_utilization_start
        return int(round(fraction * self.total_bins))

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

    def update(self, sim_time: float) -> None:
        if not self._started or self._finished:
            return
        if self.is_finished(sim_time):
            self._finished = True
            print(
                f"Layout development mode '{self._config.name}' finished after "
                f"{self._config.runtime_minutes} min (SU={self.storage_utilization:.3f})"
            )
            return

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
