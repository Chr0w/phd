import os
import random
from dataclasses import dataclass
from typing import Optional

DEFAULT_START_DELAY_SECONDS = 1.0
DEFAULT_LAYOUT_NAME = "paper_test.yaml"


def layouts_dir(user: str | None = None) -> str:
    if not user:
        user = os.environ.get("USER", "unknown")
    return f"/home/{user}/devcontainer/ros2_ws/shared_files/layouts"

@dataclass(frozen=True)
class LayoutDevelopmentModeConfig:
    name: str
    runtime_minutes: float
    storage_utilization_start: float
    storage_utilization_target: float
    event_period_seconds: float
    event_random_actions: int
    dynamic_agents: int = 0
    # Cosine pulse between low and high. One half period is low-to-high.
    # None keeps the open-loop drift from start toward target.
    storage_pulse_low: Optional[float] = None
    storage_pulse_high: Optional[float] = None
    storage_pulse_half_period_minutes: Optional[float] = None

class Setup():
    def __init__(self, name: str, user: str, mission_file_path: str, map_usd_path: str = None, seed_nr: int = 1, robot_prim_name: str = "mir_bot_3", start_position: list = None, only_plan_anchor_waypoints: bool = False, run_waypoints_in_order: bool = False, start_delay_seconds_override: float = None, runtime_minutes: float = 0.0, storage_utilization_start: float = 0.0, storage_utilization_target: float = 0.0, event_period_seconds: float = 0.0, event_random_actions: int = 0, dynamic_agents: int = 0, storage_pulse_low: float = None, storage_pulse_high: float = None, storage_pulse_half_period_minutes: float = None, layout_name: str = DEFAULT_LAYOUT_NAME) -> None:
        self._name = name
        self._user = user
        self._mission_file = mission_file_path
        self._map_usd_path = map_usd_path
        self._layout_name = layout_name
        self._seed_nr = seed_nr
        self._robot_prim_name = robot_prim_name
        self._start_position = start_position
        self._only_plan_anchor_waypoints = only_plan_anchor_waypoints
        self._run_waypoints_in_order = run_waypoints_in_order
        self._start_delay_seconds_override = start_delay_seconds_override
        self._layout_development = LayoutDevelopmentModeConfig(
            name=name,
            runtime_minutes=runtime_minutes,
            storage_utilization_start=storage_utilization_start,
            storage_utilization_target=storage_utilization_target,
            event_period_seconds=event_period_seconds,
            event_random_actions=event_random_actions,
            dynamic_agents=dynamic_agents,
            storage_pulse_low=storage_pulse_low,
            storage_pulse_high=storage_pulse_high,
            storage_pulse_half_period_minutes=storage_pulse_half_period_minutes,
        )
    @property
    def name(self):
        return self._name
    
    @property
    def user(self):
        return self._user
    
    @property
    def mission_file(self):
        return self._mission_file

    @property
    def map_usd_path(self):
        return self._map_usd_path

    @property
    def layout_name(self):
        return self._layout_name

    @property
    def layout_path(self):
        return os.path.join(layouts_dir(self._user), self._layout_name)
    
    @property
    def seed_nr(self):
        return self._seed_nr
    
    @property
    def robot_prim_name(self):
        return self._robot_prim_name

    @property
    def start_position(self):
        return self._start_position

    @property
    def layout_development(self):
        return self._layout_development

    @property
    def only_plan_anchor_waypoints(self):
        return self._only_plan_anchor_waypoints

    @property
    def run_waypoints_in_order(self):
        return self._run_waypoints_in_order

    @property
    def start_delay_seconds_override(self):
        return self._start_delay_seconds_override

    @property
    def start_delay_seconds(self):
        if self._start_delay_seconds_override is None:
            return DEFAULT_START_DELAY_SECONDS
        return float(self._start_delay_seconds_override)

    def set_seed_nr(self, seed_nr: int):
        self._seed_nr = seed_nr

def get_all_setups(user: str = None):
    """
    Get all available setups.
    
    Args:
        user: User name (optional, defaults to environment USER)
    
    Returns:
        list: List of Setup objects
    """
    if user is None:
        user = os.environ.get("USER", "unknown")

    mir_bot = "mir_bot_4"
    setups = [
        Setup(
            name="slam_mapping",
            user=user,
            mission_file_path=f"/home/{user}/isaac_sim_files/robots/{mir_bot}/{mir_bot}.usd",
            map_usd_path=f"/home/{user}/isaac_sim_files/maps/warehouse/01.usd",
            seed_nr=1,
            robot_prim_name=mir_bot,
            only_plan_anchor_waypoints=False,
            run_waypoints_in_order=True,
            runtime_minutes=20,
            storage_utilization_start=100.0,
            storage_utilization_target=100.0,
            event_period_seconds=0,
            event_random_actions=0,
        ),
        Setup(
            name="setup_1",
            user=user,
            mission_file_path=f"/home/{user}/isaac_sim_files/robots/{mir_bot}/{mir_bot}.usd",
            map_usd_path=f"/home/{user}/isaac_sim_files/maps/warehouse/01.usd",
            seed_nr=1,
            robot_prim_name=mir_bot,
            only_plan_anchor_waypoints=True,
            run_waypoints_in_order=False,
            runtime_minutes=15,
            storage_utilization_start=0.0,
            storage_utilization_target=0.8,
            event_period_seconds=10,
            event_random_actions=5,
        ),
        Setup(
            name="test_reach",
            user=user,
            mission_file_path=f"/home/{user}/isaac_sim_files/robots/{mir_bot}/{mir_bot}.usd",
            map_usd_path=f"/home/{user}/isaac_sim_files/maps/test_reach/test_reach.usd",
            seed_nr=1,
            robot_prim_name=mir_bot,
            only_plan_anchor_waypoints=False,
            run_waypoints_in_order=True,
            start_delay_seconds_override=5,
            runtime_minutes=30,
            storage_utilization_start=0.75,
            storage_utilization_target=0.75,
            event_period_seconds=0,
            event_random_actions=0,
        ),
        Setup(
            name="test_dynamics",
            user=user,
            mission_file_path=f"/home/{user}/isaac_sim_files/robots/{mir_bot}/{mir_bot}.usd",
            map_usd_path=f"/home/{user}/isaac_sim_files/maps/warehouse/01.usd",
            seed_nr=1,
            robot_prim_name=mir_bot,
            only_plan_anchor_waypoints=False,
            run_waypoints_in_order=False,
            runtime_minutes=15,
            storage_utilization_start=0.0,
            storage_utilization_target=0.0,
            event_period_seconds=0,
            event_random_actions=0,
            dynamic_agents=100,
        ),
        Setup(
            name="paper_test_1",
            user=user,
            mission_file_path=f"/home/{user}/isaac_sim_files/robots/{mir_bot}/{mir_bot}.usd",
            map_usd_path=f"/home/{user}/isaac_sim_files/maps/warehouse/01.usd",
            seed_nr=1,
            robot_prim_name=mir_bot,
            only_plan_anchor_waypoints=False,
            run_waypoints_in_order=False,
            runtime_minutes=10080,  # 7 days
            # Cosine starts at the trough. Peaks at 85% after 3 hours, back at 15% after 6.
            storage_utilization_start=0.15,
            storage_utilization_target=0.85,
            event_period_seconds=5,
            event_random_actions=50,
            dynamic_agents=10,
            storage_pulse_low=0.15,
            storage_pulse_high=0.85,
            storage_pulse_half_period_minutes=180.0,
        ),
        Setup(
            name="test_degeneracy",
            user=user,
            mission_file_path=f"/home/{user}/isaac_sim_files/robots/{mir_bot}/{mir_bot}.usd",
            map_usd_path=f"/home/{user}/isaac_sim_files/maps/warehouse/01.usd",
            seed_nr=1,
            robot_prim_name=mir_bot,
            only_plan_anchor_waypoints=False,
            run_waypoints_in_order=True,
            runtime_minutes=1,
            storage_utilization_start=1.0,
            storage_utilization_target=1.0,
            event_period_seconds=0,
            event_random_actions=0,
        ),
        Setup(
            name="baseline",
            user=user,
            mission_file_path=f"/home/{user}/isaac_sim_files/robots/{mir_bot}/{mir_bot}.usd",
            map_usd_path=f"/home/{user}/isaac_sim_files/maps/warehouse/01.usd",
            seed_nr=1,
            robot_prim_name=mir_bot,
            only_plan_anchor_waypoints=False,
            run_waypoints_in_order=True,
            runtime_minutes=10,
            storage_utilization_start=0.5,
            storage_utilization_target=0.5,
            event_period_seconds=0,
            event_random_actions=0,
        ),        
    ]
    
    return setups


def get_setup_from_name(name: str):
    setups = get_all_setups()
    for setup in setups:
        if setup.name == name:
            return setup
    return None

def get_setup_from_seed_nr(seed_nr: int):
    setups = get_all_setups()
    for setup in setups:
        if setup.seed_nr == seed_nr:
            return setup
    return None

def get_random_setup(user: str = None):
    """
    Get a random setup from all available setups.
    
    Args:
        user: User name (optional, defaults to environment USER)
    
    Returns:
        Setup: A randomly selected Setup object
    """
    setups = get_all_setups(user)
    return random.choice(setups)
