import os
import random
from mission import Mission, MissionType, Waypoint, StatusType

class Setup():
    def __init__(self, name: str, user: str, mission_file_path: str, map_usd_path: str = None, seed_nr: int = 1, robot_prim_name: str = "mir_bot_2") -> None:
        self._name = name
        self._user = user
        self._mission_file = mission_file_path
        self._map_usd_path = map_usd_path
        self._seed_nr = seed_nr
        self._robot_prim_name = robot_prim_name
        self._waypoints = []
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
    def seed_nr(self):
        return self._seed_nr
    
    @property
    def robot_prim_name(self):
        return self._robot_prim_name

    @property
    def waypoints(self):
        return self._waypoints

    def set_waypoints(self, waypoints: list[Waypoint]):
        self._waypoints = waypoints

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
    
    setups = [
        Setup(
            name="setup_1",
            user=user,
            mission_file_path=f"/home/{user}/isaac_sim_files/robots/mir_bot_2/mir_bot_2.usd",
            map_usd_path=f"/home/{user}/isaac_sim_files/map_2_for_import.usd",
            seed_nr=1,
            robot_prim_name="mir_bot_2"
        ),
        # Add more setups here as needed
        # Setup(
        #     name="setup_2",
        #     user=user,
        #     mission_file_path=f"/home/{user}/isaac_sim_files/robots/other_robot/other_robot.usd",
        #     map_usd_path=f"/home/{user}/isaac_sim_files/map_1_for_import.usd",
        #     seed_nr=2,
        #     robot_prim_name="other_robot"
        # ),
    ]
    
    return setups


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

