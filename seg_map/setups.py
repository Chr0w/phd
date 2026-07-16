import os
import random

class Setup():
    def __init__(self, name: str, user: str, mission_file_path: str, map_usd_path: str = None, seed_nr: int = 1, robot_prim_name: str = "mir_bot_3", start_position: list = None, layout_development_mode: str = None) -> None:
        self._name = name
        self._user = user
        self._mission_file = mission_file_path
        self._map_usd_path = map_usd_path
        self._seed_nr = seed_nr
        self._robot_prim_name = robot_prim_name
        self._start_position = start_position
        self._layout_development_mode = layout_development_mode
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
    def start_position(self):
        return self._start_position

    @property
    def layout_development_mode(self):
        return self._layout_development_mode

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

    setups = [
        Setup(
            name="setup_1",
            user=user,
            mission_file_path=f"/home/{user}/isaac_sim_files/robots/mir_bot_3/mir_bot_3.usd",
            map_usd_path=f"/home/{user}/isaac_sim_files/maps/warehouse/01.usd",
            seed_nr=1,
            robot_prim_name="mir_bot_3",
            layout_development_mode="fill_up",
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
