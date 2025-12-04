import numpy as np
import os
from pxr import Gf

try:
    import yaml
except Exception:
    yaml = None


def quaternion_to_yaw_radians(quaternion):
    """Convert quaternion to yaw angle in radians around Z-axis"""
    # Extract quaternion components (w, x, y, z)
    w, x, y, z = quaternion
    
    # Calculate yaw angle from quaternion
    # yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    yaw_radians = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    
    return yaw_radians

def yaw_degrees_to_quaternion(yaw_degrees):
    """
    Convert a yaw angle in degrees to a quaternion.
    
    Args:
        yaw_degrees (float): Yaw angle in degrees around the z-axis
        
    Returns:
        Gf.Quatd: USD quaternion object
    """
    # Convert degrees to radians
    yaw_radians = np.radians(yaw_degrees)
    half_yaw = yaw_radians / 2.0
    
    w = np.cos(half_yaw)
    x = 0.0
    y = 0.0
    z = np.sin(half_yaw)
    
    return Gf.Quatd(w, x, y, z)

def setup_missions(create_waypoint_func, missions_file_path=None):
    """
    Load missions from YAML file and create Mission objects.
    
    Args:
        create_waypoint_func: Function to create waypoint objects
        missions_file_path: Path to the missions YAML file (optional)
    
    Returns:
        tuple: (missions_list, current_mission_number, all_missions_completed, new_mission, start_position)
    """
    from mission import Mission, MissionType, StatusType
    
    # Default missions file path if not provided
    if missions_file_path is None:
        missions_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "missions", "mission_1.yaml")
    else:
        missions_file = missions_file_path
        
    waypoints = []
    start_position = []
    try:
        if yaml is not None:
            with open(missions_file, "r") as f:
                data = yaml.safe_load(f)
                waypoints = data.get("waypoints", []) if data else []
                # Read start_position from YAML if present
                if data and "start_position" in data:
                    start_position = data["start_position"]
    except FileNotFoundError:
        print(f"Missions file not found: {missions_file}")
    except Exception as e:
        print(f"Failed to load missions file {missions_file}: {e}")

    missions = []
    for i, wp in enumerate(waypoints):
        try:
            wx, wy = float(wp[0]), float(wp[1])
        except Exception:
            continue
        missions.append(Mission(i, MissionType.MOVE_TO_WAYPOINT, create_waypoint_func(wx, wy)))

    if missions:
        missions[0].set_status(StatusType.IN_PROGRESS)
    current_mission_number = 0
    all_missions_completed = False
    new_mission = True

    return missions, current_mission_number, all_missions_completed, new_mission, start_position

def check_if_at_waypoint(robot_get_world_pose_fn, mission, distance_threshold=0.3):
    """
    Determine whether the robot has reached the mission's waypoint.

    Args:
        robot_get_world_pose_fn: Callable returning (position_np, orientation_quat)
        mission: Mission instance with get_waypoint() -> Waypoint(x, y)
        distance_threshold: Maximum XY distance to consider "at waypoint"

    Returns:
        bool: True if robot is within the distance_threshold of the waypoint
    """
    robot_current_position, _ = robot_get_world_pose_fn()
    waypoint = mission.get_waypoint()
    waypoint_position = np.array([waypoint.x, waypoint.y, robot_current_position[2]])
    distance = np.linalg.norm(robot_current_position - waypoint_position)
    return distance < distance_threshold


def add_noise_to_array(array, noise_std=1.01):
    """
    Add noise to an array.
    """
    return array + np.random.normal(0, noise_std, array.shape)