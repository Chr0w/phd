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


def yaw_degrees_to_orientation(yaw_degrees):
    """Return (w, x, y, z) numpy quaternion for Isaac Sim set_world_pose."""
    yaw_radians = np.radians(yaw_degrees)
    half_yaw = yaw_radians / 2.0
    return np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)])


def get_start_yaw_degrees(start_position):
    """Extract yaw in degrees from start_position [x, y] or [x, y, yaw]."""
    if start_position and len(start_position) >= 3:
        return float(start_position[2])
    return 0.0


def read_start_info_from_yaml(yaml_path=None):
    """
    Read start position and seed from YAML file.
    This function reads the file fresh each time it's called.
    
    Args:
        yaml_path: Path to the YAML file containing start position (defaults to /home/{USER}/devcontainer/ros2_ws/start_pos/start_pos.yaml)
    
    Returns:
        tuple: (start_position, seed_nr) where start_position is [x, y] or [x, y, yaw_degrees] and seed_nr is int
    """
    if yaml_path is None:
        user = os.environ.get("USER")
        yaml_path = f"/home/{user}/devcontainer/ros2_ws/start_pos/start_pos.yaml"
    if yaml is None:
        raise ImportError("yaml module is required to read start position. Install it with: pip install pyyaml")
    try:
        with open(yaml_path, "r") as f:
            start_pos_data = yaml.safe_load(f)
            if start_pos_data and "robot_start_x" in start_pos_data and "robot_start_y" in start_pos_data:
                x = float(start_pos_data["robot_start_x"])
                y = float(start_pos_data["robot_start_y"])
                start_position = [x, y]
                if "robot_start_yaw" in start_pos_data:
                    start_position.append(float(start_pos_data["robot_start_yaw"]))

                # Read seed
                if "seed" in start_pos_data:
                    seed_nr = int(start_pos_data["seed"])
                else:
                    raise ValueError(f"start_pos.yaml must contain 'seed' field. Got: {start_pos_data}")
                
                return start_position, seed_nr
            else:
                raise ValueError(f"start_pos.yaml must contain 'robot_start_x' and 'robot_start_y' fields. Got: {start_pos_data}")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Start position file not found: {yaml_path}") from e
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Failed to read start position from {yaml_path}: {e}") from e


def setup_missions(create_waypoint_func, missions_file_path=None):
    """
    Load missions from YAML file and create Mission objects.
    
    Args:
        create_waypoint_func: Function to create waypoint objects
        missions_file_path: Path to the missions YAML file
    
    Returns:
        tuple: (missions_list, current_mission_number, all_missions_completed, new_mission, start_position)
    """
    from mission import Mission, MissionType, StatusType

    waypoint_list = []
    start_position = []

    if missions_file_path is None:
        missions_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "missions", "mission_1.yaml")
    else:
        missions_file = missions_file_path

    try:
        if yaml is not None:
            with open(missions_file, "r") as f:
                data = yaml.safe_load(f)
                waypoint_list = data.get("waypoints", []) if data else []
                if data and "start_position" in data:
                    start_position = data["start_position"]
    except FileNotFoundError:
        print(f"Missions file not found: {missions_file}")
    except Exception as e:
        print(f"Failed to load missions file {missions_file}: {e}")

    missions = []
    for i, wp in enumerate(waypoint_list):
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


class square:
    def __init__(self, ll, ur, color, name):
        self.ll = ll # Lower left
        self.ur = ur # Upper right
        self.color = color
        self.name = name
        self.x_length = ur[0] - ll[0]
        self.y_length = ur[1] - ll[1]
        self.x = (ur[0] + ll[0]) / 2
        self.y = (ur[1] + ll[1]) / 2


def check_square_overlap(single_square, square_list):
    """
    Check if a single square overlaps with any squares in a list.
    
    Args:
        single_square: square object with ll and ur attributes
        square_list: list of square objects with ll and ur attributes
    
    Returns:
        bool: True if there is any overlap, False otherwise
    """
    # Extract coordinates from the single square
    x1_min, y1_min = single_square.ll
    x1_max, y1_max = single_square.ur
    
    # Check overlap with each square in the list
    for square in square_list:
        # Extract coordinates from the current square in the list
        x2_min, y2_min = square.ll
        x2_max, y2_max = square.ur
        
        # Check if rectangles overlap
        if (x1_min < x2_max and x1_max > x2_min and 
            y1_min < y2_max and y1_max > y2_min):
            return True  # Found an overlap
    
    return False  # No overlaps found


def get_integer_coordinates_in_square(square):
    """
    Get all integer coordinates within a square object.
    
    Args:
        square: Square object with lower_left and upper_right attributes
    
    Returns:
        list of tuples containing all integer coordinates within the square
    """
    # Extract coordinates from the square object
    lower_left = square.ll
    upper_right = square.ur
    
    x_min, y_min = lower_left
    x_max, y_max = upper_right
    
    # Ensure we have valid coordinates
    if x_min > x_max or y_min > y_max:
        raise ValueError("Lower left coordinates must be less than upper right coordinates")
    
    # Get all integer coordinates within the square
    coordinates = []
    for x in range(int(x_min), int(x_max)):
        for y in range(int(y_min), int(y_max)):
            coordinates.append((x, y))
    
    return coordinates