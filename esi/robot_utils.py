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


def setup_missions(create_waypoint_func, missions_file_path=None, waypoints=None):
    """
    Load missions from YAML file or use provided waypoints and create Mission objects.
    
    Args:
        create_waypoint_func: Function to create waypoint objects
        missions_file_path: Path to the missions YAML file (optional, ignored if waypoints provided)
        waypoints: List of Waypoint objects (optional, takes precedence over missions_file_path)
    
    Returns:
        tuple: (missions_list, current_mission_number, all_missions_completed, new_mission, start_position)
    """
    from mission import Mission, MissionType, StatusType, Waypoint
    
    waypoint_list = []
    start_position = []
    
    # If waypoints are provided directly, use them
    if waypoints is not None:
        waypoint_list = waypoints
    else:
        # Otherwise, load from YAML file
        if missions_file_path is None:
            missions_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "missions", "mission_1.yaml")
        else:
            missions_file = missions_file_path
            
        try:
            if yaml is not None:
                with open(missions_file, "r") as f:
                    data = yaml.safe_load(f)
                    waypoint_list = data.get("waypoints", []) if data else []
                    # Read start_position from YAML if present
                    if data and "start_position" in data:
                        start_position = data["start_position"]
        except FileNotFoundError:
            print(f"Missions file not found: {missions_file}")
        except Exception as e:
            print(f"Failed to load missions file {missions_file}: {e}")

    missions = []
    for i, wp in enumerate(waypoint_list):
        try:
            # Handle both Waypoint objects and coordinate lists
            if isinstance(wp, Waypoint):
                wx, wy = wp.x, wp.y
            else:
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


def generate_free_space_from_waypoints(waypoints, width=2.0, color=None):
    """
    Generate free space rectangles connecting adjacent waypoints in a 3x3 grid.
    
    Creates individual 2m wide rectangles (paths) connecting each waypoint to its 
    adjacent waypoints (horizontally, vertically, and diagonally) in a 3x3 grid layout.
    Each connection is a separate rectangle, creating paths between waypoints.
    
    Args:
        waypoints: List of Waypoint objects (must be 9 waypoints in 3x3 grid order)
        width: Width of the free space rectangles in meters (default: 2.0)
        color: Color array for squares (optional, defaults to green)
    
    Returns:
        list: List of dictionaries with square data: {'ll': [x, y], 'ur': [x, y], 'color': array, 'name': str}
    """
    from mission import Waypoint
    
    if len(waypoints) != 9:
        raise ValueError(f"Expected 9 waypoints for a 3x3 grid, got {len(waypoints)}")
    
    if color is None:
        color = np.array([0.0, 1.0, 0.0])  # Green
    
    # Organize waypoints into 3x3 grid
    # Assuming waypoints are ordered: row by row, left to right
    grid = [[None for _ in range(3)] for _ in range(3)]
    for idx, wp in enumerate(waypoints):
        row = idx // 3
        col = idx % 3
        grid[row][col] = wp
    
    squares = []
    square_count = 0
    processed_pairs = set()  # Track which pairs we've already processed
    
    # For each waypoint, create rectangles to adjacent waypoints
    for row in range(3):
        for col in range(3):
            wp1 = grid[row][col]
            if wp1 is None:
                continue
            
            # Check all 8 adjacent positions (horizontal, vertical, diagonal)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue  # Skip self
                    
                    new_row = row + dr
                    new_col = col + dc
                    
                    # Check if adjacent position is valid
                    if 0 <= new_row < 3 and 0 <= new_col < 3:
                        wp2 = grid[new_row][new_col]
                        if wp2 is None:
                            continue
                        
                        # Create a unique identifier for this pair to avoid duplicates
                        pair_id = tuple(sorted([(row, col), (new_row, new_col)]))
                        
                        # Only create rectangle once per pair
                        if pair_id not in processed_pairs:
                            processed_pairs.add(pair_id)
                            
                            # Create rectangle from wp1 to wp2
                            # Calculate direction vector
                            dx = wp2.x - wp1.x
                            dy = wp2.y - wp1.y
                            length = np.sqrt(dx**2 + dy**2)
                            
                            if length == 0:
                                continue
                            
                            # Normalize direction
                            dir_x = dx / length
                            dir_y = dy / length
                            
                            # Perpendicular vector (for width) - normalized
                            perp_x = -dir_y
                            perp_y = dir_x
                            
                            # Half width
                            half_width = width / 2.0
                            
                            # Shrink the path slightly from each waypoint to reduce overlap
                            # This creates distinct paths rather than one big blob
                            shrink_factor = 0.3  # Shrink by 30% from each end
                            start_offset = length * shrink_factor
                            end_offset = length * shrink_factor
                            
                            # Calculate start and end points (shrunk from waypoints)
                            start_x = wp1.x + dir_x * start_offset
                            start_y = wp1.y + dir_y * start_offset
                            end_x = wp2.x - dir_x * end_offset
                            end_y = wp2.y - dir_y * end_offset
                            
                            # Calculate the 4 corners of the rectangle
                            # The rectangle extends from start to end, with width perpendicular to the line
                            
                            # Corner 1: Start point + perpendicular offset
                            c1_x = start_x + perp_x * half_width
                            c1_y = start_y + perp_y * half_width
                            
                            # Corner 2: Start point - perpendicular offset
                            c2_x = start_x - perp_x * half_width
                            c2_y = start_y - perp_y * half_width
                            
                            # Corner 3: End point - perpendicular offset
                            c3_x = end_x - perp_x * half_width
                            c3_y = end_y - perp_y * half_width
                            
                            # Corner 4: End point + perpendicular offset
                            c4_x = end_x + perp_x * half_width
                            c4_y = end_y + perp_y * half_width
                            
                            # Find bounding box (ll and ur) for this specific rectangle
                            x_coords = [c1_x, c2_x, c3_x, c4_x]
                            y_coords = [c1_y, c2_y, c3_y, c4_y]
                            
                            ll = [min(x_coords), min(y_coords)]
                            ur = [max(x_coords), max(y_coords)]
                            
                            square_count += 1
                            squares.append({
                                'll': ll,
                                'ur': ur,
                                'color': color,
                                'name': f"free_space_{square_count}"
                            })
    
    return squares


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