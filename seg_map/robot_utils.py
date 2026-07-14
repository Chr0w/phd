import glob
import heapq
import math
import os
import random

import numpy as np
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
        yaml_path: Path to the YAML file containing start position (defaults to /home/{USER}/devcontainer/ros2_ws/shared_files/start_pos.yaml)
    
    Returns:
        tuple: (start_position, seed_nr) where start_position is [x, y] or [x, y, yaw_degrees] and seed_nr is int
    """
    if yaml_path is None:
        user = os.environ.get("USER")
        yaml_path = f"/home/{user}/devcontainer/ros2_ws/shared_files/start_pos.yaml"
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


def _waypoint_id_sort_key(waypoint_id: str) -> int:
    suffix = waypoint_id.removeprefix("waypoint_")
    return int(suffix) if suffix.isdigit() else 0


def load_layout_config(layout_path: str) -> dict:
    if yaml is None:
        raise ImportError("yaml module is required to read layout config. Install it with: pip install pyyaml")
    if not os.path.isfile(layout_path):
        print(f"Layout file not found: {layout_path}")
        return {}
    with open(layout_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def layout_waypoint_entries(layout: dict) -> list[tuple[str, float, float]]:
    """Return sorted (waypoint_id, x_m, y_m) entries from a layout config."""
    entries: list[tuple[str, float, float, int]] = []
    for waypoint_data in layout.get("waypoints", []):
        waypoint_id = str(waypoint_data.get("id", ""))
        if not waypoint_id:
            continue

        position = waypoint_data.get("position")
        if not position or len(position) < 2:
            continue

        entries.append(
            (
                waypoint_id,
                float(position[0]),
                float(position[1]),
                _waypoint_id_sort_key(waypoint_id),
            )
        )

    entries.sort(key=lambda item: item[3])
    return [(waypoint_id, x, y) for waypoint_id, x, y, _sort_key in entries]


NUM_WAYPOINT_PLANS = 300
DEFAULT_START_WAYPOINT_ID = "waypoint_1"


def waypoint_plans_path() -> str:
    plans_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plans")
    return os.path.join(plans_dir, "waypoint_plans.yaml")


def bin_assets_dir(user: str | None = None, size: str = "large") -> str:
    if user is None:
        user = os.environ.get("USER", "unknown")
    if size not in {"large", "small"}:
        raise ValueError(f"Unsupported bin asset size: {size}")
    return f"/home/{user}/isaac_sim_files/assets/{size}"


def list_bin_assets(user: str | None = None, size: str = "large") -> list[str]:
    return sorted(glob.glob(os.path.join(bin_assets_dir(user, size), "*.usd")))


def layout_section_ids(layout: dict) -> list[str]:
    return [str(section.get("id", f"section_{index}")) for index, section in enumerate(layout.get("sections", []), start=1)]


def section_bins(layout: dict, section_id: str, size: str) -> list[dict]:
    for section in layout.get("sections", []):
        if str(section.get("id")) != section_id:
            continue

        bins: list[dict] = []
        for bin_data in section.get("bins", []):
            if str(bin_data.get("size", "")) != size:
                continue

            upper_left = bin_data.get("upper_left_m")
            lower_right = bin_data.get("lower_right_m")
            if not upper_left or not lower_right:
                continue

            bins.append(
                {
                    "number": int(bin_data.get("number", len(bins) + 1)),
                    "size": size,
                    "center_x": (float(upper_left[0]) + float(lower_right[0])) / 2.0,
                    "center_y": (float(upper_left[1]) + float(lower_right[1])) / 2.0,
                }
            )
        bins.sort(key=lambda item: item["number"])
        return bins

    return []


def layout_waypoint_positions(layout: dict) -> dict[str, tuple[float, float]]:
    return {
        waypoint_id: (x, y)
        for waypoint_id, x, y in layout_waypoint_entries(layout)
    }


def build_waypoint_graph(layout: dict) -> dict[str, list[tuple[str, float]]]:
    positions = layout_waypoint_positions(layout)
    graph: dict[str, list[tuple[str, float]]] = {waypoint_id: [] for waypoint_id in positions}

    for connection in layout.get("connections", []):
        start_id = str(connection.get("from", ""))
        end_id = str(connection.get("to", ""))
        if start_id not in positions or end_id not in positions:
            continue

        start_x, start_y = positions[start_id]
        end_x, end_y = positions[end_id]
        distance = math.hypot(end_x - start_x, end_y - start_y)
        graph[start_id].append((end_id, distance))
        graph[end_id].append((start_id, distance))

    return graph


def shortest_waypoint_path(
    graph: dict[str, list[tuple[str, float]]],
    start_id: str,
    goal_id: str,
) -> list[str] | None:
    if start_id == goal_id:
        return [start_id]
    if start_id not in graph or goal_id not in graph:
        return None

    distances = {start_id: 0.0}
    previous: dict[str, str] = {}
    heap: list[tuple[float, str]] = [(0.0, start_id)]
    visited: set[str] = set()

    while heap:
        distance, node_id = heapq.heappop(heap)
        if node_id in visited:
            continue
        visited.add(node_id)
        if node_id == goal_id:
            break

        for neighbor_id, edge_cost in graph.get(node_id, []):
            next_distance = distance + edge_cost
            if neighbor_id not in distances or next_distance < distances[neighbor_id]:
                distances[neighbor_id] = next_distance
                previous[neighbor_id] = node_id
                heapq.heappush(heap, (next_distance, neighbor_id))

    if goal_id not in previous and goal_id != start_id:
        return None

    path = [goal_id]
    while path[-1] != start_id:
        path.append(previous[path[-1]])
    path.reverse()
    return path


def generate_waypoint_plans(
    layout: dict,
    num_plans: int = NUM_WAYPOINT_PLANS,
    seed: int = 1,
    start_waypoint_id: str = DEFAULT_START_WAYPOINT_ID,
) -> list[dict]:
    graph = build_waypoint_graph(layout)
    positions = layout_waypoint_positions(layout)
    if not positions:
        return []

    waypoint_ids = list(positions.keys())
    if start_waypoint_id not in positions:
        start_waypoint_id = sorted(waypoint_ids, key=_waypoint_id_sort_key)[0]

    rng = random.Random(seed)
    current_waypoint_id = start_waypoint_id
    plans: list[dict] = []

    for plan_number in range(1, num_plans + 1):
        candidates = [waypoint_id for waypoint_id in waypoint_ids if waypoint_id != current_waypoint_id]
        rng.shuffle(candidates)

        target_waypoint_id = None
        path = None
        for candidate_id in candidates:
            candidate_path = shortest_waypoint_path(graph, current_waypoint_id, candidate_id)
            if candidate_path:
                target_waypoint_id = candidate_id
                path = candidate_path
                break

        if target_waypoint_id is None or path is None:
            raise RuntimeError(f"No reachable waypoint from {current_waypoint_id} for plan {plan_number}")

        plans.append(
            {
                "plan": plan_number,
                "start": current_waypoint_id,
                "target": target_waypoint_id,
                "path": path,
            }
        )
        current_waypoint_id = target_waypoint_id

    return plans


def save_waypoint_plans(plans: list[dict], output_path: str, seed: int) -> None:
    if yaml is None:
        raise ImportError("yaml module is required to save waypoint plans. Install it with: pip install pyyaml")

    payload = {
        "num_plans": len(plans),
        "seed": seed,
        "plans": plans,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def setup_missions_from_plans(
    plans: list[dict],
    positions: dict[str, tuple[float, float]],
):
    """
    Create move-to-waypoint missions from generated plans.

    Each plan contributes one mission per intermediate waypoint on its path.
    """
    from mission import Mission, MissionType, StatusType, Waypoint

    missions = []
    mission_number = 0
    for plan in plans:
        path = plan.get("path", [])
        for waypoint_id in path[1:]:
            x, y = positions[waypoint_id]
            mission = Mission(
                mission_number,
                MissionType.MOVE_TO_WAYPOINT,
                Waypoint(x, y),
                plan_number=int(plan["plan"]),
                waypoint_id=waypoint_id,
                plan_start=str(plan["start"]),
                plan_target=str(plan["target"]),
                plan_path=[str(node_id) for node_id in path],
            )
            missions.append(mission)
            mission_number += 1

    if missions:
        missions[0].set_status(StatusType.IN_PROGRESS)
    current_mission_number = 0
    all_missions_completed = False
    new_mission = True

    return missions, current_mission_number, all_missions_completed, new_mission

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