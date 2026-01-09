# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

# Note: checkout the required tutorials at https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html


# Add the current directory to Python path to find local modules
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


import isaac_sim_utils as isu
import robot_utils
from setups import get_random_setup
import ros2_utils
from robot_logger import RobotLogger
from mission import Mission, MissionType, Waypoint, StatusType

import random
import numpy as np
from typing import List, Dict, Tuple, Optional

from pxr import UsdPhysics, PhysxSchema, Gf, PhysicsSchemaTools, UsdGeom
import omni
import omni.graph.core as og
from omni.isaac.core import SimulationContext
from omni.isaac.core.objects import DynamicCuboid
from omni.physx.scripts import physicsUtils
from omni.isaac.core.objects import VisualCuboid, FixedCuboid
from omni.isaac.core.utils.rotations import quat_to_rot_matrix, rot_matrix_to_quat
import omni.isaac.core.utils.prims as prim_utils

from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.api.robots import Robot
from isaacsim.storage.native import get_assets_root_path

import isaacsim.core.utils.mesh as mesh_utils
import isaacsim.core.utils.stage as stage_utils


# To link this repo with isaac sim:
# cd ~/isaacsim/exts/isaacsim.examples.interactive/isaacsim/examples/interactive
# ln -s /home/${USER}/phd/esi/ user_examples

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

class polygon:
    def __init__(self, coordinates, color, name):
        """
        Args:
            coordinates: List of (x, y) tuples defining the polygon vertices
            color: Color array for the polygon
            name: Name identifier for the polygon
        """
        self.coordinates = coordinates  # List of (x, y) tuples
        self.color = color
        self.name = name

class ESI(BaseSample):

    def register_sim_step_callback(self):
        print("Registering sim step callback")
        self._world.add_physics_callback("sim_step", callback_fn=self.custom_simulation_step)


    def __init__(self) -> None:
        super().__init__()

        # Box dimensions: 2x2m boxes, so circumscribed radius (center to corner) = sqrt(1^2 + 1^2) = sqrt(2) ≈ 1.414m
        self.box_circumscribed_radius = np.sqrt(2.0)  # For 2x2m box

        self.Setup = get_random_setup()


        # Create waypoints
        waypoints = [
            Waypoint(x=10, y=10),
            Waypoint(x=25, y=10),
            Waypoint(x=40, y=10),
            Waypoint(x=10, y=25),
            Waypoint(x=25, y=25),
            Waypoint(x=40, y=25),
            Waypoint(x=10, y=40),
            Waypoint(x=25, y=40),
            Waypoint(x=40, y=40)
        ]

        polygon_color = np.array([1.0, 0.2, 0.2])
        self.occupiable_space_polygons_ = [
            polygon(coordinates=[(14.6, 12), (23, 12), (23, 20.4)], color=polygon_color, name="occupiable_space_polygon_1"),
            polygon(coordinates=[(12, 15), (12, 23), (20, 23)], color=polygon_color, name="occupiable_space_polygon_2"),
            polygon(coordinates=[(35.4, 12), (27, 12), (27, 20.4)], color=polygon_color, name="occupiable_space_polygon_3"),
            polygon(coordinates=[(38, 15), (38, 23), (30, 23)], color=polygon_color, name="occupiable_space_polygon_4"),
            polygon(coordinates=[(14.6, 38), (23, 38), (23, 29.6)], color=polygon_color, name="occupiable_space_polygon_5"),
            polygon(coordinates=[(12, 35), (12, 27), (20, 27)], color=polygon_color, name="occupiable_space_polygon_6"),
            polygon(coordinates=[(35.4, 38), (27, 38), (27, 29.6)], color=polygon_color, name="occupiable_space_polygon_7"),
            polygon(coordinates=[(38, 35), (38, 27), (30, 27)], color=polygon_color, name="occupiable_space_polygon_8"),
            polygon(coordinates=[(0, 0), (8, 0), (8, 50), (0, 50)], color=polygon_color, name="occupiable_space_polygon_9"),
            polygon(coordinates=[(8, 0), (50, 0), (50, 8), (8, 8)], color=polygon_color, name="occupiable_space_polygon_10"),
            polygon(coordinates=[(42, 8), (50, 8), (50, 50), (42, 50)], color=polygon_color, name="occupiable_space_polygon_11"),
            polygon(coordinates=[(8, 42), (42, 42), (42, 50), (8, 50)], color=polygon_color, name="occupiable_space_polygon_12"),
        ]

        self.Setup.set_waypoints(waypoints)

        self._previous_speed = 0.0
        self._previous_angular_velocity_ = 0.0
        self.previous_time_ = 0.0
        self.dt = 0.0
        
        # Map editing variables
        self._free_spaces: List['square'] = []  # List of square objects representing free areas
        self._box_positions: Dict[str, Tuple[Tuple[float, float], 'square']] = {}  # Dictionary mapping prim_name to (position, square) tuples
        self._moved_boxes: set[str] = set()  # Set of box names that have already been moved
        
        # ROS2 publisher wrapper
        self._map_integrity_pub = ros2_utils.MapIntegrityPublisher()

        return

    def _publish_map_integrity_ratio(self):
        """Publish the map integrity ratio (untouched boxes / total boxes)"""
        if not self._box_positions:
            return
        total_boxes = len(self._box_positions)
        untouched_boxes = ros2_utils.compute_untouched_boxes(self._box_positions, self._moved_boxes)
        integrity_ratio = ros2_utils.compute_integrity_ratio(total_boxes, untouched_boxes)
        self._map_integrity_pub.publish_ratio(integrity_ratio)





    def create_waypoint(self, x, y):
        """
        Create a visual waypoint marker and return a Waypoint object
        
        Args:
            x (float): X coordinate of the waypoint
            y (float): Y coordinate of the waypoint
            
        Returns:
            Waypoint: Waypoint object with the given coordinates
        """
        # Create a purple dynamic cube at the waypoint position
        waypoint_name = f"waypoint_{x}_{y}".replace(".", "_")
        waypoint_prim_path = f"/World/waypoints/{waypoint_name}"
        
        # Create a static visual cube as waypoint marker
        try:
            waypoint_cube = self._world.scene.add(
                VisualCuboid(
                    prim_path=waypoint_prim_path,
                    name=waypoint_name,
                    position=np.array([x, y, 0.1]),  # Slightly above ground
                    scale=np.array([0.3, 0.3, 0.1]),  # Flat cube
                    color=np.array([0.5, 0.0, 0.8])  # Purple color
                )
            )
        except Exception as e:
            print(f"Waypoint {waypoint_name} already exists: {e}")
            # Return the Waypoint object anyway so missions can still work
            return Waypoint(x, y)
        
        # Remove collision API from the waypoint cube (it's just a visual marker)
        prim = self._stage.GetPrimAtPath(waypoint_prim_path)
        if prim:
            # Remove any existing collision APIs
            collision_api = UsdPhysics.CollisionAPI.Get(self._stage, waypoint_prim_path)
            if collision_api:
                collision_api.GetPrim().RemoveAPI(UsdPhysics.CollisionAPI)
        
        # Return the Waypoint object
        return Waypoint(x, y)



    def setup_scene(self):
        isu.create_dome_light()
        self._world = self.get_world()
        self._stage = omni.usd.get_context().get_stage()
        isu.add_reference_to_stage(usd_path=self.Setup.map_usd_path, prim_path=f"/map")
        isu.add_reference_to_stage(usd_path=self.Setup.mission_file, prim_path=f"/{self.Setup.robot_prim_name}")
        self._robot = self._world.scene.add(Robot(prim_path=f"/{self.Setup.robot_prim_name}", name=self.Setup.robot_prim_name))

        self._misisons, self._current_mission_number, self._all_missions_completed, self._new_mission, start_position = robot_utils.setup_missions(self.create_waypoint, waypoints=self.Setup.waypoints)
        
        # Move robot to start position (priority: Setup.start_position > mission file start_position > first waypoint > default)
        if self.Setup.start_position and len(self.Setup.start_position) >= 2:
            start_x = float(self.Setup.start_position[0])
            start_y = float(self.Setup.start_position[1])
        elif start_position and len(start_position) >= 2:
            start_x = float(start_position[0])
            start_y = float(start_position[1])
        elif self.Setup.waypoints and len(self.Setup.waypoints) > 0:
            # Use first waypoint as start position
            start_x = float(self.Setup.waypoints[0].x)
            start_y = float(self.Setup.waypoints[0].y)
        else:
            # Default start position
            start_x = 0.0
            start_y = 0.0
        # Move robot to start position
        isu.translate_object(self._stage, f"/{self.Setup.robot_prim_name}", Gf.Vec3f(start_x, start_y, 0.0))
        # isu.rotate_object(self._stage, "/mir_bot_1", -90.0)

        # Move map coordinate system so that (0,0) is at bottom left corner
        isu.translate_object(self._stage, f"/{self.Setup.robot_prim_name}/map", Gf.Vec3f(-start_x, -start_y, 0.0))

        return


    def move_towards_target(self, mission):
        robot_current_position, robot_current_orientation = self._robot.get_world_pose()
        waypoint = mission.get_waypoint()
        
        # Calculate the direction vector from robot to waypoint
        direction_vector = np.array([waypoint.x - robot_current_position[0], 
                                   waypoint.y - robot_current_position[1]])
        
        # Calculate the required yaw angle to face the target
        target_yaw_radians = np.arctan2(direction_vector[1], direction_vector[0])
        
        # Get current robot yaw from quaternion
        current_yaw_radians = robot_utils.quaternion_to_yaw_radians(robot_current_orientation)
        
        # Calculate the angular difference
        yaw_error = target_yaw_radians - current_yaw_radians
        
        # Normalize angle to [-π, π]
        while yaw_error > np.pi:
            yaw_error -= 2 * np.pi
        while yaw_error < -np.pi:
            yaw_error += 2 * np.pi
        
        # Calculate angular velocity based on error (proportional control)
        spin_direction = -1.0 if yaw_error < 0 else 1.0

        if abs(yaw_error - self._previous_angular_velocity_) > 0.01:
            angular_velocity_z = yaw_error
        else:
            angular_velocity_z = self._previous_angular_velocity_ + spin_direction * 0.01

        # Limit angular velocity to prevent overshooting
        max_angular_velocity = 1.5  # rad/s
        angular_velocity_z = np.clip(angular_velocity_z, -max_angular_velocity, max_angular_velocity)
        
        # Set angular velocity for smooth turning
        angular_velocity = np.array([0.0, 0.0, angular_velocity_z])
        self._robot.set_angular_velocity(angular_velocity)

        max_deviation_rad = np.deg2rad(1.0)  # 1 degree in radians
        noisy_angular_velocity = angular_velocity 
        noisy_angular_velocity += np.array([0.0, 0.0, np.random.normal(0.0, max_deviation_rad)]) # about 1 deg
        
        # Calculate forward direction vector based on current orientation
        forward_direction = np.array([np.cos(current_yaw_radians), np.sin(current_yaw_radians)])
        
        # Set movement speed (adjust as needed)
        top_speed = 2.0
        speed = 1.0  # meters per second

        # Convert waypoint to numpy array for distance calculation
        distance = np.linalg.norm(robot_current_position - np.array([waypoint.x, waypoint.y, robot_current_position[2]]))

        if distance < 0.5:
            speed = max(speed * distance/2.5, 0.5)

        if speed - self._previous_speed > 0.005: 
            speed = self._previous_speed + 0.005
        
        if speed > top_speed:
            speed = top_speed
        
        # Calculate linear velocity in world frame
        linear_velocity_world = np.array([forward_direction[0] * speed, 
                                        forward_direction[1] * speed, 
                                        0.0])  # No vertical movement
        
        # Set the robot's linear velocity
        self._robot.set_linear_velocity(linear_velocity_world)

        noisy_linear_velocity_world = linear_velocity_world 
        noisy_linear_velocity_world += np.array([np.random.normal(0.0, 0.05), np.random.normal(0.0, 0.05), 0.0])
        self._previous_speed = speed
        self._previous_angular_velocity_ = angular_velocity_z
        
        return yaw_error



    def step_move_to_waypoint(self, mission, time):
        self.move_towards_target(mission)

        done = robot_utils.check_if_at_waypoint(self._robot.get_world_pose, mission)
        if done:
            mission.set_status(StatusType.SUCCESS)
            print(f"setting mission status to SUCCESS")
            return mission
        

        return mission

    def stop_motion(self):
        self._robot.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
        self._robot.set_angular_velocity(np.array([0.0, 0.0, 0.0]))
        return

    def step_pause(self, mission, time):
        print(f"On pause...")
        return

    def step_mission(self, time):
        if self._new_mission:
            print(f"Starting mission nr {self._current_mission_number} (of {len(self._misisons)}")
            self._new_mission = False

        # Check if we have completed all missions
        if self._current_mission_number >= len(self._misisons):
            if not self._all_missions_completed:
                print(f"All missions completed - stopping mission execution")
                self._all_missions_completed = True
            return
        
        misssion = self._misisons[self._current_mission_number]
        # print(misssion)
        if misssion.get_status() != StatusType.IN_PROGRESS:
            print(f"Error: Mission nr. {self._current_mission_number} is not in progress")
            return
        
        if misssion.get_type() == MissionType.MOVE_TO_WAYPOINT:
            misssion = self.step_move_to_waypoint(misssion, time)
        elif misssion.get_type() == MissionType.PAUSE:
            misssion = self.step_pause(misssion, time)
        else:
            print(f"Error: Mission nr. {self._current_mission_number} is not a valid mission type")
            return

        if misssion.get_status() == StatusType.SUCCESS:
            self.stop_motion()
            print(f"Mission {self._current_mission_number} completed successfully!")
            self._current_mission_number += 1
            self._new_mission = True
            if self._current_mission_number < len(self._misisons):
                self._misisons[self._current_mission_number].set_status(StatusType.IN_PROGRESS)
            else:
                print(f"All missions completed - stopping simulation.")
                # Stop the physics simulation
                self._world.stop()
                return
        if misssion.get_status() == StatusType.FAILED:
            self.stop_motion()
            print(f"Mission {self._current_mission_number} failed!")
            return


    def custom_simulation_step(self, step_size):
        time = self._simulation_context.current_time
        self.dt = self.dt + time - self.previous_time_

        # Debug: Show simulation time occasionally
        if int(time) != int(self.previous_time_):
            print(f"Simulation time: {time:.2f}s, boxes available: {len(self._box_positions)}")
            # Publish map integrity ratio every second
            self._publish_map_integrity_ratio()

        if self.dt > 4:
            self.edit_map()
            self.dt = 0.0

        self.step_mission(time)

        self.previous_time_ = time


    async def setup_post_load(self):
        self._world = self.get_world()
        self._robot = self._world.scene.get_object(self.Setup.robot_prim_name)

        isu.set_camera_view(eye=[50,25,100], target=[50,25,0])
        await isu.disable_gravity(UsdPhysics.Scene.Define(omni.usd.get_context().get_stage(), "/physicsScene"))

        self._simulation_context = SimulationContext()

        self.previous_time_ = self._simulation_context.current_time

        self.register_sim_step_callback()
        return

    async def setup_pre_reset(self):
        print("Pre Reset")
        # Stop robot motion and clear missions
        self.stop_motion()
        self._misisons = []
        self._current_mission_number = 0
        self._all_missions_completed = False
        
        # Reset edit_map state
        self._moved_boxes = set()
        print("Reset edit_map state for clean restart")
        return

    async def setup_post_reset(self):
        self._world = self.get_world()
        # Re-acquire the robot object after reset
        try:
            self._robot = self._world.scene.get_object(self.Setup.robot_prim_name)
        except Exception:
            # If not found, leave it as is; spawn/setup_scene should recreate it
            self._robot = None

        # Recreate the simulation context used by the sim step callback
        self._simulation_context = SimulationContext()

        # Ensure the sim-step callback is registered after a reset so missions get stepped
        try:
            self.register_sim_step_callback()
        except Exception as e:
            print(f"Warning: failed to register sim step callback after reset: {e}")

        self._misisons, self._current_mission_number, self._all_missions_completed, self._new_mission, start_position = robot_utils.setup_missions(self.create_waypoint, waypoints=self.Setup.waypoints)
        
        # Move robot to start position (priority: Setup.start_position > mission file start_position > first waypoint > default)
        if self.Setup.start_position and len(self.Setup.start_position) >= 2:
            start_x = float(self.Setup.start_position[0])
            start_y = float(self.Setup.start_position[1])
        elif start_position and len(start_position) >= 2:
            start_x = float(start_position[0])
            start_y = float(start_position[1])
        elif self.Setup.waypoints and len(self.Setup.waypoints) > 0:
            # Use first waypoint as start position
            start_x = float(self.Setup.waypoints[0].x)
            start_y = float(self.Setup.waypoints[0].y)
        else:
            # Default start position
            start_x = 0.0
            start_y = 0.0
        isu.translate_object(self._stage, f"/{self.Setup.robot_prim_name}", Gf.Vec3f(start_x, start_y, 0.0))
        
        # Move map coordinate system so that (0,0) is at bottom left corner
        isu.translate_object(self._stage, f"/{self.Setup.robot_prim_name}/map", Gf.Vec3f(-start_x, -start_y, 0.0))
        
        print("Post Reset")
        return

    def get_integer_coordinates_in_square(self, square):
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


    async def add_cube_at(self, sq):
        world = self.get_world()

        fancy_cube = world.scene.add(
            FixedCuboid(prim_path=f"/World/cubes/cube_{sq.name}", 
            name=sq.name, 
            position=np.array([sq.x, sq.y, 0.0]),
            scale=np.array([sq.x_length, sq.y_length, 0.01]),
            color=sq.color,
            ))

    async def add_polygon_at(self, poly):
        """
        Add a polygon mesh to the world.
        
        Args:
            poly: polygon instance with coordinates (list of (x, y) tuples), color, and name
        """
        stage = omni.usd.get_context().get_stage()
        prim_path = f"/World/polygons/polygon_{poly.name}"
        
        # Create the mesh prim
        mesh_prim = prim_utils.create_prim(
            prim_path,
            "Mesh",
            position=np.array([0.0, 0.0, 0.0])  # Position will be handled by vertex coordinates
        )
        
        # Get the mesh and set its geometry
        mesh = UsdGeom.Mesh.Get(stage, prim_path)
        
        # Convert coordinates to 3D vertices (z=0 for flat polygon)
        vertices = np.array([
            [coord[0], coord[1], 0.0] for coord in poly.coordinates
        ], dtype=np.float32)
        
        # Set points (vertices)
        points_attr = mesh.CreatePointsAttr()
        points_attr.Set(vertices)
        
        # Triangulate the polygon (fan triangulation from first vertex)
        num_vertices = len(poly.coordinates)
        if num_vertices < 3:
            raise ValueError("Polygon must have at least 3 vertices")
        
        # Create triangles: (0, 1, 2), (0, 2, 3), (0, 3, 4), ...
        face_vertex_indices = []
        for i in range(1, num_vertices - 1):
            face_vertex_indices.extend([0, i, i + 1])
        
        face_vertex_indices = np.array(face_vertex_indices, dtype=np.int32)
        face_vertex_counts = np.array([3] * (num_vertices - 2), dtype=np.int32)
        
        face_vertex_indices_attr = mesh.CreateFaceVertexIndicesAttr()
        face_vertex_indices_attr.Set(face_vertex_indices)
        
        face_vertex_counts_attr = mesh.CreateFaceVertexCountsAttr()
        face_vertex_counts_attr.Set(face_vertex_counts)
        
        # Set color
        color_attr = mesh.CreateDisplayColorAttr()
        color_attr.Set([tuple(poly.color)])

    def is_point_in_polygon(self, point: Tuple[float, float], poly: 'polygon') -> bool:
        """
        Check if a point is inside a polygon using ray casting algorithm.
        
        Args:
            point: (x, y) tuple
            poly: polygon object with coordinates attribute
        
        Returns:
            bool: True if point is inside polygon, False otherwise
        """
        x, y = point
        n = len(poly.coordinates)
        inside = False
        
        p1x, p1y = poly.coordinates[0]
        for i in range(1, n + 1):
            p2x, p2y = poly.coordinates[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def distance_to_polygon_edge(self, point: Tuple[float, float], poly: 'polygon') -> float:
        """
        Calculate the minimum distance from a point to any edge of a polygon.
        
        Args:
            point: (x, y) tuple
            poly: polygon object with coordinates attribute
        
        Returns:
            float: Minimum distance to polygon edge (0 if point is outside polygon)
        """
        px, py = point
        min_distance = float('inf')
        n = len(poly.coordinates)
        
        for i in range(n):
            p1 = poly.coordinates[i]
            p2 = poly.coordinates[(i + 1) % n]
            
            x1, y1 = p1
            x2, y2 = p2
            
            # Vector from p1 to p2
            dx = x2 - x1
            dy = y2 - y1
            
            # Vector from p1 to point
            px1 = px - x1
            py1 = py - y1
            
            # Project point onto line segment
            dot = px1 * dx + py1 * dy
            len_sq = dx * dx + dy * dy
            
            if len_sq == 0:
                # Degenerate edge (p1 == p2)
                dist = np.sqrt(px1 * px1 + py1 * py1)
            else:
                # Parameter t: 0 = at p1, 1 = at p2
                t = max(0, min(1, dot / len_sq))
                
                # Closest point on line segment
                closest_x = x1 + t * dx
                closest_y = y1 + t * dy
                
                # Distance from point to closest point on segment
                dist = np.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)
            
            min_distance = min(min_distance, dist)
        
        return min_distance
    
    def is_point_in_occupiable_space(self, point: Tuple[float, float]) -> bool:
        """
        Check if a point is inside any of the occupiable space polygons.
        
        Args:
            point: (x, y) tuple
        
        Returns:
            bool: True if point is inside any occupiable space polygon, False otherwise
        """
        for poly in self.occupiable_space_polygons_:
            if self.is_point_in_polygon(point, poly):
                return True
        return False
    
    def is_box_fit_in_occupiable_space(self, point: Tuple[float, float], box_circumscribed_radius: float) -> bool:
        """
        Check if a box (with given circumscribed radius) fits within any occupiable space polygon.
        The box fits if the distance from the center to the nearest polygon edge is >= the circumscribed radius.
        
        Args:
            point: (x, y) tuple - center of the box
            box_circumscribed_radius: float - distance from center to corner of the box
        
        Returns:
            bool: True if box fits within any occupiable space polygon, False otherwise
        """
        for poly in self.occupiable_space_polygons_:
            # First check if center is inside the polygon
            if self.is_point_in_polygon(point, poly):
                # Then check if distance to edge is sufficient
                dist_to_edge = self.distance_to_polygon_edge(point, poly)
                if dist_to_edge >= box_circumscribed_radius:
                    return True
        return False

    def check_square_overlap(self, single_square, square_list):
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

    def check_distance_to_boxes(self, position: Tuple[float, float], min_distance: float) -> bool:
        """
        Check if a position is too close to any existing box.
        
        Args:
            position: (x, y) position to check
            min_distance: Minimum distance required to other boxes in meters
            
        Returns:
            True if position is too close to any box, False otherwise
        """
        x, y = position
        min_distance_squared = min_distance * min_distance  # Avoid sqrt for performance
        
        for prim_name, (box_pos, _) in self._box_positions.items():
            box_x, box_y = box_pos
            
            # Calculate squared distance (faster than sqrt)
            dx = x - box_x
            dy = y - box_y
            distance_squared = dx * dx + dy * dy
            
            # If squared distance is less than minimum squared, position is too close
            if distance_squared < min_distance_squared:
                return True
        
        return False

    def get_random_not_free_space(self, free_spaces, seed, min_distance_to_boxes=1.0):
        # Set seed for reproducible "random" placement
        np.random.seed(seed)
        
        do_continue = True
        max_attempts = 500  # Reduced from 1000 to prevent long delays
        attempts = 0
        
        while do_continue and attempts < max_attempts:
            random_space = (np.random.randint(2, 48), np.random.randint(2, 48))
            sq = square([random_space[0] -1, random_space[1] -1], [random_space[0] + 1, random_space[1] + 1], np.array([0.0, 1.0, 0.0]), "1")
        
            # Check if box fits in occupiable space (considering box size and rotation)
            box_fits = self.is_box_fit_in_occupiable_space(random_space, self.box_circumscribed_radius)
            
            # Check overlap with free spaces
            overlaps_free_space = self.check_square_overlap(sq, free_spaces)
            
            # Check distance to other boxes
            too_close_to_box = self.check_distance_to_boxes(random_space, min_distance_to_boxes)
            
            # Box must fit in occupiable space, not overlap free spaces, and not be too close to other boxes
            do_continue = not box_fits or overlaps_free_space or too_close_to_box
            attempts += 1
        
        if attempts >= max_attempts:
            print(f"Warning: Could not find valid position after {max_attempts} attempts, trying systematic search")
            # Try systematic search instead of random
            return self.get_systematic_position(free_spaces, min_distance_to_boxes)
 
        return random_space, sq
    
    def get_systematic_position(self, free_spaces, min_distance_to_boxes=1.0):
        """Systematic search for valid position when random search fails"""
        print(f"Starting systematic search for position with {min_distance_to_boxes}m minimum distance")
        
        # Try positions in a grid pattern
        for x in range(1, 49, 2):  # Step by 2 for efficiency
            for y in range(1, 49, 2):
                position = (x, y)
                sq = square([x-1, y-1], [x+1, y+1], np.array([0.0, 1.0, 0.0]), "systematic")
                
                # Check if box fits in occupiable space (considering box size and rotation)
                box_fits = self.is_box_fit_in_occupiable_space(position, self.box_circumscribed_radius)
                
                # Check overlap with free spaces
                overlaps_free_space = self.check_square_overlap(sq, free_spaces)
                
                # Check distance to other boxes
                too_close_to_box = self.check_distance_to_boxes(position, min_distance_to_boxes)
                
                # Box must fit in occupiable space, not overlap free spaces, and not be too close to other boxes
                if box_fits and not overlaps_free_space and not too_close_to_box:
                    print(f"Found systematic position: {position}")
                    return position, sq
        
        # If systematic search fails, try with reduced distance
        print("Systematic search failed, trying with reduced distance")
        return self.get_systematic_position(free_spaces, min_distance_to_boxes * 0.7)

    async def _on_edit_world_event_async(self):
        from isaacsim.core.utils.stage import get_current_stage
        from isaacsim.core.utils.mesh import get_mesh_vertices_relative_to

        stage = get_current_stage()
        mesh_prim = stage.GetPrimAtPath("/map/WoodenCrate_A1_1/WoodenCrate_A2")
        coord_prim = stage.GetPrimAtPath("/map")

        if not mesh_prim or not mesh_prim.IsValid():
            print("Mesh prim not found: /map/WoodenCrate_A1_1/WoodenCrate_A2")
            return
        if not coord_prim or not coord_prim.IsValid():
            print("Coord prim not found: /map")
            return

        try:
            # Vertices in map frame
            vertices = get_mesh_vertices_relative_to(mesh_prim, coord_prim)
            print(vertices)
        except Exception as e:
            print(f"Error fetching vertices: {e}")


    async def _on_add_objects_event_async(self):

        asset_path = f"/home/{self.Setup.user}/isaac_sim_files/collection/wooden_box_2x2m/wooden_box_2x2m.usd"
        prim_name_1 = "/map/WoodenCrate_A1_1"
        prim_name_2 = "/map/WoodenCrate_A1_2"
        isu.spawn_object(asset_path, prim_name_1)
        isu.spawn_object(asset_path, prim_name_2)
        isu.translate_object(self._stage, prim_name_2, [10.0, 45.0, 0])

        overlap = isu.prims_overlap_obb(prim_name_1, prim_name_2)
        print(f"overlap: {overlap}")


        # Define free space from waypoints
        free_space_data = robot_utils.generate_free_space_from_waypoints(self.Setup.waypoints)
        print(f"Generated {len(free_space_data)} free space rectangles")
        self._free_spaces = []
        self._box_positions = {}
        self._moved_boxes = set()
        
        asset_path = f"/home/{self.Setup.user}/isaac_sim_files/collection/wooden_box_2x2m/wooden_box_2x2m.usd"

        # Create polygons for occupiable spaces
        for p in self.occupiable_space_polygons_:
            await self.add_polygon_at(p)


        # Set fixed seed for reproducible box arrangement
        random.seed(self.Setup.seed_nr)
        
        for i in range(20):
            prim_name = f"/map/WoodenCrate_A1_{i}"
            random_pos, sq = self.get_random_not_free_space(self._free_spaces, seed=self.Setup.seed_nr + i, min_distance_to_boxes=3.0)
            self._free_spaces.append(sq)

            # print("random_pos: ", random_pos)
            isu.spawn_object(asset_path, prim_name)
            isu.translate_object(self._stage, prim_name, Gf.Vec3f(random_pos[0], random_pos[1], 0.0))
            
            # Apply fixed rotation to initial box placement (reproducible)
            initial_rotation = random.uniform(0, 360)
            isu.rotate_object(self._stage, prim_name, initial_rotation)
            
            # Apply collision API to the prim
            prim = self._stage.GetPrimAtPath(prim_name)
            UsdPhysics.CollisionAPI.Apply(prim)
            
            # Store box position and square for later editing
            self._box_positions[prim_name] = (random_pos, sq)
        
        print(f"Created {len(self._box_positions)} boxes for edit_map functionality (reproducible arrangement)")

        return

    def edit_map(self) -> None:
        """
        Edit the map by moving a random box to a new location
        """
        # Check if we have any boxes to move
        if not self._box_positions:
            print(f"edit_map: No boxes available to move (box_positions is empty)")
            return
        
        # Get boxes that haven't been moved yet
        unmoved_boxes = [name for name in self._box_positions.keys() if name not in self._moved_boxes]
        
        # If all boxes have been moved, reset the moved boxes set and continue
        if not unmoved_boxes:
            return
            print("All boxes have been moved, resetting moved boxes list")
            self._moved_boxes = set()
            unmoved_boxes = list(self._box_positions.keys())
        
        # Pick a random box from unmoved boxes
        prim_name = random.choice(unmoved_boxes)
        old_pos, old_sq = self._box_positions[prim_name]
        
        print(f"Moving box {prim_name} from position {old_pos}")
        
        # Remove the old position from free spaces
        # if old_sq in self._free_spaces:
        #     self._free_spaces.remove(old_sq)
        
        # Find a new random position outside the free area with minimum distance to other boxes
        
        # Get current simulation time for deterministic seeds
        current_time = isu.get_sim_time(self._simulation_context)
        
        # Use deterministic seed based on box name and time for reproducible positioning
        #position_seed = hash(prim_name + str(int(current_time))) % 10000
        new_pos, new_sq = self.get_random_not_free_space(self._free_spaces, seed=self.Setup.seed_nr, min_distance_to_boxes=3.0)
        
        # Generate reproducible rotation based on box name and time
        # This ensures the same box gets the same rotation every time
        #rotation_seed = hash(prim_name + str(int(current_time))) % 1000
        random.seed(self.Setup.seed_nr)
        random_rotation = random.uniform(0, 360)
        
        # Move the box to the new position
        isu.translate_object(self._stage, prim_name, Gf.Vec3f(new_pos[0], new_pos[1], 0.0))
        
        # Apply random rotation to the box
        isu.rotate_object(self._stage, prim_name, random_rotation)
        
        # Add the new position to free spaces
        self._free_spaces.append(new_sq)
        
        # Update the box positions dictionary
        self._box_positions[prim_name] = (new_pos, new_sq)
        
        # Mark this box as moved
        self._moved_boxes.add(prim_name)
        
        # print(f"Moved box {prim_name} to new position {new_pos} with rotation {random_rotation:.1f}°")
        print(f"Remaining unmoved boxes: {len(self._box_positions) - len(self._moved_boxes)}")
        
        # Publish updated map integrity ratio
        self._publish_map_integrity_ratio()

        
