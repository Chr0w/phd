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
import math
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


import isaac_sim_utils as isu
import robot_utils
from setups import get_random_setup, get_setup_from_name
import ros2_utils
from mission import Mission, MissionType, Waypoint, StatusType
from polygons import polygon, is_box_fit_in_occupiable_space, add_polygon_at, try_position_in_polygon

import random
import numpy as np
from typing import List, Dict, Tuple, Optional

try:
    import yaml
except ImportError:
    yaml = None

from pxr import UsdPhysics, Gf, UsdGeom
import omni
from omni.isaac.core import SimulationContext
from omni.isaac.core.objects import VisualCuboid
from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.core.api.robots import Robot


# To link this repo with isaac sim:
# cd ~/isaacsim/exts/isaacsim.examples.interactive/isaacsim/examples/interactive
# ln -s /home/${USER}/phd/seg_map/ user_examples

class SegMap(BaseSample):

    def register_sim_step_callback(self):
        print("Registering sim step callback")
        self._world.add_physics_callback("sim_step", callback_fn=self.custom_simulation_step)

    def _pick_random_waypoint_path(self, waypoints: List[Waypoint], start_position: Tuple[float, float], n: int) -> List[Waypoint]:
        """
        Pick n random waypoints starting from the closest to start_position.
        Each subsequent waypoint must be connected to the previous one.
        
        Args:
            waypoints: List of all available waypoints
            start_position: (x, y) starting position
            n: Number of waypoints to pick
            
        Returns:
            List of selected waypoints in order
        """
        if not waypoints or n <= 0:
            return []
        
        # Create a dictionary for quick lookup by number
        waypoint_dict = {wp.nr: wp for wp in waypoints}
        
        # Find the waypoint closest to start position
        start_x, start_y = start_position
        closest_waypoint = min(waypoints, key=lambda wp: np.sqrt((wp.x - start_x)**2 + (wp.y - start_y)**2))
        
        selected = [closest_waypoint]
        current_waypoint = closest_waypoint
        
        # Pick n-1 more waypoints, each connected to the previous
        for _ in range(n - 1):
            # Get all connected waypoints
            connected_nrs = current_waypoint.connected_numbers
            available_connected = [
                waypoint_dict[nr] for nr in connected_nrs 
                if nr in waypoint_dict
            ]
            
            # Exclude the previous waypoint to avoid going back and forth
            if len(selected) > 1:
                previous_waypoint = selected[-2]  # Get the waypoint before current
                available_connected = [
                    wp for wp in available_connected 
                    if wp != previous_waypoint
                ]
            
            if not available_connected:
                # No connected waypoints available, break
                break
            
            # Randomly pick one of the connected waypoints (excluding previous)
            next_waypoint = np.random.choice(available_connected)
            selected.append(next_waypoint)
            current_waypoint = next_waypoint
        
        return selected

    def _generate_waypoint_path(self):
        """
        Generate waypoint path based on current start position.
        Reads start position fresh and regenerates waypoints.
        """
        # Get starting position - crash if not provided
        if not self.Setup.start_position or len(self.Setup.start_position) < 2:
            raise ValueError("start_position must be provided in Setup. No fallbacks allowed.")
        start_pos = (float(self.Setup.start_position[0]), float(self.Setup.start_position[1]))
        
        # Pick random waypoint path (adjust n as needed)
        selected_waypoints = self._pick_random_waypoint_path(self.waypoints, start_pos, n=30)
        
        print([wp.nr for wp in selected_waypoints])
        self.Setup.set_waypoints(selected_waypoints)

    def __init__(self) -> None:
        super().__init__()

        # Box dimensions: 2x2m boxes, so circumscribed radius (center to corner) = sqrt(1^2 + 1^2) = sqrt(2) ≈ 1.414m
        self.box_circumscribed_radius = np.sqrt(2.0)  # For 2x2m box

        # Read start position and seed from YAML file
        start_position, seed_nr = robot_utils.read_start_info_from_yaml()
        # Get setup by name
        self.Setup = get_setup_from_name("setup_1")
        # Set the start position and seed on the setup
        self.Setup._start_position = start_position
        self.Setup._seed_nr = seed_nr

        # Create waypoints
        self.waypoints = [
            Waypoint(x=10, y=10, nr=1, connected_numbers=[2, 4, 5]),
            Waypoint(x=25, y=10, nr=2, connected_numbers=[1, 3, 5]),
            Waypoint(x=40, y=10, nr=3, connected_numbers=[2, 5, 6]),
            Waypoint(x=10, y=25, nr=4, connected_numbers=[1, 5, 7]),
            Waypoint(x=25, y=25, nr=5, connected_numbers=[1, 2, 3, 4, 6, 7, 8, 9]),
            Waypoint(x=40, y=25, nr=6, connected_numbers=[3, 5, 9]),
            Waypoint(x=10, y=40, nr=7, connected_numbers=[4, 5, 8]),
            Waypoint(x=25, y=40, nr=8, connected_numbers=[5, 7, 9]),
            Waypoint(x=40, y=40, nr=9, connected_numbers=[5, 6, 8]),

        ]

        polygon_color = np.array([0.8, 0.6, 0.4])
        self.occupiable_space_polygons_ = [
            polygon(coordinates=[(14.6, 12), (23, 12), (23, 20.4)], color=polygon_color, name="occupiable_space_polygon_1"),
            polygon(coordinates=[(12, 15), (12, 23), (20, 23)], color=polygon_color, name="occupiable_space_polygon_2"),
            polygon(coordinates=[(35.4, 12), (27, 12), (27, 20.4)], color=polygon_color, name="occupiable_space_polygon_3"),
            polygon(coordinates=[(38, 15), (38, 23), (30, 23)], color=polygon_color, name="occupiable_space_polygon_4"),
            polygon(coordinates=[(14.6, 38), (23, 38), (23, 29.6)], color=polygon_color, name="occupiable_space_polygon_5"),
            polygon(coordinates=[(12, 35), (12, 27), (20, 27)], color=polygon_color, name="occupiable_space_polygon_6"),
            polygon(coordinates=[(35.4, 38), (27, 38), (27, 29.6)], color=polygon_color, name="occupiable_space_polygon_7"),
            polygon(coordinates=[(38, 35), (38, 27), (30, 27)], color=polygon_color, name="occupiable_space_polygon_8"),
            polygon(coordinates=[(0, 0), (8, 0), (8, 25), (0, 25)], color=polygon_color, name="occupiable_space_polygon_9"),
            polygon(coordinates=[(0, 25), (8, 25), (8, 50), (0, 50)], color=polygon_color, name="occupiable_space_polygon_10"),
            polygon(coordinates=[(8, 0), (29, 0), (29, 8), (8, 8)], color=polygon_color, name="occupiable_space_polygon_11"),
            polygon(coordinates=[(29, 0), (50, 0), (50, 8), (29, 8)], color=polygon_color, name="occupiable_space_polygon_12"),
            polygon(coordinates=[(42, 8), (50, 8), (50, 29), (42, 29)], color=polygon_color, name="occupiable_space_polygon_13"),
            polygon(coordinates=[(42, 29), (50, 29), (50, 50), (42, 50)], color=polygon_color, name="occupiable_space_polygon_14"),
            polygon(coordinates=[(8, 42), (25, 42), (25, 50), (8, 50)], color=polygon_color, name="occupiable_space_polygon_15"),
            polygon(coordinates=[(25, 42), (42, 42), (42, 50), (25, 50)], color=polygon_color, name="occupiable_space_polygon_16"),
        ]

        # Generate waypoint path based on current start position
        self._generate_waypoint_path()

        self._previous_speed = 0.0
        self._previous_angular_velocity_ = 0.0
        self.previous_time_ = 0.0
        self.dt = 0.0
        
        # Map editing variables
        self._box_positions: Dict[str, Tuple[float, float]] = {}
        self._original_box_positions: Dict[str, Tuple[float, float]] = {}  # Store original positions
        self._original_box_rotations: Dict[str, float] = {}  # Store original rotations
        self._moved_boxes: set[str] = set()
        self._returned_boxes: set[str] = set()  # Track boxes that have been returned to original position
        
        # ROS2 publisher wrapper
        self._map_integrity_pub = ros2_utils.MapIntegrityPublisher()

    def _publish_map_integrity_ratio(self):
        """Publish the map integrity ratio (untouched boxes / total boxes)"""
        if not self._box_positions:
            return
        total_boxes = len(self._box_positions)
        untouched_boxes = ros2_utils.compute_untouched_boxes(self._box_positions, self._moved_boxes)
        integrity_ratio = ros2_utils.compute_integrity_ratio(total_boxes, untouched_boxes)
        self._map_integrity_pub.publish_ratio(integrity_ratio)
        # print(f"Map integrity ratio: {integrity_ratio}")

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
                    position=np.array([x, y, 0.0]),  # Slightly above ground
                    scale=np.array([0.3, 0.3, 0.01]),  # Flat cube
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
        
        return Waypoint(x, y)

    def _setup_missions_and_robot_position(self):
        """
        Common setup code for missions and robot positioning.
        Sets up missions and moves robot to start position.
        """
        self._misisons, self._current_mission_number, self._all_missions_completed, self._new_mission, start_position = robot_utils.setup_missions(self.create_waypoint, waypoints=self.Setup.waypoints)
        
        # Move robot to start position - crash if not provided
        if not self.Setup.start_position or len(self.Setup.start_position) < 2:
            raise ValueError("start_position must be provided in Setup. No fallbacks allowed.")
        start_x = float(self.Setup.start_position[0])
        start_y = float(self.Setup.start_position[1])
        # Move robot to start position
        isu.translate_object(self._stage, f"/{self.Setup.robot_prim_name}", Gf.Vec3f(start_x, start_y, 0.0))
        # isu.rotate_object(self._stage, "/mir_bot_1", -90.0)

        # Move map coordinate system so that (0,0) is at bottom left corner
        isu.translate_object(self._stage, f"/{self.Setup.robot_prim_name}/map", Gf.Vec3f(-start_x, -start_y, 0.0))

    def setup_scene(self):
        isu.create_dome_light()
        self._world = self.get_world()
        self._stage = omni.usd.get_context().get_stage()
        isu.add_reference_to_stage(usd_path=self.Setup.map_usd_path, prim_path=f"/map")
        isu.add_reference_to_stage(usd_path=self.Setup.mission_file, prim_path=f"/{self.Setup.robot_prim_name}")
        self._robot = self._world.scene.add(Robot(prim_path=f"/{self.Setup.robot_prim_name}", name=self.Setup.robot_prim_name))

        # Re-read start position and seed from YAML file (in case it changed) and regenerate waypoints
        start_position, seed_nr = robot_utils.read_start_info_from_yaml()
        self.Setup = get_setup_from_name("setup_1")
        self.Setup._start_position = start_position
        self.Setup._seed_nr = seed_nr
        self._generate_waypoint_path()

        self._setup_missions_and_robot_position()

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
        self._previous_speed = speed
        self._previous_angular_velocity_ = angular_velocity_z
        
        return yaw_error

    def step_move_to_waypoint(self, mission, time):
        self.move_towards_target(mission)

        done = robot_utils.check_if_at_waypoint(self._robot.get_world_pose, mission)
        if done:
            mission.set_status(StatusType.SUCCESS)
            print(f"Setting mission status to SUCCESS")
            return mission
        return mission

    def stop_motion(self):
        self._robot.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
        self._robot.set_angular_velocity(np.array([0.0, 0.0, 0.0]))

    def step_pause(self, mission, time):
        print(f"On pause...")

    def step_mission(self, time):
        if self._new_mission:
            print(f"Starting mission nr {self._current_mission_number} (of {len(self._misisons)})")
            self._new_mission = False

        # Check if we have completed all missions
        if self._current_mission_number >= len(self._misisons):
            if not self._all_missions_completed:
                print(f"All missions completed - stopping mission execution")
                self._all_missions_completed = True
            return
        
        misssion = self._misisons[self._current_mission_number]
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
                self._world.stop()
                return
        if misssion.get_status() == StatusType.FAILED:
            self.stop_motion()
            print(f"Mission {self._current_mission_number} failed!")

    def custom_simulation_step(self, step_size):
        time = self._simulation_context.current_time
        self.dt = self.dt + time - self.previous_time_

        # Debug: Show simulation time occasionally
        if int(time) != int(self.previous_time_):
            print(f"Simulation time: {time:.2f}s")
            # Publish map integrity ratio every second
            self._publish_map_integrity_ratio()

        # if self.dt > 4:
        #     self.edit_map()
        #     self.dt = 0.0

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

    async def setup_pre_reset(self):
        print("Pre Reset")
        self.stop_motion()
        self._misisons = []
        self._current_mission_number = 0
        self._all_missions_completed = False
        self._moved_boxes = set()
        self._returned_boxes = set()
        self._original_box_rotations = {}
        print("Reset edit_map state for clean restart")

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

        # Re-read start position and seed from YAML file (in case it changed) and regenerate waypoints
        start_position, seed_nr = robot_utils.read_start_info_from_yaml()
        self.Setup = get_setup_from_name("setup_1")
        self.Setup._start_position = start_position
        self.Setup._seed_nr = seed_nr
        self._generate_waypoint_path()

        # Ensure the sim-step callback is registered after a reset so missions get stepped
        try:
            self.register_sim_step_callback()
        except Exception as e:
            print(f"Warning: failed to register sim step callback after reset: {e}")

        self._setup_missions_and_robot_position()
        print("Post Reset")

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
        min_distance_squared = min_distance * min_distance
        
        for prim_name, box_pos in self._box_positions.items():
            box_x, box_y = box_pos
            
            dx = x - box_x
            dy = y - box_y
            distance_squared = dx * dx + dy * dy
            
            if distance_squared < min_distance_squared:
                return True
        
        return False

    def get_random_not_free_space(self, seed, min_distance_to_boxes):
        # Create a new random state for this box to ensure reproducibility
        rng = np.random.RandomState(seed)
        max_attempts = 500
        attempts = 0
        
        while attempts < max_attempts:
            random_space = (rng.randint(2, 48), rng.randint(2, 48))
        
            box_fits = is_box_fit_in_occupiable_space(random_space, self.box_circumscribed_radius, self.occupiable_space_polygons_)
            too_close_to_box = self.check_distance_to_boxes(random_space, min_distance_to_boxes)
            
            if box_fits and not too_close_to_box:
                return random_space
            attempts += 1
        
        print(f"Warning: Could not find valid position after {max_attempts} attempts, trying systematic search")
        return self.get_systematic_position(min_distance_to_boxes)
    
    def get_systematic_position(self, min_distance_to_boxes):
        """Systematic search for valid position when random search fails"""
        print(f"Starting systematic search for position with {min_distance_to_boxes}m minimum distance")
        
        for x in range(1, 49, 2):
            for y in range(1, 49, 2):
                position = (x, y)
                
                box_fits = is_box_fit_in_occupiable_space(position, self.box_circumscribed_radius, self.occupiable_space_polygons_)
                too_close_to_box = self.check_distance_to_boxes(position, min_distance_to_boxes)
                
                if box_fits and not too_close_to_box:
                    print(f"Found systematic position: {position}")
                    return position
        
        print("Systematic search failed, trying with reduced distance")
        return self.get_systematic_position(min_distance_to_boxes * 0.7)

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
        
        self._box_positions = {}
        self._original_box_positions = {}
        self._original_box_rotations = {}
        self._moved_boxes = set()
        self._returned_boxes = set()

        # Create polygons for occupiable spaces
        # for p in self.occupiable_space_polygons_:
        #     await add_polygon_at(p, self._stage)

        # Set fixed seed for reproducible box arrangement
        np.random.seed(self.Setup.seed_nr)
        random.seed(self.Setup.seed_nr)
        
        for i in range(20):
            prim_name = f"/map/WoodenCrate_A1_{i}"
            # Use a combined seed that includes both seed_nr and box index for reproducibility
            box_seed = self.Setup.seed_nr * 1000 + i
            np.random.seed(box_seed)
            random_pos = self.get_random_not_free_space(seed=box_seed, min_distance_to_boxes=2 * self.box_circumscribed_radius)

            isu.spawn_object(asset_path, prim_name)
            isu.translate_object(self._stage, prim_name, Gf.Vec3f(random_pos[0], random_pos[1], 0.0))
            
            # Apply fixed rotation to initial box placement (reproducible)
            # Use numpy random for consistency with position generation
            np.random.seed(box_seed + 10000)  # Offset to get different random sequence for rotation
            initial_rotation = np.random.uniform(0, 360)
            isu.rotate_object(self._stage, prim_name, initial_rotation)
            
            # Apply collision API to the prim
            prim = self._stage.GetPrimAtPath(prim_name)
            UsdPhysics.CollisionAPI.Apply(prim)
            
            # Store box position and rotation for later editing
            self._box_positions[prim_name] = random_pos
            self._original_box_positions[prim_name] = random_pos  # Store original position
            self._original_box_rotations[prim_name] = initial_rotation  # Store original rotation
        
        print(f"Created {len(self._box_positions)} boxes for edit_map functionality (reproducible arrangement)")

    def edit_map(self) -> None:
        """
        Edit the map by moving a random box to a new location.
        First picks a random occupiable polygon, then tries positions within it.
        When all boxes have been moved, move them back to original positions one by one.
        """
        if not self._box_positions:
            print(f"edit_map: No boxes available to move (box_positions is empty)")
            return
        
        unmoved_boxes = [name for name in self._box_positions.keys() if name not in self._moved_boxes]
        
        # If we've started returning boxes, continue until all are returned
        # OR if all boxes have been moved, start moving them back to original positions
        if self._returned_boxes or not unmoved_boxes:
            # Get boxes that haven't been returned yet (use original_box_positions as source of truth)
            unreturned_boxes = [name for name in self._original_box_positions.keys() if name not in self._returned_boxes]
            if not unreturned_boxes:
                # All boxes have been returned, stop simulation (handled after last box is moved)
                return
            
            # Pick a random box to return to original position
            prim_name = random.choice(unreturned_boxes)
            original_pos = self._original_box_positions[prim_name]
            original_rotation = self._original_box_rotations[prim_name]
            current_pos = self._box_positions[prim_name]
            print(f"Returning box {prim_name} from position {current_pos} to original position {original_pos}")
            
            # Move box back to original position and rotation
            isu.translate_object(self._stage, prim_name, Gf.Vec3f(original_pos[0], original_pos[1], 0.0))
            isu.rotate_object(self._stage, prim_name, original_rotation)
            
            # Update position and mark as returned
            self._box_positions[prim_name] = original_pos
            self._returned_boxes.add(prim_name)
            # Remove from moved_boxes since it's back in original position (counts as untouched again)
            self._moved_boxes.discard(prim_name)
            remaining = len(self._original_box_positions) - len(self._returned_boxes)
            print(f"Remaining boxes to return: {remaining}")
            self._publish_map_integrity_ratio()
            
            # If all boxes have been returned, stop the simulation
            if remaining == 0:
                print("All boxes returned to original positions - stopping simulation.")
                self._world.stop()
            return
        
        # Normal flow: move a box to a new position
        prim_name = random.choice(unmoved_boxes)
        old_pos = self._box_positions[prim_name]
        print(f"Moving box {prim_name} from position {old_pos}")

        # Try to find a new position by randomly selecting polygons
        new_pos = None
        max_attempts = 100
        
        # Track only the last 3 polygons used to avoid immediately repeating them
        recent_polygons = []
        
        while new_pos is None:
            # Get polygons that aren't in the last 3 used
            available_polygons = [p for p in self.occupiable_space_polygons_ if p not in recent_polygons]
            
            # If all polygons are in recent list, use all polygons (shouldn't happen with 16+ polygons)
            if not available_polygons:
                available_polygons = self.occupiable_space_polygons_
            
            # Randomly pick a polygon from available ones (using numpy for better randomness)
            poly = np.random.choice(available_polygons)
            
            # Add to recent list and keep only last 3
            recent_polygons.append(poly)
            if len(recent_polygons) > math.floor(len(self.occupiable_space_polygons_)*0.2):
                recent_polygons.pop(0)
            
            new_pos = try_position_in_polygon(
                poly, 
                self.box_circumscribed_radius,
                2 * self.box_circumscribed_radius,  # Need 2x radius to prevent overlap
                self.check_distance_to_boxes,
                max_attempts=max_attempts
            )
            
            if new_pos is None:
                max_attempts *= 2
                print(f"No valid position found in polygon {poly.name}, increasing attempts to {max_attempts} and trying another polygon")
        
        # Generate random rotation (use box name for variation, but still somewhat random)
        # Use hash of box name + current time to get different rotation per box
        rotation_seed = hash(prim_name) % 10000
        np.random.seed(rotation_seed)
        random_rotation = np.random.uniform(0, 360)
        
        # Move the box to the new position
        isu.translate_object(self._stage, prim_name, Gf.Vec3f(new_pos[0], new_pos[1], 0.0))
        
        # Apply random rotation to the box
        isu.rotate_object(self._stage, prim_name, random_rotation)
        
        # Update the box positions dictionary
        self._box_positions[prim_name] = new_pos
        self._moved_boxes.add(prim_name)
        print(f"Remaining unmoved boxes: {len(self._box_positions) - len(self._moved_boxes)}")
        self._publish_map_integrity_ratio()
