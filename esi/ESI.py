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

from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.api.robots import Robot
from isaacsim.storage.native import get_assets_root_path



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

class ESI(BaseSample):

    def register_sim_step_callback(self):
        print("Registering sim step callback")
        self._world.add_physics_callback("sim_step", callback_fn=self.custom_simulation_step)


    def __init__(self) -> None:
        super().__init__()

        self._USER = os.environ.get("USER")
        self._import_robot_usd_path = f"/home/{self._USER}/isaac_sim_files/float_bot_3.usd"
        self._import_map_usd_path = f"/home/{self._USER}/isaac_sim_files/map_2_for_import.usd"
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
                    scale=np.array([0.5, 0.5, 0.1]),  # Flat cube
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
        isu.add_reference_to_stage(usd_path=self._import_map_usd_path, prim_path=f"/map")
        isu.add_reference_to_stage(usd_path=self._import_robot_usd_path, prim_path=f"/float_bot")
        self._robot = self._world.scene.add(Robot(prim_path="/float_bot", name="float_bot"))

        start_x = 25.0
        start_y = 25.0
        # Move robot to start position
        isu.translate_object(self._stage, "/float_bot", Gf.Vec3f(start_x, start_y, 0.0))
        # isu.rotate_object(self._stage, "/float_bot", -90.0)

        # Move map coordinate system so that (0,0) is at bottom left corner
        isu.translate_object(self._stage, "/float_bot/map", Gf.Vec3f(-start_x, -start_y, 0.0))

        self._misisons, self._current_mission_number, self._all_missions_completed, self._new_mission = robot_utils.setup_missions(self.create_waypoint)        

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
        
        # Calculate forward direction vector based on current orientation
        forward_direction = np.array([np.cos(current_yaw_radians), np.sin(current_yaw_radians)])
        
        # Set movement speed (adjust as needed)
        top_speed = 1.0
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
        self._robot = self._world.scene.get_object("float_bot")

        isu.set_camera_view([25,25,50],[25,25,0])
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
            self._robot = self._world.scene.get_object("float_bot")
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

        self._misisons, self._current_mission_number, self._all_missions_completed, self._new_mission = robot_utils.setup_missions(self.create_waypoint)
        print("Post Reset")
        return

    def world_cleanup(self):
        # Cleanup ROS2 resources
        if hasattr(self, '_map_integrity_pub') and self._map_integrity_pub is not None:
            self._map_integrity_pub.shutdown()
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

    def get_random_not_free_space(self, free_spaces, seed=42, min_distance_to_boxes=1.0):
        # Set seed for reproducible "random" placement
        np.random.seed(seed)
        
        do_continue = True
        max_attempts = 500  # Reduced from 1000 to prevent long delays
        attempts = 0
        
        while do_continue and attempts < max_attempts:
            random_space = (np.random.randint(2, 48), np.random.randint(2, 48))
            sq = square([random_space[0] -1, random_space[1] -1], [random_space[0] + 1, random_space[1] + 1], np.array([0.0, 1.0, 0.0]), "1")
        
            # Check overlap with free spaces
            overlaps_free_space = self.check_square_overlap(sq, free_spaces)
            
            # Check distance to other boxes
            too_close_to_box = self.check_distance_to_boxes(random_space, min_distance_to_boxes)
            
            do_continue = overlaps_free_space or too_close_to_box
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
                
                # Check overlap with free spaces
                overlaps_free_space = self.check_square_overlap(sq, free_spaces)
                
                # Check distance to other boxes
                too_close_to_box = self.check_distance_to_boxes(position, min_distance_to_boxes)
                
                if not overlaps_free_space and not too_close_to_box:
                    print(f"Found systematic position: {position}")
                    return position, sq
        
        # If systematic search fails, try with reduced distance
        print("Systematic search failed, trying with reduced distance")
        return self.get_systematic_position(free_spaces, min_distance_to_boxes * 0.7)

    async def _on_edit_world_event_async(self):
        print("yea")


    async def _on_add_objects_event_async(self):

        # Define free space
        square_1 = square([15,15], [35,35], np.array([0.0, 1.0, 0]), "1")
        self._free_spaces = []
        self._box_positions = {}
        self._moved_boxes = set()
        
        asset_path = f"/home/{self._USER}/isaac_sim_files/collection/wooden_box_2x2m/wooden_box_2x2m.usd"

        for i in range(1,2):
            await self.add_cube_at(eval(f"square_{i}"))
            self._free_spaces.append(eval(f"square_{i}"))

        # Set fixed seed for reproducible box arrangement
        random.seed(42)
        
        for i in range(20):
            prim_name = f"/map/WoodenCrate_A1_{i}"
            random_pos, sq = self.get_random_not_free_space(self._free_spaces, seed=42 + i, min_distance_to_boxes=3.0)
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
        position_seed = hash(prim_name + str(int(current_time))) % 10000
        new_pos, new_sq = self.get_random_not_free_space(self._free_spaces, seed=position_seed, min_distance_to_boxes=3.0)
        
        # Generate reproducible rotation based on box name and time
        # This ensures the same box gets the same rotation every time
        rotation_seed = hash(prim_name + str(int(current_time))) % 1000
        random.seed(rotation_seed)
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

        
