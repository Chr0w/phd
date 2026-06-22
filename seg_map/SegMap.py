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
from setups import get_setup_from_name
import ros2_utils
from mission import Mission, MissionType, Waypoint, StatusType
from polygons import polygon, is_box_fit_in_occupiable_space, add_polygon_at, try_position_in_polygon

import random
import numpy as np
from typing import Dict, Tuple, Optional

from pxr import Usd, UsdPhysics, Gf, UsdGeom, Sdf
import omni
import omni.kit.app
from isaacsim.examples.base.base_sample_experimental import BaseSample
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.simulation_manager import SimulationManager


# To link this repo with isaac sim:
# cd ~/isaacsim/exts/isaacsim.examples.interactive/isaacsim/examples/interactive/user_examples
# ln -sfn /home/${USER}/phd/seg_map seg_map

class SegMap(BaseSample):

    def register_sim_step_callback(self):
        print("Registering sim step callback")
        self.deregister_sim_step_callback()
        self._sim_step_callback_subscription = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(
                self._on_sim_step_event,
                name="seg_map_sim_step",
            )
        )

    def deregister_sim_step_callback(self):
        self._sim_step_callback_subscription = None

    def _on_sim_step_event(self, event):
        try:
            self.custom_simulation_step(event.payload.get("dt", 0.0))
        except Exception as exc:
            print(f"Disabling seg_map sim callback after error: {exc}")
            self.deregister_sim_step_callback()
            self._missions_started = False

    def start_missions(self):
        self._ensure_robot_asset()
        self._missions_started = True
        self._mission_loop_logged = False
        self.previous_time_ = SimulationManager.get_simulation_time()
        self.register_sim_step_callback()
        app_utils.play()

    def _start_missions_from_timeline(self, time):
        self._ensure_robot_asset()
        self._missions_started = True
        self._mission_loop_logged = False
        self.previous_time_ = time
        print("Mission loop armed from timeline play")

    def _ensure_robot_ready(self):
        self._ensure_robot_asset()
        return self._robot_prim().IsValid()

    def _robot_world_pose(self):
        prim = self._robot_prim()
        transform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
        transform.Orthonormalize()
        position = np.array(transform.ExtractTranslation(), dtype=np.float32)
        rotation = transform.ExtractRotationQuat()
        orientation = np.array([rotation.GetReal(), *rotation.GetImaginary()], dtype=np.float32)
        return position, orientation

    def _robot_prim(self):
        robot_path = f"/{self.Setup.robot_prim_name}"
        try:
            for prim in self._stage.Traverse():
                if prim.GetPath().pathString == robot_path:
                    return prim
        except Exception:
            return Usd.Prim()
        return Usd.Prim()

    def _set_robot_world_pose(self, position=None, orientation=None):
        robot_prim_path = f"/{self.Setup.robot_prim_name}"
        if position is not None:
            self._set_prim_translation(robot_prim_path, position)
        if orientation is not None:
            yaw_degrees = np.degrees(robot_utils.quaternion_to_yaw_radians(orientation))
            self._set_prim_yaw(robot_prim_path, yaw_degrees)

    def _set_prim_translation(self, prim_path, translation):
        prim = self._prim_by_path(prim_path)
        if not prim or not prim.IsValid():
            return
        xformable = UsdGeom.Xformable(prim)
        translate_op = self._get_or_add_xform_op(xformable, UsdGeom.XformOp.TypeTranslate, xformable.AddTranslateOp)
        translate_op.Set(Gf.Vec3d(*np.asarray(translation, dtype=float)))

    def _set_prim_yaw(self, prim_path, yaw_degrees):
        prim = self._prim_by_path(prim_path)
        if not prim or not prim.IsValid():
            return
        xformable = UsdGeom.Xformable(prim)
        orient_op = self._get_or_add_xform_op(xformable, UsdGeom.XformOp.TypeOrient, xformable.AddOrientOp)
        orient_op.Set(robot_utils.yaw_degrees_to_quaternion(yaw_degrees))

    def _get_or_add_xform_op(self, xformable, op_type, add_op_fn):
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == op_type:
                return op
        return add_op_fn()

    def _prim_by_path(self, prim_path):
        prim_path = str(prim_path)
        try:
            for prim in self._stage.Traverse():
                if prim.GetPath().pathString == prim_path:
                    return prim
        except Exception:
            return Usd.Prim()
        return Usd.Prim()

    def _set_robot_velocity(self, linear=None, angular=None):
        position, orientation = self._robot_world_pose()
        yaw_radians = robot_utils.quaternion_to_yaw_radians(orientation)
        step_size = max(0.0, min(float(getattr(self, "_current_step_size", 1.0 / 60.0)), 0.1))
        if angular is not None:
            yaw_radians += float(np.asarray(angular)[2]) * step_size
        if linear is not None:
            position = position + np.asarray(linear, dtype=np.float32) * step_size
        half_yaw = yaw_radians / 2.0
        self._set_robot_world_pose(
            position=position,
            orientation=np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)]),
        )

    def __init__(self) -> None:
        super().__init__()


        self.manual_control = False
        # Box dimensions: 2x2m boxes, so circumscribed radius (center to corner) = sqrt(1^2 + 1^2) = sqrt(2) ≈ 1.414m
        self.box_circumscribed_radius = np.sqrt(2.0)  # For 2x2m box

        self._selected_setup_name = "setup_2"
        self._load_setup_from_yaml()

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

        self._previous_speed = 0.0
        self._previous_angular_velocity_ = 0.0
        self.previous_time_ = 0.0
        self.dt = 0.0
        self._current_step_size = 1.0 / 60.0
        self._next_waypoint_number = 1
        self._sim_step_callback_subscription = None
        self._missions_started = False
        self._mission_loop_logged = False
        
        # Map editing variables
        self._box_positions: Dict[str, Tuple[float, float]] = {}
        self._original_box_positions: Dict[str, Tuple[float, float]] = {}  # Store original positions
        self._original_box_rotations: Dict[str, float] = {}  # Store original rotations
        self._moved_boxes: set[str] = set()
        self._returned_boxes: set[str] = set()  # Track boxes that have been returned to original position
        
        # ROS2 publisher wrapper
        self._map_integrity_pub = ros2_utils.MapIntegrityPublisher()
        self._teleop_sub = ros2_utils.TeleopCommandSubscriber('/cmd_vel')
        self._teleop_warned_unavailable = False

    def _load_setup_from_yaml(self):
        """Load the selected setup and apply start position/seed from YAML."""
        start_position, seed_nr = robot_utils.read_start_info_from_yaml()
        self.Setup = get_setup_from_name(self._selected_setup_name)
        if self.Setup is None:
            raise ValueError(f"{self._selected_setup_name} not found in setups")
        self.Setup._start_position = start_position
        self.Setup._seed_nr = seed_nr

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
        waypoint_name = f"wp_{self._next_waypoint_number:02d}"
        self._next_waypoint_number += 1
        waypoint_prim_path = f"/map/waypoints/{waypoint_name}"
        
        # Create a static visual cube as waypoint marker
        try:
            waypoint_cube = UsdGeom.Cube.Define(self._stage, waypoint_prim_path)
            waypoint_cube.CreateSizeAttr(1.0)
            waypoint_cube.CreateDisplayColorAttr([Gf.Vec3f(0.5, 0.0, 0.8)])
            waypoint_xform = UsdGeom.Xformable(waypoint_cube.GetPrim())
            waypoint_xform.AddTranslateOp().Set(Gf.Vec3f(x, y, 0.0))
            waypoint_xform.AddScaleOp().Set(Gf.Vec3f(0.3, 0.3, 0.01))
            print(f"Created waypoint {waypoint_name} at {x}, {y}")
        except Exception as e:
            print(f"Failed to create waypoint marker {waypoint_name}: {e}")
            # Return the Waypoint object anyway so missions can still work
            return Waypoint(x, y)
        
        # Remove collision API from the waypoint cube (it's just a visual marker)
        prim = self._stage.GetPrimAtPath(Sdf.Path(waypoint_prim_path))
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
        self._next_waypoint_number = 1
        self._misisons, self._current_mission_number, self._all_missions_completed, self._new_mission, _ = robot_utils.setup_missions(
            self.create_waypoint, missions_file_path=self.Setup.missions_yaml_path
        )

    def _position_robot_asset(self):
        # Move robot to start position - crash if not provided
        if not self.Setup.start_position or len(self.Setup.start_position) < 2:
            raise ValueError("start_position must be provided in Setup. No fallbacks allowed.")
        start_x = float(self.Setup.start_position[0])
        start_y = float(self.Setup.start_position[1])
        start_yaw = robot_utils.get_start_yaw_degrees(self.Setup.start_position)
        robot_prim_path = f"/{self.Setup.robot_prim_name}"

        self._set_prim_translation(robot_prim_path, (start_x, start_y, 0.0))
        self._set_prim_yaw(robot_prim_path, start_yaw)

        # Move map coordinate system so that (0,0) is at bottom left corner
        self._set_prim_translation(f"/{self.Setup.robot_prim_name}/map", (-start_x, -start_y, 0.0))

    def _ensure_robot_asset(self):
        robot_prim_path = f"/{self.Setup.robot_prim_name}"
        robot_prim = self._robot_prim()
        if not robot_prim or not robot_prim.IsValid():
            isu.add_reference_to_stage(usd_path=self.Setup.mission_file, prim_path=robot_prim_path)
            self._position_robot_asset()

    def setup_scene(self):
        self.deregister_sim_step_callback()
        self._missions_started = False
        isu.create_dome_light()
        self._stage = omni.usd.get_context().get_stage()
        isu.add_reference_to_stage(usd_path=self.Setup.map_usd_path, prim_path=f"/map")
        self._robot = None

        self._load_setup_from_yaml()
        self._setup_missions_and_robot_position()

    def move_towards_target(self, mission):
        # return
        robot_current_position, robot_current_orientation = self._robot_world_pose()
        waypoint = mission.get_waypoint()
        
        # Calculate the direction vector from robot to waypoint
        direction_vector = np.array([waypoint.x - robot_current_position[0], 
                                   waypoint.y - robot_current_position[1]])
        
        # Calculate the required yaw angle to face the target
        target_yaw_radians = np.arctan2(direction_vector[1], direction_vector[0])
        
        # Get current robot yaw from quaternion
        current_yaw_radians = robot_utils.quaternion_to_yaw_radians(robot_current_orientation)

        self._enforce_planar_orientation()
        
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
        self._set_robot_velocity(angular=angular_velocity)
        

        # print(f"Current yaw: {current_yaw_radians}, Target yaw: {target_yaw_radians}, Yaw error: {yaw_error}")
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
        self._set_robot_velocity(linear=linear_velocity_world)
        self._previous_speed = speed
        self._previous_angular_velocity_ = angular_velocity_z
        
        return yaw_error

    def step_move_to_waypoint(self, mission, time):
        self.move_towards_target(mission)

        done = robot_utils.check_if_at_waypoint(self._robot_world_pose, mission)
        if done:
            mission.set_status(StatusType.SUCCESS)
            print(f"Setting mission status to SUCCESS")
            return mission
        return mission

    def stop_motion(self):
        self._previous_speed = 0.0
        self._previous_angular_velocity_ = 0.0

    def _enforce_planar_orientation(self):
        robot_current_position, robot_current_orientation = self._robot_world_pose()
        current_yaw_radians = robot_utils.quaternion_to_yaw_radians(robot_current_orientation)
        half_yaw = current_yaw_radians / 2.0
        planar_orientation = np.array([
            np.cos(half_yaw),  # w
            0.0,               # x (roll)
            0.0,               # y (pitch)
            np.sin(half_yaw),  # z (yaw)
        ])
        self._set_robot_world_pose(
            position=robot_current_position,
            orientation=planar_orientation,
        )

    def _step_manual_control(self):
        if not self._teleop_sub.ok:
            if not self._teleop_warned_unavailable:
                print("manual_control is enabled but ROS2 /cmd_vel subscriber is not available.")
                self._teleop_warned_unavailable = True
            self.stop_motion()
            return

        self._teleop_sub.spin_once()
        linear_x, linear_y, angular_z = self._teleop_sub.get_latest_command(stale_timeout_sec=0.5)

        robot_current_position, robot_current_orientation = self._robot_world_pose()
        current_yaw_radians = robot_utils.quaternion_to_yaw_radians(robot_current_orientation)

        # Convert base-frame cmd_vel to world-frame velocity, then apply directly.
        cos_yaw = np.cos(current_yaw_radians)
        sin_yaw = np.sin(current_yaw_radians)
        linear_velocity_world = np.array([
            linear_x * cos_yaw - linear_y * sin_yaw,
            linear_x * sin_yaw + linear_y * cos_yaw,
            0.0,
        ])
        self._set_robot_velocity(linear=linear_velocity_world, angular=np.array([0.0, 0.0, angular_z]))

        self._previous_speed = float(np.linalg.norm(linear_velocity_world[:2]))
        self._previous_angular_velocity_ = float(angular_z)

    def step_pause(self, mission, time):
        print(f"On pause...")

    def step_mission(self, time):
        #return
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
                app_utils.stop()
                return
        if misssion.get_status() == StatusType.FAILED:
            self.stop_motion()
            print(f"Mission {self._current_mission_number} failed!")

    def custom_simulation_step(self, step_size):
        if not app_utils.is_playing():
            return
        current_stage = omni.usd.get_context().get_stage()
        if current_stage is None or getattr(self, "_stage", None) is not current_stage:
            self.deregister_sim_step_callback()
            self._missions_started = False
            return
        time = SimulationManager.get_simulation_time()
        if not self._missions_started:
            self._start_missions_from_timeline(time)
        if not self._ensure_robot_ready():
            return

        self._current_step_size = step_size or max(0.0, time - self.previous_time_)
        if not self._mission_loop_logged:
            print(f"Mission loop active, step_size={self._current_step_size:.4f}")
            self._mission_loop_logged = True
        self.dt = self.dt + time - self.previous_time_

        # Debug: Show simulation time occasionally
        if int(time) != int(self.previous_time_):
            print(f"Simulation time: {time:.2f}s")
            # Publish map integrity ratio every second
            self._publish_map_integrity_ratio()

        # if self.dt > 4:
        #     self.edit_map()
        #     self.dt = 0.0

        self._enforce_planar_orientation()

        if self.manual_control:
            self._step_manual_control()
        else:
            self.step_mission(time)
        self.previous_time_ = time

    async def setup_post_load(self):
        self._robot = None

        app_utils.stop()
        await omni.kit.app.get_app().next_update_async()
        self._ensure_robot_asset()

        isu.set_camera_view(eye=[-45,-75,55], target=[-12,-12,0])
        await isu.disable_gravity(self._stage)

        self.previous_time_ = SimulationManager.get_simulation_time()
        self.register_sim_step_callback()
        app_utils.stop()

    async def setup_pre_reset(self):
        print("Pre Reset")
        self.deregister_sim_step_callback()
        self.stop_motion()
        self._missions_started = False
        self._mission_loop_logged = False
        self._misisons = []
        self._current_mission_number = 0
        self._all_missions_completed = False
        self._moved_boxes = set()
        self._returned_boxes = set()
        self._original_box_rotations = {}
        print("Reset edit_map state for clean restart")

    async def setup_post_reset(self):
        self._robot = None

        self._load_setup_from_yaml()

        self._setup_missions_and_robot_position()
        app_utils.stop()
        await omni.kit.app.get_app().next_update_async()
        self._ensure_robot_asset()
        self.register_sim_step_callback()
        app_utils.stop()
        print("Post Reset")

    async def setup_post_clear(self):
        self.physics_cleanup()

    def physics_cleanup(self):
        self.deregister_sim_step_callback()
        self._missions_started = False
        self._mission_loop_logged = False

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
        import isaacsim.core.experimental.utils.stage as stage_utils

        stage = stage_utils.get_current_stage(backend="usd")
        mesh_prim = stage.GetPrimAtPath(Sdf.Path("/map/WoodenCrate_A1_1/WoodenCrate_A2"))
        coord_prim = stage.GetPrimAtPath(Sdf.Path("/map"))

        if not mesh_prim or not mesh_prim.IsValid():
            print("Mesh prim not found: /map/WoodenCrate_A1_1/WoodenCrate_A2")
            return
        if not coord_prim or not coord_prim.IsValid():
            print("Coord prim not found: /map")
            return

        try:
            # Vertices in map frame
            mesh_points = UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get()
            xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            mesh_to_world = xform_cache.GetLocalToWorldTransform(mesh_prim)
            world_to_coord = xform_cache.GetLocalToWorldTransform(coord_prim).GetInverse()
            vertices = np.array([
                world_to_coord.Transform(mesh_to_world.Transform(point))
                for point in mesh_points
            ])
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
        
        for i in range(10):
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
            prim = self._stage.GetPrimAtPath(Sdf.Path(prim_name))
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
                app_utils.stop()
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
