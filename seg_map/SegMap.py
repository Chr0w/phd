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
from setups import get_setup_from_name
import ros2_utils
from layout_development import LayoutDevelopmentController, BinAssetManager, get_mode_config
from mission import MissionType, StatusType

import numpy as np
from typing import Dict, Optional

from pxr import Usd, UsdPhysics, Gf, UsdGeom, Sdf
import omni
import omni.kit.app
from isaacsim.examples.base.base_sample_experimental import BaseSample
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.simulation_manager import SimulationManager

_SHOW_BIN_VISUALIZATION = False

_BIN_SIZE_M = {"small": 0.5, "medium": 1.5, "large": 4.0}
_BIN_HEIGHT_M = 0.02
_BIN_COLOR = Gf.Vec3f(0.0, 0.8, 0.0)
_BIN_OPACITY = 0.5


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
        self._start_layout_development(self.previous_time_)
        self.register_sim_step_callback()
        app_utils.play()

    def _start_missions_from_timeline(self, time):
        self._ensure_robot_asset()
        self._missions_started = True
        self._mission_loop_logged = False
        self.previous_time_ = time
        self._start_layout_development(time)
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
        self._selected_setup_name = "setup_1"
        self._load_setup_from_yaml()

        self._previous_speed = 0.0
        self._previous_angular_velocity_ = 0.0
        self.previous_time_ = 0.0
        self.dt = 0.0
        self._current_step_size = 1.0 / 60.0
        self._layout_config = {}
        self._waypoint_plans = []
        self._plan_count = 0
        self._last_printed_plan_number = None
        self._sim_step_callback_subscription = None
        self._missions_started = False
        self._mission_loop_logged = False
        self._layout_dev: Optional[LayoutDevelopmentController] = None
        self._layout_dev_started = False
        self._manual_asset_manager: Optional[BinAssetManager] = None
        self._section_bin_marker_paths: Dict[str, list[str]] = {}
        
        # ROS2 publisher wrapper
        self._storage_utilization_pub = ros2_utils.StorageUtilizationPublisher()
        self._sim_progress_pub = ros2_utils.SimProgressPublisher()
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

    def _publish_storage_utilization(self):
        """Publish storage utilization (occupied bins / total bins)."""
        if not self._layout_dev:
            return
        ratio = ros2_utils.compute_storage_utilization(
            self._layout_dev.occupied_count,
            self._layout_dev.total_bins,
        )
        self._storage_utilization_pub.publish_ratio(ratio)

    def _publish_sim_progress(self, sim_time: float):
        """Publish test progress (total time, percent complete, minutes passed/left)."""
        if not self._layout_dev:
            return
        progress = self._layout_dev.test_progress(sim_time)
        if progress is None:
            return
        self._sim_progress_pub.publish_progress(
            progress.total_test_time_minutes,
            progress.percentage_complete,
            progress.minutes_passed,
            progress.minutes_left,
            progress.estimated_real_minutes_to_completion,
        )

    def _ensure_asset_manager(self) -> BinAssetManager:
        if self._layout_dev:
            return self._layout_dev.asset_manager
        if not self._layout_config:
            self._load_layout_config()
        if self._manual_asset_manager is None:
            self._manual_asset_manager = BinAssetManager(
                self._stage,
                self.Setup.user,
                int(self.Setup.seed_nr),
            )
            self._manual_asset_manager.initialize(self._layout_config)
        return self._manual_asset_manager

    def _configure_visual_cube(self, prim_path, translate, scale, color, opacity=None):
        cube = UsdGeom.Cube.Define(self._stage, prim_path)
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([color])
        if opacity is not None:
            cube.CreateDisplayOpacityAttr([opacity])

        xform = UsdGeom.Xformable(cube.GetPrim())
        translate_op = self._get_or_add_xform_op(
            xform, UsdGeom.XformOp.TypeTranslate, xform.AddTranslateOp
        )
        translate_op.Set(Gf.Vec3f(*translate))
        scale_op = self._get_or_add_xform_op(
            xform, UsdGeom.XformOp.TypeScale, xform.AddScaleOp
        )
        scale_op.Set(Gf.Vec3f(*scale))

        prim = self._stage.GetPrimAtPath(Sdf.Path(prim_path))
        if prim:
            collision_api = UsdPhysics.CollisionAPI.Get(self._stage, prim_path)
            if collision_api:
                collision_api.GetPrim().RemoveAPI(UsdPhysics.CollisionAPI)

    def _layout_yaml_path(self):
        user = os.environ.get("USER", self.Setup.user)
        return f"/home/{user}/devcontainer/ros2_ws/shared_files/layout.yaml"

    def _load_layout_config(self):
        self._layout_config = robot_utils.load_layout_config(self._layout_yaml_path())
        return self._layout_config

    def _init_layout_development(self):
        mode_name = self.Setup.layout_development_mode
        if not mode_name:
            self._layout_dev = None
            return
        config = get_mode_config(mode_name)
        self._layout_dev = LayoutDevelopmentController(
            self._stage,
            self._layout_config,
            config,
            self.Setup.user,
            int(self.Setup.seed_nr),
        )
        self._layout_dev_started = False

    def _start_layout_development(self, sim_time: float):
        if not self._layout_dev or self._layout_dev_started:
            return
        self._layout_dev.start(sim_time)
        self._layout_dev_started = True

    def _init_asset_spawn_state(self):
        self._section_bin_marker_paths = {}

    def _stop_simulation(self):
        app_utils.stop()

    def _spawn_layout_waypoint(self, waypoint_id, x, y):
        prim_path = f"/map/waypoints/{waypoint_id}"
        self._configure_visual_cube(
            prim_path,
            translate=(x, y, 0.0),
            scale=(0.3, 0.3, 0.01),
            color=Gf.Vec3f(0.5, 0.0, 0.8),
        )

    def _spawn_layout_waypoints(self, layout=None):
        layout = layout if layout is not None else self._layout_config
        waypoint_entries = robot_utils.layout_waypoint_entries(layout)
        for waypoint_id, x, y in waypoint_entries:
            self._spawn_layout_waypoint(waypoint_id, x, y)
        print(f"Spawned {len(waypoint_entries)} layout waypoint marker(s)")

    def _spawn_bin_cube(self, prim_path, center_x, center_y, size_m):
        self._configure_visual_cube(
            prim_path,
            translate=(center_x, center_y, _BIN_HEIGHT_M / 2.0),
            scale=(size_m, size_m, _BIN_HEIGHT_M),
            color=_BIN_COLOR,
            opacity=_BIN_OPACITY,
        )

    def _spawn_layout_bins(self, layout=None):
        if not _SHOW_BIN_VISUALIZATION:
            return

        layout = layout if layout is not None else self._layout_config
        if not layout:
            return

        spawned = 0
        for section in layout.get("sections", []):
            section_id = str(section.get("id", "section"))
            UsdGeom.Xform.Define(self._stage, f"/map/bins/{section_id}")
            section_bin_paths: list[str] = []
            for bin_data in section.get("bins", []):
                size_name = str(bin_data.get("size", "small"))
                size_m = _BIN_SIZE_M.get(size_name)
                if size_m is None:
                    print(f"Unknown bin size '{size_name}' in section {section_id}, skipping")
                    continue

                upper_left = bin_data.get("upper_left_m")
                lower_right = bin_data.get("lower_right_m")
                if not upper_left or not lower_right:
                    print(f"Bin {bin_data.get('number')} in {section_id} missing coordinates, skipping")
                    continue

                center_x = (float(upper_left[0]) + float(lower_right[0])) / 2.0
                center_y = (float(upper_left[1]) + float(lower_right[1])) / 2.0
                bin_number = int(bin_data.get("number", spawned + 1))
                prim_path = f"/map/bins/{section_id}/bin_{bin_number:03d}"
                self._spawn_bin_cube(prim_path, center_x, center_y, size_m)
                section_bin_paths.append(prim_path)
                spawned += 1

            if section_bin_paths:
                self._section_bin_marker_paths[section_id] = section_bin_paths

        print(f"Spawned {spawned} layout bin marker(s) from {self._layout_yaml_path()}")

    def _setup_missions_and_robot_position(self):
        """
        Common setup code for missions and robot positioning.
        Generates 300 random shortest-path plans and creates missions from them.
        """
        self._load_layout_config()
        self._spawn_layout_waypoints()
        self._init_layout_development()

        seed = int(self.Setup.seed_nr)
        self._waypoint_plans = robot_utils.generate_waypoint_plans(
            self._layout_config,
            num_plans=robot_utils.NUM_WAYPOINT_PLANS,
            seed=seed,
            start_waypoint_id=robot_utils.DEFAULT_START_WAYPOINT_ID,
        )
        self._plan_count = len(self._waypoint_plans)

        plans_path = robot_utils.waypoint_plans_path()
        robot_utils.save_waypoint_plans(self._waypoint_plans, plans_path, seed)
        print(f"Saved {self._plan_count} waypoint plans to {plans_path}")

        positions = robot_utils.layout_waypoint_positions(self._layout_config)
        self._misisons, self._current_mission_number, self._all_missions_completed, self._new_mission = (
            robot_utils.setup_missions_from_plans(self._waypoint_plans, positions)
        )
        self._last_printed_plan_number = None

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
        self._layout_dev = None
        self._layout_dev_started = False
        self._manual_asset_manager = None
        isu.create_dome_light()
        self._stage = omni.usd.get_context().get_stage()
        isu.add_reference_to_stage(usd_path=self.Setup.map_usd_path, prim_path=f"/map")
        self._load_layout_config()
        self._init_asset_spawn_state()
        self._spawn_layout_bins()
        self._robot = None

        self._load_setup_from_yaml()
        self._setup_missions_and_robot_position()

    def _maybe_print_plan_start(self, mission):
        plan_number = mission.get_plan_number()
        if plan_number is None or plan_number == self._last_printed_plan_number:
            return

        self._last_printed_plan_number = plan_number
        path = mission.get_plan_path() or []
        path_text = " -> ".join(path)
        print(
            f"Starting plan {plan_number} of {self._plan_count}: "
            f"{mission.get_plan_start()} -> {mission.get_plan_target()} "
            f"(path: {path_text})"
        )

    def move_towards_target(self, mission):
        self._maybe_print_plan_start(mission)
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

        if speed - self._previous_speed > 0.05: 
            speed = self._previous_speed + 0.05
        
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
            mission = self._misisons[self._current_mission_number]
            self._maybe_print_plan_start(mission)
            print(
                f"Starting mission leg {self._current_mission_number + 1} "
                f"(of {len(self._misisons)}) "
                f"plan {mission.get_plan_number()} -> {mission.get_waypoint_id()}"
            )
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
                self._stop_simulation()
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
            self._publish_storage_utilization()
            self._publish_sim_progress(time)

        if self._layout_dev:
            self._layout_dev.update(time)
            if self._layout_dev.is_finished(time):
                self._stop_simulation()
                return

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
        self._layout_dev_started = False
        self._manual_asset_manager = None
        self._last_printed_plan_number = None
        print("Reset spawn state for clean restart")

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

    async def _on_spawn_all_objects_event_async(self):
        if not self._layout_config:
            self._load_layout_config()

        section_ids = robot_utils.layout_section_ids(self._layout_config)
        if not section_ids:
            print("No sections found in layout config")
            return

        asset_manager = self._ensure_asset_manager()
        asset_manager.clear_all()

        total_spawned = 0
        for section_id in section_ids:
            total_spawned += asset_manager.spawn_section(section_id)
            await omni.kit.app.get_app().next_update_async()

        print(
            f"Spawned {total_spawned} object(s) across "
            f"{len(section_ids)} section(s) (all bins filled)"
        )

    async def _on_clear_all_objects_event_async(self):
        asset_manager = self._ensure_asset_manager()
        asset_manager.clear_all()
        print("Cleared all spawned objects")
