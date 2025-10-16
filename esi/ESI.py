# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

# Note: checkout the required tutorials at https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html


import sys
import os

# Add the current directory to Python path to find local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from pxr import UsdPhysics, PhysxSchema, Gf, PhysicsSchemaTools, UsdGeom
import omni
from omni.isaac.core import SimulationContext
from omni.physx.scripts import physicsUtils
from omni.isaac.core.utils.stage import add_reference_to_stage
from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.core.utils.viewports import set_camera_view
from omni.isaac.core.objects import DynamicCuboid
import numpy as np
import omni.isaac.core.utils.prims as prim_utils
import omni.graph.core as og
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.api.robots import Robot
from omni.isaac.core.utils.rotations import quat_to_rot_matrix, rot_matrix_to_quat
from robot_logger import RobotLogger
from mission import Mission, MissionType, Waypoint, StatusType
from omni.isaac.core.objects import VisualCuboid, FixedCuboid

from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
try:
    import yaml
except Exception:
    yaml = None

# To link this repo with isaac sim:
# cd ~/isaacsim/exts/isaacsim.examples.interactive/isaacsim/examples/interactive
# ln -s /home/${USER}/phd/esi/ user_examples

class square:
    def __init__(self, ll, ur, color, name):
        self.ll = ll
        self.ur = ur
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

        return

    def create_dome_light(self):
        light_1 = prim_utils.create_prim(
            "/World/dome_light",
            "DomeLight",
            position=np.array([0.0, 0.0, 20.0]),
            attributes={
                "inputs:intensity": 1e3,
            }
        )

    def set_camera_view(self):
        # set_camera_view(eye=[-8.0, -25, 30], target=[16.00, 16.00, 0.00], camera_prim_path="/OmniverseKit_Persp")
        set_camera_view(eye=[25.0, 25.0, 100], target=[25.00, 25.00, 0.00], camera_prim_path="/OmniverseKit_Persp")

    def spawn_object(self, asset_path, prim_path):
        add_reference_to_stage(usd_path=asset_path, prim_path=prim_path)
        
        # Remove physical properties to make objects non-physical
        prim = self._stage.GetPrimAtPath(prim_path)
        if prim:
            # Remove RigidBodyAPI if it exists
            rigid_body_api = UsdPhysics.RigidBodyAPI.Get(self._stage, prim_path)
            if rigid_body_api:
                rigid_body_api.GetPrim().RemoveAPI(UsdPhysics.RigidBodyAPI)
            
            # Remove CollisionAPI if it exists
            collision_api = UsdPhysics.CollisionAPI.Get(self._stage, prim_path)
            if collision_api:
                collision_api.GetPrim().RemoveAPI(UsdPhysics.CollisionAPI)
            
            # Remove any physics schemas
            physics_schemas = prim.GetAppliedSchemas()
            for schema in physics_schemas:
                if "Physics" in schema:
                    prim.RemoveAPI(schema)

    def translate_object(self, prim_path, translate):
        box_mesh = UsdGeom.Mesh.Get(self._stage, prim_path)
        physicsUtils.set_or_add_translate_op(box_mesh, translate=translate)

    def rotate_object(self, prim_path, rotation):
        """
        Rotate an object around the z-axis by a given yaw angle in degrees.
        
        Args:
            prim_path (str): The path to the object to rotate
            rotation (float): Yaw angle in degrees around the z-axis
        """
        rotation_quaternion = self.yaw_to_quaternion(rotation)
        box_mesh = UsdGeom.Mesh.Get(self._stage, prim_path)
        physicsUtils.set_or_add_orient_op(box_mesh, rotation_quaternion)

    def yaw_to_quaternion(self, yaw_degrees):
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
    
    def _quaternion_to_yaw_radians(self, quaternion):
        """Convert quaternion to yaw angle in radians around Z-axis"""
        # Extract quaternion components (w, x, y, z)
        w, x, y, z = quaternion
        
        # Calculate yaw angle from quaternion
        # yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        yaw_radians = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        
        return yaw_radians

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

    def setup_missions(self):
        # Load missions from YAML file in esi/missions/mission_1.yaml
        missions_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "missions", "mission_1.yaml")
        waypoints = []
        try:
            if yaml is not None:
                with open(missions_file, "r") as f:
                    data = yaml.safe_load(f)
                    waypoints = data.get("waypoints", []) if data else []
            else:
                # Very small fallback parser: look for lines like '- [x, y]'
                print("Warning: PyYAML not available, using fallback parser for missions file")
                with open(missions_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("- [") and line.endswith("]"):
                            content = line[3:-1]
                            parts = [p.strip() for p in content.split(",")]
                            if len(parts) >= 2:
                                try:
                                    x = float(parts[0])
                                    y = float(parts[1])
                                    waypoints.append([x, y])
                                except Exception:
                                    pass
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
            missions.append(Mission(i, MissionType.MOVE_TO_WAYPOINT, self.create_waypoint(wx, wy)))

        if missions:
            missions[0].set_status(StatusType.IN_PROGRESS)
        self._current_mission_number = 0
        self._all_missions_completed = False
        self._new_mission = True

        return missions


    def setup_scene(self):
        self.create_dome_light()
        self._world = self.get_world()
        self._stage = omni.usd.get_context().get_stage()
        add_reference_to_stage(usd_path=self._import_map_usd_path, prim_path=f"/map")
        add_reference_to_stage(usd_path=self._import_robot_usd_path, prim_path=f"/float_bot")
        self._robot = self._world.scene.add(Robot(prim_path="/float_bot", name="float_bot"))

        start_x = 25.0
        start_y = 25.0
        # Move robot to start position
        self.translate_object("/float_bot", Gf.Vec3f(start_x, start_y, 0.0))
        # self.rotate_object("/float_bot", -90.0)

        # Move map coordinate system so that (0,0) is at bottom left corner
        self.translate_object("/float_bot/map", Gf.Vec3f(-start_x, -start_y, 0.0))

        self._misisons = self.setup_missions()        

        return

    async def disable_gravity(self):
        stage = omni.usd.get_context().get_stage()
        scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(0)
    
        return


    def check_if_at_waypoint(self, mission, time):
        robot_current_position, robot_current_orientation = self._robot.get_world_pose()
        waypoint = mission.get_waypoint()
        # Convert waypoint to numpy array for distance calculation
        waypoint_position = np.array([waypoint.x, waypoint.y, robot_current_position[2]]) 
        distance = np.linalg.norm(robot_current_position - waypoint_position)
        return distance < 0.3


    def move_towards_target(self, mission):
        robot_current_position, robot_current_orientation = self._robot.get_world_pose()
        waypoint = mission.get_waypoint()
        
        # Calculate the direction vector from robot to waypoint
        direction_vector = np.array([waypoint.x - robot_current_position[0], 
                                   waypoint.y - robot_current_position[1]])
        
        # Calculate the required yaw angle to face the target
        target_yaw_radians = np.arctan2(direction_vector[1], direction_vector[0])
        
        # Get current robot yaw from quaternion
        current_yaw_radians = self._quaternion_to_yaw_radians(robot_current_orientation)
        
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
        max_angular_velocity = 2.0  # rad/s
        angular_velocity_z = np.clip(angular_velocity_z, -max_angular_velocity, max_angular_velocity)
        print(f"Yaw error (radians): {yaw_error}, Angular velocity (rad/s): {angular_velocity_z}")
        
        # Set angular velocity for smooth turning
        angular_velocity = np.array([0.0, 0.0, angular_velocity_z])
        self._robot.set_angular_velocity(angular_velocity)
        
        # Set linear velocity in the direction the robot is facing
        # Get current robot yaw for forward direction
        current_yaw_radians = self._quaternion_to_yaw_radians(robot_current_orientation)
        
        # Calculate forward direction vector based on current orientation
        forward_direction = np.array([np.cos(current_yaw_radians), np.sin(current_yaw_radians)])
        
        # Set movement speed (adjust as needed)
        top_speed = 2.5
        speed = 1.0  # meters per second

        # Convert waypoint to numpy array for distance calculation
        distance = np.linalg.norm(robot_current_position - np.array([waypoint.x, waypoint.y, robot_current_position[2]]))

        if distance < 0.5:
            speed = max(speed * distance/2.5, 0.5)

        if speed - self._previous_speed > 0.005: 
            speed = self._previous_speed + 0.005
        
        if speed > top_speed:
            speed = top_speed

        print(f"Speed: {speed}")
        print(f"previous speed: {self._previous_speed}")
        
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

        done = self.check_if_at_waypoint(mission, time)
        if done:
            mission.set_status(StatusType.SUCCESS)
            print(f"setting mission status to SUCCESS")
            return mission
        
        # print(f"Robot current position: {robot_current_position}")
        # done = check_at_waypoint()

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
        self.step_mission(time)


    async def setup_post_load(self):
        self._world = self.get_world()
        self._robot = self._world.scene.get_object("float_bot")

        self.set_camera_view()
        await self.disable_gravity()

        self._simulation_context = SimulationContext()

        self.register_sim_step_callback()
        return

    async def setup_pre_reset(self):
        print("Pre Reset")
        # Stop robot motion and clear missions
        self.stop_motion()
        self._misisons = []
        self._current_mission_number = 0
        self._all_missions_completed = False
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

        self._misisons = self.setup_missions()
        print("Post Reset")
        return

    def world_cleanup(self):
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


    def get_random_not_free_space(self, free_spaces, seed=42):
        # Set seed for reproducible "random" placement
        np.random.seed(seed)
        
        do_continue = True
        while do_continue:
            random_space = (np.random.randint(1, 49), np.random.randint(1, 49))
            sq = square([random_space[0] -1, random_space[1] -1], [random_space[0] + 1, random_space[1] + 1], np.array([0.0, 1.0, 0.0]), "1")
        
            do_continue = self.check_square_overlap(sq, free_spaces)
 
        return random_space, sq


    async def _on_add_objects_event_async(self):

        # Define free space
        square_1 = square([15,15], [35,35], np.array([0.0, 1.0, 0]), "1")
        free_spaces = []
        
        asset_path = f"/home/{self._USER}/isaac_sim_files/collection/wooden_box_2x2m/wooden_box_2x2m.usd"

        for i in range(1,2):
            await self.add_cube_at(eval(f"square_{i}"))
            free_spaces.append(eval(f"square_{i}"))

        for i in range(50):
            prim_name = f"/map/WoodenCrate_A1_{i}"
            random_pos, sq = self.get_random_not_free_space(free_spaces, seed=42 + i)
            free_spaces.append(sq)

            # print("random_pos: ", random_pos)
            self.spawn_object(asset_path, prim_name)
            self.translate_object(prim_name, Gf.Vec3f(random_pos[0], random_pos[1], 0.0))
            # Apply collision API to the prim
            prim = self._stage.GetPrimAtPath(prim_name)
            UsdPhysics.CollisionAPI.Apply(prim)



        return

        

