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

from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path

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
        self._import_robot_usd_path = f"/home/{self._USER}/isaac_sim_files/float_bot_2.usd"
        self._import_map_usd_path = f"/home/{self._USER}/isaac_sim_files/map_2_for_import.usd"

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

    def setup_missions(self):
        mission_1 = Mission(0, MissionType.MOVE_TO_WAYPOINT, Waypoint(0.0, 37.5))
        mission_2 = Mission(1, MissionType.MOVE_TO_WAYPOINT, Waypoint(0.0, 40.0))
        mission_1.set_status(StatusType.IN_PROGRESS)
        return [mission_1, mission_2]


    def setup_scene(self):
        self.create_dome_light()
        self._world = self.get_world()
        self._stage = omni.usd.get_context().get_stage()
        add_reference_to_stage(usd_path=self._import_map_usd_path, prim_path=f"/map")
        add_reference_to_stage(usd_path=self._import_robot_usd_path, prim_path=f"/float_bot")
        self._robot = self._world.scene.add(Robot(prim_path="/float_bot", name="float_bot"))

        self.translate_object("/float_bot", Gf.Vec3f(7.5, 37.5, 0.0))
        self.rotate_object("/float_bot", -90.0)

        self._misisons = self.setup_missions()
        for mission in self._misisons:
            print(mission)

        self._current_mission_number = 0

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
        print(f"Distance to waypoint: {distance}")
        return distance < 0.1

    def step_move_to_waypoint(self, mission, time):
        print(f"Moving to waypoint {mission.get_waypoint()}")

        done = self.check_if_at_waypoint(mission, time)
        if done:
            mission.set_status(StatusType.SUCCESS)
            print(f"setting mission status to SUCCESS")
            return mission
        
        # print(f"Robot current position: {robot_current_position}")
        # done = check_at_waypoint()

        return mission

    def step_pause(self, mission, time):
        print(f"On pause...")
        return

    def step_mission(self, time):
        # Check if we have completed all missions
        if self._current_mission_number >= len(self._misisons):
            print(f"All missions completed")
            return
        
        print(f"Stepping mission nr. {self._current_mission_number}")
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
            print(f"Mission {self._current_mission_number} completed successfully!")
            self._current_mission_number += 1
            if self._current_mission_number < len(self._misisons):
                self._misisons[self._current_mission_number].set_status(StatusType.IN_PROGRESS)
                print(f"Starting mission {self._current_mission_number}")
            else:
                print(f"All missions completed")
                return
        # print(f"after stepping -")
        # print(f"Mission nr. {self._current_mission_number} status: {misssion.get_status()}")



    def custom_simulation_step(self, step_size):
        time = self._simulation_context.current_time
        self.step_mission(time)

        # Define time intervals with both linear and angular velocities (in robot's local frame)
        # Format: (time_threshold, linear_velocity, angular_velocity)
        # linear_velocity: [forward, left, up] in robot's frame
        # angular_velocity: [roll, pitch, yaw] in robot's frame

        # speed_scalar = 1

        # cmd_1 = (0, np.array([speed_scalar*1.0, 0.0, 0.0]), np.array([0.0, 0.0, speed_scalar*0.0]))
        # cmd_2 = (4, np.array([speed_scalar*0.0, 0.0, 0.0]), np.array([0.0, 0.0, speed_scalar*0.5]))       
        # cmd_3 = (6, np.array([speed_scalar*1.0, 0.0, 0.0]), np.array([0.0, 0.0, speed_scalar*0.0]))       
        # cmd_4 = (12, np.array([speed_scalar*0.0, 0.0, 0.0]), np.array([0.0, 0.0, speed_scalar*0.5]))       
        # cmd_5 = (14, np.array([speed_scalar*1.0, 0.0, 0.0]), np.array([0.0, 0.0, speed_scalar*0.0]))       
        # cmd_6 = (20, np.array([speed_scalar*0.0, 0.0, 0.0]), np.array([0.0, 0.0, speed_scalar*0.5]))       
        # cmd_7 = (22, np.array([speed_scalar*1.0, 0.0, 0.0]), np.array([0.0, 0.0, speed_scalar*0.0]))       
        # cmd_8 = (26, np.array([speed_scalar*0.0, 0.0, 0.0]), np.array([0.0, 0.0, speed_scalar*0.5]))       
        # cmd_9 = (28, np.array([speed_scalar*1.0, 0.0, 0.0]), np.array([0.0, 0.0, speed_scalar*0.0]))   
        # cmd_10 = (35, np.array([speed_scalar*0.0, 0.0, 0.0]), np.array([0.0, 0.0, speed_scalar*0.5]))       
        # cmd_11 = (37, np.array([speed_scalar*1.0, 0.0, 0.0]), np.array([0.0, 0.0, speed_scalar*0.0])) 
        # cmd_12 = (41, np.array([speed_scalar*0.0, 0.0, 0.0]), np.array([0.0, 0.0, speed_scalar*0.5]))       
        # cmd_13 = (43, np.array([speed_scalar*1.0, 0.0, 0.0]), np.array([0.0, 0.0, speed_scalar*0.0])) 
        # cmd_14 = (46, np.array([speed_scalar*0.0, 0.0, 0.0]), np.array([0.0, 0.0, speed_scalar*0.5]))       
        # cmd_15 = (48, np.array([speed_scalar*1.0, 0.0, 0.0]), np.array([0.0, 0.0, speed_scalar*0.0])) 
        # cmd_16 = (50, np.array([speed_scalar*0.0, 0.0, 0.0]), np.array([0.0, 0.0, speed_scalar*0.5]))       
        # cmd_17 = (53, np.array([speed_scalar*1.0, 0.0, 0.0]), np.array([0.0, 0.0, speed_scalar*0.0]))     
        # stop = (55, np.array([speed_scalar*0.0, 0.0, 0.0]), np.array([0.0, 0.0, speed_scalar*0.0]))         

        # velocity_schedule = [
        #     cmd_1,
        #     cmd_2,  
        #     cmd_3,
        #     cmd_4,
        #     cmd_5,
        #     cmd_6,
        #     cmd_7,
        #     cmd_8,
        #     cmd_9,
        #     cmd_10,
        #     cmd_11,
        #     cmd_12,
        #     cmd_13,
        #     cmd_14,
        #     cmd_15,
        #     cmd_16,
        #     cmd_17,
        #     stop
        # ]
        
        # # Find the appropriate velocities based on current time
        # current_linear_velocity_local = np.array([0.0, 0.0, 0.0])  # Default linear velocity (robot frame)
        # current_angular_velocity_local = np.array([0.0, 0.0, 0.0]) # Default angular velocity (robot frame)
        
        # for threshold_time, linear_vel, angular_vel in velocity_schedule:
        #     if time >= threshold_time:
        #         current_linear_velocity_local = linear_vel
        #         current_angular_velocity_local = angular_vel
        
        # # Get robot's current orientation
        # position, orientation = self._robot.get_world_pose()
        
        # # Transform linear velocity from robot's local frame to world frame
        # # Convert quaternion to rotation matrix
        # rot_matrix = quat_to_rot_matrix(orientation)
        
        # # Transform linear velocity: world_vel = R * local_vel
        # current_linear_velocity_world = rot_matrix @ current_linear_velocity_local
        
        # # Angular velocity is already in world frame (no transformation needed)
        # # But if you want it relative to robot's frame, you would need to transform it too
        # current_angular_velocity_world = current_angular_velocity_local
        
        # # Set the robot's velocities in world frame
        # self._robot.set_linear_velocity(current_linear_velocity_world)
        # self._robot.set_angular_velocity(current_angular_velocity_world)


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
        return

    async def setup_post_reset(self):
        self._world = self.get_world()
        self.register_sim_step_callback()
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
            DynamicCuboid(prim_path=f"/World/cubes/cube_{sq.name}", 
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


    def get_random_not_free_space(self, free_spaces):

        do_continue = True
        while do_continue:
            random_space = (np.random.randint(1, 49), np.random.randint(1, 49))
            sq = square([random_space[0] -1, random_space[1] -1], [random_space[0] + 1, random_space[1] + 1], np.array([0.0, 1.0, 0.0]), "1")
        
            do_continue = self.check_square_overlap(sq, free_spaces)
 
        return random_space, sq


    async def _on_add_objects_event_async(self):

        # Define free space
        square_1 = square([5,0], [10,50], np.array([0.0, 1.0, 0.0]), "1")
        square_2 = square([20,0], [25,50], np.array([0.0, 1.0, 0.0]), "2")
        square_3 = square([35,0], [40,50], np.array([0.0, 1.0, 0.0]), "3")
        square_4 = square([0,10], [50,15], np.array([0.0, 1.0, 0.0]), "4")
        square_5 = square([0,35], [50,40], np.array([0.0, 1.0, 0.0]), "5")
        free_spaces = []
        
        asset_path = f"/home/{self._USER}/isaac_sim_files/collection/wooden_box_2x2m/wooden_box_2x2m.usd"

        for i in range(1,6):
            await self.add_cube_at(eval(f"square_{i}"))
            free_spaces.append(eval(f"square_{i}"))

        for i in range(50):
            prim_name = f"/WoodenCrate_A1_{i}"
            random_pos, sq = self.get_random_not_free_space(free_spaces)
            free_spaces.append(sq)

            # print("random_pos: ", random_pos)
            self.spawn_object(asset_path, prim_name)
            self.translate_object(prim_name, Gf.Vec3f(random_pos[0], random_pos[1], 0.0))
            # Apply collision API to the prim
            prim = self._stage.GetPrimAtPath(prim_name)
            UsdPhysics.CollisionAPI.Apply(prim)



        return

