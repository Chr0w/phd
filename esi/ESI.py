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

# To link this repo with isaac sim:
# cd ~/isaacsim/exts/isaacsim.examples.interactive/isaacsim/examples/interactive
# ln -s /home/${USER}/phd/esi/ user_examples

class ESI(BaseSample):

    def register_sim_step_callback(self):
        print("Registering sim step callback")
        self._world.add_physics_callback("sim_step", callback_fn=self.custom_simulation_step)


    def __init__(self) -> None:
        super().__init__()

        USER = os.environ.get("USER")
        self._import_robot_usd_path = f"/home/{USER}/isaac_sim_files/float_bot_2.usd"
        self._import_map_usd_path = f"/home/{USER}/isaac_sim_files/map_1_for_import.usd"

        # Initialize robot logger
        # self._robot_logger = RobotLogger(log_interval=0.1, stop_logging_time=15.0)

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
        set_camera_view(eye=[-25.0, -35, 30], target=[0.00, 0.00, 0.00], camera_prim_path="/OmniverseKit_Persp")

    def setup_scene(self):
        self.create_dome_light()
        world = self.get_world()
        add_reference_to_stage(usd_path=self._import_map_usd_path, prim_path=f"/map")
        add_reference_to_stage(usd_path=self._import_robot_usd_path, prim_path=f"/float_bot")
        self._robot = self._world.scene.add(Robot(prim_path="/float_bot", name="float_bot"))

        return

    async def disable_gravity(self):
        stage = omni.usd.get_context().get_stage()
        scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(0)
    
        return


    def print_cube_info(self):
        position, orientation = self._robot.get_world_pose()
        current_time = self._simulation_context.current_time
        
        
        # Log robot pose data
        # try:
        #     logging_stopped = self._robot_logger.log_robot_pose(current_time, position, orientation)
        #     if logging_stopped:
        #         print("Logging has been stopped and CSV file saved!")
        # except Exception as e:
        #     print(f"Error in logging: {e}")


    def custom_simulation_step(self, step_size):
        time = self._simulation_context.current_time

        # Define time intervals with both linear and angular velocities (in robot's local frame)
        # Format: (time_threshold, linear_velocity, angular_velocity)
        # linear_velocity: [forward, left, up] in robot's frame
        # angular_velocity: [roll, pitch, yaw] in robot's frame

        speed_scalar = 1

        cmd_1 = (0, np.array([speed_scalar*1.0, 0.0, 0.0]), np.array([0.0, 0.0, speed_scalar*0.0]))
        cmd_2 = (4, np.array([speed_scalar*0.0, 0.0, 0.0]), np.array([0.0, 0.0, speed_scalar*0.5]))       
        cmd_3 = (6, np.array([speed_scalar*1.0, 0.0, 0.0]), np.array([0.0, 0.0, speed_scalar*0.0]))       
        cmd_4 = (12, np.array([speed_scalar*0.0, 0.0, 0.0]), np.array([0.0, 0.0, speed_scalar*0.5]))       
        cmd_5 = (14, np.array([speed_scalar*1.0, 0.0, 0.0]), np.array([0.0, 0.0, speed_scalar*0.0]))       
        cmd_6 = (20, np.array([speed_scalar*0.0, 0.0, 0.0]), np.array([0.0, 0.0, speed_scalar*0.5]))       
        cmd_7 = (22, np.array([speed_scalar*1.0, 0.0, 0.0]), np.array([0.0, 0.0, speed_scalar*0.0]))       
        cmd_8 = (26, np.array([speed_scalar*0.0, 0.0, 0.0]), np.array([0.0, 0.0, speed_scalar*0.5]))       
        cmd_9 = (28, np.array([speed_scalar*1.0, 0.0, 0.0]), np.array([0.0, 0.0, speed_scalar*0.0]))   
        cmd_10 = (35, np.array([speed_scalar*0.0, 0.0, 0.0]), np.array([0.0, 0.0, speed_scalar*0.5]))       
        cmd_11 = (37, np.array([speed_scalar*1.0, 0.0, 0.0]), np.array([0.0, 0.0, speed_scalar*0.0])) 
        cmd_12 = (41, np.array([speed_scalar*0.0, 0.0, 0.0]), np.array([0.0, 0.0, speed_scalar*0.5]))       
        cmd_13 = (43, np.array([speed_scalar*1.0, 0.0, 0.0]), np.array([0.0, 0.0, speed_scalar*0.0])) 
        cmd_14 = (46, np.array([speed_scalar*0.0, 0.0, 0.0]), np.array([0.0, 0.0, speed_scalar*0.5]))       
        cmd_15 = (48, np.array([speed_scalar*1.0, 0.0, 0.0]), np.array([0.0, 0.0, speed_scalar*0.0])) 
        cmd_16 = (50, np.array([speed_scalar*0.0, 0.0, 0.0]), np.array([0.0, 0.0, speed_scalar*0.5]))       
        cmd_17 = (53, np.array([speed_scalar*1.0, 0.0, 0.0]), np.array([0.0, 0.0, speed_scalar*0.0]))     
        stop = (55, np.array([speed_scalar*0.0, 0.0, 0.0]), np.array([0.0, 0.0, speed_scalar*0.0]))         

        velocity_schedule = [
            cmd_1,
            cmd_2,  
            cmd_3,
            cmd_4,
            cmd_5,
            cmd_6,
            cmd_7,
            cmd_8,
            cmd_9,
            cmd_10,
            cmd_11,
            cmd_12,
            cmd_13,
            cmd_14,
            cmd_15,
            cmd_16,
            cmd_17,
            stop
        ]
        
        # Find the appropriate velocities based on current time
        current_linear_velocity_local = np.array([0.0, 0.0, 0.0])  # Default linear velocity (robot frame)
        current_angular_velocity_local = np.array([0.0, 0.0, 0.0]) # Default angular velocity (robot frame)
        
        for threshold_time, linear_vel, angular_vel in velocity_schedule:
            if time >= threshold_time:
                current_linear_velocity_local = linear_vel
                current_angular_velocity_local = angular_vel
        
        # Get robot's current orientation
        position, orientation = self._robot.get_world_pose()
        
        # Transform linear velocity from robot's local frame to world frame
        # Convert quaternion to rotation matrix
        rot_matrix = quat_to_rot_matrix(orientation)
        
        # Transform linear velocity: world_vel = R * local_vel
        current_linear_velocity_world = rot_matrix @ current_linear_velocity_local
        
        # Angular velocity is already in world frame (no transformation needed)
        # But if you want it relative to robot's frame, you would need to transform it too
        current_angular_velocity_world = current_angular_velocity_local
        
        # Set the robot's velocities in world frame
        self._robot.set_linear_velocity(current_linear_velocity_world)
        self._robot.set_angular_velocity(current_angular_velocity_world)

        # self.print_cube_info()


    async def setup_post_load(self):
        self._world = self.get_world()
        self._robot = self._world.scene.get_object("float_bot")
        # self._robot.set_world_pose(np.array([-10.0, 10.0, 0.2]), np.array([0.0, 0.0, 0.0, 1.0]))
        # self._robot.set_linear_velocity(np.array([0.0, 0.0, 0.0]))

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
