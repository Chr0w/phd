# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

# Note: checkout the required tutorials at https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html


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

# To link this repo with isaac sim:
# cd ~/isaacsim/exts/isaacsim.examples.interactive/isaacsim/examples/interactive
# ln -s /home/${USER}/phd/esi/ user_examples

class ESI(BaseSample):
    def __init__(self) -> None:
        super().__init__()

        self._import_robot_usd_path = "/home/chrdam/isaac_sim_files/mockbot_2_for_import.usd"
        self._import_map_usd_path = "/home/chrdam/isaac_sim_files/map_1_for_import.usd"

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
        set_camera_view(eye=[30.0, 30, 30], target=[0.00, 0.00, 0.00], camera_prim_path="/OmniverseKit_Persp")

    def setup_scene(self):
        self.create_dome_light()
        world = self.get_world()
        add_reference_to_stage(usd_path=self._import_map_usd_path, prim_path=f"/map")
        add_reference_to_stage(usd_path=self._import_robot_usd_path, prim_path=f"/mockbot_2")
        self._robot = self._world.scene.add(Robot(prim_path="/mockbot_2", name="mockbot_2"))

        return

    async def disable_gravity(self):
        stage = omni.usd.get_context().get_stage()
        scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(0)
    
        return


    def print_cube_info(self):
        position, orientation = self._robot.get_world_pose()
        print(f"Cube Position: {position}")
        print(f"Sim time: {self._simulation_context.current_time}")
        print("---------")


    def custom_simulation_step(self, step_size):
        time = self._simulation_context.current_time

        # Define time intervals with both linear and angular velocities (in robot's local frame)
        # Format: (time_threshold, linear_velocity, angular_velocity)
        # linear_velocity: [forward, left, up] in robot's frame
        # angular_velocity: [roll, pitch, yaw] in robot's frame
        velocity_schedule = [
            (0.0, np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),   # Move forward, no rotation
            (3.0, np.array([0.5, 0.0, 0.0]), np.array([0.0, 0.0, 0.5])),   # Move forward slowly while turning left
            (6.0, np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),   # Move forward, no rotation
            (9.0, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, -0.5])),  # Stop moving, turn right
            (12.0, np.array([2.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),  # Move forward fast, no rotation
            (15.0, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),  # Stop everything
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

        self.print_cube_info()


    async def setup_post_load(self):
        self._world = self.get_world()
        self._robot = self._world.scene.get_object("mockbot_2")
        # self._robot.set_world_pose(np.array([-10.0, 10.0, 0.2]), np.array([0.0, 0.0, 0.0, 1.0]))
        # self._robot.set_linear_velocity(np.array([0.0, 0.0, 0.0]))

        self.set_camera_view()
        # await self.disable_gravity()

        self._simulation_context = SimulationContext()

        # Declare callbacks
        self._world.add_physics_callback("sim_step", callback_fn=self.custom_simulation_step)

        return





    async def setup_pre_reset(self):
        return

    async def setup_post_reset(self):
        return

    def world_cleanup(self):
        return
