# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from pxr import UsdPhysics, PhysxSchema, Gf, PhysicsSchemaTools, UsdGeom
import omni
from omni.isaac.core.utils.stage import add_reference_to_stage
from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.core.utils.viewports import set_camera_view
from omni.isaac.core.objects import DynamicCuboid
import numpy as np
import omni.isaac.core.utils.prims as prim_utils
import omni.graph.core as og
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.api.robots import Robot

# Note: checkout the required tutorials at https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html


class ESI(BaseSample):
    def __init__(self) -> None:
        super().__init__()

        self._import_robot_usd_path = "/home/chrdam/isaac_sim_files/mockbot_2_for_import.usd"
        self._import_map_usd_path = "/home/chrdam/isaac_sim_files/map_1_for_import.usd"

        return

    def setup_scene(self):

        light_1 = prim_utils.create_prim(
            "/World/dome_light",
            "DomeLight",
            position=np.array([0.0, 0.0, 20.0]),
            attributes={
                "inputs:intensity": 1e3,
            }
        )

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
        print("-----")


    def custom_simulation_step(self, step_size):
        # This runs every simulation step
        # You can access objects, print info, etc.
        self.print_cube_info()
        pass

    async def setup_post_load(self):
        self._world = self.get_world()
        # self._jetbot = self._world.scene.get_object("mockbot_2")
        self._robot = self._world.scene.get_object("mockbot_2")
        self._robot.set_linear_velocity(np.array([1.0, 0.0, 0.0]))


        # self._world.add_physics_callback("sim_step", callback_fn=self.print_cube_info)
        set_camera_view(eye=[30.0, 30, 30], target=[0.00, 0.00, 0.00], camera_prim_path="/OmniverseKit_Persp")
        await self.disable_gravity()
        self._world.add_physics_callback("sim_step", callback_fn=self.custom_simulation_step)

        return





    async def setup_pre_reset(self):
        return

    async def setup_post_reset(self):
        return

    def world_cleanup(self):
        return
