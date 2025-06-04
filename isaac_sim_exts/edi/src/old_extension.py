
import sys
sys.path.append('/home/mircrda/phd/isaac_sim_exts/edi/src')

import omni.ext
import omni.ui as ui
import omni.usd
from pxr import UsdGeom
import numpy as np
from isaacsim.core.api.objects import DynamicCuboid
from sim import hi

from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.core.utils.types import ArticulationAction
#from omni.isaac.core.articulations import Articulation
from isaacsim.core.prims import SingleArticulation

from isaacsim.robot.wheeled_robots.controllers import DifferentialController
import isaacsim.core.utils.stage as stage_utils
from omni.isaac.core.scenes import Scene


# from omni.isaac.core.scenes import Scene
# from omni.isaac.core.prims import XFormPrimView
# from omni.isaac.core.utils.nucleus import get_assets_root_path
# import omni.isaac.core.utils.stage as stage_utils
# from omni.isaac.core.robots import Robot
# from omni.isaac.core.articulations import Articulation

# class mobile_robot(WheeledRobot):

class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return

    # This function is called to setup the assets in the scene for the first time
    # Class variables should not be assigned here, since this function is not called
    # after a hot-reload, its only called to load the world starting from an EMPTY stage
    def setup_scene(self):
        # A world is defined in the BaseSample, can be accessed everywhere EXCEPT __init__
        #world = self.get_world()
        print("Setting up scene...")

        return
    


# Any class derived from `omni.ext.IExt` in top level module (defined in `python.modules` of `extension.toml`) will be
# instantiated when extension gets enabled and `on_startup(ext_id)` will be called. Later when extension gets disabled
# on_shutdown() is called.
class MyExtension(omni.ext.IExt):
    
    
    
    # ext_id is current extension id. It can be used with extension manager to query additional information, like where
    # this extension is located on filesystem.
    def on_startup(self, ext_id):
        print("[maticodes.project.first] MyExtension startup")
        #self.hello_world = HelloWorld()
        #self.hello_world.setup_scene()
        #robot = self.hello_world._jetbot

        #world = self.get_world()

        self._window = ui.Window("My Window", width=300, height=500)
        with self._window.frame:
            with ui.VStack():
                ui.Label("Some Label")

                def move():
                    print("moving robot")


                    assets_root_path = get_assets_root_path()
                    asset_path = assets_root_path + "/Isaac/Robots/Clearpath/Dingo/dingo.usd"

                    scene = Scene()

                    stage_utils.add_reference_to_stage(usd_path=asset_path, prim_path="/World/Fancy_Robot")

                    jetbot = scene.add(
                            WheeledRobot(
                                prim_path="/World/Fancy_Robot",
                                name="fancy_robot",
                                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                                create_robot=True,
                                usd_path=asset_path,
                            )
                        )

                    action = ArticulationAction(joint_velocities = np.array([1.14, 1.42]))
                    jetbot.apply_wheel_actions(action)                    

                    # assets_root_path = get_assets_root_path()
                    # asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
                    # stage_utils.add_reference_to_stage(usd_path=asset_path, prim_path="/World/Fancy_Robot")

                    # jetbot_prim_path = "/jetbot"
                    # jetbot = WheeledRobot(prim_path=jetbot_prim_path, 
                    #                       name="Joan", 
                    #                       wheel_dof_names=["left_wheel_joint", "right_wheel_joint"])
                    # throttle = 1.0
                    # steering = 0.5
                    # controller = DifferentialController(name="simple_control", wheel_radius=0.035, wheel_base=0.1)
                    # cmd = [throttle, steering]
                    #jetbot.apply_wheel_actions(controller.forward(cmd))

                def on_click():
                    print("created cube!")

                    # DynamicCuboid(
                    #     prim_path="/new_cube_2",
                    #     name="cube_1",
                    #     position=np.array([0, 0, 1.0]),
                    #     scale=np.array([0.6, 0.5, 0.2]),
                    #     size=1.0,
                    #     color=np.array([255, 0, 0]),
                    #     )

                    #hi()

                    stage = omni.usd.get_context().get_stage()
                    selection = omni.usd.get_context().get_selection()
                    for prim_path in selection.get_selected_prim_paths():
                        parent = stage.GetPrimAtPath(prim_path)
                        cube = UsdGeom.Cube.Define(stage, parent.GetPath().AppendPath("Cube"))

                ui.Button("Create Cubes", clicked_fn=lambda: on_click())
                ui.Button("Move robot", clicked_fn=lambda: move())

    def on_shutdown(self):
        print("[maticodes.project.first] MyExtension shutdown")
