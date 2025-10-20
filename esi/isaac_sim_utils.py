
from isaacsim.core.utils.viewports import set_camera_view as _set_camera_view
from omni.isaac.core.utils.stage import add_reference_to_stage as _add_reference_to_stage
import omni.isaac.core.utils.prims as prim_utils
import numpy as np
from omni.physx.scripts import physicsUtils
from pxr import UsdGeom, UsdPhysics, Gf
import omni

def set_camera_view(eye=[0,0,0], target=[0,0,0], camera_prim_path="/OmniverseKit_Persp"):
    _set_camera_view(eye=eye, target=target, camera_prim_path=camera_prim_path)

def add_reference_to_stage(usd_path, prim_path):
    _add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

def spawn_object(asset_path, prim_path):
    add_reference_to_stage(usd_path=asset_path, prim_path=prim_path)
    
    # Get the stage
    stage = omni.usd.get_context().get_stage()
    
    # Remove physical properties to make objects non-physical
    prim = stage.GetPrimAtPath(prim_path)
    if prim:
        # Remove RigidBodyAPI if it exists
        rigid_body_api = UsdPhysics.RigidBodyAPI.Get(stage, prim_path)
        if rigid_body_api:
            rigid_body_api.GetPrim().RemoveAPI(UsdPhysics.RigidBodyAPI)
        
        # Remove CollisionAPI if it exists
        collision_api = UsdPhysics.CollisionAPI.Get(stage, prim_path)
        if collision_api:
            collision_api.GetPrim().RemoveAPI(UsdPhysics.CollisionAPI)
        
        # Remove any physics schemas
        physics_schemas = prim.GetAppliedSchemas()
        for schema in physics_schemas:
            if "Physics" in schema:
                prim.RemoveAPI(schema)


def create_dome_light(stage_path="/World/dome_light"):
    light_1 = prim_utils.create_prim(
        stage_path,
        "DomeLight",
        position=np.array([0.0, 0.0, 20.0]),
        attributes={
            "inputs:intensity": 1e3,
        }
    )

def translate_object(stage, prim_path, translate):
    box_mesh = UsdGeom.Mesh.Get(stage, prim_path)
    physicsUtils.set_or_add_translate_op(box_mesh, translate=translate)

def rotate_object(stage, prim_path, rotation):
    """
    Rotate an object around the z-axis by a given yaw angle in degrees.
    
    Args:
        stage: The USD stage
        prim_path (str): The path to the object to rotate
        rotation (float): Yaw angle in degrees around the z-axis
    """
    import robot_utils
    rotation_quaternion = robot_utils.yaw_degrees_to_quaternion(rotation)
    box_mesh = UsdGeom.Mesh.Get(stage, prim_path)
    physicsUtils.set_or_add_orient_op(box_mesh, rotation_quaternion)


async def disable_gravity(scene):
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(0)
    return

def get_sim_time(sim_context):
    return sim_context.current_time