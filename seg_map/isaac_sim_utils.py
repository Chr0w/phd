
import math
from fractions import Fraction

import numpy as np
from omni.physx.scripts import physicsUtils
from pxr import Usd, UsdGeom, UsdPhysics, Gf, UsdLux, Sdf
import omni
import carb.settings
from isaacsim.core.rendering_manager import ViewportManager, RenderingManager
from isaacsim.core.simulation_manager import SimulationManager
import isaacsim.core.experimental.utils.stage as stage_utils

_SETTING_RATE_LIMIT_ENABLED = "/app/runLoops/main/rateLimitEnabled"
_SETTING_RATE_LIMIT_FREQUENCY = "/app/runLoops/main/rateLimitFrequency"
_SETTING_FABRIC_DEFAULT_SIM_PERIOD_NUMERATOR = "/app/settings/fabricDefaultSimPeriodNumerator"
_SETTING_FABRIC_DEFAULT_SIM_PERIOD_DENOMINATOR = "/app/settings/fabricDefaultSimPeriodDenominator"


def configure_real_time_factor(rtf: float = 2.0, loop_hz: float = 60.0) -> None:
    """
    Target RTF = sim_elapsed / wall_elapsed at the given loop rate.

    At RTF=2 and loop_hz=60, each 60 Hz tick advances 2/60 s of sim time, so one
    wall second contains two sim seconds.
    """
    if rtf <= 0.0:
        raise ValueError(f"rtf must be positive, got {rtf}")
    if loop_hz <= 0.0:
        raise ValueError(f"loop_hz must be positive, got {loop_hz}")

    sim_dt = rtf / loop_hz
    time_codes_per_second = loop_hz / rtf

    SimulationManager.setup_simulation(dt=sim_dt)
    RenderingManager.set_dt(1.0 / loop_hz)

    settings = carb.settings.get_settings()
    settings.set_bool(_SETTING_RATE_LIMIT_ENABLED, True)
    settings.set_float(_SETTING_RATE_LIMIT_FREQUENCY, loop_hz)
    settings.set_bool("/app/player/useFixedTimeStepping", True)

    timeline = omni.timeline.get_timeline_interface()
    timeline.set_target_framerate(loop_hz)
    timeline.set_time_codes_per_second(time_codes_per_second)
    timeline.set_play_every_frame(True)

    stage = stage_utils.get_current_stage(backend="usd")
    if stage:
        with Usd.EditContext(stage, stage.GetRootLayer()):
            stage.SetTimeCodesPerSecond(time_codes_per_second)

    sim_period = Fraction(sim_dt).limit_denominator(1_000_000_000)
    settings.set_int(_SETTING_FABRIC_DEFAULT_SIM_PERIOD_NUMERATOR, sim_period.numerator)
    settings.set_int(_SETTING_FABRIC_DEFAULT_SIM_PERIOD_DENOMINATOR, sim_period.denominator)

    try:
        from omni.kit.loop import _loop as omni_loop

        loop_runner = omni_loop.acquire_loop_interface()
        if hasattr(loop_runner, "set_manual_step_size"):
            loop_runner.set_manual_step_size(sim_dt)
        if hasattr(loop_runner, "set_manual_mode"):
            loop_runner.set_manual_mode(True)
    except Exception:
        pass

    print(
        f"Configured simulation RTF={rtf:g} at {loop_hz:g} Hz "
        f"(sim_dt={sim_dt:.6f}s, timeCodesPerSecond={time_codes_per_second:g})"
    )


def _get_prim(stage, prim_path):
    prim_path = str(prim_path)
    prim = stage.GetPrimAtPath(Sdf.Path(prim_path))
    if prim and prim.IsValid():
        return prim
    return None

def set_camera_view(eye=[0,0,0], target=[0,0,0], camera_prim_path="/OmniverseKit_Persp"):
    ViewportManager.set_camera_view(camera=camera_prim_path, eye=eye, target=target)

def add_reference_to_stage(usd_path, prim_path):
    stage_utils.add_reference_to_stage(usd_path=usd_path, path=prim_path)

def spawn_object(asset_path, prim_path):
    add_reference_to_stage(usd_path=asset_path, prim_path=prim_path)
    
    # Get the stage
    stage = omni.usd.get_context().get_stage()
    
    # Remove physical properties to make objects non-physical
    prim = _get_prim(stage, prim_path)
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


def _strip_rigid_bodies_recursive(stage, prim_path):
    """Remove RigidBodyAPI only, preserving mesh colliders for lidar."""
    prim = _get_prim(stage, prim_path)
    if not prim or not prim.IsValid():
        return

    stack = [prim]
    while stack:
        current = stack.pop()
        if UsdPhysics.RigidBodyAPI.Get(stage, current.GetPath()):
            current.RemoveAPI(UsdPhysics.RigidBodyAPI)
        stack.extend(current.GetChildren())


def _yaw_degrees_to_quat_h(yaw_degrees):
    half_yaw = math.radians(float(yaw_degrees)) / 2.0
    return Gf.Quath(math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))


EMPTY_PROTO_INDEX = 0


def _ensure_empty_prototype(stage, prototypes_root):
    """Collision-free prototype used for hidden instancer slots."""
    empty_path = f"{prototypes_root}/proto_{EMPTY_PROTO_INDEX:02d}"
    prim = _get_prim(stage, empty_path)
    if prim and prim.IsValid():
        return empty_path
    UsdGeom.Xform.Define(stage, empty_path)
    return empty_path


def _set_slot_proto_index(instancer, slot_index, proto_index):
    proto_indices_attr = instancer.GetProtoIndicesAttr()
    proto_indices = list(proto_indices_attr.Get() if proto_indices_attr else [])
    if slot_index >= len(proto_indices):
        return
    proto_indices[slot_index] = proto_index
    if proto_indices_attr:
        proto_indices_attr.Set(proto_indices)
    else:
        instancer.CreateProtoIndicesAttr(proto_indices)


def spawn_point_instancer(stage, instancer_path, instances):
    """
    Spawn many assets efficiently via UsdGeomPointInstancer.

    Each instance dict requires: asset_path, x, y, yaw_degrees.
    Optional keys: z (defaults to 0).

    Prototype assets keep their preset collider/mesh geometry for lidar but
    have RigidBodyAPI stripped so they do not participate in simulation.
    """
    if not instances:
        return 0
    sync_point_instancer(stage, instancer_path, instances, {})
    return len(instances)


def sync_point_instancer(stage, instancer_path, instances, asset_to_proto_index=None):
    """
    Create or update a PointInstancer, reusing existing prototype prims when possible.

    Returns an updated asset_path -> prototype_index map for the instancer.
    """
    if not instances:
        prim = _get_prim(stage, instancer_path)
        if prim and prim.IsValid():
            stage.RemovePrim(Sdf.Path(instancer_path))
        return {}

    asset_to_proto_index = dict(asset_to_proto_index or {})
    prototypes_root = f"{instancer_path}/Prototypes"
    instancer_prim = _get_prim(stage, instancer_path)
    created = instancer_prim is None or not instancer_prim.IsValid()

    if created:
        instancer = UsdGeom.PointInstancer.Define(stage, instancer_path)
    else:
        instancer = UsdGeom.PointInstancer(instancer_prim)

    next_index = (max(asset_to_proto_index.values()) + 1) if asset_to_proto_index else 0
    prototypes_changed = created

    for asset_path in dict.fromkeys(instance["asset_path"] for instance in instances):
        if asset_path in asset_to_proto_index:
            continue
        proto_path = f"{prototypes_root}/proto_{next_index:02d}"
        add_reference_to_stage(asset_path, proto_path)
        _strip_rigid_bodies_recursive(stage, proto_path)
        asset_to_proto_index[asset_path] = next_index
        next_index += 1
        prototypes_changed = True

    if prototypes_changed:
        max_index = max(asset_to_proto_index.values())
        proto_paths = [f"{prototypes_root}/proto_{i:02d}" for i in range(max_index + 1)]
        prototypes_rel = instancer.GetPrototypesRel()
        if prototypes_rel:
            prototypes_rel.SetTargets([Sdf.Path(path) for path in proto_paths])
        else:
            instancer.CreatePrototypesRel().SetTargets([Sdf.Path(path) for path in proto_paths])

    positions = []
    orientations = []
    proto_indices = []
    for instance in instances:
        positions.append(
            Gf.Vec3f(
                float(instance["x"]),
                float(instance["y"]),
                float(instance.get("z", 0.0)),
            )
        )
        orientations.append(_yaw_degrees_to_quat_h(instance["yaw_degrees"]))
        proto_indices.append(asset_to_proto_index[instance["asset_path"]])

    positions_attr = instancer.GetPositionsAttr()
    if positions_attr:
        positions_attr.Set(positions)
    else:
        instancer.CreatePositionsAttr(positions)

    orientations_attr = instancer.GetOrientationsAttr()
    if orientations_attr:
        orientations_attr.Set(orientations)
    else:
        instancer.CreateOrientationsAttr(orientations)

    proto_indices_attr = instancer.GetProtoIndicesAttr()
    if proto_indices_attr:
        proto_indices_attr.Set(proto_indices)
    else:
        instancer.CreateProtoIndicesAttr(proto_indices)

    return asset_to_proto_index


def _get_point_instancer(stage, instancer_path):
    instancer_prim = _get_prim(stage, instancer_path)
    if not instancer_prim or not instancer_prim.IsValid():
        return None
    return UsdGeom.PointInstancer(instancer_prim)


def _ensure_pool_prototypes(stage, instancer_path, pool_assets, asset_to_proto_index):
    prototypes_root = f"{instancer_path}/Prototypes"
    _ensure_empty_prototype(stage, prototypes_root)
    proto_map = dict(asset_to_proto_index or {})
    next_index = (max(proto_map.values()) + 1) if proto_map else 1
    prototypes_changed = not proto_map

    for asset_path in pool_assets:
        if asset_path in proto_map:
            continue
        proto_path = f"{prototypes_root}/proto_{next_index:02d}"
        add_reference_to_stage(asset_path, proto_path)
        _strip_rigid_bodies_recursive(stage, proto_path)
        proto_map[asset_path] = next_index
        next_index += 1
        prototypes_changed = True

    return proto_map, prototypes_changed


def create_point_instancer_shell(stage, instancer_path, instances, pool_assets):
    """
    Create a fixed-slot PointInstancer with every slot pre-positioned and hidden.

    Hidden slots use an empty collision-free prototype so lidar ignores them.
    """
    instancer = UsdGeom.PointInstancer.Define(stage, instancer_path)
    proto_map, prototypes_changed = _ensure_pool_prototypes(
        stage, instancer_path, pool_assets, {}
    )
    prototypes_root = f"{instancer_path}/Prototypes"
    max_index = max(proto_map.values()) if proto_map else EMPTY_PROTO_INDEX
    if prototypes_changed and max_index >= EMPTY_PROTO_INDEX:
        proto_paths = [f"{prototypes_root}/proto_{i:02d}" for i in range(max_index + 1)]
        instancer.CreatePrototypesRel().SetTargets([Sdf.Path(path) for path in proto_paths])

    positions = []
    orientations = []
    for instance in instances:
        positions.append(
            Gf.Vec3f(
                float(instance["x"]),
                float(instance["y"]),
                float(instance.get("z", 0.0)),
            )
        )
        orientations.append(_yaw_degrees_to_quat_h(instance["yaw_degrees"]))

    slot_count = len(instances)
    instancer.CreatePositionsAttr(positions)
    instancer.CreateOrientationsAttr(orientations)
    instancer.CreateProtoIndicesAttr([EMPTY_PROTO_INDEX] * slot_count)
    instancer.CreateInvisibleIdsAttr(list(range(slot_count)))
    return proto_map


def set_point_instancer_proto_indices(stage, instancer_path, proto_indices):
    instancer = _get_point_instancer(stage, instancer_path)
    if not instancer:
        return
    proto_indices_attr = instancer.GetProtoIndicesAttr()
    if proto_indices_attr:
        proto_indices_attr.Set(proto_indices)
    else:
        instancer.CreateProtoIndicesAttr(proto_indices)


def set_point_instancer_invisible_ids(stage, instancer_path, invisible_ids):
    instancer = _get_point_instancer(stage, instancer_path)
    if not instancer:
        return
    invisible_attr = instancer.GetInvisibleIdsAttr()
    if invisible_attr:
        invisible_attr.Set(invisible_ids)
    else:
        instancer.CreateInvisibleIdsAttr(invisible_ids)


def hide_point_instancer_slot(stage, instancer_path, slot_index):
    instancer = _get_point_instancer(stage, instancer_path)
    if not instancer:
        return
    _set_slot_proto_index(instancer, slot_index, EMPTY_PROTO_INDEX)
    invisible_attr = instancer.GetInvisibleIdsAttr()
    invisible = list(invisible_attr.Get() if invisible_attr else [])
    if slot_index in invisible:
        return
    invisible.append(slot_index)
    invisible.sort()
    set_point_instancer_invisible_ids(stage, instancer_path, invisible)


def show_point_instancer_slot(stage, instancer_path, slot_index, asset_proto_index):
    instancer = _get_point_instancer(stage, instancer_path)
    if not instancer:
        return
    _set_slot_proto_index(instancer, slot_index, asset_proto_index)
    invisible_attr = instancer.GetInvisibleIdsAttr()
    invisible = list(invisible_attr.Get() if invisible_attr else [])
    if slot_index not in invisible:
        return
    invisible.remove(slot_index)
    set_point_instancer_invisible_ids(stage, instancer_path, invisible)


def create_dome_light(stage_path="/World/dome_light"):
    stage = stage_utils.get_current_stage(backend="usd")
    light = UsdLux.DomeLight.Define(stage, stage_path)
    light.CreateIntensityAttr(1e3)
    translate_object(stage, stage_path, Gf.Vec3f(0.0, 0.0, 20.0))

def _get_xformable(stage, prim_path):
    prim = _get_prim(stage, prim_path)
    if not prim or not prim.IsValid():
        return None
    if prim.IsA(UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(prim)
        return mesh
    return UsdGeom.Xformable(prim)


def translate_object(stage, prim_path, translate):
    xformable = _get_xformable(stage, prim_path)
    if xformable:
        physicsUtils.set_or_add_translate_op(xformable, translate=translate)

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
    xformable = _get_xformable(stage, prim_path)
    if xformable:
        physicsUtils.set_or_add_orient_op(xformable, rotation_quaternion)


async def disable_gravity(stage=None):
    stage = stage or stage_utils.get_current_stage(backend="usd")
    physics_scenes = [
        UsdPhysics.Scene(prim)
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.Scene)
    ]
    if not physics_scenes:
        physics_scenes = [UsdPhysics.Scene.Define(stage, "/PhysicsScene")]

    for scene in physics_scenes:
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(0.0)
    return


def _bbox_to_obb_components(stage, imageable):
    """Return (center, axes[3,3], half_extents[3]) for the prim's local bound in world space.
    axes rows are unit vectors in world frame.
    """
    import numpy as np
    # Build a bbox cache with common purposes
    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes, useExtentsHint=True)
    # Local bound gives an axis-aligned box in the prim's local space
    local_bbox = cache.ComputeLocalBound(imageable.GetPrim())
    rng = local_bbox.GetBox() if hasattr(local_bbox, "GetBox") else local_bbox.GetRange()
    pmin = rng.GetMin()
    pmax = rng.GetMax()
    local_center = Gf.Vec3d((pmin[0] + pmax[0]) * 0.5, (pmin[1] + pmax[1]) * 0.5, (pmin[2] + pmax[2]) * 0.5)
    local_half = Gf.Vec3d((pmax[0] - pmin[0]) * 0.5, (pmax[1] - pmin[1]) * 0.5, (pmax[2] - pmin[2]) * 0.5)

    # World transform of the prim
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    world_mat = xform_cache.GetLocalToWorldTransform(imageable.GetPrim())

    # Center in world
    center_world = world_mat.Transform(local_center)

    # Axes are the columns of the upper-left 3x3 of the matrix
    r0 = Gf.Vec3d(world_mat[0][0], world_mat[1][0], world_mat[2][0])
    r1 = Gf.Vec3d(world_mat[0][1], world_mat[1][1], world_mat[2][1])
    r2 = Gf.Vec3d(world_mat[0][2], world_mat[1][2], world_mat[2][2])
    axes = [r0, r1, r2]
    # Account for non-uniform scale by using column norms
    scales = [r.GetLength() for r in axes]
    axes_unit = [ (axes[i] / scales[i]) if scales[i] > 1e-12 else Gf.Vec3d(0.0,0.0,0.0) for i in range(3) ]
    half_ext = [ local_half[i] * scales[i] for i in range(3) ]

    axes_np = np.array([[axes_unit[0][0], axes_unit[0][1], axes_unit[0][2]],
                        [axes_unit[1][0], axes_unit[1][1], axes_unit[1][2]],
                        [axes_unit[2][0], axes_unit[2][1], axes_unit[2][2]]], dtype=float)
    center_np = np.array([center_world[0], center_world[1], center_world[2]], dtype=float)
    half_np = np.array([half_ext[0], half_ext[1], half_ext[2]], dtype=float)
    return center_np, axes_np, half_np


def _obb_overlap_sat(c1, A1, e1, c2, A2, e2):
    """Separating Axis Theorem for two OBBs.
    c*: (3,), A*: (3,3) rows are unit axes, e*: (3,).
    """
    import numpy as np
    EPS = 1e-8
    # Rotation from box2 into box1 coordinates
    R = A1 @ A2.T
    absR = np.abs(R) + EPS
    # Translation in box1 frame
    t = A1 @ (c2 - c1)

    # Test axes L = A1[i]
    for i in range(3):
        ra = e1[i]
        rb = e2[0]*absR[i,0] + e2[1]*absR[i,1] + e2[2]*absR[i,2]
        if abs(t[i]) > ra + rb:
            return False

    # Test axes L = A2[i]
    for i in range(3):
        ra = e1[0]*absR[0,i] + e1[1]*absR[1,i] + e1[2]*absR[2,i]
        rb = e2[i]
        if abs(t[0]*R[0,i] + t[1]*R[1,i] + t[2]*R[2,i]) > ra + rb:
            return False

    # Test cross products A1[i] x A2[j]
    for i in range(3):
        for j in range(3):
            ra = e1[(i+1)%3]*absR[(i+2)%3,j] + e1[(i+2)%3]*absR[(i+1)%3,j]
            rb = e2[(j+1)%3]*absR[i,(j+2)%3] + e2[(j+2)%3]*absR[i,(j+1)%3]
            lhs = abs(t[(i+2)%3]*R[(i+1)%3,j] - t[(i+1)%3]*R[(i+2)%3,j])
            if lhs > ra + rb:
                return False
    return True


def prims_overlap_obb(prim_path_a: str, prim_path_b: str) -> bool:
    """Return True if two prims' oriented bounding boxes overlap (world frame)."""
    stage = omni.usd.get_context().get_stage()
    prim_a = _get_prim(stage, prim_path_a)
    prim_b = _get_prim(stage, prim_path_b)
    if not (prim_a and prim_a.IsValid() and prim_b and prim_b.IsValid()):
        return False

    def _first_mesh_descendant(prim):
        """Return the prim itself if it has geometry; otherwise first Mesh descendant (depth-first)."""
        if UsdGeom.Mesh(prim):
            return prim
        stack = list(prim.GetChildren())
        while stack:
            p = stack.pop(0)
            if UsdGeom.Mesh(p):
                return p
            stack.extend(p.GetChildren())
        return prim  # fallback

    prim_a_geom = _first_mesh_descendant(prim_a)
    prim_b_geom = _first_mesh_descendant(prim_b)

    img_a = UsdGeom.Imageable(prim_a_geom)
    img_b = UsdGeom.Imageable(prim_b_geom)
    c1, A1, e1 = _bbox_to_obb_components(stage, img_a)
    c2, A2, e2 = _bbox_to_obb_components(stage, img_b)
    return _obb_overlap_sat(c1, A1, e1, c2, A2, e2)

def get_sim_time(sim_context):
    return sim_context.current_time