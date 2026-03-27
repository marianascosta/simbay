# ==========================================
# 1. IMPORTS
# ==========================================
import logging
import os

import mujoco
import numpy as np

from src.robots import MujocoRobot
from src.utils.tracing import set_span_attributes
from src.utils.tracing import trace_call

from .constants import DEFAULT_OBJECT_PROPS
from .constants import FRANKA_HOME_QPOS

logger = logging.getLogger("simbay.mujoco_utils")


# ==========================================
# 2. CORE MUJOCO LOADER
# ==========================================
@trace_call("simbay.mujoco_utils", "mujoco.load_model")
def load_mujoco_model(xml_path):
    """
    Reads the XML from the path and builds a fresh, independent C++ model.
    The OS caches this file in RAM automatically, making it safe and lightning-fast
    to call this 100 times in a loop for the Particle Filter.
    """
    set_span_attributes({"mujoco.xml_path": str(xml_path)})
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)  # type: ignore
        data = mujoco.MjData(model)  # type: ignore
    except ValueError as e:
        raise ValueError(f"Error loading MuJoCo XML from {xml_path}: {e}")

    # Configure initial state
    data.qpos[:7] = FRANKA_HOME_QPOS
    data.ctrl[:7] = FRANKA_HOME_QPOS
    mujoco.mj_forward(model, data)  # type: ignore

    set_span_attributes({"mujoco.nq": int(model.nq), "mujoco.nv": int(model.nv)})
    return model, data


# ==========================================
# 3. PHYSICS MODIFIER
# ==========================================
@trace_call("simbay.mujoco_utils", "mujoco.modify_object_properties")
def modify_object_properties(model, data, body_name, props):
    """
    Modifies a MuJoCo body (size, mass, friction, position).
    """
    set_span_attributes(
        {
            "mujoco.body_name": str(body_name),
            "mujoco.body_mass": float(props["mass"]),
        }
    )
    try:
        body_id = model.body(body_name).id
        geom_id = model.body_geomadr[body_id]
    except KeyError:
        logger.error(
            {
                "event": "mujoco_body_not_found",
                "msg": f"Could not find MuJoCo body '{body_name}'.",
                "body_name": body_name,
            }
        )
        return

    # A. Save State
    saved_qpos = data.qpos.copy()
    saved_qvel = data.qvel.copy()
    saved_ctrl = data.ctrl.copy()

    # B. Apply Geometry / Physics
    type_map = {
        "box": mujoco.mjtGeom.mjGEOM_BOX,  # type: ignore
        "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,  # type: ignore
        "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,  # type: ignore
        "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,  # type: ignore
    }

    geom_type = props["type"]
    if geom_type in type_map:
        model.geom_type[geom_id] = type_map[geom_type]

    model.geom_size[geom_id] = np.array(props["size"])
    model.body_mass[body_id] = props["mass"]
    model.geom_friction[geom_id] = np.array(props["friction"])

    mujoco.mj_setConst(model, data)  # type: ignore

    # C. Restore State
    data.qpos[:] = saved_qpos
    data.qvel[:] = saved_qvel
    data.ctrl[:] = saved_ctrl

    # D. Apply Position
    jnt_adr = model.body_jntadr[body_id]
    if jnt_adr != -1:
        qpos_adr = model.jnt_qposadr[jnt_adr]
        data.qpos[qpos_adr : qpos_adr + 3] = np.array(props["pos"])

        dof_adr = model.jnt_dofadr[jnt_adr]
        data.qvel[dof_adr : dof_adr + 6] = 0.0

    mujoco.mj_forward(model, data)  # type: ignore


@trace_call("simbay.mujoco_utils", "mujoco.prepare_model_for_warp")
def prepare_model_for_warp(model) -> None:
    """Patch MuJoCo model for the simplified batched Warp collision path."""
    model.opt.noslip_iterations = 0
    model.geom_margin[:] = 0.0
    model.geom_gap[:] = 0.0

    # Preserve the previous batched-backend collision model: only the floor,
    # fingertip-pad boxes, and the object participate in contact. Mesh and
    # sphere geoms are disabled to keep Warp aligned with the prior runtime
    # behavior and force signal characteristics.
    for geom_index in range(model.ngeom):
        geom_type = model.geom_type[geom_index]
        if geom_type in (mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_SPHERE):
            model.geom_contype[geom_index] = 0
            model.geom_conaffinity[geom_index] = 0


# ==========================================
# 4. FACTORY
# ==========================================
@trace_call("simbay.mujoco_utils", "mujoco.initialize_env")
def initialize_mujoco_env(object_properties=DEFAULT_OBJECT_PROPS):
    set_span_attributes(
        {
            "mujoco.object_mass": float(object_properties["mass"]),
            "mujoco.object_type": str(object_properties["type"]),
        }
    )
    xml_path = os.path.join("assets", "franka_fr3_v2", "scene.xml")

    model, data = load_mujoco_model(xml_path)
    modify_object_properties(model, data, "object", object_properties)

    return MujocoRobot(model, data)
