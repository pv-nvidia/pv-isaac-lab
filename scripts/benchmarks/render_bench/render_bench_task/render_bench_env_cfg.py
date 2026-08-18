# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Env config for the Franka-Cabinet render-only benchmark. Adds a TiledCamera
with ``MultiBackendRendererCfg`` + the standard ``simple_shading_*`` preset
infrastructure so the existing benchmark/visualize tooling drives this scene
the same way it drives ShadowHand Vision.

Camera convention notes:
    ``convention="world"`` is ``+X forward, +Z up``. The Cabinet scene uses a
    front-facing perspective camera; a top-down camera builder is retained as
    the neutral default for the data-type preset shell.
"""

from __future__ import annotations

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

# ----------------------------------------------------------------------
# Shared physics + camera preset infrastructure
# ----------------------------------------------------------------------


@configclass
class _RenderBenchPhysicsCfg(PresetCfg):
    """Physics backend presets — pick via ``presets=newton_mjwarp`` (default)
    or ``presets=physx`` (requires Isaac Sim)."""

    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(solver="newton", integrator="implicitfast", njmax=200, nconmax=70),
        num_substeps=2,
    )
    physx: PhysxCfg = PhysxCfg()
    default = newton_mjwarp


# Top-down quaternion: 90° around +Y → camera-forward +X→-Z (look straight down).
_TOP_DOWN_ROT = (0.0, 0.7071, 0.0, 0.7071)


def _top_down_camera(
    pos: tuple[float, float, float],
    focal_length: float = 15.0,
    width: int = 256,
    height: int = 256,
) -> CameraCfg:
    """Build a top-down camera at ``pos`` with the given focal length.
    Smaller focal_length = wider FOV. Default 256×256 per tile."""
    return CameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=CameraCfg.OffsetCfg(pos=pos, rot=_TOP_DOWN_ROT, convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 50.0),
        ),
        width=width,
        height=height,
        renderer_cfg=MultiBackendRendererCfg(),
    )


def _perspective_camera(
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float],
    focal_length: float = 18.0,
    width: int = 256,
    height: int = 256,
    warp_enable_shadows: bool = False,
) -> CameraCfg:
    """Build a perspective camera with explicit quaternion orientation.

    ``rot`` is ``(qx, qy, qz, qw)`` in ``convention="world"`` (+X camera-forward).
    Pick ``pos`` and ``rot`` to aim the camera at the workspace from a 3/4 angle."""
    return CameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 50.0),
        ),
        width=width,
        height=height,
        renderer_cfg=MultiBackendRendererCfg(
            newton_renderer=NewtonWarpRendererCfg(enable_shadows=warp_enable_shadows),
        ),
    )


def _tiled_with_camera(camera_base: CameraCfg) -> _RenderBenchTiledCameraCfg:
    """Wrap a CameraCfg in the data-type PresetCfg shell so all standard
    ``simple_shading_*`` / ``rgb`` / ``albedo`` / ``depth`` presets are available."""
    return _RenderBenchTiledCameraCfg(
        default=camera_base.replace(data_types=["rgb"]),
        rgb=camera_base.replace(data_types=["rgb"]),
        albedo=camera_base.replace(data_types=["albedo"]),
        depth=camera_base.replace(data_types=["depth"]),
        simple_shading_constant_diffuse=camera_base.replace(data_types=["simple_shading_constant_diffuse"]),
        simple_shading_diffuse_mdl=camera_base.replace(data_types=["simple_shading_diffuse_mdl"]),
        simple_shading_full_mdl=camera_base.replace(data_types=["simple_shading_full_mdl"]),
    )


# Default top-down camera (overridden per-variant for framing).
_BASE_CAMERA = _top_down_camera(pos=(0.5, 0.0, 3.5), focal_length=15.0)


@configclass
class _RenderBenchTiledCameraCfg(PresetCfg):
    """Data-type presets — pick via ``presets=simple_shading_full_mdl``, etc."""

    default: CameraCfg = _BASE_CAMERA.replace(data_types=["rgb"])
    rgb: CameraCfg = _BASE_CAMERA.replace(data_types=["rgb"])
    albedo: CameraCfg = _BASE_CAMERA.replace(data_types=["albedo"])
    depth: CameraCfg = _BASE_CAMERA.replace(data_types=["depth"])
    simple_shading_constant_diffuse: CameraCfg = _BASE_CAMERA.replace(data_types=["simple_shading_constant_diffuse"])
    simple_shading_diffuse_mdl: CameraCfg = _BASE_CAMERA.replace(data_types=["simple_shading_diffuse_mdl"])
    simple_shading_full_mdl: CameraCfg = _BASE_CAMERA.replace(data_types=["simple_shading_full_mdl"])


# ----------------------------------------------------------------------
# Base env cfg.
# ----------------------------------------------------------------------


@configclass
class RenderBenchBaseEnvCfg(DirectRLEnvCfg):
    decimation: int = 2
    episode_length_s: float = 60.0

    action_space: int = 1
    observation_space: int = 1
    state_space: int = 0

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=2, physics=_RenderBenchPhysicsCfg())
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=4.0, replicate_physics=True)
    tiled_camera: _RenderBenchTiledCameraCfg = _RenderBenchTiledCameraCfg()

    articulations: dict = {}
    rigid_objects: dict = {}
    # Override default ground-plane z (top surface of the flat ground).
    ground_plane_z: float = 0.0
    # When True, replace the USD GroundPlaneCfg (which Warp's renderer doesn't
    # see — it only knows about simulation meshes) with a large flat
    # CuboidCfg routed through the standard RigidObject path. Both OVRTX
    # and Warp render the same geometry, giving an apple-to-apple
    # comparison (without this, Warp's background is just the clear color,
    # so primary-ray-miss + shadow-ray costs differ).
    use_flat_ground: bool = False
    flat_ground_size: tuple = (50.0, 50.0)  # XY extent (m); single per-env cuboid
    flat_ground_thickness: float = 0.1
    flat_ground_color: tuple = (0.5, 0.5, 0.5)
    dome_light_intensity: float = 2000.0
    # Optional override: if set, use this light cfg instead of the default
    # DomeLight. Useful for DistantLight (directional sun-like) scenes.
    light_cfg: sim_utils.LightCfg | None = None
    light_orientation: tuple = (0.0, 0.0, 0.0, 1.0)

    write_image_to_file: bool = False

    # Per-joint sinusoidal animation amplitude (radians). 0 = no animation;
    # each articulation's joints oscillate around their default pose with a
    # random per-(env, joint) phase, clamped to the joint's soft limits.
    joint_animation_amplitude: float = 0.6
    joint_animation_freq_hz: float = 0.5


# ----------------------------------------------------------------------
# Franka + Sektion cabinet scene (mirrors Isaac-Franka-Cabinet-Direct-v0).
# ----------------------------------------------------------------------


@configclass
class RenderBenchFrankaCabinetEnvCfg(RenderBenchBaseEnvCfg):
    """Franka Panda + Sektion cabinet (4 articulated joints: 2 drawers, 2
    doors). Sinusoidal animation drives all joints so the rendered frames
    show drawers sliding in/out and doors swinging — the cabinet adds
    significant articulated-geometry complexity to the scene.

    Poses match the canonical ``Isaac-Franka-Cabinet-Direct-v0`` task:
      Franka: pos=(1.0, 0, 0), rot=180° around Y (faces -X toward cabinet)
      Cabinet: pos=(0.0, 0, 0.4), rot=180° around Z (opens toward -X)
    """

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=3.0, replicate_physics=True)

    # Front view with no yaw / no roll — pitch-only camera. Sits in front of
    # the cabinet (+X side, the drawer-facing side), elevated, pitched ~35°
    # down so the workspace reads as a slight bird's-eye. Tighter focal so
    # the Franka + cabinet fill most of the frame.
    #   pos        = (2.0, 0.0, 1.5)
    #   look-at    = (0.4, 0, 0.4)  →  forward = (-0.824, 0, -0.566)
    #   pitch      = ~34.5° down,  yaw = 0,  roll = 0
    # Quaternion (qxyzw) = (-0.296, 0, 0.955, 0) — qw=0, qx and qz only,
    # corresponds to a 180° rotation around the (-X, 0, +Z) axis.
    tiled_camera: _RenderBenchTiledCameraCfg = _tiled_with_camera(
        _perspective_camera(
            pos=(2.0, 0.0, 1.5),
            rot=(-0.296, 0.0, 0.955, 0.0),
            focal_length=24.0,
            warp_enable_shadows=True,
        )
    )

    # USD DistantLight matching Warp's hardcoded directional
    # ``(-0.57735, 0.57735, -0.57735)``. Orientation is the quaternion that
    # rotates USD ``DistantLight``'s default ``-Z`` to that direction.
    light_cfg: sim_utils.LightCfg | None = sim_utils.DistantLightCfg(
        intensity=200.0,
        exposure=0.0,
        angle=0.0,
        color=(1.0, 1.0, 1.0),
        normalize=True,
    )
    light_orientation: tuple = (0.3251, 0.3251, 0.0, 0.8881)

    articulations: dict = {
        # Franka mounted at +X facing -X (toward cabinet). HighPD config so
        # the joints track sinusoidal commands smoothly.
        "robot": FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="/World/envs/env_.*/Robot",
            init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(
                pos=(1.0, 0.0, 0.0),
                rot=(0.0, 0.0, 1.0, 0.0),  # 180° around Y
            ),
        ),
        # Sektion cabinet: 4 articulated joints (door_left, door_right,
        # drawer_top, drawer_bottom). Loaded as ArticulationCfg so it clones
        # reliably to every env via the standard IsaacLab path.
        "cabinet": ArticulationCfg(
            prim_path="/World/envs/env_.*/Cabinet",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
                activate_contact_sensors=False,
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.4),
                rot=(0.0, 0.0, 0.0, 1.0),  # 180° around Z
                joint_pos={
                    "door_left_joint": 0.0,
                    "door_right_joint": 0.0,
                    "drawer_bottom_joint": 0.0,
                    "drawer_top_joint": 0.0,
                },
            ),
            actuators={
                "drawers": ImplicitActuatorCfg(
                    joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                    effort_limit_sim=87.0,
                    stiffness=10.0,
                    damping=1.0,
                ),
                "doors": ImplicitActuatorCfg(
                    joint_names_expr=["door_left_joint", "door_right_joint"],
                    effort_limit_sim=87.0,
                    stiffness=10.0,
                    damping=2.5,
                ),
            },
        ),
    }
    rigid_objects: dict = {}

    # Slightly slower / smaller-amplitude joint animation than the default so
    # the drawers and doors move at a clearly-visible-but-not-frantic rate.
    joint_animation_amplitude: float = 0.4
    joint_animation_freq_hz: float = 0.35

    # Use the flat-color CuboidCfg ground (visible in BOTH OVRTX and Warp)
    # instead of the USD GroundPlaneCfg (which only OVRTX sees). Necessary
    # for fair OVRTX-vs-Warp profile comparison — without this, Warp gets
    # extra primary-ray-misses + skipped shadow rays.
    use_flat_ground: bool = True
