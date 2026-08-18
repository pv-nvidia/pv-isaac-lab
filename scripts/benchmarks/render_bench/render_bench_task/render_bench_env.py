# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct env for render-only benchmarking.

Loops the cfg's ``articulations`` and ``rigid_objects`` dicts to populate the
scene, then drives every articulation joint with a deterministic sinusoidal
animation so the rendered frames show motion (the Franka arm sweeping and the
cabinet drawers/doors sliding). The env produces no rewards — it exists purely
to feed geometry to the OVRTX / Newton-Warp renderers under a fixed camera.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import Camera, save_images_to_file

if TYPE_CHECKING:
    from .render_bench_env_cfg import RenderBenchBaseEnvCfg


class RenderBenchEnv(DirectRLEnv):
    cfg: RenderBenchBaseEnvCfg

    # --- scene construction --------------------------------------------------

    def _setup_scene(self):
        self._articulations: dict[str, Articulation] = {}
        for name, art_cfg in (self.cfg.articulations or {}).items():
            self._articulations[name] = Articulation(art_cfg)

        self._rigid_objects: dict[str, RigidObject] = {}
        for name, ro_cfg in (self.cfg.rigid_objects or {}).items():
            self._rigid_objects[name] = RigidObject(ro_cfg)

        self._tiled_camera = Camera(self.cfg.tiled_camera)

        if self.cfg.use_flat_ground:
            # Flat-color ground via CuboidCfg routed through RigidObject so
            # both OVRTX and Warp see the same geometry (apple-to-apple
            # comparison). The USD GroundPlaneCfg path produces a USD-only
            # plane that Warp's render_megakernel doesn't see.
            sx, sy = self.cfg.flat_ground_size
            sz = self.cfg.flat_ground_thickness
            top_at = self.cfg.ground_plane_z
            ground_rigid_cfg = RigidObjectCfg(
                prim_path="/World/envs/env_.*/Floor",
                spawn=sim_utils.CuboidCfg(
                    size=(sx, sy, sz),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=self.cfg.flat_ground_color, metallic=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, top_at - sz / 2.0)),
            )
            self._rigid_objects["__floor"] = RigidObject(ground_rigid_cfg)
        else:
            # USD-loaded grid floor. ``color=(1,1,1)`` overrides the cfg
            # default (0,0,0) so the USD's base diffuse texture color shows
            # through under Newton; Warp's renderer doesn't see this plane at
            # all (clear color shows through instead).
            ground_cfg = sim_utils.GroundPlaneCfg(color=(1.0, 1.0, 1.0))
            ground_cfg.func(
                "/World/defaultGroundPlane",
                ground_cfg,
                translation=(0.0, 0.0, self.cfg.ground_plane_z),
            )

        # Always spawn a dim ambient DomeLight so non-lit surfaces don't go
        # pure black under Newton's renderer (which lacks Kit's tone mapping
        # and ambient defaults).
        dome_cfg = sim_utils.DomeLightCfg(intensity=self.cfg.dome_light_intensity, color=(0.75, 0.75, 0.75))
        dome_cfg.func("/World/Light", dome_cfg)
        # Optional directional light (e.g. DistantLight) on top of the ambient.
        if self.cfg.light_cfg is not None:
            self.cfg.light_cfg.func(
                "/World/LightDirectional",
                self.cfg.light_cfg,
                orientation=self.cfg.light_orientation,
            )

        # Clone env_0 -> env_1..N. Current develop replaced
        # scene.clone_environments() with the cloner.clone_plan_from_env_0 +
        # replicate pattern (see isaaclab_tasks core cartpole_direct_env).
        src, dest = "/World/envs/env_0", "/World/envs/env_{}"
        pos = cloner.grid_transforms(self.scene.num_envs, self.scene.cfg.env_spacing, device=self.device)[0]
        plan = cloner.clone_plan_from_env_0(src, dest, self.scene.num_envs, self.device, pos)
        cloner.replicate(plan, stage=self.scene.stage)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        for name, art in self._articulations.items():
            self.scene.articulations[name] = art
        for name, ro in self._rigid_objects.items():
            self.scene.rigid_objects[name] = ro
        self.scene.sensors["tiled_camera"] = self._tiled_camera

    # --- physics step --------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._lazy_init_motion()
        if self.cfg.joint_animation_amplitude > 0.0:
            self._sin_animate()

    def _apply_action(self) -> None:
        pass

    def _lazy_init_motion(self):
        if hasattr(self, "_motion_init_done"):
            return
        self._motion_init_done = True

        # Sinusoidal phases per (env, joint) for every articulation. Use a
        # DETERMINISTIC hash of (env_idx, joint_idx) so the simulation is
        # reproducible across runs and rendering backends — otherwise the
        # default torch RNG state varies process-to-process and OVRTX vs.
        # Warp runs end up at different joint poses at the same frame even
        # though the physics solver itself is deterministic.
        self._anim_phases: dict[str, torch.Tensor] = {}
        for name, art in self._articulations.items():
            default_pos = art.data.default_joint_pos.torch
            n_envs, n_joints = default_pos.shape
            env_idx = torch.arange(n_envs, device=self.device, dtype=default_pos.dtype).unsqueeze(1)
            joint_idx = torch.arange(n_joints, device=self.device, dtype=default_pos.dtype).unsqueeze(0)
            # Coprime primes spread phases pseudo-uniformly across (env, joint).
            phase_int = (env_idx * 7919.0 + joint_idx * 6553.0) % 10007.0
            self._anim_phases[name] = phase_int * (2.0 * math.pi / 10007.0)
        self._anim_t = 0.0

    # --- Sinusoidal joint animation -----------------------------------------

    def _sin_animate(self):
        self._anim_t += self.cfg.sim.dt * self.cfg.decimation
        omega = 2.0 * math.pi * self.cfg.joint_animation_freq_hz
        for name, art in self._articulations.items():
            default_pos = art.data.default_joint_pos.torch
            phase = self._anim_phases[name]
            target = default_pos + self.cfg.joint_animation_amplitude * torch.sin(omega * self._anim_t + phase)
            soft_limits = art.data.soft_joint_pos_limits.torch
            target = torch.clamp(target, soft_limits[..., 0], soft_limits[..., 1])
            art.set_joint_position_target_index(target=target)

    # --- DirectRLEnv plumbing ------------------------------------------------

    def _get_observations(self) -> dict:
        data_type = self.cfg.tiled_camera.data_types[0]
        out = self._tiled_camera.data.output
        if isinstance(out, dict) and data_type in out and self.cfg.write_image_to_file:
            try:
                cam = out[data_type]
                if not torch.is_tensor(cam):
                    cam = torch.from_dlpack(cam)
                img = cam.float()
                if img.max() > 1.5:
                    img = img / 255.0
                save_images_to_file(img[:, ..., :3], f"render_bench_{data_type}.png")
            except Exception as e:
                print(f"[render_bench] write_image_to_file failed: {e}")
        return {"policy": torch.zeros((self.num_envs, 1), device=self.device)}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self):
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros_like(time_out), time_out
