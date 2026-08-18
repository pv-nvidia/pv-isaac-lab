# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vendored copy of wahuang-rtx/IsaacLab render-bench Franka-Cabinet scene.

Source: https://github.com/wahuang-rtx/IsaacLab/tree/wahuang/render-bench-wip
        source/isaaclab_tasks/isaaclab_tasks/direct/render_bench/  (commit 18b17b80)

Registered under a distinct id (``Repro-RenderBench-Franka-Cabinet-v0``) so it
never clashes with an in-tree ``Isaac-RenderBench-Franka-Cabinet-v0`` if one is
present.

``_setup_scene`` was ported to current develop: the removed
``scene.clone_environments()`` is replaced by the
``cloner.clone_plan_from_env_0`` + ``cloner.replicate`` pattern (see the core
``cartpole_direct_env`` task).
"""

import gymnasium as gym

gym.register(
    id="Repro-RenderBench-Franka-Cabinet-v0",
    entry_point=f"{__name__}.render_bench_env:RenderBenchEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.render_bench_env_cfg:RenderBenchFrankaCabinetEnvCfg"},
)
