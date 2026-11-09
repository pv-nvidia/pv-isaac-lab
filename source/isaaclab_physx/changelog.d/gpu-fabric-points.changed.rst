* **Breaking:** Changed Kit RTX rendering of simulation-authored ``Points`` prims, such as Newton MPM particle
  clouds, to use GPU Fabric on a CUDA simulation device. The particle positions were copied to CPU Fabric on
  every visual update, which synchronized the host with the simulation. There is no CPU fallback for CUDA
  particles: a Kit build older than ``isaaclab_physx.renderers.fabric.MIN_GPU_POINTS_KIT_BUILD`` raises a
  ``RuntimeError`` when it binds ``Points`` on a CUDA device, because such builds render GPU Fabric ``Points``
  incorrectly (OMPE-106051). Migration: use the Isaac Sim release that Isaac Lab pins, or a newer one whose
  Kit contains the fix. CPU simulation devices and scenes without ``Points`` prims are unaffected.
