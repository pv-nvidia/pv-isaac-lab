* **Breaking:** Changed Kit RTX rendering of simulation-authored ``Points`` prims, such as Newton MPM particle
  clouds, to use GPU Fabric on a CUDA simulation device. The particle positions were copied to CPU Fabric on
  every visual update, which synchronized the host with the simulation. There is no CPU fallback for CUDA
  particles. Kit builds without the GPU ``Points`` fix (OMPE-106051) render such points incorrectly: a cloud
  with one point can stay frozen, and a cloud can lose its color. Migration: use an Isaac Sim release whose
  Kit contains the fix. Isaac Sim 6.2.0.0 does not contain it. CPU simulation devices and scenes without
  ``Points`` prims are unaffected.
