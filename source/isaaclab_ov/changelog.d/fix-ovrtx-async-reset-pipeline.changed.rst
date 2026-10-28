* Changed asynchronous OVRTX rendering (:attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.async_rendering`)
  to keep its pipeline on environment resets and at start-up. Waiting for a fresh image on each reset
  made every RL step wait for its own capture, so asynchronous rendering gave no speedup. Reset
  environments now show their previous episode for one step, and camera outputs stay zero until the
  second capture.
