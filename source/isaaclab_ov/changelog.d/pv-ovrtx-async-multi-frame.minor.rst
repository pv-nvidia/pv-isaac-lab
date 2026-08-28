Added
^^^^^

* Added multi-frame asynchronous rendering latency on the OVRTX legacy scene-ownership path:
  :attr:`~isaaclab.renderers.RendererCfg.async_rendering` frame counts above one keep that many
  renders in flight. The ovstage path still sustains at most one frame — it retains a single
  committed snapshot that renders read in place — so larger values are clamped to ``1`` there with
  a warning.
