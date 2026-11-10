* Added :class:`~isaaclab.sim.UnsupportedSceneError` for a scene that the active runtime cannot simulate or
  render. A visualizer step of :class:`~isaaclab.sim.SimulationContext` did not let an error stop the
  simulation: it logged the error and closed the visualizer. The step now lets this error class through.
