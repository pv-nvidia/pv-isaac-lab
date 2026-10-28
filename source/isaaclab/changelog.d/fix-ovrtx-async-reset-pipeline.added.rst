* Added the ``blank_first_frame`` option to ``isaaclab.utils.images.CameraFrameStack`` and the camera
  image observation terms, such as :class:`~isaaclab.envs.mdp.observations.image_rgb`. It replaces the first
  frame after a reset with zeros, so that delayed renderers, synchronous renderers, and real cameras give
  the policy the same first frame.
