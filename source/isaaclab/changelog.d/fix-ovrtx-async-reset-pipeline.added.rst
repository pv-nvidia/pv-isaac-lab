* Added the ``blank_first_frame`` option to ``isaaclab.utils.images.CameraFrameStack`` and the camera
  image observation terms, such as :class:`~isaaclab.envs.mdp.observations.image_rgb`. It gives the first
  frame after a reset zero pixels before normalization, so that delayed renderers, synchronous renderers,
  and real cameras give the policy the same first frame.
