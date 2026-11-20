* Changed the split search of the OVRTX renderer to compute the split from the ``ovrtx.RenderFailedError`` of a
  failed first frame instead of halving the environments per render product. The search needs fewer failed frames,
  and a note on the error explains why the search stops when a split cannot help.
