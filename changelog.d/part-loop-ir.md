### Changed

- Internal: the builder's loop over the parts of a program of segments
  carries its values in IR, and the renaming of values in its text
  (`renameValue`) is gone; no program changes (#568).
