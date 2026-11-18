### Added

- **The writer of a list of rigid units** (D422, slice S4a of D417, #905). `md_exec.write_unit_frames` runs after the last op of a step that writes the positions. It is one kernel, a thread for each member of a unit, and it:
  - packs the f32 frames that the loops over the list read;
  - checks every place against its unit's covering radius by the convex rule, and stops the run where one lies beyond it;
  - tests the units' reference points against the least half-margin of the segments, in f32 with a counted margin of rounding.

  Its results are flags that the host reads at the refresh, after it has issued the reciprocal sum. The per-step readback of the refresh's own test is gone: on an RTX 4090, ubiquitin in OPC idles the device 17.7 µs a step instead of 37.4. The route stays for tests only (`MDIR_RIGID_UNITS_TEST`) until S4b, so no run changes.

### Fixed

- **D421's step profile under NPT.** The figures a step were 1.263 times too large in all three columns, because the profile counted the steps by a loop that the barostat's steps skip. They are now counted by the gather of PME.
