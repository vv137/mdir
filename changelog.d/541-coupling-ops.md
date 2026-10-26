### Added

- The ops of the thermostats and of the removal of the motion of the
  center of mass in `dyn` (`dyn.momentum`, `dyn.kinetic`,
  `dyn.half_step_excess`, `dyn.velocity_rescaling`, `dyn.nose_hoover`,
  `dyn.shift`, `dyn.scale`, `dyn.noise`, `dyn.to_bath`), the claims of
  `dyn.program` checked against its body with the new `no_global_sums`, and
  the pass `dyn-expand-couplings`, which expands the ops into the text the
  driver writes. The driver does not write them yet; no program changes
  (#541, step 2a; D326, docs/dynamics.md, Section 2.2).
