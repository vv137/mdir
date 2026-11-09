### Added

- **`mdir check` warns when the force of PME's direct sum jumps by more than 1 kJ/mol/nm at the cutoff**, as a `beta` chosen for a longer cutoff or a loose `tolerance` makes it; it names the jump and the pairs a frame that mixed and double precision may cut differently, and runs are unchanged (D[pme-cutoff-jump-warning], #793).
