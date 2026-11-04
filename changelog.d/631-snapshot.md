### Fixed

- **Internal**: the equivalence check of the builder emits each case of both builds from one reading of its control file, so a control rewritten during the check is not reported as a difference, and a test shows it (#631).
