### Added

- **`[performance] rigid_units = "AUTO" | "ON" | "OFF"`** (D424, slice S4b of D417, #914). `ON` evaluates the pairs of the rigid kinds of D403 by units over segmented lists: the two most numerous kinds of two or more particles whose molecules exclude all their own pairs and none outside them. It refuses, with the reason, a program where that cannot apply (the CPU target, a precision other than `MIXED`, the deterministic mode, a minimization, an open cell, `[free_energy]`, LJPME, a PME kernel other than `GAUSSIAN`, tunables and `observe`, no kind it takes, a covering reach beyond half the least face width of the cell, and frames that write forces where a taken kind has a site, #916). `OFF` is the general route; `AUTO`, the default, takes the general route until its rule is measured, so no program changes without the key. A covering radius is widened to the positions the run begins with where they lie beyond it. `mdir check` (text and JSON), the log, the manifest (`rigid_units_*`), and Python's `Program.plan` state the route and, for each kind, whether it is taken and why. In Python, `Execution.rigid_units` (`mdir.RigidUnits`). With `ON`, #795's six systems run at 0.73 to 0.90 of `main`'s wall on an RTX 4090.

### Removed

- **`MDIR_RIGID_UNITS_TEST`**, the route for tests of D421, replaced by the key, with no alias.
