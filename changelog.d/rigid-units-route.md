### Added

- **`[performance] rigid_units = "AUTO" | "ON" | "OFF"`** (D[rigid-units-route], slice S4b of D417, #914). `ON` evaluates the pairs of the rigid kinds of D403 by units, over segmented lists, and refuses with the reason a program where that cannot apply; `OFF` is the general route; `AUTO`, the default, takes the general route until its rule is measured. `mdir check`, the log, and the manifest state the route and, for each kind, whether it is taken and why.

### Removed

- **`MDIR_RIGID_UNITS_TEST`**, the route for tests of D421, replaced by the key, with no alias.
