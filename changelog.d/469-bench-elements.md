### Fixed

- **`bench.py prepare` writes the right elements into the GROMACS topologies of the Amber suite's systems with repartitioned masses**, taking them from the atom types instead of the masses, and checks them; their water is now rigid by SETTLE, as in the other systems (#469).
