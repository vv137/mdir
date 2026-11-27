### Added

- **`mdir check` reports the elements of a topology and where each comes
  from** (the file, or a fallback: the name and the mass, the mass, none),
  the constraints by kind (bonds of hydrogen, rigid waters), and warns
  (`element_group_mass`, `element_light_atom`, `element_type_name`) where
  the mass of a heavy atom with its hydrogens, a light atom of one bond,
  or the name of a type contradicts an element, as in GROMACS topologies
  of repartitioned masses whose at.num were guessed from the masses. The
  run is unchanged (#469, D[gromacs-hmr-elements]).
