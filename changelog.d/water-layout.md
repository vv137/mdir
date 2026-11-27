### Changed

- **A list of rigid units reads each copy of a taken kind as consecutive particles** (D[water-layout], #794). The order of the state already keeps a copy's particles together. The list therefore has no table of members, and the writer of the frames takes the places in their order. `rigid_units = "ON"` does not take a kind whose molecules are not of consecutive particles in the input, and `mdir check` gives that reason.

### Performance

- **The loop over the pairs of rigid units no longer looks up each member in a table** (D[water-layout], #794). On an RTX 4090 in mixed precision with `rigid_units = "ON"`, the loop is 1.6 to 4.0% faster per call on TIP3P, JAC, Factor IX, and Cellulose.
