### Changed

- **A list of rigid units reads each copy of a taken kind as consecutive particles** (D[water-layout], #794). The order of the state already keeps a copy's particles together. The list therefore has no table of members, and the writer of the frames takes the places in their order. `rigid_units = "ON"` does not take a kind whose molecules are not of consecutive particles in the input, and `mdir check` gives that reason.

### Performance

- **The step with `rigid_units = "ON"` is 1.7 to 3.7% shorter** (D[water-layout], #794), on average over two jobs of two rounds. Measured on an RTX 4090 in mixed precision at 2 fs, against `main` with `ON`, on TIP3P boxes of 9,989 and 19,880 atoms and on JAC, Factor IX, and Cellulose under NVE and NPT. The writer of the frames is 14 to 39% faster per call. The loop over the pairs no longer looks up each member in a table, and it is about 3% faster on all of these systems except Cellulose. Programs under `AUTO` and `OFF` are unchanged.
