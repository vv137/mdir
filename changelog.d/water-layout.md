### Changed

- **A list of rigid units reads each copy of a taken kind as consecutive particles** (D[water-layout], #794). The order of the state already keeps a copy's particles together. The list therefore has no table of members, and the writer of the frames takes the places in their order. Under `rigid_units = "ON"` a kind whose molecules are not of consecutive particles in the input is not taken. Its particles run in the rest, and `mdir check` and the run warn (`rigid_units_scattered`). Where no kind is taken and a kind is passed over for this alone, the run takes the general route and says why; a refusal depends only on the reasons of before. No input is refused.

### Performance

- **The step with `rigid_units = "ON"` is 1.7 to 3.7% shorter** (D[water-layout], #794), on average over two jobs of two rounds. Measured on an RTX 4090 in mixed precision at 2 fs, against `main` with `ON`, on TIP3P boxes of 9,989 and 19,880 atoms and on JAC, Factor IX, and Cellulose under NVE and NPT. The writer of the frames is 16 to 41% faster per call, and the loop over the pairs, which no longer looks up each member in a table, 2.4 to 7.3%. The step with `ON` on ubiquitin in OPC is unchanged within 0.6%, and on the large box of OPC 2.0% shorter. Programs under `AUTO` and `OFF` are unchanged.
