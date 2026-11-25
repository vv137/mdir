# D[box-f32-exact] Use one representable periodic cell per step

In mixed and single precision, round each independent component of the reduced
cell in nm to binary32 at input, checkpoint restore, frame replacement, and
barostat scaling. Widen those values exactly for the state and all binary64
consumers. This removes a difference of periods between particle wrapping and
the translations of retained pair images (#966). Double precision keeps the
supplied cell. The setup log and manifest report the six signed changes.
