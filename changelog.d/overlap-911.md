### Changed

- **The optional second-stream PME schedule overlaps independent direct pair traversals inside nested step loops**; the join now precedes the first dependent operation, and the default schedule stays serialized (D[overlap-911], #911).
