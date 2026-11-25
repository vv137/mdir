### Changed

- Reduce the cost of checked bonded force and energy accumulation on the
  GPU by using block-scope and device-scope relaxed integer adds.
  [D437](docs/decisions/bonded-word-atomics.md)
