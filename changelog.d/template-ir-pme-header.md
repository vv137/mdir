### Changed

- **No template of the lowering is instantiated by rewriting its text** (D[template-ir-pme-header], #739). The PME templates take the aliases of the types of an instance and the shapes of the tiles from a header above their unchanged text, and the names of the instance in IR through the symbol table; `instantiatePMETemplates` is removed. The lowered programs do not change (`docs/pme.md`, Section 3).
