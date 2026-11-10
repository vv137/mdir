### Added

- **A guide to the internals for contributors**: `docs/internals.md` summarizes the stages of the lowering and where each lives, the design patterns of the code (emitters, named primitives in templates, op interfaces, strategies chosen by checked preconditions and measured policy, the `TermBuilder` registry, facts in the IR, the target boundary), what to use when adding a term, a pass, a kernel, a strategy, a kept structure, or a target, and short notes on GPU practice; `CONTRIBUTING.md`, `README.md`, `docs/overview.md`, and Appendix B link it (#817).
