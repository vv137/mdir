### Added

- **The derivative of the product of force matching in the positions** (D[force-matching-hvp], #771 part 1): the request `tangent_forces` of `md.evaluate`, the forces of the tangent of a potential, gives the Hessian-vector product $g\,\nabla U + \mathsf H\,\bar{\mathbf G}$ in a variant of the product program of the frame evaluator, for pair terms (whose tangent reads the displacement), terms over tuples, and terms of the absolute positions; the tangent of a reciprocal sum and virtual sites are refused by name ([docs/ops.md](docs/ops.md)).
