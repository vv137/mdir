### Performance

- Mixed-precision Gaussian PME can generate five device kernels and apply
  the reciprocal influence during the X pass. Observed steps also accumulate
  energy and virial in X. The setup log and manifest record the resolved
  plans (D435, #961).
  Mixed Gaussian TOWERS AUTO includes the four folding meshes on `sm_89`.
