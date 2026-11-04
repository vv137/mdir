### Changed

- **Since 0.2.0 Langevin dynamics (`LANGEVIN`) with constraints takes the g-BAOAB step of Leimkuhler and Matthews (2016), so its trajectories differ from those of 0.1.0** ([docs/decisions/D265.md](docs/decisions/D265.md)). With rigid water the state sits within 0.4 K of the bath, where the earlier step left it +1.4 K above at 2 fs and +4.8 K above at 4 fs. Under Langevin dynamics the log's `TEMPERATURE` is that of $K_\text{half}$, the kinetic energy of the half steps around each step. 0.2.0 shipped this without an entry (D265, #268, #736).
