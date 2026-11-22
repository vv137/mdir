### Changed

- **The errors of the builder are ASCII as well** (#813). The refusals of a `pairlist_distance` beyond the reach of the neighbor matrix or the groups, of a cell less than twice the cutoff, and of a prolate PME grid that does not reach its tolerance spell the angstrom `Ang` and a product `x`, as the rest of the text of `mdir` does since D396; `scripts/check-before-push.sh` refuses a non-ASCII character in a string literal of `lib/Driver/Builder*.cpp` and `lib/Driver/Term*.cpp`.
