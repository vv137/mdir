### Fixed

- **A run that searches its skins generates its host code with the fast scheduler again** (D150, #829). The candidates of the search left LLVM's `-pre-RA-sched` counted as set, so `mdir run` took it for a scheduler of the command line and generated its own host code with the default one, which can take minutes on a large program; a scheduler given on the command line is now also kept through the search.
