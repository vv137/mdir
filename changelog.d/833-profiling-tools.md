### Fixed

- The CPU sampler finds the JIT code again in an `mdir` that exports no dynamic symbols, `prof.py` attributes the f64 loops over pairs to the pairs, and `bench.py` has the one-thread proxy variants `mp1` and `dp1` (#833).
