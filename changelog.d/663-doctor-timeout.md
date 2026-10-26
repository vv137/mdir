### Added

- **`mdir doctor --timeout=<seconds>`**: the limit of each check run, 120 s
  by default as before (0 waits without a limit); a run stopped at the
  limit says so and names the option. The doctor tests pass 600 s, since a
  full suite on a shared node reached 120 s (#663).
