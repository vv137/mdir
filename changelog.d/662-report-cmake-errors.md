### Fixed

- **A failed configure names CMake's error**: the report of a failed
  step of `scripts/release/publish.sh` shows the log's `CMake Error at ...`
  lines with the two lines of their message, besides the `FAILED:` and
  `error:` lines (#662).
