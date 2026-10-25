### Fixed

- **The shards of a suite cover it**: a shard needs `--order=lexical`, which
  `lit.cfg.py` now requires, since the default order followed the times
  recorded in each build tree and two trees took overlapping shards; the
  check of the expected unsupported tests compares a shard's own tests
  (D[shard-order], #608).
