### Fixed

- **`md_alg.sort_by_key` refuses a side output that views one of its own buffers, its `by`, or another side output** (#871). Its verifier compared the values alone, so a `memref.cast`, `subview`, `memory_space_cast`, or `reinterpret_cast` of the order (or of the keys, counts, start, cursor, or scatter) passed in `writes` or `adds`. It now follows the views to their buffer by the check of `md_alg.bin_members` (D409), one helper of the two ops. No program builds such an op; the lowered programs, with the PTX of every kernel, are unchanged.
