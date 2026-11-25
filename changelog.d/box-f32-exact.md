### Fixed

- Mixed and single precision use one binary32-exact step cell for periodic geometry. The barostat accumulates sub-ulp pressure changes in binary64, rescales positions between consecutive step cells, and checkpoints the master cell for continuation (D[box-f32-exact], #966).
