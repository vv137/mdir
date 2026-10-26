### Fixed

- **A full compile cache no longer lists its directory at every store**: a
  store that takes the cache over its bound now evicts the entries used
  least recently down to 90% of the bound, not to the bound, so the stores
  after it fit without a listing until a tenth of the bound is written
  again. At the bound, every store had listed the whole directory under the
  lock of `size`, which serialized suites on a network file system and
  made Python simulations time out (#675; D234 updated).
