### Changed

- **One entry point for the container and binary builds**:
  `scripts/packaging/build.sh runtime|tarball|wheels|ci|ci-prebuilt` builds
  each target with the container program of `DOCKER`, a log, the commit
  check, and, for the CI image alone, the push; `publish.sh`, `ci.yml`, and
  `ci-image.yml` build through it, and `packaging/README.md` lists the
  targets in one table. It replaces `scripts/ci/build-image.sh`, whose
  `--push` is now `build.sh ci --clone main --push --login` (#639).
