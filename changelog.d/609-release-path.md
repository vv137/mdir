### Added

- **The release path is checked before release day**:
  `scripts/release/check-path.sh` runs the edits of a release on a copy of
  the tree and checks the changelog's move, the notes with absolute links,
  and versions pinned in tests or code, and the notes of a GitHub Release
  now link into the repository at the tag (#609).
