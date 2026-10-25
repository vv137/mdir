### Fixed

- **A pull request with a pending decision passes the checks**: the table of
  `docs/decisions.md` lists numbered decisions only, so a PR that adds
  `docs/decisions/<label>.md` leaves it as it is, and the release path
  (`check-path.sh`) and `decision-number.sh check --pending` pass on it;
  the check of the release path no longer fails now and then while its
  temporary repository is removed (#602, #609).
