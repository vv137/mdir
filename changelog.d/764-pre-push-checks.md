### Added

- **A pre-push hook runs the checks of CI locally**: `scripts/check-before-push.sh` runs the `checks` job of CI in seconds, and `git config core.hooksPath scripts/hooks` makes git run it before every push (#764).
