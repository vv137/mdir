### Added

- **The checks before a push refuse control characters in text files**: `scripts/check-before-push.sh` fails, with the file and line, when a tracked text file holds a byte in `\x00`–`\x08` or `\x0b`–`\x1f`, or a tracked Markdown file a tab, as an escape in a shell or Python string leaves in TeX (#768).

### Fixed

- **The TeX of D373 and of the validation table of `docs/python-frames.md`** had `\times` and `\boldsymbol\theta` turned into tabs and a backspace; they are restored (#768).
