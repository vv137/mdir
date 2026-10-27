### Fixed

- **`backend` and `stage_field_precision` of `[[energy.model]]`, `scaling` of `[[energy.external]]`, and `weighting` of a term over groups are read without regard to case**, as the other choices of the control file are: `backend = "stages"` was refused, and now means `"STAGES"`. `mdir.ModelTerm` takes its `backend` and `stage_field_precision` the same way (#551).
