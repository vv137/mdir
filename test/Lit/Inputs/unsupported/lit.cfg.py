# A suite whose list of expected unsupported tests (expected-unsupported.txt
# here) differs from what its features leave unsupported: needs-c.test is
# unsupported without an entry, and runs.test has an entry but runs
# (test/mdir_lit.py, #608). It declares the feature a; -Dfeatures=a,c adds c.
import os
import sys

import lit.formats

config.name = "unsupported"
config.test_format = lit.formats.ShTest()
config.suffixes = [".test"]
config.excludes = ["lit.cfg.py"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = lit_config.params["out"]
config.available_features.update(
    lit_config.params.get("features", "a").split(","))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import mdir_lit

mdir_lit.check_shards(lit_config)
mdir_lit.check_expected_unsupported(config, lit_config)
