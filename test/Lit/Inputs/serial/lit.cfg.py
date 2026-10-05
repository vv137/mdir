# A suite of tests that only sleep and record when they ran, scheduled as
# the MDIR suite schedules its tests on a GPU (test/mdir_lit.py). It needs no
# device: the feature cuda is declared here.
import os
import sys

import lit.formats

config.name = "serial"
config.test_format = lit.formats.ShTest()
config.suffixes = [".test"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = lit_config.params["out"]
config.available_features.update(["cuda", "hdf5", "cuda-sanitizer-free"])
config.substitutions.append(("%python", sys.executable))
config.substitutions.append(("%record", os.path.join(os.path.dirname(__file__), "record.py")))
config.substitutions.append(("%log", lit_config.params["out"]))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import mdir_lit

mdir_lit.serialize_gpu_tests(config, lit_config)
