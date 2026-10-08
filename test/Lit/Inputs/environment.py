"""environment.py CUDA_CACHE_PATH TMPDIR: the values that lit was given, empty
for none, against what this test received (test/Lit/environment.test)."""
import os
import sys
import tempfile

cache, base = sys.argv[1], sys.argv[2]
assert os.environ.get("CUDA_CACHE_PATH", "") == cache, (os.environ.get("CUDA_CACHE_PATH"), cache)
print("the cache of the driver is that of the suite")
here = os.path.realpath(tempfile.gettempdir())
assert not base or here.startswith(os.path.realpath(base) + os.sep), (here, base)
print("temporary files are under the directory of the suite")
