"""Where the compile cache is and whom it trusts (D[compile-cache-default]).

  compile_cache_default.py ROOT TARGET PRECISION WORK
                           [--location] [--security] [--size] [--salt]

Each scenario is a process of its own that compiles and runs the program
of compile_cache.py (the dipeptide, 20 deterministic steps) and prints the
counts of its compile: host objects compiled, read, rejected, and stored,
and on a GPU the PTX and cubins generated and read and the GPU entries
stored. A scenario that takes the default directory has no
MDIR_COMPILE_CACHE_DIR, an XDG_CACHE_HOME under WORK, and
MDIR_COMPILE_CACHE_TEST_LOCAL=1, so that the default is WORK's whatever the
file system of WORK. The state after the steps of each scenario must equal
that of a run without a cache bit for bit.

- --location: MDIR_COMPILE_CACHE=off makes no directory; the default is
  made private (0700, its parent too), stored to cold, and read warm; a
  relative XDG_CACHE_HOME is ignored for HOME/.cache.
- --security: the warm default directory made writable by others (0777)
  is refused: nothing is read, written, or touched, though its entries are
  whole and valid, and a warning says why; made private again, they are
  read. A parent that the group may write, the user's, loses that
  permission, and the entries are read. A default that is
  another user's directory (a link to `/`) is refused. A directory named by
  MDIR_COMPILE_CACHE_DIR that others may write is used, with a warning.
- --size: MDIR_COMPILE_CACHE_MAX_MB=0 turns the cache off, and a value that
  is not a number is reported and keeps the bound.
- --salt: MDIR_COMPILE_CACHE_SALT_OVERRIDE, which stands for another build
  of MDIR, misses every entry of every kind on a warm cache and stores as
  many again; the same override then hits, and so does no override.
"""
import json
import os
import pathlib
import stat
import subprocess
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


def main():
    root, target, precision, work = sys.argv[1:5]
    options = set(sys.argv[5:])
    work = pathlib.Path(work)
    work.mkdir(parents=True, exist_ok=True)
    gpu = target == "GPU"

    def run(name, env=None, default=None):
        """A child of compile_cache.py; `default` is the XDG_CACHE_HOME of a
        scenario of the default directory, else WORK/named is named."""
        environment = dict(os.environ)
        for key in ("MDIR_COMPILE_CACHE", "MDIR_COMPILE_CACHE_MAX_MB",
                    "MDIR_COMPILE_CACHE_SALT_OVERRIDE",
                    "MDIR_COMPILE_CACHE_TEST_LOCAL"):
            environment.pop(key, None)
        if default is not None:
            environment.pop("MDIR_COMPILE_CACHE_DIR", None)
            environment["XDG_CACHE_HOME"] = str(default)
            environment["MDIR_COMPILE_CACHE_TEST_LOCAL"] = "1"
        else:
            environment["MDIR_COMPILE_CACHE_DIR"] = str(work / "named")
        environment.update(env or {})
        out = work / f"{name}.npz"
        result = subprocess.run(
            [sys.executable, str(HERE / "compile_cache.py"), "--child", root,
             target, precision, "1", str(out), ""],
            env=environment, check=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True)
        return json.loads(result.stdout.splitlines()[-1]), out, result.stderr

    reference = None

    def report(name, stats, out):
        a, b = np.load(reference), np.load(out)
        same = all(a[k].tobytes() == b[k].tobytes() for k in a.files)
        line = (f"{name}: compiled {stats['host_compiled']} hits "
                f"{stats['cache_hits']} rejected {stats['cache_rejected']} "
                f"stored {stats['cache_stored']}")
        if gpu:
            line += (f" gpu ptx {stats['gpu_ptx_compiled']} read "
                     f"{stats['gpu_ptx_hits']} cubin {stats['gpu_cubin_compiled']}"
                     f" read {stats['gpu_cubin_hits']} stored "
                     f"{stats['gpu_cache_stored']}")
        print(line + " state " + ("same" if same else "DIFFERENT"), flush=True)

    def files(directory):
        """Every entry of the cache at `directory`, with its size and times."""
        directory = pathlib.Path(directory)
        if not directory.is_dir():
            return {}
        return {str(p.relative_to(directory)): (p.stat().st_size,
                                                p.stat().st_mtime_ns)
                for p in directory.rglob("*")
                if p.is_file() and p.suffix in (".o", ".ptx", ".cubin")}

    def kinds(directory):
        names = files(directory)
        return " ".join(f"{suffix} {sum(n.endswith('.' + suffix) for n in names)}"
                        for suffix in ("o", "ptx", "cubin"))

    def mode(path):
        return oct(stat.S_IMODE(os.stat(path).st_mode))[2:]

    def warned(stderr, text):
        return "warned" if text in stderr else "silent"

    stats, reference, _ = run("off", env={"MDIR_COMPILE_CACHE": "off"},
                              default=work / "xdg-off")
    report("off", stats, reference)
    print("off: default", "made" if (work / "xdg-off").exists() else "absent")

    xdg = work / "xdg"
    default = xdg / "mdir" / "compile"
    if options & {"--location", "--security", "--salt"}:
        stats, out, _ = run("default-cold", default=xdg)
        report("default-cold", stats, out)
        print(f"default-cold: directory {mode(default)} parent "
              f"{mode(default.parent)} entries {kinds(default)}")
        stats, out, _ = run("default-warm", default=xdg)
        report("default-warm", stats, out)
    if "--location" in options:
        home = work / "home"
        home.mkdir(exist_ok=True)
        stats, out, _ = run("relative-xdg", default="relative/cache",
                            env={"HOME": str(home)})
        report("relative-xdg", stats, out)
        print("relative-xdg: entries in HOME/.cache "
              + kinds(home / ".cache" / "mdir" / "compile"))

    if "--security" in options:
        # Times long past, so that a read, which touches its entry, shows.
        for p in default.rglob("*"):
            if p.is_file():
                os.utime(p, ns=(10**18, 10**18))
        before = files(default)
        os.chmod(default, 0o777)
        stats, out, err = run("open-default", default=xdg)
        report("open-default", stats, out)
        print("open-default: " + warned(err, "refused the default directory")
              + ", entries " + ("unchanged" if files(default) == before
                                else "CHANGED"))
        os.chmod(default, 0o700)
        stats, out, _ = run("private-again", default=xdg)
        report("private-again", stats, out)
        os.chmod(default.parent, 0o770)
        stats, out, err = run("open-parent", default=xdg)
        report("open-parent", stats, out)
        print(f"open-parent: {warned(err, 'refused')}, parent "
              f"{mode(default.parent)}")
        foreign = work / "xdg-foreign"
        (foreign / "mdir").mkdir(parents=True, exist_ok=True)
        link = foreign / "mdir" / "compile"
        if not link.is_symlink():
            link.symlink_to("/")
        stats, out, err = run("foreign-default", default=foreign)
        report("foreign-default", stats, out)
        print("foreign-default: " + warned(err, "belongs to user 0"))
        named = work / "named"
        named.mkdir(exist_ok=True)
        os.chmod(named, 0o777)
        stats, out, err = run("open-named")
        report("open-named", stats, out)
        print("open-named: " + warned(err, "group or others may write the "
                                           "compile cache"))

    if "--size" in options:
        sized = work / "xdg-sized"
        stats, out, _ = run("size-zero", default=sized,
                            env={"MDIR_COMPILE_CACHE_MAX_MB": "0"})
        report("size-zero", stats, out)
        print("size-zero: default", "made" if sized.exists() else "absent")
        stats, out, err = run("size-malformed", default=sized,
                              env={"MDIR_COMPILE_CACHE_MAX_MB": "2GB"})
        report("size-malformed", stats, out)
        print("size-malformed: " + warned(err, "MDIR_COMPILE_CACHE_MAX_MB='2GB'"
                                               " is not a whole number"))

    if "--salt" in options:
        before = set(files(default))
        stats, out, _ = run("other-build", default=xdg,
                            env={"MDIR_COMPILE_CACHE_SALT_OVERRIDE": "other"})
        report("other-build", stats, out)
        new = {n: v for n, v in files(default).items() if n not in before}
        print("other-build: new entries " + " ".join(
            f"{s} {sum(n.endswith('.' + s) for n in new)}"
            for s in ("o", "ptx", "cubin")))
        stats, out, _ = run("other-build-again", default=xdg,
                            env={"MDIR_COMPILE_CACHE_SALT_OVERRIDE": "other"})
        report("other-build-again", stats, out)
        stats, out, _ = run("this-build", default=xdg)
        report("this-build", stats, out)


if __name__ == "__main__":
    main()
