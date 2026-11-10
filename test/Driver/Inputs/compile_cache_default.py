"""Where the compile cache is and whom it trusts (D397).

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
  permission, and the entries are read. A default that is a symbolic link,
  to another user's directory (`/`) or to a private directory of the
  user's that holds whole and valid entries, is refused, and so is a
  default whose parent (`mdir`) is a link to the user's private one:
  another user may swap a link that passes the checks (they would follow
  it) for one to a directory of their own. Nothing in the target of a link
  is read or touched. A directory named by MDIR_COMPILE_CACHE_DIR that
  others may write is used, with a warning.
- --size: MDIR_COMPILE_CACHE_MAX_MB=0 turns the cache off, and a value that
  is not a number, or whose bytes do not fit in 64 bits, is reported and
  keeps the bound.
- --salt: a stored key holds the build ID of the image that compiled it
  (the Python extension, read from its ELF notes here);
  MDIR_COMPILE_CACHE_SALT_OVERRIDE, which stands for another build
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
        print("foreign-default: " + warned(err, "is a symbolic link"))
        # Links to the user's own private directories, which pass every
        # check of the owner and the mode that follows a link.
        before = files(default)
        linked = work / "xdg-linked"
        (linked / "mdir").mkdir(parents=True, exist_ok=True)
        os.chmod(linked / "mdir", 0o700)
        if not (linked / "mdir" / "compile").is_symlink():
            (linked / "mdir" / "compile").symlink_to(default)
        stats, out, err = run("linked-default", default=linked)
        report("linked-default", stats, out)
        print("linked-default: " + warned(err, "is a symbolic link")
              + ", entries " + ("unchanged" if files(default) == before
                                else "CHANGED"))
        linked = work / "xdg-linked-parent"
        linked.mkdir(exist_ok=True)
        if not (linked / "mdir").is_symlink():
            (linked / "mdir").symlink_to(default.parent)
        stats, out, err = run("linked-parent", default=linked)
        report("linked-parent", stats, out)
        print("linked-parent: " + warned(err, "is a symbolic link")
              + ", entries " + ("unchanged" if files(default) == before
                                else "CHANGED"))
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
        # 2^44 MiB is 2^64 bytes, one more than 64 bits hold.
        huge = str(2**44)
        stats, out, err = run("size-overflow", default=sized,
                              env={"MDIR_COMPILE_CACHE_MAX_MB": huge})
        report("size-overflow", stats, out)
        print("size-overflow: " + warned(err, f"MDIR_COMPILE_CACHE_MAX_MB='{huge}'"
                                              " is not a whole number"))

    if "--salt" in options:
        # The key of a stored host object names this build: the build ID of
        # the extension that compiled it.
        import mdir
        extension = next(pathlib.Path(mdir.__file__).parent.glob("_core*.so"))
        keys = [read_key(p) for p in sorted((default / "host").glob("*.o"))]
        print("build-id: " + ("the extension's" if keys and all(
            f"image build-id {build_id(extension)}\n" in k for k in keys)
            else "MISSING"))
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


def read_key(path):
    """The key of the cache entry at `path`: after the magic tag and the
    length of the key (CompileCache.cpp)."""
    data = path.read_bytes()
    length = int.from_bytes(data[8:16], "little")
    return data[16:16 + length].decode()


def build_id(path):
    """The GNU build ID of the ELF file at `path`, from its PT_NOTE
    segments, in lowercase hex."""
    data = path.read_bytes()
    assert data[:4] == b"\x7fELF" and data[4] == 2 and data[5] == 1
    phoff = int.from_bytes(data[32:40], "little")
    phentsize = int.from_bytes(data[54:56], "little")
    phnum = int.from_bytes(data[56:58], "little")
    for i in range(phnum):
        header = data[phoff + i * phentsize:phoff + (i + 1) * phentsize]
        if int.from_bytes(header[0:4], "little") != 4:  # PT_NOTE
            continue
        offset = int.from_bytes(header[8:16], "little")
        end = offset + int.from_bytes(header[32:40], "little")
        while offset + 12 <= end:
            namesz, descsz, kind = (int.from_bytes(data[offset + 4 * j:
                                                        offset + 4 * j + 4],
                                                   "little")
                                    for j in range(3))
            name = data[offset + 12:offset + 12 + namesz]
            desc = offset + 12 + (namesz + 3) // 4 * 4
            if kind == 3 and name == b"GNU\0":  # NT_GNU_BUILD_ID
                return data[desc:desc + descsz].hex()
            offset = desc + (descsz + 3) // 4 * 4
    return None


if __name__ == "__main__":
    main()
