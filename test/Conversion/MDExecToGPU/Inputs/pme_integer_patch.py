"""Exercise the emitted PME kernels with an independent direct integer grid."""
import argparse
import itertools
import math
import re
import struct
import subprocess
from pathlib import Path


def rounded(x, precision):
    return struct.unpack("f", struct.pack("f", x))[0] if precision == "f32" else x


def fixture(precision, case):
    dims = (9, 9, 7)
    particles = [(1.5, 2.5, 3.5, 0.875), (0.5, 7.5, 6.5, -0.5625),
                 (5.5, 4.5, 0.5, 0.125), (8.5, 6.5, 6.5, -0.25)]
    if case == "occupancy":
        particles += [particles[0]] * 15
    if case == "charge":
        particles[1] = (*particles[1][:3], -3.0)
    if case == "refuse":
        particles[0] = (*particles[0][:3], 2.0**60)
    starts = [tuple((int(u) - 3) % k for u, k in zip(p, dims)) for p in particles]
    cells = [math.ceil(k / 8) for k in dims]
    keys = [8*((x//8*cells[1]+y//8)*cells[2]+z//8)
            + 4*(x%8//4) + 2*(y%8//4) + z%8//4 for x, y, z in starts]
    offsets = [sum(key < j for key in keys) for j in range(8*math.prod(cells)+1)]
    grid = [0] * math.prod(dims)
    # Every coordinate has fractional part 1/2; cubic cardinal weights
    # are (1,23,23,1)/48. Round at the mesh precision after each multiply.
    cast = lambda x: rounded(x, precision)
    weights = [cast(n * cast(1/6)) for n in (0.125, 2.875, 2.875, 0.125)]
    scale = 2**(23 if precision == "f32" else 40)
    for pos, base in zip(particles, starts):
        for x, y, z in itertools.product(range(4), repeat=3):
            ix, iy, iz = [(s+j) % k for s, j, k in zip(base, (x,y,z), dims)]
            grid[(ix*dims[1]+iy)*dims[2]+iz] += round(cast(
                cast(cast(pos[3]*scale*weights[x])*weights[y])*weights[z]))
    return particles, keys, offsets, grid


def program(precision, case, source, opt):
    particles, keys, offsets, grid = fixture(precision, case)
    fixture_ir = source.read_text().replace("f32", precision)
    args = [opt, "--md-exec-assign-algebra", "--convert-md-exec-to-gpu"]
    module = subprocess.run(args, input=fixture_ir, text=True, check=True,
                            stdout=subprocess.PIPE).stdout
    print("module attributes {gpu.container_module} {")
    def kernel(name):
        match = re.search(r"^  func.func private @("+name+r"_p[0-9]\w+)\(.*?^  }",
                          module, re.M | re.S)
        print(match[0])
        return match[1]
    spread = kernel("mdrt_gpu_pme_spread_ordered")
    convert = kernel("mdrt_gpu_pme_patch_real")
    gather = kernel("mdrt_gpu_pme_gather_ordered")
    print("func.func private @printI64(i64)\nfunc.func private @printNewline()")
    print("func.func @main() {")
    for i in sorted({0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 64, 128, 256, len(particles),
                     len(offsets), len(grid), *range(len(particles)), *range(len(offsets))}):
        print(f"%i{i} = arith.constant {i} : index")
    real = precision
    specs = [("p", f"4x{real}", len(particles)), ("off","i32",len(offsets)),
             ("grid","i32",len(grid)), ("wide","i64",len(grid)), ("flag","i32",1),
             ("out",real,len(grid)), ("ids","i32",len(particles)),
             ("xyz",f"3x{real}",len(particles))]
    types = {}
    for name,t,n in specs:
        types[name] = f"memref<?x{t}, 1>"
        print(f"%{name} = gpu.alloc (%i{n}) : {types[name]}")
        print(f"%h{name} = memref.alloc(%i{n}) : memref<?x{t}>")
    copies = itertools.count()
    def copy(name, to_device):
        gt = types[name]
        ht = gt.replace(", 1>", ">")
        operands = (name,"h"+name,gt,ht) if to_device else ("h"+name,name,ht,gt)
        dst,src,dtype,stype = operands
        n = next(copies)
        print(f"%wait{n} = gpu.wait async")
        print(f"%copy{n} = gpu.memcpy async [%wait{n}] %{dst}, %{src} : {dtype}, {stype}")
        print(f"gpu.wait [%copy{n}]")
    for i,off in enumerate(offsets):
        print(f"%off{i} = arith.constant {off} : i32")
        print(f"memref.store %off{i}, %hoff[%i{i}] : memref<?xi32>")
    copy("off", True)
    print("%box = arith.constant dense<10.0> : vector<3xf64>")
    for run,(threads,extent) in enumerate([(64,11),(128,6),(256,0)]):
        ids = sorted(range(len(keys)), key=lambda j:(keys[j], -j if run == 1 else j))
        for i,j in enumerate(ids):
            for axis,v in enumerate(particles[j]):
                print(f"%v{run}_{i}_{axis} = arith.constant {v:.17e} : {real}")
                print(f"memref.store %v{run}_{i}_{axis}, %hp[%i{i},%i{axis}] : memref<?x4x{real}>")
            print(f"%id{run}_{i} = arith.constant {j} : i32")
            print(f"memref.store %id{run}_{i}, %hids[%i{i}] : memref<?xi32>")
        copy("p", True)
        copy("ids", True)
        names = ["p","off","grid","wide","flag"]
        operands = ", ".join("%"+n for n in names)
        signature = ", ".join(types[n] for n in names)
        print(f"func.call @{spread}({operands}, %i9,%i9,%i7,%i{threads},%i{extent}) : "
              f"({signature}, index,index,index,index,index) -> ()")
        print(f"func.call @{convert}(%grid,%wide,%flag,%out) : "
              f"({types['grid']},{types['wide']},{types['flag']},{types['out']}) -> ()")
        copy("flag", False)
        print(f"%status{run} = memref.load %hflag[%i0] : memref<?xi32>")
        print(f"%s{run} = arith.extsi %status{run} : i32 to i64")
        print(f"func.call @printI64(%s{run}) : (i64) -> ()\nfunc.call @printNewline() : () -> ()")
        narrow = precision == "f32" and case == "normal"
        result,word = ("grid","i32") if narrow else ("wide","i64")
        copy(result, False)
        print(f"scf.for %j = %i0 to %i{len(grid)} step %i1 {{")
        print(f"%word = memref.load %h{result}[%j] : memref<?x{word}>")
        if narrow:
            print("%signed = arith.extsi %word : i32 to i64")
        print(f"func.call @printI64(%{'signed' if narrow else 'word'}) : (i64) -> ()")
        print("func.call @printNewline() : () -> ()\n}")
        print(f"func.call @{gather}(%p,%ids,%out,%box,%i9,%i9,%i7,%xyz,%i{threads}) : "
              f"({types['p']},{types['ids']},{types['out']},vector<3xf64>,index,index,index,"
              f"{types['xyz']},index) -> ()")
        copy("xyz", False)
        print(f"scf.for %atom = %i0 to %i{len(particles)} step %i1 {{")
        print("scf.for %ax = %i0 to %i3 step %i1 {")
        print(f"%component = memref.load %hxyz[%atom,%ax] : memref<?x3x{real}>")
        print(f"%bits = arith.bitcast %component : {real} to {real.replace('f','i')}")
        if real == "f32":
            print("%bits64 = arith.extsi %bits : i32 to i64")
        print(f"func.call @printI64(%{'bits64' if real == 'f32' else 'bits'}) : (i64) -> ()")
        print("func.call @printNewline() : () -> ()\n}\n}")
    print("return\n}\n}")


def check(precision, case, path):
    particles, _, _, grid = fixture(precision, case)
    values = iter(map(int, path.read_text().split()))
    first = None
    for _ in range(3):
        assert next(values) == int(precision == "f64" or case != "normal")
        measured = [next(values) for _ in grid]
        assert measured == grid, next((v for v in zip(measured,grid) if v[0]!=v[1]), None)
        force = [next(values) for _ in range(3*len(particles))]
        assert any(force) and (first is None or force == first)
        first = force
    assert next(values, None) is None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["emit", "check"])
    parser.add_argument("precision", choices=["f32", "f64"])
    parser.add_argument("case", choices=["normal","occupancy","charge","refuse"])
    parser.add_argument("path", type=Path)
    parser.add_argument("--opt")
    args = parser.parse_args()
    if args.action == "emit":
        program(args.precision, args.case, args.path, args.opt)
    else:
        check(args.precision, args.case, args.path)
