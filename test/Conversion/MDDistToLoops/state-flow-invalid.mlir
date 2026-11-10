// D[distributed-migration]: only a md_dist.state crosses control flow; a
// layout, halo, or field carried by scf.for apart from its state is refused
// by the executor with the reason, since the verifier cannot follow its
// epoch through a block argument.
//
// RUN: mdir-opt --md-exec-assign-algebra --convert-md-dist-to-loops %s \
// RUN:   --split-input-file -verify-diagnostics

!vec = !md.field<@atoms, 3 x f64>
!own = !md_dist.field<@dd, owned, 3 x f64>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])
func.func @field_in_loop(%x: !vec, %cell: !md.cell, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %l = md_dist.partition %x, %cell : !vec -> !md_dist.layout<@dd>
  %x0 = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  // expected-error @+1 {{carries a value of md_dist other than a md_dist.state}}
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%a = %x0) -> (!own) {
    scf.yield %a : !own
  }
  return
}

// -----

!vec = !md.field<@atoms, 3 x f64>
!own = !md_dist.field<@dd, owned, 3 x f64>
!lay = !md_dist.layout<@dd>
md.particle_set @atoms
md_dist.decomposition @dd on(@atoms) grid([2, 1, 1])
func.func @layout_in_branch(%x: !vec, %cell: !md.cell, %c: i1) {
  %l = md_dist.partition %x, %cell : !vec -> !lay
  %x0 = md_dist.distribute %x, %l image(%cell) : !vec -> !own
  // expected-error @+1 {{carries a value of md_dist other than a md_dist.state}}
  %r = scf.if %c -> (!lay) {
    scf.yield %l : !lay
  } else {
    %l2 = md_dist.migrate %l, %x0, %cell : !lay, !own -> !lay
    scf.yield %l2 : !lay
  }
  return
}
