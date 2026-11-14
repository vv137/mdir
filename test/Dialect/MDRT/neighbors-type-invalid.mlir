// What the parser of !mdrt.neighbors refuses (D[neighbors-contract]).
//
// RUN: mdir-opt %s --split-input-file --verify-diagnostics

// The guarantee is always spelled (the maintainer's ruling 1 on #882): a
// bare type would claim one silently.
// expected-error@+1 {{expected ','}}
func.func private @f(!mdrt.neighbors<@atoms>)

// -----

// expected-error@+1 {{expected the traversal 'directed' or 'unique', got 'both'}}
func.func private @f(!mdrt.neighbors<@atoms, both, minimum, masked, none>)

// -----

// expected-error@+1 {{expected the images 'minimum' or 'explicit', got 'plain'}}
func.func private @f(!mdrt.neighbors<@atoms, directed, plain, masked, none>)

// -----

// expected-error@+1 {{expected the exclusions 'masked' or 'removed', got 'kept'}}
func.func private @f(!mdrt.neighbors<@atoms, directed, minimum, kept, none>)

// -----

// expected-error@+1 {{expected the organization 'none' or 'grouped_by' or 'sorted_within' or 'stable', got 'sorted'}}
func.func private @f(!mdrt.neighbors<@atoms, directed, minimum, masked, sorted(owner, item)>)

// -----

// expected-error@+1 {{expected the group 'owner' or 'group', got 'cell'}}
func.func private @f(!mdrt.neighbors<@atoms, directed, minimum, masked, grouped_by(cell)>)

// -----

// expected-error@+1 {{expected the key 'item' or 'search', got 'z'}}
func.func private @f(!mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, z)>)
