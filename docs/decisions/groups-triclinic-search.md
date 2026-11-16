# D[groups-triclinic-search]

Recorded at its merge.

**In a triclinic cell the list kernel of the groups build (D404) bounds the window of each row of cells. A row of z is visited once even when a group's range of z is as wide as the grid. Its window of y is the hull, over the extreme periods $t_z$ of the row within the range, of the windows of those periods, each narrowed to the slice of the reach that the row's slab cuts. The windows of x are bounded the same way, over the extreme periods $t_z$ and $t_y$. The images kernel of D115 moves along a lattice vector only on the axes where the group needs another image. An orthorhombic cell keeps its program.** Issue #895, blocking S2 of D417 (#894, the maintainer's option (b)).

## The defect

The groups build puts the places of a triclinic cell in the brick $[0, a_x) \times [0, b_y) \times [0, c_z)$ and its grid of cells of half the reach. It finds a group's candidates row by row: for each row of z within the reach of the group's box, the window of y; for each row of y, the window of x. Across a face of z the row $t_z$ periods away is that of $\mathbf c\, t_z$, so its windows of y and x move by $-t_z (c_x, c_y)$. Across a face of y they move by $-t_y b_x$. There were two problems.

1. **A range of z as wide as the grid made the windows of y and x the whole grid, and a range of y that wide made the window of x the whole grid.** The range was clamped to the grid, so each row is visited once, and the code did not know which period a row stood for. In the reduced cell of a rhombic dodecahedron $c_z$ is short: 44.5 Å for the solvated systems of #795, six cells of half a reach of 13.9 Å. A group's range of z is then often the whole grid, so the group read most of the places.
2. **The images kernel took all 27 lattice moves of a candidate in every group that it ran for.** It runs for a group whose box and reach are as wide as an edge of the brick along some axis. Its groups are those of the short $c_z$.

On the units proxies of D417's step 1 (RTX 5090), the triclinic list kernel and the images kernel took about 600 and 120 µs a build. Ubiquitin's orthorhombic cell, with as many places, took 76 µs, and its images kernel 1 µs.

## The bound of a row's window

Take a group's box with center $\mathbf c_b$, half widths $\mathbf h$, and the reach $R$. Let row $z$ be the slab $[z\,\ell_z, (z+1)\,\ell_z)$, with $\ell_z = c_z / n_z$ the edge of a cell along z, in the coordinates of the brick unwrapped by its period. The gap between the slab and the box along z is

$$g_z = \max\bigl(0,\; z\,\ell_z - (c_{b,z} + h_z),\; (c_{b,z} - h_z) - (z+1)\,\ell_z\bigr).$$

Any point of the slab within $R$ of the box has its y within $h_y + \sqrt{R^2 - g_z^2}$ of $c_{b,y}$, and no point of it is within $R$ when $g_z \ge R$. The window of y of the row is narrowed to that half width. The center of period $t_z$ is moved by $-t_z\, c_y$, and the window then covers the cells of y.

A row of z that the range takes for two periods takes the hull of the two narrowed windows. Each window is bounded by its own center and half width, and a period whose gap is at least $R$ adds nothing. The center moves linearly with $t_z$, so the two extreme periods bound every period between them. More than two periods occur only when a range of z spans twice the grid; such a row takes the un-narrowed window, which is the hull of all the periods.

For the window of x of a row of y the same holds, with the gap $g_y$ of the y-slab and $g_z$: the half width is $h_x + \sqrt{R^2 - g_z^2 - g_y^2}$, the center is $c_{b,x} - t_z c_x - t_y b_x$, and the hull is over the four extreme pairs $(t_z, t_y)$.

The windows are formed in f32, as before. The square of the reach that narrows them is widened by $2^{-12}$, so the rounding of a window never drops a cell from which the box test takes a candidate. A window is the whole grid only where its hull is as wide. Each cell is still visited once, and the list kernel still takes each candidate in the image nearest to the center of the box. So the lists hold the same pairs in the same images, which is D413's contract: each pair once for each of its images within the reach. Only the order in which a group's candidates are found can change.

## The images kernel

The images kernel of D115 takes the images of a candidate a lattice vector away from that of the list kernel: the codes $\Delta n_c$, then y rounded again and $\Delta n_b$, then x rounded again and $\Delta n_a$, each from $-1$ to $1$.

Along an axis where the group's box and the reach are narrower than the brick, $w + 2R < L$, the image that the pass and the rounding give is the only one within the reach. So the move along that axis is 0. A group runs the codes that move only along its axes of $w + 2R \ge L$; its other codes take no work. The test is the group's, so it is uniform over the warp.

## What stays

- An orthorhombic cell: none of this code is in its program.
- The minimum image, the lattice shifts of the places and of the entries, the encoding of the entries, and the blocks.
- The grid of cells of half the reach. A grid of a third of the reach was measured and was slower, with more rows and more runs (Section "Measurements").
