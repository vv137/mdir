# References

Last updated: 2026-10-02.

This document lists the literature behind the methods, algorithms,
derivations, force fields, and formats that the MDIR documentation and
source code name.

**How to cite.** A document cites an entry by its key in brackets, linked
to the entry: [[Bussi2007]](references.md#bussi2007). A comment in the
source code cites the key or the short form "Bussi et al., J. Chem. Phys.
126, 014101 (2007)". A key is usually the family name of the first author
and the year of publication, written in ASCII; a book by two authors is
keyed by both names, and a manual by the name of the program.

**Verification.** The bibliographic data of every entry was checked on
2026-09-29, or on the day a later entry was added, against a primary
source: the Crossref record of the DOI, the DataCite record for a Zenodo
DOI, or the page of the publisher. An entry
whose data could not be checked is marked **UNVERIFIED**, with the reason.
Authors are listed in full up to six; beyond six, the first three are
followed by "et al."

Entries are sorted by key.

### Abraham2015

M. J. Abraham, T. Murtola, R. Schulz, S. Páll, J. C. Smith, B. Hess,
E. Lindahl, "GROMACS: High performance molecular simulations through
multi-level parallelism from laptops to supercomputers," *SoftwareX*
**1–2**, 19–25 (2015).
[doi:10.1016/j.softx.2015.06.001](https://doi.org/10.1016/j.softx.2015.06.001)

Used for: identifying GROMACS, a reference engine for validation and a
system discussed in [prior-art.md](prior-art.md).

### AllenTildesley2017

M. P. Allen, D. J. Tildesley, *Computer Simulation of Liquids*, 2nd ed.
(Oxford University Press, Oxford, 2017).
[doi:10.1093/oso/9780198803195.001.0001](https://doi.org/10.1093/oso/9780198803195.001.0001)

Used for: textbook treatment of periodic boundaries and the minimum image
convention, cell (linked) lists, Verlet neighbor lists and the half-skin
displacement criterion for rebuilding them, the long-range correction for
dispersion, and velocities drawn from the Maxwell–Boltzmann distribution.

### Andersen1983

H. C. Andersen, "Rattle: A 'velocity' version of the shake algorithm for
molecular dynamics calculations," *J. Comput. Phys.* **52**, 24–34 (1983).
[doi:10.1016/0021-9991(83)90014-1](https://doi.org/10.1016/0021-9991(83)90014-1)

Used for: RATTLE, the velocity constraints under velocity Verlet (M1).

### Anderson2020

J. A. Anderson, J. Glaser, S. C. Glotzer, "HOOMD-blue: A Python package
for high-performance molecular dynamics and hard particle Monte Carlo
simulations," *Comput. Mater. Sci.* **173**, 109363 (2020).
[doi:10.1016/j.commatsci.2019.109363](https://doi.org/10.1016/j.commatsci.2019.109363)

Used for: identifying HOOMD-blue, a system discussed in
[prior-art.md](prior-art.md).

### Ballenegger2012

V. Ballenegger, J. J. Cerdà, C. Holm, "How to convert SPME to P3M:
influence functions and error estimates," *J. Chem. Theory Comput.* **8**,
936–947 (2012).
[doi:10.1021/ct2001792](https://doi.org/10.1021/ct2001792)

Used for: influence functions of mesh Ewald other than that of smooth
particle mesh Ewald, of which `influence = "OPTIMAL"` takes a factor
for each edge (pme-m1.md, Section 1.1).

### Berendsen1984

H. J. C. Berendsen, J. P. M. Postma, W. F. van Gunsteren, A. DiNola,
J. R. Haak, "Molecular dynamics with coupling to an external bath,"
*J. Chem. Phys.* **81**, 3684–3690 (1984).
[doi:10.1063/1.448118](https://doi.org/10.1063/1.448118)

Used for: the Berendsen barostat, which stochastic cell rescaling extends
with a noise term.

### Bernetti2020

M. Bernetti, G. Bussi, "Pressure control using stochastic cell
rescaling," *J. Chem. Phys.* **153**, 114107 (2020).
[doi:10.1063/5.0020514](https://doi.org/10.1063/5.0020514)

Used for: stochastic cell rescaling, the barostat of M1 (D50, D72); its
integrators, the cost of evaluating after a scaling, the effective energy,
and the dependence on the period of coupling (D77).

### Berthelot1898

D. Berthelot, "Sur le mélange des gaz," *C. R. Hebd. Séances Acad. Sci.*
**126**, 1703–1706 (1898). No DOI.

**UNVERIFIED**: the volume (Gallica, ark:/12148/bpt6k3082d) was located,
but the pages could not be opened to check the title and page range.

Used for: the geometric mean of the energy parameter in the
Lorentz–Berthelot mixing rule.

### Blackman2021

D. Blackman, S. Vigna, "Scrambled linear pseudorandom number generators,"
*ACM Trans. Math. Softw.* **47**(4), 1–32 (2021).
[doi:10.1145/3460772](https://doi.org/10.1145/3460772)

Used for: xoshiro256\*\*, the generator of the initial velocities in the
driver.

### Blondel1996

A. Blondel, M. Karplus, "New formulation for derivatives of torsion
angles and improper torsion angles in molecular mechanics: Elimination of
singularities," *J. Comput. Chem.* **17**, 1132–1141 (1996).
[doi:10.1002/(SICI)1096-987X(19960715)17:9&lt;1132::AID-JCC5&gt;3.0.CO;2-T](https://doi.org/10.1002/(SICI)1096-987X(19960715)17:9%3C1132::AID-JCC5%3E3.0.CO;2-T)

Used for: the derivative of a dihedral with respect to the positions of
its four particles, which has no singularity where three particles are
in line.

### BoxMuller1958

G. E. P. Box, M. E. Muller, "A note on the generation of random normal
deviates," *Ann. Math. Stat.* **29**, 610–611 (1958).
[doi:10.1214/aoms/1177706645](https://doi.org/10.1214/aoms/1177706645)

Used for: normal deviates from uniform ones, for the initial velocities
and the thermostat.

### Brunken2025

C. Brunken, O. Peltre, H. Chomet, et al., "Machine learning interatomic
potentials: library for efficient training, model development and
simulation of molecular systems," arXiv:2505.22397 (2025).
[doi:10.48550/arXiv.2505.22397](https://doi.org/10.48550/arXiv.2505.22397)

Used for: related work, a library of learned potentials (MACE, NequIP,
ViSNet) with wrappers for ASE and JAX MD.

### Bussi2007

G. Bussi, D. Donadio, M. Parrinello, "Canonical sampling through velocity
rescaling," *J. Chem. Phys.* **126**, 014101 (2007).
[doi:10.1063/1.2408420](https://doi.org/10.1063/1.2408420)

Used for: stochastic velocity rescaling, the thermostat of M1 (D50).

### Bussi2009

G. Bussi, T. Zykova-Timan, M. Parrinello, "Isothermal-isobaric molecular
dynamics using stochastic velocity rescaling," *J. Chem. Phys.* **130**,
074101 (2009).
[doi:10.1063/1.3073889](https://doi.org/10.1063/1.3073889)

Used for: the second-order barostat that M1 considered and did not take
(D50).

### Darden1993

T. Darden, D. York, L. Pedersen, "Particle mesh Ewald: An N⋅log(N)
method for Ewald sums in large systems," *J. Chem. Phys.* **98**,
10089–10092 (1993).
[doi:10.1063/1.464397](https://doi.org/10.1063/1.464397)

Used for: particle mesh Ewald (M1).

### Daw1984

M. S. Daw, M. I. Baskes, "Embedded-atom method: Derivation and
application to impurities, surfaces, and other defects in metals,"
*Phys. Rev. B* **29**, 6443–6453 (1984).
[doi:10.1103/PhysRevB.29.6443](https://doi.org/10.1103/PhysRevB.29.6443)

Used for: the embedded-atom method (EAM), the example of a many-body
potential with several stages.

### deBuyl2014

P. de Buyl, P. H. Colberg, F. Höfling, "H5MD: A structured, efficient, and
portable file format for molecular data," *Comput. Phys. Commun.* **185**,
1546–1553 (2014).
[doi:10.1016/j.cpc.2014.01.018](https://doi.org/10.1016/j.cpc.2014.01.018)

Used for: H5MD, the format of checkpoints (D26).

### deJong2013

D. H. de Jong, G. Singh, W. F. D. Bennett, et al., "Improved parameters
for the Martini coarse-grained protein force field," *J. Chem. Theory
Comput.* **9**, 687–697 (2013).
[doi:10.1021/ct300646g](https://doi.org/10.1021/ct300646g)

Used for: Martini 2.2, the force field of the first target of M1 (D46),
now deferred (D53).

### Dickson2022

C. J. Dickson, R. C. Walker, I. R. Gould, "Lipid21: Complex lipid membrane
simulations with AMBER," *J. Chem. Theory Comput.* **18**, 1726–1736
(2022).
[doi:10.1021/acs.jctc.1c01217](https://doi.org/10.1021/acs.jctc.1c01217)

Used for: Lipid21, the force field of the POPC bilayer on which the
semi-isotropic barostat is compared with GROMACS (D119).

### Eastman2017

P. Eastman, J. Swails, J. D. Chodera, et al., "OpenMM 7: Rapid
development of high performance algorithms for molecular dynamics,"
*PLoS Comput. Biol.* **13**, e1005659 (2017).
[doi:10.1371/journal.pcbi.1005659](https://doi.org/10.1371/journal.pcbi.1005659)

Used for: identifying OpenMM, whose custom forces define the syntax of
energy expressions (D22) and which is discussed in
[prior-art.md](prior-art.md).

### Essmann1995

U. Essmann, L. Perera, M. L. Berkowitz, T. Darden, H. Lee, L. G. Pedersen,
"A smooth particle mesh Ewald method," *J. Chem. Phys.* **103**,
8577–8593 (1995).
[doi:10.1063/1.470117](https://doi.org/10.1063/1.470117)

Used for: smooth particle mesh Ewald, with B-spline charge spreading, the
influence function, and the reciprocal virial (M1).

### Ewald1921

P. P. Ewald, "Die Berechnung optischer und elektrostatischer
Gitterpotentiale," *Ann. Phys.* **369**, 253–287 (1921).
[doi:10.1002/andp.19213690304](https://doi.org/10.1002/andp.19213690304)

Used for: the Ewald sum, of which particle mesh Ewald is a fast
evaluation, and against which it is validated.

### Fennell2006

C. J. Fennell, J. D. Gezelter, "Is the Ewald summation still necessary?
Pairwise alternatives to the accepted standard for long-range
electrostatics," *J. Chem. Phys.* **124**, 234104 (2006).
[doi:10.1063/1.2206581](https://doi.org/10.1063/1.2206581)

Used for: damped shifted force electrostatics, a cutoff-based
alternative named in the design review.

### Fuchs2025

P. Fuchs, W. Chen, S. Thaler, J. Zavadlav, "chemtrain-deploy: A parallel
and scalable framework for machine learning potentials in million-atom
MD simulations," *J. Chem. Theory Comput.* **21**(15), 7550–7560 (2025).
[doi:10.1021/acs.jctc.5c00996](https://doi.org/10.1021/acs.jctc.5c00996)

Used for: related work, the deployment of potentials defined in JAX
inside LAMMPS on several GPUs.

### Fukunishi2002

H. Fukunishi, O. Watanabe, S. Takada, "On the Hamiltonian replica
exchange method for efficient sampling of biomolecular systems:
Application to protein structure prediction," *J. Chem. Phys.* **116**,
9058–9067 (2002).
[doi:10.1063/1.1472510](https://doi.org/10.1063/1.1472510)

Used for: Hamiltonian replica exchange, a protocol of `ensemble`.

### Galtsov2025

I. S. Galtsov, R. V. Muratov, G. V. Vyskvarko, S. A. Murzov, S. A.
Dyachkov, P. R. Levashov, "MDcraft -- a modern molecular dynamics
simulation package with machine learning potentials support,"
arXiv:2511.22951 (2025).
[doi:10.48550/arXiv.2511.22951](https://doi.org/10.48550/arXiv.2511.22951)

Used for: related work, an engine with a Python interface over a core in
C++, parallel with MPI, that runs learned potentials.

### Gratl2022

F. A. Gratl, S. Seckler, H.-J. Bungartz, P. Neumann, "N ways to simulate
short-range particle systems: Automated algorithm selection with the
node-level library AutoPas," *Comput. Phys. Commun.* **273**, 108262
(2022).
[doi:10.1016/j.cpc.2021.108262](https://doi.org/10.1016/j.cpc.2021.108262)

Used for: identifying AutoPas, a model for the joint planner.

### GromacsManual2025

M. Abraham, A. Alekseenko, B. Andrews, et al., *GROMACS 2025.4 Manual*
(Zenodo, 2025).
[doi:10.5281/zenodo.17671776](https://doi.org/10.5281/zenodo.17671776)

Used for: the definitions of the `potential-shift`, `potential-switch`
(the fifth-degree switching polynomial), and `force-switch` modifiers,
which the MDIR truncations `shift`, `switch`, and `force_switch` match
and against which the tests check them; the sign convention of the
GROMACS virial; the formats `.top`, `.itp`, and `.gro`; the update groups of
the domain decomposition of GROMACS (Section "Domain decomposition"),
which the disjoint union of the groups of the constraints parallels (D83).

### Hairer2003

E. Hairer, C. Lubich, G. Wanner, "Geometric numerical integration
illustrated by the Störmer–Verlet method," *Acta Numer.* **12**, 399–450
(2003).
[doi:10.1017/S0962492902000144](https://doi.org/10.1017/S0962492902000144)

Used for: the properties `symplectic` and `time_reversible` of velocity
Verlet and leapfrog, and why neither conserves energy exactly (B9).

### He2024

Y. He, A. Podobas, S. Markidis, "Leveraging MLIR for loop vectorization
and GPU porting of FFT libraries," in *Euro-Par 2023: Parallel Processing
Workshops*, Lecture Notes in Computer Science (Springer, Cham, 2024),
pp. 207–218.
[doi:10.1007/978-3-031-50684-0_16](https://doi.org/10.1007/978-3-031-50684-0_16)

Used for: related work, FFTc, a language for FFTs on dialects of MLIR;
MDIR calls cuFFT and pocketfft for the transforms of PME.

### Hess2008

B. Hess, C. Kutzner, D. van der Spoel, E. Lindahl, "GROMACS 4:
Algorithms for highly efficient, load-balanced, and scalable molecular
simulation," *J. Chem. Theory Comput.* **4**, 435–447 (2008).
[doi:10.1021/ct700301q](https://doi.org/10.1021/ct700301q)

Used for: eighth-shell domain decomposition with dynamic load balancing
and separate PME ranks, discussed in [prior-art.md](prior-art.md).

### HockneyEastwood1988

R. W. Hockney, J. W. Eastwood, *Computer Simulation Using Particles*
(IOP Publishing, Bristol, 1988).
[doi:10.1887/0852743920](https://doi.org/10.1887/0852743920)

Used for: the leapfrog integrator, and the particle–particle
particle–mesh (PPPM) method.

### Hoogerbrugge1992

P. J. Hoogerbrugge, J. M. V. A. Koelman, "Simulating microscopic
hydrodynamic phenomena with dissipative particle dynamics," *Europhys.
Lett.* **19**, 155–160 (1992).
[doi:10.1209/0295-5075/19/3/001](https://doi.org/10.1209/0295-5075/19/3/001)

Used for: dissipative particle dynamics (DPD), the example of a pairwise
thermostat that needs a neighborhood.

### Hub2014

J. S. Hub, B. L. de Groot, H. Grubmüller, G. Groenhof, "Quantifying
artifacts in Ewald simulations of inhomogeneous systems with a net
charge," *J. Chem. Theory Comput.* **10**, 381–390 (2014).
[doi:10.1021/ct400626b](https://doi.org/10.1021/ct400626b)

Used for: the energy of the uniform background that neutralizes a system
with a net charge under Ewald summation, $-\pi Q^2 / (2V\beta^2)$ in units of
$f$ (pme-m1.md).

### Izadi2014

S. Izadi, R. Anandakrishnan, A. V. Onufriev, "Building water models: A
different approach," *J. Phys. Chem. Lett.* **5**, 3863–3871 (2014).
[doi:10.1021/jz501780a](https://doi.org/10.1021/jz501780a)

Used for: OPC, a water model with four sites, whose extra point is a
virtual site (design-m1.md, Section 19).

### Jones1924

J. E. Jones, "On the determination of molecular fields. —II. From the
equation of state of a gas," *Proc. R. Soc. Lond. A* **106**, 463–477
(1924).
[doi:10.1098/rspa.1924.0082](https://doi.org/10.1098/rspa.1924.0082)

Used for: the Lennard-Jones potential, the pair term of M0.

### Jorgensen1983

W. L. Jorgensen, J. Chandrasekhar, J. D. Madura, R. W. Impey, M. L. Klein,
"Comparison of simple potential functions for simulating liquid water,"
*J. Chem. Phys.* **79**, 926–935 (1983).
[doi:10.1063/1.445869](https://doi.org/10.1063/1.445869)

Used for: TIP3P, the water model of M1 (D53); also TIP4P, which M1 cannot
run.

### Jung2018

J. Jung, C. Kobayashi, Y. Sugita, "Kinetic energy definition in velocity
Verlet integration for accurate pressure evaluation," *J. Chem. Phys.*
**148**, 164109 (2018).
[doi:10.1063/1.5008438](https://doi.org/10.1063/1.5008438)

Used for: the kinetic energy from the half steps, which the pressure
takes (D45).

### Jung2019

J. Jung, C. Kobayashi, Y. Sugita, "Optimal temperature evaluation in
molecular dynamics simulations with a large time step," *J. Chem. Theory
Comput.* **15**, 84–94 (2019).
[doi:10.1021/acs.jctc.8b00874](https://doi.org/10.1021/acs.jctc.8b00874)

Used for: the mean of the kinetic energies at the step and at the half
steps, which the temperature takes (D45).

### Jung2026

J. Jung, D. Ugarte La Torre, C. Kobayashi, K. Ozaki, Y. Sugita,
"Optimized M-SHAKE constraint implementations for GPU-accelerated
molecular dynamics: Balancing precision and performance across
architectures," *J. Comput. Chem.* **47**(25), e70492 (2026).
[doi:10.1002/jcc.70492](https://doi.org/10.1002/jcc.70492)

Used for: the error of the velocities that the rounding of a constraint
in single precision makes, the change of a step divided by the step, as
the source of the drift of the energy, and M-SHAKE for the waters in
mixed precision, whose change is a small number (D112).

### Kelley2025

B. Kelley, S. Rajamanickam, "LAPIS: A performance portable, high
productivity compiler framework," arXiv:2509.25605 (2025).
[doi:10.48550/arXiv.2509.25605](https://doi.org/10.48550/arXiv.2509.25605)

Used for: related work, a compiler on MLIR for sparse and dense linear
algebra that emits Kokkos.

### Krautler2001

V. Kräutler, W. F. van Gunsteren, P. H. Hünenberger, "A fast SHAKE
algorithm to solve distance constraint equations for small molecules in
molecular dynamics simulations," *J. Comput. Chem.* **22**(5), 501–508
(2001).
[doi:10.1002/1096-987X(20010415)22:5<501::AID-JCC1021>3.0.CO;2-V](https://doi.org/10.1002/1096-987X(20010415)22:5%3C501::AID-JCC1021%3E3.0.CO;2-V)

Used for: M-SHAKE, the iterations of Newton on all the bonds of a group
at once: the groups of SHAKE, and the rigid waters below double
precision (D112).

### Laio2002

A. Laio, M. Parrinello, "Escaping free-energy minima," *Proc. Natl. Acad.
Sci. USA* **99**, 12562–12566 (2002).
[doi:10.1073/pnas.202427399](https://doi.org/10.1073/pnas.202427399)

Used for: metadynamics, the example of a history-dependent bias in the
design review.

### Lattner2021

C. Lattner, M. Amini, U. Bondhugula, et al., "MLIR: Scaling compiler
infrastructure for domain specific computation," in *2021 IEEE/ACM
International Symposium on Code Generation and Optimization (CGO)*
(IEEE, 2021), pp. 2–14.
[doi:10.1109/CGO51591.2021.9370308](https://doi.org/10.1109/CGO51591.2021.9370308)

Used for: MLIR, the compiler infrastructure that MDIR is built on.

### LeGrand2013

S. Le Grand, A. W. Götz, R. C. Walker, "SPFP: Speed without
compromise—A mixed precision model for GPU accelerated molecular dynamics
simulations," *Comput. Phys. Commun.* **184**, 374–380 (2013).
[doi:10.1016/j.cpc.2012.09.022](https://doi.org/10.1016/j.cpc.2012.09.022)

Used for: accumulation in fixed point with integer atomic additions,
whose result does not depend on the order of the threads.

### Li2014

P. Li, K. M. Merz, "Taking into account the ion-induced dipole
interaction in the nonbonded model of ions," *J. Chem. Theory Comput.*
**10**, 289–297 (2014).
[doi:10.1021/ct400751u](https://doi.org/10.1021/ct400751u)

Used for: the 12-6-4 model of ions, which the readers of M1 reject.

### Lorentz1881

H. A. Lorentz, "Ueber die Anwendung des Satzes vom Virial in der
kinetischen Theorie der Gase," *Ann. Phys.* **248**, 127–136 (1881).
[doi:10.1002/andp.18812480110](https://doi.org/10.1002/andp.18812480110)

Used for: the arithmetic mean of the size parameter in the
Lorentz–Berthelot mixing rule.

### Louwerse2006

M. J. Louwerse, E. J. Baerends, "Calculation of pressure in case of
periodic boundary conditions," *Chem. Phys. Lett.* **421**, 138–141
(2006).
[doi:10.1016/j.cplett.2006.01.087](https://doi.org/10.1016/j.cplett.2006.01.087)

Used for: the virial of pair forces with minimum-image displacements
under periodic boundaries, from which the pressure follows (B8).

### Lyubartsev1992

A. P. Lyubartsev, A. A. Martsinovski, S. V. Shevkunov,
P. N. Vorontsov-Velyaminov, "New approach to Monte Carlo calculation of
the free energy: Method of expanded ensembles," *J. Chem. Phys.* **96**,
1776–1783 (1992).
[doi:10.1063/1.462133](https://doi.org/10.1063/1.462133)

Used for: expanded ensembles, a protocol of `ensemble`.

### MacKerell2004

A. D. MacKerell, M. Feig, C. L. Brooks, "Extending the treatment of
backbone energetics in protein force fields: Limitations of gas-phase
quantum mechanics in reproducing protein conformational distributions in
molecular dynamics simulations," *J. Comput. Chem.* **25**, 1400–1415
(2004).
[doi:10.1002/jcc.20065](https://doi.org/10.1002/jcc.20065)

Used for: CMAP, the correction map of two backbone dihedrals, a bicubic
patch in each cell of a grid (design-m1.md, Section 20).

### Maier2015

J. A. Maier, C. Martinez, K. Kasavajhala, L. Wickstrom, K. E. Hauser,
C. Simmerling, "ff14SB: Improving the accuracy of protein side chain and
backbone parameters from ff99SB," *J. Chem. Theory Comput.* **11**,
3696–3713 (2015).
[doi:10.1021/acs.jctc.5b00255](https://doi.org/10.1021/acs.jctc.5b00255)

Used for: ff14SB, the protein force field of M1 (D53).

### Marrink2007

S. J. Marrink, H. J. Risselada, S. Yefimov, D. P. Tieleman, A. H. de Vries,
"The MARTINI force field: Coarse grained model for biomolecular
simulations," *J. Phys. Chem. B* **111**, 7812–7824 (2007).
[doi:10.1021/jp071097f](https://doi.org/10.1021/jp071097f)

Used for: the Martini 2 force field, including its lipids, deferred
(D53).

### Martinez2009

L. Martínez, R. Andrade, E. G. Birgin, J. M. Martínez, "PACKMOL: A package
for building initial configurations for molecular dynamics simulations,"
*J. Comput. Chem.* **30**, 2157–2164 (2009).
[doi:10.1002/jcc.21224](https://doi.org/10.1002/jcc.21224)

Used for: packing the POPC bilayer of the comparison of D119, whose close
contacts led to D120.

### Marsaglia2000

G. Marsaglia, W. W. Tsang, "A simple method for generating gamma
variables," *ACM Trans. Math. Softw.* **26**, 363–372 (2000).
[doi:10.1145/358407.358414](https://doi.org/10.1145/358407.358414)

Used for: the sum of the squares of normal deviates that the thermostat
takes, drawn as a gamma deviate (`runtime/mdrt.c`).

### Merz2018

P. T. Merz, M. R. Shirts, "Testing for physical validity in molecular
simulations," *PLOS ONE* **13**, e0202764 (2018).
[doi:10.1371/journal.pone.0202764](https://doi.org/10.1371/journal.pone.0202764)

Used for: the tests of physical validity that the roadmap plans for the
mixed precision and the couplings: the convergence of the fluctuation of
the energy with the step, the distribution of the kinetic energy and its
equipartition, and the ensemble tests at two states, which that paper
applies to isotropic coupling only.

### Metropolis1953

N. Metropolis, A. W. Rosenbluth, M. N. Rosenbluth, A. H. Teller,
E. Teller, "Equation of state calculations by fast computing machines,"
*J. Chem. Phys.* **21**, 1087–1092 (1953).
[doi:10.1063/1.1699114](https://doi.org/10.1063/1.1699114)

Used for: the Metropolis acceptance test (`dyn.metropolis`).

### Miyamoto1992

S. Miyamoto, P. A. Kollman, "Settle: An analytical version of the SHAKE
and RATTLE algorithm for rigid water models," *J. Comput. Chem.* **13**,
952–962 (1992).
[doi:10.1002/jcc.540130805](https://doi.org/10.1002/jcc.540130805)

Used for: SETTLE, the constraints of rigid water in double precision
(M1; below it, M-SHAKE, D112).

### Moses2020

W. S. Moses, V. Churavy, "Instead of rewriting foreign code for machine
learning, automatically synthesize fast gradients," in *Advances in
Neural Information Processing Systems 33 (NeurIPS 2020)* (Curran
Associates, 2020), pp. 12472–12485. No DOI;
[proceedings page](https://proceedings.neurips.cc/paper_files/paper/2020/hash/9332c513ef44b682e9347822c2e457ac-Abstract.html).

Used for: Enzyme, automatic differentiation at the level of LLVM, which
is not the default path of MDIR (D2).

### Mudalige2012

G. R. Mudalige, M. B. Giles, I. Reguly, C. Bertolli, P. H. J. Kelly,
"OP2: An active library framework for solving unstructured mesh-based
applications on multi-core and many-core architectures," in *2012
Innovative Parallel Computing (InPar)* (IEEE, 2012), pp. 1–12.
[doi:10.1109/InPar.2012.6339594](https://doi.org/10.1109/InPar.2012.6339594)

Used for: OP2, the origin of loops with access descriptors.

### Musaelian2023

A. Musaelian, S. Batzner, A. Johansson, et al., "Learning local
equivariant representations for large-scale atomistic dynamics," *Nat.
Commun.* **14**, 579 (2023).
[doi:10.1038/s41467-023-36329-y](https://doi.org/10.1038/s41467-023-36329-y)

Used for: Allegro, the example of a strictly local learned potential.

### Pall2013

S. Páll, B. Hess, "A flexible algorithm for calculating pair interactions
on SIMD architectures," *Comput. Phys. Commun.* **184**, 2641–2650
(2013).
[doi:10.1016/j.cpc.2013.06.003](https://doi.org/10.1016/j.cpc.2013.06.003)

Used for: cluster pair lists (clusters of a fixed number of particles,
lists of cluster pairs searched by bounding boxes, masks of interactions
and exclusions, the 8 × 4 layout of the CUDA kernels), discussed in
[prior-art.md](prior-art.md); the tile structure of MDIR (D82) follows
them, see [tiles-m1.md](tiles-m1.md).

### Pall2020

S. Páll, A. Zhmurov, P. Bauer, et al., "Heterogeneous parallelization
and acceleration of molecular dynamics simulations in GROMACS," *J. Chem.
Phys.* **153**, 134110 (2020).
[doi:10.1063/5.0018516](https://doi.org/10.1063/5.0018516)

Used for: the dual pair list with dynamic (rolling) pruning, the pruning
of cluster pairs on the GPU, and the fixed lifetime of the pair list with
a buffer estimated from a tolerance of the energy drift, which the tile
structure of MDIR (D82) follows or departs from; see
[tiles-m1.md](tiles-m1.md).

### Park2024

Y. Park, J. Kim, S. Hwang, S. Han, "Scalable parallel algorithm for graph
neural network interatomic potentials in molecular dynamics
simulations," *J. Chem. Theory Comput.* **20**, 4857–4868 (2024).
[doi:10.1021/acs.jctc.4c00190](https://doi.org/10.1021/acs.jctc.4c00190)

Used for: SevenNet, the example of a message-passing learned potential
with per-layer communication.

### Peng2025

M. J. Peng, W. S. Moses, O. Zinenko, C. Dubach, "Sound and modular
activity analysis for automatic differentiation in MLIR," *Proc. ACM
Program. Lang.* **9**(OOPSLA2), 2087–2114 (2025).
[doi:10.1145/3763125](https://doi.org/10.1145/3763125)

Used for: related work, an activity analysis for automatic
differentiation on MLIR, proved sound as an abstract interpretation.

### Quentrec1973

B. Quentrec, C. Brot, "New method for searching for neighbors in
molecular dynamics computations," *J. Comput. Phys.* **13**, 430–432
(1973).
[doi:10.1016/0021-9991(73)90046-6](https://doi.org/10.1016/0021-9991(73)90046-6)

Used for: binning particles into cells to search for neighbors (cell
lists), the first stage of every neighbor build.

### Rathgeber2012

F. Rathgeber, G. R. Markall, L. Mitchell, et al., "PyOP2: A high-level
framework for performance-portable simulations on unstructured meshes,"
in *2012 SC Companion: High Performance Computing, Networking Storage and
Analysis* (IEEE, 2012), pp. 1116–1123.
[doi:10.1109/SC.Companion.2012.134](https://doi.org/10.1109/SC.Companion.2012.134)

Used for: PyOP2, with OP2 the origin of loops with access descriptors.

### ReactantJl

EnzymeAD, "Reactant.jl," a compiler of Julia functions to MLIR, with
automatic differentiation by EnzymeMLIR and executables for CPUs, GPUs,
and TPUs through XLA, under the MIT license. No paper;
[github.com/EnzymeAD/Reactant.jl](https://github.com/EnzymeAD/Reactant.jl)
(read 2026-10-02).

Used for: related work, a compiler on MLIR with differentiation at the
level of MLIR.

### Reinecke2019

M. Reinecke, "pocketfft," a library of fast Fourier transforms in C,
Max-Planck-Society (2010–2019), under the 3-clause BSD license.
[gitlab.mpcdf.mpg.de/mtr/pocketfft](https://gitlab.mpcdf.mpg.de/mtr/pocketfft)

Used for: the FFT of particle mesh Ewald on the host (D64, pme-m1.md).

### Ryckaert1977

J.-P. Ryckaert, G. Ciccotti, H. J. C. Berendsen, "Numerical integration
of the cartesian equations of motion of a system with constraints:
molecular dynamics of n-alkanes," *J. Comput. Phys.* **23**, 327–341
(1977).
[doi:10.1016/0021-9991(77)90098-5](https://doi.org/10.1016/0021-9991(77)90098-5)

Used for: SHAKE, the constraints of the bonds of hydrogen (M1).

### Salmon2011

J. K. Salmon, M. A. Moraes, R. O. Dror, D. E. Shaw, "Parallel random
numbers: As easy as 1, 2, 3," in *Proceedings of 2011 International
Conference for High Performance Computing, Networking, Storage and
Analysis (SC '11)* (ACM, 2011), pp. 1–12.
[doi:10.1145/2063384.2063405](https://doi.org/10.1145/2063384.2063405)

Used for: counter-based random numbers (Random123) and the Philox
generator, Philox 4×32 with 10 rounds in M1 (`runtime/mdrt.c`), checked
against the known answers of Random123 (`test/Runtime/random.test`).

### SalomonFerrer2013

R. Salomon-Ferrer, A. W. Götz, D. Poole, S. Le Grand, R. C. Walker,
"Routine microsecond molecular dynamics simulations with AMBER on GPUs.
2. Explicit solvent particle mesh Ewald," *J. Chem. Theory Comput.* **9**,
3878–3888 (2013).
[doi:10.1021/ct400314y](https://doi.org/10.1021/ct400314y)

Used for: how pmemd.cuda, the reference of the goal of performance, builds
its neighbor list (Section 3.3): particles sorted into boxes of the
cutoff plus the skin and along a Hilbert curve of 4 × 4 × 4 within each,
then taken in groups of 16 or 32 that share the particles within the
extended cutoff of any of them, from the box and its 13 neighbors on the
leading edge (a half shell); the direct sum iterates over that list a warp
at a time, with the exclusions masked. Measured against in
`scripts/experiments/neighbor-structures` (2026-10-01).

### Saunders2018

W. R. Saunders, J. Grant, E. H. Müller, "A domain specific language for
performance portable molecular dynamics algorithms," *Comput. Phys.
Commun.* **224**, 119–135 (2018).
[doi:10.1016/j.cpc.2017.11.006](https://doi.org/10.1016/j.cpc.2017.11.006)

Used for: PPMD, the origin of the particle and pair loops of `md_exec`
(P11 to P18); see [prior-art.md](prior-art.md).

### Schoenholz2020

S. S. Schoenholz, E. D. Cubuk, "JAX MD: A framework for differentiable
physics," in *Advances in Neural Information Processing Systems 33
(NeurIPS 2020)* (Curran Associates, 2020), pp. 11428–11441. No DOI;
[proceedings page](https://proceedings.neurips.cc/paper_files/paper/2020/hash/83d3d4b6c9579515e1679aca8cbc8033-Abstract.html).

Used for: identifying JAX-MD, a system discussed in
[prior-art.md](prior-art.md).

### SchottVerdugo2019

S. Schott-Verdugo, H. Gohlke, "PACKMOL-Memgen: A simple-to-use,
generalized workflow for membrane-protein–lipid-bilayer system building,"
*J. Chem. Inf. Model.* **59**, 2522–2528 (2019).
[doi:10.1021/acs.jcim.9b00269](https://doi.org/10.1021/acs.jcim.9b00269)

Used for: the input of PACKMOL for the POPC bilayer of D119.

### Shirts2007

M. R. Shirts, D. L. Mobley, J. D. Chodera, V. S. Pande, "Accurate and
efficient corrections for missing dispersion interactions in molecular
simulations," *J. Phys. Chem. B* **111**, 13052–13063 (2007).
[doi:10.1021/jp0735987](https://doi.org/10.1021/jp0735987)

Used for: the correction for the dispersion beyond the cutoff, to the
energy and the pressure (M1).

### Shirts2013

M. R. Shirts, "Simple quantitative tests to validate sampling from
thermodynamic ensembles," *J. Chem. Theory Comput.* **9**(2), 909–926
(2013).
[doi:10.1021/ct300688p](https://doi.org/10.1021/ct300688p)

Used for: the tests of the ensembles of the thermostat and the barostat,
from the ratio of the distributions of two runs at different temperatures
or pressures (`scripts/validation/ensembles`).

### Slattery2022

S. Slattery, S. T. Reeve, C. Junghans, et al., "Cabana: A performance
portable library for particle-based simulations," *J. Open Source Softw.*
**7**(72), 4115 (2022).
[doi:10.21105/joss.04115](https://doi.org/10.21105/joss.04115)

Used for: identifying Cabana, a model for the runtime.

### Steele2014

G. L. Steele, D. Lea, C. H. Flood, "Fast splittable pseudorandom number
generators," in *Proceedings of the 2014 ACM International Conference on
Object Oriented Programming Systems Languages & Applications (OOPSLA '14)*
(ACM, 2014), pp. 453–472.
[doi:10.1145/2660193.2660195](https://doi.org/10.1145/2660193.2660195)

Used for: SplitMix, whose 64-bit variant (splitmix64) seeds the state of
the generator of the initial velocities. The mixing constants in the code
are those of the common 64-bit variant, not of the paper.

### Steinbach1994

P. J. Steinbach, B. R. Brooks, "New spherical-cutoff methods for
long-range forces in macromolecular simulation," *J. Comput. Chem.* **15**,
667–683 (1994).
[doi:10.1002/jcc.540150702](https://doi.org/10.1002/jcc.540150702)

Used for: force switching and potential switching as truncations of
nonbonded terms at a cutoff.

### Stoddard1973

S. D. Stoddard, J. Ford, "Numerical experiments on the stochastic
behavior of a Lennard-Jones gas system," *Phys. Rev. A* **8**, 1504–1512
(1973).
[doi:10.1103/PhysRevA.8.1504](https://doi.org/10.1103/PhysRevA.8.1504)

Used for: the shifted-force potential, `u(r) − u(r_c) − (r − r_c) u'(r_c)`,
the truncation `force_shift`.

### Sugita1999

Y. Sugita, Y. Okamoto, "Replica-exchange molecular dynamics method for
protein folding," *Chem. Phys. Lett.* **314**, 141–151 (1999).
[doi:10.1016/S0009-2614(99)01123-9](https://doi.org/10.1016/S0009-2614(99)01123-9)

Used for: temperature replica exchange, a protocol of `ensemble`.

### Swope1982

W. C. Swope, H. C. Andersen, P. H. Berens, K. R. Wilson, "A computer
simulation method for the calculation of equilibrium constants for the
formation of physical clusters of molecules: Application to small water
clusters," *J. Chem. Phys.* **76**, 637–649 (1982).
[doi:10.1063/1.442716](https://doi.org/10.1063/1.442716)

Used for: the velocity Verlet integrator (D21).

### Tersoff1988

J. Tersoff, "New empirical approach for the structure and energy of
covalent systems," *Phys. Rev. B* **37**, 6991–7000 (1988).
[doi:10.1103/PhysRevB.37.6991](https://doi.org/10.1103/PhysRevB.37.6991)

Used for: the Tersoff potential, an example of a many-body term.

### Thaler2026

F. Thaler, S. Keller, "GPU-native compressed neighbor lists with a
space-filling-curve data layout," arXiv:2602.19873 (2026).
[doi:10.48550/arXiv.2602.19873](https://doi.org/10.48550/arXiv.2602.19873)

Used for: lists of clusters along a Hilbert curve whose indices are
stored as deltas in nibbles, 3.6 bytes a particle against 12 for GROMACS;
weighed against the lists of groups in the roadmap (Section 2).

### Thompson2009

A. P. Thompson, S. J. Plimpton, W. Mattson, "General formulation of
pressure and stress tensor for arbitrary many-body interaction potentials
under periodic boundary conditions," *J. Chem. Phys.* **131**, 154107
(2009).
[doi:10.1063/1.3245303](https://doi.org/10.1063/1.3245303)

Used for: the virial and the pressure of terms of more than two particles
under periodic boundaries, from the forces and the displacements within
a tuple (M1, Section 4.1 of [design-m1.md](design-m1.md)).

### Thompson2022

A. P. Thompson, H. M. Aktulga, R. Berger, et al., "LAMMPS - a flexible
simulation tool for particle-based materials modeling at the atomic,
meso, and continuum scales," *Comput. Phys. Commun.* **271**, 108171
(2022).
[doi:10.1016/j.cpc.2021.108171](https://doi.org/10.1016/j.cpc.2021.108171)

Used for: identifying LAMMPS, a system discussed in
[prior-art.md](prior-art.md).

### Tian2020

C. Tian, K. Kasavajhala, K. A. A. Belfon, L. Raguette, H. Huang,
A. N. Migues, J. Bickel, Y. Wang, J. Pincay, Q. Wu, C. Simmerling, "ff19SB:
Amino-acid-specific protein backbone parameters trained against quantum
mechanics energy surfaces in solution," *J. Chem. Theory Comput.* **16**,
528–552 (2020).
[doi:10.1021/acs.jctc.9b00591](https://doi.org/10.1021/acs.jctc.9b00591)

Used for: ff19SB, the protein force field of the target of M1 (D65), whose
CMAP terms are corrections of $\phi$ and $\psi$ for each amino acid.

### Tiesinga2021

E. Tiesinga, P. J. Mohr, D. B. Newell, B. N. Taylor, "CODATA recommended
values of the fundamental physical constants: 2018," *Rev. Mod. Phys.*
**93**, 025010 (2021).
[doi:10.1103/RevModPhys.93.025010](https://doi.org/10.1103/RevModPhys.93.025010)

Used for: the Boltzmann and Avogadro constants, from which the driver
takes the Boltzmann constant in kJ/(mol K) and the factor of the unit of
pressure.

### Tironi1995

I. G. Tironi, R. Sperb, P. E. Smith, W. F. van Gunsteren, "A generalized
reaction field method for molecular dynamics simulations," *J. Chem.
Phys.* **102**, 5451–5459 (1995).
[doi:10.1063/1.469273](https://doi.org/10.1063/1.469273)

Used for: reaction-field electrostatics, a cutoff-based alternative named
in the design review.

### Verlet1967

L. Verlet, "Computer 'experiments' on classical fluids. I.
Thermodynamical properties of Lennard-Jones molecules," *Phys. Rev.*
**159**, 98–103 (1967).
[doi:10.1103/PhysRev.159.98](https://doi.org/10.1103/PhysRev.159.98)

Used for: the Verlet integrator, of which velocity Verlet and leapfrog
are forms, and the neighbor list built with a cutoff plus a skin (the
Verlet list).

### VijayKumar1987

S. Vijay-Kumar, C. E. Bugg, W. J. Cook, "Structure of ubiquitin refined at
1.8 Å resolution," *J. Mol. Biol.* **194**, 531–544 (1987).
[doi:10.1016/0022-2836(87)90679-6](https://doi.org/10.1016/0022-2836(87)90679-6)

Used for: PDB entry 1UBQ, the protein of the comparison with GROMACS
(white paper, Section 10.6) and of the tutorial `examples/ubiquitin`.

### Wang2011

L. Wang, R. A. Friesner, B. J. Berne, "Replica exchange with solute
scaling: A more efficient version of replica exchange with solute
tempering (REST2)," *J. Phys. Chem. B* **115**, 9431–9438 (2011).
[doi:10.1021/jp204407d](https://doi.org/10.1021/jp204407d)

Used for: REST2, a protocol of `ensemble`.
