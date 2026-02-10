# 3D Lid-Driven Cavity (Oseen + SUPG/PSPG) in Firedrake — Matrix-Free Geometric Multigrid

This repository contains a Firedrake implementation of a **3D lid-driven cavity** solver designed specifically to explore:

- **matrix-free** operator application,
- **geometric multigrid (GMG)** on a mesh hierarchy,
- stabilized **equal-order** (P1–P1) and Taylor–Hood (P2–P1) discretizations,
- and practical performance behavior (iterations vs wall-time vs memory) on CPU + MPI.

The goal is not to build a “full Navier–Stokes production code”, but to create a **clean testbed** for multigrid + Krylov behavior on the (velocity, pressure) system, similar in spirit to a “Poisson multigrid test”, but on a harder saddle-point problem.

---

## What this solver is doing 

### 1) Time stepping with an semi implicit-style linearization
At each time step, we solve a **linearized** incompressible flow system where the convective velocity is **frozen** from the previous state. 

- If you run “one linear solve per time step”, the convection field is taken from the previous time step.
- If you later add a Picard loop inside each time step, you can update the linearization velocity between inner iterations (not enabled by default in the stable/working baseline).

This approach is **more stable** than putting all convection purely on the right-hand side explicitly, while still being much cheaper than a full Newton solve.

### 2) Stabilization for equal-order elements (SUPG/PSPG)
For **P1–P1** (equal-order) velocity/pressure, the pressure space is not stable by itself for incompressible flow. To make it work robustly, we add:

- **SUPG**-style stabilization for momentum advection-dominated behavior
- **PSPG** stabilization to control pressure modes for equal-order pairs

This makes P1–P1 usable in practice, including in 3D and with MPI.

> Note: We intentionally avoid adding raw grad-div stabilization by default in this repo, since it can distort the cavity vortex structure unless carefully tuned.

### 3) Monolithic coupled solve
We solve velocity and pressure together as a **single coupled system** using GMRES. This is different from segregated approaches (like SIMPLE/PISO style splitting).

---

## Multigrid + Matrix-Free: why this is interesting

### Matrix-free operator
When `mat_type` is set to `matfree`, Firedrake applies the operator without assembling the global sparse matrix. This can:

- reduce memory (no AIJ storage on fine levels),
- reduce assembly overhead,
- enable strong-scaling experiments where assembly becomes expensive.

### Geometric multigrid (GMG) on a hierarchy
We use a **MeshHierarchy** so that multigrid is geometric: coarse grids are created by uniform coarsening (reverse refinement). This avoids the heavy graph/coarsening metadata typical of algebraic multigrid.

The coarse level is typically assembled and solved directly (LU), while the fine levels remain matrix-free.

---

## Mesh and problem setup

### Domain
- Unit cube cavity: `[0,1] x [0,1] x [0,1]`

### Boundary conditions (standard cavity)
- No-slip on all walls
- Lid motion on the top boundary (typically `y = 1`): velocity is set to `(1, 0, 0)`

### Mesh type
- Firedrake’s `UnitCubeMesh` uses a **tetrahedral mesh** (important for DOF counts vs hex meshes).

---

## Running the code

## RUNNING (MPI) AND BASIC CONTROLS

Important: avoid oversubscription when using MPI.
Set OpenMP threads to 1 so you don’t accidentally run (MPI ranks × OpenMP threads) total threads.

Example: 16 MPI ranks, equal-order P1–P1

export OMP_NUM_THREADS=1
mpiexec -n 16 python cavity_3d_oseen_mg.py --base 3 --levels 4 --Re 10 --cfl 1 --steps 25 --equal --pc mg --no-output -ksp_converged_reason

---

## HOW TO MAKE THE GRID COARSER / FINER

The finest resolution is controlled by:
--base   : cells per direction on the coarsest mesh
--levels : number of uniform refinements

Finest cells per direction is:
finestN = base * 2^levels

Examples:
--base 3 --levels 4  gives  finestN = 48
--base 4 --levels 4  gives  finestN = 64
--base 4 --levels 3  gives  finestN = 32

Practical note: for some 3D setups, geometric multigrid can fail or become fragile with too few levels. In our tests, 4 levels was often much more robust than 3 for this coupled (u,p) system.

---

## SOLVER OPTIONS

Preconditioner choices:
--pc mg    uses geometric multigrid (matrix-free on most levels)
--pc gamg  uses PETSc GAMG (assembled operator)

Smoother choices for MG (used only when --pc mg):
--smoother jacobi  fast to try, sometimes fragile for saddle-point systems
--smoother patch   stronger and more robust, but needs overlap and costs more setup

---

## OBSERVATIONS (ITERATIONS vs WALL TIME vs MEMORY)

These are practical outcomes observed while testing (not universal truths).

1. MG can reduce GMRES iterations significantly
   With GMG, the coupled solve often converges in roughly 10–25 iterations per step (depends on CFL, Re, mesh, and element choice).

2. For 2D GAMG can be “more iterations but faster”
   In 2D runs, GAMG required many more GMRES iterations per solve (for example 80–150), but the overall wall time was still low because the per-iteration cost was small and the implementation is highly optimized.
   In 3D the performance of MG is faster and it also uses less RAM.

3. Memory behavior: matrix-free GMG vs assembled GAMG
   Matrix-free GMG tends to use less RAM than fully assembled approaches on fine grids because it avoids storing the large sparse matrix.
   However, peak memory can still rise during solves due to Krylov basis storage (GMRES) and multigrid level data.
   A common pattern is that RSS increases while the solve is active, then drops when the solve finishes. This is consistent with temporary Krylov workspace and solver objects.

4. Back-of-the-envelope RAM comparison vs OpenFOAM (practical note)
   On similar DOF scales of 1M and core counts (16 cores with 64 GB RAM setup), Firedrake monolithic (u,p) plus stabilization plus Krylov+MG can use maround 5GB RAM as against 2-3 GB for the OpenFOAM SIMPLE-style segregated solve.
   This comparison is not 1:1 because OpenFOAM often solves U and p in separate stages (segregated), while this Firedrake implementation is monolithic and may allocate larger Krylov workspaces. Mesh type (tet vs poly/hex) and discretization also matter.
   Takeaway: FEM taking ~2× the RAM of a segregated FV SIMPLE solve make FEM somewhat more practical and attractive; though it still appears to be a long shot from an FV based solver
---

## COMMON PITFALLS / TROUBLESHOOTING

1. MPI appears “stuck” when printing
   In Firedrake, some operations (notably assemble) are collective across MPI ranks. If only rank 0 calls assemble inside a conditional print block, the run can deadlock.
   Fix pattern: all ranks do the assemble, only rank 0 prints. (The working code follows this rule.)

2. MG failures with too few levels
   With coupled saddle-point operators (u,p), MG can fail due to coarse-grid issues or weak smoothers. If you see DIVERGED_PC_FAILED:

* try --levels 4
* try a stronger smoother (patch)
* or switch to --pc gamg as a baseline.

3. Equal-order behavior
   P1–P1 needs PSPG/SUPG. If you disable stabilization on P1–P1, expect instability.

---

## OUTPUTS

By default the code writes ParaView-ready output (*.pvd) containing velocity and pressure.
Use --no-output to disable I/O for performance tests.


## Acknowledgements
This experimental code is made possible by the Firedrake project and its authors/contributors. The mesh heirarchy and preconditioner configurations used here follow established Firedrake demo patterns.
