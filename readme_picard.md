# Firedrake 3D Driven Cavity (BDF2 + Picard + MatFree MG + Checkpoint/Restart + Re Ramp)

This repository contains a Firedrake-based solver for the incompressible Navier–Stokes driven cavity problem in 3D, using:

- Time stepping: BDF2 (with BDF1 startup on a fresh run)
- Nonlinearity: Picard (Oseen) iterations for the convective term
- Linear solver: PETSc Krylov (default fgmres) with matrix-free operator
- Preconditioner: geometric multigrid (MeshHierarchy) with Chebyshev + Jacobi smoothing
- Stabilization: SUPG/PSPG (optional, can be disabled)
- Checkpoint/restart: per-rank NPZ checkpoint files + JSON metadata
- Optional Reynolds number ramp: linearly ramp Re(t) from Re_start to Re_target over a given time window

The code is intended for research/learning and for running long simulations that need safe restart.

---

## Files

- `cavity_3d.py` (or similarly named script)
  Main solver script.

Checkpoint outputs (generated):
- `cavity3d_checkpoint.json`
- `cavity3d_checkpoint.rank0.npz`, `cavity3d_checkpoint.rank1.npz`, ...

VTK output (generated):
- `cavity3d_bdf2.pvd` and associated `.vtu` files

---

## Requirements

- Python 3
- Firedrake installed and working (MPI-enabled)
- PETSc (comes with Firedrake installs)
- mpi4py
- NumPy

Optional:
- psutil (nicer memory reporting; otherwise the script falls back to resource.getrusage)

---

## Problem Setup

Domain:
- Unit cube mesh (UnitCubeMesh) refined via MeshHierarchy

Boundary conditions (typical Firedrake UnitCubeMesh boundary ids):
- Lid: y = 1 face (boundary id 4) moving in +x direction
- Walls: no-slip on all other faces (1, 2, 3, 5, 6)

Notes:
- The boundary id mapping is the standard UnitCubeMesh mapping:
  1: x=0, 2: x=1, 3: y=0, 4: y=1, 5: z=0, 6: z=1

---

## Discretization

Velocity/pressure spaces:
- Default is equal-order P1-P1 when passing --equal
- Otherwise uses P2-P1

Pressure handling:
- The script includes an optional pressure penalty term eps_p*(p,q) controlled by --p-penalty
- If --p-penalty 0 is used, the script sets a pressure nullspace (constant pressure mode)

Stabilization:
- SUPG/PSPG is enabled by default
- Disable with --no-stab

---

## Time Stepping

- Fresh run:
  - Step 1 uses BDF1 (Backward Euler) for startup
  - Steps >= 2 use BDF2
- Restart run:
  - Continues using BDF2 (and does not redo startup)

Time step control:
- dt is computed using CFL: dt = CFL * hmin / max(umax, 1)
- dt is clamped to [--dt-min, --dt-max]
- If --t-end is provided, dt is also clipped to not overshoot t_end

---

## Picard Iterations

Within each time step, Picard iterations linearize convection by freezing the advecting velocity:

- Start Picard with u_adv = u^n
- Solve linear Oseen system for (u^(k), p^(k))
- Update u_adv = u^(k) and repeat

Picard monitoring prints:
- |du|: L2 norm of velocity change between successive Picard iterates
- rel: relative update norm, |du| / ||u_prev|| (L2)
- ||F||: a residual norm computed by assembling a residual form into a dual-space vector (see "Residual Norm Notes" below)

Stopping rule (typical in this repo):
- Stop when (||F|| < picard_atol) AND (rel < picard_rtol)

Controls:
- --picard-maxit
- --picard-rtol
- --picard-atol
- --picard-monitor

---

## Reynolds Number Ramp (Optional)

You pass --Re as the target Reynolds number. If ramping is enabled, the effective Re changes with time:

- --Re-start  starting Re at t=0
- --Re-ramp   duration in seconds (if 0, ramp is disabled)

Re(t) behavior:
- For t < ramp_time, Re_eff(t) increases linearly from Re_start to Re_target
- For t >= ramp_time, Re_eff(t) = Re_target
- Viscosity is updated each time step as nu = 1 / Re_eff(t)

This is useful for difficult target Reynolds numbers where directly starting at high Re causes nonconvergence.

---

## Checkpoint / Restart

Checkpoint format:
- Per MPI rank: <prefix>.rank<r>.npz
- Metadata JSON: <prefix>.json

What is stored:
- u^n, u^{n-1}, p^n, p^{n-1} (local arrays per rank)
- step, time, dt_last, mesh/discretization metadata, MPI size

Controls:
- --checkpoint-prefix   (default: cavity3d_checkpoint)
- --checkpoint-freq     write every N steps (0 disables periodic checkpointing)
- --restart             read checkpoint and continue

Important restart rules:
- Restart must use the same mesh/discretization arguments and the same MPI size.
- If MPI size changes, rank-local arrays will not match.

Signal behavior:
- On Ctrl+C (SIGINT) or SIGTERM, the script sets a stop flag,
  finishes the current step, attempts a checkpoint write, then exits cleanly.

---

## Running

### Basic run (single process)
python cavity_3d.py --base 4 --levels 3 --Re 500 --cfl 1 --t-end 2.0 --steps 1000000 --equal --picard-maxit 20 --picard-monitor --qdeg 4

### MPI run
mpiexec -n 16 python cavity_3d.py --base 4 --levels 4 --Re 2000 --cfl 1 --t-end 10.0 --steps 1000000 --equal --picard-maxit 20 --picard-monitor --qdeg 4 --checkpoint-freq 10

### Restart from checkpoint
mpiexec -n 16 python cavity_3d.py --base 4 --levels 4 --Re 2000 --cfl 1 --t-end 10.0 --steps 1000000 --equal --picard-maxit 20 --picard-monitor --qdeg 4 --checkpoint-freq 10 --restart

### Enable Re ramp from 100 to target Re over 1 second
mpiexec -n 16 python cavity_3d.py --base 4 --levels 4 --Re 2000 --Re-start 100 --Re-ramp 1.0 --cfl 1 --t-end 10.0 --steps 1000000 --equal --picard-maxit 20 --picard-monitor --qdeg 4 --checkpoint-freq 10

---

## PETSc Solver Options

The script sets solver defaults in Python via solver_parameters, but you can also pass PETSc options on the command line.

Common examples:
- -ksp_converged_reason
- -ksp_monitor
- -ksp_rtol <val>
- -ksp_atol <val>
- -ksp_max_it <val>

Note:
- If you override tolerances on the command line, make sure the script does not overwrite them later.
- You can validate what PETSc is using with:
  -ksp_view
  -ksp_converged_reason
  -ksp_monitor_true_residual (if supported in your PETSc build)

---

## Residual Norm Notes (||F||)

The printed ||F|| in this code is computed by:
- assembling a residual form into a dual-space vector (Cofunction)
- applying boundary condition zeroing on the residual vector
- taking the PETSc Vec norm of that assembled residual vector

This is a useful diagnostic, but it is not necessarily:
- the L2 norm of the strong-form PDE residual, nor
- directly comparable across meshes or discretizations without care

If you need a specific residual definition (e.g., L2 norm of strong residual, or a properly scaled weak residual),
modify the residual evaluation section accordingly.

---

## Output

VTK output:
- Written every --write-freq steps and on step 1
- File: cavity3d_bdf2.pvd

Memory output:
- Printed after setup and periodically every --mem-freq steps (plus step 1)
- Shows RSS max per rank and sum across ranks

---

## Tips / Troubleshooting

- If high target Re does not converge from rest, use Re ramping.
- If restart appears to do nothing, check the checkpoint metadata:
  - step and t values
  - confirm your --t-end is greater than checkpoint time
- If checkpoint writing fails:
  - ensure the working directory is writable
  - ensure no stale *.tmp files remain
  - verify that the prefix path exists (the script attempts to create it)


---

## Acknowledgements

- Firedrake project
- PETSc project
- mpi4py and NumPy
