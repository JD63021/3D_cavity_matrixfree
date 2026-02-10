#!/usr/bin/env python3
"""
3D lid-driven cavity (Oseen / explicit convection) in Firedrake

Goal: Poisson-test style *matrix-free geometric multigrid* on the coupled (u,p) block.

Key features:
- MeshHierarchy (geometric MG)
- matfree operator
- MG with either:
    (a) Jacobi smoothing (fast to try, may be weak for saddle point)
    (b) Patch smoothing (stronger; needs overlap)

- CFL-based dt
- Pressure handled via nullspace (no pinning => robust across MG levels)
- ParaView-ready output via VTKFile(.pvd)

BC convention assumed for UnitCubeMesh boundary ids:
  1: x=0, 2: x=1, 3: y=0, 4: y=1, 5: z=0, 6: z=1
We drive the "lid" on y=1 (id=4) with velocity (1,0,0), all other faces no-slip.

Examples:
  Serial:
    export OMP_NUM_THREADS=1
    python cavity_3d_oseen_mg.py --base 4 --levels 3 --Re 100 --cfl 0.5 --steps 50 -ksp_converged_reason

  MPI:
    export OMP_NUM_THREADS=1
    mpiexec -n 16 python cavity_3d_oseen_mg.py --base 4 --levels 3 --Re 100 --cfl 0.5 --steps 50 -ksp_monitor -ksp_converged_reason
"""

import argparse
import sys
from time import perf_counter

# ----------------------------
# Parse args BEFORE importing Firedrake
# ----------------------------
parser = argparse.ArgumentParser(
    description="3D lid-driven cavity (Oseen) with SUPG/PSPG and matfree geometric MG"
)

# Mesh controls (geometric MG needs a hierarchy)
parser.add_argument("--base", type=int, default=4,
                    help="Coarsest mesh cells/dir (UnitCubeMesh(base,base,base))")
parser.add_argument("--levels", type=int, default=3,
                    help="Uniform refinements; finest = base * 2^levels")
parser.add_argument("--N", type=int, default=None,
                    help="(Legacy) Finest cells/dir. If set, base is computed from N and --levels.")

# Physics/time stepping
parser.add_argument("--Re", type=float, default=100.0, help="Reynolds number (nu=1/Re)")
parser.add_argument("--steps", type=int, default=50, help="Number of time steps")
parser.add_argument("--cfl", type=float, default=1.0, help="Target CFL number (dt computed from this)")
parser.add_argument("--dt-min", type=float, default=1e-6, help="Minimum dt clamp")
parser.add_argument("--dt-max", type=float, default=1e-1, help="Maximum dt clamp")

# Discretization options
parser.add_argument("--equal", action="store_true",
                    help="Use equal-order P1-P1 (else P2-P1 Taylor-Hood).")
parser.add_argument("--no-stab", action="store_true",
                    help="Disable SUPG/PSPG (recommended only for Taylor-Hood).")

# Solver options
parser.add_argument("--pc", choices=["mg", "gamg"], default="mg",
                    help="Preconditioner. 'mg' = geometric multigrid (needs hierarchy). "
                         "'gamg' = PETSc GAMG (assembled).")
parser.add_argument("--smoother", choices=["jacobi", "patch"], default="jacobi",
                    help="MG level smoother (only used if --pc mg). Patch is stronger but uses overlap.")
parser.add_argument("--rtol", type=float, default=1e-8, help="KSP relative tolerance")
parser.add_argument("--maxit", type=int, default=200, help="KSP max iterations")

# Output
parser.add_argument("--write-freq", type=int, default=5, help="VTK write frequency (in steps)")
parser.add_argument("--vtk", type=str, default="cavity3d.pvd", help="Output PVD filename")

# Parse everything, keep unknown PETSc args, intercept a couple common ones
args, petsc_argv = parser.parse_known_args()

want_ksp_monitor = False
want_ksp_reason = False
filtered = []
for a in petsc_argv:
    if a == "-ksp_monitor":
        want_ksp_monitor = True
    elif a == "-ksp_converged_reason":
        want_ksp_reason = True
    else:
        filtered.append(a)

# Only pass remaining PETSc args to Firedrake/PETSc init
sys.argv = [sys.argv[0]] + filtered

# ----------------------------
# Now import Firedrake
# ----------------------------
from firedrake import *  # noqa
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.rank


def main():
    # ---- resolve finest/base ----
    if args.N is not None:
        finestN = int(args.N)
        base = finestN // (2 ** args.levels)
        if base * (2 ** args.levels) != finestN:
            raise ValueError(f"--N={finestN} is not divisible by 2^levels={2**args.levels}. "
                             f"Choose N = base*2^levels.")
        args.base = base
    finestN = args.base * (2 ** args.levels)

    Re = float(args.Re)
    nu = Constant(1.0 / Re)

    # ---- mesh hierarchy (required for geometric MG) ----
    dparams = None
    if args.pc == "mg" and args.smoother == "patch":
        # Needed so vertex-star patches exist in parallel
        dparams = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}

    base_mesh = UnitCubeMesh(args.base, args.base, args.base, distribution_parameters=dparams)
    mh = MeshHierarchy(base_mesh, args.levels)
    mesh = mh[-1]

    # Uniform h estimate (for CFL)
    hmin = 1.0 / float(finestN)

    # ---- spaces ----
    if args.equal:
        V = VectorFunctionSpace(mesh, "CG", 1)
        Q = FunctionSpace(mesh, "CG", 1)
    else:
        V = VectorFunctionSpace(mesh, "CG", 2)
        Q = FunctionSpace(mesh, "CG", 1)
    Z = V * Q

    # Unknown/test/trial
    up = Function(Z, name="up")   # (u,p) at n+1
    u, p = split(up)
    v, q = TestFunctions(Z)
    u_trial, p_trial = TrialFunctions(Z)

    # Explicit convection velocity (u_n)
    u_n = Function(V, name="u_n")
    u_n.assign(Constant((0.0, 0.0, 0.0)))

    # dt as Constant (updated each step)
    dt = Constant(float(args.dt_max))

    # ---- BCs ----
    # UnitCubeMesh boundary ids assumed:
    # 1 x=0, 2 x=1, 3 y=0, 4 y=1, 5 z=0, 6 z=1
    # Lid at y=1 (id 4), velocity (1,0,0)
    bc_walls = DirichletBC(Z.sub(0), Constant((0.0, 0.0, 0.0)), (1, 2, 3, 5, 6))
    bc_lid   = DirichletBC(Z.sub(0), Constant((1.0, 0.0, 0.0)), (4,))
    bcs = [bc_walls, bc_lid]

    # ---- pressure nullspace (preferred over pinning; survives MG) ----
    nullspace = MixedVectorSpaceBasis(
        Z, [Z.sub(0), VectorSpaceBasis(constant=True, comm=mesh.comm)]
    )

    # ---- stabilization parameter tau(u_n,dt) ----
    h = CellDiameter(mesh)
    u_mag = sqrt(dot(u_n, u_n) + 1e-12)
    tau = 1.0 / sqrt((2.0/dt)**2 + (2.0*u_mag/h)**2 + (4.0*nu/h**2)**2)

    # ---- Oseen-like linear system (explicit convection with u_n) ----
    # (1/dt) u + (u_n·∇)u - nu Δu + ∇p = (1/dt) u_n
    # div(u) = 0
    F = (
        (1.0/dt) * inner(u_trial, v) * dx
        + inner(dot(grad(u_trial), u_n), v) * dx
        + nu * inner(grad(u_trial), grad(v)) * dx
        - inner(p_trial, div(v)) * dx
        + inner(div(u_trial), q) * dx
        - (1.0/dt) * inner(u_n, v) * dx
    )

    if not args.no_stab:
        # "Strong-ish" momentum residual without viscous second-derivative term
        r_m = (1.0/dt) * u_trial + dot(grad(u_trial), u_n) + grad(p_trial) - (1.0/dt) * u_n

        # SUPG: test with (u_n · ∇v)
        F += inner(r_m, tau * dot(grad(v), u_n)) * dx
        # PSPG: test with ∇q
        F += inner(r_m, tau * grad(q)) * dx

    a = lhs(F)
    L = rhs(F)

    # ---- solver parameters ----
    if args.pc == "gamg":
        # GAMG needs assembled operator (keep this option for comparison/debug)
        sp = {
            "mat_type": "aij",
            "ksp_type": "gmres",
            "ksp_rtol": args.rtol,
            "ksp_max_it": args.maxit,
            "pc_type": "gamg",
        }
    else:
        # Geometric MG with matfree operator (Poisson-test style)
        sp = {
            "mat_type": "matfree",
            "ksp_type": "gmres",
            "ksp_rtol": args.rtol,
            "ksp_max_it": args.maxit,
            "pc_type": "mg",

            # Coarse solve: assembled + direct
            "mg_coarse": {
                "mat_type": "aij",
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            }
        }

        if args.smoother == "jacobi":
            sp["mg_levels"] = {
                "ksp_type": "chebyshev",
                "ksp_max_it": 1,
                "pc_type": "jacobi",
            }
        else:
            # Patch smoother (stronger, usually much better for coupled systems)
            sp["mg_levels"] = {
                "ksp_type": "chebyshev",
                "ksp_max_it": 1,
                "pc_type": "python",
                "pc_python_type": "firedrake.PatchPC",
                "patch": {
                    "pc_patch": {
                        "construct_type": "star",
                        "construct_dim": 0,
                        "sub_mat_type": "seqdense",
                        "dense_inverse": True,
                        "save_operators": True,
                        "precompute_element_tensors": True,
                    },
                    "sub_ksp_type": "preonly",
                    "sub_pc_type": "lu",
                },
            }

    if want_ksp_monitor:
        sp["ksp_monitor"] = None
    if want_ksp_reason:
        sp["ksp_converged_reason"] = None

    problem = LinearVariationalProblem(a, L, up, bcs=bcs)
    solver = LinearVariationalSolver(problem, solver_parameters=sp, nullspace=nullspace)

    # ---- output ----
    u_sol, p_sol = up.subfunctions
    u_sol.rename("Velocity")
    p_sol.rename("Pressure")
    vtk = VTKFile(args.vtk)

    # ---- helpers ----
    def compute_umax(u_fun: Function) -> float:
        arr = u_fun.dat.data_ro
        local = float(np.max(np.linalg.norm(arr, axis=1))) if arr.size else 0.0
        return comm.allreduce(local, op=MPI.MAX)

    # ---- run ----
    if rank == 0:
        disc = "P1-P1" if args.equal else "P2-P1"
        stab = "SUPG/PSPG" if not args.no_stab else "no-stab"
        print(f"3D cavity | {disc} | {stab}")
        print(f"mesh: base={args.base}, levels={args.levels} => finestN={finestN}")
        print(f"Re={Re}, steps={args.steps}, CFL={args.cfl}, ranks={comm.size}")
        print(f"solver: pc={args.pc}" + (f" (smoother={args.smoother})" if args.pc == "mg" else ""))
        print(f"{'step':>6}  {'t':>10}  {'dt':>10}  {'umax':>10}  {'||u||_L2':>12}")

    t = 0.0
    t0 = perf_counter()
    for n in range(1, args.steps + 1):
        umax = compute_umax(u_n)
        uref = max(umax, 1.0)  # lid speed ~ 1; keeps dt sane early
        dt_val = args.cfl * hmin / uref
        dt_val = min(max(dt_val, args.dt_min), args.dt_max)
        dt.assign(dt_val)
        t += dt_val

        solver.solve()

        # update explicit convection velocity
        u_n.assign(u_sol)

        if (n % args.write_freq == 0) or (n == 1):
            vtk.write(u_sol, p_sol, time=t)

        do_print = (n % 5 == 0) or (n == 1)

        # assemble is collective: MUST be called by all ranks
        if do_print:
            unorm = float(sqrt(assemble(inner(u_sol, u_sol) * dx)))
            if rank == 0:
                print(f"{n:6d}  {t:10.4e}  {dt_val:10.3e}  {umax:10.3e}  {unorm:12.6e}")

    t1 = perf_counter()
    if rank == 0:
        print(f"Total wall time: {t1 - t0:.3f} s")
        print(f"Wrote: {args.vtk}")


if __name__ == "__main__":
    main()

