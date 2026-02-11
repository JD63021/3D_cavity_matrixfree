#!/usr/bin/env python3
"""
3D driven cavity: Picard nonlinear convection with matfree geometric MG (MeshHierarchy)
and Chebyshev+Jacobi smoothing + restart checkpoints (NPZ per-rank)
+ TRUE residual Picard stopping
+ Re ramping: linearly ramp Re(t) from Re_start to Re_target over ramp_time seconds.

Ramp behavior:
- You still pass --Re as the TARGET Re (final value after ramp).
- New args:
    --Re-start   : starting Re at t=0 (default 100)
    --Re-ramp    : ramp duration in seconds (default 1.0). If 0 => no ramp.
- Effective viscosity is nu(t) = 1 / Re_eff(t), where
    Re_eff(t) = Re_start + (Re_target - Re_start) * min(t/ramp_time, 1).

IMPORTANT:
- This updates nu (a Constant) every time step based on current time t.
- It also rebuilds the linear forms/solver ONCE per time step (as before), so nu changes are applied.

Everything else stays the same.
"""

import os
import argparse
import sys
import json
import signal
from time import perf_counter
import resource  # RSS fallback

# ----------------------------
# Optional psutil (nicer RSS)
# ----------------------------
try:
    import psutil  # type: ignore
    HAVE_PSUTIL = True
except Exception:
    HAVE_PSUTIL = False

os.environ.setdefault("OMP_NUM_THREADS", "1")

# ----------------------------
# Parse args BEFORE importing Firedrake
# ----------------------------
parser = argparse.ArgumentParser(
    description="3D cavity Picard + matfree MG, timestep-only PC rebuild, BDF2 + restart checkpoints (NPZ) + true residual + Re ramp"
)

parser.add_argument("--base", type=int, default=8)
parser.add_argument("--levels", type=int, default=2)
parser.add_argument("--N", type=int, default=None,
                    help="If set, uses finestN=N with base = N/2^levels (must be integer).")

# Target Re (final)
parser.add_argument("--Re", type=float, default=100.0, help="TARGET Reynolds number after ramp.")
# Ramp controls
parser.add_argument("--Re-start", type=float, default=100.0, help="Starting Re at t=0 for ramp.")
parser.add_argument("--Re-ramp", type=float, default=1.0, help="Ramp duration in seconds. 0 disables ramp.")

parser.add_argument("--steps", type=int, default=50,
                    help="Maximum number of time steps (always honored).")
parser.add_argument("--t-end", type=float, default=None,
                    help="Final simulation time. If set, stop when t >= t_end (in addition to steps).")

parser.add_argument("--cfl", type=float, default=1.0)
parser.add_argument("--dt-min", type=float, default=1e-6)
parser.add_argument("--dt-max", type=float, default=1e-1)

parser.add_argument("--equal", action="store_true",
                    help="Use equal-order P1-P1 (else P2-P1).")
parser.add_argument("--no-stab", action="store_true",
                    help="Disable SUPG/PSPG.")

parser.add_argument("--qdeg", type=int, default=4,
                    help="Quadrature degree for operator integrals.")

parser.add_argument("--rtol", type=float, default=1e-3)
parser.add_argument("--maxit", type=int, default=200)
parser.add_argument("--divtol", type=float, default=1e30)

parser.add_argument("--mg-level-sweeps", type=int, default=2)

parser.add_argument("--picard-maxit", type=int, default=10)
parser.add_argument("--picard-rtol", type=float, default=1e-6,
                    help="Picard relative tolerance on ||du||/||u_prev||.")
parser.add_argument("--picard-atol", type=float, default=1e-5,
                    help="Picard absolute tolerance on TRUE residual norm ||F||.")
parser.add_argument("--picard-monitor", action="store_true")

# pressure penalty (default ON)
parser.add_argument("--p-penalty", type=float, default=1e-15,
                    help="Pressure mass penalty eps_p*(p,q). Use 0 to restore pressure nullspace.")

parser.add_argument("--write-freq", type=int, default=5)
parser.add_argument("--vtk", type=str, default="cavity3d_bdf2.pvd")

# ----------------------------
# Restart / checkpointing (NPZ)
# ----------------------------
parser.add_argument("--checkpoint-prefix", type=str, default="cavity3d_checkpoint",
                    help="Prefix for checkpoint files. Writes <prefix>.rank<i>.npz and <prefix>.json")
parser.add_argument("--checkpoint-freq", type=int, default=50,
                    help="Write checkpoint every N steps. 0 disables periodic checkpointing.")
parser.add_argument("--restart", action="store_true",
                    help="Restart from checkpoint if files exist.")

# Memory reporting
parser.add_argument("--mem-freq", type=int, default=10,
                    help="Print memory usage every N steps (and at step 1). 0 disables.")
parser.add_argument("--no-psutil", action="store_true",
                    help="Do not use psutil even if installed; use resource.getrusage fallback.")

args, petsc_argv = parser.parse_known_args()

# Preserve PETSc monitoring flags if user passes them (keep all other PETSc flags!)
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
sys.argv = [sys.argv[0]] + filtered

# ----------------------------
# Now import Firedrake
# ----------------------------
from firedrake import *  # noqa
from firedrake import Cofunction
from firedrake.exceptions import ConvergenceError
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.rank

# ----------------------------
# Graceful stop flag (Ctrl+C / SIGTERM)
# ----------------------------
STOP_REQUESTED = False


def _handle_stop_signal(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


signal.signal(signal.SIGINT, _handle_stop_signal)
signal.signal(signal.SIGTERM, _handle_stop_signal)


def main():
    # ---- resolve finest/base ----
    if args.N is not None:
        finestN = int(args.N)
        base = finestN // (2 ** args.levels)
        if base * (2 ** args.levels) != finestN:
            raise ValueError("Choose N = base*2^levels.")
        args.base = base
    finestN = args.base * (2 ** args.levels)

    Re_target = float(args.Re)
    Re_start = float(args.Re_start)
    ramp_time = float(args.Re_ramp)

    # nu is a Constant we will update every time step based on t
    nu = Constant(1.0 / max(Re_start, 1e-16))

    # 3D mesh hierarchy
    base_mesh = UnitCubeMesh(args.base, args.base, args.base)
    mh = MeshHierarchy(base_mesh, args.levels)
    mesh = mh[-1]

    # crude hmin for CFL (uniform refinement => ~1/finestN)
    hmin = 1.0 / float(finestN)

    # spaces
    if args.equal:
        V = VectorFunctionSpace(mesh, "CG", 1)  # 3D vector
        Q = FunctionSpace(mesh, "CG", 1)
    else:
        V = VectorFunctionSpace(mesh, "CG", 2)
        Q = FunctionSpace(mesh, "CG", 1)
    Z = V * Q

    # ---- exact DOF counts (global + max local) ----
    Vdim_g = V.dim()
    Qdim_g = Q.dim()
    Vdim_l = V.dof_dset.layout_vec.getLocalSize()
    Qdim_l = Q.dof_dset.layout_vec.getLocalSize()
    Vdim_lmax = comm.allreduce(Vdim_l, op=MPI.MAX)
    Qdim_lmax = comm.allreduce(Qdim_l, op=MPI.MAX)

    # ---- memory reporting utilities ----
    use_psutil = (HAVE_PSUTIL and (not args.no_psutil))

    def current_rss_mb() -> float:
        if use_psutil:
            p = psutil.Process(os.getpid())  # type: ignore[name-defined]
            return float(p.memory_info().rss) / (1024.0**2)
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0

    def report_mem(tag: str = ""):
        rss_mb = current_rss_mb()
        rss_max = comm.allreduce(rss_mb, op=MPI.MAX)
        rss_sum = comm.allreduce(rss_mb, op=MPI.SUM)
        if rank == 0:
            print(f"[MEM]{tag}  RSS max/rank = {rss_max:8.1f} MB | RSS sum = {rss_sum/1024.0:8.2f} GB")

    # Unknowns/tests/trials
    up = Function(Z, name="up")
    u, p = split(up)
    v, q = TestFunctions(Z)
    u_trial, p_trial = TrialFunctions(Z)

    # History (velocity)
    u_n = Function(V, name="u_n")
    u_n.assign(Constant((0.0, 0.0, 0.0)))  # u^n
    u_nm1 = Function(V, name="u_nm1")
    u_nm1.assign(u_n)  # u^{n-1}

    # History (pressure) only for initial guess convenience
    p_n = Function(Q, name="p_n")
    p_n.assign(0.0)
    p_nm1 = Function(Q, name="p_nm1")
    p_nm1.assign(p_n)

    # Picard advection and iteration bookkeeping
    u_adv = Function(V, name="u_adv")
    u_adv.assign(u_n)
    u_prev_it = Function(V, name="u_prev_it")

    dt = Constant(float(args.dt_max))

    # Boundary IDs (typical): 1:x=0 2:x=1 3:y=0 4:y=1 5:z=0 6:z=1
    bc_lid = DirichletBC(Z.sub(0), Constant((1.0, 0.0, 0.0)), (4,))
    bc_walls = DirichletBC(Z.sub(0), Constant((0.0, 0.0, 0.0)), (1, 2, 3, 5, 6))
    bcs = [bc_lid, bc_walls]

    # Pressure nullspace handling
    if args.p_penalty > 0.0:
        nullspace = None
    else:
        nullspace = MixedVectorSpaceBasis(
            Z, [Z.sub(0), VectorSpaceBasis(constant=True, comm=mesh.comm)]
        )

    dxq = dx(metadata={"quadrature_degree": int(args.qdeg)})

    # tau(u_adv, dt): SAME for BDF1 and BDF2 (uses current nu)
    h = CellDiameter(mesh)
    u_mag = sqrt(dot(u_adv, u_adv) + 1e-12)
    tau = 1.0 / sqrt((2.0 / dt) ** 2 + (2.0 * u_mag / h) ** 2 + (4.0 * nu / h ** 2) ** 2)

    eps_p = Constant(float(args.p_penalty))

    # ----------------------------
    # Helper: Re ramp
    # ----------------------------
    def Re_eff(time_val: float) -> float:
        if ramp_time <= 0.0:
            return Re_target
        s = min(max(time_val / ramp_time, 0.0), 1.0)
        return Re_start + (Re_target - Re_start) * s

    # ----------------------------
    # Solver parameters
    # ----------------------------
    sp = {
        "mat_type": "matfree",
        "ksp_type": "gmres",
        "ksp_rtol": args.rtol,
        "ksp_max_it": args.maxit,
        "ksp_divtol": args.divtol,
        "ksp_initial_guess_nonzero": True,
        "pc_type": "mg",
        "mg_coarse": {
            "mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        "mg_levels": {
            "ksp_type": "chebyshev",
            "ksp_max_it": max(1, int(args.mg_level_sweeps)),
            "pc_type": "jacobi",
        },
    }
    if want_ksp_monitor:
        sp["ksp_monitor"] = None
    if want_ksp_reason:
        sp["ksp_converged_reason"] = None

    # output
    u_sol, p_sol = up.subfunctions
    u_sol.rename("Velocity")
    p_sol.rename("Pressure")
    vtk = VTKFile(args.vtk)

    def compute_umax(u_fun: Function) -> float:
        arr = u_fun.dat.data_ro  # shape (nlocal, 3)
        local = float(np.max(np.linalg.norm(arr, axis=1))) if arr.size else 0.0
        return comm.allreduce(local, op=MPI.MAX)

    # ----------------------------
    # Residual storage
    # ----------------------------
    R_true = Cofunction(Z.dual(), name="R_true")

    def _zero_cofunction(cof: Cofunction):
        try:
            for d in cof.dat:
                d.data[:] = 0.0
        except TypeError:
            cof.dat.data[:] = 0.0

    # ----------------------------
    # NPZ checkpointing (per-rank) + JSON meta
    # ----------------------------
    chk_prefix = args.checkpoint_prefix
    chk_meta = chk_prefix + ".json"

    chk_dir = os.path.dirname(os.path.abspath(chk_prefix))
    if rank == 0:
        os.makedirs(chk_dir, exist_ok=True)
    comm.barrier()

    def chk_rank_file(r: int) -> str:
        return f"{chk_prefix}.rank{r}.npz"

    CHK_DISABLED = False

    def _write_meta_atomic(meta: dict):
        if rank == 0:
            tmp = chk_meta + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, sort_keys=True)
            os.replace(tmp, chk_meta)
        comm.barrier()

    def _read_meta():
        meta = None
        if rank == 0 and os.path.exists(chk_meta):
            with open(chk_meta, "r", encoding="utf-8") as f:
                meta = json.load(f)
        meta = comm.bcast(meta, root=0)
        return meta

    def save_checkpoint(step: int, time_val: float):
        """
        Atomic NPZ checkpoint:
        - each rank writes to <prefix>.rank<i>.npz.tmp then renames
        - IMPORTANT: write via file-handle so NumPy does NOT auto-append ".npz"
        - rank 0 writes JSON meta AFTER all ranks succeed
        """
        tmp_file = chk_rank_file(rank) + ".tmp"
        final_file = chk_rank_file(rank)

        if os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except Exception:
                pass
        comm.barrier()

        u_n_arr = np.array(u_n.dat.data_ro, copy=True)
        u_nm1_arr = np.array(u_nm1.dat.data_ro, copy=True)
        p_n_arr = np.array(p_n.dat.data_ro, copy=True)
        p_nm1_arr = np.array(p_nm1.dat.data_ro, copy=True)

        with open(tmp_file, "wb") as f:
            np.savez(f, u_n=u_n_arr, u_nm1=u_nm1_arr, p_n=p_n_arr, p_nm1=p_nm1_arr)

        comm.barrier()
        os.replace(tmp_file, final_file)
        comm.barrier()

        meta = {
            "step": int(step),
            "t": float(time_val),
            "dt_last": float(dt.dat.data_ro[0]),
            "base": int(args.base),
            "levels": int(args.levels),
            "finestN": int(finestN),
            "equal": bool(args.equal),
            "Re_target": float(Re_target),
            "Re_start": float(Re_start),
            "Re_ramp": float(ramp_time),
            "p_penalty": float(args.p_penalty),
            "comm_size": int(comm.size),
            "V_local_shape": list(u_n.dat.data_ro.shape),
            "Q_local_shape": list(p_n.dat.data_ro.shape),
        }
        _write_meta_atomic(meta)

        if rank == 0:
            print(f"[CHK] wrote checkpoint at step={step}, t={time_val:.6e} -> {chk_prefix}.rank*.npz")

    def safe_save_checkpoint(step: int, time_val: float):
        nonlocal CHK_DISABLED
        if CHK_DISABLED:
            return
        try:
            save_checkpoint(step, time_val)
        except Exception as e:
            if rank == 0:
                print(f"[CHK][ERROR] checkpoint write failed at step={step}, t={time_val:.6e}: {e}")
                print("[CHK][WARN] disabling further checkpoint writes (simulation continues).")
            CHK_DISABLED = True
            comm.barrier()

    def load_checkpoint():
        meta = _read_meta()
        if meta is None:
            if rank == 0:
                print("[CHK] no metadata file found; cannot restart.")
            return None

        def warn(msg: str):
            if rank == 0:
                print(f"[CHK][WARN] {msg}")

        if int(meta.get("comm_size", comm.size)) != int(comm.size):
            warn(f"checkpoint comm_size={meta.get('comm_size')} != current comm.size={comm.size} (restart likely invalid)")
        if int(meta.get("base", args.base)) != int(args.base):
            warn(f"checkpoint base={meta.get('base')} != args.base={args.base}")
        if int(meta.get("levels", args.levels)) != int(args.levels):
            warn(f"checkpoint levels={meta.get('levels')} != args.levels={args.levels}")
        if bool(meta.get("equal", args.equal)) != bool(args.equal):
            warn(f"checkpoint equal={meta.get('equal')} != args.equal={args.equal}")

        fpath = chk_rank_file(rank)
        if not os.path.exists(fpath):
            if rank == 0:
                print(f"[CHK] missing rank file {fpath}; cannot restart.")
            return None
        comm.barrier()

        data = np.load(fpath)
        u_n_arr = data["u_n"]
        u_nm1_arr = data["u_nm1"]
        p_n_arr = data["p_n"]
        p_nm1_arr = data["p_nm1"]

        if u_n_arr.shape != u_n.dat.data_ro.shape:
            warn(f"u_n shape mismatch: file {u_n_arr.shape} vs current {u_n.dat.data_ro.shape}")
        if p_n_arr.shape != p_n.dat.data_ro.shape:
            warn(f"p_n shape mismatch: file {p_n_arr.shape} vs current {p_n.dat.data_ro.shape}")

        u_n.dat.data[:] = u_n_arr
        u_nm1.dat.data[:] = u_nm1_arr
        p_n.dat.data[:] = p_n_arr
        p_nm1.dat.data[:] = p_nm1_arr

        comm.barrier()
        if rank == 0:
            print(f"[CHK] restarted from step={meta.get('step')}, t={float(meta.get('t', 0.0)):.6e}")
        return meta

    # ----------------------------
    # Header prints
    # ----------------------------
    if rank == 0:
        disc = "P1-P1" if args.equal else "P2-P1"
        stab = "SUPG/PSPG" if not args.no_stab else "no-stab"
        print("---- DOF counts (exact) ----")
        print(f"finestN = {finestN}")
        print(f"Discretization: {disc}")
        print(f"Velocity space V : global dofs = {Vdim_g:,}, max local = {Vdim_lmax:,}")
        print(f"Pressure space Q : global dofs = {Qdim_g:,}, max local = {Qdim_lmax:,}")
        print(f"Total (V+Q)      : global dofs = {(Vdim_g + Qdim_g):,}")
        print("---------------------------")
        print(f"3D cavity BDF2 | {disc} | {stab} | qdeg={args.qdeg} | ppen={args.p_penalty:g}")
        print(f"mesh: base={args.base}, levels={args.levels} => finestN={finestN}")
        print(f"Re target={Re_target}, Re_start={Re_start}, Re_ramp={ramp_time}s")
        print(f"steps={args.steps}, t_end={args.t_end}, CFL={args.cfl}, ranks={comm.size}")
        print("PC refresh: timestep-only (no rebuild inside Picard)")
        print("Time stepping: step1=BDF1 (fresh only), steps>=2=BDF2")
        print(f"Linear: rtol={args.rtol}, maxit={args.maxit}, divtol={args.divtol:g}")
        print("Picard stop: (||F|| < atol) AND (||du||/||u_prev|| < rtol)")
        print(f"Picard: maxit={args.picard_maxit}, rtol={args.picard_rtol}, atol={args.picard_atol}  (atol is TRUE residual)")
        print(f"Checkpoint(NPZ): prefix={chk_prefix}, freq={args.checkpoint_freq} (0=off), restart={args.restart}")
        print(f"{'step':>6}  {'mode':>6}  {'t':>10}  {'dt':>10}  {'umax':>10}  {'Re_eff':>10}  {'||u||_L2':>12}  {'mem_maxMB':>10}  {'mem_sumGB':>10}")

    report_mem(" after setup")

    # ----------------------------
    # Restart if requested
    # ----------------------------
    t = 0.0
    start_step = 1

    if args.restart:
        meta = load_checkpoint()
        if meta is not None:
            t = float(meta.get("t", 0.0))
            last_step = int(meta.get("step", 0))
            start_step = max(1, last_step + 1)

    if args.t_end is not None and t >= float(args.t_end):
        if rank == 0:
            print(f"t={t:.6e} already >= t_end={float(args.t_end):.6e}; nothing to do.")
        return

    # ----------------------------
    # Time loop
    # ----------------------------
    wall0 = perf_counter()
    last_completed_step = start_step - 1

    try:
        for n in range(start_step, args.steps + 1):
            if args.t_end is not None and t >= float(args.t_end):
                break

            # dt from CFL using u^n
            umax = compute_umax(u_n)
            uref = max(umax, 1.0)
            dt_val = args.cfl * hmin / uref
            dt_val = min(max(dt_val, args.dt_min), args.dt_max)

            # don't overshoot t_end
            if args.t_end is not None:
                remaining = float(args.t_end) - t
                if remaining <= 0.0:
                    break
                if dt_val > remaining:
                    dt_val = remaining

            dt.assign(dt_val)
            t += dt_val

            # Update nu via Re ramp at the CURRENT time t
            Re_now = Re_eff(t)
            nu.assign(1.0 / max(Re_now, 1e-16))

            # Choose scheme
            if n == 1 and start_step == 1:
                mode = "BDF1"
            else:
                mode = "BDF2"

            # (Re)build forms for this time step (so nu(t) is captured)
            dt_inv = 1.0 / dt
            bdf2_fac = 1.0 / (2.0 * dt)

            # linearized forms depend on u_adv; defined as UFL expr so OK to rebuild each step
            F1 = (
                dt_inv * inner(u_trial, v) * dxq
                + inner(dot(grad(u_trial), u_adv), v) * dxq
                + nu * inner(grad(u_trial), grad(v)) * dxq
                - inner(p_trial, div(v)) * dxq
                + inner(div(u_trial), q) * dxq
                + eps_p * inner(p_trial, q) * dxq
                - dt_inv * inner(u_n, v) * dxq
            )
            if not args.no_stab:
                r_m1 = dt_inv * u_trial + dot(grad(u_trial), u_adv) + grad(p_trial) - dt_inv * u_n
                F1 += inner(r_m1, tau * dot(grad(v), u_adv)) * dxq
                F1 += inner(r_m1, tau * grad(q)) * dxq
            a1 = lhs(F1)
            L1 = rhs(F1)

            F2 = (
                (3.0 * bdf2_fac) * inner(u_trial, v) * dxq
                + inner(dot(grad(u_trial), u_adv), v) * dxq
                + nu * inner(grad(u_trial), grad(v)) * dxq
                - inner(p_trial, div(v)) * dxq
                + inner(div(u_trial), q) * dxq
                + eps_p * inner(p_trial, q) * dxq
                - bdf2_fac * inner(4.0 * u_n - u_nm1, v) * dxq
            )
            if not args.no_stab:
                r_m2 = (3.0 * bdf2_fac) * u_trial + dot(grad(u_trial), u_adv) + grad(p_trial) - bdf2_fac * (4.0 * u_n - u_nm1)
                F2 += inner(r_m2, tau * dot(grad(v), u_adv)) * dxq
                F2 += inner(r_m2, tau * grad(q)) * dxq
            a2 = lhs(F2)
            L2 = rhs(F2)

            # TRUE residual forms for this step
            F1_true = (
                dt_inv * inner(u, v) * dxq
                + inner(dot(grad(u), u), v) * dxq
                + nu * inner(grad(u), grad(v)) * dxq
                - inner(p, div(v)) * dxq
                + inner(div(u), q) * dxq
                + eps_p * inner(p, q) * dxq
                - dt_inv * inner(u_n, v) * dxq
            )
            if not args.no_stab:
                r_m1_true = dt_inv * u + dot(grad(u), u) + grad(p) - dt_inv * u_n
                F1_true += inner(r_m1_true, tau * dot(grad(v), u)) * dxq
                F1_true += inner(r_m1_true, tau * grad(q)) * dxq

            F2_true = (
                (3.0 * bdf2_fac) * inner(u, v) * dxq
                + inner(dot(grad(u), u), v) * dxq
                + nu * inner(grad(u), grad(v)) * dxq
                - inner(p, div(v)) * dxq
                + inner(div(u), q) * dxq
                + eps_p * inner(p, q) * dxq
                - bdf2_fac * inner(4.0 * u_n - u_nm1, v) * dxq
            )
            if not args.no_stab:
                r_m2_true = (3.0 * bdf2_fac) * u + dot(grad(u), u) + grad(p) - bdf2_fac * (4.0 * u_n - u_nm1)
                F2_true += inner(r_m2_true, tau * dot(grad(v), u)) * dxq
                F2_true += inner(r_m2_true, tau * grad(q)) * dxq

            def true_residual_norm(local_mode: str) -> float:
                _zero_cofunction(R_true)
                if local_mode == "BDF1":
                    assemble(F1_true, tensor=R_true)
                else:
                    assemble(F2_true, tensor=R_true)
                for bc in bcs:
                    bc.zero(R_true)
                with R_true.dat.vec_ro as rv:
                    return float(rv.norm())

            # Choose linear system and build solver once per time step
            if mode == "BDF1":
                a_form, L_form = a1, L1
            else:
                a_form, L_form = a2, L2

            # Picard starts from last time step
            u_adv.assign(u_n)
            u_sol.assign(u_n)
            p_sol.assign(p_n)

            solver = LinearVariationalSolver(
                LinearVariationalProblem(a_form, L_form, up, bcs=bcs),
                solver_parameters=sp,
                nullspace=nullspace,
            )

            for k in range(1, args.picard_maxit + 1):
                u_prev_it.assign(u_sol)

                try:
                    solver.solve()
                except ConvergenceError:
                    solver = LinearVariationalSolver(
                        LinearVariationalProblem(a_form, L_form, up, bcs=bcs),
                        solver_parameters=sp,
                        nullspace=nullspace,
                    )
                    solver.solve()

                du = float(np.sqrt(assemble(inner(u_sol - u_prev_it, u_sol - u_prev_it) * dx)))
                u_prev_norm = float(np.sqrt(assemble(inner(u_prev_it, u_prev_it) * dx)))
                rel = du / max(u_prev_norm, 1e-16)

                # update advection then compute true residual
                u_adv.assign(u_sol)
                rnorm = true_residual_norm(mode)

                if args.picard_monitor and rank == 0:
                    print(f"  Picard {k:2d}: |du|={du:.3e}, rel={rel:.3e}, ||F||={rnorm:.3e}")

                if (rnorm < args.picard_atol) and (rel < args.picard_rtol):
                    break

            # Shift history
            u_nm1.assign(u_n)
            p_nm1.assign(p_n)
            u_n.assign(u_sol)
            p_n.assign(p_sol)

            last_completed_step = n

            # VTK output
            if (n % args.write_freq == 0) or (n == 1):
                vtk.write(u_sol, p_sol, time=t)

            # checkpointing
            do_chk = (args.checkpoint_freq > 0) and (n % args.checkpoint_freq == 0)
            if do_chk or STOP_REQUESTED:
                safe_save_checkpoint(n, t)
                if STOP_REQUESTED:
                    if rank == 0:
                        print("[STOP] stop requested; checkpoint attempt done; exiting.")
                    break

            # printing
            do_print = (n % 5 == 0) or (n == 1)
            do_mem = (args.mem_freq > 0) and (n == 1 or n % args.mem_freq == 0)

            if do_print:
                unorm = float(sqrt(assemble(inner(u_sol, u_sol) * dx)))
            else:
                unorm = None

            rss_max_mb = None
            rss_sum_gb = None
            if do_mem:
                rss_mb = current_rss_mb()
                rss_max_mb = comm.allreduce(rss_mb, op=MPI.MAX)
                rss_sum_mb = comm.allreduce(rss_mb, op=MPI.SUM)
                rss_sum_gb = rss_sum_mb / 1024.0

            if do_print and rank == 0:
                if rss_max_mb is None:
                    print(f"{n:6d}  {mode:>6}  {t:10.4e}  {dt_val:10.3e}  {umax:10.3e}  {Re_now:10.3e}  {unorm:12.6e}  {'':>10}  {'':>10}")
                else:
                    print(f"{n:6d}  {mode:>6}  {t:10.4e}  {dt_val:10.3e}  {umax:10.3e}  {Re_now:10.3e}  {unorm:12.6e}  {rss_max_mb:10.1f}  {rss_sum_gb:10.2f}")

    except KeyboardInterrupt:
        if rank == 0:
            print("\n[INT] KeyboardInterrupt caught: writing checkpoint of last completed state and exiting...")
        if last_completed_step >= 0:
            safe_save_checkpoint(last_completed_step, t)

    wall1 = perf_counter()
    if rank == 0:
        print(f"Stopped at step={last_completed_step}, t={t:.6e}")
        print(f"Total wall time: {wall1 - wall0:.3f} s")
        print(f"Wrote: {args.vtk}")

    report_mem(" end")


if __name__ == "__main__":
    main()

