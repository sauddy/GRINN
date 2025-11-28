import argparse
import os
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from config import (DIMENSION,FD_N_2D,FD_N_3D,PERTURBATION_TYPE,SLICE_Z,a,cs,harmonics,
    num_of_waves,tmax,tmin,wave,xmin,ymin,zmin,N_0,N_r,rho_o,)
from core.data_generator import input_taker, req_consts_calc
from core.initial_conditions import fun_rho_0,fun_vx_0,fun_vy_0,fun_vz_0,initialize_shared_velocity_fields
from core.losses import ASTPN, pde_residue
from core.model_architecture import PINN
from numerical_solvers.LAX_torch import lax_solution_warm_start_torch,lax_solution_torch,lax_solution_3d_sinusoidal_torch


# ========================= Collocation / PINN loss =========================


def prepare_collocation_points(
    lam: float,
    num_waves: float,
    tmax_cfg: float,
    device: torch.device,
) -> Tuple[ASTPN, List[torch.Tensor], List[torch.Tensor]]:
    xmax = xmin + lam * num_waves
    ymax = ymin + lam * num_waves if DIMENSION >= 2 else ymin

    # For 3D, ASTPN expects rmin/rmax including z; we just re-use zmin and zmax=zmin+lam*num_waves
    if DIMENSION == 3:
        zmax = zmin + lam * num_waves
        rmin = [xmin, ymin, zmin, tmin]
        rmax = [xmax, ymax, zmax, tmax_cfg]
    elif DIMENSION == 2:
        rmin = [xmin, ymin, tmin]
        rmax = [xmax, ymax, tmax_cfg]
    else:
        rmin = [xmin, tmin]
        rmax = [xmax, tmax_cfg]

    model = ASTPN(
        rmin=rmin,
        rmax=rmax,
        N_0=N_0,
        N_b=0,
        N_r=N_r,
        dimension=DIMENSION,
    )

    collocation_domain = model.geo_time_coord(option="Domain")
    collocation_ic = model.geo_time_coord(option="IC")

    collocation_domain = [tensor.to(device) for tensor in collocation_domain]
    collocation_ic = [tensor.to(device) for tensor in collocation_ic]
    return model, collocation_domain, collocation_ic


def compute_pinn_loss(
    model: ASTPN,
    net: PINN,
    collocation_domain: Sequence[torch.Tensor],
    collocation_ic: Sequence[torch.Tensor],
    lam: float,
    rho_1: float,
    jeans: float,
    v_1: float,
) -> float:
    mse = nn.MSELoss()

    rho_0 = fun_rho_0(rho_1, lam, collocation_ic)
    vx_0 = fun_vx_0(lam, jeans, v_1, collocation_ic)
    vy_0 = fun_vy_0(lam, jeans, v_1, collocation_ic) if model.dimension >= 2 else None
    vz_0 = fun_vz_0(lam, jeans, v_1, collocation_ic) if model.dimension == 3 else None

    net_ic_out = net(collocation_ic)
    rho_ic_out = net_ic_out[:, 0:1]
    vx_ic_out = net_ic_out[:, 1:2]

    if model.dimension == 2:
        vy_ic_out = net_ic_out[:, 2:3]
    elif model.dimension == 3:
        vy_ic_out = net_ic_out[:, 2:3]
        vz_ic_out = net_ic_out[:, 3:4]

    mse_rho_ic = mse(rho_ic_out, rho_0)
    mse_vx_ic = mse(vx_ic_out, vx_0)

    if model.dimension >= 2:
        mse_vy_ic = mse(vy_ic_out, vy_0)
    if model.dimension == 3 and vz_0 is not None:
        mse_vz_ic = mse(vz_ic_out, vz_0)

    if isinstance(collocation_domain, (list, tuple)):
        colloc_shifted = list(collocation_domain)
    else:
        colloc_shifted = [
            collocation_domain[:, i : i + 1] for i in range(collocation_domain.shape[1])
        ]

    if model.dimension == 1:
        rho_r, vx_r, phi_r = pde_residue(colloc_shifted, net, dimension=1)
    elif model.dimension == 2:
        rho_r, vx_r, vy_r, phi_r = pde_residue(colloc_shifted, net, dimension=2)
    else:
        rho_r, vx_r, vy_r, vz_r, phi_r = pde_residue(colloc_shifted, net, dimension=3)

    mse_rho = torch.mean(rho_r**2)
    mse_vx = torch.mean(vx_r**2)
    if model.dimension >= 2:
        mse_vy = torch.mean(vy_r**2)
    if model.dimension == 3:
        mse_vz = torch.mean(vz_r**2)
    mse_phi = torch.mean(phi_r**2)

    if model.dimension == 1:
        loss = mse_rho_ic + mse_vx_ic + mse_rho + mse_vx + mse_phi
    elif model.dimension == 2:
        loss = (
            mse_rho_ic
            + mse_vx_ic
            + mse_vy_ic
            + mse_rho
            + mse_vx
            + mse_vy
            + mse_phi
        )
    else:
        loss = (
            mse_rho_ic
            + mse_vx_ic
            + mse_vy_ic
            + mse_vz_ic
            + mse_rho
            + mse_vx
            + mse_vy
            + mse_vz
            + mse_phi
        )

    return float(loss.item())


# ========================= PINN vs FD comparison =========================


def _pinn_eval_on_grid_2d(
    net: PINN, X: np.ndarray, Y: np.ndarray, t_end: float, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = torch.tensor(
        np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32, device=device
    )
    times = torch.full((coords.shape[0], 1), float(t_end), dtype=torch.float32, device=device)
    net_inputs = [coords[:, 0:1], coords[:, 1:2], times]
    net.eval()
    with torch.no_grad():
        preds = net(net_inputs).cpu().numpy()
    rho_pred = preds[:, 0].reshape(X.shape)
    vx_pred = preds[:, 1].reshape(X.shape)
    vy_pred = preds[:, 2].reshape(X.shape)
    return rho_pred, vx_pred, vy_pred


def _pinn_eval_on_grid_3d_slice(
    net: PINN,
    X: np.ndarray,
    Y: np.ndarray,
    z_val: float,
    t_end: float,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coords = torch.tensor(
        np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32, device=device
    )
    z = torch.full((coords.shape[0], 1), float(z_val), dtype=torch.float32, device=device)
    times = torch.full((coords.shape[0], 1), float(t_end), dtype=torch.float32, device=device)
    net_inputs = [coords[:, 0:1], coords[:, 1:2], z, times]
    net.eval()
    with torch.no_grad():
        preds = net(net_inputs).cpu().numpy()
    rho_pred = preds[:, 0].reshape(X.shape)
    vx_pred = preds[:, 1].reshape(X.shape)
    vy_pred = preds[:, 2].reshape(X.shape)
    vz_pred = preds[:, 3].reshape(X.shape)
    return rho_pred, vx_pred, vy_pred, vz_pred


def pinn_vs_fd_mse(
    net: PINN,
    lam: float,
    rho_1: float,
    v_1: float,
    t_end: float,
    grid_n: int,
    device: torch.device,
    return_fields: bool = False,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """
    General comparison wrapper:
    - 2D power_spectrum: warm-start FD using shared velocity fields
    - 2D sinusoidal: direct 2D LAX solver
    - 3D sinusoidal: 3D LAX solver, then compare on a z-slice
    """
    pert = str(PERTURBATION_TYPE).lower()

    if DIMENSION == 2 and pert == "power_spectrum":
        # --- 2D power spectrum with warm-start ---
        Lx = lam * num_of_waves
        Ly = lam * num_of_waves

        x_lin = np.linspace(xmin, xmin + Lx, grid_n, endpoint=False)
        y_lin = np.linspace(ymin, ymin + Ly, grid_n, endpoint=False)
        X, Y = np.meshgrid(x_lin, y_lin, indexing="ij")

        rho_ic = rho_o * np.ones((grid_n, grid_n), dtype=np.float64)

        # Use same shared velocity fields as training
        vx_shared, vy_shared = initialize_shared_velocity_fields(
            lam, num_of_waves, v_1, seed=None
        )
        if vx_shared.shape != rho_ic.shape:
            from scipy.ndimage import zoom

            scale_x = grid_n / vx_shared.shape[0]
            scale_y = grid_n / vx_shared.shape[1]
            vx_ic = zoom(vx_shared, (scale_x, scale_y), order=1)
            vy_ic = zoom(vy_shared, (scale_x, scale_y), order=1)
        else:
            vx_ic = vx_shared
            vy_ic = vy_shared

        snapshots = lax_solution_warm_start_torch(
            rho_ic=rho_ic,
            vx_ic=vx_ic,
            vy_ic=vy_ic,
            x_grid=x_lin,
            y_grid=y_lin,
            t_start=0.0,
            t_end=t_end,
            nu=0.25,
            gravity=True,
            save_times=[t_end],
        )
        rho_fd, vx_fd, vy_fd, _, _, _ = snapshots[t_end]

        rho_pred, vx_pred, vy_pred = _pinn_eval_on_grid_2d(net, X, Y, t_end, device)

        metrics: Dict[str, float] = {
            "rho_mse": float(np.mean((rho_pred - rho_fd) ** 2)),
            "vx_mse": float(np.mean((vx_pred - vx_fd) ** 2)),
            "vy_mse": float(np.mean((vy_pred - vy_fd) ** 2)),
        }

        fields: Dict[str, np.ndarray] = {}
        if return_fields:
            fields = {
                "x": X,
                "y": Y,
                "rho_fd": rho_fd,
                "rho_pred": rho_pred,
                "vx_fd": vx_fd,
                "vx_pred": vx_pred,
                "vy_fd": vy_fd,
                "vy_pred": vy_pred,
            }
        return metrics, fields

    if DIMENSION == 2 and pert == "sinusoidal":
        # --- 2D sinusoidal: direct FD solver from IC definition ---
        Lx = lam * num_of_waves
        Ly = lam * num_of_waves
        x_lin = np.linspace(xmin, xmin + Lx, grid_n, endpoint=False)
        y_lin = np.linspace(ymin, ymin + Ly, grid_n, endpoint=False)
        X, Y = np.meshgrid(x_lin, y_lin, indexing="ij")

        # Use 2D LAX solver with sinusoidal setup
        _, rho_fd, vx_fd, vy_fd, _, _, _ = lax_solution_torch(
            time_val=t_end,
            N=grid_n,
            nu=0.25,
            lam=lam,
            num_of_waves=num_of_waves,
            rho_1=rho_1,
            gravity=True,
        )

        rho_pred, vx_pred, vy_pred = _pinn_eval_on_grid_2d(net, X, Y, t_end, device)

        metrics = {
            "rho_mse": float(np.mean((rho_pred - rho_fd) ** 2)),
            "vx_mse": float(np.mean((vx_pred - vx_fd) ** 2)),
            "vy_mse": float(np.mean((vy_pred - vy_fd) ** 2)),
        }
        fields: Dict[str, np.ndarray] = {}
        if return_fields:
            fields = {
                "x": X,
                "y": Y,
                "rho_fd": rho_fd,
                "rho_pred": rho_pred,
                "vx_fd": vx_fd,
                "vx_pred": vx_pred,
                "vy_fd": vy_fd,
                "vy_pred": vy_pred,
            }
        return metrics, fields

    if DIMENSION == 3 and pert == "sinusoidal":
        # --- 3D sinusoidal: use full 3D solver, compare on z-slice ---
        N = grid_n
        x_arr, y_arr, z_arr, rho_fd_3d, vx_fd_3d, vy_fd_3d, vz_fd_3d, _, _, _ = (
            lax_solution_3d_sinusoidal_torch(
                time_val=t_end,
                N=N,
                nu=0.25,
                lam=lam,
                num_of_waves=num_of_waves,
                rho_1=rho_1,
                gravity=True,
            )
        )
        X, Y, Z = np.meshgrid(x_arr, y_arr, z_arr, indexing="ij")

        # Choose z-slice nearest to SLICE_Z * Lz
        Lz = lam * num_of_waves
        target_z = SLICE_Z * Lz
        z_index = int(np.clip(np.round(target_z / (Lz / N)), 0, N - 1))

        rho_fd = rho_fd_3d[:, :, z_index]
        vx_fd = vx_fd_3d[:, :, z_index]
        vy_fd = vy_fd_3d[:, :, z_index]
        vz_fd = vz_fd_3d[:, :, z_index]
        X_slice = X[:, :, z_index]
        Y_slice = Y[:, :, z_index]
        z_val = z_arr[z_index]

        rho_pred, vx_pred, vy_pred, vz_pred = _pinn_eval_on_grid_3d_slice(
            net, X_slice, Y_slice, float(z_val), t_end, device
        )

        metrics = {
            "rho_mse": float(np.mean((rho_pred - rho_fd) ** 2)),
            "vx_mse": float(np.mean((vx_pred - vx_fd) ** 2)),
            "vy_mse": float(np.mean((vy_pred - vy_fd) ** 2)),
            "vz_mse": float(np.mean((vz_pred - vz_fd) ** 2)),
        }
        fields: Dict[str, np.ndarray] = {}
        if return_fields:
            fields = {
                "x": X_slice,
                "y": Y_slice,
                "rho_fd": rho_fd,
                "rho_pred": rho_pred,
                "vx_fd": vx_fd,
                "vx_pred": vx_pred,
                "vy_fd": vy_fd,
                "vy_pred": vy_pred,
                "vz_fd": vz_fd,
                "vz_pred": vz_pred,
            }
        return metrics, fields

    raise NotImplementedError(
        f"FD comparison not implemented for DIMENSION={DIMENSION}, "
        f"PERTURBATION_TYPE={PERTURBATION_TYPE}"
    )


# ========================= Plotting =========================


def plot_pinn_loss(
    loss_a: float,
    loss_b: float,
    t_end: float,
    label_a: str,
    label_b: str,
    outdir: str,
) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    labels = [label_a, label_b]
    values = [loss_a, loss_b]
    colors = ["#1f77b4", "#ff7f0e"]

    ax.bar(labels, values, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("Total PINN loss (log scale)")
    ax.set_title(f"PINN Loss on [0, {t_end}]")
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.2e}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    out_path = os.path.join(outdir, "pinn_loss.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_fd_metrics(
    metrics_a: Dict[str, float],
    metrics_b: Dict[str, float],
    t_end: float,
    label_a: str,
    label_b: str,
    outdir: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    # Keep canonical ordering where possible
    all_keys = ["rho_mse", "vx_mse", "vy_mse", "vz_mse"]
    keys = [k for k in all_keys if k in metrics_a and k in metrics_b]
    labels = [k.replace("_mse", "").upper() for k in keys]
    x = np.arange(len(keys))
    width = 0.35

    values_a = [metrics_a[k] for k in keys]
    values_b = [metrics_b[k] for k in keys]

    ax.bar(x - width / 2, values_a, width, label=label_a, color="#1f77b4")
    ax.bar(x + width / 2, values_b, width, label=label_b, color="#ff7f0e")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MSE")
    ax.set_title(f"FD MSE at t = {t_end}")
    ax.legend()
    ax.set_yscale("log")

    for idx, val in enumerate(values_a):
        ax.text(x[idx] - width / 2, val, f"{val:.2e}", ha="center", va="bottom", fontsize=8)
    for idx, val in enumerate(values_b):
        ax.text(x[idx] + width / 2, val, f"{val:.2e}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(outdir, "fd_mse.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_field_panels(
    field_key: str,
    fields_a: Dict[str, np.ndarray],
    fields_b: Dict[str, np.ndarray],
    t_end: float,
    label_a: str,
    label_b: str,
    outdir: str,
    symmetric: bool = False,
) -> None:
    label_map = {
        "rho": "Density ρ",
        "vx": "Velocity vx",
        "vy": "Velocity vy",
        "vz": "Velocity vz",
    }
    cmap_field = "viridis" if not symmetric else "coolwarm"
    cmap_error = "magma"

    arrays = [
        fields_a[f"{field_key}_fd"],
        fields_a[f"{field_key}_pred"],
        fields_b[f"{field_key}_fd"],
        fields_b[f"{field_key}_pred"],
    ]

    if symmetric:
        limit = max(np.max(np.abs(arr)) for arr in arrays)
        if limit == 0:
            limit = 1.0
        vmin, vmax = -limit, limit
    else:
        vmin = min(np.min(arr) for arr in arrays)
        vmax = max(np.max(arr) for arr in arrays)
        if vmin == vmax:
            vmin -= 1.0
            vmax += 1.0

    err_limit = max(
        np.max(np.abs(fields[f"{field_key}_pred"] - fields[f"{field_key}_fd"]))
        for fields in (fields_a, fields_b)
    )
    if err_limit == 0:
        err_limit = 1.0

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.0))
    row_entries = [(label_a, fields_a), (label_b, fields_b)]

    for row, (row_label, field_dict) in enumerate(row_entries):
        fd = field_dict[f"{field_key}_fd"]
        pred = field_dict[f"{field_key}_pred"]
        error = np.abs(pred - fd)

        im0 = axes[row, 0].imshow(fd, origin="lower", cmap=cmap_field, vmin=vmin, vmax=vmax)
        im1 = axes[row, 1].imshow(pred, origin="lower", cmap=cmap_field, vmin=vmin, vmax=vmax)
        im2 = axes[row, 2].imshow(
            error, origin="lower", cmap=cmap_error, vmin=0.0, vmax=err_limit
        )

        axes[row, 0].set_title("FD reference" if row == 0 else "")
        axes[row, 1].set_title("PINN prediction" if row == 0 else "")
        axes[row, 2].set_title("|Prediction − FD|" if row == 0 else "")
        axes[row, 0].set_ylabel(row_label)

        for col, im in enumerate((im0, im1, im2)):
            cbar = fig.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
            if col < 2:
                cbar.ax.set_ylabel(label_map.get(field_key, field_key), rotation=270, labelpad=12)
            else:
                cbar.ax.set_ylabel("Absolute error", rotation=270, labelpad=12)

        for col in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    fig.suptitle(f"{label_map.get(field_key, field_key)} comparison at t = {t_end}", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = os.path.join(outdir, f"{field_key}_comparison.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ========================= CLI / Main =========================


def load_model(path: str, lam: float, num_waves: float, device: torch.device) -> PINN:
    net = PINN(n_harmonics=harmonics)
    xmax = xmin + lam * num_waves
    if DIMENSION >= 2:
        ymax = ymin + lam * num_waves
    else:
        ymax = ymin
    if DIMENSION == 3:
        zmax = zmin + lam * num_waves
        rmin = [xmin, ymin, zmin]
        rmax = [xmax, ymax, zmax]
    elif DIMENSION == 2:
        rmin = [xmin, ymin]
        rmax = [xmax, ymax]
    else:
        rmin = [xmin]
        rmax = [xmax]
    net.set_domain(rmin=rmin, rmax=rmax, dimension=DIMENSION)
    state_dict = torch.load(path, map_location=device)
    net.load_state_dict(state_dict)
    net = net.to(device)
    return net


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two PINN models against FD ground truth. "
            "Supports 2D power spectrum, 2D sinusoidal, and 3D sinusoidal "
            "(governed by config.py PERTURBATION_TYPE and DIMENSION)."
        )
    )
    parser.add_argument(
        "--model-a",
        type=str,
        required=True,
        help="Path to the first model checkpoint.",
    )
    parser.add_argument(
        "--model-b",
        type=str,
        required=True,
        help="Path to the second model checkpoint.",
    )
    parser.add_argument(
        "--label-a",
        type=str,
        default=None,
        help="Label for the first model (used in plots).",
    )
    parser.add_argument(
        "--label-b",
        type=str,
        default=None,
        help="Label for the second model (used in plots).",
    )
    parser.add_argument(
        "--time-a",
        type=float,
        default=3.0,
        help="Training end time for the first model (for labeling only).",
    )
    parser.add_argument(
        "--time-b",
        type=float,
        default=4.0,
        help="Training end time for the second model (for labeling only).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="model_fd_comparison",
        help="Directory to save comparison plots.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Force device (default: auto-detect).",
    )
    parser.add_argument(
        "--fd-grid",
        type=int,
        default=None,
        help="Grid size for FD evaluation (per dimension, default from config).",
    )
    parser.add_argument(
        "--fd-time",
        type=float,
        default=4.0,
        help="Physical time at which to compare with FD solution.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda"
        if (args.device == "cuda" or (args.device is None and torch.cuda.is_available()))
        else "cpu"
    )
    print(f"Using device: {device}")

    # Derive problem parameters from config
    lam, rho_1, num_waves_eff, tmax_cfg, _, _, _ = input_taker(
        wave, a, num_of_waves, tmax, N_0, 0, N_r
    )
    jeans, alpha_phys = req_consts_calc(lam, rho_1)
    is_power = str(PERTURBATION_TYPE).lower() == "power_spectrum"
    v_1 = a * cs if is_power else alpha_phys

    # FD grid size
    if args.fd_grid is not None:
        grid_n = args.fd_grid
    else:
        if DIMENSION == 3:
            grid_n = FD_N_3D
        else:
            grid_n = FD_N_2D

    label_a = args.label_a or f"t={args.time_a} model"
    label_b = args.label_b or f"t={args.time_b} model"

    net_a = load_model(args.model_a, lam, num_waves_eff, device)
    net_b = load_model(args.model_b, lam, num_waves_eff, device)

    model, collocation_domain, collocation_ic = prepare_collocation_points(
        lam, num_waves_eff, args.fd_time, device
    )

    print("\n[Step 1] Computing PINN loss contributions on collocation sets...")
    loss_a = compute_pinn_loss(
        model, net_a, collocation_domain, collocation_ic, lam, rho_1, jeans, v_1
    )
    loss_b = compute_pinn_loss(
        model, net_b, collocation_domain, collocation_ic, lam, rho_1, jeans, v_1
    )

    print(f"  PINN loss ({label_a}) on [0,{args.fd_time}]: {loss_a:.4e}")
    print(f"  PINN loss ({label_b}) on [0,{args.fd_time}]: {loss_b:.4e}")

    print("\n[Step 2] Comparing against FD ground truth...")
    fd_metrics_a, fields_a = pinn_vs_fd_mse(
        net_a,
        lam,
        rho_1,
        v_1,
        t_end=args.fd_time,
        grid_n=grid_n,
        device=device,
        return_fields=True,
    )
    fd_metrics_b, fields_b = pinn_vs_fd_mse(
        net_b,
        lam,
        rho_1,
        v_1,
        t_end=args.fd_time,
        grid_n=grid_n,
        device=device,
        return_fields=True,
    )

    print(f"  FD comparison ({label_a}):", fd_metrics_a)
    print(f"  FD comparison ({label_b}):", fd_metrics_b)

    os.makedirs(args.outdir, exist_ok=True)

    plot_pinn_loss(loss_a, loss_b, args.fd_time, label_a, label_b, args.outdir)
    plot_fd_metrics(fd_metrics_a, fd_metrics_b, args.fd_time, label_a, label_b, args.outdir)

    # Always plot rho, vx, vy; plot vz only if present (3D)
    plot_field_panels("rho", fields_a, fields_b, args.fd_time, label_a, label_b, args.outdir)
    plot_field_panels(
        "vx", fields_a, fields_b, args.fd_time, label_a, label_b, args.outdir, symmetric=True
    )
    plot_field_panels(
        "vy", fields_a, fields_b, args.fd_time, label_a, label_b, args.outdir, symmetric=True
    )
    if "vz_fd" in fields_a and "vz_fd" in fields_b:
        plot_field_panels(
            "vz",
            fields_a,
            fields_b,
            args.fd_time,
            label_a,
            label_b,
            args.outdir,
            symmetric=True,
        )

    print(f"\nSaved comparison plots to {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
