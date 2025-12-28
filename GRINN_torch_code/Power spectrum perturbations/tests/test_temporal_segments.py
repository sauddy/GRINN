"""
Segmented PINN test harness inspired by
Mao et al. (2025) https://arxiv.org/abs/2509.20447.

The script trains separate PINNs on overlapping temporal intervals:
[0, T1], [T1 - Δt, T2], … and stitches their predictions for
visualization/analysis without touching the main training pipeline.
"""
import argparse
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

# Make sure config + modules are importable both in repo layout and in packed Kaggle script
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATE_ROOTS = [
    _THIS_DIR,
    os.path.dirname(_THIS_DIR),
    os.path.dirname(os.path.dirname(_THIS_DIR)),
]
for _root in _CANDIDATE_ROOTS:
    if not _root or _root in sys.path:
        continue
    sys.path.insert(0, _root)
    if os.path.exists(os.path.join(_root, "config.py")):
        break

from config import (
    DIMENSION,
    N_0,
    N_r,
    PERTURBATION_TYPE,
    RANDOM_SEED,
    STARTUP_DT,
    a,
    cs,
    harmonics,
    iteration_adam_2D,
    iteration_lbgfs_2D,
    N_GRID,
    num_layers,
    num_neurons,
    num_of_waves,
    rho_o,
    tmax as TMAX_CFG,
    wave,
    xmin,
    ymin,
)
from core.data_generator import input_taker, req_consts_calc
from core.initial_conditions import initialize_shared_velocity_fields
from core.losses import ASTPN
from core.model_architecture import PINN
from training.trainer import train
import training.physics as physics  # noqa: E402 (needed for IC overrides)
from visualization.Plotting_2D import (
    create_5x3_comparison_table,
    set_shared_velocity_fields,
)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


@dataclass
class SegmentWindow:
    """Temporal window description."""

    start: float
    end: float


@dataclass
class TrainedSegment:
    """Holds a trained network and its active window."""

    net: PINN
    window: SegmentWindow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train segmented PINNs over overlapping temporal windows "
            "and generate density/velocity comparison tables."
        )
    )
    parser.add_argument(
        "--window-ends",
        type=float,
        nargs="+",
        default=None,
        help=(
            "List of monotonically increasing end-times [T1 T2 ...]. "
            "Windows are built as [t_start, T1], [T1-overlap, T2], … . "
            "Defaults to using config.tmax."
        ),
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.1,
        help="Temporal overlap Δt between consecutive windows (default: 0.1).",
    )
    parser.add_argument(
        "--t-start",
        type=float,
        default=STARTUP_DT,
        help="Initial window start time (default: config.STARTUP_DT).",
    )
    parser.add_argument(
        "--adam-iters",
        type=int,
        default=iteration_adam_2D,
        help="Override Adam iterations per segment (default: config.iteration_adam_2D).",
    )
    parser.add_argument(
        "--lbfgs-iters",
        type=int,
        default=iteration_lbgfs_2D,
        help="Override LBFGS iterations per segment (default: config.iteration_lbgfs_2D).",
    )
    parser.add_argument(
        "--use-lbfgs",
        action="store_true",
        default=False,
        help="Enable LBFGS phase per segment (defaults to False). "
        "When disabled, lbfgs-iters is ignored.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device string (default: auto-detected).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("tests", "segmented_outputs"),
        help="Directory to store diagnostic logs and plots.",
    )
    parser.add_argument(
        "--overlap-samples",
        type=int,
        default=20000,
        help="Number of labeled overlap points transferred from the previous network.",
    )
    parser.add_argument(
        "--overlap-weight",
        type=float,
        default=1.0,
        help="Relative weight for overlap data loss (set 0 to disable).",
    )
    parser.add_argument(
        "--overlap-batch-size",
        type=int,
        default=2048,
        help="Batch size used when sampling overlap data during training.",
    )
    return parser.parse_args()


def build_windows(
    t_start: float, overlap: float, window_ends: Optional[Sequence[float]], tmax: float
) -> List[SegmentWindow]:
    """Create sequential windows following [0,T1],[T1-Δt,T2],… policy."""
    if not window_ends:
        window_ends = [float(tmax)]

    windows: List[SegmentWindow] = []
    prev_end = None
    for end in window_ends:
        if windows and end <= windows[-1].end:
            raise ValueError("window_ends must be strictly increasing.")
        if end > tmax + 1e-9:
            raise ValueError(f"Window end {end} exceeds configured tmax={tmax}.")

        start = t_start if prev_end is None else max(t_start, prev_end - overlap)
        if start >= end:
            raise ValueError(
                f"Invalid window [{start}, {end}]. Ensure overlap < window length."
            )
        windows.append(SegmentWindow(start=start, end=end))
        prev_end = end

    return windows


@contextmanager
def override_initial_conditions(
    rho_fn: Optional[Callable] = None,
    vx_fn: Optional[Callable] = None,
    vy_fn: Optional[Callable] = None,
):
    """Temporarily swap IC helper functions used by training.physics."""

    original_rho = getattr(physics, "fun_rho_0", None)
    original_vx = getattr(physics, "fun_vx_0", None)
    original_vy = getattr(physics, "fun_vy_0", None)

    if rho_fn:
        physics.fun_rho_0 = rho_fn
    if vx_fn:
        physics.fun_vx_0 = vx_fn
    if vy_fn and DIMENSION >= 2:
        physics.fun_vy_0 = vy_fn

    try:
        yield
    finally:
        if original_rho is not None:
            physics.fun_rho_0 = original_rho
        elif hasattr(physics, "fun_rho_0"):
            del physics.fun_rho_0

        if original_vx is not None:
            physics.fun_vx_0 = original_vx
        elif hasattr(physics, "fun_vx_0"):
            del physics.fun_vx_0

        if DIMENSION >= 2:
            if original_vy is not None:
                physics.fun_vy_0 = original_vy
            elif hasattr(physics, "fun_vy_0"):
                del physics.fun_vy_0


def make_transfer_ic_functions(
    prev_net: Optional[PINN],
    transfer_time: float,
    device: torch.device,
) -> Tuple[Optional[Callable], Optional[Callable], Optional[Callable]]:
    """Create IC helper functions that query the previous network."""

    if prev_net is None:
        return None, None, None

    prev_net = prev_net.eval()
    prev_device = next(prev_net.parameters()).device

    def eval_prev(colloc: Sequence[torch.Tensor]) -> torch.Tensor:
        inputs = []
        for idx, tensor in enumerate(colloc):
            base = tensor.detach().clone()
            if idx == len(colloc) - 1:
                base.fill_(transfer_time)
            inputs.append(base.to(prev_device))
        with torch.no_grad():
            predictions = prev_net(inputs)
        return predictions.to(device)

    def rho_fn(_rho_1, _lam, colloc):
        return eval_prev(colloc)[:, 0:1].detach()

    def vx_fn(_lam, _jeans, _v1, colloc):
        return eval_prev(colloc)[:, 1:2].detach()

    def vy_fn(_lam, _jeans, _v1, colloc):
        # For DIMENSION == 1, vy is unused but keep signature consistent.
        outputs = eval_prev(colloc)
        if outputs.size(1) < 3:
            return torch.zeros_like(outputs[:, 1:2])
        return outputs[:, 2:3].detach()

    return rho_fn, vx_fn, vy_fn


def generate_overlap_dataset(
    prev_net: Optional[PINN],
    overlap_start: float,
    overlap_end: float,
    sample_count: int,
    xmin_val: float,
    xmax_val: float,
    ymin_val: float,
    ymax_val: float,
    device: torch.device,
) -> Optional[Dict[str, torch.Tensor]]:
    """Sample labeled data from the previous network across the overlap window."""
    if (
        prev_net is None
        or overlap_end <= overlap_start
        or sample_count <= 0
    ):
        return None

    prev_net = prev_net.eval()
    with torch.no_grad():
        x = torch.rand(sample_count, 1, device=device) * (xmax_val - xmin_val) + xmin_val
        t = torch.rand(sample_count, 1, device=device) * (overlap_end - overlap_start) + overlap_start

        inputs = [x]
        y = None
        if DIMENSION >= 2:
            y = torch.rand(sample_count, 1, device=device) * (ymax_val - ymin_val) + ymin_val
            inputs.append(y)
        if DIMENSION >= 3:
            raise NotImplementedError("3D overlap sampling not yet supported")
        inputs.append(t)

        outputs = prev_net(inputs)

    dataset = {
        'x': x.detach(),
        'y': y.detach() if y is not None else None,
        'z': None,
        't': t.detach(),
        'rho': outputs[:, 0:1].detach(),
        'vx': outputs[:, 1:2].detach(),
        'vy': outputs[:, 2:3].detach() if outputs.shape[1] >= 3 else None,
        'phi': outputs[:, 3:4].detach() if outputs.shape[1] >= 4 else None,
        'count': sample_count,
    }
    return dataset


def train_segment(
    segment_idx: int,
    window: SegmentWindow,
    lam: float,
    rho_1: float,
    jeans: float,
    v_1: float,
    xmax: float,
    ymax: float,
    device: torch.device,
    use_lbfgs: bool,
    adam_iters: int,
    lbfgs_iters: int,
    prev_net: Optional[PINN],
    data_terms: Optional[List[Dict[str, object]]] = None,
) -> PINN:
    """Train a PINN on a single temporal window."""
    print(f"\n=== Training segment {segment_idx}: [{window.start:.3f}, {window.end:.3f}] ===")

    net = PINN(num_neurons=num_neurons, num_layers=num_layers, n_harmonics=harmonics)
    net = net.to(device)
    net.set_domain(rmin=[xmin, ymin], rmax=[xmax, ymax], dimension=DIMENSION)

    model = ASTPN(
        rmin=[xmin, ymin, window.start],
        rmax=[xmax, ymax, window.end],
        N_0=N_0,
        N_b=0,
        N_r=N_r,
        dimension=DIMENSION,
    )

    collocation_domain = model.geo_time_coord(option="Domain")
    collocation_IC = model.geo_time_coord(option="IC")
    collocation_IC[-1].data.fill_(window.start)

    mse_cost = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    optimizerL = (
        torch.optim.LBFGS(net.parameters(), line_search_fn="strong_wolfe")
        if use_lbfgs and lbfgs_iters > 0
        else None
    )

    rho_fn, vx_fn, vy_fn = make_transfer_ic_functions(prev_net, window.start, device)

    with override_initial_conditions(rho_fn, vx_fn, vy_fn):
        train(
            model=model,
            net=net,
            collocation_domain=collocation_domain,
            collocation_IC=collocation_IC,
            optimizer=optimizer,
            optimizerL=optimizerL if optimizerL is not None else optimizer,
            iteration_adam=adam_iters,
            iterationL=(lbfgs_iters if use_lbfgs else 0),
            mse_cost_function=mse_cost,
            closure=None,
            rho_1=rho_1,
            lam=lam,
            jeans=jeans,
            v_1=v_1,
            device=device,
            data_terms=data_terms,
        )

    net.eval()
    return net


class SegmentedEnsemble:
    """Callable wrapper that routes evaluation requests to the right network."""

    def __init__(self, segments: List[TrainedSegment]):
        self.segments = segments
        sample_net = self.segments[0].net
        net_device = next(sample_net.parameters()).device
        dummy_coord = torch.zeros(1, 1, device=net_device)
        with torch.no_grad():
            if DIMENSION >= 2:
                sample_out = sample_net([dummy_coord, dummy_coord, dummy_coord])
            else:
                sample_out = sample_net([dummy_coord, dummy_coord])
        self.output_dim = sample_out.shape[1]

    def __call__(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        if DIMENSION >= 2:
            x, y, t = inputs
        else:
            x, t = inputs
            y = None
        device = x.device
        dtype = x.dtype
        output = torch.zeros(x.shape[0], self.output_dim, device=device, dtype=dtype)
        assigned = torch.zeros(x.shape[0], dtype=torch.bool, device=device)
        tol = 1e-6

        for segment in self.segments:
            start, end = segment.window.start - tol, segment.window.end + tol
            mask = ((t >= start) & (t <= end)).squeeze()
            if not mask.any():
                continue
            net = segment.net
            net_device = next(net.parameters()).device
            x_sel = x[mask].to(net_device)
            t_sel = t[mask].to(net_device)
            if DIMENSION >= 2 and y is not None:
                y_sel = y[mask].to(net_device)
                inputs_sel = [x_sel, y_sel, t_sel]
            else:
                inputs_sel = [x_sel, t_sel]
            preds = net(inputs_sel).to(device)
            output[mask] = preds
            assigned[mask] = True

        if not torch.all(assigned):
            remaining = ~assigned
            earliest_window_start = self.segments[0].window.start + tol
            if torch.any(t[remaining] > earliest_window_start):
                raise RuntimeError(
                    "Some evaluation times fall outside the provided segment windows."
                )
            first_segment = self.segments[0]
            net = first_segment.net
            net_device = next(net.parameters()).device
            x_sel = x[remaining].to(net_device)
            t_sel = t[remaining].to(net_device)
            if DIMENSION >= 2 and y is not None:
                y_sel = y[remaining].to(net_device)
                inputs_sel = [x_sel, y_sel, t_sel]
            else:
                inputs_sel = [x_sel, t_sel]
            preds = net(inputs_sel).to(device)
            output[remaining] = preds
            assigned[remaining] = True

        return output


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    lam, rho_1, _, tmax_cfg, *_ = input_taker(wave, a, num_of_waves, TMAX_CFG, N_0, 0, N_r)
    jeans, alpha = req_consts_calc(lam, rho_1)

    if str(PERTURBATION_TYPE).lower() == "sinusoidal":
        k_mag = 2.0 * np.pi / lam
        base_rho = rho_o if rho_o != 0 else 1.0
        v_1 = (rho_1 / base_rho) * (alpha / k_mag)
    else:
        v_1 = a * cs

    xmax = xmin + lam * num_of_waves
    ymax = ymin + lam * num_of_waves

    windows = build_windows(args.t_start, args.overlap, args.window_ends, tmax_cfg)
    print("Configured temporal windows:")
    for idx, window in enumerate(windows):
        print(f"  {idx}: [{window.start:.3f}, {window.end:.3f}]")

    if str(PERTURBATION_TYPE).lower() == "power_spectrum":
        vx_np, vy_np = initialize_shared_velocity_fields(
            lam, num_of_waves, v_1, seed=RANDOM_SEED
        )
        set_shared_velocity_fields(vx_np, vy_np)

    segments: List[TrainedSegment] = []
    prev_net: Optional[PINN] = None
    for idx, window in enumerate(windows):
        overlap_dataset = None
        data_terms: List[Dict[str, object]] = []
        if prev_net is not None:
            prev_window_end = windows[idx - 1].end
            overlap_end = min(prev_window_end, window.end)
            overlap_dataset = generate_overlap_dataset(
                prev_net,
                overlap_start=window.start,
                overlap_end=overlap_end,
                sample_count=max(0, int(args.overlap_samples)),
                xmin_val=xmin,
                xmax_val=xmax,
                ymin_val=ymin,
                ymax_val=ymax,
                device=device,
            )
            if overlap_dataset is not None and args.overlap_weight > 0:
                batch_cap = overlap_dataset['count']
                overlap_batch = max(1, min(int(args.overlap_batch_size), batch_cap))
                data_terms.append({
                    'dataset': overlap_dataset,
                    'weight': float(args.overlap_weight),
                    'batch_size': overlap_batch,
                    'label': f"OVERLAP_{idx}"
                })

        net = train_segment(
            segment_idx=idx,
            window=window,
            lam=lam,
            rho_1=rho_1,
            jeans=jeans,
            v_1=v_1,
            xmax=xmax,
            ymax=ymax,
            device=device,
            use_lbfgs=args.use_lbfgs,
            adam_iters=args.adam_iters,
            lbfgs_iters=args.lbfgs_iters,
            prev_net=prev_net,
            data_terms=data_terms if data_terms else None,
        )
        segments.append(TrainedSegment(net=net, window=window))
        prev_net = net

    ensemble = SegmentedEnsemble(segments)
    final_tmax = windows[-1].end
    initial_params = (xmin, xmax, ymin, ymax, rho_1, alpha, lam, args.output_dir, final_tmax)

    print("\nGenerating density comparison table...")
    create_5x3_comparison_table(ensemble, initial_params, which="density", N=N_GRID)

    print("\nGenerating velocity comparison table...")
    create_5x3_comparison_table(ensemble, initial_params, which="velocity", N=N_GRID)

    print(f"\nSegmented PINN test completed. Outputs stored in {args.output_dir}")


if __name__ == "__main__":
    main()

