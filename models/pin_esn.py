"""
Paradigm B: Physics-Informed Echo State Network (PIN-ESN) with Closed-Form RLS Readout.
Implements:
1. PhysicsInformedESN:
   - High-dimensional recurrent reservoir (N_res = 128) satisfying the Echo State Property
     (spectral radius rho(W_res) = 0.95 < 1.0, sparse random connectivity).
   - Inputs: Quadrotor kinematic state/motor wrench (18-dim) and delayed physics residual
     r_phys_{t-1} (6-dim) -> 24-dim combined input vector.
   - Closed-form Recursive Least Squares (RLS) adaptive readout head (Theta in R^(129 x 6))
     with exponential forgetting factor lambda_rls = 0.995.
   - In-place tensor rank-1 updates (torch.addr) for microsecond execution latency (< 200 us).
   - Strictly convex optimization: Zero local minima, zero stochastic replay buffer variance,
     guaranteed convergence, and deterministic execution.
"""

import time
from typing import Tuple, Optional, Union, Dict, Any
import numpy as np
import torch


class PhysicsInformedESN:
    """
    Visual-Aided Physics-Informed Echo State Network (V-PIN-ESN) with Closed-Form RLS Readout.
    Estimates and compensates unmodeled external wrench F_ext in R^6 using dual physical
    kinematic features, delayed Euler-Lagrange residuals, and visual innovation residuals.
    """

    def __init__(
        self,
        state_dim: int = 18,
        phys_dim: int = 6,
        vis_dim: int = 3,
        output_dim: int = 6,
        reservoir_dim: int = 128,
        spectral_radius: float = 0.95,
        sparsity: float = 0.15,
        leaking_rate: float = 0.35,
        input_scale: float = 0.15,
        lambda_rls: float = 0.995,
        delta_p0: float = 50.0,
        device: Optional[torch.device] = None,
        seed: int = 42
    ):
        self.state_dim = state_dim
        self.phys_dim = phys_dim
        self.vis_dim = vis_dim
        self.input_dim = state_dim + phys_dim + vis_dim  # 18 + 6 + 3 = 27
        self.output_dim = output_dim          # 6 (fx, fy, fz, tau_x, tau_y, tau_z)
        self.reservoir_dim = reservoir_dim    # 128
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leaking_rate = leaking_rate
        self.lambda_rls = lambda_rls
        self.inv_lambda_rls = 1.0 / lambda_rls
        self.delta_p0 = delta_p0
        self.device = device or torch.device("cpu")
        self.seed = seed

        # Set deterministic seed for reproducibility
        rng = np.random.RandomState(seed)

        # 1. Random Input Projection Matrix W_in: (reservoir_dim, input_dim)
        w_in_np = rng.uniform(-input_scale, input_scale, (self.reservoir_dim, self.input_dim)).astype(np.float32)
        self.W_in = torch.from_numpy(w_in_np).to(self.device)

        # 2. Sparse Recurrent Reservoir Matrix W_res: (reservoir_dim, reservoir_dim)
        raw_res = rng.randn(self.reservoir_dim, self.reservoir_dim).astype(np.float32)
        mask = rng.rand(self.reservoir_dim, self.reservoir_dim) < self.sparsity
        raw_res = raw_res * mask

        # Scale to strictly enforce the Echo State Property: rho(W_res) = spectral_radius < 1
        eigenvalues = np.linalg.eigvals(raw_res)
        max_eig = float(np.max(np.abs(eigenvalues)))
        if max_eig > 1e-7:
            scaled_res = raw_res * (self.spectral_radius / max_eig)
        else:
            scaled_res = raw_res

        self.W_res = torch.from_numpy(scaled_res).to(self.device)
        self.bias_res = torch.from_numpy(rng.uniform(-0.02, 0.02, self.reservoir_dim).astype(np.float32)).to(self.device)

        # 3. Reservoir State h in R^reservoir_dim
        self.h = torch.zeros(self.reservoir_dim, dtype=torch.float32, device=self.device)

        # 4. Readout Head Theta in R^((reservoir_dim + 1) x output_dim)
        self.param_dim = self.reservoir_dim + 1  # 129
        self.Theta = torch.zeros((self.param_dim, self.output_dim), dtype=torch.float32, device=self.device)

        # 5. Inverse Correlation Covariance Matrix P in R^(param_dim x param_dim)
        self.P = torch.eye(self.param_dim, dtype=torch.float32, device=self.device) * self.delta_p0

        # Preallocated buffers for in-place rank-1 updates (avoids GC allocations)
        self.u_t = torch.zeros(self.input_dim, dtype=torch.float32, device=self.device)
        self.pre_act = torch.zeros(self.reservoir_dim, dtype=torch.float32, device=self.device)
        self.phi = torch.ones(self.param_dim, dtype=torch.float32, device=self.device)
        self.P_phi = torch.zeros(self.param_dim, dtype=torch.float32, device=self.device)
        self.k = torch.zeros(self.param_dim, dtype=torch.float32, device=self.device)
        self.f_hat = torch.zeros(self.output_dim, dtype=torch.float32, device=self.device)
        self.r_prev = torch.zeros(self.phys_dim, dtype=torch.float32, device=self.device)

    def reset(self):
        """Resets reservoir state, parameter matrix, and covariance matrix."""
        self.h.zero_()
        self.Theta.zero_()
        self.P.copy_(torch.eye(self.param_dim, dtype=torch.float32, device=self.device) * self.delta_p0)
        self.phi.fill_(1.0)
        self.r_prev.zero_()
        self.u_t.zero_()

    @torch.no_grad()
    def predict(
        self,
        x: Union[np.ndarray, torch.Tensor],
        delta_p_vis: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward prediction of external wrench without updating RLS weights.

        Args:
            x: Quadrotor state feature vector (18-dim).
            delta_p_vis: Optional visual innovation residual (3-dim).

        Returns:
            f_hat: Predicted external wrench in R^6 as NumPy array.
            phi: Augmented feature vector [h_t; 1] in R^129 as NumPy array.
        """
        if isinstance(x, np.ndarray):
            x_t = torch.from_numpy(x).float().to(self.device).ravel()
        else:
            x_t = x.to(self.device).ravel()

        self.u_t[:self.state_dim] = x_t
        self.u_t[self.state_dim : self.state_dim + self.phys_dim] = self.r_prev

        if delta_p_vis is not None:
            if isinstance(delta_p_vis, np.ndarray):
                dp_t = torch.from_numpy(delta_p_vis).float().to(self.device).ravel()[:self.vis_dim]
            else:
                dp_t = delta_p_vis.to(self.device).ravel()[:self.vis_dim]
            self.u_t[self.state_dim + self.phys_dim :] = dp_t
        else:
            self.u_t[self.state_dim + self.phys_dim :].zero_()

        # Reservoir update
        torch.mv(self.W_in, self.u_t, out=self.pre_act)
        self.pre_act.addmv_(self.W_res, self.h)
        self.pre_act.add_(self.bias_res)
        self.pre_act.tanh_()
        self.h.mul_(1.0 - self.leaking_rate).add_(self.pre_act, alpha=self.leaking_rate)

        self.phi[:self.reservoir_dim] = self.h
        self.phi[self.reservoir_dim] = 1.0

        torch.mv(self.Theta.t(), self.phi, out=self.f_hat)
        return self.f_hat.cpu().numpy(), self.phi.cpu().numpy()

    @torch.no_grad()
    def update(
        self,
        x: Union[np.ndarray, torch.Tensor],
        r_phys: Union[np.ndarray, torch.Tensor],
        delta_p_vis: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Executes one complete closed-form recursive update cycle:
        1. Forward step through reservoir using [x_t, r_prev, delta_p_vis].
        2. Computes predicted wrench f_hat.
        3. Closed-form RLS adaptation of Theta driven by physics residual r_phys.
        4. Saves r_phys as r_prev for next step.
        5. Returns updated wrench estimate and execution latency in microseconds.
        """
        t0 = time.perf_counter_ns()

        if isinstance(x, np.ndarray):
            x_t = torch.from_numpy(x).float().to(self.device).ravel()
        else:
            x_t = x.to(self.device).ravel()

        if isinstance(r_phys, np.ndarray):
            r_t = torch.from_numpy(r_phys).float().to(self.device).ravel()
        else:
            r_t = r_phys.to(self.device).ravel()

        # 1. Forward pass through reservoir with delayed residual and visual innovation
        self.u_t[:self.state_dim] = x_t
        self.u_t[self.state_dim : self.state_dim + self.phys_dim] = self.r_prev

        if delta_p_vis is not None:
            if isinstance(delta_p_vis, np.ndarray):
                dp_t = torch.from_numpy(delta_p_vis).float().to(self.device).ravel()[:self.vis_dim]
            else:
                dp_t = delta_p_vis.to(self.device).ravel()[:self.vis_dim]
            self.u_t[self.state_dim + self.phys_dim :] = dp_t
        else:
            self.u_t[self.state_dim + self.phys_dim :].zero_()

        torch.mv(self.W_in, self.u_t, out=self.pre_act)
        self.pre_act.addmv_(self.W_res, self.h)
        self.pre_act.add_(self.bias_res)
        self.pre_act.tanh_()
        self.h.mul_(1.0 - self.leaking_rate).add_(self.pre_act, alpha=self.leaking_rate)

        self.phi[:self.reservoir_dim] = self.h
        self.phi[self.reservoir_dim] = 1.0

        # 2. Compute RLS Kalman gain: k = (P * phi) / (lambda + phi^T * P * phi)
        torch.mv(self.P, self.phi, out=self.P_phi)
        denom = self.lambda_rls + float(torch.dot(self.phi, self.P_phi))
        self.k.copy_(self.P_phi / denom)

        # 3. Covariance update: P = (P - k * (P_phi)^T) / lambda
        torch.addr(self.P, self.k, self.P_phi, alpha=-1.0, beta=1.0, out=self.P)
        self.P.mul_(self.inv_lambda_rls)

        # 4. Parameter update: Theta = Theta + k * (r_phys)^T
        torch.addr(self.Theta, self.k, r_t, alpha=1.0, beta=1.0, out=self.Theta)

        # 5. Compute adapted wrench estimate
        torch.mv(self.Theta.t(), self.phi, out=self.f_hat)

        # 6. Save current residual as delayed input for next time step
        self.r_prev.copy_(r_t)

        t1 = time.perf_counter_ns()
        step_latency_us = (t1 - t0) / 1000.0

        return self.f_hat.cpu().numpy(), step_latency_us

    def get_stats(self) -> Dict[str, Any]:
        """Returns diagnostic statistics of the reservoir and readout head."""
        eigvals = np.linalg.eigvals(self.W_res.cpu().numpy())
        return {
            "reservoir_dim": self.reservoir_dim,
            "param_dim": self.param_dim,
            "trainable_params": self.Theta.numel(),  # 129 * 6 = 774
            "spectral_radius": float(np.max(np.abs(eigvals))),
            "reservoir_energy": float(torch.norm(self.h).item()),
            "theta_norm": float(torch.norm(self.Theta).item()),
            "p_trace": float(torch.trace(self.P).item())
        }


if __name__ == "__main__":
    print("=" * 75)
    print(" Verifying Paradigm B: Physics-Informed Echo State Network (PIN-ESN)")
    print("=" * 75)

    esn = PhysicsInformedESN(reservoir_dim=128)
    stats = esn.get_stats()
    print(f"Reservoir Dimension:  {stats['reservoir_dim']}")
    print(f"Trainable Readout:    {stats['trainable_params']} parameters (Theta in R^{esn.param_dim}x{esn.output_dim})")
    print(f"Echo State Property:  Spectral Radius rho = {stats['spectral_radius']:.4f} (< 1.0: {stats['spectral_radius'] < 1.0})")

    # Benchmark closed-loop disturbance adaptation
    print("\nSimulating 1,000 closed-loop adaptation cycles under 500 Hz flight load...")
    latencies = []
    dummy_x = np.random.randn(18).astype(np.float32)
    ground_truth_wind = np.array([5.0, -3.5, 2.0, 0.05, -0.02, 0.01], dtype=np.float32)

    # In closed-loop flight: r_phys = F_ext_true - F_ext_hat
    f_hat_curr = np.zeros(6, dtype=np.float32)

    for step in range(1000):
        # Physics residual is the difference between actual wind and what we've compensated so far
        r_phys = (ground_truth_wind - f_hat_curr) + 0.01 * np.random.randn(6).astype(np.float32)
        f_hat_curr, lat = esn.update(dummy_x, r_phys)
        latencies.append(lat)

    warm_lat = latencies[20:]
    mean_lat = float(np.mean(warm_lat))
    p95_lat = float(np.percentile(warm_lat, 95))
    max_lat = float(np.max(warm_lat))

    print(f"\nExecution Latency Benchmark:")
    print(f"  • Mean Latency: {mean_lat:.2f} µs")
    print(f"  • p95 Latency:  {p95_lat:.2f} µs")
    print(f"  • Max Latency:  {max_lat:.2f} µs")
    print(f"  • 500 Hz Budget Utilization: {(mean_lat / 2000.0) * 100:.2f}% (Deadline: 2,000 µs)")

    print(f"\nFinal Disturbance Compensation:")
    print(f"  • Ground Truth Wrench:     {ground_truth_wind[:3]}")
    print(f"  • PIN-ESN Wrench Estimate: {f_hat_curr[:3]}")
    error = float(np.linalg.norm(ground_truth_wind - f_hat_curr))
    print(f"  • Residual Tracking Error: {error:.4f} N (Target converged: {error < 0.1})")

    assert stats["spectral_radius"] < 1.0, "Spectral radius must be < 1.0!"
    assert mean_lat < 500.0, f"Mean latency {mean_lat} µs exceeded real-time budget!"
    assert error < 0.1, f"PIN-ESN failed to converge! Residual error: {error}"
    print("\n✔ Paradigm B (PIN-ESN) verification passed successfully!")
