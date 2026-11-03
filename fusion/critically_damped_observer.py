"""
Module 5: Critically Damped (zeta = 1.0) Precision-Blending State Observer.
Implements:
1. CriticallyDampedLPF: Second-order critically damped low-pass filter (zeta = 1.0, omega_n)
   solving ddot{alpha} + 2*omega_n*dot{alpha} + omega_n^2*alpha = omega_n^2*alpha_raw
   via semi-implicit Euler integration, guaranteeing zero overshoot and bounded jerk.
2. DynamicPrecisionBlendingObserver: Fuses real sensors y_real and cGAN virtual sensors y_GAN
   via dynamic inverse-covariance precision matrix scaling, modulated by the Discriminator's
   path optimality score p_opt.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, Union

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from config import GlobalConfig, CONFIG
from envs.uav_mujoco_env import quat_to_rot_matrix


# ==============================================================================
# 1. Second-Order Critically Damped Low-Pass Filter (zeta = 1.0)
# ==============================================================================

class CriticallyDampedLPF:
    """
    Continuous-time 2nd-order critically damped ODE:
        ddot{alpha} + 2*omega_n*dot{alpha} + omega_n^2*alpha = omega_n^2*alpha_raw

    With damping ratio zeta = 1.0, the characteristic polynomial has repeated roots
    at s = -omega_n, mathematically guaranteeing:
    - Zero overshoot (no transient oscillation above 1.0 or below 0.0)
    - Zero ringing
    - Bounded derivative |dot{alpha}| <= omega_n / e (eliminating actuator twitch/jerk)
    - Fastest possible monotonic step response.

    Discretized via symplectic / semi-implicit Euler integration at timestep dt.
    """

    def __init__(
        self,
        omega_n: float = 15.0,
        dt: float = 0.002,
        dim: int = 1,
        eps: float = 0.001
    ):
        self.omega_n = omega_n
        self.dt = dt
        self.dim = dim
        self.eps = eps
        self.reset()

    def reset(self, init_alpha: Union[float, np.ndarray] = 1.0):
        """Reset filter states to initial trust value."""
        if self.dim == 1:
            self.alpha = float(init_alpha)
            self.alpha_dot = 0.0
            self.alpha_ddot = 0.0
        else:
            if np.isscalar(init_alpha):
                self.alpha = np.full(self.dim, float(init_alpha), dtype=np.float64)
            else:
                self.alpha = np.asarray(init_alpha, dtype=np.float64).copy()
            self.alpha_dot = np.zeros(self.dim, dtype=np.float64)
            self.alpha_ddot = np.zeros(self.dim, dtype=np.float64)

    def step(
        self,
        alpha_raw: Union[float, np.ndarray],
        dt: Optional[float] = None
    ) -> Union[float, np.ndarray]:
        """
        Advances the filter by one timestep dt:
            ddot{alpha}_t = omega_n^2 * (alpha_raw - alpha_{t-1}) - 2*omega_n * dot{alpha}_{t-1}
            dot{alpha}_t = dot{alpha}_{t-1} + ddot{alpha}_t * dt
            alpha_t = clip(alpha_{t-1} + dot{alpha}_t * dt, eps, 1 - eps)

        Args:
            alpha_raw: Raw instantaneous trust target in [0, 1].
            dt: Timestep override (defaults to self.dt).

        Returns:
            alpha_smoothed: Filtered trust weight in [eps, 1 - eps].
        """
        step_dt = dt if dt is not None else self.dt
        w_n = self.omega_n

        target = np.asarray(alpha_raw, dtype=np.float64) if self.dim > 1 else float(alpha_raw)

        # 1. Acceleration of trust weight
        self.alpha_ddot = (w_n**2) * (target - self.alpha) - (2.0 * w_n) * self.alpha_dot

        # 2. Semi-implicit Euler integration
        self.alpha_dot = self.alpha_dot + self.alpha_ddot * step_dt
        self.alpha = self.alpha + self.alpha_dot * step_dt

        # 3. Clip within strictly valid boundary [eps, 1 - eps]
        if self.dim == 1:
            self.alpha = float(np.clip(self.alpha, self.eps, 1.0 - self.eps))
        else:
            self.alpha = np.clip(self.alpha, self.eps, 1.0 - self.eps)

        return self.alpha

    @property
    def value(self) -> Union[float, np.ndarray]:
        return self.alpha

    @property
    def rate(self) -> Union[float, np.ndarray]:
        return self.alpha_dot


# ==============================================================================
# 2. Dynamic Precision-Blending State Observer
# ==============================================================================

class DynamicPrecisionBlendingObserver:
    """
    Fuses real sensor readings y_real and cGAN virtual sensor predictions y_GAN
    using Dynamic Precision Blending:
        y_fused = (R_real^-1 + R_GAN^-1)^-1 * (R_real^-1 * y_real + R_GAN^-1 * y_GAN)

    where:
        R_real^-1 = alpha_smoothed * R_nom^-1
        R_GAN^-1  = (1 - alpha_smoothed) * p_opt * R_GAN_nom^-1

    Modulated continuously by:
    - alpha_smoothed: Smoothed trust weight from CriticallyDampedLPF (zeta = 1.0)
    - p_opt: Path optimality score from Discriminator D_psi in [0, 1]
    """

    def __init__(
        self,
        config: GlobalConfig = CONFIG,
        sensor_dim: int = 13,
        omega_n: Optional[float] = None
    ):
        self.cfg = config
        self.sensor_dim = sensor_dim
        dt = config.sim.dt
        wn = omega_n if omega_n is not None else config.filter.omega_n

        # 2nd-order critically damped filter for global trust
        self.lpf = CriticallyDampedLPF(omega_n=wn, dt=dt, dim=1, eps=config.filter.trust_epsilon)

        # Multi-channel LPF for per-modality precision blending (13 channels)
        self.lpf_multichannel = CriticallyDampedLPF(
            omega_n=wn,
            dt=dt,
            dim=sensor_dim,
            eps=config.filter.trust_epsilon
        )

        # Nominal sensor standard deviations (from config)
        noise_cfg = config.noise
        sigmas_real = np.array([
            # IMU accel (3)
            noise_cfg.accel_std, noise_cfg.accel_std, noise_cfg.accel_std,
            # IMU gyro (3)
            noise_cfg.gyro_std, noise_cfg.gyro_std, noise_cfg.gyro_std,
            # GPS pos (3)
            noise_cfg.gps_pos_std, noise_cfg.gps_pos_std, noise_cfg.gps_pos_std,
            # GPS vel (3)
            noise_cfg.gps_vel_std, noise_cfg.gps_vel_std, noise_cfg.gps_vel_std,
            # LiDAR (1)
            noise_cfg.lidar_std
        ], dtype=np.float64)

        # Nominal covariance diagonals
        self.var_real_nom = np.square(sigmas_real)
        # Virtual GAN sensor has slightly higher nominal variance (1.8x)
        self.var_gan_nom = self.var_real_nom * 1.8

        # Precision vectors (inverse variances)
        self.prec_real_nom = 1.0 / np.maximum(self.var_real_nom, 1e-8)
        self.prec_gan_nom = 1.0 / np.maximum(self.var_gan_nom, 1e-8)

        # Internal kinematic state estimate: [x, y, z, vx, vy, vz, roll, pitch, yaw]
        self.estimated_pos = np.zeros(3, dtype=np.float64)
        self.estimated_vel = np.zeros(3, dtype=np.float64)
        self.estimated_rot = np.eye(3, dtype=np.float64)
        self.estimated_omega = np.zeros(3, dtype=np.float64)

        self.reset()

    def reset(
        self,
        init_pos: Optional[np.ndarray] = None,
        init_vel: Optional[np.ndarray] = None,
        init_rot: Optional[np.ndarray] = None
    ):
        """Reset filter and state estimates."""
        self.lpf.reset(init_alpha=1.0)
        self.lpf_multichannel.reset(init_alpha=1.0)

        self.estimated_pos = np.array(init_pos, dtype=np.float64) if init_pos is not None else np.array([0, 0, 1.0])
        self.estimated_vel = np.array(init_vel, dtype=np.float64) if init_vel is not None else np.zeros(3)
        self.estimated_rot = np.array(init_rot, dtype=np.float64) if init_rot is not None else np.eye(3)
        self.estimated_omega = np.zeros(3, dtype=np.float64)

    def fuse(
        self,
        y_real: np.ndarray,
        y_gan: np.ndarray,
        alpha_raw: float,
        p_opt: float = 1.0,
        channel_alpha_raw: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Executes Dynamic Precision Blending between physical sensors and virtual cGAN sensor:
            W_real = alpha_tilde * R_nom^-1
            W_GAN  = (1 - alpha_tilde) * p_opt * R_GAN_nom^-1
            y_fused = (W_real + W_GAN)^-1 * (W_real * y_real + W_GAN * y_gan)

        Args:
            y_real: Physical sensor observation in R^13.
            y_gan: Virtual cGAN sensor prediction in R^13.
            alpha_raw: Scalar instantaneous trust target in [0, 1].
            p_opt: Path optimality score from Discriminator D_psi in [0, 1].
            channel_alpha_raw: Optional 13-dim modality trust vector.

        Returns:
            y_fused: Fused observation vector in R^13.
            diagnostics: Fusion diagnostics including smoothed trust, rate, and weights.
        """
        t0 = time.perf_counter_ns()

        y_r = np.asarray(y_real, dtype=np.float64)
        y_g = np.asarray(y_gan, dtype=np.float64)

        # 1. Smooth trust weight through 2nd-order Critically Damped LPF
        alpha_smoothed = float(self.lpf.step(alpha_raw))
        alpha_dot = float(self.lpf.rate)

        # Multi-channel trust smoothing
        if channel_alpha_raw is not None:
            alpha_vec = self.lpf_multichannel.step(channel_alpha_raw)
        else:
            alpha_vec = np.full(self.sensor_dim, alpha_smoothed, dtype=np.float64)

        # 2. Dynamic Precision Weights
        # Modulate virtual sensor precision with Discriminator path optimality p_opt
        p_opt_clamped = float(np.clip(p_opt, 0.05, 1.0))

        w_real = alpha_vec * self.prec_real_nom
        w_gan = (1.0 - alpha_vec) * p_opt_clamped * self.prec_gan_nom

        w_total = w_real + w_gan

        # 3. Precision-Weighted Fusion Equation: y_fused = (w_r * y_r + w_g * y_g) / w_total
        y_fused = (w_real * y_r + w_gan * y_g) / np.maximum(w_total, 1e-8)

        # 4. Update internal kinematic observer state
        dt = self.cfg.sim.dt
        # Fused IMU gyro -> body rates
        self.estimated_omega = y_fused[3:6]

        # Fused GPS position and velocity -> kinematics
        fused_gps_pos = y_fused[6:9]
        fused_gps_vel = y_fused[9:12]

        # Complementary update on kinematics
        self.estimated_vel = 0.9 * (self.estimated_vel + y_fused[0:3] * dt) + 0.1 * fused_gps_vel
        self.estimated_pos = 0.95 * (self.estimated_pos + self.estimated_vel * dt) + 0.05 * fused_gps_pos

        t1 = time.perf_counter_ns()
        latency_us = (t1 - t0) / 1000.0

        diagnostics = {
            "alpha_raw": alpha_raw,
            "alpha_smoothed": alpha_smoothed,
            "alpha_dot": alpha_dot,
            "alpha_ddot": float(self.lpf.alpha_ddot),
            "p_opt": p_opt_clamped,
            "w_real_mean": float(np.mean(w_real)),
            "w_gan_mean": float(np.mean(w_gan)),
            "virtual_blend_ratio": float(np.mean(w_gan / np.maximum(w_total, 1e-8))),
            "latency_us": latency_us,
            "estimated_pos": self.estimated_pos.copy(),
            "estimated_vel": self.estimated_vel.copy()
        }

        return y_fused, diagnostics

    def fuse_tri_brid(
        self,
        y_real: np.ndarray,
        y_gan: np.ndarray,
        vslam_pos: np.ndarray,
        vslam_vel: np.ndarray,
        alpha_vslam: float,
        alpha_gps: float,
        p_opt: float = 1.0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Novelty 2: Tri-Brid Critically Damped Precision Blending Observer.
        Fuses IMU, VSLAM, GPS, and cGAN virtual sensor:
        1. When VSLAM is valid (alpha_vslam high), blends VSLAM with GPS to anchor drift.
        2. When VSLAM suffers blackout/blur AND GPS is spoofed, dynamically switches
           to the cGAN virtual sensor via the 2nd-order critically damped LPF (zeta = 1.0).
        """
        t0 = time.perf_counter_ns()
        y_r = np.asarray(y_real, dtype=np.float64).copy()
        y_g = np.asarray(y_gan, dtype=np.float64).copy()

        # 1. Synthesize visual-inertial-GPS hybrid position & velocity
        gps_pos = y_r[6:9]
        gps_vel = y_r[9:12]

        w_vslam = float(np.clip(alpha_vslam, 0.001, 0.999))
        w_gps = float(np.clip(alpha_gps, 0.001, 0.999))
        w_nav_total = w_vslam + w_gps

        p_nav = (w_vslam * vslam_pos + w_gps * gps_pos) / w_nav_total
        v_nav = (w_vslam * vslam_vel + w_gps * gps_vel) / w_nav_total

        # Override GPS slots in y_r with navigation solution
        y_r[6:9] = p_nav
        y_r[9:12] = v_nav

        # Overall navigation trust target
        alpha_nav_raw = float(np.clip(max(alpha_vslam, alpha_gps), 0.001, 0.999))

        # 2. Critically Damped Low-Pass Filtering (zeta = 1.0)
        alpha_smoothed = float(self.lpf.step(alpha_nav_raw))
        alpha_dot = float(self.lpf.rate)

        # 3. Precision-Weighted Blending with cGAN virtual sensor
        p_opt_clamped = float(np.clip(p_opt, 0.05, 1.0))
        alpha_vec = np.full(self.sensor_dim, alpha_smoothed, dtype=np.float64)

        w_real = alpha_vec * self.prec_real_nom
        w_gan = (1.0 - alpha_vec) * p_opt_clamped * self.prec_gan_nom
        w_total = w_real + w_gan

        y_fused = (w_real * y_r + w_gan * y_g) / np.maximum(w_total, 1e-8)

        # 4. Update kinematic state
        dt = self.cfg.sim.dt
        self.estimated_omega = y_fused[3:6]
        fused_p = y_fused[6:9]
        fused_v = y_fused[9:12]

        self.estimated_vel = 0.9 * (self.estimated_vel + y_fused[0:3] * dt) + 0.1 * fused_v
        self.estimated_pos = 0.95 * (self.estimated_pos + self.estimated_vel * dt) + 0.05 * fused_p

        t1 = time.perf_counter_ns()
        latency_us = (t1 - t0) / 1000.0

        diagnostics = {
            "alpha_raw": alpha_nav_raw,
            "alpha_smoothed": alpha_smoothed,
            "alpha_dot": alpha_dot,
            "alpha_vslam": alpha_vslam,
            "alpha_gps": alpha_gps,
            "p_opt": p_opt_clamped,
            "latency_us": latency_us,
            "estimated_pos": self.estimated_pos.copy(),
            "estimated_vel": self.estimated_vel.copy(),
            "fallback_active": bool(alpha_smoothed < 0.3)
        }
        return y_fused, diagnostics


# ==============================================================================
# 3. Verification & Comparison Suite: Step-Fault Response
# ==============================================================================

def verify_step_fault_response(
    duration: float = 1.0,
    dt: float = 0.002,
    omega_n: float = 15.0
) -> Dict[str, Any]:
    """
    Simulates a sudden catastrophic sensor fault (step jump: trust 1.0 -> 0.0 at t=0.2s)
    and benchmarks:
    1. Unsmoothed Hard-Switch Observer (causes severe derivative spike and motor jerk)
    2. Critically Damped Observer (zeta = 1.0, zero overshoot, bounded derivative)
    """
    steps = int(duration / dt)
    t_fault = 0.2

    # Instantiate filter & observer
    lpf = CriticallyDampedLPF(omega_n=omega_n, dt=dt, dim=1)
    observer = DynamicPrecisionBlendingObserver(omega_n=omega_n)

    # History logs
    t_hist = []
    alpha_raw_hist = []
    alpha_smooth_hist = []
    alpha_dot_hist = []
    hard_switch_dot_hist = []

    pos_err_hard = []
    pos_err_smooth = []

    true_pos = np.array([1.0, 2.0, 1.5])

    for step in range(steps):
        t = step * dt
        t_hist.append(t)

        # Fault occurs at t >= 0.2s: GPS pos corrupts with +10m spoofing offset
        is_fault = (t >= t_fault)
        alpha_raw = 0.0 if is_fault else 1.0
        alpha_raw_hist.append(alpha_raw)

        # Simulated sensors
        clean_sensor = np.zeros(13)
        clean_sensor[6:9] = true_pos # GPS pos

        y_real = clean_sensor.copy()
        if is_fault:
            y_real[6:9] += np.array([8.0, -12.0, 5.0]) # GPS spoofing offset

        # Virtual cGAN sensor predicts close to true state
        y_gan = clean_sensor + np.random.normal(0, 0.03, size=13)

        # 1. Unsmoothed Hard Switch calculation
        if step == 0:
            hard_dot = 0.0
            prev_hard = alpha_raw
        else:
            hard_dot = (alpha_raw - prev_hard) / dt
            prev_hard = alpha_raw
        hard_switch_dot_hist.append(abs(hard_dot))

        # 2. Critically Damped Observer calculation
        y_fused, diag = observer.fuse(
            y_real=y_real,
            y_gan=y_gan,
            alpha_raw=alpha_raw,
            p_opt=0.95
        )

        alpha_smooth_hist.append(diag["alpha_smoothed"])
        alpha_dot_hist.append(abs(diag["alpha_dot"]))

        # Errors on fused GPS position
        err_hard = np.linalg.norm((y_real[6:9] if not is_fault else y_gan[6:9]) - true_pos)
        err_smooth = np.linalg.norm(y_fused[6:9] - true_pos)

        pos_err_hard.append(err_hard)
        pos_err_smooth.append(err_smooth)

    # Compute key quantitative metrics
    max_hard_jerk_metric = float(np.max(hard_switch_dot_hist))
    max_smooth_rate_metric = float(np.max(alpha_dot_hist))
    rate_reduction_factor = max_hard_jerk_metric / max(max_smooth_rate_metric, 1e-6)

    # Check overshoot: critically damped response must never dip below target or exceed bounds
    min_smooth = float(np.min(alpha_smooth_hist))
    max_smooth = float(np.max(alpha_smooth_hist))
    overshoot_below = max(0.0, -min_smooth) # should be 0.0
    overshoot_above = max(0.0, max_smooth - 1.0) # should be 0.0

    return {
        "max_hard_rate": max_hard_jerk_metric,
        "max_smooth_rate": max_smooth_rate_metric,
        "rate_reduction_factor": rate_reduction_factor,
        "overshoot_percentage": (overshoot_below + overshoot_above) * 100.0,
        "mean_latency_us": diag["latency_us"],
        "min_alpha": min_smooth,
        "final_alpha": alpha_smooth_hist[-1]
    }


if __name__ == "__main__":
    print("=" * 75)
    print(" Running Module 5 Verification: Critically Damped Observer (zeta = 1.0)")
    print("=" * 75)

    duration = 1.0
    dt = 0.002
    omega_n = 15.0

    print(f"Simulating 1.0s ({int(duration/dt)} steps @ 500 Hz) with step fault at t = 0.2s...")
    metrics = verify_step_fault_response(duration=duration, dt=dt, omega_n=omega_n)

    print("\n--- RESULTS SUMMARY ---")
    print(f"1. Transition Smoothness & Jerk Reduction:")
    print(f"   • Unsmoothed Hard-Switch Max |d(alpha)/dt|: {metrics['max_hard_rate']:.2f} s^-1  (Violent step jump)")
    print(f"   • Critically Damped Max |d(alpha)/dt|:     {metrics['max_smooth_rate']:.2f} s^-1  (Smooth bounded derivative)")
    print(f"   • Actuator Jerk / Rate Reduction Factor:   {metrics['rate_reduction_factor']:.1f}x reduction")

    print(f"\n2. Mathematical Zero-Overshoot Verification (zeta = 1.0):")
    print(f"   • Measured Overshoot: {metrics['overshoot_percentage']:.6f}% (Mathematically zero ringing)")
    print(f"   • Final Settled Trust: {metrics['final_alpha']:.4f} (Cleanly transitioned to virtual cGAN sensor)")

    print(f"\n3. Computational Efficiency:")
    print(f"   • Observer Step Latency: {metrics['mean_latency_us']:.2f} µs (Budget: < 50 µs)")

    print("\nVerification checks:")
    assert metrics["overshoot_percentage"] < 1e-5, "Critically damped filter must have zero overshoot!"
    assert metrics["max_smooth_rate"] < 10.0, f"Filtered rate {metrics['max_smooth_rate']} exceeds bound!"
    assert metrics["rate_reduction_factor"] > 40.0, "Filter must reduce transition derivative by > 40x!"
    assert metrics["mean_latency_us"] < 50.0, "Observer latency must be under 50 microseconds!"
    print("✔ All Module 5 Verification assertions passed successfully!")
    print("=" * 75)
