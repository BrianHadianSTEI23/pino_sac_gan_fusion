"""
SMAC 3D State-Lattice Simulator and SE(3) Geometric Tracking Controller.
Generates kinodynamically feasible 3D reference trajectories (emulating ROS 2 Nav2 SMAC Lattice)
and controls the quadrotor along the planned path using SE(3) geometric control.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from config import GlobalConfig, CONFIG
from envs.uav_mujoco_env import UAVMujocoEnv


@dataclass
class ReferenceState:
    """Desired 3D kinematic waypoint reference state."""
    pos: np.ndarray        # Desired position (3,)
    vel: np.ndarray        # Desired linear velocity (3,)
    acc: np.ndarray        # Desired linear acceleration (3,)
    yaw: float             # Desired yaw heading (rad)
    yaw_rate: float        # Desired yaw rate (rad/s)


class SMACLatticePath:
    """
    Emulates ROS 2 Nav2 SMAC 3D State Lattice global trajectory generator.
    Produces kinodynamically feasible 3D trajectories with smooth derivatives
    and local horizon windows for cGAN conditioning.
    """

    def __init__(
        self,
        radius_x: float = 2.5,
        radius_y: float = 2.0,
        z_base: float = 1.5,
        z_amp: float = 0.5,
        period: float = 12.0
    ):
        self.rx = radius_x
        self.ry = radius_y
        self.z_base = z_base
        self.z_amp = z_amp
        self.period = period
        self.w0 = 2.0 * np.pi / period

    def get_reference_at(self, t: float) -> ReferenceState:
        """
        Computes the exact analytic 3D reference state at time t.
        Uses a smooth ascending figure-8 (Lemniscate) curve to exercise
        multiaxial coupling in SE(3).
        """
        w = self.w0
        # Position
        x = self.rx * np.sin(w * t)
        y = self.ry * np.sin(2.0 * w * t)
        z = self.z_base + self.z_amp * np.sin(w * t)
        pos = np.array([x, y, z], dtype=np.float64)

        # Velocity (first derivative)
        vx = self.rx * w * np.cos(w * t)
        vy = self.ry * (2.0 * w) * np.cos(2.0 * w * t)
        vz = self.z_amp * w * np.cos(w * t)
        vel = np.array([vx, vy, vz], dtype=np.float64)

        # Acceleration (second derivative)
        ax = -self.rx * (w**2) * np.sin(w * t)
        ay = -self.ry * ((2.0 * w)**2) * np.sin(2.0 * w * t)
        az = -self.z_amp * (w**2) * np.sin(w * t)
        acc = np.array([ax, ay, az], dtype=np.float64)

        # Yaw aligned tangentially to path direction
        yaw = float(np.arctan2(vy, vx))
        # Numerical yaw rate approximation
        eps = 1e-4
        vx_next = self.rx * w * np.cos(w * (t + eps))
        vy_next = self.ry * (2.0 * w) * np.cos(2.0 * w * (t + eps))
        yaw_next = float(np.arctan2(vy_next, vx_next))
        yaw_rate = float((np.unwrap([yaw, yaw_next])[1] - yaw) / eps)

        return ReferenceState(
            pos=pos,
            vel=vel,
            acc=acc,
            yaw=yaw,
            yaw_rate=yaw_rate
        )

    def get_local_window(
        self,
        t: float,
        horizon_steps: int = 10,
        dt: float = 0.05
    ) -> np.ndarray:
        """
        Extracts future waypoint window P_lattice[t : t+k] of shape (horizon_steps, 4)
        [x, y, z, yaw] for conditioning the virtual sensor cGAN.
        """
        window = np.zeros((horizon_steps, 4), dtype=np.float64)
        for i in range(horizon_steps):
            t_future = t + (i + 1) * dt
            ref = self.get_reference_at(t_future)
            window[i, 0:3] = ref.pos
            window[i, 3] = ref.yaw
        return window


def vee_map(S: np.ndarray) -> np.ndarray:
    """Vee operator mapping skew-symmetric so(3) matrix to R^3."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=np.float64)


class SE3GeometricController:
    """
    Geometric Tracking Controller on SE(3) manifold (Lee et al., 2010).
    Computes required collective thrust and 3-axis rotational moments,
    then allocates them to 4 motor thrust commands via the mixer matrix.
    """

    def __init__(self, config: GlobalConfig = CONFIG):
        self.cfg = config
        self.m = config.uav.mass
        self.g = config.sim.gravity
        self.J = np.diag([config.uav.ixx, config.uav.iyy, config.uav.izz])

        # Control gains tuned for Skydio quadrotor
        self.Kp = np.diag([22.0, 22.0, 32.0])
        self.Kv = np.diag([8.0, 8.0, 10.0])
        self.KR = np.diag([12.0, 12.0, 4.0])
        self.Kw = np.diag([1.6, 1.6, 0.6])

        # Motor mixer allocation matrix (X-configuration)
        d = config.uav.arm_length / np.sqrt(2.0)  # ~0.15556 m
        c_tau = config.uav.k_torque               # 0.015 N*m / N
        self.allocation_matrix = np.array([
            [1.0,     1.0,     1.0,     1.0],     # Collective thrust Fz
            [  d,       d,      -d,      -d],     # Roll torque tau_x
            [ -d,       d,       d,      -d],     # Pitch torque tau_y
            [-c_tau,  c_tau,  -c_tau,  c_tau]     # Yaw torque tau_z
        ], dtype=np.float64)
        self.inv_allocation = np.linalg.inv(self.allocation_matrix)

    def compute_control(
        self,
        current_state: Dict[str, Any],
        ref: ReferenceState
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Computes 4-rotor thrust commands [T1, T2, T3, T4] in Newtons.

        Args:
            current_state: State dictionary from UAVMujocoEnv.
            ref: Desired ReferenceState from SMACLatticePath.

        Returns:
            ctrl: 4-dim motor thrust commands in Newtons.
            diagnostics: Dictionary containing tracking errors and wrenches.
        """
        pos = current_state["pos"]
        vel = current_state["lin_vel"]
        R = current_state["rot_matrix"]
        omega = current_state["ang_vel"]

        # 1. Translational tracking errors
        e_p = pos - ref.pos
        e_v = vel - ref.vel

        # 2. Desired total force in world frame
        f_gravity = np.array([0.0, 0.0, self.m * self.g], dtype=np.float64)
        f_feedforward = self.m * ref.acc
        F_des = -self.Kp @ e_p - self.Kv @ e_v + f_gravity + f_feedforward

        # Body z-axis (third column of current rotation matrix)
        z_b = R[:, 2]
        collective_thrust = float(np.dot(F_des, z_b))
        collective_thrust = np.clip(collective_thrust, 0.0, 40.0)

        # 3. Construct desired rotation matrix R_des
        z_b_des = F_des / (np.linalg.norm(F_des) + 1e-8)
        x_c_des = np.array([np.cos(ref.yaw), np.sin(ref.yaw), 0.0], dtype=np.float64)
        y_b_des = np.cross(z_b_des, x_c_des)
        norm_yb = np.linalg.norm(y_b_des)
        if norm_yb < 1e-6:
            y_b_des = np.array([0.0, 1.0, 0.0])
        else:
            y_b_des /= norm_yb
        x_b_des = np.cross(y_b_des, z_b_des)
        R_des = np.column_stack([x_b_des, y_b_des, z_b_des])

        # 4. Attitude error in SO(3)
        e_R_matrix = 0.5 * (R_des.T @ R - R.T @ R_des)
        e_R = vee_map(e_R_matrix)

        # 5. Angular rate error
        omega_des = np.array([0.0, 0.0, ref.yaw_rate], dtype=np.float64)
        e_omega = omega - (R.T @ R_des @ omega_des)

        # 6. Desired body torque
        tau_des = -self.KR @ e_R - self.Kw @ e_omega + np.cross(omega, self.J @ omega)

        # 7. Mixer allocation to 4 motors
        wrench = np.array([collective_thrust, tau_des[0], tau_des[1], tau_des[2]])
        ctrl = self.inv_allocation @ wrench
        ctrl = np.clip(ctrl, 0.0, 10.0)

        diagnostics = {
            "pos_error_norm": float(np.linalg.norm(e_p)),
            "vel_error_norm": float(np.linalg.norm(e_v)),
            "att_error_norm": float(np.linalg.norm(e_R)),
            "collective_thrust": collective_thrust,
            "tau_des": tau_des
        }

        return ctrl, diagnostics


def run_benchmark(duration_sec: float = 5.0) -> Dict[str, Any]:
    """
    Runs a benchmark comparison between clean nominal flight and
    flight under mission disturbance & sensor faults.
    Returns tracking and residual statistics.
    """
    env = UAVMujocoEnv()
    path = SMACLatticePath()
    controller = SE3GeometricController()

    dt = env.dt
    total_steps = int(duration_sec / dt)

    results = {}

    for mode in ["clean", "faulty"]:
        enable_faults = (mode == "faulty")
        # Reset to initial waypoint
        ref_0 = path.get_reference_at(0.0)
        state = env.reset(init_pos=ref_0.pos, init_vel=ref_0.vel)

        r_phys_norms = []
        r_sens_norms = []
        pos_errors = []

        for step in range(total_steps):
            t = env.sim_time
            ref = path.get_reference_at(t)
            ctrl, diag = controller.compute_control(state, ref)

            obs, r_phys, r_sens, info = env.step(ctrl, enable_faults=enable_faults)
            state = info["true_state"]

            r_phys_norms.append(float(np.linalg.norm(r_phys)))
            r_sens_norms.append(float(np.linalg.norm(r_sens)))
            pos_errors.append(diag["pos_error_norm"])

        results[mode] = {
            "r_phys_mean": float(np.mean(r_phys_norms)),
            "r_phys_max": float(np.max(r_phys_norms)),
            "r_sens_mean": float(np.mean(r_sens_norms)),
            "r_sens_max": float(np.max(r_sens_norms)),
            "pos_rmse": float(np.sqrt(np.mean(np.square(pos_errors)))),
            "final_pos_error": float(pos_errors[-1])
        }

    return results


if __name__ == "__main__":
    print("=" * 70)
    print(" Running Module 1 Verification: MuJoCo UAV Simulation & Fault Injector")
    print("=" * 70)

    duration = 5.0
    print(f"Simulating 5.0 seconds ({int(duration / 0.002)} steps @ 500 Hz)...")
    res = run_benchmark(duration_sec=duration)

    print("\n--- RESULTS SUMMARY ---")
    print(f"1. Clean Flight (Nominal Noise Only):")
    print(f"   • Mean ||r_phys||: {res['clean']['r_phys_mean']:.6f} N  (Exact physics verified: ≈ 0.0)")
    print(f"   • Max  ||r_phys||: {res['clean']['r_phys_max']:.6f} N")
    print(f"   • Mean ||r_sens||: {res['clean']['r_sens_mean']:.6f}   (Nominal sensor noise floor)")
    print(f"   • Max  ||r_sens||: {res['clean']['r_sens_max']:.6f}")
    print(f"   • Position RMSE:   {res['clean']['pos_rmse']:.4f} m")

    print(f"\n2. Faulty Flight (Wind Gust @ t=4s + Thermal IMU Drift @ t>=2s):")
    print(f"   • Mean ||r_phys||: {res['faulty']['r_phys_mean']:.6f} N  (Disturbance detected!)")
    print(f"   • Max  ||r_phys||: {res['faulty']['r_phys_max']:.6f} N  (Wind force peak)")
    print(f"   • Mean ||r_sens||: {res['faulty']['r_sens_mean']:.6f}   (IMU thermal ramp divergence)")
    print(f"   • Max  ||r_sens||: {res['faulty']['r_sens_max']:.6f}")
    print(f"   • Position RMSE:   {res['faulty']['pos_rmse']:.4f} m")

    print("\nVerification checks:")
    assert res['clean']['r_phys_mean'] < 1e-4, "Clean physics residual should evaluate to ~0.0!"
    assert res['faulty']['r_phys_max'] > 4.0, "Wind disturbance should induce detectable r_phys spike!"
    assert res['faulty']['r_sens_max'] > res['clean']['r_sens_max'], "Fault injector should diverge r_sens!"
    print("✔ All verification assertions passed successfully!")
    print("=" * 70)
