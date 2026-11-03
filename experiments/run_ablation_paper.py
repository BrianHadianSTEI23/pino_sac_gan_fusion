"""
Module 6: End-to-End Ablation Runner & Publication Plots (run_ablation_paper.py).
Runs the full 20-second closed-loop flight mission under severe compound disturbances
(wind gust t=4-9s, thermal IMU drift t>=2s, GPS spoofing/LiDAR blackout t=10-15s).
Evaluates 6 controllers head-to-head:
(a) Classical EKF
(b) Variant 0 [RLS Baseline]
(c) Variant 1 [SAC Freeze 0]
(d) Variant 2 [SAC Freeze 1]
(e) Variant 3 [SAC Freeze 2]
(f) Variant 4 [SAC Freeze 3]
Generates 4 IEEE publication figures (PNG and PDF) and outputs a formatted LaTeX table.
"""

import sys
import os
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg") # Headless rendering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from config import GlobalConfig, CONFIG
from envs.uav_mujoco_env import UAVMujocoEnv
from envs.smac_lattice_sim import SMACLatticePath, SE3GeometricController
from models.pillar_b_sn_mlp import SpectrallyNormalizedMLP, NoiseAndFaultAdapter, extract_mlp_features
from models.continual_adapters import (
    create_continual_adapter,
    RLSHeadAdapter,
    OnlineAnnealedSACAdapter,
    PIN_ESN_Adapter
)
from models.physics_cgan import (
    PhysicsConditionedGenerator,
    PathOptimalityDiscriminator,
    pack_cgan_condition
)
from fusion.critically_damped_observer import DynamicPrecisionBlendingObserver


# ==============================================================================
# 1. Classical EKF Baseline (No Trust Gating, No cGAN Fallback)
# ==============================================================================

class ClassicalEKF:
    """
    Standard Extended Kalman Filter baseline for quadrotor state estimation:
    - Assumes fixed sensor noise covariance R_nom.
    - No physics consistency checking (r_phys ignored).
    - No virtual sensor fallback (vulnerable to spoofing and drift).
    - External disturbance wrench is assumed zero (no live adaptation).
    """

    def __init__(self, config: GlobalConfig = CONFIG):
        self.dt = config.sim.dt
        self.pos = np.array([0.0, 0.0, 1.0])
        self.vel = np.zeros(3)
        self.rot = np.eye(3)
        self.omega = np.zeros(3)

        # State error covariance P (12x12: pos, vel, rot_err, omega)
        self.P = np.eye(12) * 0.1

    def step(self, obs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        t0 = time.perf_counter_ns()
        dt = self.dt

        # IMU prediction
        acc = obs["imu_accel"] - np.array([0.0, 0.0, -9.81])
        self.vel = self.vel + acc * dt
        self.pos = self.pos + self.vel * dt
        self.omega = obs["imu_gyro"]

        # GPS measurement update (unfiltered, direct trust)
        gps_pos = obs["gps_pos"]
        gps_vel = obs["gps_vel"]

        # Simple Kalman innovation gain K = 0.08
        self.pos = 0.92 * self.pos + 0.08 * gps_pos
        self.vel = 0.90 * self.vel + 0.10 * gps_vel

        latency_us = (time.perf_counter_ns() - t0) / 1000.0

        return {
            "estimated_pos": self.pos.copy(),
            "estimated_vel": self.vel.copy(),
            "estimated_rot": self.rot.copy(),
            "estimated_omega": self.omega.copy(),
            "f_ext_hat": np.zeros(6),
            "latency_us": latency_us,
            "alpha_raw": 1.0,
            "alpha_smoothed": 1.0,
            "p_opt": 1.0
        }


# ==============================================================================
# 2. Closed-Loop Mission Simulation Engine
# ==============================================================================

def run_single_mission(
    controller_type: str,
    duration: float = 20.0,
    checkpoints_dir: str = "./checkpoints",
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Executes a 20-second flight mission under full compound faults:
    - Wind gust: t in [4, 9]s
    - Thermal IMU drift: t >= 2s
    - GPS spoofing & LiDAR blackout: t in [10, 15]s
    """
    dev = device or torch.device("cpu")
    env = UAVMujocoEnv()
    path = SMACLatticePath()
    tracking_controller = SE3GeometricController()

    dt = env.dt
    total_steps = int(duration / dt)

    # Initialize cGAN Virtual Sensor & Critic
    cgan_G = PhysicsConditionedGenerator(95, 16, 13).to(dev)
    cgan_D = PathOptimalityDiscriminator(95, 13).to(dev)

    # Load pre-trained weights if available
    g_path = os.path.join(checkpoints_dir, "cgan_generator.pt")
    d_path = os.path.join(checkpoints_dir, "cgan_discriminator.pt")
    mlp_path = os.path.join(checkpoints_dir, "sn_mlp_base.pt")

    if os.path.exists(g_path):
        cgan_G.load_state_dict(torch.load(g_path, map_location=dev))
    if os.path.exists(d_path):
        cgan_D.load_state_dict(torch.load(d_path, map_location=dev))

    cgan_G.eval()
    cgan_D.eval()

    # Observer & Fault Adapter
    observer = DynamicPrecisionBlendingObserver()
    fault_adapter = NoiseAndFaultAdapter()

    # Base SN-MLP
    base_mlp = SpectrallyNormalizedMLP(18, 256, 6).to(dev)
    if os.path.exists(mlp_path):
        base_mlp.load_state_dict(torch.load(mlp_path, map_location=dev))

    # Continual Adapter or EKF
    is_ekf = (controller_type == "Classical EKF")
    continual_adapter = None

    if is_ekf:
        estimator = ClassicalEKF()
    else:
        variant_map = {
            "Variant 0 [RLS]": 0,
            "Variant 1 [SAC Freeze 0]": 1,
            "Variant 2 [SAC Freeze 1]": 2,
            "Variant 3 [SAC Freeze 2]": 3,
            "Variant 4 [SAC Freeze 3]": 4,
            "Paradigm B [PIN-ESN]": 5
        }
        v_idx = variant_map[controller_type]
        continual_adapter = create_continual_adapter(v_idx, base_mlp=base_mlp, device=dev)

    # Reset environment to initial waypoint
    ref_0 = path.get_reference_at(0.0)
    state = env.reset(init_pos=ref_0.pos, init_vel=ref_0.vel)
    observer.reset(init_pos=ref_0.pos, init_vel=ref_0.vel)

    # Logging structures
    history = {
        "time": [],
        "true_pos": [],
        "ref_pos": [],
        "est_pos": [],
        "r_phys": [],
        "r_sens": [],
        "r_vis": [],
        "alpha_raw": [],
        "alpha_smoothed": [],
        "alpha_vslam": [],
        "p_opt": [],
        "latencies_us": [],
        "pos_errors": [],
        "control_jerk": []
    }

    past_sensor_window = []
    prev_pos = state["pos"].copy()
    prev_vel = state["lin_vel"].copy()
    prev_acc = np.zeros(3)

    for step in range(total_steps):
        t = env.sim_time
        ref = path.get_reference_at(t)

        # 1. State feedback to tracking controller
        if is_ekf:
            control_feedback_state = {
                "pos": estimator.pos,
                "lin_vel": estimator.vel,
                "rot_matrix": state["rot_matrix"], # attitude
                "ang_vel": estimator.omega
            }
        else:
            control_feedback_state = {
                "pos": observer.estimated_pos,
                "lin_vel": observer.estimated_vel,
                "rot_matrix": state["rot_matrix"],
                "ang_vel": observer.estimated_omega
            }

        ctrl, diag = tracking_controller.compute_control(control_feedback_state, ref)

        # 2. Advance physics with mission compound faults
        obs, r_phys, r_sens, info = env.step(ctrl, enable_faults=True)
        true_st = info["true_state"]

        current_sensor = np.concatenate([
            obs["imu_accel"],
            obs["imu_gyro"],
            obs["gps_pos"],
            obs["gps_vel"],
            np.array([obs["lidar_down"]])
        ])

        past_sensor_window.append(current_sensor)
        if len(past_sensor_window) > 5:
            past_sensor_window.pop(0)

        # Compute jerk: d^3(x)/dt^3
        curr_acc = true_st["lin_acc"]
        jerk = np.linalg.norm(curr_acc - prev_acc) / dt
        prev_acc = curr_acc.copy()

        # 3. State Estimation & Continual Adaptation
        t_est_start = time.perf_counter_ns()

        if is_ekf:
            est_res = estimator.step(obs)
            alpha_raw = 1.0
            alpha_smoothed = 1.0
            alpha_vslam = 1.0
            r_vis = 0.0
            p_opt = 1.0
            f_ext_hat = np.zeros(6)
            est_pos = est_res["estimated_pos"]
        else:
            # (A) Compute trust scores via Fault Adapter & Physics-Consistent Visual Gating (Novelty 1)
            trust_scores = fault_adapter.update(r_phys, r_sens)
            alpha_raw = trust_scores["global"]

            vg_diag = fault_adapter.compute_visual_gating(
                v_vslam=obs.get("vslam_vel", obs["gps_vel"]),
                v_prev=prev_vel,
                rot_matrix=true_st["rot_matrix"],
                motor_thrusts=info["ctrl"],
                f_ext_hat=locals().get("f_ext_hat", None),
                dt=dt,
                vslam_valid=obs.get("vslam_valid", True),
                vslam_quality=obs.get("vslam_quality", 1.0)
            )
            alpha_vslam = vg_diag["alpha_vslam"]
            r_vis = vg_diag["r_vis"]

            # (B) Virtual cGAN Sensor Inference (evaluated at 100 Hz / step % 5)
            if step % 5 == 0 or 'y_gan_cached' not in locals():
                lat_window = path.get_local_window(t, horizon_steps=5, dt=0.02)
                past_arr = np.array(past_sensor_window if len(past_sensor_window) == 5 else [current_sensor]*5)
                c_vec = pack_cgan_condition(lat_window, past_arr, info["ctrl"], r_phys)

                with torch.no_grad():
                    c_t = torch.from_numpy(c_vec).float().unsqueeze(0).to(dev)
                    y_gan_t = cgan_G(c_t)
                    p_opt_t = cgan_D(c_t, y_gan_t)
                    y_gan_cached = y_gan_t.cpu().numpy().squeeze(0)
                    p_opt_cached = float(p_opt_t.item())

            y_gan = y_gan_cached
            p_opt = p_opt_cached

            # (C) Tri-Brid Critically Damped Precision Blending (Novelty 2)
            y_fused, obs_diag = observer.fuse_tri_brid(
                y_real=current_sensor,
                y_gan=y_gan,
                vslam_pos=obs.get("vslam_pos", obs["gps_pos"]),
                vslam_vel=obs.get("vslam_vel", obs["gps_vel"]),
                alpha_vslam=alpha_vslam,
                alpha_gps=trust_scores.get("gps_pos", 1.0),
                p_opt=p_opt
            )
            alpha_smoothed = obs_diag["alpha_smoothed"]
            est_pos = observer.estimated_pos

            # (D) Continual Disturbance Adaptation with Visual-Aided Reservoir (Novelty 3)
            mlp_feat = extract_mlp_features(
                lin_vel=true_st["lin_vel"],
                ang_vel=true_st["ang_vel"],
                rot_matrix=true_st["rot_matrix"],
                tau_motors=info["ctrl"][:3]
            )

            delta_p_vis = (obs["vslam_pos"] - est_pos) if obs.get("vslam_valid", True) else np.zeros(3)

            if isinstance(continual_adapter, PIN_ESN_Adapter):
                f_ext_hat, _ = continual_adapter.update(mlp_feat, r_phys, delta_p_vis=delta_p_vis)
            elif isinstance(continual_adapter, RLSHeadAdapter):
                f_ext_hat, _ = continual_adapter.update(mlp_feat, r_phys)
            elif isinstance(continual_adapter, OnlineAnnealedSACAdapter):
                f_ext_hat, _ = continual_adapter.select_action(mlp_feat, deterministic=True)
                rew = continual_adapter.compute_reward(r_phys, r_sens, f_ext_hat)
                continual_adapter.buffer.push(mlp_feat, f_ext_hat, rew, mlp_feat)
                if step % 10 == 0:
                    continual_adapter.update_parameters()

        total_step_lat_us = (time.perf_counter_ns() - t_est_start) / 1000.0

        # Log history
        history["time"].append(t)
        history["true_pos"].append(true_st["pos"].copy())
        history["ref_pos"].append(ref.pos.copy())
        history["est_pos"].append(est_pos.copy())
        history["r_phys"].append(float(np.linalg.norm(r_phys)))
        history["r_sens"].append(float(np.linalg.norm(r_sens)))
        history["r_vis"].append(float(r_vis))
        history["alpha_raw"].append(alpha_raw)
        history["alpha_smoothed"].append(alpha_smoothed)
        history["alpha_vslam"].append(float(alpha_vslam))
        history["p_opt"].append(p_opt)
        history["latencies_us"].append(total_step_lat_us)
        history["pos_errors"].append(float(np.linalg.norm(true_st["pos"] - ref.pos)))
        history["control_jerk"].append(jerk)

        state = true_st

    # Compute summary metrics
    pos_errs = np.array(history["pos_errors"])
    ate = float(np.mean(pos_errs))
    rmse = float(np.sqrt(np.mean(np.square(pos_errs))))
    max_jerk = float(np.max(history["control_jerk"][10:])) # ignore step 0 boundary
    mean_lat = float(np.mean(history["latencies_us"]))

    return {
        "name": controller_type,
        "history": history,
        "ate": ate,
        "rmse": rmse,
        "max_jerk": max_jerk,
        "mean_latency_us": mean_lat
    }


# ==============================================================================
# 3. Publication Figures & LaTeX Table Generation
# ==============================================================================

def generate_publication_artifacts(
    all_results: Dict[str, Dict[str, Any]],
    output_dir: str = "./figures"
):
    """
    Renders IEEE publication-grade plots (Matplotlib/Seaborn style) and
    outputs a LaTeX summary table.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({
        "font.size": 11,
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "lines.linewidth": 1.8
    })

    palette = {
        "Classical EKF": "#e74c3c",          # Red
        "Variant 0 [RLS]": "#2ecc71",         # Green
        "Variant 1 [SAC Freeze 0]": "#3498db",# Blue
        "Variant 2 [SAC Freeze 1]": "#9b59b6",# Purple
        "Variant 3 [SAC Freeze 2]": "#f39c12",# Orange
        "Variant 4 [SAC Freeze 3]": "#1abc9c",# Teal
        "Paradigm B [PIN-ESN]": "#e67e22"     # Deep Amber
    }

    # --------------------------------------------------------------------------
    # Figure 1: 3D Trajectory Tracking vs. SMAC Lattice
    # --------------------------------------------------------------------------
    fig1 = plt.figure(figsize=(9, 7))
    ax1 = fig1.add_subplot(111, projection="3d")

    # Plot Reference Path
    sample_key = "Variant 0 [RLS]" if "Variant 0 [RLS]" in all_results else list(all_results.keys())[0]
    sample_hist = all_results[sample_key]["history"]
    ref_p = np.array(sample_hist["ref_pos"])
    ax1.plot(ref_p[:, 0], ref_p[:, 1], ref_p[:, 2], "k--", label="SMAC Lattice Reference", linewidth=2.2)

    # Plot representative controllers (EKF, RLS, SAC-0, and PIN-ESN)
    fig1_keys = ["Classical EKF", "Variant 0 [RLS]", "Variant 1 [SAC Freeze 0]", "Paradigm B [PIN-ESN]"]
    for key in fig1_keys:
        if key in all_results:
            res = all_results[key]
            tp = np.array(res["history"]["true_pos"])
            ax1.plot(tp[:, 0], tp[:, 1], tp[:, 2], label=key, color=palette[key], alpha=0.9)

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title("Fig 1: 3D Trajectory Tracking under Compound Disturbances & Spoofing")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig1.savefig(os.path.join(output_dir, "fig1_3d_trajectory_tracking.png"), dpi=300)
    fig1.savefig(os.path.join(output_dir, "fig1_3d_trajectory_tracking.pdf"))
    plt.close(fig1)

    # --------------------------------------------------------------------------
    # Figure 2: Position Tracking RMSE & Control Jerk
    # --------------------------------------------------------------------------
    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    t_axis = sample_hist["time"]

    # Shaded fault windows
    for ax in (ax2a, ax2b):
        ax.axvspan(4.0, 9.0, color="#f39c12", alpha=0.15, label="Wind Gust [4-9s]" if ax == ax2a else None)
        ax.axvspan(10.0, 15.0, color="#e74c3c", alpha=0.15, label="GPS Spoofing / Blackout [10-15s]" if ax == ax2a else None)

    for key in fig1_keys:
        if key in all_results:
            res = all_results[key]
            ax2a.plot(t_axis, res["history"]["pos_errors"], label=key, color=palette[key])
            # Smooth jerk for visualization
            jerk_smooth = np.convolve(res["history"]["control_jerk"], np.ones(25)/25, mode="same")
            ax2b.plot(t_axis, jerk_smooth, label=key, color=palette[key])

    ax2a.set_ylabel("Position Error (m)")
    ax2a.set_title("Fig 2: Tracking Error and Control Jerk Comparison")
    ax2a.legend(loc="upper right", framealpha=0.9)
    ax2a.grid(True, linestyle=":", alpha=0.6)

    ax2b.set_xlabel("Mission Time (s)")
    ax2b.set_ylabel(r"Control Jerk $\| \dddot{x} \|$ ($\mathrm{m/s^3}$)")
    ax2b.set_yscale("log")
    ax2b.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, "fig2_rmse_and_control_jerk.png"), dpi=300)
    fig2.savefig(os.path.join(output_dir, "fig2_rmse_and_control_jerk.pdf"))
    plt.close(fig2)

    # --------------------------------------------------------------------------
    # Figure 3: Explainable AI Dashboard (Physics, Sensors, Visual Gating & Trust)
    # --------------------------------------------------------------------------
    fig3, (ax3a, ax3b, ax3c, ax3d) = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    disp_key = "Paradigm B [PIN-ESN]" if "Paradigm B [PIN-ESN]" in all_results else ("Variant 0 [RLS]" if "Variant 0 [RLS]" in all_results else list(all_results.keys())[0])
    rls_hist = all_results[disp_key]["history"]

    # (a) Physics residual
    ax3a.plot(t_axis, rls_hist["r_phys"], color="#2c3e50", label=r"Euler-Lagrange Residual $\| r_{\mathrm{phys}} \|$")
    ax3a.axhline(2.5, color="#e74c3c", linestyle="--", alpha=0.7, label=r"Threshold $\tau_{\mathrm{thresh}}$")
    ax3a.axvspan(4.0, 9.0, color="#f39c12", alpha=0.15, label="Wind Gust [4-9s]")
    ax3a.set_ylabel(r"$\| r_{\mathrm{phys}} \|$ (N)")
    ax3a.set_title(f"Fig 3: Explainable AI Dashboard - {disp_key} (Physics Residual, Visual Gating & Trust)")
    ax3a.legend(loc="upper right")
    ax3a.grid(True, linestyle=":", alpha=0.6)

    # (b) Sensor residual
    ax3b.plot(t_axis, rls_hist["r_sens"], color="#8e44ad", label=r"Sensor Residual $\| r_{\mathrm{sens}} \|$")
    ax3b.axvspan(10.0, 15.0, color="#e74c3c", alpha=0.15, label="GPS Spoofing / Blackout [10-15s]")
    ax3b.set_ylabel(r"$\| r_{\mathrm{sens}} \|$")
    ax3b.legend(loc="upper right")
    ax3b.grid(True, linestyle=":", alpha=0.6)

    # (c) Novelty 1: Physics-Consistent Visual Gating Residual
    r_vis_plot = rls_hist.get("r_vis", [0.0]*len(t_axis))
    ax3c.plot(t_axis, r_vis_plot, color="#d35400", linewidth=1.6, label=r"Visual Gating Residual $\| r_{\mathrm{vis}} \|$ (PC-VG)")
    ax3c.axhline(1.8, color="#c0392b", linestyle=":", alpha=0.8, label=r"Visual Gate $\tau_{\mathrm{vis}}$")
    ax3c.axvspan(10.0, 15.0, color="#e74c3c", alpha=0.15)
    ax3c.set_ylabel(r"$\| r_{\mathrm{vis}} \|$ (m/s)")
    ax3c.legend(loc="upper right")
    ax3c.grid(True, linestyle=":", alpha=0.6)

    # (d) Novelty 2: Tri-Brid Trust weights (VSLAM, Smoothed Trust, Path Optimality)
    if "alpha_vslam" in rls_hist:
        ax3d.plot(t_axis, rls_hist["alpha_vslam"], color="#e67e22", linestyle="-.", label=r"VSLAM Trust $\alpha_{\mathrm{vslam}}$ (PC-VG)")
    ax3d.plot(t_axis, rls_hist["alpha_smoothed"], color="#27ae60", linewidth=2.0, label=r"Smoothed Trust $\tilde{\alpha}_t$ ($\zeta=1.0$)")
    ax3d.plot(t_axis, rls_hist["p_opt"], color="#2980b9", linestyle="--", label=r"Path Optimality $p_{\mathrm{opt}}$ ($D_\psi$)")
    ax3d.axvspan(10.0, 15.0, color="#e74c3c", alpha=0.15)
    ax3d.set_xlabel("Mission Time (s)")
    ax3d.set_ylabel("Trust Weight")
    ax3d.set_ylim(-0.05, 1.05)
    ax3d.legend(loc="lower left")
    ax3d.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig3.savefig(os.path.join(output_dir, "fig3_explainable_ai_dashboard.png"), dpi=300)
    fig3.savefig(os.path.join(output_dir, "fig3_explainable_ai_dashboard.pdf"))
    plt.close(fig3)

    # --------------------------------------------------------------------------
    # Figure 4: Ablation Bar & Pareto Frontier (RMSE vs. Latency)
    # --------------------------------------------------------------------------
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(11, 4.8))

    short_names_map = {
        "Variant 0 [RLS]": "RLS",
        "Variant 1 [SAC Freeze 0]": "SAC-F0",
        "Variant 2 [SAC Freeze 1]": "SAC-F1",
        "Variant 3 [SAC Freeze 2]": "SAC-F2",
        "Variant 4 [SAC Freeze 3]": "SAC-F3",
        "Paradigm B [PIN-ESN]": "PIN-ESN"
    }

    variants = [k for k in all_results.keys() if k != "Classical EKF"]
    rmses = [all_results[v]["rmse"] for v in variants]
    latencies = [all_results[v]["mean_latency_us"] for v in variants]
    bar_colors = [palette[v] for v in variants]
    short_names = [short_names_map.get(v, v) for v in variants]

    # Bar chart of RMSE
    bars = ax4a.bar(short_names, rmses, color=bar_colors, edgecolor="black", alpha=0.85)
    ax4a.set_ylabel("Position RMSE (m)")
    ax4a.set_title("Tracking RMSE across Ablation Variants")
    ax4a.grid(True, linestyle=":", alpha=0.6, axis="y")
    for bar in bars:
        yval = bar.get_height()
        ax4a.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f"{yval:.3f}", ha="center", va="bottom", fontsize=9)

    # Pareto Frontier: Latency vs RMSE
    for i, v in enumerate(variants):
        ax4b.scatter(latencies[i], rmses[i], color=palette[v], s=120, edgecolor="black", zorder=4, label=short_names[i])
        ax4b.annotate(short_names[i], (latencies[i] * 1.05, rmses[i]), fontsize=10, weight="bold")

    ax4b.set_xlabel(r"Mean Step Latency ($\mu\mathrm{s}$)")
    ax4b.set_ylabel("Position RMSE (m)")
    ax4b.set_xscale("log")
    ax4b.set_title(r"Pareto Frontier (Latency vs. Accuracy)")
    ax4b.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig4.savefig(os.path.join(output_dir, "fig4_ablation_pareto_chart.png"), dpi=300)
    fig4.savefig(os.path.join(output_dir, "fig4_ablation_pareto_chart.pdf"))
    plt.close(fig4)

    # --------------------------------------------------------------------------
    # LaTeX Summary Table
    # --------------------------------------------------------------------------
    latex_table = "\\begin{table}[t]\n\\centering\n"
    latex_table += "\\caption{Comparative Benchmark under Compound Environmental Disturbances and Sensor Spoofing}\n"
    latex_table += "\\label{tab:ablation_results}\n"
    latex_table += "\\begin{tabular}{lcccc}\n\\toprule\n"
    latex_table += "\\textbf{Architecture / Adaptation} & \\textbf{ATE (m)} & \\textbf{RMSE (m)} & \\textbf{Max Jerk (m/s$^3$)} & \\textbf{Step Latency ($\\mu$s)} \\\\\n\\midrule\n"

    all_keys = ["Classical EKF"] if "Classical EKF" in all_results else []
    all_keys += variants
    for key in all_keys:
        m = all_results[key]
        name_clean = key.replace("[", "(").replace("]", ")")
        latex_table += f"{name_clean:<28} & {m['ate']:.4f} & {m['rmse']:.4f} & {m['max_jerk']:>8.2f} & {m['mean_latency_us']:>8.1f} \\\\\n"

    latex_table += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

    with open(os.path.join(output_dir, "table_metrics.tex"), "w") as f:
        f.write(latex_table)

    return latex_table


def main():
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description="Run closed-loop ablation paper benchmark.")
    parser.add_argument("--duration", type=float, default=20.0, help="Mission duration in seconds (default: 20.0)")
    parser.add_argument("--checkpoints-dir", type=str, default="./checkpoints", help="Path to pre-trained checkpoints")
    parser.add_argument("--figures-dir", type=str, default="./figures", help="Output directory for publication figures")
    parser.add_argument("--only-paradigm-b", action="store_true", default=False, help="Fast benchmark only Paradigm B against Classical EKF and RLS")
    parser.add_argument("--no-pin-esn", action="store_true", default=False, help="Disable Paradigm B [PIN-ESN] in ablation suite")
    parser.add_argument("--render-cam", action="store_true", default=False, help="Render front camera snapshot for visual telemetry verification")
    args = parser.parse_args()

    device = torch.device("cpu")
    print("=" * 80)
    print(" Running Module 6: End-to-End Paper Ablation Benchmark")
    print("=" * 80)
    print(f"Mission Duration: {args.duration}s ({int(args.duration/0.002)} steps @ 500 Hz)")
    print("Compound faults scheduled:")
    print("  • Wind Gust:       t in [4.0, 9.0]s")
    print("  • IMU Thermal:     t >= 2.0s (continuous drift ramp)")
    print("  • GPS Spoofing:    t in [10.0, 15.0]s")
    print("  • LiDAR Blackout:  t in [10.0, 15.0]s")

    if getattr(args, "only_paradigm_b", False):
        controllers = [
            "Classical EKF",
            "Variant 0 [RLS]",
            "Paradigm B [PIN-ESN]"
        ]
    else:
        controllers = [
            "Classical EKF",
            "Variant 0 [RLS]",
            "Variant 1 [SAC Freeze 0]",
            "Variant 2 [SAC Freeze 1]",
            "Variant 3 [SAC Freeze 2]",
            "Variant 4 [SAC Freeze 3]"
        ]
        if not args.no_pin_esn:
            controllers.append("Paradigm B [PIN-ESN]")

    results = {}
    for ctrl in controllers:
        print(f"\n>> Simulating: {ctrl} ...")
        t_start = time.time()
        res = run_single_mission(
            controller_type=ctrl,
            duration=args.duration,
            checkpoints_dir=args.checkpoints_dir,
            device=device
        )
        sim_elapsed = time.time() - t_start
        print(f"   Done in {sim_elapsed:.2f}s | RMSE: {res['rmse']:.4f} m | ATE: {res['ate']:.4f} m | Latency: {res['mean_latency_us']:.1f} µs")
        results[ctrl] = res

    # Generate Figures and LaTeX table
    print("\n" + "=" * 80)
    print(" Generating IEEE Publication Figures & Summary Table...")
    print("=" * 80)
    latex_tbl = generate_publication_artifacts(results, output_dir=args.figures_dir)

    # Render and save Approach A forward-facing camera snapshot for visual telemetry
    try:
        sample_env = UAVMujocoEnv()
        sample_env.reset()
        cam_frame = sample_env.render_front_cam(height=240, width=320)
        if cam_frame is not None:
            cam_path = os.path.join(args.figures_dir, "front_cam_sample.png")
            matplotlib.image.imsave(cam_path, cam_frame)
            print(f"  • front_cam_sample.png (Approach A telemetry camera view)")
        sample_env.close()
    except Exception:
        pass

    print("\nSaved Figures in ./figures/:")
    print("  • fig1_3d_trajectory_tracking.png & .pdf")
    print("  • fig2_rmse_and_control_jerk.png & .pdf")
    print("  • fig3_explainable_ai_dashboard.png & .pdf (4-panel: residuals, visual gating & trust)")
    print("  • fig4_ablation_pareto_chart.png & .pdf")
    print("  • table_metrics.tex")

    print("\n" + "-" * 80)
    print(" FORMATTED LATEX SUMMARY TABLE:")
    print("-" * 80)
    print(latex_tbl)
    print("-" * 80)
    print("✔ Module 6 Ablation benchmark completed successfully!")


if __name__ == "__main__":
    main()
