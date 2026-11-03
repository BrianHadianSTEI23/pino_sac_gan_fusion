"""
Module 6: Offline Pre-Training Script (train_offline.py).
Collects rollout trajectories along the SMAC Lattice in the MuJoCo quadrotor environment,
pre-trains the PhysicsConditionedGenerator + PathOptimalityDiscriminator (Module 4)
and the SpectrallyNormalizedMLP base weights (Module 2), and saves checkpoints to ./checkpoints/.
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from config import GlobalConfig, CONFIG
from envs.uav_mujoco_env import UAVMujocoEnv
from envs.smac_lattice_sim import SMACLatticePath, SE3GeometricController
from models.pillar_b_sn_mlp import SpectrallyNormalizedMLP, extract_mlp_features
from models.physics_cgan import (
    PhysicsConditionedGenerator,
    PathOptimalityDiscriminator,
    pack_cgan_condition,
    train_cgan_step
)


def collect_offline_trajectories(
    num_rollouts: int = 50,
    trajectory_duration: float = 4.0,
    dt: float = 0.002
) -> Dict[str, np.ndarray]:
    """
    Executes flight rollouts along the SMAC 3D State Lattice with random initial conditions
    and minor aerodynamic disturbances to collect offline dataset for cGAN and SN-MLP pre-training.

    Returns dataset dict containing:
        - conditions: (N, 95)
        - real_sensors: (N, 13)
        - prev_sensors: (N, 13)
        - tau_motors: (N, 4)
        - mlp_inputs: (N, 18)
        - mlp_wrenches: (N, 6)
    """
    env = UAVMujocoEnv()
    path = SMACLatticePath()
    controller = SE3GeometricController()

    steps_per_rollout = int(trajectory_duration / dt)
    history_len = 5 # 5 past steps of sensors
    future_horizon = 5 # 5 future lattice waypoints

    conditions_list = []
    real_sensors_list = []
    prev_sensors_list = []
    tau_motors_list = []
    mlp_inputs_list = []
    mlp_wrenches_list = []

    print(f"Collecting {num_rollouts} flight trajectories ({steps_per_rollout} steps each @ 500 Hz)...")

    for ep in range(num_rollouts):
        # Slightly randomized initial position around start waypoint
        ref_0 = path.get_reference_at(0.0)
        init_pos = ref_0.pos + np.random.normal(0, 0.05, size=3)
        init_vel = ref_0.vel + np.random.normal(0, 0.02, size=3)
        state = env.reset(init_pos=init_pos, init_vel=init_vel)

        # Random wind gust disturbance for 30% of rollouts to train disturbance estimator
        has_wind = (ep % 3 == 0)
        wind_wrench = np.random.uniform(-3.0, 3.0, size=6) if has_wind else np.zeros(6)

        past_sensor_buffer = []

        for step in range(steps_per_rollout):
            t = env.sim_time
            ref = path.get_reference_at(t)
            ctrl, _ = controller.compute_control(state, ref)

            # Step physics
            obs, r_phys, r_sens, info = env.step(
                ctrl=ctrl,
                external_wrench=wind_wrench if has_wind else None,
                enable_faults=False # Clean training dataset for nominal sensor distribution
            )

            current_sensor = np.concatenate([
                obs["imu_accel"],
                obs["imu_gyro"],
                obs["gps_pos"],
                obs["gps_vel"],
                np.array([obs["lidar_down"]])
            ])

            past_sensor_buffer.append(current_sensor)
            if len(past_sensor_buffer) > history_len:
                past_sensor_buffer.pop(0)

            # Once buffer is filled, record training sample
            if len(past_sensor_buffer) == history_len and step < steps_per_rollout - 1:
                # Local future lattice window
                lattice_window = path.get_local_window(t, horizon_steps=future_horizon, dt=0.02)
                past_sensors_arr = np.array(past_sensor_buffer) # (5, 13)
                tau_m = info["ctrl"] # 4 motor commands

                c_t = pack_cgan_condition(
                    lattice_window=lattice_window,
                    past_sensors=past_sensors_arr,
                    tau_motors=tau_m,
                    r_phys=r_phys
                )

                # Feature vector for SN-MLP
                st = info["true_state"]
                mlp_in = extract_mlp_features(
                    lin_vel=st["lin_vel"],
                    ang_vel=st["ang_vel"],
                    rot_matrix=st["rot_matrix"],
                    tau_motors=tau_m[:3]
                )

                # We will record next step's sensor reading in the next iteration
                conditions_list.append(c_t)
                prev_sensors_list.append(current_sensor)
                tau_motors_list.append(tau_m)
                mlp_inputs_list.append(mlp_in)
                mlp_wrenches_list.append(wind_wrench if has_wind else np.zeros(6))

            state = info["true_state"]

    # Align real next sensors (shift by 1)
    N = len(conditions_list)
    # Generate target next sensors from recorded sequences
    real_sensors_arr = np.array(prev_sensors_list)[1:]
    # Append one duplicate for alignment
    real_sensors_arr = np.vstack([real_sensors_arr, real_sensors_arr[-1:]])

    dataset = {
        "conditions": np.array(conditions_list, dtype=np.float32),
        "real_sensors": real_sensors_arr.astype(np.float32),
        "prev_sensors": np.array(prev_sensors_list, dtype=np.float32),
        "tau_motors": np.array(tau_motors_list, dtype=np.float32),
        "mlp_inputs": np.array(mlp_inputs_list, dtype=np.float32),
        "mlp_wrenches": np.array(mlp_wrenches_list, dtype=np.float32)
    }

    print(f"Collected total {N} state-sensor-condition samples.")
    return dataset


def train_offline_cgan_and_mlp(
    dataset: Dict[str, np.ndarray],
    epochs: int = 10,
    batch_size: int = 64,
    checkpoint_dir: str = "./checkpoints",
    device: Optional[torch.device] = None
):
    """
    Pre-trains:
    1. PhysicsConditionedGenerator G_zeta + PathOptimalityDiscriminator D_psi
    2. SpectrallyNormalizedMLP base trunk weights
    """
    dev = device or torch.device("cpu")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(" Phase 1: Pre-Training Physics-Conditioned cGAN Virtual Sensor")
    print("=" * 70)

    # 1. Models
    G = PhysicsConditionedGenerator(cond_dim=95, latent_dim=16, sensor_dim=13).to(dev)
    D = PathOptimalityDiscriminator(cond_dim=95, sensor_dim=13).to(dev)

    g_opt = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_opt = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # DataLoader
    cond_t = torch.from_numpy(dataset["conditions"]).to(dev)
    real_t = torch.from_numpy(dataset["real_sensors"]).to(dev)
    prev_t = torch.from_numpy(dataset["prev_sensors"]).to(dev)
    tau_t = torch.from_numpy(dataset["tau_motors"]).to(dev)

    cgan_loader = DataLoader(
        TensorDataset(cond_t, real_t, prev_t, tau_t),
        batch_size=batch_size,
        shuffle=True
    )

    for epoch in range(1, epochs + 1):
        g_losses, d_losses, phys_losses = [], [], []

        for batch_cond, batch_real, batch_prev, batch_tau in cgan_loader:
            metrics = train_cgan_step(
                generator=G,
                discriminator=D,
                g_optimizer=g_opt,
                d_optimizer=d_opt,
                real_sensors=batch_real,
                prev_sensors=batch_prev,
                cond=batch_cond,
                tau_motors=batch_tau,
                lambda_adv=1.0,
                lambda_l1=10.0,
                lambda_phys=0.2
            )
            g_losses.append(metrics["loss_G_total"])
            d_losses.append(metrics["loss_D_total"])
            phys_losses.append(metrics["loss_G_phys"])

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss G: {np.mean(g_losses):.4f} | Loss D: {np.mean(d_losses):.4f} | Phys Res: {np.mean(phys_losses):.2f}")

    # Save cGAN checkpoints
    torch.save(G.state_dict(), os.path.join(checkpoint_dir, "cgan_generator.pt"))
    torch.save(D.state_dict(), os.path.join(checkpoint_dir, "cgan_discriminator.pt"))
    print(f"✔ cGAN Generator & Discriminator checkpoints saved to {checkpoint_dir}/")

    print("\n" + "=" * 70)
    print(" Phase 2: Pre-Training Spectrally Normalized MLP Base Trunk")
    print("=" * 70)

    mlp = SpectrallyNormalizedMLP(input_dim=18, hidden_dim=256, output_dim=6).to(dev)
    mlp_opt = optim.Adam(mlp.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    mlp_in_t = torch.from_numpy(dataset["mlp_inputs"]).to(dev)
    mlp_tgt_t = torch.from_numpy(dataset["mlp_wrenches"]).to(dev)

    mlp_loader = DataLoader(
        TensorDataset(mlp_in_t, mlp_tgt_t),
        batch_size=batch_size,
        shuffle=True
    )

    for epoch in range(1, epochs + 1):
        epoch_losses = []
        for x_b, y_b in mlp_loader:
            mlp_opt.zero_grad()
            pred = mlp(x_b)
            loss = loss_fn(pred, y_b)
            loss.backward()
            mlp_opt.step()
            epoch_losses.append(loss.item())

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - SN-MLP Wrench MSE: {np.mean(epoch_losses):.6f}")

    torch.save(mlp.state_dict(), os.path.join(checkpoint_dir, "sn_mlp_base.pt"))
    print(f"✔ SpectrallyNormalizedMLP base weights saved to {checkpoint_dir}/")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Pre-train cGAN and SN-MLP base weights offline.")
    parser.add_argument("--num-rollouts", type=int, default=20, help="Number of flight rollouts to collect (default: 20)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of pre-training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size (default: 64)")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Output directory for checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Offline Pre-Training on device: {device}")

    dataset = collect_offline_trajectories(num_rollouts=args.num_rollouts, trajectory_duration=3.0)
    train_offline_cgan_and_mlp(
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        device=device
    )


if __name__ == "__main__":
    main()
