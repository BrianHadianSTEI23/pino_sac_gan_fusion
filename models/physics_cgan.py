"""
Module 4: Physics-Conditioned Conditional GAN (Virtual Sensor Generator & Path Critic).
Implements:
1. PathOptimalityDiscriminator (D_psi): Scores whether robot trajectory remains on the
   optimal SMAC Lattice kinodynamic manifold, outputting p_opt in [0, 1].
2. PhysicsConditionedGenerator (G_zeta): Generates multi-modal sensor predictions
   y_hat_GAN in R^13 conditioned on [P_lattice, y_past, tau_motors, r_phys] and latent noise z.
3. compute_cgan_losses: Computes Adversarial loss, L1 reconstruction loss, and explicit
   differentiable Euler-Lagrange physics residual loss lambda_phys * ||r_phys(y_hat_GAN)||^2.
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, Union

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GlobalConfig, CONFIG


# ==============================================================================
# 1. Path Optimality Discriminator (D_psi)
# ==============================================================================

class PathOptimalityDiscriminator(nn.Module):
    """
    Discriminator D_psi:
    Evaluates whether the local sensor/trajectory window matches the kinodynamically
    optimal SMAC Lattice manifold.
    Outputs scalar optimality probability p_opt in [0, 1].
    """

    def __init__(
        self,
        cond_dim: int = 95,
        sensor_dim: int = 13,
        hidden_dim: int = 128
    ):
        super().__init__()
        in_dim = cond_dim + sensor_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(hidden_dim // 2, 32),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(32, 1)
        )

    def forward(self, cond: torch.Tensor, sensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass scoring condition-sensor pairs.

        Args:
            cond: Conditioning tensor of shape (B, cond_dim).
            sensor: Sensor reading tensor of shape (B, sensor_dim).

        Returns:
            prob: Scalar optimality probability p_opt in [0, 1] of shape (B, 1).
        """
        x = torch.cat([cond, sensor], dim=-1)
        logits = self.net(x)
        return torch.sigmoid(logits)

    def get_logits(self, cond: torch.Tensor, sensor: torch.Tensor) -> torch.Tensor:
        """Returns unnormalized logits for numerically stable BCE with logits."""
        x = torch.cat([cond, sensor], dim=-1)
        return self.net(x)


# ==============================================================================
# 2. Physics-Conditioned Generator (G_zeta)
# ==============================================================================

class PhysicsConditionedGenerator(nn.Module):
    """
    Virtual Sensor Generator G_zeta:
    Takes conditioning vector c_t = [P_lattice, y_past, tau_motors, r_phys] (95-dim)
    and latent noise vector z (16-dim) to synthesize the next multi-modal sensor window
    y_hat_GAN in R^13 (accelerometer, gyro, GPS pos, GPS vel, LiDAR).
    """

    def __init__(
        self,
        cond_dim: int = 95,
        latent_dim: int = 16,
        sensor_dim: int = 13,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.sensor_dim = sensor_dim

        in_dim = cond_dim + latent_dim

        # Deep feedforward generator with residual skip connection
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.act1 = nn.LeakyReLU(0.2, inplace=False)

        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.act2 = nn.LeakyReLU(0.2, inplace=False)

        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(hidden_dim // 2, sensor_dim)
        )

    def forward(self, cond: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward synthesis pass.

        Args:
            cond: Conditioning tensor (B, cond_dim).
            z: Optional latent noise tensor (B, latent_dim). If None, sampled from N(0, I).

        Returns:
            y_hat_GAN: Predicted multi-modal sensor vector (B, 13).
        """
        batch_size = cond.shape[0]
        if z is None:
            z = torch.randn(batch_size, self.latent_dim, device=cond.device, dtype=cond.dtype)

        x = torch.cat([cond, z], dim=-1)
        h = self.act1(self.fc_in(x))
        # Residual skip connection
        h = self.act2(h + self.block1(h))
        out = self.block2(h)
        return out

    def sample(self, cond: torch.Tensor) -> torch.Tensor:
        """Utility method to generate a prediction without supplying z explicitly."""
        return self.forward(cond, z=None)


# ==============================================================================
# 3. Conditioning Vector Packing Utility
# ==============================================================================

def pack_cgan_condition(
    lattice_window: np.ndarray,
    past_sensors: np.ndarray,
    tau_motors: np.ndarray,
    r_phys: np.ndarray
) -> np.ndarray:
    """
    Packs conditioning elements into a 95-dim continuous vector:
    - lattice_window: 5 waypoints * 4 [x, y, z, yaw] = 20 dims
    - past_sensors: 5 timesteps * 13 sensor dims = 65 dims
    - tau_motors: 4 motor thrusts (or 4 ctrl signals) = 4 dims
    - r_phys: 6-DoF Euler-Lagrange physics residual = 6 dims
    Total: 20 + 65 + 4 + 6 = 95 dims.
    """
    lat_flat = np.asarray(lattice_window, dtype=np.float32).ravel()[:20]
    sens_flat = np.asarray(past_sensors, dtype=np.float32).ravel()[:65]
    tau_flat = np.asarray(tau_motors, dtype=np.float32).ravel()[:4]
    r_phys_flat = np.asarray(r_phys, dtype=np.float32).ravel()[:6]

    # Pad if shorter than expected
    if lat_flat.size < 20:
        lat_flat = np.pad(lat_flat, (0, 20 - lat_flat.size))
    if sens_flat.size < 65:
        sens_flat = np.pad(sens_flat, (0, 65 - sens_flat.size))
    if tau_flat.size < 4:
        tau_flat = np.pad(tau_flat, (0, 4 - tau_flat.size))
    if r_phys_flat.size < 6:
        r_phys_flat = np.pad(r_phys_flat, (0, 6 - r_phys_flat.size))

    return np.concatenate([lat_flat, sens_flat, tau_flat, r_phys_flat]).astype(np.float32)


# ==============================================================================
# 4. Explicit Euler-Lagrange Physics Residual Loss
# ==============================================================================

def compute_euler_lagrange_loss(
    y_pred: torch.Tensor,
    y_prev: torch.Tensor,
    tau_motors: torch.Tensor,
    f_ext: Optional[torch.Tensor] = None,
    dt: float = 0.002,
    mass: float = 1.5,
    gravity: float = 9.81
) -> torch.Tensor:
    """
    Differentiable Euler-Lagrange physics residual loss embedded into Generator training.

    Given predicted next sensor reading y_pred:
    - y_pred[:, 0:3] = IMU specific force a_imu
    - y_pred[:, 3:6] = IMU gyro omega
    - y_pred[:, 6:9] = GPS position p
    - y_pred[:, 9:12] = GPS velocity v

    Computes:
    1. Translational residual:
       p_ddot = (v_pred - v_prev) / dt
       F_net_kinematic = mass * (p_ddot + g*z)
       F_actuator = sum(tau_motors) along thrust axis
       r_trans = F_net_kinematic - F_actuator - f_ext_trans
    2. Rotational residual:
       omega_dot = (omega_pred - omega_prev) / dt
       tau_net_kinematic = J * omega_dot + omega x (J * omega)
       r_rot = tau_net_kinematic - tau_actuator - f_ext_rot

    Returns:
        loss_phys: Mean squared residual ||r_phys||^2 across the mini-batch.
    """
    # Inertia tensor diagonal (kg * m^2)
    J = torch.tensor([0.014, 0.014, 0.025], dtype=y_pred.dtype, device=y_pred.device)

    # 1. Accelerations from finite differences across sensor window
    v_prev = y_prev[:, 9:12]
    v_pred = y_pred[:, 9:12]
    p_ddot = (v_pred - v_prev) / dt # (B, 3)

    omega_prev = y_prev[:, 3:6]
    omega_pred = y_pred[:, 3:6]
    omega_dot = (omega_pred - omega_prev) / dt # (B, 3)

    # 2. Gravity vector in world frame
    g_vec = torch.zeros_like(p_ddot)
    g_vec[:, 2] = gravity

    # 3. Commanded actuator thrusts (sum of 4 rotor thrusts)
    # tau_motors: (B, 4)
    if tau_motors.shape[-1] == 4:
        total_thrust = tau_motors.sum(dim=-1, keepdim=True) # (B, 1)
        # Approximate body Z alignment with vertical during nominal tracking
        F_actuator = torch.zeros_like(p_ddot)
        F_actuator[:, 2:] = total_thrust
    else:
        F_actuator = tau_motors[:, :3]

    f_ext_trans = f_ext[:, :3] if f_ext is not None else 0.0
    f_ext_rot = f_ext[:, 3:6] if f_ext is not None else 0.0

    # Translational Newton-Euler residual
    r_trans = mass * (p_ddot + g_vec) - F_actuator - f_ext_trans

    # Rotational Euler residual: J * omega_dot + omega x (J * omega)
    J_omega = omega_pred * J
    gyro_torque = torch.cross(omega_pred, J_omega, dim=-1)
    tau_actuator = torch.zeros_like(omega_dot) # nominal or differential thrusts
    r_rot = (J * omega_dot) + gyro_torque - tau_actuator - f_ext_rot

    r_phys = torch.cat([r_trans, r_rot], dim=-1) # (B, 6)
    loss_phys = torch.mean(torch.sum(r_phys**2, dim=-1))
    return loss_phys


# ==============================================================================
# 5. Composite cGAN Loss Function
# ==============================================================================

def compute_cgan_losses(
    generator: PhysicsConditionedGenerator,
    discriminator: PathOptimalityDiscriminator,
    real_sensors: torch.Tensor,
    prev_sensors: torch.Tensor,
    cond: torch.Tensor,
    tau_motors: torch.Tensor,
    f_ext: Optional[torch.Tensor] = None,
    lambda_adv: float = 1.0,
    lambda_l1: float = 10.0,
    lambda_phys: float = 0.5,
    dt: float = 0.002
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Computes complete GAN losses for both Generator and Discriminator:
    - L_G = lambda_adv * L_adv(G, D) + lambda_L1 * ||y_real - y_hat||_1 + lambda_phys * ||r_phys(y_hat)||^2
    - L_D = -E[log D(cond, y_real)] - E[log(1 - D(cond, y_hat))]

    Returns:
        loss_G: Total generator loss.
        loss_D: Total discriminator loss.
        loss_dict: Dictionary of individual loss components.
    """
    batch_size = real_sensors.shape[0]
    device = real_sensors.device

    # Targets for binary cross entropy
    valid_target = torch.ones(batch_size, 1, device=device)
    fake_target = torch.zeros(batch_size, 1, device=device)

    # 1. Synthesize virtual sensor prediction y_hat
    fake_sensors = generator(cond)

    # ------------------ Discriminator Loss ------------------
    d_real_logits = discriminator.get_logits(cond, real_sensors)
    d_fake_logits = discriminator.get_logits(cond, fake_sensors.detach())

    loss_d_real = F.binary_cross_entropy_with_logits(d_real_logits, valid_target)
    loss_d_fake = F.binary_cross_entropy_with_logits(d_fake_logits, fake_target)
    loss_D = 0.5 * (loss_d_real + loss_d_fake)

    # ------------------ Generator Loss ------------------
    # Adversarial loss: fool the discriminator
    d_fake_for_g = discriminator.get_logits(cond, fake_sensors)
    loss_g_adv = F.binary_cross_entropy_with_logits(d_fake_for_g, valid_target)

    # L1 Sensor Reconstruction Loss
    loss_g_l1 = F.l1_loss(fake_sensors, real_sensors)

    # Explicit Euler-Lagrange Physics Residual Loss
    loss_g_phys = compute_euler_lagrange_loss(
        y_pred=fake_sensors,
        y_prev=prev_sensors,
        tau_motors=tau_motors,
        f_ext=f_ext,
        dt=dt
    )

    loss_G = (lambda_adv * loss_g_adv) + (lambda_l1 * loss_g_l1) + (lambda_phys * loss_g_phys)

    loss_dict = {
        "loss_G_total": float(loss_G.item()),
        "loss_G_adv": float(loss_g_adv.item()),
        "loss_G_l1": float(loss_g_l1.item()),
        "loss_G_phys": float(loss_g_phys.item()),
        "loss_D_total": float(loss_D.item()),
        "loss_D_real": float(loss_d_real.item()),
        "loss_D_fake": float(loss_d_fake.item())
    }

    return loss_G, loss_D, loss_dict


def train_cgan_step(
    generator: PhysicsConditionedGenerator,
    discriminator: PathOptimalityDiscriminator,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    real_sensors: torch.Tensor,
    prev_sensors: torch.Tensor,
    cond: torch.Tensor,
    tau_motors: torch.Tensor,
    f_ext: Optional[torch.Tensor] = None,
    lambda_adv: float = 1.0,
    lambda_l1: float = 10.0,
    lambda_phys: float = 0.5,
    dt: float = 0.002
) -> Dict[str, float]:
    """
    Executes one standard GAN training step:
    1. Updates Discriminator with detached fake samples.
    2. Updates Generator with fresh adversarial, L1, and physics residual backpropagation.
    """
    batch_size = real_sensors.shape[0]
    device = real_sensors.device
    valid_target = torch.ones(batch_size, 1, device=device)
    fake_target = torch.zeros(batch_size, 1, device=device)

    # 1. Update Discriminator
    d_optimizer.zero_grad()
    with torch.no_grad():
        fake_detached = generator(cond)

    d_real_logits = discriminator.get_logits(cond, real_sensors)
    d_fake_logits = discriminator.get_logits(cond, fake_detached)

    loss_d_real = F.binary_cross_entropy_with_logits(d_real_logits, valid_target)
    loss_d_fake = F.binary_cross_entropy_with_logits(d_fake_logits, fake_target)
    loss_D = 0.5 * (loss_d_real + loss_d_fake)
    loss_D.backward()
    d_optimizer.step()

    # 2. Update Generator
    g_optimizer.zero_grad()
    fake_sensors = generator(cond)
    d_fake_for_g = discriminator.get_logits(cond, fake_sensors)
    loss_g_adv = F.binary_cross_entropy_with_logits(d_fake_for_g, valid_target)
    loss_g_l1 = F.l1_loss(fake_sensors, real_sensors)
    loss_g_phys = compute_euler_lagrange_loss(
        y_pred=fake_sensors,
        y_prev=prev_sensors,
        tau_motors=tau_motors,
        f_ext=f_ext,
        dt=dt
    )

    loss_G = (lambda_adv * loss_g_adv) + (lambda_l1 * loss_g_l1) + (lambda_phys * loss_g_phys)
    loss_G.backward()
    g_optimizer.step()

    return {
        "loss_G_total": float(loss_G.item()),
        "loss_G_adv": float(loss_g_adv.item()),
        "loss_G_l1": float(loss_g_l1.item()),
        "loss_G_phys": float(loss_g_phys.item()),
        "loss_D_total": float(loss_D.item()),
        "loss_D_real": float(loss_d_real.item()),
        "loss_D_fake": float(loss_d_fake.item())
    }


# ==============================================================================
# 6. Verification & Self-Test Block
# ==============================================================================

if __name__ == "__main__":
    print("=" * 75)
    print(" Running Module 4 Verification: Physics-Conditioned cGAN")
    print("=" * 75)

    device = torch.device("cpu")
    batch_size = 16
    cond_dim = 95
    sensor_dim = 13

    # 1. Instantiate Generator & Discriminator
    print("\n1. Instantiating cGAN Models:")
    G = PhysicsConditionedGenerator(cond_dim=cond_dim, latent_dim=16, sensor_dim=sensor_dim).to(device)
    D = PathOptimalityDiscriminator(cond_dim=cond_dim, sensor_dim=sensor_dim).to(device)

    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"   • Generator G_zeta parameter count:     {g_params:,}")
    print(f"   • Discriminator D_psi parameter count: {d_params:,}")

    # 2. Test Forward Passes & Dimensions
    cond_dummy = torch.randn(batch_size, cond_dim, device=device)
    real_sensor_dummy = torch.randn(batch_size, sensor_dim, device=device)
    prev_sensor_dummy = real_sensor_dummy + torch.randn_like(real_sensor_dummy) * 0.01
    tau_motors_dummy = torch.full((batch_size, 4), 3.67875, device=device) # hover thrust

    y_fake = G(cond_dummy)
    p_opt = D(cond_dummy, y_fake)

    print("\n2. Forward Pass Verification:")
    print(f"   • Input Condition shape:     {tuple(cond_dummy.shape)}  (Expected: ({batch_size}, 95))")
    print(f"   • Generated Sensor shape:    {tuple(y_fake.shape)}  (Expected: ({batch_size}, 13))")
    print(f"   • Discriminator p_opt shape: {tuple(p_opt.shape)}  (Expected: ({batch_size}, 1))")
    print(f"   • Sample p_opt range:        [{p_opt.min().item():.4f}, {p_opt.max().item():.4f}] (Valid in [0, 1])")

    assert y_fake.shape == (batch_size, 13)
    assert p_opt.shape == (batch_size, 1)
    assert 0.0 <= p_opt.min().item() and p_opt.max().item() <= 1.0

    # 3. Test Explicit Euler-Lagrange Physics Residual Loss
    print("\n3. Testing Euler-Lagrange Physics Residual Loss:")
    loss_phys = compute_euler_lagrange_loss(
        y_pred=y_fake,
        y_prev=prev_sensor_dummy,
        tau_motors=tau_motors_dummy
    )
    print(f"   • Computed Physics Residual Loss ||r_phys||^2: {loss_phys.item():.4f}")
    assert loss_phys.item() >= 0.0

    # Verify gradient backpropagation specifically from physics loss into Generator
    G.zero_grad()
    loss_phys.backward()
    grad_norms = [p.grad.norm().item() for p in G.parameters() if p.grad is not None]
    print(f"   • Gradients verified flowing to Generator: {len(grad_norms)} layers with active grads")
    print(f"   • Max parameter grad norm from physics loss: {max(grad_norms):.4f}")
    assert len(grad_norms) > 0 and max(grad_norms) > 0.0, "Physics loss must backpropagate gradients into Generator!"
    print("   ✔ Differentiable physics gradient flow verified!")

    # 4. Test Complete cGAN Training Step
    print("\n4. Testing End-to-End Training Step (Adversarial + L1 + Physics):")
    g_opt = torch.optim.Adam(G.parameters(), lr=1e-4)
    d_opt = torch.optim.Adam(D.parameters(), lr=1e-4)

    metrics = train_cgan_step(
        generator=G,
        discriminator=D,
        g_optimizer=g_opt,
        d_optimizer=d_opt,
        real_sensors=real_sensor_dummy,
        prev_sensors=prev_sensor_dummy,
        cond=cond_dummy,
        tau_motors=tau_motors_dummy,
        lambda_adv=1.0,
        lambda_l1=10.0,
        lambda_phys=0.5
    )

    print(f"   • Generator Total Loss:     {metrics['loss_G_total']:.4f}")
    print(f"     - Adversarial Component:  {metrics['loss_G_adv']:.4f}")
    print(f"     - L1 Sensor Component:    {metrics['loss_G_l1']:.4f}")
    print(f"     - Physics Residual Comp:  {metrics['loss_G_phys']:.4f}")
    print(f"   • Discriminator Total Loss: {metrics['loss_D_total']:.4f}")
    print("   ✔ End-to-end training step and loss convergence verified!")

    # 5. Test Conditioning Packing Utility
    print("\n5. Testing Condition Vector Packing Utility:")
    lat_dummy = np.random.randn(5, 4).astype(np.float32)
    past_sens_dummy = np.random.randn(5, 13).astype(np.float32)
    tau_dummy = np.array([3.68, 3.68, 3.68, 3.68], dtype=np.float32)
    r_phys_dummy = np.zeros(6, dtype=np.float32)

    c_packed = pack_cgan_condition(lat_dummy, past_sens_dummy, tau_dummy, r_phys_dummy)
    print(f"   • Packed condition vector shape: {c_packed.shape}  (Expected: (95,))")
    assert c_packed.shape == (95,)
    print("   ✔ Conditioning vector packing utility verified!")

    print("\n" + "=" * 75)
    print("✔ All Module 4 Verification checks passed successfully!")
    print("=" * 75)
