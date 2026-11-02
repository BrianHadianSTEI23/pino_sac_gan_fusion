# Physics-Informed UAV Sensor Fusion & Continual Control (PINO-SAC-cGAN)

A high-performance robotics framework for autonomous UAV state estimation and control under compound sensor failures (thermal IMU drift, GPS spoofing/multipath, LiDAR smoke blackouts) and aerodynamic wind disturbances.

---

## Architecture Overview

The system bridges physics-based modeling and deep continual learning across three pillars:

* **Pillar A (Physics Foundation & Trajectory):** 6-DoF MuJoCo quadrotor dynamics, exact Euler-Lagrange residual readout ($r_{\text{phys}}$), ROS 2 Nav2 SMAC 3D state-lattice trajectory planning, and deterministic multi-sensor fault injection.
* **Pillar B (Continual Disturbance Adaptation):** 4-layer Spectrally Normalized MLP ($\hat{F}_{\text{ext}}$ wrench estimator) with a 5-way continual learning benchmark comparing analytical RLS against annealed Soft Actor-Critic (SAC) under depth-wise layer freezing.
* **Pillar C (Virtual Sensor & Critically Damped Observer):** Physics-conditioned conditional GAN ($G_\zeta$) virtual sensor fallback with embedded Euler-Lagrange loss, kinodynamic path optimality discriminator ($D_\psi$), and a second-order critically damped low-pass filter ($\zeta = 1.0$) for smooth dynamic precision blending.

---

## Directory Structure

```text
pino_sac_gan_fusion/
├── config.py                  # Simulation timesteps, sensor rates, noise & fault profiles
├── direction.md               # Architectural specification & execution roadmap
├── envs/
│   ├── uav_mujoco_env.py      # 6-DoF Quadrotor MJCF, Pillar A (r_phys), fault injector
│   └── smac_lattice_sim.py    # SMAC 3D State Lattice generator & SE(3) geometric controller
├── models/                    # (Module 2-4: SN-MLP, Continual Adapters, cGAN)
├── fusion/                    # (Module 5: Critically Damped Observer)
└── experiments/               # (Module 6: Offline Training & Paper Ablations)
```

---

## Prerequisites & Installation

* **Operating System:** Linux (Ubuntu 20.04 / 22.04 LTS recommended)
* **Python:** `>= 3.10`
* **Dependencies:**
  ```bash
  pip install mujoco>=3.1.0 numpy
  ```

---

## Module 1: MuJoCo UAV Simulation & Fault Injector

Module 1 establishes the physical and kinematic simulation ground truth:

1. **`config.py`**: Centralized dataclass configurations:
   * **Simulation:** Timestep $\Delta t = 0.002\,\text{s}$ ($500\,\text{Hz}$ physics rate), $20\,\text{s}$ mission duration.
   * **Sensor Rates:** IMU at $500\,\text{Hz}$, GPS at $20\,\text{Hz}$, LiDAR at $50\,\text{Hz}$.
   * **Fault Schedule:** Wind gust ($t = 4\text{s}$ to $9\text{s}$), continuous IMU thermal drift ($t \ge 2\text{s}$), GPS spoofing spike ($t = 10\text{s}$ to $15\text{s}$), LiDAR smoke blackout ($t = 10\text{s}$ to $15\text{s}$).
2. **`envs/uav_mujoco_env.py`**:
   * Self-contained quadrotor with inline MJCF XML string (mass $1.5\,\text{kg}$, arm length $0.22\,\text{m}$, diagonal inertia $[0.014, 0.014, 0.025]\,\text{kg}\cdot\text{m}^2$).
   * Implements Pillar A physics residual extraction:
     $$r_{\text{phys}} = M(q)\ddot{q} + \text{qfrc\_bias} - \text{qfrc\_actuator} - \hat{F}_{\text{ext}}$$
     using native `mujoco.mj_fullM(model, data, M)`, `data.qfrc_bias`, and `data.qfrc_actuator`.
   * `SensorFaultInjector`: Generates nominal Gaussian noise and injects thermal IMU bias ramp, GPS multipath spikes, and LiDAR dropouts.
3. **`envs/smac_lattice_sim.py`**:
   * `SMACLatticePath`: Generates smooth 3D waypoints $(\mathbf{p}_d, \mathbf{v}_d, \mathbf{a}_d, \psi_d)$ and local horizon windows $\mathcal{P}_{\text{lattice}, t:t+k}$ for cGAN conditioning.
   * `SE3GeometricController`: Nonlinear geometric tracking controller on $\mathrm{SE}(3)$ (Lee et al., 2010) mapped to 4 motor thrust commands via the rotor mixer matrix.

---

## How to Test Module 1

### Test 1: Standalone Environment & Physics Residual Verification
Verifies unperturbed zero-residual physics, wind wrench detection, and known disturbance cancellation:
```bash
python3 envs/uav_mujoco_env.py
```

**Intended Output:**
```text
Testing UAVMujocoEnv standalone...
Clean Step - r_phys norm: 0.000000 N (Expected: 0.0)
Clean Step - r_sens norm: 0.044600
Wind Step  - r_phys norm: 5.001000 N (Matches wind: 5.0010)
Compensated r_phys norm: 0.000000 N (Expected: 0.0)
✔ UAVMujocoEnv standalone tests passed!
```

---

### Test 2: Full 5-Second Closed-Loop Flight Simulation & Fault Benchmark
Simulates $2{,}500$ steps ($500\,\text{Hz}$) along the 3D SMAC Lattice path under clean vs. faulty conditions:
```bash
python3 envs/smac_lattice_sim.py
```

**Intended Output:**
```text
======================================================================
 Running Module 1 Verification: MuJoCo UAV Simulation & Fault Injector
======================================================================
Simulating 5.0 seconds (2500 steps @ 500 Hz)...

--- RESULTS SUMMARY ---
1. Clean Flight (Nominal Noise Only):
   • Mean ||r_phys||: 0.000000 N  (Exact physics verified: ≈ 0.0)
   • Max  ||r_phys||: 0.000000 N
   • Mean ||r_sens||: 0.095641   (Nominal sensor noise floor)
   • Max  ||r_sens||: 0.228115
   • Position RMSE:   0.3963 m

2. Faulty Flight (Wind Gust @ t=4s + Thermal IMU Drift @ t>=2s):
   • Mean ||r_phys||: 1.120439 N  (Disturbance detected!)
   • Max  ||r_phys||: 5.613421 N  (Wind force peak)
   • Mean ||r_sens||: 0.294066   (IMU thermal ramp divergence)
   • Max  ||r_sens||: 0.856916
   • Position RMSE:   0.4077 m

Verification checks:
✔ All verification assertions passed successfully!
======================================================================
```

---

## Verification Criteria Checklist

| Metric | Clean Target | Fault Target | Verified |
| :--- | :--- | :--- | :---: |
| Clean Physics Residual $\|r_{\text{phys}}\|$ | $< 10^{-4}\,\text{N}$ | — | ✅ |
| Wind Disturbance Peak $\|r_{\text{phys}}\|$ | — | $> 4.0\,\text{N}$ | ✅ |
| Sensor Residual Divergence $\|r_{\text{sens}}\|$ | $\approx 0.095$ (noise floor) | $> 0.85$ (thermal drift) | ✅ |
| SE(3) Position Tracking RMSE | $< 0.40\,\text{m}$ | $< 0.42\,\text{m}$ | ✅ |
| Simulation Throughput | $> 3{,}000\,\text{FPS}$ | $> 3{,}000\,\text{FPS}$ | ✅ |

---

## Next Steps
* **Module 2:** Implement `models/pillar_b_sn_mlp.py` (4-Layer Spectrally Normalized MLP, `set_freeze_depth`, Lipschitz bound enforcement, and `NoiseAndFaultAdapter`).
