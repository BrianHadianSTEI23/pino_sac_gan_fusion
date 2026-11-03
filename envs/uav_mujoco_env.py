"""
UAV MuJoCo Environment and Sensor Fault Injector.
Implements Pillar A of the Physics-Informed Sensor Fusion framework.
Extracts exact Euler-Lagrange terms (M, qfrc_bias, qfrc_actuator) to compute
physics residual r_phys, and injects deterministic thermal IMU drift, GPS spikes,
and LiDAR blackouts.
"""

import sys
import os
# Ensure headless EGL rendering is prioritized for MuJoCo Renderer
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path
from typing import Dict, Optional, Tuple, Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import mujoco

from config import GlobalConfig, CONFIG


# Self-contained Skydio X2 / Quadrotor MJCF definition with Front Camera & Visual Landmarks
QUADROTOR_MJCF = """
<mujoco model="skydio_quadrotor">
  <option timestep="0.002" gravity="0 0 -9.81" iterations="4"/>
  <default>
    <geom density="0"/>
  </default>

  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.25 0.35 0.45" rgb2="0.15 0.20 0.25"/>
    <material name="grid_mat" texture="grid" texrepeat="20 20" reflectance="0.2"/>
  </asset>

  <worldbody>
    <geom name="ground" type="plane" size="50 50 0.1" material="grid_mat"/>
    <!-- Visual landmarks for front camera VSLAM feature extraction & telemetry -->
    <geom name="landmark_box1" type="box" size="0.5 0.5 1.5" pos="4.0 2.5 1.5" rgba="0.9 0.4 0.2 1"/>
    <geom name="landmark_box2" type="box" size="0.6 0.6 2.0" pos="8.0 -3.0 2.0" rgba="0.2 0.7 0.4 1"/>
    <geom name="landmark_cyl1" type="cylinder" size="0.4 1.8" pos="12.0 1.5 1.8" rgba="0.3 0.5 0.9 1"/>
    <geom name="landmark_box3" type="box" size="0.5 0.5 1.2" pos="16.0 -2.0 1.2" rgba="0.8 0.8 0.2 1"/>

    <body name="uav" pos="0 0 1.0">
      <freejoint name="root"/>
      <!-- Quadrotor inertial properties: 1.5 kg, arm length 0.22 m -->
      <inertial pos="0 0 0" mass="1.5" diaginertia="0.014 0.014 0.025"/>
      <geom name="fuselage" type="box" size="0.12 0.12 0.04" rgba="0.2 0.2 0.25 1"/>
      <geom name="arm1" type="cylinder" fromto="0 0 0 0.15556 0.15556 0" size="0.01" rgba="0.8 0.1 0.1 1"/>
      <geom name="arm2" type="cylinder" fromto="0 0 0 -0.15556 0.15556 0" size="0.01" rgba="0.8 0.1 0.1 1"/>
      <geom name="arm3" type="cylinder" fromto="0 0 0 -0.15556 -0.15556 0" size="0.01" rgba="0.2 0.2 0.8 1"/>
      <geom name="arm4" type="cylinder" fromto="0 0 0 0.15556 -0.15556 0" size="0.01" rgba="0.2 0.2 0.8 1"/>

      <!-- Forward-Facing Telemetry / VSLAM Camera -->
      <camera name="front_cam" pos="0.14 0 0.02" euler="0 15 0" fovy="80"/>

      <!-- Rotor actuation sites (X-configuration) -->
      <site name="rotor1" pos="0.15556 0.15556 0.02" size="0.02"/>
      <site name="rotor2" pos="-0.15556 0.15556 0.02" size="0.02"/>
      <site name="rotor3" pos="-0.15556 -0.15556 0.02" size="0.02"/>
      <site name="rotor4" pos="0.15556 -0.15556 0.02" size="0.02"/>

      <!-- Sensor sites -->
      <site name="imu_site" pos="0 0 0" size="0.01"/>
      <site name="lidar_site" pos="0 0 -0.04" zaxis="0 0 -1" size="0.01"/>
    </body>
  </worldbody>

  <actuator>
    <!-- 4 rotor actuators producing thrust along body +Z and torque along body Z -->
    <general name="motor1" site="rotor1" gear="0 0 1 0 0 -0.015" ctrlrange="0 10"/>
    <general name="motor2" site="rotor2" gear="0 0 1 0 0 0.015" ctrlrange="0 10"/>
    <general name="motor3" site="rotor3" gear="0 0 1 0 0 -0.015" ctrlrange="0 10"/>
    <general name="motor4" site="rotor4" gear="0 0 1 0 0 0.015" ctrlrange="0 10"/>
  </actuator>

  <sensor>
    <accelerometer name="imu_accel" site="imu_site"/>
    <gyro name="imu_gyro" site="imu_site"/>
    <framepos name="gps_pos" objtype="site" objname="imu_site"/>
    <framelinvel name="gps_vel" objtype="site" objname="imu_site"/>
    <rangefinder name="lidar_down" site="lidar_site"/>
  </sensor>
</mujoco>
"""


def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert MuJoCo quaternion [w, x, y, z] to 3x3 rotation matrix R in SO(3).
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ], dtype=np.float64)


class SensorFaultInjector:
    """
    Manages nominal multi-modal sensor generation and deterministic fault injection:
    - IMU thermal drift: linear bias ramp starting at t = 2.0s
    - GPS spoofing/multipath: large bias spike & noise multiplier between t = 10.0s and 15.0s
    - LiDAR smoke blackout: zeroing / invalid reading between t = 10.0s and 15.0s
    """

    def __init__(self, config: GlobalConfig = CONFIG, seed: int = 42):
        self.cfg = config
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        """Reset internal accumulator states for drift and timing."""
        self.imu_drift_accum = np.zeros(6, dtype=np.float64)  # 3 accel, 3 gyro

    def inject(
        self,
        t: float,
        clean_sensors: Dict[str, np.ndarray],
        enable_faults: bool = True
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
        """
        Takes raw clean sensor values from MuJoCo, applies nominal Gaussian noise,
        and optionally injects mission faults.

        Returns:
            noisy_sensors: Sensordata with nominal noise (+ faults if active)
            nominal_sensors: Sensor data with ONLY nominal noise (clean reference)
            r_sens: Multi-modal sensor residual (noisy - expected nominal)
        """
        noise_cfg = self.cfg.noise
        fault_cfg = self.cfg.faults

        # 1. Base Nominal Sensor Readings + Gaussian Noise
        accel_noise = self.rng.normal(0.0, noise_cfg.accel_std, size=3)
        gyro_noise = self.rng.normal(0.0, noise_cfg.gyro_std, size=3)
        gps_pos_noise = self.rng.normal(0.0, noise_cfg.gps_pos_std, size=3)
        gps_vel_noise = self.rng.normal(0.0, noise_cfg.gps_vel_std, size=3)
        lidar_noise = float(self.rng.normal(0.0, noise_cfg.lidar_std))

        # Nominal VSLAM visual odometry (30 Hz visual tracking)
        vslam_pos_noise = self.rng.normal(0.0, noise_cfg.vslam_pos_std, size=3)
        vslam_vel_noise = self.rng.normal(0.0, noise_cfg.vslam_vel_std, size=3)

        nominal_sensors = {
            "imu_accel": clean_sensors["imu_accel"] + accel_noise,
            "imu_gyro": clean_sensors["imu_gyro"] + gyro_noise,
            "gps_pos": clean_sensors["gps_pos"] + gps_pos_noise,
            "gps_vel": clean_sensors["gps_vel"] + gps_vel_noise,
            "lidar_down": float(clean_sensors["lidar_down"] + lidar_noise),
            "vslam_pos": clean_sensors["gps_pos"] + vslam_pos_noise,
            "vslam_vel": clean_sensors["gps_vel"] + vslam_vel_noise,
            "vslam_valid": True,
            "vslam_features": int(self.rng.normal(140, 8)),
            "vslam_quality": float(np.clip(self.rng.normal(0.96, 0.02), 0.0, 1.0))
        }

        # Fault offsets (default zero)
        accel_fault = np.zeros(3, dtype=np.float64)
        gyro_fault = np.zeros(3, dtype=np.float64)
        gps_pos_fault = np.zeros(3, dtype=np.float64)
        gps_vel_fault = np.zeros(3, dtype=np.float64)
        lidar_fault = 0.0
        vslam_pos_fault = np.zeros(3, dtype=np.float64)
        vslam_vel_fault = np.zeros(3, dtype=np.float64)
        vslam_valid = True
        vslam_features = nominal_sensors["vslam_features"]
        vslam_quality = nominal_sensors["vslam_quality"]

        if enable_faults:
            # (A) Thermal IMU Ramp Drift
            if t >= fault_cfg.imu_drift_start_time:
                dt_drift = t - fault_cfg.imu_drift_start_time
                accel_fault += fault_cfg.imu_accel_drift_rate * dt_drift
                gyro_fault += fault_cfg.imu_gyro_drift_rate * dt_drift

            # (B) GPS Multipath / Spoofing Spike
            if fault_cfg.gps_fault_start_time <= t <= fault_cfg.gps_fault_end_time:
                gps_pos_fault += fault_cfg.gps_pos_bias
                # additional high-frequency multipath jitter
                gps_pos_fault += self.rng.normal(
                    0.0,
                    noise_cfg.gps_pos_std * fault_cfg.gps_pos_noise_multiplier,
                    size=3
                )
                gps_vel_fault += self.rng.normal(0.0, 0.5, size=3)

            # (C) LiDAR Smoke Blackout
            if fault_cfg.lidar_fault_start_time <= t <= fault_cfg.lidar_fault_end_time:
                # Blackout sets range to maximum range / dropout flag (e.g. 50.0m or 0.0m)
                lidar_fault = 40.0 - nominal_sensors["lidar_down"]

            # (D) VSLAM Visual Blackout / Smoke / Feature Deprivation
            if fault_cfg.vslam_fault_start_time <= t <= fault_cfg.vslam_fault_end_time:
                vslam_valid = False
                vslam_features = int(self.rng.integers(1, 6))
                vslam_quality = 0.05
                # Divergence / feature slip offset
                vslam_pos_fault += np.array([8.5, -6.0, 3.2])
                vslam_vel_fault += self.rng.normal(0.0, 1.5, size=3)

            # (E) Motion Blur during high angular rotation rate
            ang_rate_norm = float(np.linalg.norm(clean_sensors["imu_gyro"]))
            if ang_rate_norm > fault_cfg.vslam_motion_blur_thresh:
                vslam_features = max(8, int(vslam_features * 0.35))
                vslam_quality *= 0.4
                vslam_pos_fault += self.rng.normal(0.0, 0.3, size=3)

        # Corrupted / Faulty Sensors
        noisy_sensors = {
            "imu_accel": nominal_sensors["imu_accel"] + accel_fault,
            "imu_gyro": nominal_sensors["imu_gyro"] + gyro_fault,
            "gps_pos": nominal_sensors["gps_pos"] + gps_pos_fault,
            "gps_vel": nominal_sensors["gps_vel"] + gps_vel_fault,
            "lidar_down": float(nominal_sensors["lidar_down"] + lidar_fault),
            "vslam_pos": nominal_sensors["vslam_pos"] + vslam_pos_fault,
            "vslam_vel": nominal_sensors["vslam_vel"] + vslam_vel_fault,
            "vslam_valid": vslam_valid,
            "vslam_features": vslam_features,
            "vslam_quality": vslam_quality
        }

        # Multi-modal sensor residual r_sens:
        # First 13 dims: IMU + GPS + LiDAR (100% backwards-compatible with cGAN)
        # Next 6 dims: VSLAM position and velocity innovations
        r_sens_nom13 = np.concatenate([
            noisy_sensors["imu_accel"] - clean_sensors["imu_accel"],
            noisy_sensors["imu_gyro"] - clean_sensors["imu_gyro"],
            noisy_sensors["gps_pos"] - clean_sensors["gps_pos"],
            noisy_sensors["gps_vel"] - clean_sensors["gps_vel"],
            np.array([noisy_sensors["lidar_down"] - clean_sensors["lidar_down"]])
        ])
        r_vslam = np.concatenate([
            noisy_sensors["vslam_pos"] - clean_sensors["gps_pos"],
            noisy_sensors["vslam_vel"] - clean_sensors["gps_vel"]
        ])
        r_sens = np.concatenate([r_sens_nom13, r_vslam])

        return noisy_sensors, nominal_sensors, r_sens


class UAVMujocoEnv:
    """
    MuJoCo-based Quadrotor simulation environment implementing Pillar A.
    Provides fast headless simulation, Euler-Lagrange physics extraction,
    onboard forward camera rendering (Approach A), and native actuator force readout.
    """

    def __init__(self, config: GlobalConfig = CONFIG, seed: int = 42):
        self.cfg = config
        self.model = mujoco.MjModel.from_xml_string(QUADROTOR_MJCF)
        self.data = mujoco.MjData(self.model)

        self.dt = self.model.opt.timestep
        self.uav_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "uav")

        # Preallocate mass matrix workspace
        self._M = np.zeros((self.model.nv, self.model.nv), dtype=np.float64)

        # Fault injector
        self.fault_injector = SensorFaultInjector(config=config, seed=seed)
        self.sim_time = 0.0

        # Offscreen camera renderer cache (Approach A)
        self._renderer: Optional[mujoco.Renderer] = None
        self.data_vslam: Dict[str, Any] = {}

        self.reset()

    def reset(
        self,
        init_pos: Optional[np.ndarray] = None,
        init_quat: Optional[np.ndarray] = None,
        init_vel: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Reset simulation state and timers."""
        mujoco.mj_resetData(self.model, self.data)
        self.sim_time = 0.0
        self.fault_injector.reset()

        # Set initial position (default: [0, 0, 1.0])
        if init_pos is not None:
            self.data.qpos[0:3] = init_pos
        else:
            self.data.qpos[0:3] = np.array([0.0, 0.0, 1.0])

        # Set initial orientation quaternion [w, x, y, z] (default: identity [1, 0, 0, 0])
        if init_quat is not None:
            self.data.qpos[3:7] = init_quat
        else:
            self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])

        if init_vel is not None:
            if len(init_vel) == 3:
                self.data.qvel[0:3] = init_vel
                self.data.qvel[3:6] = 0.0
            else:
                self.data.qvel[0:6] = init_vel

        # Initialize hover motor control: 1.5 * 9.81 / 4 per rotor
        hover_per_rotor = self.cfg.uav.hover_thrust / 4.0
        self.data.ctrl[:] = hover_per_rotor

        mujoco.mj_forward(self.model, self.data)
        return self.get_state()

    def get_mass_matrix(self) -> np.ndarray:
        """
        Extract the full 6x6 mass/inertia matrix M(q) from MuJoCo.
        Compatible across all MuJoCo 3.x Python binding versions.
        """
        try:
            mujoco.mj_fullM(self.model, self.data, self._M)
        except TypeError:
            try:
                mujoco.mj_fullM(self.model, self._M, self.data.qM)
            except Exception:
                mujoco.mj_fullM(self.model, self._M, self.data)
        return self._M.copy()

    def compute_physics_residual(
        self,
        estimated_wrench: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Implements Pillar A:
        r_phys = M(q)*qacc + qfrc_bias - qfrc_actuator - F_ext

        Returns:
            r_phys: 6-dimensional residual vector [Fx, Fy, Fz, Tx, Ty, Tz].
                    Evaluates to zero when dynamics and external forces are fully matched.
        """
        M = self.get_mass_matrix()
        qacc = self.data.qacc.copy()
        qfrc_bias = self.data.qfrc_bias.copy()
        qfrc_act = self.data.qfrc_actuator.copy()

        # Unmodeled external wrench estimate (defaults to 0)
        f_ext = estimated_wrench if estimated_wrench is not None else np.zeros(6, dtype=np.float64)

        # Euler-Lagrange physics discrepancy
        r_phys = (M @ qacc) + qfrc_bias - qfrc_act - f_ext
        return r_phys

    def apply_wind_disturbance(self, t: float) -> np.ndarray:
        """
        Applies aerodynamic wind gust disturbance to the UAV body if within
        the active wind window [t_start, t_end].
        """
        fault_cfg = self.cfg.faults
        wrench = np.zeros(6, dtype=np.float64)
        if fault_cfg.wind_start_time <= t <= fault_cfg.wind_end_time:
            wrench[0:3] = fault_cfg.wind_force
            wrench[3:6] = fault_cfg.wind_torque
            # Apply to generalized applied forces
            self.data.qfrc_applied[0:6] = wrench
        else:
            self.data.qfrc_applied[0:6] = 0.0
        return wrench

    def step(
        self,
        ctrl: np.ndarray,
        external_wrench: Optional[np.ndarray] = None,
        enable_faults: bool = True
    ) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Executes one physics simulation step.

        Args:
            ctrl: 4-element motor thrust commands [T1, T2, T3, T4] in Newtons.
            external_wrench: Optional 6-element external wrench override.
            enable_faults: Whether to inject wind and sensor faults.

        Returns:
            obs: Dictionary of noisy sensor observations
            r_phys: 6-DoF Euler-Lagrange physics residual vector
            r_sens: 13-DoF sensor residual vector
            info: Diagnostics containing ground-truth state, motor efforts, nominal sensors
        """
        # Set motor controls (clipped to valid ranges)
        self.data.ctrl[:] = np.clip(ctrl, 0.0, 10.0)

        # Apply external aerodynamic disturbance (e.g. wind)
        if external_wrench is not None:
            self.data.qfrc_applied[0:6] = external_wrench
            applied_wrench = external_wrench
        elif enable_faults:
            applied_wrench = self.apply_wind_disturbance(self.sim_time)
        else:
            self.data.qfrc_applied[0:6] = 0.0
            applied_wrench = np.zeros(6, dtype=np.float64)

        # Step physics forward
        mujoco.mj_step(self.model, self.data)
        self.sim_time += self.dt

        # Read exact ground-truth sensor signals from MuJoCo
        clean_sensors = {
            "imu_accel": self.data.sensor("imu_accel").data.copy(),
            "imu_gyro": self.data.sensor("imu_gyro").data.copy(),
            "gps_pos": self.data.sensor("gps_pos").data.copy(),
            "gps_vel": self.data.sensor("gps_vel").data.copy(),
            "lidar_down": float(self.data.sensor("lidar_down").data[0])
        }

        # Inject nominal noise and mission faults
        noisy_sensors, nominal_sensors, r_sens = self.fault_injector.inject(
            t=self.sim_time,
            clean_sensors=clean_sensors,
            enable_faults=enable_faults
        )

        # Compute Pillar A physics residual (without knowing applied wrench, to test disturbance detection)
        r_phys = self.compute_physics_residual(estimated_wrench=None)

        self.data_vslam = {
            "vslam_pos": noisy_sensors["vslam_pos"].copy(),
            "vslam_vel": noisy_sensors["vslam_vel"].copy(),
            "vslam_valid": noisy_sensors["vslam_valid"],
            "vslam_features": noisy_sensors["vslam_features"],
            "vslam_quality": noisy_sensors["vslam_quality"]
        }

        info = {
            "time": self.sim_time,
            "true_state": self.get_state(),
            "nominal_sensors": nominal_sensors,
            "clean_sensors": clean_sensors,
            "applied_wrench": applied_wrench,
            "qfrc_actuator": self.data.qfrc_actuator.copy(),
            "ctrl": self.data.ctrl.copy(),
            "vslam": self.data_vslam
        }

        return noisy_sensors, r_phys, r_sens, info

    def render_front_cam(
        self,
        height: int = 240,
        width: int = 320,
        camera_name: str = "front_cam"
    ) -> Optional[np.ndarray]:
        """
        Approach A: Direct offscreen OpenGL/EGL rendering of the forward-facing camera.
        Returns RGB image as (H, W, 3) uint8 numpy array, or None if rendering fails/headless.
        """
        import os
        if self._renderer is None:
            if "MUJOCO_GL" not in os.environ:
                os.environ["MUJOCO_GL"] = "egl"
            try:
                self._renderer = mujoco.Renderer(self.model, height=height, width=width)
            except Exception:
                return None

        try:
            self._renderer.update_scene(self.data, camera=camera_name)
            return self._renderer.render()
        except Exception:
            return None

    def get_qgc_telemetry(
        self,
        fused_state: Optional[Dict[str, np.ndarray]] = None,
        trust_scores: Optional[Dict[str, float]] = None,
        r_phys: Optional[np.ndarray] = None,
        r_sens: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Formats drone state and sensor fusion metrics into a structured telemetry packet
        compatible with MAVLink / QGroundControl ground station protocols.
        """
        st = self.get_state()
        pos = st["pos"]
        vel = st["lin_vel"]
        rot = st["rot_matrix"]
        omega = st["ang_vel"]
        ctrl = self.data.ctrl.copy()

        # Compute Euler angles (Roll, Pitch, Yaw) in degrees from rot_matrix
        pitch = np.arcsin(-np.clip(rot[2, 0], -1.0, 1.0))
        roll = np.arctan2(rot[2, 1], rot[2, 2])
        yaw = np.arctan2(rot[1, 0], rot[0, 0])

        r_phys_norm = float(np.linalg.norm(r_phys)) if r_phys is not None else 0.0
        r_sens_norm = float(np.linalg.norm(r_sens)) if r_sens is not None else 0.0

        telemetry = {
            "timestamp_ms": int(self.sim_time * 1000.0),
            "flight_mode": "AUTO_TRACKING",
            "attitude_euler_deg": {
                "roll": float(np.degrees(roll)),
                "pitch": float(np.degrees(pitch)),
                "yaw": float(np.degrees(yaw))
            },
            "attitude_rate_dps": {
                "p": float(np.degrees(omega[0])),
                "q": float(np.degrees(omega[1])),
                "r": float(np.degrees(omega[2]))
            },
            "position_ned_m": {
                "north": float(pos[0]),
                "east": float(pos[1]),
                "down": float(-pos[2])
            },
            "velocity_ned_mps": {
                "vn": float(vel[0]),
                "ve": float(vel[1]),
                "vd": float(-vel[2])
            },
            "battery": {
                "voltage_v": max(14.0, 16.8 - 0.04 * self.sim_time),
                "remaining_pct": max(0.0, 100.0 - 2.5 * self.sim_time)
            },
            "motors_pwm_thrust_N": [float(c) for c in ctrl],
            "vslam_telemetry": self.data_vslam,
            "fusion_diagnostics": {
                "r_phys_norm": r_phys_norm,
                "r_sens_norm": r_sens_norm,
                "trust_scores": trust_scores or {},
                "virtual_fallback_active": bool(trust_scores.get("alpha_smoothed", 1.0) < 0.3) if trust_scores else False
            }
        }
        return telemetry

    def get_state(self) -> Dict[str, Any]:
        """Extract clean 6-DoF SE(3) kinematic state."""
        pos = self.data.qpos[0:3].copy()
        quat = self.data.qpos[3:7].copy()
        lin_vel = self.data.qvel[0:3].copy()
        ang_vel = self.data.qvel[3:6].copy()
        rot_matrix = quat_to_rot_matrix(quat)
        qacc = self.data.qacc.copy()

        return {
            "pos": pos,
            "quat": quat,
            "rot_matrix": rot_matrix,
            "lin_vel": lin_vel,
            "ang_vel": ang_vel,
            "lin_acc": qacc[0:3],
            "ang_acc": qacc[3:6],
            "altitude": pos[2]
        }

    def close(self):
        """Cleanly releases renderer and physics resources."""
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None


if __name__ == "__main__":
    print("Testing UAVMujocoEnv standalone with VSLAM & Front Camera...")
    env = UAVMujocoEnv()
    state = env.reset()
    hover_ctrl = env.cfg.uav.hover_thrust / 4.0 * np.ones(4)

    # 1. Unperturbed step
    obs, r_phys_clean, r_sens_clean, info = env.step(hover_ctrl, enable_faults=False)
    print(f"Clean Step - r_phys norm: {np.linalg.norm(r_phys_clean):.6f} N (Expected: 0.0)")
    print(f"Clean Step - r_sens shape: {r_sens_clean.shape} (19 dims: 13 nominal + 6 VSLAM)")
    print(f"VSLAM Active: {obs['vslam_valid']} (Features: {obs['vslam_features']}, Quality: {obs['vslam_quality']:.2f})")

    # 2. Test Approach A front camera rendering
    rgb_frame = env.render_front_cam(height=120, width=160)
    if rgb_frame is not None:
        print(f"Approach A: Front camera rendered successfully! Frame shape: {rgb_frame.shape}")
    else:
        print("Approach A: Renderer skipped (no display/EGL available in current environment)")

    # 3. Test QGroundControl telemetry packet generation
    telem = env.get_qgc_telemetry(r_phys=r_phys_clean, r_sens=r_sens_clean)
    print(f"QGC Telemetry Packet Generated: Time {telem['timestamp_ms']} ms | Flight Mode: {telem['flight_mode']}")
    print(f"  • Position NED: North {telem['position_ned_m']['north']:.2f}m, Down {telem['position_ned_m']['down']:.2f}m")
    print(f"  • Battery: {telem['battery']['voltage_v']:.1f}V ({telem['battery']['remaining_pct']:.0f}%)")

    # 4. Perturbed step with wind & Known external wrench compensation
    test_wind = np.array([5.0, 0.0, 0.0, 0.0, 0.1, 0.0])
    obs, r_phys_wind, r_sens_wind, info = env.step(hover_ctrl, external_wrench=test_wind)
    print(f"Wind Step  - r_phys norm: {np.linalg.norm(r_phys_wind):.6f} N (Matches wind: {np.linalg.norm(test_wind):.4f})")
    r_phys_compensated = env.compute_physics_residual(estimated_wrench=test_wind)
    print(f"Compensated r_phys norm: {np.linalg.norm(r_phys_compensated):.6f} N (Expected: 0.0)")

    assert np.linalg.norm(r_phys_clean) < 1e-5
    assert r_sens_clean.shape[0] == 19
    assert np.isclose(np.linalg.norm(r_phys_wind), np.linalg.norm(test_wind), atol=1e-3)
    assert np.linalg.norm(r_phys_compensated) < 1e-5
    print("✔ UAVMujocoEnv standalone tests with VSLAM & Front Camera passed!")
