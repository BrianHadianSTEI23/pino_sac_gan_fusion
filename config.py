"""
Configuration module for Physics-Informed UAV Sensor Fusion and Control.
Defines simulation timesteps, sensor rates, fault injection profiles,
and physical parameters for the quadrotor platform.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class SimConfig:
    """Simulation time and physics configuration."""
    dt: float = 0.002             # MuJoCo physics step (500 Hz)
    sim_duration: float = 20.0    # Default mission duration in seconds
    substeps: int = 1             # MuJoCo solver iterations per step
    gravity: float = 9.81         # m/s^2


@dataclass
class UAVConfig:
    """Quadrotor geometric and inertial parameters."""
    mass: float = 1.5             # Total mass (kg)
    arm_length: float = 0.22      # Distance from CoM to rotor (m)
    k_thrust: float = 0.15        # Thrust coefficient per rotor (N / normalized ctrl unit)
    k_torque: float = 0.015       # Yaw torque coefficient per rotor (N*m / normalized ctrl unit)
    
    # Inertia tensor diagonal (kg * m^2)
    ixx: float = 0.014
    iyy: float = 0.014
    izz: float = 0.025
    
    # Motor control limits [min, max] normalized
    ctrl_min: float = 0.0
    ctrl_max: float = 1.0
    hover_thrust: float = mass * 9.81  # ~14.715 N collective (~3.68 N/rotor)


@dataclass
class SensorRates:
    """Sensor sampling intervals (in seconds)."""
    imu_dt: float = 0.002         # 500 Hz (every physics step)
    gps_dt: float = 0.050         # 20 Hz (every 25 steps)
    lidar_dt: float = 0.020       # 50 Hz (every 10 steps)


@dataclass
class SensorNoiseConfig:
    """Nominal sensor noise standard deviations (clean conditions)."""
    # IMU: accelerometer (m/s^2), gyro (rad/s)
    accel_std: float = 0.02
    gyro_std: float = 0.005
    
    # GPS: position (m), velocity (m/s)
    gps_pos_std: float = 0.05
    gps_vel_std: float = 0.02
    
    # LiDAR: distance / range (m)
    lidar_std: float = 0.01


@dataclass
class FaultProfileConfig:
    """
    Deterministic fault injection schedule across the mission timeline:
    - Wind gust disturbance: sudden aerodynamic external wrench
    - Thermal IMU drift: linear bias ramp on gyroscope/accelerometer
    - GPS multipath/spoofing: high-amplitude bias offset + noise spike
    - LiDAR smoke blackout: complete loss / zeroing / divergence of ranges
    """
    # Wind disturbance
    wind_start_time: float = 4.0
    wind_end_time: float = 9.0
    wind_force: np.ndarray = field(default_factory=lambda: np.array([4.5, -3.0, 1.5]))  # External force (N)
    wind_torque: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.08, -0.04])) # External torque (N*m)

    # Thermal IMU ramp drift
    imu_drift_start_time: float = 2.0
    imu_accel_drift_rate: np.ndarray = field(default_factory=lambda: np.array([0.15, -0.10, 0.20]))   # (m/s^2)/s
    imu_gyro_drift_rate: np.ndarray = field(default_factory=lambda: np.array([0.02, 0.03, -0.015]))    # (rad/s)/s

    # GPS Spoofing / Multipath spike
    gps_fault_start_time: float = 10.0
    gps_fault_end_time: float = 15.0
    gps_pos_bias: np.ndarray = field(default_factory=lambda: np.array([5.0, -8.0, 3.5]))  # Position offset (m)
    gps_pos_noise_multiplier: float = 10.0

    # LiDAR smoke / sensor blackout
    lidar_fault_start_time: float = 10.0
    lidar_fault_end_time: float = 15.0
    lidar_dropout_prob: float = 1.0  # 1.0 = total blackout (returns large / invalid readings)


@dataclass
class FilterConfig:
    """Dynamic precision blending and critically damped filter parameters."""
    omega_n: float = 15.0         # Natural frequency (rad/s)
    zeta: float = 1.0             # Critically damped (zeta = 1.0, zero overshoot)
    tau_thresh: float = 2.5       # Instantaneous trust residual threshold
    sigmoid_slope: float = 2.0    # Slope factor k for raw trust mapping
    trust_epsilon: float = 0.01   # Boundary clipping [eps, 1 - eps]


@dataclass
class GlobalConfig:
    """Consolidated system configuration."""
    sim: SimConfig = field(default_factory=SimConfig)
    uav: UAVConfig = field(default_factory=UAVConfig)
    rates: SensorRates = field(default_factory=SensorRates)
    noise: SensorNoiseConfig = field(default_factory=SensorNoiseConfig)
    faults: FaultProfileConfig = field(default_factory=FaultProfileConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)


# Default singleton instance
CONFIG = GlobalConfig()
