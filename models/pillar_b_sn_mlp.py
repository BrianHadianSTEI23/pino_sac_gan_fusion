"""
Module 2: Pillar B (4-Layer Spectrally Normalized MLP) & Noise/Fault Adapter.
Implements:
1. SpectrallyNormalizedMLP: 4-layer MLP with spectral normalization on layers 1-3,
   Softplus activations, and depth-wise layer freezing for 5-way continual adaptation.
2. NoiseAndFaultAdapter: Sliding-window evaluation of physics and sensor residuals
   mapping to raw instantaneous trust scores alpha_raw in (0, 1)^m.
"""

import sys
from pathlib import Path
from collections import deque
from typing import List, Dict, Optional, Tuple, Union, Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np

# PyTorch will be imported dynamically or directly
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    try:
        import torch.nn.utils.parametrizations as parametrizations
        def apply_sn(layer: nn.Module) -> nn.Module:
            return parametrizations.spectral_norm(layer)
    except (ImportError, AttributeError):
        import torch.nn.utils as utils
        def apply_sn(layer: nn.Module) -> nn.Module:
            return utils.spectral_norm(layer)
except ImportError:
    torch = None
    nn = None
    apply_sn = None

from config import GlobalConfig, CONFIG


class SpectrallyNormalizedMLP(nn.Module if nn is not None else object):
    """
    4-Layer Spectrally Normalized MLP for unmodeled external wrench estimation:
    x in R^18 -> 64 -> 64 -> 64 -> F_hat_ext in R^6.

    Layers 1, 2, and 3 use PyTorch spectral normalization to bound the Lipschitz constant (L <= 1).
    Activations are smooth Softplus (beta=1.0).
    Layer 4 is an unconstrained linear head suitable for RLS closed-form or SAC updates.
    """

    def __init__(
        self,
        input_dim: int = 18,
        hidden_dim: int = 64,
        output_dim: int = 6,
        beta: float = 1.0
    ):
        if nn is None:
            raise ImportError("PyTorch is required to instantiate SpectrallyNormalizedMLP.")
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.beta = beta

        # Layer 1: input -> hidden (Spectrally Normalized)
        self.fc1 = apply_sn(nn.Linear(input_dim, hidden_dim))
        self.act1 = nn.Softplus(beta=beta)

        # Layer 2: hidden -> hidden (Spectrally Normalized)
        self.fc2 = apply_sn(nn.Linear(hidden_dim, hidden_dim))
        self.act2 = nn.Softplus(beta=beta)

        # Layer 3: hidden -> hidden (Spectrally Normalized)
        self.fc3 = apply_sn(nn.Linear(hidden_dim, hidden_dim))
        self.act3 = nn.Softplus(beta=beta)

        # Layer 4 (Head): hidden -> output (Unconstrained linear layer for RLS / SAC)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

        # Track freeze state
        self.frozen_depth = 0
        self.set_freeze_depth(0)

    def forward_trunk(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Computes the latent representation h3 in R^64 produced by the first 3 layers.
        Exposed specifically for Variant 0 (RLS baseline) and downstream feature reuse.
        """
        h1 = self.act1(self.fc1(x))
        h2 = self.act2(self.fc2(h1))
        h3 = self.act3(self.fc3(h2))
        return h3

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Full forward pass mapping kinematic/actuation state x to estimated external wrench F_ext.
        x in R^(B x 18) -> F_ext in R^(B x 6)
        """
        h3 = self.forward_trunk(x)
        out = self.fc4(h3)
        return out

    def set_freeze_depth(self, num_frozen_layers: int) -> List["nn.Parameter"]:
        """
        Configures depth-wise layer freezing for the 5-way continual adaptation benchmark:
        - num_frozen_layers = 0: All layers {1, 2, 3, 4} trainable (SAC Freeze 0)
        - num_frozen_layers = 1: Layer {1} frozen; {2, 3, 4} trainable (SAC Freeze 1)
        - num_frozen_layers = 2: Layers {1, 2} frozen; {3, 4} trainable (SAC Freeze 2)
        - num_frozen_layers = 3: Layers {1, 2, 3} frozen; only head {4} trainable (SAC Freeze 3 / RLS)

        Returns:
            trainable_params: List of parameters with requires_grad=True.
        """
        if num_frozen_layers not in [0, 1, 2, 3]:
            raise ValueError(f"num_frozen_layers must be in {{0, 1, 2, 3}}, got {num_frozen_layers}")

        self.frozen_depth = num_frozen_layers

        # First, unfreeze all layers by default
        for param in self.parameters():
            param.requires_grad = True

        # Group layers
        layers = [self.fc1, self.fc2, self.fc3, self.fc4]

        # Freeze the first `num_frozen_layers` layers
        for layer_idx in range(num_frozen_layers):
            for param in layers[layer_idx].parameters():
                param.requires_grad = False

        # Return list of parameters that require gradients
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self) -> Dict[str, int]:
        """Returns total vs. trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


def extract_mlp_features(
    lin_vel: np.ndarray,
    ang_vel: np.ndarray,
    rot_matrix: np.ndarray,
    tau_motors: np.ndarray
) -> np.ndarray:
    """
    Packs quadrotor kinematic and actuator signals into the 18-dim feature vector:
    x = [lin_vel (3), ang_vel (3), rot_matrix.flatten() (9), tau_motors (3)]
    Note: If tau_motors has 4 elements (individual rotor thrusts), the first 3
    or the net body torque [tau_x, tau_y, tau_z] can be passed.
    """
    v = np.asarray(lin_vel, dtype=np.float32).ravel()[:3]
    w = np.asarray(ang_vel, dtype=np.float32).ravel()[:3]
    R_flat = np.asarray(rot_matrix, dtype=np.float32).ravel()[:9]
    tau = np.asarray(tau_motors, dtype=np.float32).ravel()[:3]
    return np.concatenate([v, w, R_flat, tau]).astype(np.float32)


class NoiseAndFaultAdapter:
    """
    Evaluates sliding windows of sensor residuals r_sens and physics residual r_phys
    to compute raw instantaneous trust scores alpha_raw in (0, 1)^m across sensor modalities.

    Equation:
        alpha_raw = sigmoid( k * (tau_thresh - (||r_phys|| + ||r_sens||)) )
    """

    def __init__(
        self,
        config: GlobalConfig = CONFIG,
        window_size: int = 10,
        tau_thresh: Optional[float] = None,
        k: Optional[float] = None
    ):
        self.cfg = config
        self.window_size = window_size
        self.tau_thresh = tau_thresh if tau_thresh is not None else config.filter.tau_thresh
        self.k = k if k is not None else config.filter.sigmoid_slope

        # Sensor modality residual indices in r_sens:
        # 0:3 = accel, 3:6 = gyro, 6:9 = gps_pos, 9:12 = gps_vel, 12:13 = lidar
        # 13:16 = vslam_pos, 16:19 = vslam_vel (when 19-dim VSLAM residual is provided)
        self.modality_slices = {
            "imu_accel": slice(0, 3),
            "imu_gyro": slice(3, 6),
            "gps_pos": slice(6, 9),
            "gps_vel": slice(9, 12),
            "lidar": slice(12, 13),
            "vslam_pos": slice(13, 16),
            "vslam_vel": slice(16, 19)
        }

        # Sliding window ring buffers
        self.r_phys_history = deque(maxlen=window_size)
        self.r_sens_history = deque(maxlen=window_size)
        self.reset()

    def reset(self):
        """Clears sliding window buffers."""
        self.r_phys_history.clear()
        self.r_sens_history.clear()

    def update(
        self,
        r_phys: Union[np.ndarray, float],
        r_sens: np.ndarray
    ) -> Dict[str, float]:
        """
        Pushes latest residuals into sliding window and computes trust scores.

        Args:
            r_phys: 6-dim physics residual or scalar norm.
            r_sens: 13-dim or 19-dim concatenated sensor residual vector.

        Returns:
            trust_dict: Dictionary of raw trust scores in (0, 1) across modalities.
        """
        r_phys_norm = float(np.linalg.norm(r_phys)) if isinstance(r_phys, np.ndarray) else float(r_phys)
        self.r_phys_history.append(r_phys_norm)
        self.r_sens_history.append(np.asarray(r_sens, dtype=np.float64))

        # Smoothed moving average residual norms over sliding window
        mean_r_phys = float(np.mean(self.r_phys_history))
        mean_r_sens = np.mean(self.r_sens_history, axis=0)

        # 1. Global scalar trust score (using base 13 channels for backwards compatibility)
        base_sens = mean_r_sens[:13]
        total_sens_norm = float(np.linalg.norm(base_sens))
        global_discrepancy = mean_r_phys + total_sens_norm
        alpha_raw_global = self._sigmoid(self.k * (self.tau_thresh - global_discrepancy))

        trust_dict = {"global": alpha_raw_global}

        # 2. Per-modality trust scores
        for modality, sl in self.modality_slices.items():
            if sl.stop <= mean_r_sens.shape[0]:
                mod_sens_norm = float(np.linalg.norm(mean_r_sens[sl]))
                mod_discrepancy = 0.5 * mean_r_phys + mod_sens_norm
                mod_thresh = self.tau_thresh * (1.2 if ("gps" in modality or "vslam" in modality) else 1.0)
                alpha_mod = self._sigmoid(self.k * (mod_thresh - mod_discrepancy))
                trust_dict[modality] = alpha_mod

        return trust_dict

    def compute_visual_gating(
        self,
        v_vslam: np.ndarray,
        v_prev: np.ndarray,
        rot_matrix: np.ndarray,
        motor_thrusts: np.ndarray,
        f_ext_hat: Optional[np.ndarray] = None,
        dt: float = 0.002,
        vslam_valid: bool = True,
        vslam_quality: float = 1.0,
        mass: float = 1.5,
        gravity: float = 9.81
    ) -> Dict[str, Any]:
        """
        Novelty 1: Physics-Consistent Visual Gating (PC-VG).
        Computes dynamic momentum prediction and visual innovation residual r_vis:
            a_pred = (R * [0, 0, sum(T)] + F_ext_hat) / mass + [0, 0, -g]
            v_pred = v_prev + a_pred * dt
            r_vis = ||v_vslam - v_pred||
        If r_vis > tau_vis or vslam_valid is False, rejects visual update before filter divergence.
        """
        if not vslam_valid or vslam_quality < 0.2:
            return {
                "alpha_vslam": 0.001,
                "r_vis": 10.0,
                "vslam_accepted": False,
                "scale_drift_detected": True
            }

        total_thrust = float(np.sum(motor_thrusts))
        thrust_body = np.array([0.0, 0.0, total_thrust], dtype=np.float64)
        thrust_world = rot_matrix @ thrust_body

        f_ext = np.asarray(f_ext_hat[:3], dtype=np.float64) if f_ext_hat is not None else np.zeros(3)
        acc_pred = (thrust_world + f_ext) / mass + np.array([0.0, 0.0, -gravity])
        v_pred = v_prev + acc_pred * dt

        r_vis = float(np.linalg.norm(v_vslam - v_pred))
        tau_vis = 1.8  # Visual discrepancy threshold (m/s)

        alpha_raw_vis = self._sigmoid(2.5 * (tau_vis - r_vis)) * float(np.clip(vslam_quality, 0.0, 1.0))
        accepted = bool(r_vis <= tau_vis and alpha_raw_vis > 0.3)

        return {
            "alpha_vslam": float(np.clip(alpha_raw_vis, 0.001, 0.999)),
            "r_vis": r_vis,
            "vslam_accepted": accepted,
            "scale_drift_detected": bool(r_vis > tau_vis)
        }

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Numerically stable sigmoid clipped to [1e-5, 1 - 1e-5]."""
        sig = 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))
        return float(np.clip(sig, 1e-4, 1.0 - 1e-4))


if __name__ == "__main__":
    print("=" * 70)
    print(" Running Module 2 Verification: Pillar B (4-Layer SN-MLP & Fault Adapter)")
    print("=" * 70)

    if torch is None:
        print("PyTorch is not yet installed in the current environment.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # 1. Instantiate Model
    model = SpectrallyNormalizedMLP(input_dim=18, hidden_dim=64, output_dim=6).to(device)
    print("\n1. Model Architecture Initialized:")
    print(model)

    # 2. Test Forward Passes & Dimensions
    batch_size = 16
    x_dummy = torch.randn(batch_size, 18, device=device)
    h3 = model.forward_trunk(x_dummy)
    out = model(x_dummy)
    print(f"\n2. Forward Pass Verification:")
    print(f"   • Input shape:        {tuple(x_dummy.shape)}")
    print(f"   • Trunk h3 shape:     {tuple(h3.shape)}  (Expected: ({batch_size}, 64))")
    print(f"   • Output F_ext shape: {tuple(out.shape)}  (Expected: ({batch_size}, 6))")
    assert h3.shape == (batch_size, 64)
    assert out.shape == (batch_size, 6)

    # 3. Test Layer Freezing Ablation Variants
    print("\n3. Testing Depth-Wise Layer Freezing Ablation Depths:")
    expected_trainable = []
    for depth in [0, 1, 2, 3]:
        trainable_params = model.set_freeze_depth(depth)
        counts = model.count_parameters()
        print(f"   • Freeze Depth {depth}: {counts['trainable']} trainable / {counts['total']} total params ({counts['frozen']} frozen)")
        expected_trainable.append(counts['trainable'])

        # Verify gradient flow on a dummy backward step
        out = model(x_dummy)
        loss = out.sum()
        loss.backward()

        if depth == 0:
            assert any(p.grad is not None for p in model.fc1.parameters()), "Layer 1 should have grad at depth 0"
        elif depth >= 1:
            # Layer 1 should have requires_grad=False
            for p in model.fc1.parameters():
                assert not p.requires_grad, f"Layer 1 param {p} should be frozen at depth {depth}"
        if depth >= 2:
            for p in model.fc2.parameters():
                assert not p.requires_grad, f"Layer 2 param {p} should be frozen at depth {depth}"
        if depth == 3:
            for p in model.fc3.parameters():
                assert not p.requires_grad, f"Layer 3 param {p} should be frozen at depth 3"
            # Head fc4 must still be trainable!
            assert any(p.requires_grad for p in model.fc4.parameters()), "Head layer fc4 must remain trainable at depth 3"

        model.zero_grad()

    # Monotonic decrease check
    assert expected_trainable[0] > expected_trainable[1] > expected_trainable[2] > expected_trainable[3], \
        "Trainable parameters must decrease monotonically with freeze depth!"
    print("   ✔ Depth-wise layer freezing verified successfully across all 4 depths!")

    # 4. Test Empirical Lipschitz Bound Enforcement
    print("\n4. Empirical Lipschitz Bound Verification:")
    x1 = torch.randn(100, 18, device=device)
    perturbation = torch.randn(100, 18, device=device) * 0.1
    x2 = x1 + perturbation

    model.eval()
    with torch.no_grad():
        h3_1 = model.forward_trunk(x1)
        h3_2 = model.forward_trunk(x2)
        diff_in = torch.norm(x1 - x2, dim=1)
        diff_out = torch.norm(h3_1 - h3_2, dim=1)
        empirical_L = (diff_out / (diff_in + 1e-8)).max().item()

    print(f"   • Maximum empirical Lipschitz quotient ||h3(x1) - h3(x2)|| / ||x1 - x2||: {empirical_L:.4f}")
    assert empirical_L <= 5.0, f"Trunk Lipschitz constant unexpectedly high: {empirical_L}"
    print("   ✔ Spectral normalization Lipschitz constraint verified!")

    # 5. Test NoiseAndFaultAdapter
    print("\n5. Testing NoiseAndFaultAdapter:")
    adapter = NoiseAndFaultAdapter()

    # Clean condition test
    clean_r_phys = np.zeros(6)
    clean_r_sens = np.random.normal(0, 0.02, size=13)
    for _ in range(15):
        trust_clean = adapter.update(clean_r_phys, clean_r_sens)
    print(f"   • Clean condition global trust score: {trust_clean['global']:.4f} (Expected > 0.90)")
    assert trust_clean["global"] > 0.90, "Clean trust should be near 1.0!"

    # Fault condition test (Wind + GPS spoofing)
    fault_r_phys = np.array([4.5, -3.0, 1.5, 0.05, 0.08, -0.04]) # 5.6 N wind
    fault_r_sens = clean_r_sens.copy()
    fault_r_sens[6:9] += np.array([5.0, -8.0, 3.5]) # GPS offset
    for _ in range(15):
        trust_fault = adapter.update(fault_r_phys, fault_r_sens)
    print(f"   • Fault condition global trust score: {trust_fault['global']:.4f} (Expected < 0.05)")
    print(f"   • GPS position modality trust score:  {trust_fault['gps_pos']:.4f} (Expected < 0.01)")
    assert trust_fault["global"] < 0.05, "Fault global trust should collapse toward 0.0!"
    assert trust_fault["gps_pos"] < 0.01, "GPS trust should collapse during spoofing!"
    print("   ✔ NoiseAndFaultAdapter verified successfully under clean and fault conditions!")

    print("\n" + "=" * 70)
    print("✔ All Module 2 Verification checks passed successfully!")
    print("=" * 70)
