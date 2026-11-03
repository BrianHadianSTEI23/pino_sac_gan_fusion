"""
Module 3: The 5-Way Online Continual Adaptation Engine (RLS vs. Annealed SAC).
Implements:
1. RLSHeadAdapter: Analytical closed-form Recursive Least Squares (lambda=0.995)
   updating the linear head W_4 of SpectrallyNormalizedMLP driven by r_phys.
2. OnlineAnnealedSACAdapter: Soft Actor-Critic with depth-wise layer freezing
   (Freeze 0, 1, 2, 3), Twin Q-Critics, lightweight ring replay buffer,
   and exponential temperature annealing (alpha_t -> 0) for jitter-free deployment.
3. 5-Way Continual Adaptation Benchmark suite comparing variants 0-4.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List, Union

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import GlobalConfig, CONFIG
from models.pillar_b_sn_mlp import SpectrallyNormalizedMLP, extract_mlp_features
from models.pin_esn import PhysicsInformedESN


# ==============================================================================
# 1. Variant 0: Convex Recursive Least Squares (RLS) Head Adapter
# ==============================================================================

class RLSHeadAdapter:
    """
    Variant 0: Analytical closed-form Recursive Least Squares updater.
    Freezes layers {1, 2, 3} of SpectrallyNormalizedMLP and updates head layer W_4 (and b_4)
    in closed form with forgetting factor lambda_rls = 0.995.

    The error signal driving adaptation is the Euler-Lagrange physics residual:
    e_t = (F_hat_ext + r_phys) - F_hat_ext = r_phys in R^6.
    """

    def __init__(
        self,
        model: SpectrallyNormalizedMLP,
        lambda_rls: float = 0.995,
        delta: float = 100.0,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.lambda_rls = lambda_rls
        self.device = device or next(model.parameters()).device

        # Freeze trunk layers {1, 2, 3}
        self.model.set_freeze_depth(3)

        self.hidden_dim = model.hidden_dim   # 64
        self.output_dim = model.output_dim   # 6
        self.param_dim = self.hidden_dim + 1 # 65 (weights + bias)

        # Parameter matrix Theta: (65, 6) where Theta[:64, :] is W4^T and Theta[64, :] is b4
        self.Theta = np.zeros((self.param_dim, self.output_dim), dtype=np.float64)

        # Initialize Theta from current model weights if available
        with torch.no_grad():
            w4 = model.fc4.weight.cpu().numpy() # (6, 64)
            b4 = model.fc4.bias.cpu().numpy()   # (6,)
            self.Theta[:self.hidden_dim, :] = w4.T
            self.Theta[self.hidden_dim, :] = b4

        # Inverse correlation covariance matrix P: (65, 65)
        self.P = np.eye(self.param_dim, dtype=np.float64) * delta

    def predict(
        self,
        x: Union[np.ndarray, torch.Tensor],
        precomputed_h3: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes forward pass: h3 in R^64 and predicted wrench F_hat in R^6.

        Args:
            x: Input feature vector in R^18 or (B, 18).
            precomputed_h3: Optional precomputed trunk features h3 (64,) to avoid redundant forward pass.

        Returns:
            f_hat: Predicted external wrench (output_dim,)
            phi: Augmented feature vector [h3, 1] (param_dim,)
        """
        if precomputed_h3 is not None:
            h3 = precomputed_h3
        else:
            if isinstance(x, np.ndarray):
                x_t = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
            else:
                x_t = x.to(self.device)
                if x_t.dim() == 1:
                    x_t = x_t.unsqueeze(0)

            with torch.no_grad():
                h3 = self.model.forward_trunk(x_t).cpu().numpy().ravel() # (64,)

        phi = np.empty(self.param_dim, dtype=np.float64)
        phi[:self.hidden_dim] = h3
        phi[self.hidden_dim] = 1.0

        f_hat = phi @ self.Theta # (6,)
        return f_hat, phi

    def update(
        self,
        x: Union[np.ndarray, torch.Tensor],
        r_phys: np.ndarray,
        precomputed_h3: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Performs one analytical closed-form RLS recursion step:
            k = P * phi / (lambda + phi^T * P * phi)
            P = (P - k * phi^T * P) / lambda
            Theta = Theta + k * e^T  (where e = r_phys)

        Args:
            x: Kinematic input feature vector in R^18.
            r_phys: Euler-Lagrange physics residual in R^6.
            precomputed_h3: Optional precomputed trunk features h3.

        Returns:
            f_hat: Predicted external wrench after update.
            step_latency_us: Execution latency in microseconds.
        """
        t0 = time.perf_counter_ns()

        f_hat, phi = self.predict(x, precomputed_h3=precomputed_h3)

        # Innovation / Error signal is directly the physics residual r_phys
        e = np.asarray(r_phys, dtype=np.float64)

        # RLS Gain vector computation: k = P * phi / (lambda + phi^T * P * phi)
        P_phi = self.P @ phi # (65,)
        denom = self.lambda_rls + float(phi @ P_phi)
        k = P_phi / denom # (65,)

        # Covariance update: P = (P - k * phi^T * P) / lambda
        self.P = (self.P - np.outer(k, P_phi)) / self.lambda_rls

        # Parameter matrix update: Theta = Theta + k * e^T
        self.Theta += np.outer(k, e)

        # Synchronize updated weights back to PyTorch model fc4
        self.model.fc4.weight.data.copy_(torch.from_numpy(self.Theta[:self.hidden_dim, :].T))
        self.model.fc4.bias.data.copy_(torch.from_numpy(self.Theta[self.hidden_dim, :]))

        f_hat_updated = phi @ self.Theta
        t1 = time.perf_counter_ns()
        step_latency_us = (t1 - t0) / 1000.0

        return f_hat_updated, step_latency_us


# ==============================================================================
# 2. Replay Buffer & Twin Critic Networks for SAC
# ==============================================================================

class RingReplayBuffer:
    """
    Lightweight, preallocated circular ring buffer for online SAC updates.
    Stored as contiguous memory buffers on CPU/GPU for microsecond insertions.
    """

    def __init__(self, state_dim: int = 18, action_dim: int = 6, capacity: int = 10000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool = False
    ):
        """Insert transition at current pointer index."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Uniform random mini-batch sampling."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[indices]),
            torch.from_numpy(self.actions[indices]),
            torch.from_numpy(self.rewards[indices]),
            torch.from_numpy(self.next_states[indices]),
            torch.from_numpy(self.dones[indices])
        )


class TwinQCritic(nn.Module):
    """
    Twin Q-Critics Q1(s, a) and Q2(s, a) evaluating state-wrench pairs.
    input: state (18) + action/wrench (6) = 24 -> 64 -> 64 -> 1.
    """

    def __init__(self, state_dim: int = 18, action_dim: int = 6, hidden_dim: int = 64):
        super().__init__()
        in_dim = state_dim + action_dim

        # Q1 Architecture
        self.q1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Softplus(beta=1.0),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(beta=1.0),
            nn.Linear(hidden_dim, 1)
        )

        # Q2 Architecture
        self.q2 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Softplus(beta=1.0),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(beta=1.0),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)


# ==============================================================================
# 3. Variants 1–4: Online Annealed Soft Actor-Critic Adapter
# ==============================================================================

class OnlineAnnealedSACAdapter:
    """
    Implements online continual Soft Actor-Critic with depth-wise layer freezing:
    - freeze_depth = 0: Freeze 0 (All layers unfrozen) -> Variant 1
    - freeze_depth = 1: Freeze 1 (Layer 1 frozen)       -> Variant 2
    - freeze_depth = 2: Freeze 2 (Layers 1-2 frozen)     -> Variant 3
    - freeze_depth = 3: Freeze 3 (Layers 1-3 frozen)     -> Variant 4

    Features exponential temperature annealing:
        alpha_t = alpha_0 * exp(-beta * step) -> 0
    guaranteeing smooth, zero-jitter transition to deterministic policy evaluation.
    """

    def __init__(
        self,
        actor_model: SpectrallyNormalizedMLP,
        freeze_depth: int = 0,
        state_dim: int = 18,
        action_dim: int = 6,
        alpha_0: float = 0.2,
        beta_decay: float = 0.0005,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        device: Optional[torch.device] = None
    ):
        self.device = device or next(actor_model.parameters()).device
        self.actor = actor_model.to(self.device)
        self.freeze_depth = freeze_depth

        # Configure exact layer freeze depth
        self.trainable_actor_params = self.actor.set_freeze_depth(freeze_depth)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Entropy temperature parameters
        self.alpha_0 = alpha_0
        self.beta_decay = beta_decay
        self.step_count = 0
        self.current_alpha = alpha_0

        # Max wrench bounds [Fx, Fy, Fz, Tx, Ty, Tz]
        self.max_force = 12.0   # Newtons
        self.max_torque = 0.5   # N*m
        self.action_scale = torch.tensor(
            [self.max_force, self.max_force, self.max_force, self.max_torque, self.max_torque, self.max_torque],
            dtype=torch.float32,
            device=self.device
        )

        # Log std head for stochastic Gaussian policy: mu comes from actor.fc4, log_std is parametrized
        self.log_std_head = nn.Linear(actor_model.hidden_dim, action_dim).to(self.device)
        self.trainable_actor_params += list(self.log_std_head.parameters())

        # Twin Q-Critics and Target Critics
        self.critic = TwinQCritic(state_dim, action_dim, hidden_dim=64).to(self.device)
        self.critic_target = TwinQCritic(state_dim, action_dim, hidden_dim=64).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers (only trainable actor weights are updated!)
        self.actor_optimizer = optim.Adam(self.trainable_actor_params, lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Replay Buffer
        self.buffer = RingReplayBuffer(state_dim, action_dim, capacity=buffer_capacity)

        # Reward weights: R = -(w1*||r_phys||^2 + w2*||r_sens||^2 + w3*||d_F_ext||^2)
        self.w1 = 1.0
        self.w2 = 0.1
        self.w3 = 0.05
        self.prev_action = np.zeros(action_dim, dtype=np.float32)

    def get_current_entropy_temp(self) -> float:
        """Computes exponential temperature decay alpha_t = alpha_0 * exp(-beta * t)."""
        self.current_alpha = float(self.alpha_0 * np.exp(-self.beta_decay * self.step_count))
        return self.current_alpha

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, float]:
        """
        Samples external wrench action from Gaussian policy or returns deterministic mean.

        Returns:
            action: 6-dim external wrench prediction in Newtons / N*m.
            alpha_t: Current entropy temperature.
        """
        self.step_count += 1
        alpha_t = self.get_current_entropy_temp()

        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            h3 = self.actor.forward_trunk(state_t)
            mu = self.actor.fc4(h3)

            # Clamp mean within normalized range [-1, 1] then scale
            mu_scaled = torch.tanh(mu) * self.action_scale

            if deterministic or alpha_t < 1e-3:
                # Deterministic mean execution for jitter-free deployment
                action = mu_scaled.cpu().numpy().squeeze(0)
            else:
                log_std = torch.clamp(self.log_std_head(h3), -5.0, 1.0)
                std = torch.exp(log_std) * np.sqrt(max(alpha_t, 1e-4))
                eps = torch.randn_like(mu)
                action_raw = torch.tanh(mu + std * eps) * self.action_scale
                action = action_raw.cpu().numpy().squeeze(0)

        return action, alpha_t

    def compute_reward(
        self,
        r_phys: np.ndarray,
        r_sens: np.ndarray,
        action: np.ndarray
    ) -> float:
        """
        Calculates composite continual adaptation reward:
        R_t = -(w1 * ||r_phys||^2 + w2 * ||r_sens||^2 + w3 * ||action - prev_action||^2)
        """
        r_phys_norm_sq = float(np.sum(np.square(r_phys)))
        r_sens_norm_sq = float(np.sum(np.square(r_sens)))
        delta_act_sq = float(np.sum(np.square(action - self.prev_action)))

        reward = -(self.w1 * r_phys_norm_sq + self.w2 * r_sens_norm_sq + self.w3 * delta_act_sq)
        self.prev_action = action.copy()
        return reward

    def update_parameters(self) -> Tuple[float, float, float]:
        """
        Executes one gradient update step on Actor and Twin Critics.
        Returns: (critic_loss, actor_loss, latency_us).
        """
        if self.buffer.size < self.batch_size:
            return 0.0, 0.0, 0.0

        t0 = time.perf_counter_ns()

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # ----------------- 1. Critic Update -----------------
        with torch.no_grad():
            next_h3 = self.actor.forward_trunk(next_states)
            next_mu = self.actor.fc4(next_h3)
            next_log_std = torch.clamp(self.log_std_head(next_h3), -5.0, 1.0)
            next_std = torch.exp(next_log_std)
            next_actions = torch.tanh(next_mu + next_std * torch.randn_like(next_mu)) * self.action_scale

            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_value = rewards + (1.0 - dones) * self.gamma * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(current_q1, target_value) + nn.functional.mse_loss(current_q2, target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ----------------- 2. Actor Update -----------------
        h3 = self.actor.forward_trunk(states)
        mu = self.actor.fc4(h3)
        log_std = torch.clamp(self.log_std_head(h3), -5.0, 1.0)
        std = torch.exp(log_std)
        pred_actions = torch.tanh(mu + std * torch.randn_like(mu)) * self.action_scale

        q1_pred = self.critic.q1_forward(states, pred_actions)
        alpha_t = self.get_current_entropy_temp()

        # Log prob approximation
        log_prob = -0.5 * torch.sum(((pred_actions - mu) / (std + 1e-6))**2, dim=-1, keepdim=True)
        actor_loss = -(q1_pred - alpha_t * log_prob).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------- 3. Target Polyak Averaging -----------------
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        t1 = time.perf_counter_ns()
        latency_us = (t1 - t0) / 1000.0

        return float(critic_loss.item()), float(actor_loss.item()), latency_us


# ==============================================================================
# 3b. Paradigm B: Physics-Informed Echo State Network (PIN-ESN) Adapter
# ==============================================================================

class PIN_ESN_Adapter:
    """
    Paradigm B Adapter: Physics-Informed Echo State Network with closed-form RLS readout.
    Wraps PhysicsInformedESN to provide a uniform predict / update continual adaptation interface.
    """

    def __init__(
        self,
        reservoir_dim: int = 128,
        lambda_rls: float = 0.995,
        device: Optional[torch.device] = None,
        seed: int = 42
    ):
        self.device = device or torch.device("cpu")
        self.model = PhysicsInformedESN(
            state_dim=18,
            phys_dim=6,
            output_dim=6,
            reservoir_dim=reservoir_dim,
            spectral_radius=0.95,
            sparsity=0.15,
            leaking_rate=0.35,
            input_scale=0.15,
            lambda_rls=lambda_rls,
            delta_p0=50.0,
            device=self.device,
            seed=seed
        )
        self.hidden_dim = reservoir_dim
        self.output_dim = 6
        self.param_dim = self.model.param_dim

    def predict(
        self,
        x: Union[np.ndarray, torch.Tensor],
        precomputed_h3: Optional[np.ndarray] = None,
        delta_p_vis: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward prediction of disturbance wrench."""
        return self.model.predict(x, delta_p_vis=delta_p_vis)

    def update(
        self,
        x: Union[np.ndarray, torch.Tensor],
        r_phys: Union[np.ndarray, torch.Tensor],
        precomputed_h3: Optional[np.ndarray] = None,
        delta_p_vis: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Tuple[np.ndarray, float]:
        """Closed-form RLS adaptation step."""
        return self.model.update(x, r_phys, delta_p_vis=delta_p_vis)

    def reset(self):
        """Resets reservoir state and covariance."""
        self.model.reset()

    def get_stats(self) -> Dict[str, Any]:
        """Returns reservoir diagnostics."""
        return self.model.get_stats()


# ==============================================================================
# 4. 5-Way Adaptation Manager / Factory (Extended with Paradigm B)
# ==============================================================================

def create_continual_adapter(
    variant_idx: Union[int, str],
    base_mlp: Optional[SpectrallyNormalizedMLP] = None,
    device: Optional[torch.device] = None
) -> Union[RLSHeadAdapter, OnlineAnnealedSACAdapter, PIN_ESN_Adapter]:
    """
    Factory creating one of the continual adaptation variants:
    - Variant 0: RLS closed-form baseline (Head W4 updated analytically)
    - Variant 1: SAC Freeze 0 (All 4 layers updated via SAC gradients)
    - Variant 2: SAC Freeze 1 (Layer 1 frozen, layers 2, 3, 4 updated)
    - Variant 3: SAC Freeze 2 (Layers 1, 2 frozen, layers 3, 4 updated)
    - Variant 4: SAC Freeze 3 (Layers 1, 2, 3 frozen, only head 4 updated)
    - Variant 5 / 'pin_esn': Paradigm B [PIN-ESN] (Recurrent reservoir + closed-form RLS)
    """
    device = device or torch.device("cpu")
    mlp = base_mlp if base_mlp is not None else SpectrallyNormalizedMLP(18, 64, 6).to(device)

    if variant_idx == 0:
        return RLSHeadAdapter(model=mlp, lambda_rls=0.995, device=device)
    elif variant_idx in [1, 2, 3, 4]:
        freeze_depth = variant_idx - 1 # 1->0, 2->1, 3->2, 4->3
        return OnlineAnnealedSACAdapter(
            actor_model=mlp,
            freeze_depth=freeze_depth,
            device=device
        )
    elif variant_idx in [5, "pin_esn", "pin-esn", "Paradigm B [PIN-ESN]"]:
        return PIN_ESN_Adapter(reservoir_dim=128, lambda_rls=0.995, device=device)
    else:
        raise ValueError(f"Unknown variant index {variant_idx}. Must be 0, 1, 2, 3, 4, or 5 (PIN-ESN).")


# ==============================================================================
# 5. Verification & Benchmarking Suite
# ==============================================================================

def benchmark_5way_continual_adaptation(
    num_steps: int = 100,
    include_pin_esn: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Benchmarks single-step execution latency (microseconds), parameter counts,
    and memory across all plasticity-stability variants and Paradigm B.
    """
    device = torch.device("cpu")
    variant_names = [
        "Variant 0 [RLS Baseline]",
        "Variant 1 [SAC Freeze 0]",
        "Variant 2 [SAC Freeze 1]",
        "Variant 3 [SAC Freeze 2]",
        "Variant 4 [SAC Freeze 3]"
    ]
    if include_pin_esn:
        variant_names.append("Paradigm B [PIN-ESN]")

    results = {}

    for idx, name in enumerate(variant_names):
        mlp = SpectrallyNormalizedMLP(18, 64, 6).to(device)
        v_key = 5 if "PIN-ESN" in name else idx
        adapter = create_continual_adapter(v_key, base_mlp=mlp, device=device)

        latencies_us = []
        dummy_state = np.random.randn(18).astype(np.float32)
        dummy_r_phys = np.array([4.5, -3.0, 1.5, 0.05, 0.08, -0.04]) # ~5.6 N wind
        dummy_r_sens = np.random.randn(13).astype(np.float32) * 0.05

        if isinstance(adapter, (RLSHeadAdapter, PIN_ESN_Adapter)):
            # Benchmark analytical update latency
            for _ in range(num_steps):
                _, lat = adapter.update(dummy_state, dummy_r_phys)
                latencies_us.append(lat)
            if isinstance(adapter, RLSHeadAdapter):
                trainable_params = adapter.Theta.size # 65 * 6 = 390
                total_params = mlp.count_parameters()["total"]
            else:
                stats = adapter.get_stats()
                trainable_params = stats["trainable_params"]
                total_params = adapter.model.W_in.numel() + adapter.model.W_res.numel() + trainable_params
        else:
            # Warm up SAC buffer
            for _ in range(adapter.batch_size + 5):
                act, _ = adapter.select_action(dummy_state)
                rew = adapter.compute_reward(dummy_r_phys, dummy_r_sens, act)
                next_st = dummy_state + 0.01 * np.random.randn(18).astype(np.float32)
                adapter.buffer.push(dummy_state, act, rew, next_st)

            # Benchmark SAC online update latency
            for _ in range(num_steps):
                act, _ = adapter.select_action(dummy_state, deterministic=True)
                rew = adapter.compute_reward(dummy_r_phys, dummy_r_sens, act)
                next_st = dummy_state + 0.01 * np.random.randn(18).astype(np.float32)
                adapter.buffer.push(dummy_state, act, rew, next_st)
                _, _, lat = adapter.update_parameters()
                latencies_us.append(lat)

            counts = mlp.count_parameters()
            trainable_params = counts["trainable"]
            total_params = counts["total"]

        # Discard first 5 steps as JIT / cache warmup
        warm_latencies = latencies_us[5:]
        mean_lat = float(np.mean(warm_latencies))
        p95_lat = float(np.percentile(warm_latencies, 95))

        results[name] = {
            "mean_latency_us": mean_lat,
            "p95_latency_us": p95_lat,
            "trainable_params": trainable_params,
            "total_params": total_params
        }

    return results


if __name__ == "__main__":
    print("=" * 75)
    print(" Running Module 3 Verification: 5-Way Online Continual Adaptation Engine")
    print("=" * 75)

    print("Benchmarking single-step execution latency (100 steps each)...")
    res = benchmark_5way_continual_adaptation(num_steps=100)

    print("\n" + "-" * 75)
    print(f"{'Variant':<28} | {'Trainable':<10} | {'Mean Latency':<14} | {'p95 Latency':<12}")
    print("-" * 75)
    for name, m in res.items():
        print(f"{name:<28} | {m['trainable_params']:<10} | {m['mean_latency_us']:>8.2f} µs     | {m['p95_latency_us']:>8.2f} µs")
    print("-" * 75)

    # 1. Latency & Budget Assertions
    # At 500 Hz, the control period is 2000 µs (2 ms).
    # RLS should comfortably execute within the sub-millisecond real-time budget (< 1000 µs).
    rls_lat = res["Variant 0 [RLS Baseline]"]["mean_latency_us"]
    print(f"\n1. RLS Execution Latency: {rls_lat:.2f} µs (< 1000 µs sub-ms budget: {rls_lat < 1000.0})")
    assert rls_lat < 1000.0, f"RLS latency {rls_lat} µs exceeded sub-millisecond budget!"

    # 2. Entropy Annealing Verification
    print("\n2. Jitter-Free Entropy Annealing Verification:")
    mlp = SpectrallyNormalizedMLP(18, 64, 6)
    sac_adapter = OnlineAnnealedSACAdapter(mlp, freeze_depth=0, alpha_0=0.2, beta_decay=0.015)
    alphas = []
    actions_stoch = []
    dummy_st = np.ones(18, dtype=np.float32)

    for step in range(500):
        act, a_t = sac_adapter.select_action(dummy_st, deterministic=False)
        alphas.append(a_t)
        actions_stoch.append(act)

    initial_jitter = float(np.mean(np.std(actions_stoch[:50], axis=0)))
    tail_jitter = float(np.mean(np.std(actions_stoch[-50:], axis=0)))

    print(f"   • Initial Temperature alpha_0: {alphas[0]:.4f}")
    print(f"   • Annealed Temperature @ step 500: {alphas[-1]:.6f} (Decayed to ~ 0.0)")
    print(f"   • Initial Action Std Dev (Steps 1-50):   {initial_jitter:.4f} N")
    print(f"   • Annealed Action Std Dev (Steps 450-500): {tail_jitter:.4f} N (Decayed by {100*(1 - tail_jitter/initial_jitter):.1f}%)")

    # Deterministic deployment test (zero exploration jitter)
    det_actions = [sac_adapter.select_action(dummy_st, deterministic=True)[0] for _ in range(50)]
    det_jitter = float(np.mean(np.std(det_actions, axis=0)))
    print(f"   • Deterministic Deployment Action Std Dev: {det_jitter:.6f} N (Zero Jitter: True)")

    assert alphas[-1] < 0.001, "Entropy temperature should decay near zero!"
    assert tail_jitter < initial_jitter * 0.2, "Exploration jitter must decay significantly as alpha_t -> 0!"
    assert det_jitter < 1e-6, "Deterministic action selection must have zero exploration jitter!"
    print("   ✔ Entropy annealing and jitter elimination verified!")

    # 3. Closed-Form RLS Disturbance Tracking Verification
    print("\n3. Closed-Form RLS Disturbance Tracking Verification:")
    mlp_rls = SpectrallyNormalizedMLP(18, 64, 6)
    rls = RLSHeadAdapter(mlp_rls, lambda_rls=0.99)
    true_wind_wrench = np.array([4.5, -3.0, 1.5, 0.0, 0.0, 0.0])

    f_pred = np.zeros(6)
    for step in range(150):
        # Physics residual discrepancy e = true_wind - f_pred
        r_phys_sim = true_wind_wrench - f_pred
        f_pred, _ = rls.update(dummy_st, r_phys_sim)

    error = np.linalg.norm(f_pred - true_wind_wrench)
    print(f"   • True Wind Wrench:      {np.round(true_wind_wrench[:3], 3)} N")
    print(f"   • RLS Estimated Wrench:  {np.round(f_pred[:3], 3)} N")
    print(f"   • Tracking Error Norm:   {error:.6f} N (Expected < 0.1 N)")
    assert error < 0.1, f"RLS failed to converge: error = {error}"
    print("   ✔ RLS convex convergence verified!")

    print("\n" + "=" * 75)
    print("✔ All Module 3 Verification checks passed successfully!")
    print("=" * 75)
