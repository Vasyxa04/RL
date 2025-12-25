import argparse
import os
import random
import sys
from collections import deque, namedtuple
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment


BIG_NET = "sumo_rl/nets/big-intersection/big-intersection.net.xml"
BIG_ROUTE = "sumo_rl/nets/big-intersection/routes.rou.xml"
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten_observation(obs) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float32)
    return arr.reshape(-1)


class LinearSchedule:
    def __init__(self, start: float, end: float, duration: int) -> None:
        self.start = start
        self.end = end
        self.duration = max(1, duration)

    def value(self, step: int) -> float:
        progress = min(1.0, step / self.duration)
        return self.start + progress * (self.end - self.start)


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, epsilon: float = 1e-3) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = epsilon
        self.buffer: List[Transition] = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done) -> None:
        transition = Transition(flatten_observation(state), action, reward, flatten_observation(next_state), done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float) -> Tuple[Transition, np.ndarray, np.ndarray]:
        assert len(self.buffer) >= batch_size, "Not enough elements in the replay buffer"
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: len(self.buffer)]

        scaled_prios = prios ** self.alpha
        probs = scaled_prios / scaled_prios.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        batch = Transition(*zip(*samples))
        return batch, indices, weights.astype(np.float32)

    def update_priorities(self, indices: Sequence[int], priorities: Sequence[float]) -> None:
        for idx, priority in zip(indices, priorities):
            adjusted = max(float(priority), self.epsilon)
            self.priorities[idx] = adjusted
            self.max_priority = max(self.max_priority, adjusted)

    def __len__(self) -> int:
        return len(self.buffer)


class DuelingDQN(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_dim, size))
            layers.append(nn.ReLU())
            prev_dim = size
        self.feature_extractor = nn.Sequential(*layers)

        advantage_layers = [
            nn.Linear(prev_dim, prev_dim),
            nn.ReLU(),
            nn.Linear(prev_dim, action_dim),
        ]
        value_layers = [
            nn.Linear(prev_dim, prev_dim),
            nn.ReLU(),
            nn.Linear(prev_dim, 1),
        ]
        self.advantage_stream = nn.Sequential(*advantage_layers)
        self.value_stream = nn.Sequential(*value_layers)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_extractor:
            x = self.feature_extractor(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


def soft_update(online_net: nn.Module, target_net: nn.Module, tau: float) -> None:
    for target_param, param in zip(target_net.parameters(), online_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def select_action(q_network: nn.Module, observation: np.ndarray, epsilon: float, action_space, device) -> int:
    if random.random() < epsilon:
        return int(action_space.sample())
    obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = q_network(obs_tensor)
    return int(q_values.argmax(dim=1).item())


def make_env(args, out_csv_name=None):
    return SumoEnvironment(
        net_file=BIG_NET,
        route_file=BIG_ROUTE,
        single_agent=True,
        out_csv_name=out_csv_name,
        use_gui=args.use_gui,
        num_seconds=args.num_seconds,
        yellow_time=args.yellow_time,
        min_green=args.min_green,
        max_green=args.max_green,
        delta_time=args.delta_time,
    )


def run_baseline(args):
    try:
        from stable_baselines3 import DQN as SB3DQN
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "stable-baselines3 must be installed to run the reference implementation "
            "(pip install stable-baselines3)"
        ) from exc

    print("Training SB3 DQN baseline...")
    env = make_env(args, out_csv_name=os.path.join(args.results_dir, f"{args.csv_prefix}_sb3"))
    model = SB3DQN(
        env=env,
        policy="MlpPolicy",
        learning_rate=args.learning_rate,
        learning_starts=args.learning_starts,
        buffer_size=args.buffer_size,
        train_freq=args.train_frequency,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.sb3_exploration_fraction,
        exploration_final_eps=args.sb3_final_eps,
        gamma=args.gamma,
        verbose=1,
    )
    model.learn(total_timesteps=args.total_timesteps)
    env.close()


def compute_loss(batch, weights, online_net, target_net, gamma, device):
    states = torch.as_tensor(np.stack(batch.state), dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch.action, dtype=torch.int64, device=device).unsqueeze(-1)
    rewards = torch.as_tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(-1)
    next_states = torch.as_tensor(np.stack(batch.next_state), dtype=torch.float32, device=device)
    dones = torch.as_tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(-1)
    weights_tensor = torch.as_tensor(weights, dtype=torch.float32, device=device).unsqueeze(-1)

    q_values = online_net(states).gather(1, actions)

    with torch.no_grad():
        next_online_actions = online_net(next_states).argmax(dim=1, keepdim=True)
        next_target_q = target_net(next_states).gather(1, next_online_actions)
        targets = rewards + (1.0 - dones) * gamma * next_target_q

    td_errors = targets - q_values
    per_sample_loss = F.smooth_l1_loss(q_values, targets, reduction="none")
    loss = (weights_tensor * per_sample_loss).mean()
    return loss, td_errors.detach().squeeze(-1).cpu().numpy()


def run_custom(args):
    print("Training custom DQN agent (dueling + double + prioritized replay)...")
    device = resolve_device(args.device)
    seed_everything(args.seed)

    env = make_env(args)
    obs_shape = env.observation_space.shape
    obs_dim = int(np.prod(obs_shape))
    action_dim = env.action_space.n

    online_net = DuelingDQN(obs_dim, action_dim, args.hidden_sizes).to(device)
    target_net = DuelingDQN(obs_dim, action_dim, args.hidden_sizes).to(device)
    target_net.load_state_dict(online_net.state_dict())
    optimizer = optim.AdamW(online_net.parameters(), lr=args.learning_rate, eps=1e-8)

    replay_buffer = PrioritizedReplayBuffer(args.buffer_size, alpha=args.per_alpha, epsilon=args.per_epsilon)
    epsilon_schedule = LinearSchedule(args.epsilon_start, args.epsilon_final, args.epsilon_decay)
    beta_schedule = LinearSchedule(args.per_beta_start, args.per_beta_end, args.total_timesteps)

    obs, _ = env.reset()
    obs = flatten_observation(obs)
    episode_reward = 0.0
    episode_len = 0
    episode_idx = 1
    returns_window = deque(maxlen=20)

    csv_prefix = os.path.join(args.results_dir, f"{args.csv_prefix}_custom_run{args.run_id}")
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    for step in range(1, args.total_timesteps + 1):
        epsilon = epsilon_schedule.value(step)
        action = select_action(online_net, obs, epsilon, env.action_space, device)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        scaled_reward = np.clip(reward * args.reward_scale, -args.reward_clip, args.reward_clip)

        replay_buffer.add(obs, action, scaled_reward, next_obs, float(done))

        obs = flatten_observation(next_obs)
        episode_reward += reward
        episode_len += 1

        if (
            step >= args.learning_starts
            and len(replay_buffer) >= args.batch_size
            and step % args.train_frequency == 0
        ):
            batch, indices, weights = replay_buffer.sample(args.batch_size, beta_schedule.value(step))
            loss, td_errors = compute_loss(batch, weights, online_net, target_net, args.gamma, device)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online_net.parameters(), args.max_grad_norm)
            optimizer.step()

            replay_buffer.update_priorities(indices, np.abs(td_errors) + args.per_epsilon)
            soft_update(online_net, target_net, args.target_tau)

        if done:
            env.save_csv(csv_prefix, episode_idx)
            returns_window.append(episode_reward)

            if step % args.log_interval == 0 or returns_window.maxlen == len(returns_window):
                mean_return = float(np.mean(returns_window)) if returns_window else episode_reward
                print(
                    f"[custom] step={step} episode={episode_idx} len={episode_len} "
                    f"return={episode_reward:.2f} mean_last={mean_return:.2f} epsilon={epsilon:.3f} "
                    f"buffer={len(replay_buffer)}"
                )

            obs, _ = env.reset()
            obs = flatten_observation(obs)
            episode_reward = 0.0
            episode_len = 0
            episode_idx += 1

    torch.save(online_net.state_dict(), os.path.join(args.results_dir, f"dqn_custom_run{args.run_id}.pt"))
    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="DQN training on the big intersection scenario.")
    parser.add_argument("--agent", choices=["baseline", "custom", "both"], default="custom")
    parser.add_argument("--total-timesteps", type=int, default=150_000)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-starts", type=int, default=2_000)
    parser.add_argument("--train-frequency", type=int, default=4)
    parser.add_argument("--target-update-interval", type=int, default=1_000)
    parser.add_argument("--target-tau", type=float, default=0.005)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-final", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=int, default=60_000)
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta-start", type=float, default=0.4)
    parser.add_argument("--per-beta-end", type=float, default=1.0)
    parser.add_argument("--per-epsilon", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--reward-clip", type=float, default=20.0)
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[256, 256])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log-interval", type=int, default=1_000)
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--results-dir", type=str, default="outputs/big-intersection")
    parser.add_argument("--csv-prefix", type=str, default="dqn_big-intersection")
    parser.add_argument("--use-gui", action="store_true")
    parser.add_argument("--num-seconds", type=int, default=5_400)
    parser.add_argument("--yellow-time", type=int, default=4)
    parser.add_argument("--min-green", type=int, default=5)
    parser.add_argument("--max-green", type=int, default=60)
    parser.add_argument("--delta-time", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--sb3-exploration-fraction", type=float, default=0.05)
    parser.add_argument("--sb3-final-eps", type=float, default=0.01)
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    if args.agent in {"baseline", "both"}:
        run_baseline(args)

    if args.agent in {"custom", "both"}:
        run_custom(args)


if __name__ == "__main__":
    main()
