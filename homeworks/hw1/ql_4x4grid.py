import argparse
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict

import numpy as np


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy


NET_FILE = "sumo_rl/nets/4x4-Lucas/4x4.net.xml"
ROUTE_FILE = "sumo_rl/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml"


@dataclass
class LinearDecaySchedule:
    """Simple linear epsilon schedule."""

    initial: float
    minimum: float
    decay_steps: int
    step_count: int = field(default=0, init=False)

    def step(self) -> float:
        if self.decay_steps <= 0:
            return self.minimum
        ratio = min(1.0, self.step_count / self.decay_steps)
        value = self.initial + (self.minimum - self.initial) * ratio
        self.step_count += 1
        return value


class TabularQLearningAgent:
    """Custom Q-learning implementation with optimistic starts and linear epsilon decay."""

    def __init__(
        self,
        starting_state,
        n_actions: int,
        alpha: float,
        gamma: float,
        epsilon_schedule: LinearDecaySchedule,
        optimistic_init: float = 0.0,
    ) -> None:
        self.state = starting_state
        self.num_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_schedule = epsilon_schedule
        self.q_table = defaultdict(lambda: np.full(self.num_actions, optimistic_init, dtype=np.float32))
        self.last_action = None

    def set_state(self, encoded_state) -> None:
        self.state = encoded_state
        self.last_action = None

    def act(self) -> int:
        epsilon = self.epsilon_schedule.step()
        if np.random.random() < epsilon:
            action = np.random.randint(self.num_actions)
        else:
            q_values = self.q_table[self.state]
            max_value = np.max(q_values)
            greedy_actions = np.where(q_values == max_value)[0]
            action = int(np.random.choice(greedy_actions))

        self.last_action = action
        return action

    def learn(self, next_state, reward: float, done: bool) -> None:
        if self.last_action is None:
            return

        current_q = self.q_table[self.state][self.last_action]
        next_q_values = self.q_table[next_state]
        target = reward if done else reward + self.gamma * np.max(next_q_values)
        self.q_table[self.state][self.last_action] = current_q + self.alpha * (target - current_q)
        self.state = next_state


def parse_args():
    parser = argparse.ArgumentParser(description="Q-learning experiments on the 4x4 grid network.")
    parser.add_argument("--runs", type=int, default=30, help="Number of random seeds to evaluate.")
    parser.add_argument("--episodes", type=int, default=4, help="Number of episodes per run.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate for both agents.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for both agents.")
    parser.add_argument(
        "--agent",
        choices=["baseline", "custom", "both"],
        default="both",
        help="Choose which implementation to run.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/4x4", help="Directory for CSV logs.")
    parser.add_argument("--use-gui", action="store_true", help="Enable SUMO GUI for debugging.")
    parser.add_argument("--num-seconds", type=int, default=80000, help="Simulation horizon.")
    parser.add_argument("--delta-time", type=int, default=5, help="Action step length.")
    parser.add_argument("--min-green", type=int, default=5, help="Minimum green phase duration.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--epsilon-start", type=float, default=0.2, help="Initial epsilon for the custom agent.")
    parser.add_argument("--epsilon-final", type=float, default=0.01, help="Final epsilon for the custom agent.")
    parser.add_argument("--epsilon-decay", type=int, default=15000, help="Number of steps for epsilon decay.")
    parser.add_argument("--optimistic-q", type=float, default=0.0, help="Initial Q-value for unseen states.")
    parser.add_argument("--baseline-eps", type=float, default=0.05, help="Initial epsilon for baseline agent.")
    parser.add_argument("--baseline-eps-min", type=float, default=0.005, help="Minimum epsilon for baseline agent.")
    parser.add_argument("--baseline-decay", type=float, default=1.0, help="Decay for baseline epsilon-greedy.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_env(args) -> SumoEnvironment:
    return SumoEnvironment(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        use_gui=args.use_gui,
        num_seconds=args.num_seconds,
        min_green=args.min_green,
        delta_time=args.delta_time,
    )


def run_baseline(args) -> None:
    print("Running reference Q-learning implementation...")
    env = build_env(args)
    try:
        for run in range(1, args.runs + 1):
            seed_everything(args.seed + run)
            initial_states = env.reset()
            ql_agents = {
                ts: QLAgent(
                    starting_state=env.encode(initial_states[ts], ts),
                    state_space=env.observation_space,
                    action_space=env.action_space,
                    alpha=args.alpha,
                    gamma=args.gamma,
                    exploration_strategy=EpsilonGreedy(
                        initial_epsilon=args.baseline_eps,
                        min_epsilon=args.baseline_eps_min,
                        decay=args.baseline_decay,
                    ),
                )
                for ts in env.ts_ids
            }

            for episode in range(1, args.episodes + 1):
                if episode != 1:
                    initial_states = env.reset()
                    for ts in initial_states.keys():
                        ql_agents[ts].state = env.encode(initial_states[ts], ts)

                done = {"__all__": False}
                while not done["__all__"]:
                    actions = {ts: agent.act() for ts, agent in ql_agents.items()}
                    s, r, done, _ = env.step(action=actions)
                    for agent_id in s.keys():
                        ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])

                env.save_csv(os.path.join(args.output_dir, "baseline", f"ql-4x4grid_run{run}"), episode)
    finally:
        env.close()


def run_custom(args) -> None:
    print("Running custom Q-learning implementation...")
    env = build_env(args)
    try:
        for run in range(1, args.runs + 1):
            seed_everything(args.seed + 10_000 + run)
            initial_states = env.reset()
            custom_agents: Dict[str, TabularQLearningAgent] = {}
            for ts in env.ts_ids:
                epsilon_schedule = LinearDecaySchedule(
                    initial=args.epsilon_start,
                    minimum=args.epsilon_final,
                    decay_steps=args.epsilon_decay,
                )
                custom_agents[ts] = TabularQLearningAgent(
                    starting_state=env.encode(initial_states[ts], ts),
                    n_actions=env.action_spaces(ts).n,
                    alpha=args.alpha,
                    gamma=args.gamma,
                    epsilon_schedule=epsilon_schedule,
                    optimistic_init=args.optimistic_q,
                )

            for episode in range(1, args.episodes + 1):
                if episode != 1:
                    initial_states = env.reset()
                    for ts in initial_states.keys():
                        custom_agents[ts].set_state(env.encode(initial_states[ts], ts))

                done = {"__all__": False}
                while not done["__all__"]:
                    actions = {ts: agent.act() for ts, agent in custom_agents.items()}
                    next_states, rewards, done, _ = env.step(action=actions)

                    for agent_id, next_obs in next_states.items():
                        encoded_state = env.encode(next_obs, agent_id)
                        custom_agents[agent_id].learn(
                            next_state=encoded_state,
                            reward=rewards[agent_id],
                            done=done["__all__"],
                        )

                env.save_csv(os.path.join(args.output_dir, "custom", f"ql-4x4grid_run{run}"), episode)
    finally:
        env.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.agent in {"baseline", "both"}:
        run_baseline(args)

    if args.agent in {"custom", "both"}:
        run_custom(args)


if __name__ == "__main__":
    main()
