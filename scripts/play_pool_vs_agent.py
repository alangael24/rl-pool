#!/usr/bin/env python3
import argparse
import glob
import os
import sys

import numpy as np
import torch

import pufferlib
import pufferlib.pytorch
from pufferlib.pufferl import load_config, load_env, load_policy


CURRENT_PLAYER_OBS_IDX = 80
NUM_BALLS = 16
BALL_STRIDE = 5
VX_OFFSET = 2
VY_OFFSET = 3
STATIONARY_EPS = 1e-6


def resolve_model_path(path):
    if path != "latest":
        return path

    candidates = glob.glob("experiments/puffer_pool_*/model_*.pt")
    if not candidates:
        raise FileNotFoundError("No checkpoints found under experiments/puffer_pool_*/model_*.pt")
    return max(candidates, key=os.path.getctime)


def is_human_turn(obs, human_player):
    return float(obs[human_player][CURRENT_PLAYER_OBS_IDX]) > 0.5


def is_stationary(obs_for_player):
    # Ball features are [x, y, vx, vy, alive] repeated 16 times.
    for i in range(NUM_BALLS):
        base = i * BALL_STRIDE
        vx = float(obs_for_player[base + VX_OFFSET])
        vy = float(obs_for_player[base + VY_OFFSET])
        if abs(vx) > STATIONARY_EPS or abs(vy) > STATIONARY_EPS:
            return False
    return True


def read_shot(last_dir, last_power):
    prompt = (
        f"Tu tiro dir[0-15] power[0-4] "
        f"(actual {last_dir} {last_power}, Enter=igual, q=salir): "
    )
    raw = input(prompt).strip().lower()
    if raw in ("q", "quit", "exit"):
        return None
    if raw == "":
        return int(last_dir), int(last_power)

    parts = raw.split()
    if len(parts) != 2:
        print("Formato invalido. Usa: <dir> <power> (ejemplo: 12 4)")
        return int(last_dir), int(last_power)

    try:
        direction = int(parts[0]) % 16
        power = int(parts[1]) % 5
    except ValueError:
        print("Entrada invalida. Deben ser enteros.")
        return int(last_dir), int(last_power)

    return direction, power


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="latest",
        help="Path to checkpoint .pt or 'latest'")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--human-player", type=int, default=0, choices=[0, 1])
    parser.add_argument("--max-physics-steps", type=int, default=64)
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    print(f"Loading model: {model_path}")

    env_name = "puffer_pool"
    cfg = load_config(env_name)
    cfg["train"]["device"] = args.device
    cfg["load_model_path"] = model_path
    cfg["env"]["num_envs"] = 1
    cfg["env"]["render_mode"] = "human"
    cfg["env"]["max_physics_steps"] = args.max_physics_steps
    cfg["vec"] = dict(backend="Serial", num_envs=1, seed=args.seed)

    vecenv = load_env(env_name, cfg)
    policy = load_policy(cfg, vecenv, env_name)
    ob, _ = vecenv.reset(seed=args.seed)
    driver = vecenv.driver_env

    if vecenv.num_agents != 2:
        raise RuntimeError(f"Expected 2 agents for pool, got {vecenv.num_agents}")

    state = {}
    if cfg["train"]["use_rnn"]:
        state = dict(
            lstm_h=torch.zeros(vecenv.num_agents, policy.hidden_size, device=args.device),
            lstm_c=torch.zeros(vecenv.num_agents, policy.hidden_size, device=args.device),
        )

    last_dir, last_power = 0, 4
    prompted_this_turn = False

    try:
        while True:
            driver.render()

            with torch.no_grad():
                ob_t = torch.as_tensor(ob).to(args.device)
                logits, _ = policy.forward_eval(ob_t, state)
                agent_action, _, _ = pufferlib.pytorch.sample_logits(logits)
                action = agent_action.cpu().numpy().reshape(vecenv.action_space.shape)

            action = action.astype(np.int32, copy=False)

            human_turn = is_human_turn(ob, args.human_player)
            stationary = is_stationary(ob[args.human_player])

            if human_turn and stationary and not prompted_this_turn:
                shot = read_shot(last_dir, last_power)
                if shot is None:
                    print("Saliendo.")
                    break
                last_dir, last_power = shot
                prompted_this_turn = True

            if not (human_turn and stationary):
                prompted_this_turn = False

            action[args.human_player, 0] = last_dir
            action[args.human_player, 1] = last_power
            ob, _, _, _, _ = vecenv.step(action)
    finally:
        vecenv.close()


if __name__ == "__main__":
    sys.exit(main())
