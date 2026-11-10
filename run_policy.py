#!/usr/bin/env python3
"""Load the pen-click play policy and sample joint-command chunks from one state.

Run from the root of a clone of FAR-hands/play_policy (it provides the model
code):

  python /path/to/run_policy.py --ckpt /path/to/penclick_play_policy_step39000.pt
  python /path/to/run_policy.py --ckpt ... --episode data/playpolicy_penclick/th_1_329.npz --frame 500

Without --episode the state is the training mean, which only shows that the
model loads and runs.
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.cwd()))
from play_policy.data import Normalizer  # noqa: E402
from play_policy.model import build_policy  # noqa: E402


def load_policy(ckpt_path, device='cpu'):
  """Returns (policy, state normaliser, action normaliser, checkpoint dict)."""
  ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
  a = types.SimpleNamespace(
      H=ck['horizon'], **{k: ck[k] for k in (
          'n_action_steps', 'patch_size', 'd_model', 'depth', 'n_heads',
          'mlp_ratio', 'dsed', 'schedule', 'n_timesteps', 'n_sample_steps')})
  policy = build_policy('dit', ck['state_dim'], a, action_dim=ck['action_dim'])
  policy.load_state_dict(ck['policy'])
  policy.to(device).eval()
  return (policy, Normalizer.from_state_dict(ck['norm_state']),
          Normalizer.from_state_dict(ck['norm_action']), ck)


@torch.no_grad()
def sample_chunks(policy, norm_s, norm_a, state, n=1, seed=0, device='cpu'):
  """state: (49,) = 16 joint positions (rad) + 33 touch values.

  Returns (n, 200, 16) joint commands in rad; the first 50 steps are the ones
  meant to be executed before asking again.
  """
  cond = torch.as_tensor(norm_s.normalize(state[None]), dtype=torch.float32,
                         device=device).repeat(n, 1)
  gen = torch.Generator(device=device).manual_seed(seed)
  chunk = policy(cond, generator=gen).cpu().numpy()
  return chunk * norm_a.scale + norm_a.center


def main():
  p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  p.add_argument('--ckpt', required=True)
  p.add_argument('--episode', help='a file written by build_dataset.py')
  p.add_argument('--frame', type=int, default=0)
  p.add_argument('--samples', type=int, default=4)
  args = p.parse_args()
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  policy, norm_s, norm_a, ck = load_policy(args.ckpt, device)
  print(f"loaded step {ck['step']}: state {ck['state_dim']} -> "
        f"{ck['horizon']} x {ck['action_dim']} commands, on {device}")

  recorded = None
  if args.episode:
    d = np.load(args.episode)
    state = np.concatenate([d['joint_pos'], d['touch']], 1)[args.frame]
    recorded = d['cmd_q'][args.frame:args.frame + ck['horizon']]
  else:
    state = np.asarray(norm_s.center, dtype=np.float32)

  chunks = sample_chunks(policy, norm_s, norm_a, state, args.samples, device=device)
  n_exec = ck['n_action_steps']
  print(f'sampled {chunks.shape[0]} chunks of shape {chunks.shape[1:]} (rad)')
  print(f'spread between samples over the first {n_exec} steps: '
        f'{chunks[:, :n_exec].std(0).mean():.4f} rad (mean std per joint and step)')
  if recorded is not None and len(recorded) == ck['horizon']:
    err = np.abs(chunks[:, :n_exec] - recorded[None, :n_exec]).mean()
    print(f'mean absolute difference from the recorded commands: {err:.4f} rad')
  names = ck['meta']['cmd_names']
  print('first command of sample 0:')
  for name, v in zip(names, chunks[0, 0]):
    print(f'  {name:8s} {v:+.3f}')


if __name__ == '__main__':
  main()
