#!/usr/bin/env python3
"""Score a trained pen-click play policy on the held-out th_12 recordings.

For random held-out windows, compares the policy's predicted joint-command
chunk with the commands that were really sent, in radians, against a
do-nothing baseline that holds the current measured joint position.

Run from the play_policy repo root:

  python /path/to/eval_policy.py --ckpt runs/play_policy/<run>/ckpt_best.pt \
      --data_dir /path/to/playpolicy_penclick --out results/policy_eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play_policy.data import Normalizer, WindowDataset  # noqa: E402
from play_policy.model import build_policy  # noqa: E402
from train_policy import load_segments  # noqa: E402


def main():
  p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  p.add_argument('--ckpt', required=True)
  p.add_argument('--data_dir', required=True)
  p.add_argument('--out', type=Path, required=True)
  p.add_argument('--windows', type=int, default=2048)
  p.add_argument('--samples', type=int, default=4)
  p.add_argument('--batch', type=int, default=256)
  args = p.parse_args()
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
  a = types.SimpleNamespace(
      H=ck['horizon'], **{k: ck[k] for k in (
          'n_action_steps', 'patch_size', 'd_model', 'depth', 'n_heads',
          'mlp_ratio', 'dsed', 'schedule', 'n_timesteps', 'n_sample_steps')})
  policy = build_policy('dit', ck['state_dim'], a, action_dim=ck['action_dim'])
  policy.load_state_dict(ck['policy'])
  policy.to(device).eval()
  norm_s = Normalizer.from_state_dict(ck['norm_state'])
  norm_a = Normalizer.from_state_dict(ck['norm_action'])

  segments, meta = load_segments(args.data_dir, min_len=a.H)
  val = [s for s in segments if s['heldout']]
  ds = WindowDataset(val, a.H, norm_s, norm_a)
  rng = np.random.default_rng(0)
  scale = torch.as_tensor(norm_a.scale, device=device)

  err_pol, err_one, err_hold = [], [], []
  for i in range(0, args.windows, args.batch):
    cond, chunk = ds.sample_batch(min(args.batch, args.windows - i), rng)
    cond, chunk = cond.to(device), chunk.to(device)
    preds = torch.stack([
        policy(cond, generator=torch.Generator(device=device).manual_seed(100 * m + i))
        for m in range(args.samples)])
    # Baseline: command = current measured joint position, held for the chunk.
    joints_rad = cond[:, :16].cpu().numpy() * norm_s.scale[:16] + norm_s.center[:16]
    hold = torch.as_tensor(norm_a.normalize(joints_rad), device=device)[:, None]
    err_pol.append(((preds.mean(0) - chunk).abs() * scale).cpu().numpy())
    err_one.append(((preds[0] - chunk).abs() * scale).cpu().numpy())
    err_hold.append(((hold - chunk).abs() * scale).cpu().numpy())
  err_pol, err_one, err_hold = (np.concatenate(e) for e in (err_pol, err_one, err_hold))

  names = meta['joint_names']
  groups = {'thumb': [i for i, n in enumerate(names) if n.startswith('th_')],
            'index': [i for i, n in enumerate(names) if n.startswith('if_')],
            'middle': [i for i, n in enumerate(names) if n.startswith('mf_')],
            'ring': [i for i, n in enumerate(names) if n.startswith('rf_')]}
  n_exec = int(a.n_action_steps)

  def summary(e):
    out = {'all_joints_executed_steps': float(e[:, :n_exec].mean()),
           'all_joints_full_chunk': float(e.mean())}
    for g, idx in groups.items():
      out[f'{g}_executed_steps'] = float(e[:, :n_exec][..., idx].mean())
    return out

  result = dict(
      ckpt=str(args.ckpt), step=int(ck['step']),
      best_val_action_mse_norm=ck.get('best_val_action_mse_norm'),
      heldout_recordings=len(val), windows=int(len(err_pol)),
      horizon=int(a.H), executed_steps=n_exec, units='mean absolute error, rad',
      policy_mean_of_samples=summary(err_pol),
      policy_single_sample=summary(err_one),
      hold_current_position=summary(err_hold))
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=1))
  print(json.dumps(result, indent=1))


if __name__ == '__main__':
  main()
