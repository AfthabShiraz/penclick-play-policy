#!/usr/bin/env python3
"""Train the PlayPolicy diffusion-transformer play policy on the pen-click data.

Runs the unmodified trainer and model from FAR-hands/play_policy
(`play_policy.train`, arXiv 2610.22340) with the paper's settings. Only the
data loading is replaced:

    state  = [joint_pos (16), touch (33)]   instead of [hand joints (22), cube pose (7)]
    action = cmd_q (16)                     instead of cmd_q22 (22)
    validation = the held-out th_12 recordings, instead of a random 15% of segments

Each recording is one contiguous segment. Run from the play_policy repo root:

  python /path/to/train_policy.py --data_dir /path/to/playpolicy_penclick [trainer flags]

The trainer's own log line "state_dim=... = 22 hand + 7 cube" is its fixed
wording; the real layout is printed by this script just before it.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
from play_policy import train as pp_train  # noqa: E402


def load_segments(data_dir, min_len=1, **_):
  dirs = [data_dir] if isinstance(data_dir, (str, Path)) else list(data_dir)
  files = sorted(f for d in dirs for f in glob.glob(str(Path(d) / '*.npz')))
  if not files:
    raise FileNotFoundError(f'no .npz under {dirs}')
  segments, meta = [], {}
  for f in files:
    d = np.load(f)
    states = np.concatenate([d['joint_pos'], d['touch']], 1).astype(np.float32)
    actions = d['cmd_q'].astype(np.float32)
    if not meta:
      meta = {
          'state_layout': 'joint_pos(16) + touch(33)',
          'joint_names': d['joint_names'].tolist(),
          'touch_names': d['touch_names'].tolist(),
          'cmd_names': d['joint_names'].tolist(),
          'touch_units': 'symlog of summed pad force (sensor counts)',
          'hz': 100.0,
      }
    if len(states) >= min_len:
      segments.append({
          'states': states, 'actions': actions, 'file': Path(f).name,
          'seg': 0, 'heldout': bool(d['heldout']), 'label': str(d['label'])})
  return segments, meta


def split_segments(segments, val_frac, seed):
  del val_frac, seed
  train = [s for s in segments if not s['heldout']]
  val = [s for s in segments if s['heldout']]
  n = lambda segs: sum(len(s['states']) for s in segs)
  print(f'pen-click data: state = joint_pos(16) + touch(33), action = cmd_q(16), 100 Hz')
  print(f'  train {len(train)} recordings / {n(train)} frames; '
        f'held-out th_12 {len(val)} recordings / {n(val)} frames')
  return train, val


pp_train.load_segments = load_segments
pp_train.split_segments = split_segments

if __name__ == '__main__':
  raise SystemExit(pp_train.main())
