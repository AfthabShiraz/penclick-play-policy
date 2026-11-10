#!/usr/bin/env python3
"""Build the play-policy training set from the pen-click recordings.

One output file per recording, on the 100 Hz grid of the sparsh_skin episode:

    t          (T,)      seconds, recording clock
    joint_pos  (T, 16)   measured joint positions (rad), interpolated from the
                         ~10 Hz joint-state stream
    touch      (T, 33)   net force on each of the 11 thumb/index/middle pads
                         (x, y, z), symlog-compressed sensor counts
    cmd_q      (T, 16)   last joint command at each step (rad)  <- the action
    joint_names, touch_names, label, heldout

joint_pos and cmd_q are both the raw hardware values, in the recording's own
joint order. State for the policy = [joint_pos, touch] (49-D).

  python build_dataset.py --raw /path/to/compose_playbacks --out data/playpolicy_penclick
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

# Taxel ranges of each pad, in the forward-kinematics order of
# `flex_taxel_fk/forces_local`. Ring finger and palm are left out because the
# demonstrations do not use them.
FINGERS = [
    {'key': 'thumb', 'pads': [('tip', 0, 30), ('mid', 30, 46), ('base', 46, 62)]},
    {'key': 'index', 'pads': [('tip', 62, 92), ('mid', 92, 108), ('low', 108, 124), ('base', 124, 140)]},
    {'key': 'middle', 'pads': [('tip', 140, 170), ('mid', 170, 186), ('low', 186, 202), ('base', 202, 218)]},
]


def symlog(x):
  return np.sign(x) * np.log1p(np.abs(x))


def main():
  p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  p.add_argument('--raw', type=Path, required=True)
  p.add_argument('--out', type=Path, default=ROOT / 'data/playpolicy_penclick')
  p.add_argument('--heldout', type=Path,
                 default=ROOT / 'heldout_th12.json')
  args = p.parse_args()

  held = {e['recording'] for e in json.loads(args.heldout.read_text())['episodes']}
  pads = [(f"{f['key']}_{name}", a, b)
          for f in FINGERS for name, a, b in f['pads']]
  touch_names = [f'{n}_{ax}' for n, _, _ in pads for ax in 'xyz']
  args.out.mkdir(parents=True, exist_ok=True)

  n = frames = 0
  worst_gap = 0.0
  for ep_path in sorted((args.raw / 'sparsh_skin').glob('episode_*.npz')):
    name = ep_path.stem[len('episode_'):]
    ep = np.load(ep_path, allow_pickle=True)
    raw = np.load(args.raw / f'{name}.npz', allow_pickle=True)
    meta = json.loads(str(ep['metadata_json']))
    T = int(meta['steps'])
    t = meta['t_start'] + np.arange(T) / meta['freq']

    state_t, pos = raw['state_t'], raw['position']
    cmd_t, cmd_q = raw['cmd_t'], raw['cmd_q']
    assert pos.shape[1] == 16 and cmd_q.shape[1] == 16, name
    assert np.all(np.diff(state_t) > 0) and np.all(np.diff(cmd_t) >= 0), name
    worst_gap = max(worst_gap, float(np.diff(state_t).max()))

    joint_pos = np.stack(
        [np.interp(t, state_t, pos[:, j]) for j in range(16)], 1)
    idx = np.clip(np.searchsorted(cmd_t, t, side='right') - 1, 0, None)
    cmd = cmd_q[idx]

    forces = ep['flex_taxel_fk/forces_local']
    assert forces.shape == (T, 368, 3), (name, forces.shape)
    touch = symlog(np.concatenate(
        [forces[:, a:b].sum(1) for _, a, b in pads], 1))

    arrays = dict(joint_pos=joint_pos, touch=touch, cmd_q=cmd)
    for k, v in arrays.items():
      assert np.isfinite(v).all(), (name, k)
    np.savez_compressed(
        args.out / f'{name}.npz',
        t=t.astype(np.float32),
        **{k: v.astype(np.float32) for k, v in arrays.items()},
        joint_names=np.asarray(raw['joint_names']),
        touch_names=np.asarray(touch_names),
        label=str(meta.get('label', '')),
        heldout=name in held)
    n += 1
    frames += T

  print(f'wrote {n} recordings, {frames} frames ({frames / 100 / 3600:.2f} h) '
        f'to {args.out}; {len(held)} listed as held out; '
        f'largest joint-state gap {worst_gap:.2f} s')


if __name__ == '__main__':
  main()
