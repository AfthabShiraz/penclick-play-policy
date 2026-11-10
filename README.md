# Pen-click play policy

A diffusion-transformer play policy trained on LEAP hand + XELA skin pen-click
recordings. It is step 1 of PlayPolicy, "Dexterity from Supervised Play"
(arXiv 2610.22340), trained with the authors' unmodified trainer and paper
settings from [FAR-hands/play_policy](https://github.com/FAR-hands/play_policy)
(commit `e41cf7c`). Only the data loading is replaced.

## What the model does

| | |
|---|---|
| Input | 49 numbers: 16 measured joint positions (rad) + 33 touch values |
| Output | the next 200 steps (2 s at 100 Hz) of 16 joint commands (rad); the first 50 are meant to be executed |
| Model | DiT diffusion policy, 7.49M parameters, 10 sampling steps |
| Training data | 514 recordings, 1.15M frames; 49 recordings of thumb take `th_12` held out |
| Training | 40,000 steps, batch 64; this is the best checkpoint by validation error (step 39,000) |

Touch is the summed x/y/z force on each of the 11 thumb, index and middle pads,
symlog-compressed. Joint positions and commands are raw hardware values in the
recordings' joint order (`th_cmc, th_axl, th_mcp, th_ipl, if_…, mf_…, rf_…`);
the names are stored in the checkpoint.

The model imitates the recorded commands. It has no goal, reward or click
signal, and it was trained on failed attempts as well as successful ones.

## Result on the held-out recordings

Mean absolute error in rad over the 50 executed steps, 2,048 windows, against a
baseline that holds the current measured joint position
(`results/policy_eval_heldout.json`):

| Joints | Policy | Hold position |
|---|---|---|
| All | 0.019 | 0.032 |
| Index | 0.018 | 0.049 |
| Middle | 0.024 | 0.048 |
| Thumb | 0.032 | 0.031 |
| Ring | 0.0009 | 0.0003 |

The thumb is no better than the baseline; the held-out set is a thumb
demonstration the model never saw. This is an offline score only. The policy
has not been run on the hand.

## Use it

```bash
git clone https://github.com/FAR-hands/play_policy && cd play_policy
git checkout e41cf7c
pip install torch "numpy<2"

python /path/to/this/repo/run_policy.py \
    --ckpt /path/to/this/repo/checkpoints/penclick_play_policy_step39000.pt
```

`run_policy.py` has two functions to import: `load_policy(ckpt)` and
`sample_chunks(policy, norm_s, norm_a, state, n)`, which returns `n` sampled
command chunks of shape `(200, 16)` in rad. It was tested with torch 2.14 and
numpy 1.26.

## Files

| File | Purpose |
|---|---|
| `checkpoints/penclick_play_policy_step39000.pt` | weights, input/output scaling, model settings, joint and touch names (86 MB; also holds the optimiser state) |
| `run_policy.py` | load the model and sample from a state |
| `build_dataset.py` | build the training files from the raw recordings (`compose_playbacks/`) |
| `train_policy.py` | train, by calling the authors' trainer on that data |
| `eval_policy.py` | produce the held-out table above |
| `heldout_th12.json` | the 49 held-out recordings |

The recordings themselves are not in this repository.

To retrain, from the `play_policy` root:

```bash
python /path/to/build_dataset.py --raw /path/to/compose_playbacks --out data/playpolicy_penclick
python /path/to/train_policy.py --data_dir data/playpolicy_penclick
python /path/to/eval_policy.py --ckpt runs/play_policy/<run>/ckpt_best.pt \
    --data_dir data/playpolicy_penclick --out results/policy_eval.json
```
