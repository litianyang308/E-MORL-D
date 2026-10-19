# E-MORL/D: Decomposition-Based Multi-Objective Reinforcement Learning with Symmetry Exploitation

Official implementation of the **E-MORL/D** algorithm.

E-MORL/D extends decomposition-based multi-objective RL (MORL/D) by exploiting
**MDP symmetries** with *MDP-homomorphic networks* (van der Pol et al., 2020).
This repository contains **the algorithm code only** (no experiment / plotting scripts).

## What's inside

| Component | Location |
|---|---|
| **E-MORL/D** algorithm (`MORLD_S`) | `morl_baselines/multi_policy/morld/morld_s.py` |
| MORL/D (used by the S-MORL/D ablation baseline) | `morl_baselines/multi_policy/morld/morld.py` |
| Symmetry-aware policies `DQN_S` / `PPO_C` | `morl_baselines/single_policy/ser/mo_dqn_s.py`, `mo_ppo.py` |
| EUPG policy-gradient policies | `morl_baselines/single_policy/esr/` |
| Symmetric Deep Sea Treasure benchmark: maps `v0_S`, `DST_2_S`, `DST_3_S`, `DST_4_S` and their Pareto fronts | `morl_baselines/common/env.py` (`DSTS_MAP`, `DSTS_PARETO_FRONT`) |
| Shared utilities (buffers, networks, scalarizations, Pareto archive, indicators, weights) | `morl_baselines/common/` |
| MDP-homomorphic networks (extended with DST / FourRoom / LunarLander / Reacher symmetry groups) | `symmetrizer/` |

The code is based on a modified version of [morl-baselines](https://github.com/LucasAlegre/morl-baselines)
(see *Acknowledgments*).

## Installation

```bash
# 1. environment (Python 3.10)
conda env create -f environment.yml
conda activate e-morl-d

# 2. install this package (provides the `morl_baselines` and `symmetrizer` modules)
pip install -e .

# 3. patch mo-gymnasium (REQUIRED for the symmetric DST maps, see patches/mo_gymnasium/)
DST_DIR=$(python -c "import mo_gymnasium, os; print(os.path.join(os.path.dirname(mo_gymnasium.__file__), 'envs'))")
cp patches/mo_gymnasium/deep_sea_treasure.py $DST_DIR/deep_sea_treasure/
cp patches/mo_gymnasium/four_room.py        $DST_DIR/four_room/
```

> GPU: `requirements.txt` installs PyTorch with CUDA 11.8. For a CPU-only machine, install
> `pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu` instead.

## Quick start

Training E-MORL/D on a symmetric Deep Sea Treasure variant (`v0_S`):

```python
import torch
from morl_baselines.common.evaluation import seed_everything
from morl_baselines.common.env import get_ref_point_and_pfs, DSTS_MAP, DSTS_PARETO_FRONT
from morl_baselines.multi_policy.morld.morld_s import MORLD_S

gamma = 0.99
env_name = "v0"  # one of: "v0", "DST_2", "DST_3", "DST_4"

env, eval_env, ref_point, _ = get_ref_point_and_pfs("deep-sea-treasure-concave-v0", gamma)
env.unwrapped.sea_map = DSTS_MAP[env_name + "_S"]
eval_env.unwrapped.sea_map = DSTS_MAP[env_name + "_S"]
known_pareto_front = DSTS_PARETO_FRONT[env_name]
env_name = "deep-sea-treasure-concave-" + env_name + "_S"

seed_everything(3047)
algo = MORLD_S(
    env=env,
    env_name=env_name,
    alg_name="E-MORL-D",
    pop_size=10,
    net_arch=[64, 64],
    symmetry=True,
    policy_name="PPO_C",          # "PPO_C" (continuous actions) or "DQN_S" (discrete)
    scalarization_method="ws",    # "ws" or "tch"
    evaluation_mode="ser",
    weight_adaptation_method="PSA",
    weight_init_method="uniform",
    sharing_mechanism=[],
    update_passes=10,
    neighborhood_size=1,
    exchange_every=100,
    shared_buffer=False,
    gamma=gamma,
    log=False,
    seed=3047,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
)
algo.train(
    eval_env=eval_env,
    total_timesteps=int(1e5) + 1,
    ref_point=ref_point,
    known_pareto_front=known_pareto_front,
)
```

Notes:

* Training results are written to `../res_s/<env_name>/` relative to the current working
  directory; loss curves to `./Loss/<env_name>/`.
* For non-DST environments, simply use
  `env, eval_env, ref_point, known_pareto_front = get_ref_point_and_pfs("<mo-gymnasium-env-id>", gamma)`
  and pass `env_name="<mo-gymnasium-env-id>"`.
* The same `MORLD_S` class with `policy_name="DQN_S"` and `symmetry=True` reproduces the
  S-MORL/D ablation baseline when run with `MORLD` (`morld.py`).

## Project structure

```
E-MORL-D/
├── morl_baselines/
│   ├── multi_policy/morld/  # MORLD_S (E-MORL/D) and MORLD (MORL/D)
│   ├── single_policy/       # DQN_S / PPO_C symmetry-aware policies, EUPG policies
│   └── common/              # utilities + symmetric DST benchmark maps (env.py)
├── symmetrizer/             # MDP-homomorphic networks (top-level package)
├── patches/mo_gymnasium/    # patched mo-gymnasium envs (required, see inside)
├── environment.yml / requirements.txt / pyproject.toml
└── LICENSE
```

## Acknowledgments

This code is based on:

* **[morl-baselines](https://github.com/LucasAlegre/morl-baselines)** (MIT License, © 2022 Lucas Alegre, Florian Felten) — modified for this work;
* **[symmetrizer](https://github.com/ElisevanderPol/symmetrizer)** — MDP homomorphic networks
  (E. van der Pol, D. E. Worrall, H. van Hoof, F. A. Oliehoek, M. Welling,
  *MDP Homomorphic Networks: Group Symmetries in Reinforcement Learning*, ICML 2020), extended with
  DST / FourRoom / LunarLander / Reacher symmetry groups.

## Citation

If you use this code, please cite our paper:

```bibtex
% TODO: add paper citation
```

## License

[MIT](LICENSE)
