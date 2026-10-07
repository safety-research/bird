# BIRD: A Benchmark for Iterative Reward Design

[![arXiv](https://img.shields.io/badge/arXiv-2610.04364-b31b1b.svg)](https://arxiv.org/abs/2610.04364)

Code for [**A Bird's-Eye View of Iterative Reward Design**](https://arxiv.org/abs/2610.04364)

[Paper](https://arxiv.org/abs/2610.04364) · [Quickstart](#quickstart) · [Methods](#methods) ·
[Reproducing the paper](#reproducing-the-paper) · [Citation](#citation)

## Installation

BIRD uses [uv](https://docs.astral.sh/uv/). The core depends only on `pyyaml` and `numpy`;
everything else is an optional extra, imported only by the code path that needs it.
`uv sync` installs exactly the extras it is given (and removes any others), so name every
extra you need in one command:

```bash
uv sync --extra test                                   # core + tests: runs fully offline
uv sync --extra test --extra metaworld --extra sb3 --extra anthropic --extra video
                                                       # + Meta-World MT10 and Gym MuJoCo with SB3 PPO,
                                                       #   Claude as generator / judge, mp4 recording
```

Runs that call Claude need `ANTHROPIC_API_KEY` in the environment.

HumanoidBench and Assistax pin simulator versions that cannot share one environment with
Meta-World, so each has its own setup script and virtual environment
(`.venv-humanoid`, `.venv-jax`); see [GPU benchmarks](#gpu-benchmarks-humanoidbench-and-assistax).

Methods judged by a vision-language model render MuJoCo frames, which needs an EGL or
OSMesa library. On a machine without one, run `bash scripts/setup_gl.sh` once (it installs
Mesa into `~/.local/opt/gl`), then `source scripts/setup_gl.sh --env` in each shell that
renders.

## Quickstart

Every command below runs offline. The `tester` profile swaps in a mock LLM, a toy
environment and a mock learner, so a whole search finishes in about a second.

```bash
uv run python3 bird.py --list-configs                      # every configuration, by directory
uv run python3 bird.py -c eureka --profile tester          # run Eureka end to end, offline
uv run python3 bird.py -c rda --print-config               # the fully resolved configuration
uv run python3 bird.py --diff eureka rda                   # the keys on which two methods differ
uv run python3 bird.py -c eureka -s generate.n_candidates=4 -s seed=7 --dry-run
uv run python3 -m pytest tests -q
```

Each run writes `runs/<name>-<config hash>-<timestamp>/` containing the fully resolved
configuration, every candidate reward, its training curve and evaluation, the prompts
and responses, and the returned reward.

### How configurations work

Every key is declared in `configs/_default.yaml` (with its default) and in
`bird/schema.py` (with its allowed values). Unknown keys are errors, and every value that
names an implementation is a key in the component registry (`bird/registry.py`), so a
configuration cannot select something that does not exist.

- `extends:` inherits from another configuration; lists replace rather than append.
- `--profile` layers an execution profile (`tester`, `dev`, `full`) on top of the method
  (order: defaults, `extends:` chain, method file, profile, `-s` overrides). A profile may
  set only execution keys (models, budgets, backend, parallelism, tracking, video), so it
  changes *how expensively* a run executes, never *which method* it is.
- `-s key=value` overrides any key on the command line.
- `-c` takes a file path, a path relative to `configs/` (`methods/eureka`,
  `hillclimb/v2_verify`), or a bare name (`eureka`, `era_u`, `v2_verify`); a bare name
  must match exactly one file under `configs/` (outside `_profiles/`), and an ambiguous
  one is refused.
- The resolved configuration is written with every run, and its hash is the run id.

## Methods

| Method | Configuration (`configs/methods/`) |
|---|---|
| Optimal Reward Search (Singh et al., 2009)† | `singh_orp` |
| Language to Rewards (Yu et al., 2023)† | `l2r` |
| Eureka (Ma et al., 2024) | `eureka` (published ablation: `eureka_no_evolution`) |
| Text2Reward (Xie et al., 2024) | `text2reward_zeroshot`, `text2reward_human`, `metaworld_text2reward_zeroshot` |
| DrEureka (Ma et al., 2024) | `dreureka` |
| REvolve (Hazra et al., 2025) | `revolve` |
| CARD (Sun et al., 2025) | `card` (five-iteration protocol: `card_iter5`) |
| RF-Agent (Gao et al., 2025) | `rf_agent` |
| R\* (Li et al., 2025) | `rstar` |
| ROSKA (Huang et al., 2025) | `roska` |
| LaRes (Li et al., 2025) | `lares` |
| Gran Turismo (Ma et al., 2025) | `gt` |
| RDA (Lee et al., 2026) | `rda` (HumanoidBench setting: `rda_humanoidbench`) |
| LIMEN (Jaswal et al., 2026) | `limen`, `limen_reward_only` |
| **ERA-U** (this paper) | `configs/era_u.yaml` |
| **ERA-S** (this paper) | `configs/era_s.yaml` |

† Included in simplified form (see the paper's appendix).

Each method file holds the values its paper and released code specify, with a short
citation for each; † marks a value where the paper and the released code disagree, ‡ a
value the paper leaves unspecified. Most are not runnable as written (they name retired
models or budgets of hundreds of millions of steps); the paper's experiments run them
under a matched protocol, described next.

`configs/hillclimb/` holds the 24 configurations of the hill-climb behind ERA-U (plus four
parents in their inheritance chain), named as in the paper's hill-climb table (`v1`,
`v2_verify`, ..., `v4_peak_noes40`); `era_u.yaml` is configuration 24 under its paper name.

## Reproducing the paper

The paper runs every method under one protocol: eight candidates per iteration, five
iterations, 1M environment steps per policy training (20M on Assistax), one RL optimizer
per benchmark, `claude-opus-5` as the reward generator and vision-language judge, and
`claude-sonnet-5` for secondary evaluator roles. `scripts/paper_cell.py` turns a (method,
task, seed) triple into the exact command:

```bash
uv run python3 scripts/paper_cell.py --list                                   # methods and the 21 tasks
uv run python3 scripts/paper_cell.py --method era_u --task mt10_push-v3 --seed 0          # print the command
uv run python3 scripts/paper_cell.py --method era_u --task mt10_push-v3 --seed 0 --dry-run
uv run python3 scripts/paper_cell.py --method eureka --task gym_hopper_hop --seed 0 --run
```

| Benchmark | Tasks | Learner | Seeds |
|---|---|---|---|
| Meta-World MT10 (v3) | 10 | Stable-Baselines3 PPO, CPU | 10 |
| Gym MuJoCo (v5) | 5 | Stable-Baselines3 PPO, CPU | 10 |
| Assistax | 4 | Assistax's IPPO in JAX, one GPU, 20M steps | 5 |
| HumanoidBench | 2 | FastTD3 with SimbaV2 networks, one GPU | 5 |

A search trains 40 policies (RF-Agent 38; CARD, which trains one candidate per iteration,
4). REvolve runs 13 islands, as its configuration specifies, except on HumanoidBench,
where the paper's runs used 3 (`paper_cell.py` sets this). As a rough guide from the
paper's runs: a Meta-World or Gym MuJoCo search takes 3–6 hours on 8 CPU cores, an
Assistax search about 8 GPU-hours and a HumanoidBench search about a day of GPU time; a
search makes on the order of a hundred LLM calls (more for methods with an in-loop
vision-language judge).

`paper_cell.py` covers the method comparison and the hill-climb. The ablation study, the
candidate-allocation sweep and the EPIC analysis are configurations of the same loop; the
keys each one varies are listed in the paper's appendix.

### GPU benchmarks: HumanoidBench and Assistax

Both run from their own virtual environment, and `paper_cell.py` prints commands that use
it. Run `--run` with that environment's interpreter.

```bash
# HumanoidBench: mujoco 3.1.6, humanoid-bench @ cb118903; the paper's cells need CUDA torch
TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash scripts/setup_humanoid.sh
.venv-humanoid/bin/python scripts/paper_cell.py --method era_u --task h1hand_walk --seed 0 --run

# Assistax: JAX, assistax @ a7d94f4e, and the benchmark's zoo of 630 pretrained human partners
bash scripts/setup_jax.sh
hf download leohink/assistax-zoo zoo.tar.gz --repo-type dataset --local-dir ~/assistax-zoo
tar -xzf ~/assistax-zoo/zoo.tar.gz -C ~/assistax-zoo
export BIRD_ZOO_PATH=~/assistax-zoo/zoo
.venv-jax/bin/python scripts/paper_cell.py --method era_u --task upstream_assistax_feeding --seed 0 --run
```

The zoo's contents are checked against a recorded digest before training.

### Reproducibility

This is a cleaned-up release of the code used for the paper. Numbers will not reproduce
exactly, since the LLM calls are sampled, and a few prompt details differ from the paper's
runs. For example, the environment description now ends with an index of the observation
fields, and REvolve's environment description is natural-language rather than the Pythonic
class abstraction used in the paper's Meta-World, Gym MuJoCo and HumanoidBench runs.
`scripts/paper_cell.py` restores the other settings those runs used.

## Reward hacking on HumanoidBench

`policies/h1hand_hb_hacks/` contains the scripted policies from the paper's Section 4.4:
`moonhop` and `crawl_rock` reach high reference reward on HumanoidBench walk and crawl by
hopping facing backward or rocking in place, which shows that those reference rewards can
be maximized without walking or crawling. The directory also holds `hop`, a supplementary
example (a two-footed forward hop that clears the walk bar the same way). Evaluate them
with `scripts/eval_policy.py`.

## Repository layout

```
bird.py              the IRD loop and the command-line interface
bird/                config system, registry, components for each stage, LLM clients
bird/envs/           environment adapters (Meta-World, Gym MuJoCo, Assistax, HumanoidBench, toy)
configs/             configurations
  _default.yaml      every key and its default
  methods/           the published methods, their variants, and zeroshot (the root they extend)
  era_u.yaml         ERA-U, the paper's unsupervised recipe
  era_s.yaml         ERA-S, the paper's supervised recipe
  hillclimb/         the hill-climb configurations behind ERA-U
  examples/          a small illustrative configuration (the JAX reward path)
  _profiles/         execution profiles (tester, dev, full, and two for HumanoidBench)
tasks/               one task specification per environment (description, horizon, metric)
policies/            scripted policies (Meta-World policies used by the hill-climb's demonstration verifier;
                     HumanoidBench reward-hacking policies; controllers for the toy and classic-control
                     tasks, used by the tests)
scripts/             setup scripts, the paper protocol, spec generators, utilities
tests/               the test suite
docs/figures/        the figures in this README
licenses/            license texts of the third-party works listed in THIRD_PARTY_NOTICES.md
```

## Tests

```bash
uv run python3 -m pytest tests -q
```

Tests that need an optional dependency (a simulator, a learner, an API key) skip when it
is absent, so the suite passes under `--extra test` alone; install the corresponding extras
to run them. Tests that render frames are the exception: they carry the `gl` marker and
fail rather than skip when a simulator is installed but no GL library is, so either set one
up (`scripts/setup_gl.sh`, see [Installation](#installation)) or deselect them with
`-m "not gl"`. `-m "not slow"` runs the fast offline subset.

## Citation

```bibtex
@article{bhamidipaty2026bird,
  title   = {A Bird's-Eye View of Iterative Reward Design},
  author  = {Bhamidipaty, Logan Mondal and Robson, Lauren and Petrini, Linda and
             Lyu, Shengrui and Ndousse, Kamal},
  journal = {arXiv preprint arXiv:2610.04364},
  year    = {2026}
}
```

## Acknowledgments

This work was done as part of the Anthropic Fellows Program.

## License

MIT (see `LICENSE`). BIRD contains code, assets and prompt text derived from third-party
works under their own licenses; see `THIRD_PARTY_NOTICES.md` and `licenses/`.
