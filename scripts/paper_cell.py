#!/usr/bin/env python3
"""Print (or run) the exact command for one cell of the paper's experiments.

A *cell* is one reward search: one method, one task, one seed. The method configs in
configs/methods/ hold each method's published values; the paper ran every method under one
matched execution protocol (Section 4 and Appendix "Experimental Details"): eight
candidates per iteration, five iterations, 1M environment steps per policy training
(20M on Assistax), and one RL optimizer per benchmark. That protocol is a handful of
command-line overrides on top of the method config, and this script is where they live.

    uv run python3 scripts/paper_cell.py --method era_u --task mt10_push-v3 --seed 0
    uv run python3 scripts/paper_cell.py --method eureka --task gym_hopper_hop --seed 3 --run
    uv run python3 scripts/paper_cell.py --list

Benchmarks and learners (Appendix "Policy training"):
  * Meta-World MT10 and Gym MuJoCo: Stable-Baselines3 PPO on CPU (`--extra metaworld --extra sb3`).
  * HumanoidBench walk/crawl: FastTD3 with SimbaV2 networks on one CUDA GPU
    (separate environment: scripts/setup_humanoid.sh).
  * Assistax: the benchmark's own IPPO trainer in JAX on one CUDA GPU, 20M steps, with the
    630-partner human zoo (separate environment: scripts/setup_jax.sh).

`--protocol paper` (the default) reproduces the settings the paper's runs used.
`--protocol current` runs today's method configs unchanged. They differ only for CARD
on MT10, Gym MuJoCo and HumanoidBench, whose config gained fidelity fixes after the
paper's runs, and for REvolve's island count on HumanoidBench (see PAPER_PINS, and the
note below it on the one REvolve setting no override can restore).
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------------------
# The 21 tasks of the method comparison (Appendix "Benchmarks and tasks").
# ---------------------------------------------------------------------------------------
MT10 = ["mt10_button-press-topdown-v3", "mt10_door-open-v3", "mt10_drawer-close-v3",
        "mt10_drawer-open-v3", "mt10_peg-insert-side-v3", "mt10_pick-place-v3",
        "mt10_push-v3", "mt10_reach-v3", "mt10_window-close-v3", "mt10_window-open-v3"]
GYM = ["gym_half_cheetah", "gym_hopper_hop", "gym_swimmer_forward",
       "gym_inverted_pendulum_balance", "gym_reacher_reach"]
ASSISTAX = ["upstream_assistax_armmanipulation", "upstream_assistax_feeding",
            "upstream_assistax_scratchitch", "upstream_assistax_teethbrushing"]
HUMANOIDBENCH = ["h1hand_walk", "h1hand_crawl"]
PAPER_TASKS = MT10 + GYM + ASSISTAX + HUMANOIDBENCH

# The four MT10 tasks the ERA hill-climb was developed on (Appendix "The hill-climb").
HILLCLIMB_TASKS = ["mt10_pick-place-v3", "mt10_push-v3", "mt10_peg-insert-side-v3",
                   "mt10_drawer-open-v3"]

# Seeds per task in the method comparison.
SEEDS = {"mt10": 10, "gym": 10, "assistax": 5, "humanoidbench": 5}


def suite_of(task: str) -> str:
    if task.startswith("mt10_"):
        return "mt10"
    if task.startswith("gym_"):
        return "gym"
    if task.startswith("upstream_assistax_"):
        return "assistax"
    if task.startswith("h1hand_"):
        return "humanoidbench"
    raise SystemExit(f"{task!r} is not a task of the paper's four benchmarks; "
                     f"see --list")


# ---------------------------------------------------------------------------------------
# Per-benchmark execution: learner, device and step budget.
# ---------------------------------------------------------------------------------------
def search_steps(suite: str) -> int:
    return 20_000_000 if suite == "assistax" else 1_000_000


def suite_overrides(suite: str) -> List[str]:
    if suite == "assistax":
        learner = ["train.backend=assistax_ppo", "generate.reward_language=jax",
                   "train.architecture=ff_nps", "train.hyperparameters.device=cuda"]
    elif suite == "humanoidbench":
        learner = ["train.backend=fasttd3", "train.architecture=simba_v2",
                   "train.hyperparameters.device=cuda", "train.n_parallel_envs=24"]
    else:
        learner = ["train.algorithm=ppo"]
    return learner + [f"train.env_steps={search_steps(suite)}",
                      "output.video.record=all", "output.tracker=none",
                      "loop.resume_from=auto"]


# ---------------------------------------------------------------------------------------
# Per-method protocol: config file, iterations, and the matched-budget overrides.
# ---------------------------------------------------------------------------------------
NATIVE = "evaluate.fitness.source=native"   # rank on the environment's own signal

#: method -> (config, iterations). Iterations are an execution-profile key, so they
#: are always passed on the command line.
METHODS: Dict[str, tuple] = {
    "eureka":     ("eureka", 5),
    "rda":        ("rda", 5),
    "card":       ("card", 3),          # CARD's published protocol (three iterations)
    "card_iter5": ("card_iter5", 5),    # CARD's entry in the method comparison
    "rf_agent":   ("rf_agent", 5),
    "revolve":    ("revolve", 5),
    "rstar":      ("rstar", 5),
    "era_u":      ("era_u", 5),
    "era_s":      ("era_s", 5),
}

#: Hill-climb configurations (Appendix table, configs/hillclimb/). Iterations follow the
#: candidate schedule each configuration was designed around.
HILLCLIMB_ITERS = {"v1": 8, "v2_verify_wide": 4, "v2_verify_short": 2,
                   "v2_verify_adapt": 4, "v4_peak_noes40": 5}
HILLCLIMB = ["v1", "v2_verify", "v2_verify_wide", "v2_verify_short", "v2_verify_adapt",
             "v2_verify_kl", "v2_verify_noes", "v2_verify_es8", "v2_verify_es8_constant",
             "v2_verify_es8_ramp", "v2_verify_es8_distinct", "v2_verify_es8_personas",
             "v2_noverify_es8", "v3_tree", "v3_all", "v3_elite", "v3_actions", "v3_puct",
             "v3_selfverify", "v3_thought", "v3_thought_elite", "v4", "v4_tree",
             "v4_peak_noes40"]
for _name in HILLCLIMB:
    METHODS[f"hillclimb/{_name}"] = (f"hillclimb/{_name}", HILLCLIMB_ITERS.get(_name, 6))


def method_overrides(method: str, suite: str) -> List[str]:
    """The matched protocol: K = 8 candidates, the final retrain at the search budget,
    and the supervised methods ranked on the environment's own signal."""
    retrain = f"final_retrain.env_steps={search_steps(suite)}"
    table = {
        "eureka":   ["generate.n_candidates=8", retrain, NATIVE],
        "rda":      ["generate.n_candidates=8"],
        "rstar":    ["generate.n_candidates=8", "generate.crossover.n=2",
                     "loop.waves=[{'llm':6,'crossover':2,'when':'always'},"
                     "{'llm':0,'crossover':2,'when':'no_archive'}]",
                     retrain, NATIVE],
        "rf_agent": ["generate.n_candidates=8", "generate.tree.horizon_trainings=38", NATIVE],
        "revolve":  ["generate.n_candidates=8", NATIVE],
        "era_u":    ["generate.n_candidates=8"],
        "era_s":    ["generate.n_candidates=8"],
    }
    return table.get(method, []) + PPO_DEFAULTS.get((method, suite), [])


#: The protocol trains every method on Meta-World and Gym MuJoCo with PPO at its library
#: defaults; CARD's config carries a SAC hyperparameter block that does not apply to PPO.
#: Applies under both protocols.
PPO_DEFAULTS: Dict[tuple, List[str]] = {
    (m, s): ["train.hyperparameters=null"]
    for m in ("card", "card_iter5") for s in ("mt10", "gym")
}


#: `--protocol paper`: settings the paper's runs used where today's config differs.
#:
#: Keyed by (method, suite); suite "*" applies everywhere. Derived by resolving every
#: method on one task per benchmark (seed 0) both ways -- the as-run command at the
#: code point each paper cell ran, and this script's command -- and diffing the two
#: resolved configurations key by key. Each as-run reconstruction reproduced the config
#: hash in that cell's recorded run id.
#:
#: CARD on MT10 and Gym MuJoCo (both the 3-iteration `card` and `card_iter5`). The CARD
#: config now uses Text2Reward's verbatim Meta-World prompt and learner, which the paper's
#: runs predate. The paper's CARD cells used
#: the repository's own class-abstraction prompt, the per-task symbol table, the default
#: reward signature, PPO's own hyperparameters (no SAC block; `train.hyperparameters` was
#: `{}`, and null is the same thing to every reader of the key), and the sb3 path's
#: default of 3 evaluation episodes per checkpoint (null means the backend's own constant).
#: Without the `train.hyperparameters` pin, `train.algorithm=ppo` would receive the SAC
#: block's batch_size 512, net_arch [256, 256, 256] and the string ent_coef "auto_0.1".
_CARD_AS_RUN = ["generate.context.env_spec=pythonic_class_abstraction",
                "generate.postprocess.symbol_mapping=per_task",
                "generate.output.signature=compute_reward_state_action_next",
                "train.hyperparameters=null",
                "evaluate.checkpoint_eval_episodes=null"]
#: CARD on HumanoidBench (`card`) ran the same four prompt and
#: evaluation settings with `train.hyperparameters: {device: cuda}`. The SAC block cannot
#: be removed there without losing `device`, so its three keys that fasttd3 accepts are
#: set back to fasttd3's own defaults (the values the paper's run used); fasttd3 ignores
#: the other six with a warning, and gamma is 0.99 either way.
_CARD_AS_RUN_FASTTD3 = [pin for pin in _CARD_AS_RUN if not pin.startswith("train.")] + [
    "train.hyperparameters.batch_size=32768", "train.hyperparameters.learning_starts=10",
    "train.hyperparameters.tau=0.1"]
PAPER_PINS: Dict[tuple, List[str]] = {
    ("card", "mt10"): _CARD_AS_RUN,
    ("card", "gym"): _CARD_AS_RUN,
    ("card", "humanoidbench"): _CARD_AS_RUN_FASTTD3,
    ("card_iter5", "mt10"): _CARD_AS_RUN,
    ("card_iter5", "gym"): _CARD_AS_RUN,
    # CARD on Assistax ran today's CARD config: no pins.
    # REvolve on HumanoidBench ran with three islands (the paper's REvolve row).
    # Every other benchmark ran the config's 13.
    ("revolve", "humanoidbench"): ["update.archive.n_islands=3"],
}
#: Paper-run settings that --protocol paper does NOT reproduce:
#:  * REvolve on MT10, Gym MuJoCo and HumanoidBench ran with
#:    `generate.context.env_spec: pythonic_class_abstraction` and no symbol table
#:    (Assistax ran today's `natural_language_only`). No override can restore it: the
#:    loader refuses that pair, and any table that satisfies the check (`per_task`)
#:    rewrites names on these environments, which the paper's runs did not do.
#:  * The HumanoidBench RDA and ERA-U cells and the Assistax RDA, ERA-U and ERA-S cells
#:    come from reruns launched with `post=[]` (no final retrain). The search is the
#:    same either way; those cells simply have no retrained score. Not pinned, so every
#:    method keeps the protocol's retrain; add `post=[]` to match those runs' compute.
#:  * Prompt text is not configuration. Every MT10, Gym MuJoCo and HumanoidBench cell
#:    predates the flat-row field map that is now appended to the environment
#:    description; the Assistax cells include it.


def paper_pins(method: str, suite: str) -> List[str]:
    out: List[str] = []
    for (m, s), pins in PAPER_PINS.items():
        if m == method and s in (suite, "*"):
            out += pins
    return out


#: How to invoke Python for each benchmark. HumanoidBench and Assistax pin simulator
#: versions that cannot share the main environment, so they run from the environments
#: scripts/setup_humanoid.sh and scripts/setup_jax.sh create.
PREFIX = {
    "mt10": ["uv", "run", "python3"],
    "gym": ["uv", "run", "python3"],
    "humanoidbench": [".venv-humanoid/bin/python"],
    "assistax": [".venv-jax/bin/python"],
}


def build(method: str, task: str, seed: int, protocol: str = "paper",
          out: Optional[str] = None, profile: str = "full") -> List[str]:
    if method not in METHODS:
        raise SystemExit(f"unknown method {method!r}; see --list")
    suite = suite_of(task)
    config, iters = METHODS[method]
    sets = (suite_overrides(suite) + method_overrides(method, suite)
            + (paper_pins(method, suite) if protocol == "paper" else []))
    merged: Dict[str, str] = {}
    for o in sets:                      # later settings win, key by key
        key, _, value = o.partition("=")
        merged[key] = value
    cmd = PREFIX[suite] + ["bird.py", "--config", config, "--profile", profile,
           "--set", f"seed={seed}", "--set", f"problem.env_id={task}",
           "--set", f"loop.n_iterations={iters}"]
    for key, value in merged.items():
        cmd += ["--set", f"{key}={value}"]
    if out:
        cmd += ["--out", out]
    return cmd


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--method", help="one of --list's methods")
    ap.add_argument("--task", help="an env id of the paper's 21 tasks (see --list)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--protocol", choices=("paper", "current"), default="paper")
    ap.add_argument("--out", help="output directory (default: bird.py's own)")
    ap.add_argument("--run", action="store_true", help="run the command instead of printing it")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve and validate the config without running anything")
    ap.add_argument("--list", action="store_true", help="list methods and tasks")
    a = ap.parse_args(argv)
    if a.list:
        print("methods:", " ".join(m for m in METHODS if not m.startswith("hillclimb/")))
        print("hill-climb:", " ".join(m for m in METHODS if m.startswith("hillclimb/")))
        for name, tasks in (("mt10", MT10), ("gym", GYM), ("assistax", ASSISTAX),
                            ("humanoidbench", HUMANOIDBENCH)):
            print(f"{name} ({SEEDS[name]} seeds):", " ".join(tasks))
        print("hill-climb development tasks:", " ".join(HILLCLIMB_TASKS))
        return 0
    if not (a.method and a.task):
        ap.error("--method and --task are required (or --list)")
    cmd = build(a.method, a.task, a.seed, a.protocol, a.out)
    if a.dry_run:
        cmd.append("--dry-run")
    if not (a.run or a.dry_run):
        print(shlex.join(cmd))
        return 0
    # run with this interpreter from the repository root
    first = cmd.index("bird.py")
    return subprocess.call([sys.executable] + cmd[first:], cwd=REPO)


if __name__ == "__main__":
    sys.exit(main())
