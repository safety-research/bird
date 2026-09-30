"""`problem.horizon` -- the episode length as a config key, truncation only.

WHAT THE KEY IS FOR. An episode's horizon is an ENVIRONMENT fact and stays one:
it is read from `tasks/<id>/shared_spec.yaml::env.horizon` and `null` (the
default) means "whatever the adapter says". But a horizon is also a value some
papers PIN as part of their method -- RDA's HumanoidBench column reports "Maximum
Environment Steps: Task-dependent (MS), 500 (HB)" (`refs/tex/rda/appendix.tex:1424`)
against HumanoidBench's shipped 1000 -- and without this key the only way to express
that would be to edit the shared spec, which re-bases the task for every OTHER
consumer. That is a schema missing a knob.

THREE PROPERTIES, and each has a failure this file is written against:

  * TRUNCATION ONLY. An integer above the environment's own horizon is refused,
    not clamped: a clamp would let a config state 2000, run 1000, and record the
    2000 nowhere.
  * IT REACHES THE ADAPTER, hence every backend, the views, the rollouts, the
    trace and the PROMPT (`describe()` renders the horizon). A key that resolved
    and reached nothing is the declared-but-unread class.
  * IT IS RECORDED AS EXECUTED (`horizon_effective` on the seed row), because a
    run that asked for 500 and a process the override never reached leave
    identical curves otherwise.
"""

import types

import numpy as np
import pytest

from pathlib import Path

from conftest import REPO, backend_param  # noqa: F401  (path setup)
from bird import registry
from bird.config import ConfigError, load
from bird.envs.base import apply_horizon


def _env(env_id="pendulum"):
    return registry.get("env", env_id)(None)


# ==========================================================================
# The key, at load
# ==========================================================================


def test_null_is_the_environments_own_and_changes_nothing():
    """The default, and the property that makes this key safe to add: `null`
    leaves the adapter exactly as it was."""
    cfg = load("rda", profile="dev", overrides={"problem.env_id": "pendulum"})
    assert cfg["problem.horizon"] is None
    env = _env()
    own = int(env.horizon)
    assert apply_horizon(env, cfg) == (own, None)
    assert int(env.horizon) == own


def test_an_integer_truncates_the_adapter_and_is_the_returned_value():
    cfg = load("rda", profile="dev", overrides={"problem.env_id": "pendulum",
                                                "problem.horizon": 7})
    env = _env()
    assert int(env.horizon) > 7
    assert apply_horizon(env, cfg) == (7, 7)
    assert int(env.horizon) == 7


def test_longer_than_the_environments_own_is_CLAMPED_at_construction_and_recorded():
    """CLAMPED at construction, not refused -- and refused at LOAD instead when
    the config chose its own env. The two sites do different things on purpose.

    WHY NOT REFUSE HERE: a paper config names no env (`configs/methods/rda.yaml`'s
    header says so), so `configs/methods/rda_humanoidbench.yaml`'s Table-1 horizon of
    500 resolves against `toy_reacher`'s 25 under every profile -- and a
    construction refusal would make that config UNRUNNABLE on the tester
    profile, which `tests/test_pipeline.py` exercises for every method point.
    The same config pins `train.env_steps: 10000000`, equally impossible for a
    toy env, and the profile handles THAT by overriding, because it is a
    profile key.
    `problem.horizon` is §0 and must not become one (a profile key is a key a
    profile may change the science with). So the pin is a CITATION until an env
    is chosen, the chosen env caps it, and both numbers are recorded.

    THE OBJECTION TO CLAMPING IS ANSWERED RATHER THAN DROPPED. It is that "a
    config could state 2000, run 1000 and record the 2000 nowhere" -- so
    `apply_horizon` returns the pair and every seed row carries
    `horizon_requested` beside `horizon_effective`. Asserted here, on the
    return value, and in `test_a_PRODUCED_seed_row_...` on a row.

    Driven through a STUB config rather than `load`, because on a spec-backed
    env the load-time rule fires first (the test below) and the only way to
    reach the construction path on its own is to hand it the value.
    """
    env = _env()
    own = int(env.horizon)
    stub = types.SimpleNamespace(get=lambda key, default=None:
                                 10 ** 6 if key == "problem.horizon" else default)
    effective, requested = apply_horizon(env, stub)
    assert (effective, requested) == (own, 10 ** 6), (
        "the clamp must return what RAN and what was ASKED FOR, or the cap is "
        "invisible in the artifact -- which was the whole objection to clamping")
    assert int(env.horizon) == own, "a clamped call must leave the adapter alone"


def test_zero_and_negative_are_refused():
    for bad in (0, -5):
        with pytest.raises(ConfigError, match=r"not an episode length"):
            load("rda", profile="dev", overrides={"problem.env_id": "pendulum",
                                                  "problem.horizon": bad})


def test_longer_than_the_spec_is_refused_at_load_on_a_chosen_env():
    """The early half, where it costs nothing: a spec's `env.horizon` is
    readable without constructing a MuJoCo model, so a command line that asks
    for a longer episode than the benchmark ships fails at config load."""
    with pytest.raises(ConfigError, match=r"longer than h1hand_package's own horizon"):
        load("rda_humanoidbench", profile="humanoid_simba",
             overrides={"problem.env_id": "h1hand_package",
                        "problem.horizon": 2000})


def test_a_config_that_chose_no_env_is_not_checked_against_the_default_one():
    """THE SCOPE, and it is the paper configs' own idiom rather than a convenience.
    A paper config here names no env -- `configs/methods/rda.yaml`'s header says so in
    as many words -- so it resolves to `_default.yaml`'s `toy_reacher`, whose
    spec horizon is 25. Checking `rda_humanoidbench`'s 500 against THAT would
    refuse the file for an environment it never meant, and would break
    `bird.py --validate-all` for the configs the rule exists to validate.

    Nothing is lost: a command line that chooses an env is checked in full (the
    test above), and a direct run is caught at construction.
    """
    cfg = load("rda_humanoidbench", profile="humanoid_simba")
    assert cfg["problem.env_id"] == "toy_reacher", (
        "if rda_humanoidbench ever pins its env, this test's premise is gone and "
        "the load-time rule should check it")
    assert cfg["problem.horizon"] == 500
    # ... and the mismatch is CLAMPED one step later rather than refused, which
    # is what keeps this config runnable on the tester profile while still
    # recording that the paper's 500 is not what ran.
    env = _env("toy_reacher")
    assert int(env.horizon) < 500
    effective, requested = apply_horizon(env, cfg)
    assert (effective, requested) == (int(_env("toy_reacher").horizon), 500)


def test_the_hb_column_pins_the_papers_500():
    """RDA Table 1's row is "Maximum Environment Steps & Task-dependent (MS),
    500 (HB)" -- split per benchmark like the four other split RL rows beside it. And it is
    half of the gamma derivation: SimbaV2 computes gamma from the episode
    length, so 500 with `action_repeat: 2` is what gives Table 1's own 0.98
    while the shipped 1000 would give 0.99."""
    cfg = load("rda_humanoidbench", profile="humanoid_simba")
    assert cfg["problem.horizon"] == 500
    hp = cfg["train.hyperparameters"]
    eff = cfg["problem.horizon"] / hp["action_repeat"]
    gamma = max(min((eff / 5 - 1) / (eff / 5), 0.995), 0.95)
    assert gamma == pytest.approx(hp["gamma"]), (
        "the pinned gamma must be the one upstream's heuristic derives from the "
        "pinned horizon and repeat")


def test_two_horizons_are_two_run_directories():
    """In `Config.hash()`, like every other key: a horizon is an experiment
    parameter, and two of them must not share a run directory -- the resume
    machinery adopts by hash."""
    a = load("rda", profile="dev", overrides={"problem.env_id": "pendulum"})
    b = load("rda", profile="dev", overrides={"problem.env_id": "pendulum",
                                              "problem.horizon": 50})
    assert a.hash() != b.hash()


# ==========================================================================
# It reaches everything that reads a horizon
# ==========================================================================


def test_the_truncated_horizon_reaches_the_prompt_and_a_rollout():
    """`describe()` renders the horizon into the prompt, and `_rollout` bounds
    its loop by it. A key that resolved and reached neither would be the
    declared-but-unread class -- and the prompt half is what a wrapper env
    would have missed."""
    from bird.components import training as T

    cfg = load("rda", profile="dev", overrides={"problem.env_id": "pendulum",
                                                "problem.horizon": 9})
    env = _env()
    own = int(env.horizon)
    apply_horizon(env, cfg)
    text = env.describe()
    # `EnvAdapter.describe` renders "An episode lasts at most {horizon} steps."
    # -- the sentence a candidate's prompt carries, so this asserts the phrase
    # rather than the digit, which could match anything.
    assert "at most 9 steps" in text, (
        "the truncated horizon must reach the rendered description; the prompt "
        "is what a wrapper-env implementation would have missed")
    assert f"at most {own} steps" not in text, (
        "the pre-truncation horizon must not also appear -- two answers to one "
        "question in one prompt")

    reward = T.compile_reward(
        "def compute_reward(state, action):\n    return 0.0, {}\n",
        type("C", (), {"cand_id": "c0", "reward_code": "", "iteration": 0})())
    traj, steps, _gt = T._rollout(
        env, lambda s: np.zeros(int(np.asarray(env.action_low).size)),
        np.random.default_rng(0), reward,
        T._error_router("exception_soft", __import__("io").StringIO()))
    assert steps <= 9, f"a rollout ran {steps} steps past a horizon of 9"


def _learner_backends():
    """Every `train_backend` registry member.

    OFF THE REGISTRY, NEVER A LITERAL LIST, and that is a lesson rather than a
    style preference. `parametrize("backend", ["mock", "tabular", "none"])`
    would omit `sb3`, `fasttd3` and `simba_v2`, the three that matter for a
    real run: deleting `horizon_effective` from `_sb3_run` would then break no
    test, invisible for exactly the reason a literal list is always
    invisible.

    `tests/test_pruning.py::_backends` is the same helper for the same reason.
    """
    registry.load_all()
    return sorted(n for (k, n) in registry._REGISTRY if k == "train_backend")


def _skip_if_unavailable(name: str) -> None:
    """`tests/test_pruning.py::_skip_if_unavailable`, plus the jax tier's backend (`assistax_ppo`).
    A skip is not a pass: these run only where the extras are installed."""
    if name == "sb3":
        pytest.importorskip("stable_baselines3")
        pytest.importorskip("gymnasium")
    # The jax tier's learner. `jax` IS in an extra, so this gate can open
    # on an install with the jax extra; there the case runs. Narrow, as the docstring requires -- it names the package and
    # swallows no assertion, so on a machine with the extra the backend can only
    # pass by routing through the dispatch.
    if name == "assistax_ppo":
        pytest.importorskip("jax")
    if name in ("fasttd3", "simba_v2"):
        pytest.importorskip("torch")


#: The torch backends run their upstream defaults otherwise -- 128 envs, batch
#: 32768, 1024-wide distributional critics, a GPU-scale set -- and this file
#: tests whether a FIELD is written, not learning.
_TINY = {"fasttd3": {"num_envs": 2, "batch_size": 32, "buffer_size": 64,
                     "critic_hidden_dim": 8, "actor_hidden_dim": 8,
                     "num_atoms": 5, "v_min": -10.0, "v_max": 10.0,
                     "learning_starts": 1, "compile": False},
         "simba_v2": {"batch_size": 32, "buffer_size": 64,
                      "critic_hidden_dim": 8, "actor_hidden_dim": 8,
                      "critic_num_bins": 5, "learning_starts": 1,
                      "compile": False}}


@pytest.mark.parametrize("backend", [backend_param(b) for b in _learner_backends()])
def test_a_PRODUCED_seed_row_carries_the_horizon_it_trained_under(backend):
    """`horizon_effective`, asserted on a row a backend ACTUALLY WROTE.

    NOT A SOURCE SCAN. A scan asserting, per MODULE, that the literal
    `"horizon_effective"` appears in the source stays green while an entire
    backend family produces rows without the field: `_sb3_run` lives in the
    same module as `_run_backend`, so `bird/components/training.py` satisfies
    the assertion on `_sb3_run`'s line while every `mock`, `tabular` and `none`
    row carries nothing. A test whose unit is the file cannot see a writer
    inside it, and `'"x"' in src` is satisfied by the literal appearing in a
    comment.

    That family is not a stub tier either: `mock` is what the tester profile
    runs, so these are precisely the rows obtainable WITHOUT a GPU -- and
    `none` and `tabular` are real method points (L2R's planner, Singh 2009),
    each free to pin a horizon.

    So: run the backend and assert the value on the row it wrote.

    PARAMETRISED OVER THE REGISTRY FAMILY, not over a literal list: listing
    `["mock", "tabular", "none"]` would silently drop `sb3`, `fasttd3` and
    `simba_v2`, the three that matter for a real run. A literal list cannot
    grow when the family does.
    `_learner_backends()` says more about why.

    The surrogates need no torch and take about a second each; the learner
    backends `importorskip` and run only where their extras are installed.
    """
    from bird.budget import Budget
    from bird.context import Context
    from bird.state import RunState
    from bird.types import Candidate

    _skip_if_unavailable(backend)
    overrides = {"problem.env_id": "pendulum", "problem.horizon": 7,
                 "train.backend": backend, "output.tracker": "none",
                 "llm.generator.provider": "mock",
                 "llm.evaluator.provider": "mock",
                 "evaluate.rollouts_per_candidate": 1,
                 "train.env_steps": 200}
    if backend in _TINY:
        overrides["train.hyperparameters"] = dict(_TINY[backend])
    if backend in ("fasttd3", "simba_v2"):
        # both need a continuous-action env, which pendulum is, and simba_v2
        # reads `train.n_parallel_envs` as its fleet width.
        overrides["train.n_parallel_envs"] = 2
        overrides["train.algorithm"] = "fasttd3" if backend == "fasttd3" else "sac"
        overrides["train.architecture"] = "mlp" if backend == "fasttd3" else "simba_v2"
    cfg = load("rda", profile="tester", overrides=overrides)
    ctx = Context(cfg=cfg, budget=Budget())
    ctx.env = registry.get("env", "pendulum")(ctx)
    assert apply_horizon(ctx.env, cfg) == (7, 7)
    cand = Candidate(
        cand_id="c0",
        reward_code="def compute_reward(state, action):\n    return 0.0, {}\n",
        iteration=0)
    res = registry.get("train_backend", backend)(ctx, RunState(), cand, 1)
    assert res.seed_metrics, f"{backend} produced no seed row"
    for row in res.seed_metrics:
        assert "horizon_effective" in row, (
            f"the {backend} backend wrote a seed row with no horizon_effective; "
            "two runs that differ in the episode length are two tasks and a "
            "results table without this field cannot tell")
        assert row["horizon_effective"] == 7, row["horizon_effective"]
        assert row["horizon_requested"] == 7, row["horizon_requested"]


def test_every_config_bearing_adapter_construction_applies_the_horizon():
    """THE STRUCTURAL GUARD, and it exists because a hand-maintained list of
    callers goes stale.

    A site that constructs an adapter WITH a config and skips the horizon
    would, since `final_retrain` is a `post:` phase, retrain at the
    environment's shipped horizon beside a search at the pinned one -- the
    exact failure `horizon_effective` exists to make visible, arriving through
    the one door it cannot see.

    The rule is structural, so the check is structural, in the same instrument
    `tests/test_no_method_branching.py` uses: an AST walk over the source for
    `registry.get("env", <...>)(ctx)` -- a construction whose argument is a
    Context, i.e. one that HAS a config -- requiring the file to apply the
    horizon. A new site will happen; this is what catches it, rather than
    someone remembering to update a list.

    Deliberately file-scoped rather than statement-scoped, and that IS the
    weaker form: it would pass a file that applied the horizon to one adapter
    and not a second. Two constructions in one file is not a shape this repo
    has, and the alternative (proving the call follows in control flow) is a
    dataflow analysis rather than a test. The scope is stated so the next
    reader knows what it does not cover -- which is the lesson of the source
    scan above.
    """
    import ast
    import warnings

    root = Path(REPO)
    offenders = []
    checked = 0
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel.startswith(("refs/", "tests/")) or "/.venv" in f"/{rel}":
            continue
        try:
            # `simplefilter` because `ast.parse` re-emits a file's own
            # DeprecationWarnings (invalid escape sequences, of which the tree
            # has a couple) and this test would otherwise add warnings to every
            # run's output for files it only reads. Not the test's finding, and
            # not the test's business.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        constructs_with_cfg = False
        for node in ast.walk(tree):
            # `<something>.get("env", ...)(<ctx>)` -- a call whose callee is
            # itself a `get("env", ...)` call and whose single argument is a
            # Name spelled `ctx` (or an attribute ending `.ctx`).
            if not isinstance(node, ast.Call) or len(node.args) != 1:
                continue
            inner = node.func
            if not (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr == "get" and inner.args
                    and isinstance(inner.args[0], ast.Constant)
                    and inner.args[0].value == "env"):
                continue
            arg = node.args[0]
            name = (arg.id if isinstance(arg, ast.Name)
                    else arg.attr if isinstance(arg, ast.Attribute) else "")
            if name == "ctx":
                constructs_with_cfg = True
        if constructs_with_cfg:
            checked += 1
            if "apply_horizon" not in path.read_text():
                offenders.append(rel)

    assert checked >= 2, (
        f"only {checked} config-bearing adapter construction(s) found; the walk "
        "has stopped matching the shape it is looking for, which would make this "
        "test pass vacuously")
    assert not offenders, (
        "these files construct an adapter with a Context -- so with a config in "
        "hand -- and never apply `problem.horizon`, which would be silently "
        f"ignored there: {offenders}")
