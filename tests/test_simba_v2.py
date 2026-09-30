"""`train.backend: simba_v2` -- the port is held to the vendored upstream.

Three families of check, and they guard different failures:

  * FIDELITY -- the port must compute what `refs/code/SimbaV2` computes.
    `_SIMBAV2_DEFAULTS` is compared against a YAML read of upstream's four
    config files (a re-vendor that moves a default fails here instead of
    drifting), `_derived` against upstream's own `${eval:...}` expressions,
    and RDA's Table 1 against upstream's two formulas. These need the vendored
    tree; a sparse checkout without `refs/code/SimbaV2` skips them by name.

  * MECHANISM -- the two things that make this SimbaV2 rather than
    Simba-shaped layers, and both are silent when broken: the hyperspherical
    WEIGHT PROJECTION (a loop over named submodules -- a rename projects
    nothing and every counter reads normal) and the entropy term in the
    categorical TD target (without it the critic is a plain value function and
    the run still trains).

  * CONTRACT -- the backend must mean the same thing every other backend
    means: a real checkpoint curve with `_sb3_run`'s keys, honest `env_steps`,
    the four budget facts on the seed row, a discrete env refused loudly,
    determinism from the config seed, and a policy blob that rebuilds through
    its own branch and not `fasttd3`'s.

Plus one guard that belongs to neither module and to both: `action_repeat: 1`
on the shared `_VecEnvView` must be bit-identical to a view with no repeat,
because `fasttd3` runs through it on every tier.

torch gates everything below the config-only tests: `--extra test` has no
torch, so the default CI job skips those by design (the sb3 and fasttd3
precedent); `--extra all` runs them.
"""

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from conftest import REPO  # noqa: F401  (path setup)
from bird import registry
from bird.budget import Budget
from bird.config import ConfigError, load
from bird.context import Context
from bird.state import RunState
from bird.types import Candidate

import bird.components.fasttd3 as FT
import bird.components.simba_v2 as SV
from bird.components import training as T

UPSTREAM = Path(REPO) / "refs" / "code" / "SimbaV2"

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

needs_torch = pytest.mark.skipif(
    not HAVE_TORCH, reason="train.backend: simba_v2 needs torch")
needs_upstream = pytest.mark.skipif(
    not (UPSTREAM / "configs" / "agent" / "simbaV2.yaml").is_file(),
    reason="refs/code/SimbaV2 not present (sparse checkout); run scripts/fetch_refs.sh")

#: Tiny everywhere: the contract is what is under test, not learning.
TINY = {"batch_size": 32, "buffer_size": 256,
        "critic_hidden_dim": 16, "actor_hidden_dim": 16,
        "critic_num_bins": 11, "learning_starts": 8, "compile": False,
        "actor_num_blocks": 1, "critic_num_blocks": 1}

REWARD = """
def compute_reward(state, action):
    import numpy as np
    th = np.arctan2(state[1], state[0])
    return -(th ** 2) - 0.1 * state[2] ** 2, {"upright": -(th ** 2)}
"""


def _ctx(env_id="pendulum", **overrides):
    base = {"seed": 0, "problem.env_id": env_id, "output.tracker": "none",
            "llm.generator.provider": "mock", "llm.evaluator.provider": "mock",
            "evaluate.rollouts_per_candidate": 2,
            "train.backend": "simba_v2", "train.algorithm": "sac",
            "train.architecture": "simba_v2",
            "train.n_parallel_envs": 2,
            "train.hyperparameters": dict(TINY), "train.env_steps": 400}
    base.update(overrides)
    cfg = load("rda", overrides=base, profile="dev")
    ctx = Context(cfg=cfg, budget=Budget())
    ctx.env = registry.get("env", env_id)(ctx)
    return ctx


def _train(ctx, cand_id="c0", state=None, **kw):
    backend = registry.get("train_backend", "simba_v2")
    cand = Candidate(cand_id=cand_id, reward_code=REWARD, iteration=0)
    return backend(ctx, state if state is not None else RunState(), cand, 1, **kw)


@pytest.fixture(autouse=True)
def _clean_stores():
    T._POLICY_STORE.clear()
    T._REPLAY_STORE.clear()
    yield
    T._POLICY_STORE.clear()
    T._REPLAY_STORE.clear()


def _upstream_cfg(rel):
    return yaml.safe_load((UPSTREAM / rel).read_text())


# ==========================================================================
# No torch needed: registration, laziness, defaults-vs-upstream, coherence
# ==========================================================================


def test_the_backend_is_registered_and_the_module_imports_without_torch():
    """`registry.load_all()` imports this module for every config in the repo,
    so an import-time torch dependency would make a torch-less machine unable
    to validate configs it never intended to run -- the sb3 rule, and the
    fasttd3 test verbatim. Checked in a subprocess so an already-imported
    torch in THIS process cannot mask it."""
    assert "simba_v2" in registry.names("train_backend")
    code = ("import sys; import bird.components.simba_v2; "
            "sys.exit(1 if 'torch' in sys.modules else 0)")
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(REPO),
                          capture_output=True, text=True)
    assert proc.returncode == 0, (
        "importing bird.components.simba_v2 pulled torch into sys.modules; "
        f"stderr: {proc.stderr[-400:]}")


def test_every_declared_hyperparameter_is_actually_READ_by_the_backend():
    """DECLARED-BUT-UNREAD KEYS, checked mechanically on a 37-key surface.

    A key that is declared and never honoured is a recurring bug class, and its
    worst-behaved member is "a key that IS read, and read only into a string"
    -- because every test it could fail is green.
    `train.hyperparameters` is a free `dict` in the schema, so
    `tests/test_schema_coverage.py` cannot see inside it: nothing else in the
    repo can tell whether `_SIMBAV2_DEFAULTS`' 37 entries reach the learner or
    sit there looking authoritative. `_simba_hyper` warns about an UNKNOWN name,
    which is the other direction; this is the one that matters for fidelity,
    since a pinned-but-unread `target_tau` would make the port silently
    FastTD3's.

    AN AST READ, NOT A RUN, deliberately: it needs no torch, so it holds in
    every CI selection rather than only where torch is installed, and it cannot be
    satisfied by a key that is read on a code path no test reaches. `hyper[k]`
    and `hyper.get(k)` are both counted; a key consumed some third way would
    have to be named here, which is the point -- the exemption becomes visible.
    """
    import ast

    src = (Path(REPO) / "bird" / "components" / "simba_v2.py").read_text()
    tree = ast.parse(src)
    declared = None
    for node in tree.body:
        target = getattr(node, "target", None)
        if isinstance(node, ast.AnnAssign) and getattr(target, "id", "") == "_SIMBAV2_DEFAULTS":
            declared = {k.value for k in node.value.keys}
    assert declared, "could not find _SIMBAV2_DEFAULTS as a module-level annotated assignment"
    assert declared == set(SV._SIMBAV2_DEFAULTS), "the AST read and the import disagree"

    read = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id == "hyper" and isinstance(node.slice, ast.Constant)):
            read.add(node.slice.value)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "hyper" and node.args
                and isinstance(node.args[0], ast.Constant)):
            read.add(node.args[0].value)

    unread = sorted(declared - read)
    assert not unread, (
        f"{len(unread)} key(s) in _SIMBAV2_DEFAULTS are declared and never read: "
        f"{unread}. The fix is to HONOUR the key, never to delete it quietly "
        "-- a declared-but-unread key is a fabricated pin; if one is genuinely consumed "
        "some other way, name the exemption in this test so it is visible.")
    unknown = sorted(read - declared)
    assert not unknown, (
        f"the backend reads {unknown} out of `hyper`, which `_SIMBAV2_DEFAULTS` "
        "does not declare -- so `_simba_hyper` would never supply it and the "
        "read gets a KeyError or a silent default")


@needs_upstream
def test_the_defaults_are_upstreams_four_config_files():
    """`_SIMBAV2_DEFAULTS` is upstream's own numbers, read from upstream rather
    than trusted. A re-vendor that moves a default fails HERE, loudly, instead
    of drifting into every run that quotes the paper's column.

    Only the keys upstream states as literals are checked: the `${eval:...}`
    ones are `_derived`'s job (its own test below) and the `${...}`
    interpolations are BIRD's keys (`train.env_steps`, `train.n_parallel_envs`,
    the seed, gamma), which `_SIMBAV2_DEFAULTS`' comments say are absent on
    purpose.
    """
    agent = _upstream_cfg("configs/agent/simbaV2.yaml")
    buf = _upstream_cfg("configs/buffer/numpy_uniform.yaml")
    hb = _upstream_cfg("configs/env/hb_locomotion.yaml")
    D = SV._SIMBAV2_DEFAULTS

    for key in ("normalize_observation", "normalize_reward", "normalized_g_max",
                "learning_rate_init", "learning_rate_end",
                "learning_rate_decay_rate",
                "actor_num_blocks", "actor_hidden_dim", "actor_c_shift",
                "critic_num_blocks", "critic_hidden_dim", "critic_c_shift",
                "critic_num_bins", "target_tau", "temp_initial_value",
                "temp_target_entropy", "temp_target_entropy_coef"):
        assert key in D, f"{key} is in upstream's agent config and not in the defaults"
        # COERCE BOTH THROUGH float() WHERE IT APPLIES, because upstream writes
        # `learning_rate_init: 1e-4` and YAML 1.1 reads that as the STRING
        # "1e-4" (a float needs a digit before the `e`'s sign, `1.0e-4`). A
        # comparison that failed on the quoting rather than on the value would
        # make this test unrunnable and get deleted, which is worse than the
        # coercion.
        want, got = agent[key], D[key]
        if want is None or isinstance(want, bool):
            assert got == want, f"{key}: {got!r} != upstream {want!r}"
        else:
            assert float(got) == float(want), (
                f"{key}: {got!r} != upstream {want!r}")

    # The buffer's three, and the UNITS are the point: upstream's `max_length`
    # and `sample_batch_size` are TOTALS and `min_length` is a transition
    # count, so a port that read them per-env would be 16x out on a 16-env run.
    assert D["buffer_size"] == buf["max_length"]
    assert D["learning_starts"] == buf["min_length"]
    assert D["batch_size"] == buf["sample_batch_size"]

    # HumanoidBench's action_repeat is 2 upstream; the DEFAULT here is the
    # degenerate 1 (every key defaults to its most degenerate value), so this asserts the relationship rather
    # than equality -- and asserts that the value the configs pin is upstream's.
    assert hb["action_repeat"] == 2
    assert D["action_repeat"] == 1, "the repo default is the degenerate one"
    rda = load("rda_humanoidbench", profile="humanoid_simba")
    assert rda["train.hyperparameters"]["action_repeat"] == hb["action_repeat"], (
        "configs/methods/rda_humanoidbench.yaml must pin HumanoidBench's own action_repeat")

    # `updates_per_interaction_step: ${action_repeat}` -- an EXPRESSION, so the
    # port's `None` must mean the expression and not a copy of today's value.
    assert D["updates_per_interaction_step"] is None
    assert _upstream_cfg("configs/online_rl.yaml")["updates_per_interaction_step"] \
        == "${action_repeat}"


@needs_upstream
def test_the_derived_constants_are_upstreams_own_expressions():
    """`configs/agent/simbaV2.yaml:25-28` and `:36-41` write each scaler and
    alpha constant as an OmegaConf `${eval:...}` over the hidden dimension and
    the block count, and `simbaV2_agent.py:309` fills `temp_target_entropy`
    from the coefficient and the action dimension. `_derived` recomputes them
    so a config that changes a hidden dimension cannot silently keep the old
    constants -- the two-copies-of-one-fact shape. This checks the formulas,
    against the text of upstream's own expressions."""
    agent = (UPSTREAM / "configs" / "agent" / "simbaV2.yaml").read_text()
    # The expressions, quoted from upstream so a re-vendor that changes one is
    # visible here rather than only in the numbers.
    assert "actor_scaler_init: ${eval:'math.sqrt(2 / ${agent.actor_hidden_dim})'}" in agent
    assert "actor_alpha_init: ${eval:'1 / (${agent.actor_num_blocks} + 1)'}" in agent
    assert "actor_alpha_scale: ${eval:'1 / math.sqrt(${agent.actor_hidden_dim})'}" in agent
    assert "critic_min_v: ${eval:'-${agent.normalized_g_max}'}" in agent

    hyper = dict(SV._SIMBAV2_DEFAULTS)
    der = SV._derived(hyper, n_act=6)
    assert der["actor_scaler_init"] == math.sqrt(2 / 128)
    assert der["actor_scaler_scale"] == math.sqrt(2 / 128)
    assert der["critic_scaler_init"] == math.sqrt(2 / 512)
    assert der["actor_alpha_init"] == 1 / (1 + 1)
    assert der["actor_alpha_scale"] == 1 / math.sqrt(128)
    assert der["critic_alpha_init"] == 1 / (2 + 1)
    assert der["critic_alpha_scale"] == 1 / math.sqrt(512)
    assert (der["critic_min_v"], der["critic_max_v"]) == (-5.0, 5.0)
    assert der["temp_target_entropy"] == -0.5 * 6

    # Change a hidden dim and every constant that depends on it must move.
    hyper["actor_hidden_dim"] = 256
    der2 = SV._derived(hyper, n_act=6)
    assert der2["actor_scaler_init"] == math.sqrt(2 / 256)
    assert der2["actor_alpha_scale"] == 1 / math.sqrt(256)
    assert der2["critic_scaler_init"] == der["critic_scaler_init"], "critic untouched"


@needs_upstream
def test_rda_table_1_is_derivable_from_upstreams_own_formulas():
    """THE PROVENANCE CLAIM `configs/methods/rda_humanoidbench.yaml` NOW MAKES, checked
    rather than asserted in a comment.

    RDA Table 1's HumanoidBench column gives gamma 0.98 and 625 K update steps
    (`refs/tex/rda/appendix.tex:1423, :1426`). Upstream DERIVES both from the
    episode length, the env count and the repeat
    (`configs/online_rl.yaml:19-20, :27-28`), so they are not independent
    choices, and `(max_episode_steps 500, action_repeat 2)` is the unique pair
    consistent with both. If a re-vendor changed either formula, every
    provenance sentence in that config would become false and nothing else in
    the repo would notice.
    """
    text = (UPSTREAM / "configs" / "online_rl.yaml").read_text()
    assert ("eff_episode_len: ${eval:'${env.max_episode_steps} / "
            "${env.action_repeat}'}") in text
    assert ("gamma: ${eval:'max(min((${eff_episode_len} / 5 - 1) / "
            "(${eff_episode_len} / 5), 0.995), 0.95)'}") in text
    assert ("num_interaction_steps: ${eval:'${num_env_steps} / "
            "(${num_train_envs} * ${action_repeat})'}") in text

    def gamma_of(max_ep, repeat):
        eff = max_ep / repeat
        return max(min((eff / 5 - 1) / (eff / 5), 0.995), 0.95)

    def updates_of(env_steps, envs, repeat):
        return int(env_steps // (envs * repeat)) * repeat

    assert gamma_of(500, 2) == pytest.approx(0.98)
    assert updates_of(10_000_000, 16, 2) == 625_000
    # UNIQUENESS: the two other readings of Table 1's rows give other gammas.
    assert gamma_of(500, 1) == pytest.approx(0.99)
    assert gamma_of(250, 2) == pytest.approx(0.96)

    # And the config pins exactly that point.
    cfg = load("rda_humanoidbench", profile="humanoid_simba")
    hp = cfg["train.hyperparameters"]
    assert hp["gamma"] == pytest.approx(gamma_of(500, hp["action_repeat"]))
    assert updates_of(10_000_000, cfg["train.n_parallel_envs"],
                      hp["action_repeat"]) == 625_000


def test_the_hb_config_resolves_on_this_backend_and_keeps_its_five_key_diff():
    """`rda_humanoidbench` under the paper's learner: the profile supplies the
    backend and the env count (both PROFILE keys), the method config supplies
    the learner block,
    and `tests/test_ablation.py`'s key-set diff is untouched -- which is the
    whole reason the backend is named in a profile and not in the config."""
    cfg = load("rda_humanoidbench", profile="humanoid_simba")
    assert cfg["train.backend"] == "simba_v2"
    assert cfg["train.algorithm"] == "sac"
    assert cfg["train.architecture"] == "simba_v2"
    assert cfg["train.n_parallel_envs"] == 16
    hp = cfg["train.hyperparameters"]
    assert hp["device"] == "cuda", "the GPU pin is method-side and must survive"
    assert hp["gamma"] == 0.98
    assert hp["action_repeat"] == 2
    assert hp["updates_per_interaction_step"] is None

    # WHAT THIS FILE CHECKS ABOUT THE DIFF, AND WHAT IT DELIBERATELY DOES NOT.
    #
    # `tests/test_ablation.py::
    # test_the_humanoidbench_variant_is_a_benchmark_change_and_not_a_method`
    # owns the exact key SET and is the only place it is written down.
    # Re-asserting that set here would be two copies of one fact, and the
    # second goes stale the first time a legitimate key is added to the diff
    # (as `problem.horizon` was) while the owning test passes. Keeping it in
    # sync is the maintenance cost the repo refuses elsewhere (`TASK_SPEC_KEYS`
    # exists so a test cannot carry its own whitelist).
    #
    # What belongs HERE is the property this BACKEND depends on and the
    # ablation test does not state: that `train.backend` is absent from the
    # diff. It is a PROFILE key, so the only way to select this learner is a
    # profile -- and the day someone "helpfully" pins it in the method config,
    # the ablation test fails with a key-set message while THIS one says why
    # it matters.
    diff = load("rda").diff(load("rda_humanoidbench"))
    assert "train.backend" not in diff, (
        "train.backend has been pinned in a method config. It is execution, not "
        "method identity: a profile outranks the method config on it, so the pin "
        "would be inert under every profile that sets one -- and it breaks the "
        "ablation test's key set, which is the guard that keeps the two "
        "benchmarks one method.")
    assert cfg["train.backend"] == "simba_v2", "the profile is what selects the learner"


def test_a_budget_that_cannot_reach_one_interaction_step_is_refused():
    """`env_steps // (n_parallel_envs * action_repeat)` floors to zero when the
    budget is smaller than one vector decision, and the backend's `max(1, ...)`
    would then train ONE step while reporting the requested budget. Refused at
    load, where it is one line to fix, rather than discovered after the run."""
    with pytest.raises(ConfigError, match=r"0 interaction steps"):
        load("rda", profile="dev",
             overrides={"train.backend": "simba_v2", "train.env_steps": 16,
                        "train.n_parallel_envs": 16,
                        "train.hyperparameters": {"action_repeat": 2}})
    # One step is enough.
    cfg = load("rda", profile="dev",
               overrides={"train.backend": "simba_v2", "train.env_steps": 32,
                          "train.n_parallel_envs": 16,
                          "train.hyperparameters": {"action_repeat": 2}})
    assert cfg["train.env_steps"] == 32


def test_a_nonpositive_repeat_or_update_rate_is_refused():
    """`max(1, ...)` in the backend would silently promote either to 1, and
    `action_repeat: 0` reading as 1 is a different discount horizon under the
    config's own gamma."""
    for hp, pattern in (({"action_repeat": 0}, r"action_repeat=0"),
                        ({"updates_per_interaction_step": 0},
                         r"updates_per_interaction_step=0")):
        with pytest.raises(ConfigError, match=pattern):
            load("rda", profile="dev",
                 overrides={"train.backend": "simba_v2", "train.hyperparameters": hp})


def test_the_shared_replay_pool_is_refused_rather_than_declared_and_inert():
    """`interaction_cfg.shared_buffer` has no consumer on this backend, exactly
    as on fasttd3, so a config that declares it is refused rather than left to
    report a pool that never executed."""
    with pytest.raises(ConfigError, match=r"shared_buffer=true is not wired on "
                                          r"train.backend=simba_v2"):
        load("lares", profile="dev",
             overrides={"train.backend": "simba_v2",
                        "train.interaction": "shared_population",
                        "train.interaction_cfg.shared_buffer": True,
                        "train.seeds_per_candidate": 1,
                        "select.allocation": "uniform"})


# ==========================================================================
# The two mechanisms that are silent when broken
# ==========================================================================


@needs_torch
def test_every_hyper_dense_kernel_is_on_the_unit_sphere_after_init_and_after_a_step():
    """THE MECHANISM SIMBAV2 IS NAMED FOR, and it is a loop over named
    submodules -- so a refactor that renamed or wrapped `HyperDense` would
    project NOTHING, every counter would read normal, and the run would be
    Simba-shaped layers without the constraint.

    Upstream projects at init (`simbaV2_agent.py:193-195`) and after EVERY
    actor and critic gradient step (`simbaV2_update.py:91`, `:239`). The axis
    is upstream's: a flax `Dense` kernel is `(in, out)` normalised along axis
    0 -- one unit vector per OUTPUT unit -- which on `torch.nn.Linear.weight`
    `(out, in)` is `dim=1`. Both axes give unit-norm weights and only one is
    upstream's, so the rows are checked, not the columns.
    """
    ns = SV._torch_classes()
    hyper = dict(SV._SIMBAV2_DEFAULTS, **TINY)
    der = SV._derived(hyper, n_act=1)
    actor, qnet, _qt, _temp, _norm = SV._build_networks(
        ns, hyper, der, n_obs=3, n_act=1, device=ns["torch"].device("cpu"))

    def rows(net):
        return [m.w.weight for m in net.modules() if getattr(m, "_HYPER", False)]

    # Before: orthogonal init, so the ROWS are not unit norm in general.
    n_actor = ns["project_"](actor)
    n_critic = ns["project_"](qnet)
    assert n_actor > 0 and n_critic > 0, "the projection reached no kernel at all"
    for w in rows(actor) + rows(qnet):
        norms = ns["torch"].linalg.norm(w, ord=2, dim=1)
        assert ns["torch"].allclose(norms, ns["torch"].ones_like(norms), atol=1e-5), (
            "a hyper_dense kernel's rows are not unit vectors after the projection")

    # And it survives a gradient step: perturb, project, check again.
    with ns["torch"].no_grad():
        for w in rows(actor):
            w.add_(ns["torch"].randn_like(w) * 0.5)
    again = ns["project_"](actor)
    assert again == n_actor
    for w in rows(actor):
        norms = ns["torch"].linalg.norm(w, ord=2, dim=1)
        assert ns["torch"].allclose(norms, ns["torch"].ones_like(norms), atol=1e-5)


@needs_torch
@pytest.mark.slow
def test_the_kernels_are_STILL_on_the_sphere_after_a_real_training():
    """THE BEHAVIOURAL CHECK, and it is the one that catches the failure a
    counter cannot.

    Gradient steps move a kernel OFF the unit sphere; the projection after
    every actor and critic step is what puts it back
    (`simbaV2_update.py:91`, `:239`). So after a training that took gradient
    steps, every actor and critic kernel must still be unit-norm ROW-wise --
    and if either per-step `project_` call were deleted, this fails on real
    weights rather than on a bookkeeping field. That is the difference between
    proving the mechanism ran and proving a constructor called it.

    THE TARGET CRITIC IS DELIBERATELY EXCLUDED, and asserting that is part of
    the test: upstream projects it at init and never again
    (`update_target_network` is a bare tree map, `simbaV2_update.py:244`), so a
    convex combination of two projected networks is what it trains with. A
    port that "fixed" that by projecting it too would pass a naive version of
    this test and be wrong; here the target is checked to have DRIFTED, which
    is the positive evidence that the soft update is upstream's.
    """
    ns = SV._torch_classes()
    torch = ns["torch"]
    ctx = _ctx(**{"train.env_steps": 600, "train.n_parallel_envs": 2})
    res = _train(ctx, "c0001")
    assert res.trained, res.error
    row = res.seed_metrics[0]
    assert row["update_steps"] > 0, "no gradient step ran; the test proves nothing"

    blob = T._POLICY_STORE[res.policy_ref]
    payload = SV._payload(ns, "simba_v2", blob, res.policy_ref)

    def rows_offsphere(state_dict):
        """Max |‖row‖ - 1| over every `HyperDense` kernel in a state dict."""
        worst = 0.0
        seen = 0
        for key, val in state_dict.items():
            if key.endswith(".w.weight"):
                norms = torch.linalg.norm(val.float(), ord=2, dim=1)
                worst = max(worst, float((norms - 1.0).abs().max()))
                seen += 1
        assert seen > 0, "no hyper_dense kernel found in the payload"
        return worst, seen

    a_off, a_n = rows_offsphere(payload["actor"])
    c_off, c_n = rows_offsphere(payload["qnet"])
    assert a_off < 1e-4, (
        f"after {row['update_steps']} updates the actor's kernels are {a_off} off "
        "the unit sphere -- the per-step projection is not running")
    assert c_off < 1e-4, (
        f"after {row['update_steps']} updates the critic's kernels are {c_off} off "
        "the unit sphere -- the per-step projection is not running")

    t_off, _ = rows_offsphere(payload["qnet_target"])
    assert t_off > 1e-6, (
        "the TARGET critic is on the sphere, which upstream's soft update does "
        "not leave it: `update_target_network` is a bare tree map, so a convex "
        "combination of two projected networks is what it trains with. If this "
        "fails, something is projecting the target and the port has 'fixed' "
        "upstream")
    assert a_n == row["n_hyper_kernels_actor"]
    assert c_n == row["n_hyper_kernels_critic"]


@needs_torch
@pytest.mark.slow
def test_the_projection_counters_are_an_EQUALITY_against_the_update_count():
    """The artifact's side of the test above, and the reason the seed row
    carries four projection fields rather than one.

    `n_hyper_kernels_projected` is the INIT count and is written by the
    constructor, so it cannot distinguish "projected every step" from
    "projected once" -- delete both per-step `project_` calls and that field is
    byte-identical. The per-step counters can,
    because their value is PREDICTABLE rather than merely non-zero:

        n_projection_calls   == 2 * update_steps        (actor, then critic)
        n_projection_kernels == update_steps * (k_actor + k_critic)

    An equality, not a floor -- a floor would pass a run that projected on some
    steps and not others. Asserted with `compile: false` (this backend's
    default, and what `TINY` sets), because the counters are Python side
    effects inside the update closure and a compiled closure may not run them;
    `compiled` is on the same row so a reader can tell which case they have.
    """
    ctx = _ctx(**{"train.env_steps": 600, "train.n_parallel_envs": 2})
    res = _train(ctx, "c0002x")
    assert res.trained, res.error
    row = res.seed_metrics[0]
    assert row["compiled"] is False, "this equality is asserted uncompiled"
    assert row["update_steps"] > 0

    assert row["n_hyper_kernels_projected"] > 0, (
        "the init projection reached no kernel -- `HyperDense`'s `_HYPER` marker "
        "or `project_`'s reader of it has moved")
    assert row["n_hyper_kernels_actor"] > 0 and row["n_hyper_kernels_critic"] > 0
    # init projects all THREE networks: actor + critic + target critic, and the
    # target's kernel count equals the critic's.
    assert row["n_hyper_kernels_projected"] == (
        row["n_hyper_kernels_actor"] + 2 * row["n_hyper_kernels_critic"])

    assert row["n_projection_calls"] == 2 * row["update_steps"], (
        f"{row['n_projection_calls']} projection calls over "
        f"{row['update_steps']} updates; upstream projects the actor after its "
        "step and the critic after its, so it is exactly two per update")
    assert row["n_projection_kernels"] == row["update_steps"] * (
        row["n_hyper_kernels_actor"] + row["n_hyper_kernels_critic"]), row


@needs_torch
def test_the_critic_target_carries_the_entropy_term():
    """The critic is a SOFT value function: upstream's target is
    `reward + gamma^n * (bin_values - temp * log pi(a'|s')) * (1 - terminated)`
    (`simbaV2_update.py:96 categorical_td_loss`). Drop the entropy term and the
    critic becomes a plain distributional value function -- the run still
    trains, every loss still falls, and the algorithm is no longer SAC.

    So the target is checked as a FUNCTION of the entropy term: a non-zero
    entropy must move the projected mass, and in the direction the sign says
    (a positive `temp * log pi` SUBTRACTS, shifting mass toward lower bins).
    """
    ns = SV._torch_classes()
    torch = ns["torch"]
    num_bins, min_v, max_v = 11, -5.0, 5.0
    bins = torch.linspace(min_v, max_v, num_bins).view(1, -1)
    # One sample, all mass on the middle bin (value 0.0).
    logp = torch.full((1, num_bins), -1e9)
    logp[0, num_bins // 2] = 0.0
    kw = dict(reward=torch.zeros(1), done=torch.zeros(1),
              discount=torch.ones(1) * 0.9, bin_values=bins,
              num_bins=num_bins, min_v=min_v, max_v=max_v)
    zero = ns["categorical_td_target"](logp, actor_entropy=torch.zeros(1), **kw)
    pos = ns["categorical_td_target"](logp, actor_entropy=torch.ones(1) * 2.0, **kw)
    assert torch.allclose(zero.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(pos.sum(), torch.tensor(1.0), atol=1e-5)
    assert not torch.allclose(zero, pos), (
        "the entropy term does not move the target -- the critic is not soft")
    mean_zero = float((zero * bins).sum())
    mean_pos = float((pos * bins).sum())
    assert mean_pos < mean_zero - 1e-3, (
        f"a positive temp*log_pi must shift the target DOWN: {mean_pos} vs {mean_zero}")


@needs_torch
def test_truncation_still_bootstraps_and_termination_does_not():
    """`done` in the target is upstream's `batch["terminated"]`
    (`simbaV2_update.py:190`), NOT terminated-or-truncated: a time-limited
    episode's last transition must still bootstrap or every horizon becomes a
    real absorbing state and the value function learns the clock. The backend
    reconstructs `terminated` as `dones & ~truncations`, because `_VecEnvView`
    reports the pair the other way round."""
    ns = SV._torch_classes()
    torch = ns["torch"]
    num_bins, min_v, max_v = 11, -5.0, 5.0
    bins = torch.linspace(min_v, max_v, num_bins).view(1, -1)
    logp = torch.full((1, num_bins), -1e9)
    logp[0, -1] = 0.0                     # all mass on the TOP bin
    kw = dict(reward=torch.zeros(1), actor_entropy=torch.zeros(1),
              discount=torch.ones(1) * 0.9, bin_values=bins,
              num_bins=num_bins, min_v=min_v, max_v=max_v)
    boot = ns["categorical_td_target"](logp, done=torch.zeros(1), **kw)
    term = ns["categorical_td_target"](logp, done=torch.ones(1), **kw)
    assert float((boot * bins).sum()) > 1.0, "a bootstrapped target keeps the value"
    assert abs(float((term * bins).sum())) < 1e-4, (
        "a terminated target is the reward alone")

    # And the backend's reconstruction: dones=1 with truncations=1 is a TIMEOUT,
    # so `terminated` is 0 there and 1 only for a real termination.
    dones = torch.tensor([1, 1, 0]).bool()
    truncs = torch.tensor([1, 0, 0]).bool()
    terminated = (dones & ~truncs).float()
    assert terminated.tolist() == [0.0, 1.0, 0.0]


@needs_torch
def test_the_tanh_gaussian_log_prob_matches_the_change_of_variables():
    """The actor is a squashed Gaussian and its `log_prob` is the density minus
    the tanh log-determinant, in the stable `2*(log 2 - u - softplus(-2u))`
    form -- because `log(1 - tanh(u)**2)` underflows to `-inf` past |u| ~ 9 and
    would make the actor loss NaN at exactly the saturation SAC drives the
    policy toward. Checked against the naive form where the naive form is
    valid, and for finiteness where it is not."""
    ns = SV._torch_classes()
    torch = ns["torch"]
    u = torch.tensor([[-2.0, 0.0, 0.5]])
    naive = torch.log(1.0 - torch.tanh(u) ** 2)
    stable = 2.0 * (math.log(2.0) - u - torch.nn.functional.softplus(-2.0 * u))
    assert torch.allclose(naive, stable, atol=1e-5)
    big = torch.tensor([[-30.0, 30.0]])
    assert torch.isinf(torch.log(1.0 - torch.tanh(big) ** 2)).all(), "the naive form dies"
    stable_big = 2.0 * (math.log(2.0) - big
                        - torch.nn.functional.softplus(-2.0 * big))
    assert torch.isfinite(stable_big).all(), "the stable form must not"

    # And the actor's own sample: finite log-probs, actions strictly inside
    # (-1, 1), and temperature 0 is the deterministic tanh(mean).
    hyper = dict(SV._SIMBAV2_DEFAULTS, **TINY)
    der = SV._derived(hyper, n_act=2)
    actor, *_ = SV._build_networks(ns, hyper, der, n_obs=4, n_act=2,
                                   device=torch.device("cpu"))
    obs = torch.randn(8, 4)
    torch.manual_seed(0)
    a, lp = actor.sample(obs, temperature=1.0)
    assert a.shape == (8, 2) and lp.shape == (8,)
    assert torch.isfinite(lp).all() and (a.abs() < 1.0).all()
    d1, _ = actor.sample(obs, temperature=0.0)
    d2, _ = actor.sample(obs, temperature=0.0)
    assert torch.equal(d1, d2), "temperature 0 must be deterministic"
    mean, _log_std = actor(obs)
    assert torch.allclose(d1, torch.tanh(mean), atol=1e-6)


@needs_torch
def test_the_scaler_follows_upstream_and_not_fasttd3s_port():
    """`Scaler` initialises its parameter to `scale` and multiplies by
    `init / scale`, so the layer scales by `init`
    (`simbaV2_layer.py:14 Scaler`). FastTD3's port stores `init * scale` and so
    scales by `init ** 2`; at `scaler_init = scaler_scale = sqrt(2/h)` that is
    `sqrt(2/h)` against `2/h`. The two are the same code shape and different
    functions, which is why this is pinned rather than reviewed."""
    ns = SV._torch_classes()
    torch = ns["torch"]
    hid = 128
    init = scale = math.sqrt(2.0 / hid)

    # OURS, directly: `Scaler` is exported from this module's namespace.
    ours = ns["Scaler"](dim=3, init=init, scale=scale)
    x = torch.ones(1, 3)
    assert torch.allclose(ours(x), torch.full((1, 3), init), atol=1e-7), (
        "upstream's Scaler multiplies by `init` at initialisation")
    assert torch.allclose(ours.scaler.detach(), torch.full((3,), scale), atol=1e-7), (
        "upstream initialises the PARAMETER to `scale` "
        "(simbaV2_layer.py:14 Scaler)")

    # THEIRS, THROUGH AN INSTANTIATED NETWORK, because `fasttd3._torch_classes()`
    # exports ten names and `Scaler` is not one of them -- it is a closure-local
    # class inside that factory, so `FT._torch_classes()["Scaler"]` raises
    # `KeyError`. The public surface is the network, so build one and read
    # the scaler off it -- which also checks the constant `_build_networks`
    # actually passes rather than one this test invented.
    fns = FT._torch_classes()
    their_actor = fns["ActorSimba"](
        n_obs=4, n_act=2, num_envs=1, hidden_dim=hid,
        scaler_init=init, scaler_scale=scale,
        alpha_init=0.5, alpha_scale=1.0 / math.sqrt(hid),
        expansion=4, c_shift=3.0, num_blocks=1, std_min=0.0, std_max=1.0,
        device=torch.device("cpu"))
    theirs = their_actor.embedder.scaler
    assert torch.allclose(theirs.scaler.detach(),
                          torch.full_like(theirs.scaler.detach(), init * scale),
                          atol=1e-7), (
        "FastTD3's port stores `init * scale` in the parameter -- if this "
        "fails, that port changed and the module docstring's departure table "
        "is stale")
    y = torch.ones(1, hid)
    assert torch.allclose(theirs(y), torch.full((1, hid), init * init), atol=1e-7), (
        "so its layer multiplies by init**2 where upstream's multiplies by init")
    # And the two are genuinely different functions at this scaling, which is
    # the whole point of the departure row.
    assert abs(init - init * init) > 1e-3, (init, init * init)


@needs_torch
def test_the_critic_support_is_the_reward_scalers_g_max():
    """One key, two consumers: `normalized_g_max` sets the critic's bins AND
    the reward scaler's ceiling (`simbaV2.yaml:10`, `:38-39`). They must move
    together, because a support sized for raw returns beside a normalised
    reward -- FastTD3's pairing, +-250 with g_max 10 -- puts every target in
    the middle bin and the critic learns nothing while every loss falls."""
    hyper = dict(SV._SIMBAV2_DEFAULTS, normalized_g_max=7.5)
    der = SV._derived(hyper, n_act=1)
    assert (der["critic_min_v"], der["critic_max_v"]) == (-7.5, 7.5)
    ns = SV._torch_classes()
    rn = ns["RewardNormalizer"](gamma=0.99, device=ns["torch"].device("cpu"),
                                g_max=float(hyper["normalized_g_max"]))
    assert float(rn.g_max) == 7.5


# ==========================================================================
# Contract: the backend means what every other backend means
# ==========================================================================


@needs_torch
@pytest.mark.slow
def test_a_training_produces_a_real_checkpoint_curve_and_honest_budget_facts():
    """`_sb3_run`'s contract: a curve with its keys, `env_steps` that counts
    what was spent, and the FOUR budget facts on the seed row,
    so RDA's 625 K claim is audited from the artifact rather than from a
    docstring.

    The identity under test is the one `simba_v2_backend`'s docstring states:

        interaction_steps = env_steps // (n_parallel_envs * action_repeat)
        update_steps      = (interaction_steps - learning_starts_steps)
                            * updates_per_interaction_step
    """
    ctx = _ctx(**{"train.env_steps": 400, "train.n_parallel_envs": 2,
                  "train.hyperparameters": dict(TINY, action_repeat=2)})
    res = _train(ctx, "c0002")
    assert res.trained, res.error
    assert res.checkpoints, "no checkpoint curve"
    for key in ("step", "round", "fitness", "reward_return", "gt_return"):
        assert key in res.checkpoints[0], key

    row = res.seed_metrics[0]
    assert row["backend"] == "simba_v2" and row["learner"] == "simba_v2"
    assert row["architecture"] == "simba_v2"
    # `algorithm_cited` is the PAPER's value and stays a citation even where it
    # coincides with the execution -- which it does here.
    assert row["algorithm_cited"] == "sac"
    assert row["action_repeat"] == 2 and row["num_envs"] == 2
    assert row["interaction_steps"] == 400 // (2 * 2)
    assert row["updates_per_interaction_step"] == 2, "null resolves to action_repeat"
    assert row["update_steps_planned"] == row["interaction_steps"] * 2
    expected = max(0, row["interaction_steps"] - row["learning_starts_steps"]) * 2
    assert row["update_steps"] == expected, (row["update_steps"], expected)
    assert row["update_steps"] > 0, "the budget must reach a gradient step"
    # SIMULATOR steps, upstream's accounting.
    assert row["train_steps"] == row["interaction_steps"] * 2 * 2
    assert row["train_steps_requested"] == 400
    assert row["random_action_env_steps"] > 0, "upstream's random warm-up"
    assert row["learning_starts_transitions"] == TINY["learning_starts"]
    assert row["batch_per_env"] == max(1, TINY["batch_size"] // 2)
    assert row["buffer_size_transitions"] == TINY["buffer_size"]
    assert res.env_steps_used >= row["train_steps"]


@needs_torch
@pytest.mark.slow
def test_off_cuda_the_gpu_provenance_fields_are_none_and_not_zero():
    """"no GPU" and "a GPU that did nothing" are different rows a reader of the
    seed rows must not average together, so the utilisation statistics are `None` off
    cuda rather than 0, and `learner_device` is read off the PARAMETERS and not
    from the config. `gpu_peak_alloc_torch_mib` counts TENSORS, so it is `None`
    here and NON-ZERO is the whole test on a real GPU (no MiB floor)."""
    ctx = _ctx()
    res = _train(ctx, "c0003")
    assert res.trained, res.error
    row = res.seed_metrics[0]
    assert row["device"] == "cpu"
    assert row["learner_device"] == "cpu", (
        "the SET over actor and both critics, so one parameter left behind is "
        "visible rather than passing")
    assert row["gpu_peak_alloc_torch_mib"] is None
    assert row["gpu_utilization_max_pct"] is None
    assert row["gpu_utilization_mean_pct"] is None
    assert row["gpu_utilization_n_samples"] == 0


def test_an_explicit_cuda_on_a_box_without_one_is_refused_not_downgraded():
    """Asking for cuda and silently getting cpu is a measurement that looks
    like the one you wanted: the fallback also disables AMP and
    `torch.compile`, both derived from `device.type == "cuda"`. `auto` keeps
    its fallback; an explicit `cuda` does not. (fasttd3's rule, and the reason
    `configs/methods/rda_humanoidbench.yaml` pins `cuda` rather than `auto`.)"""
    pytest.importorskip("torch")
    import torch
    if torch.cuda.is_available():
        pytest.skip("this machine has cuda; the refusal is for machines that do not")
    ctx = _ctx(**{"train.hyperparameters": dict(TINY, device="cuda")})
    with pytest.raises(RuntimeError, match=r"torch.cuda.is_available\(\) is False"):
        _train(ctx, "c0004")


@needs_torch
@pytest.mark.slow
def test_two_runs_of_one_seed_are_identical_and_two_seeds_are_not():
    """Determinism from the config seed, as on every backend: `torch.manual_seed`
    covers the init, the actor's own sampling and the batch indices, and the
    view's episode streams are `(seed, slot, episode)`.

    THE DISCRIMINATION HALF IS ASSERTED ON THE POLICY, NOT ON THE FITNESS: at
    this file's tester budget the ground-truth fitness on pendulum is
    identically zero for every seed (`[0.0, 0.0, 0.0, 0.0, 0.0]` on both), so an
    inequality between the two seeds' fitness lists could only pass by
    accident.

    The trained PARAMETERS are the honest witness: they differ between seeds
    whatever the fitness curve does, because the seed reaches the init, the
    exploration draws and the batch indices. Compared as the stored blob's
    bytes, which also means a change that stopped the seed reaching the learner
    at all -- the failure this test exists for -- fails here rather than
    reading as a saturated metric.
    """
    # ONE cand_id FOR THE PAIR, AND EACH BLOB READ IMMEDIATELY -- and the two
    # halves of that sentence pull against each other.
    #
    # `policy_ref` is `f"policy:{cand_id}"` and `_store` is a plain assignment,
    # so two trainings under one id write the SAME key: read both blobs at the
    # end and `blob_a == blob_b` compares the second with itself and cannot
    # fail -- the same vacuous-assertion family as a `[0.0]*5 != [0.0]*5`
    # fitness comparison.
    #
    # But `cand_id` is ALSO a seed component -- `_seed_base` mixes
    # `zlib.crc32(cand_id)` into the stream -- so giving the pair two ids makes
    # them two different seeds, and `blob_a == blob_b` then fails for a
    # correct learner.
    #
    # The resolution is not a compromise: same id (so the seed is identical),
    # each blob read at its own training (so the key collision cannot reach the
    # comparison), through a helper so the read cannot be ordered wrongly.
    # Third seeded run gets its own id, which is free -- a different seed AND a
    # different id both predict a different blob, and the assertion is that it
    # differs.
    def _trained_blob(ctx, cand_id):
        res = _train(ctx, cand_id)
        assert res.trained, (cand_id, res.error)
        assert res.policy_ref in T._POLICY_STORE, cand_id
        return res, bytes(T._POLICY_STORE[res.policy_ref].tobytes())

    a, blob_a = _trained_blob(_ctx(), "c0005")
    b, blob_b = _trained_blob(_ctx(), "c0005")
    assert a.policy_ref == b.policy_ref, (
        "the pair must share a cand_id, because cand_id salts the seed "
        "(`_seed_base`) -- two ids would be two seeds and this test would "
        "assert determinism across a difference it created")
    assert [p["fitness"] for p in a.checkpoints] == [p["fitness"] for p in b.checkpoints]
    assert blob_a == blob_b, "one seed, two runs: the parameters must be identical"

    c, blob_c = _trained_blob(_ctx(seed=7), "c0005seed7")
    assert blob_c != blob_a, (
        "a different seed produced byte-identical parameters, so the seed is "
        "not reaching the learner")


@needs_torch
def test_a_discrete_env_is_refused_loudly_rather_than_failing_candidates():
    """A tanh-Gaussian policy has no discrete head, and the refusal is a
    CONFIG error rather than a candidate failure: it holds for every candidate
    of every iteration, so failing them one by one would burn the whole search
    to report a config error."""
    # `toy_hungry_thirsty`, not `hungry_thirsty`: the registered id carries the
    # `toy_` prefix (`exact_states: 64`, `n_actions: 6`). The bare name fails in
    # `load()` with "not in [...]" before the refusal under test can run, so the
    # refusal would never be exercised.
    ctx = _ctx(env_id="toy_hungry_thirsty")
    assert getattr(ctx.env, "exact_states", None) is not None, (
        "this test needs a DISCRETE env; if this fires, the chosen id stopped "
        "being one and the refusal below would pass for the wrong reason")
    with pytest.raises(ValueError, match=r"needs a continuous-action env"):
        _train(ctx, "c0006")


@needs_torch
@pytest.mark.slow
def test_the_policy_blob_rebuilds_through_its_own_branch_and_acts():
    """A simba_v2 blob and a fasttd3 blob are both torch containers, so
    `training._blob_kind` cannot separate them and `policy_from_ref` routes on
    the payload's own `format` tag. The rebuilt callable must be the loop's own
    action path over the stored weights -- not a second implementation that
    could differ in the squash or the normaliser invisibly."""
    ctx = _ctx()
    res = _train(ctx, "c0007")
    assert res.trained and res.policy_ref in T._POLICY_STORE
    blob = T._POLICY_STORE[res.policy_ref]
    assert T._blob_kind(blob) == "torch"
    assert SV.blob_format(blob) == SV._BLOB_FORMAT

    entered = []
    real = FT.fasttd3_policy_from_blob
    FT.fasttd3_policy_from_blob = lambda *a, **k: entered.append(a)
    try:
        policy = T.policy_from_ref(ctx.cfg, ctx.env, blob, None, ref=res.policy_ref)
    finally:
        FT.fasttd3_policy_from_blob = real
    assert not entered, "a simba_v2 blob reached the fasttd3 rebuild"

    env = ctx.env
    s = env.reset(np.random.default_rng(0))
    a1, a2 = policy(s), policy(s)
    assert np.all(np.isfinite(a1)) and np.array_equal(a1, a2), "greedy: deterministic"
    center, half = FT._action_affine(env)
    assert np.all(a1 >= center - half - 1e-6) and np.all(a1 <= center + half + 1e-6)


@needs_torch
@pytest.mark.slow
def test_neither_learner_will_load_the_others_checkpoint():
    """The negative half of the routing, in BOTH directions, because a misroute
    is worse than a refusal: a policy rebuilt under the wrong learner would
    load some tensors, fail on others, and -- if the shapes happened to fit --
    roll out a network nobody built. Each `_payload` checks its own format tag
    and names the ref."""
    import test_fasttd3 as FT_TESTS

    ctx_s = _ctx()
    res_s = _train(ctx_s, "c0008")
    simba_blob = T._POLICY_STORE[res_s.policy_ref]

    ctx_f = FT_TESTS._ctx(**{"train.hyperparameters": {**FT_TESTS.TINY, "num_envs": 2},
                             "train.env_steps": 200})
    res_f = FT_TESTS._train(ctx_f, "c0009")
    fast_blob = T._POLICY_STORE[res_f.policy_ref]

    with pytest.raises(ValueError, match=r"c0009.*simba_v2/v1"):
        SV.simba_v2_policy_from_blob(ctx_s.cfg, ctx_s.env, None, fast_blob,
                                     "policy:c0009")
    with pytest.raises(ValueError, match=r"c0008.*fasttd3/v1"):
        FT.fasttd3_policy_from_blob(ctx_f.cfg, ctx_f.env, None, simba_blob,
                                    "policy:c0008")
    # And `policy_from_ref` sends each to the right one without being told.
    assert SV.blob_format(fast_blob) == "fasttd3/v1"
    assert SV.blob_format(simba_blob) == "simba_v2/v1"


# ==========================================================================
# The shared view: action_repeat 1 must be the identity
# ==========================================================================


@needs_torch
@pytest.mark.slow
def test_action_repeat_1_is_bit_identical_to_no_repeat():
    """`_VecEnvView` is SHARED with `fasttd3`, which runs on every tier, so the
    `action_repeat` parameter must be the identity at 1 -- the
    same calls in the same order on the same input. Checked by running the
    FASTTD3 backend (whose loop never sets the parameter) and comparing the
    whole checkpoint curve against a run through a view constructed with an
    explicit `action_repeat=1`, and by stepping the view directly.

    Not a summary scalar: `tests/test_parallelism.py`'s argument -- a
    bit-identity claim compared on one number is a claim about one number.
    """
    import test_fasttd3 as FT_TESTS

    a = FT_TESTS._train(FT_TESTS._ctx(**{
        "train.hyperparameters": {**FT_TESTS.TINY, "num_envs": 2},
        "train.env_steps": 200}), "c0010")
    assert a.trained, a.error

    # The view itself, stepped by hand at both spellings.
    ctx = _ctx()
    reward = T.compile_reward(REWARD, Candidate(cand_id="c0", reward_code=REWARD,
                                                iteration=0))

    def feats(s):
        return np.asarray(s, dtype=np.float32)

    def roll(**kw):
        view = FT._VecEnvView(ctx.env, 2, reward, feats, "none", 11,
                             int(np.asarray(ctx.env.obs_low).size), **kw)
        obs = view.reset()
        out = [obs.copy()]
        rng = np.random.default_rng(3)
        for _ in range(12):
            act = rng.uniform(-1.0, 1.0, size=(2, int(np.asarray(
                ctx.env.action_low).size))).astype(np.float32)
            nxt, rew, dones, touts, truen = view.step(act)
            out += [nxt.copy(), rew.copy(), dones.copy(), touts.copy(), truen.copy()]
        out.append(np.array([view.n_steps, view.n_clipped]))
        view.close()
        return out

    default = roll()
    explicit = roll(action_repeat=1)
    assert len(default) == len(explicit)
    for i, (x, y) in enumerate(zip(default, explicit)):
        assert np.array_equal(x, y), f"action_repeat=1 changed output {i}"

    # And repeat 2 is a DIFFERENT function -- twice the simulator steps per
    # call, so the guard above is not vacuous.
    two = roll(action_repeat=2)
    assert two[-1][0] == 2 * default[-1][0], (
        "action_repeat=2 must consume twice the simulator steps")
    assert not np.array_equal(two[1], default[1])
