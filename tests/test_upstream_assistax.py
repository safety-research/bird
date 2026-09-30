"""`bird/envs/upstream_assistax.py` -- upstream Assistax's own jax env classes.

THE OFFLINE HALF IS THE GATE TESTS, and they are the point of this file. The
consumer gate on this tier is the only thing standing between a candidate and
upstream's paid reward, and it can fail silently while looking healthy in
two ways: the spec PREEMPTS the factory attribute (so the tier's own bans are
never applied unless the spec carries them), and a walk that collects the CPU
tier's row callables can stop before a helper nested inside a container value.
Both are invisible to every test that asserts a pasted subset is banned.

So these tests ENUMERATE and compare rather than sample. A test that asserts
"reward_source is banned" cannot fail on the symbol nobody remembered.
"""
from __future__ import annotations

import pytest

from bird import config as bird_config
from bird import registry, tasks
from bird.envs import assistax as cpu
from bird.envs import upstream_assistax as up

ENV_ID = "upstream_assistax_scratchitch"


def _factory():
    registry.load_all()
    return registry.get("env", ENV_ID)


def _spec_gate():
    return set(bird_config.forbidden_symbols_of(tasks.index()[ENV_ID]))


# -- the gate ---------------------------------------------------------------

def test_the_spec_gate_is_a_superset_of_the_adapter_tuple():
    """The spec must carry the tuple, because the SPEC is what is consulted.

    `config.py:383` sets `gate = forbidden_symbols_of(spec)` and falls through
    to the factory attribute only `if not gate:`. A derived spec supplies its
    parent's gate, so without the union at derivation this tier's own bans are
    dead code -- present on the factory, never applied. This asserts the
    direction that actually matters.
    """
    missing = sorted(set(_factory().consumer_forbidden_symbols) - _spec_gate())
    assert not missing, (
        f"{len(missing)} symbols are on the adapter tuple but NOT in the spec, "
        f"so they gate nothing: {missing[:12]}. Re-run derive_jax_spec.py with "
        f"--union-adapter-gate {ENV_ID}.")


def test_every_cpu_row_callable_is_banned_including_the_nested_ones():
    """ENUMERATED from `_TASKS`, at any depth -- never a pasted subset.

    The nested ones are the reason this is written as a walk. A row value that
    holds a tuple, list or dict of helpers defeats a collector that tests each
    row value for callability, which finds none of them. A pasted list would
    also have passed a test that pasted the same list.

    The five rows carry no named helper nested in a container today, so the
    recursion is proven on a synthetic row rather than on `_TASKS`: the walk
    must stay a walk so that the first row to nest a helper is gated too.
    """
    def walk(obj, depth=0):
        if depth > 6:
            return []
        if callable(obj):
            name = getattr(obj, "__name__", "")
            return [name] if name.isidentifier() else []
        if isinstance(obj, dict):
            obj = list(obj.values())
        if isinstance(obj, (list, tuple, set, frozenset)):
            return [n for item in obj for n in walk(item, depth + 1)]
        return []

    def _helper_a():
        pass

    def _helper_b():
        pass

    probe = {"helpers": (_helper_a,), "by_name": {"j": [_helper_b]}, "scene": "x.xml"}
    assert {n for v in probe.values() for n in walk(v)} == {"_helper_a", "_helper_b"}, (
        "the walk does not reach callables nested in a row's containers; the "
        "recursion is not recursing")

    names = {n for row in cpu._TASKS.values() for v in row.values() for n in walk(v)}
    assert names, "the walk found no callables at all -- it is not walking"
    assert names <= _spec_gate(), sorted(names - _spec_gate())[:12]


def test_the_gate_holds_no_unreferenceable_names():
    """`<lambda>` is not a symbol a candidate can write.

    Banning it gates nothing and inflates the count, which matters because the
    count is the number two independent walks are compared on.
    """
    bad = [n for n in _factory().consumer_forbidden_symbols if not n.isidentifier()]
    assert not bad, bad


def test_the_adapter_declares_cuda_and_not_batched():
    """Two different questions; `jax_base.py:150-162` records why they are not collapsed."""
    f = _factory()
    assert f.requires_cuda is True
    assert f.batched is False


def test_the_spec_resolves_to_this_env_id():
    """`_BIRD_ID_RULES['assistax_upstream']`, and the spec dir name must match."""
    spec = tasks.index()[ENV_ID]
    assert spec.bird_env_id == ENV_ID
    assert spec.env["env_id"] == "scratchitch"


# -- measured against the installed package ---------------------------------

@pytest.mark.jax
def test_homogenisation_does_not_move_the_row():
    """The candidate's observation is the physics row, not upstream's obs dict.

    Asserted rather than reasoned about. The adapter constructs homogenised
    because the zoo was trained that way, and homogenisation DOES move the
    per-agent spaces (robot 29 -> 42, human act 3 -> 7 on scratchitch). The
    claim that it cannot move `concat(qpos, qvel, slots, tail)` is exactly the
    kind that turns out to be false, so it is measured: identical arrays, zero
    tolerance.
    """
    pytest.importorskip("assistax")
    import numpy as np

    cls = _factory().adapter_cls
    rows = {}
    for hm in ("max", None):
        sub = type(cls.__name__, (cls,), {"homogenisation_method": hm})
        env = sub(None)
        rng = np.random.default_rng(0)
        s = env.reset(rng)
        for _ in range(5):
            s, _done, _info = env.step(s, np.zeros(env.obs_low.shape[0] and 7))
        rows[hm] = np.asarray(s)
    assert rows["max"].shape == rows[None].shape
    assert np.array_equal(rows["max"], rows[None])


@pytest.mark.jax
def test_gate_covers_the_measured_metric_and_info_keys():
    """The gate vs the env it gates -- the comparison nothing was making.

    A hand-written component list can hold 15 of the 16 real `state.metrics`
    names and miss `reward_scratching`, the component of the only task this
    tier adapts, while every test stays green.
    """
    pytest.importorskip("assistax")
    import jax

    env = _factory().adapter_cls(None)
    _obs, st = env._env.reset(jax.random.PRNGKey(0))
    gate = _spec_gate()
    leaked = sorted((set(st.metrics) | {k for k in st.info if not k.startswith("first_")})
                    - gate)
    assert not leaked, f"reachable channels not in the gate: {leaked}"


@pytest.mark.jax
def test_render_returns_a_frame_rather_than_raising(gl):
    """`output.video.record: all` runs this on every candidate, not just the best.

    Renders through MuJoCo, so it requests the `gl` fixture (deselect with
    `-m "not gl"` on a box with no GL context; `scripts/setup_gl.sh`)."""
    pytest.importorskip("assistax")
    import numpy as np

    env = _factory().adapter_cls(None)
    frame = env.render(env.reset(np.random.default_rng(0)))
    assert frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8


# -- the zoo partner --------------------------------------------------------

def test_partners_without_a_zoo_path_are_refused():
    """Half-configured must not silently become the zero partner.

    THE DIRECTION THAT FAILS THIS WAY IS PARTNERS-WITHOUT-A-PATH, and that
    asymmetry is deliberate rather than a weakening. The forward case -- a
    `zoo_path` with no `zoo_partners` -- is SATISFIED instead: the adapter
    derives the split from the zoo itself (`split_partners`, deterministic in
    `seed`), which is what lets `BIRD_ZOO_PATH` alone configure a run.
    Requiring both to be written by hand would fail every run that sets
    only the path.

    The reverse has no such derivation available -- partners name uuids in a
    zoo whose location is unknown -- so it still refuses, and the refusal is
    still the one that stops a half-configured env becoming the passive
    human by default.
    """
    cls = _factory().adapter_cls
    sub = type(cls.__name__, (cls,), {"zoo_partners": {"IPPO": {"human": ["u0"]}}})
    with pytest.raises(ValueError, match="must be set together"):
        sub(None)


@pytest.mark.jax
def test_the_jax_row_matches_the_host_row_column_by_column():
    """The learner's row and the artifact's row must be the SAME function.

    PER COLUMN AND ON A STEPPED SEQUENCE, which is the whole point. A single
    `max(abs(host - jax))` over reset rows cannot discriminate ANY field that
    is zero at reset -- every quantity differenced or integrated between
    consecutive states. It reports 0.0 even against a `row_jax` that
    hardcodes `tool_vel = zeros`, which makes `tool_speed` permanently 0 on
    the row the learner is paid on and correct on every row the artifact
    records.

    A row-wide max also hides a wrong column with a small true value under a
    correct column with a large one, so the comparison is per column and the
    failure names the column. The instance is `tool_vel`; the class is
    "differenced fields are zero at reset".
    """
    pytest.importorskip("assistax")
    import numpy as np

    env = _factory().adapter_cls(None)
    names = ([f"qpos[{i}]" for i in range(int(env._env.sys.q_size()))]
             + [f"qvel[{i}]" for i in range(int(env._env.sys.qd_size()))]
             + [f"slot[{i}]" for i in range(up._N_SLOTS.get(env.task, 0))]
             + [t[0] for t in env._tail])
    rng = np.random.default_rng(0)
    host = env.reset(rng)
    rows = [(np.asarray(host), np.asarray(env.row_jax(env._last_state)), 0)]
    for k in range(1, 4):
        prev_state = env._last_state
        host, _done, _info = env.step(host, np.clip(rng.normal(0, 0.3, 7), -1, 1))
        rows.append((np.asarray(host),
                     np.asarray(env.row_jax(env._last_state, prev_state)), k))

    assert len(names) == env.obs_dim, (len(names), env.obs_dim)
    # PER COLUMN, and to float32 tolerance rather than exactly. The host row
    # is float64 and the device row is float32 (jax's default), so exact
    # equality fails on ~8-significant-figure agreement -- a dtype
    # difference, not a formula difference. The tolerance is tight enough
    # that a ZEROED column still fails loudly (0.0 vs 0.035 is not close),
    # which is the defect this test exists for, and per-column so a wrong
    # small value cannot hide under a correct large one.
    bad = [f"step {k} col {i} ({names[i]}): host={h[i]!r} jax={j[i]!r}"
           for h, j, k in rows for i in range(env.obs_dim)
           if not np.isclose(h[i], j[i], rtol=1e-6, atol=1e-7)]
    assert not bad, "host and device rows disagree:\n  " + "\n  ".join(bad[:12])


@pytest.mark.jax
def test_a_differenced_column_cannot_be_silently_zeroed():
    """The mutant a row-wide reset check cannot kill.

    Zeroing the tool-velocity columns is exactly the defect the per-column
    check exists for, so the check is exercised against it directly rather
    than trusted. Pinned on the tail's own names, not on indices, so a layout
    change moves the test with the row.
    """
    pytest.importorskip("assistax")
    import numpy as np

    env = _factory().adapter_cls(None)
    tail_names = [t[0] for t in env._tail]
    vel_cols = [env.obs_dim - len(tail_names) + tail_names.index(n)
                for n in ("tool_vx", "tool_vy", "tool_vz")]
    rng = np.random.default_rng(0)
    host = env.reset(rng)
    prev_state = env._last_state
    host, _d, _i = env.step(host, np.clip(rng.normal(0, 0.3, 7), -1, 1))
    good = np.asarray(env.row_jax(env._last_state, prev_state))
    # the real row has a non-zero tool velocity after a step ...
    assert np.any(good[vel_cols] != 0.0), good[vel_cols]
    # ... so the zeroed variant, which is what `prev_st=None` produces, must
    # differ from the host row in precisely those columns.
    zeroed = np.asarray(env.row_jax(env._last_state))
    assert np.all(zeroed[vel_cols] == 0.0)
    assert not np.array_equal(zeroed, np.asarray(host))


@pytest.mark.jax
def test_row_jax_traces_under_jit():
    """The property that makes it usable at all; `_row` fails this."""
    pytest.importorskip("assistax")
    import jax
    import numpy as np

    env = _factory().adapter_cls(None)
    env.reset(np.random.default_rng(0))
    assert jax.jit(env.row_jax)(env._last_state).shape == (env.obs_dim,)


def test_upstream_env_kwargs_cannot_be_mutated_in_place():
    """Shared by five subclasses; an in-place edit would change all of them."""
    with pytest.raises(TypeError):
        _factory().adapter_cls.upstream_env_kwargs["ctrl_cost_weight"] = 99


def test_upstream_env_kwargs_are_upstreams_cited_values():
    """`config/ippo.yaml` at a7d94f4e -- ctrl_cost_weight 0, NOT the 1e-6 default."""
    kw = _factory().adapter_cls.upstream_env_kwargs
    assert kw["ctrl_cost_weight"] == 0
    assert kw["backend"] == "mjx"
    assert kw["het_reward"] is False
    assert kw["episode_length"] == 1000


@pytest.mark.jax
def test_upstream_never_writes_the_pref_reward_into_state_reward():
    """The premise the whole gt_return formula rests on, AT NONZERO PREF.

    `LoadAgentWrapper.step` adds the partner's preference reward to the
    returned rewards DICT (`aht.py:881`) and replaces `metrics` (`:882`); its
    only state write is that metrics replacement, so `State.reward` stays
    task-only. Everything about which expression `gt_return` may use follows
    from that, so it is pinned by a test rather than by a reading of upstream.

    AN UNBOUND ENV CANNOT TEST IT. Asserting `from_all - pref == from_state -
    pref` on an UNBOUND env cannot fail: the `pref` term cancels algebraically,
    and with no partner loaded it is 0.0 anyway, so it reduces to
    `state.reward == rew["__all__"]` on the one configuration where nothing
    adds anything. The divergence occurs only WITH a partner -- the same shape
    as a row comparison over reset rows, where every differenced field is
    legitimately zero.

    So this drives a real `LoadAgentWrapper` with real stacked `pref_configs`.
    No zoo files are needed: `__init__` takes `load_agents` directly, and only
    `load_from_zoo` reads the disk. The partner population must still EXIST --
    the wrapper indexes `pref_configs["human"]` by the sampled partner, so an
    empty `load_agents` raises `KeyError: 'human'` at `reset` -- so a
    deterministic stand-in network (zero-mean, near-zero-std actions) plays
    the partner, and the preference block carries the three reward-side fields
    `compute_preference_reward` reads beside the seven observation fields.
    Every preferred range CONTAINS zero, so a near-still arm is in range and
    the preference reward is nonzero by construction: with a degenerate range
    (min == max) upstream's gaussian is 0 everywhere, the pref term vanishes,
    and the assertion below would cancel exactly as the unbound form does.
    """
    pytest.importorskip("assistax")
    import jax
    import jax.numpy as jnp
    import numpy as np
    from assistax.wrappers.aht import LoadAgentWrapper, LoadNetworkState

    import assistax

    from bird.components.assistax_ppo import PREF_OBS_FIELDS

    base = assistax.make("scratchitch", homogenisation_method="max")
    n_partners = 4
    pref = {"human": {f: jnp.linspace(0.2, 0.8, n_partners) for f in PREF_OBS_FIELDS}}
    pref["human"].update(speed_range_min=jnp.zeros(n_partners),
                         speed_range_max=jnp.ones(n_partners),
                         force_range_min=jnp.zeros(n_partners),
                         force_range_max=jnp.ones(n_partners),
                         reward_budget=jnp.full(n_partners, 1.0),
                         overall_weight=jnp.full(n_partners, 1.0),
                         touch_threshold=jnp.full(n_partners, 0.5))
    act_dim = int(base.action_space("human").shape[0])

    class _PartnerOut:
        def __init__(self, pi, hstate):
            self.pi, self.hstate = pi, hstate

    def _partner_apply(params, hstate, _inputs):
        mean = jnp.zeros((n_partners, act_dim))
        return _PartnerOut((mean, jnp.full((n_partners, act_dim), 1e-3)), hstate)

    partner = LoadNetworkState(apply_fn=_partner_apply, params=jnp.zeros((n_partners,)),
                               pop_size=n_partners)
    wrapped = LoadAgentWrapper(base, {"IPPO": {"human": partner}}, pref_configs=pref,
                               agents_expect_pref_obs=True)

    _o, st = wrapped.reset(jax.random.PRNGKey(0))
    acts = {a: jnp.zeros(base.action_space(a).shape) for a in wrapped.agents}
    _o2, st2, rew, _d, _i = wrapped.step(jax.random.PRNGKey(1), st, acts)

    inner = getattr(st2, "_state", st2)
    paid = float(np.asarray(rew["__all__"]).reshape(()))
    state_reward = float(np.asarray(inner.reward).reshape(()))
    r_pref = float(np.asarray(inner.metrics["total_pref_reward"]).reshape(()))

    assert r_pref != 0.0, (
        "the preference reward is zero, so `paid - pref == paid` and the premise "
        "below cannot discriminate; the fixture's preferred ranges no longer "
        "contain this step's speed and force")
    # The premise. If upstream ever starts writing the augmented reward into
    # State.reward, this fails and every gt_return expression must be re-read.
    assert state_reward == pytest.approx(paid - r_pref, abs=1e-5), (
        f"State.reward ({state_reward}) is not task-only: paid {paid}, "
        f"pref {r_pref}. `state.reward - pref` would now double-subtract.")


@pytest.mark.jax
def test_the_prompt_shows_the_candidate_the_observation_it_writes_against():
    """The rendered prompt must carry the spec's fields, not an empty table.

    An adapter that subclasses the plain `EnvAdapter` and never applies its
    spec renders "State variables, in order:" followed by nothing from
    `describe()`, and `generation.py` builds the prompt from exactly that.
    Every LLM candidate on this tier would then be written blind -- against a
    357-symbol gate protecting an observation the model was never shown --
    while the spec's `state_surface.flat_fields` holds all 88 entries unread.

    Asserted against the SPEC's count rather than a literal 88, so the test
    follows the layout instead of pinning a number that would have to be
    edited in two places. Nothing else compares the rendered prompt to the
    spec it claims to describe.
    """
    pytest.importorskip("assistax")
    from bird import tasks as _t

    env = _factory().adapter_cls(None)
    spec = _t.index()[ENV_ID]
    n_state = len((spec.raw.get("state_surface") or {}).get("flat_fields") or [])
    n_action = len(((spec.raw.get("env") or {}).get("spaces") or {}).get("action_fields") or [])
    assert n_state and n_action, "the spec itself carries no field table"

    assert len(env._state_fields) == n_state == env.obs_dim
    assert len(env._action_fields) == n_action == env.action_dim

    text = env.describe("state_fields_only")
    for name, _doc in list(env._state_fields)[:3] + list(env._state_fields)[-3:]:
        assert name in text, f"{name} missing from the rendered prompt"
    assert "s[0]" in text and f"a[{env.action_dim - 1}]" in text
