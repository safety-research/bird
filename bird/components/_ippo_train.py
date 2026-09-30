"""The `assistax_ppo` training call: upstream's IPPO, BIRD's reward.

Separated from `assistax_ppo.py` so that module stays the registration, and
so this half -- which reaches upstream's trainer and therefore jax -- is
imported only on the one path that runs a training.

WHAT IS UPSTREAM'S AND WHAT IS OURS, because the tier's whole claim is the
first list:

  upstream's   the env and its dynamics, the IPPO loop, the network, the
               hyperparameters (`_ippo_backend.UPSTREAM_IPPO_CONFIG`, copied
               from `config/ippo.yaml` at a7d94f4e), the partner protocol
               (`LoadAgentWrapper`, per-episode resampling), and the
               evaluation rollout.
  ours         the candidate's reward, substituted by rebinding the trainer's
               `LoadAgentWrapper` name for the duration of one call
               (`_ippo_reward_env`); the 88-wide observation row the reward is
               computed on; the train/held-out partner split; and the folding
               of the result into a `TrainResult`.

THE EVALUATION IS DELIBERATELY NOT BOUND. `make_evaluation` builds its own env
the same way `make_train` does, and it must see UPSTREAM's reward: that is
where `gt_return` comes from. Binding it too would make the evaluation report
the candidate's own reward as ground truth -- a run that scores itself.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from ..registry import register  # noqa: F401  (import-order parity with siblings)
from ..types import Candidate, TrainResult, Trajectory
from . import _ippo_backend as B
from . import _ippo_reward_env as RE

log = logging.getLogger(__name__)


class _PartnerSplitView:
    """A hand-set `zoo_partners` in `PartnerSplit`'s shape.

    Used ONLY when the adapter has `zoo_partners` but no `zoo_split` -- i.e. a
    caller set the train half directly. The seed row reads `.train`, `.seed`,
    `.heldout` and `.index_sha256` off the split (`run`, "partner_*" extras),
    so every one of those is carried here, and `tests/test_ippo_partner_split_view.py`
    derives that set from this file's source rather than from this docstring.

    What it carries is what it KNOWS: `seed` is the run's seed (the value the
    driver's own `split_partners` call used), `heldout` is EMPTY because the
    adapter never computed one -- an invented held-out half would silently
    become the R^pref population -- and `index_sha256` is "" for the same
    reason. Both read as "not derived here" on the row, which is the truth.
    """

    def __init__(self, train: Any, seed: int) -> None:
        self.train = train
        self.seed = int(seed)
        self.heldout: List[str] = []
        self.index_sha256 = ""
        self.pool_size = 0


def _partner_split(env: Any, cfg: Any) -> Any:
    """The train/held-out partner split, or None when no zoo is configured.

    Returns upstream-shaped `{algorithm: {"human": [uuid, ...]}}` for the
    TRAIN half only. The held-out half is carried back on the seed row and
    never handed to the wrapper: it is the R^pref population, and a partner
    that trained against the robot cannot also measure generalisation to
    unseen partners.
    """
    zoo_path = getattr(env, "zoo_path", None)
    if not zoo_path:
        return None
    from ..envs.assistax_zoo import DEFAULT_N_TRAIN, split_partners  # noqa: WPS433

    # ONE SOURCE OF TRUTH. The adapter derives the split at construction to
    # build `LoadAgentWrapper` with; recomputing it here would be two
    # computations of one fact, agreeing only because `split_partners` is
    # deterministic in the seed -- which is exactly the kind of agreement
    # that survives until someone changes one caller's arguments.
    full = getattr(env, "zoo_split", None)
    if full is not None:
        return full                                       # the adapter's own PartnerSplit, every field real
    already = getattr(env, "zoo_partners", None)
    if already:
        return _PartnerSplitView(already, seed=int(cfg.get("seed", 0) or 0))

    return split_partners(str(zoo_path), str(env.task),
                          seed=int(cfg.get("seed", 0) or 0),
                          n_train=DEFAULT_N_TRAIN)


def _evaluate(trainer: Any, config: Dict[str, Any], split: Any, env: Any,
              train_out: Any, seed: int) -> Any:
    """Upstream's evaluation rollout, on UPSTREAM's reward, plus the payload.

    `gt_return` is `sum(rollout reward) - sum(total_pref_reward)`, where the
    rollout reward is `infos.reward` -- upstream's logged value of the DICT
    returned by `env.step` (`ippo_ff_nps.py:860`, from `:443`).

    THE DICT, NOT `state.reward`, AND THE DIFFERENCE IS NOT COSMETIC.
    `LoadAgentWrapper.step` adds the partner's preference reward to the
    returned dict (`aht.py:881`) and replaces `metrics` (`:882`); it NEVER
    writes `State.reward`. So `state.reward` is already task-only, and
    `state.reward - total_pref_reward` would DOUBLE-SUBTRACT the pref term.

    A zero-pref fixture cannot tell the two formulas apart (measured on a bare
    env: `state.reward == rew["__all__"] == 0.17537644505500793`), so the
    test for this is at NONZERO pref.

    Returns `(payload, gt_return)`, with the payload `None` when the rollout
    carried no per-step channels to build one from -- absent rather than an
    empty dict, so "no evaluation" and "an evaluation that measured nothing"
    stay different facts in the artifact.
    """
    import jax                                            # noqa: WPS433
    import numpy as np                                    # noqa: WPS433

    # THE EVAL BINDS THE PREF-OBS SUBCLASS, AND ONLY THAT.
    #
    # Leaving `make_evaluation` entirely unbound fails: upstream's BARE
    # `LoadAgentWrapper` appends the 7 preference dims to EVERY agent, so the
    # eval env feeds a 49-wide robot observation to a network trained on 42
    # and raises `ScopeParamShapeError: "kernel" ... expected (49, 128),
    # existing (42, 128)` out of `ippo_ff_nps.py:881`. The eval must not carry
    # the REWARD override, or `gt_return` would be the candidate scoring
    # itself -- but it must carry the observation override the training env
    # has. The two overrides are separable and only one of them belongs here.
    #
    # So: `_human_only_pref_obs_class()` alone -- robot 42, human 49, reward
    # untouched and therefore upstream's, which is what `gt_return` needs.
    from .assistax_ppo import _human_only_pref_obs_class   # noqa: WPS433

    # ROWS RIDE ALONG IN `infos`: the eval wrapper is the human-only-pref-obs
    # class with the adapter's `row_jax` logged per step (`RE.row_logging_class`),
    # so the rollout comes home as the same 88-wide rows the prompt documents.
    with RE.candidate_reward_binding(
            trainer, RE.row_logging_class(_human_only_pref_obs_class(),
                                          lambda st, prev: env.row_jax(st, prev))) as eval_cls, \
         RE.log_wrapper_binding(trainer, {eval_cls.bird_row_key}):
        # the second binding keeps the row through upstream's LogWrapper, which
        # otherwise rebuilds `info` from its two preference keys (baselines.py:55)
        eval_env, run_evaluation = trainer.make_evaluation(
            config, load_zoo=split.train)
        # WIDTH PARITY, ASSERTED AT CONSTRUCTION. The mismatch above CRASHES
        # -- 49 into a 42-input Dense raises -- but that is luck: had the two
        # widths coincided, the eval env would feed the robot the 7 held-out
        # preference values silently, and every gt_return would be measured
        # on an observation the policy never trains against with nothing to
        # show for it. So the check is on the WIDTHS rather than on the
        # exception, and it runs before a single episode.
        # THE PROPERTY, then the cheap corroboration. The width equality
        # below is a proxy -- it catches a mismatched class and passes for
        # any other with a coincidentally equal width -- so the stack itself
        # is checked first.
        RE.assert_human_only_pref_obs(eval_env)
        _eval_w = int(eval_env.observation_space(env.EGO).shape[0])
        _train_w = int(config["OBS_DIM"])
        if _eval_w != _train_w:
            raise RuntimeError(
                f"eval env gives {env.EGO} a {_eval_w}-wide observation but "
                f"the policy was trained on {_train_w}. A difference of "
                f"{_eval_w - _train_w} is the preference-observation "
                f"augmentation reaching the wrong agent; bind the same "
                f"human-only subclass for evaluation as for training.")
        # ENV_STATE OFF. Logging it keeps the whole MJX pipeline state in
        # the scan's carry for every step of every eval episode: 18.2 GB of
        # program I/O, unrematerializable below 16.9 GiB, and an OOM at
        # 4.14 GiB on a 23 GB L4 GPU. The two values that would need it ride
        # `env_metrics` as scalars instead.
        #
        # A PARAMETER, NOT A SLICE: turning the flag off removes the state
        # from the carry entirely, where trimming leaves would still build
        # and thread it.
        log_cfg = trainer.EvalInfoLogConfig(env_state=False)
        infos = jax.block_until_ready(
            run_evaluation(jax.random.PRNGKey(seed),
                           train_out["runner_state"].train_state,
                           log_eval_info=log_cfg))

    metrics = jax.device_get(getattr(infos, "env_metrics", None) or {})
    # THE EGO'S SERIES, indexed out of the per-agent dicts. `__all__` is the
    # team key upstream also carries; the ego is what the policy was scored
    # on and what `gt_return` must describe.
    _rew = jax.device_get(infos.reward) if infos.reward is not None else {}
    rewards = np.asarray(_rew[env.EGO] if isinstance(_rew, dict) else _rew,
                         dtype=float)
    pref = (np.asarray(jax.device_get(metrics["total_pref_reward"]))
            if "total_pref_reward" in metrics else np.zeros_like(rewards))
    # The reduction lives in `aligned_task_return` so it is testable; see its
    # docstring for why done steps leave both series.
    _done = jax.device_get(infos.done) if getattr(infos, "done", None) is not None else {}
    if isinstance(_done, dict):
        # `__all__` is the episode boundary -- the same key upstream's
        # auto-reset selects on (`aht.py:900-903`), so the mask this defines
        # is the one the resets actually happened at. The episode-level reads
        # below must use THIS boundary and not a second notion of one.
        _done = _done.get("__all__", _done.get(env.EGO, np.zeros_like(rewards)))
    done = np.asarray(_done, dtype=bool)
    r_task, n_dropped = aligned_task_return(rewards, pref, done)
    # `r_pref` is ABSENT, not 0.0, when no partner contributed: a zero says a
    # partner ran and earned nothing, which is a different experiment from no
    # partner at all.
    r_pref = float(pref.sum()) if "total_pref_reward" in metrics else None
    gt_return = r_task

    _info = jax.device_get(getattr(infos, "info", None) or {})
    _act = jax.device_get(infos.action) if getattr(infos, "action", None) is not None else None
    trajectories = trajectories_from_eval(
        rows=_info.get(eval_cls.bird_row_key) if isinstance(_info, dict) else None,
        rewards=rewards, pref=pref, done=done,
        actions=(_act.get(env.EGO) if isinstance(_act, dict) else _act),
        env=env, k=_rollouts_k(env))

    speed = jax.device_get(metrics.get("pref_raw_speed"))
    force = jax.device_get(metrics.get("pref_raw_force"))
    if speed is None or force is None:
        # NOT a zero-filled payload. The consumer computes R^pref from these
        # channels, and a payload of zeros is indistinguishable from an
        # episode in which nothing was touched.
        return None, gt_return, r_pref, r_task, n_dropped, trajectories

    # THE DRAWN PARTNER IS ON `LoadAgentState`, not on the wrapper object
    # (`state.ag_idx["human"]`, one level in). The eval env is upstream's and
    # has no `_last_state` (that attribute belongs to OUR adapter), and
    # `EvalInfo.env_state` is not logged (see ENV_STATE OFF above), so the
    # index is read off `env_metrics`, where the human-only-pref-obs wrapper
    # writes it as `bird_ag_idx`.
    # PER STEP, and read at the SAME boundary the done mask defines. The
    # wrapper resamples the partner on auto-reset (`aht.py:900-903` selects
    # `ag_idx_re`), so a 4-episode rollout can contain 8 distinct indices:
    # "one id per episode slot" would name the wrong partner for every
    # segment after the first reset. Taking the id at each column's step 0
    # names the partner the episode STARTED with, and the distinct count
    # rides the payload so a reader can see it resampled rather than
    # inferring a single partner from a single id. Both episode-level reads
    # use this same `done` series, so the two reductions cannot drift apart.
    drawn = metrics.get("bird_ag_idx")
    if drawn is None:
        drawn = getattr(train_out["runner_state"], "ag_idx", None)
    if isinstance(drawn, dict):
        drawn = drawn.get("human")
    # (T, n_episodes) -> (n_episodes, T). `eval_rollout_payload`/`_as_2d`
    # take episodes FIRST; feeding the rollout's native orientation would make
    # `n_episodes` read as the horizon (1000) and fail validation many times
    # over, all of it downstream of one transpose.
    def _eps_first(a: Any) -> Any:
        arr = np.asarray(a)
        return arr.T if arr.ndim == 2 else arr.reshape(1, -1)

    speed_e, force_e = _eps_first(speed), _eps_first(force)
    n_eps = int(speed_e.shape[0])
    expected = int(config.get("NUM_EVAL_EPISODES", n_eps))
    if n_eps != expected:
        raise RuntimeError(
            f"eval payload has {n_eps} episodes but NUM_EVAL_EPISODES is "
            f"{expected}; the series are almost certainly transposed "
            f"((T, n_eps) vs (n_eps, T)).")

    partner_id = _partner_ids(split, drawn)
    return B.eval_rollout_payload(
        partner_id=partner_id,
        tool_speed=speed_e,
        contact_force=force_e,
        # The force carried INTO step 0: `LoadAgentState.prev_contact_force`
        # (`aht.py:927`), which the human-only-pref-obs wrapper copies into
        # `env_metrics` as `bird_prev_contact_force` every step, shifted by
        # one step (`pre_step_contact_force`). It must never be None: the
        # preference-metric consumer refuses None outright -- 0.0 is a legal
        # contact force and None cannot be read as zero.
        last_contact_force=pre_step_contact_force(
            _prev_contact_force(metrics))[0],
        partner=_partner_params(eval_env, drawn, n_eps, split),
        task=str(env.task), seed=int(seed),
        # `partner` feeds the preference-reward arguments; a bookkeeping
        # count does not belong in it and nothing there reads one. `meta` is
        # where facts about the ROLLOUT go.
        meta={
            # NOT A DEFECT SIGNAL, and the comment says so because the value
            # is normally NON-ZERO: one done step per episode is dropped from
            # both series, so a 32-episode rollout reports 32. It does not
            # mean steps went missing: it counts the terminal transition whose
            # pref metric upstream zero-pads, which is a known misalignment
            # rather than absent data. A value that is NOT a multiple of
            # n_episodes would be the surprising one.
            "gt_return_steps_dropped": n_dropped,
            # Distinct ids AT STEP 0 -- the population `partner_id` itself
            # holds -- because `require_valid_eval_payload` asserts the two
            # agree, and an equality between a step-0 list and an all-step
            # count is not an equality at all (the two differ once a rollout
            # auto-resets, e.g. 57 all-step against 31 step-0 over 32
            # episodes). Derivable, by construction, from the list above.
            "partner_distinct_ids": len({str(x) for x in partner_id}),
            # The all-step count, under its own name, because it is worth
            # keeping too: `LoadAgentWrapper`
            # resamples the partner at every mid-rollout auto-reset
            # (`aht.py:900-903`), so a rollout legitimately contains partners
            # no step-0 row names. It is the difference between "one partner,
            # broadcast" and "a fresh draw per episode", and under
            # `auto_reset` it is EXPECTED to exceed the field above.
            "partner_distinct_ids_all_steps": _distinct_ids(
                metrics.get("bird_ag_idx")),
            # The number behind the warning, kept in the ARTEFACT because a
            # warning is read once and an artefact is read later: "32 ids, 1
            # distinct vector" is what broadcasting one partner looks like.
            "distinct_parameter_vectors": _distinct_param_vectors(
                _partner_params(eval_env, drawn, n_eps, split)),
        },
        ), gt_return, r_pref, r_task, n_dropped, trajectories


def _rollouts_k(env: Any, default: int = 3) -> int:
    """How many eval episodes come home as trajectories: `evaluate.rollouts_per_
    candidate` when the adapter carries a config (`ctx.cfg` is not reachable
    from here), else RDA's K=3 -- the same default `evaluation._rollouts` uses,
    so the worker never ships fewer than the stage will read."""
    cfg = getattr(env, "_cfg", None)
    try:
        return max(1, int(cfg.get("evaluate.rollouts_per_candidate", default) or default))
    except Exception:                                      # noqa: BLE001
        return default


def _absent(reason: str) -> List[Trajectory]:
    """The fold's ABSENT answer, with its reason on the log. Six refusals share
    this exit; without the reason, telling them apart would take a rerun
    instead of a log line."""
    import logging                                        # noqa: WPS433
    logging.getLogger(__name__).warning("assistax_ppo: eval rollouts absent -- %s", reason)
    return []


def trajectories_from_eval(*, rows: Any, rewards: Any, pref: Any, done: Any,
                           actions: Any, env: Any, k: int) -> List[Trajectory]:
    """`Trajectory` objects from upstream's stacked eval logs, host side, pure.

    `rows` is `(T, n_eps, W)` as the scan stacked it (time first, the same
    orientation `_eps_first` corrects for the scalar series); `rewards`/`pref`/
    `done` are `(T, n_eps)`; `actions` `(T, n_eps, A)` or None. Each of the first
    `k` episodes becomes one trajectory of TASK reward (`reward - pref`, the same
    subtraction `aligned_task_return` makes so R_pref stays held out), cut at its
    first `done` (inclusive) or the full horizon, with `success` from the
    adapter's own any-step predicate on the banked rows. Returns `[]` -- never
    raises -- when no rows were logged, so an eval that logged nothing is an
    absent rollout rather than a crash after a 20M-step training.
    """
    import numpy as np                                    # noqa: WPS433

    if rows is None:
        return _absent("no rows were logged")
    R = np.asarray(rows, dtype=np.float64)
    if R.ndim != 3:
        return _absent(f"rows are {R.ndim}-d, not (T, W, n_eps)")
    rew = np.asarray(rewards, dtype=np.float64)
    pr = np.asarray(pref, dtype=np.float64) if pref is not None else np.zeros_like(rew)
    dn = np.asarray(done, dtype=bool) if done is not None else np.zeros(rew.shape, dtype=bool)
    if rew.ndim == 1:                                   # one episode: (T,) -> (T, 1), the others follow
        rew = rew[:, None]
        pr = pr.reshape(rew.shape)
        dn = dn.reshape(rew.shape)
    # THE LAYOUT IS MEASURED, NOT ASSUMED (scratchitch, NUM_EVAL_EPISODES 32, horizon
    # 1000): `reward` and `done` come out of upstream's eval scan as (T, n_eps) =
    # (1000, 32); the row we log in `infos` comes out as (T, W, n_eps) = (1000, 88, 32),
    # because `_env_step` does `info = tree_map(lambda x: x.swapaxes(0, 1), info)` on
    # every per-step info array (ippo_ff_nps.py:853) and the scan stacks T in front.
    # Assuming (T, n_eps, W) instead would build every "rollout" out of one row
    # COMPONENT across the 32 episodes, and hand the judge and `env.render` 32-vectors.
    # So: `rewards` is (T, n_eps) and is the authority for T and n_eps; the row's width axis is
    # the one equal to the adapter's `obs_dim` (88 here); the other two are matched to T and
    # n_eps by size. A layout whose axes cannot be told apart is refused as ABSENT -- an
    # unscored candidate is visible in the artifact, a scrambled rollout is not.
    if rew.ndim != 2:
        return _absent(f"rewards are {rew.ndim}-d, not (T, n_eps)")
    T, n_eps = int(rew.shape[0]), int(rew.shape[1])
    width = getattr(env, "obs_dim", None)
    width = int(width) if width is not None else None
    dims = list(R.shape)
    taken: set = set()
    w_ax = None
    if width is not None:
        w_cands = [i for i, d in enumerate(dims) if d == width]
        # a width equal to T or n_eps is only unambiguous if exactly one axis carries it
        w_ax = w_cands[0] if len(w_cands) == 1 else None
        if w_ax is None:
            return _absent(f"width {width} matches {len(w_cands)} axes of {dims}, not exactly one")
        taken.add(w_ax)
    t_cands = [i for i, d in enumerate(dims) if d == T and i not in taken]
    e_cands = [i for i, d in enumerate(dims) if d == n_eps and i not in taken]
    if T == n_eps:
        # square in time and episodes: keep the scan's order (T before n_eps) among the untaken axes
        rest = [i for i in range(3) if i not in taken]
        if len(rest) != 2 or any(dims[i] != T for i in rest):
            return _absent(f"T == n_eps == {T} but the untaken axes of {dims} are not both {T}")
        t_ax, e_ax = rest[0], rest[1]
    else:
        if len(t_cands) != 1 or len(e_cands) != 1 or t_cands[0] == e_cands[0]:
            return _absent(f"T={T} / n_eps={n_eps} do not pick one axis each of {dims}")
        t_ax, e_ax = t_cands[0], e_cands[0]
    if w_ax is None:
        w_ax = ({0, 1, 2} - {t_ax, e_ax}).pop()
    R = np.transpose(R, (t_ax, e_ax, w_ax))                 # -> (T, n_eps, W)
    act = np.asarray(actions) if actions is not None else None
    if act is not None and act.ndim == 3 and act.shape[0] != R.shape[0] and act.shape[1] == R.shape[0]:
        act = np.transpose(act, (1, 0, 2))
    out: List[Trajectory] = []
    for ep in range(min(int(k), R.shape[1])):
        T = R.shape[0]
        hits = np.flatnonzero(dn[:, ep]) if dn.shape == rew.shape else np.array([], dtype=int)
        length = int(hits[0]) + 1 if hits.size else T
        states = R[:length, ep, :]
        task_r = (rew[:length, ep] - pr[:length, ep]).tolist()
        try:
            success = bool(env.success(states))
        except Exception:                                  # noqa: BLE001
            success = False
        out.append(Trajectory(states=states,
                              actions=(act[:length, ep, :] if act is not None and act.ndim == 3 else None),
                              rewards=task_r, success=success, length=length,
                              ret=float(sum(task_r))))
    return out


def _prev_contact_force(metrics: Any) -> Any:
    """`prev_contact_force` off the eval rollout's logged env state.

    Walks to the `LoadAgentState` through whatever wrappers logged it
    (`LogEnvState` wraps it in the eval path), because the attribute lives on
    that state and not on the log wrapper around it.

    RAISES rather than returning None if it cannot be found. A None here is
    exactly what the preference-metric consumer refuses, so silently falling
    back to it would move the
    failure from this function -- where the traceback names the cause -- to
    the consumer, where it names only the symptom.
    """
    cf = (metrics or {}).get("bird_prev_contact_force")
    if cf is not None:
        return cf
    raise RuntimeError(
        "assistax_ppo: no `prev_contact_force` on the eval rollout's logged "
        "env state. It is carried at `aht.py:927` and `EvalInfo.env_state` is "
        "logged by default (`ippo_ff_nps.py:856`); if that has changed, the "
        "eval payload's `last_contact_force` needs a new source -- it must "
        "not become None, which the preference-metric consumer refuses.")


def aligned_task_return(rewards: Any, pref: Any, done: Any) -> Any:
    """`(mean per-episode task return, steps dropped)` from EvalInfo series.

    SHAPES ARE `(T, n_episodes)`, PER AGENT, AND THIS SIGNATURE ENFORCES IT.
    `EvalInfo.reward` and `.done` are per-agent DICTS (`ippo_ff_nps.py:857/860`)
    -- NOT batchified the way the train path does at `:443`. Passing a dict to
    `np.asarray` gives a 0-d OBJECT array, and `.astype(bool)` on it is `True`,
    so the mask would drop everything and the reduction would return 0.0 for
    every seed while `r_pref` stayed real: a zero reported as a measurement. A
    test that re-implements the arithmetic on hand-made 1-D arrays cannot see
    a dict.

    So the ego series are indexed out by the CALLER and this function refuses
    anything that is not a 2-D float array -- a dict reaching here is a
    programming error and now says so.

    Done steps leave BOTH series (see below), per episode column.
    """
    import numpy as np                                    # noqa: WPS433

    def _2d(x: Any, what: str) -> Any:
        if isinstance(x, dict):
            raise TypeError(
                f"aligned_task_return got a dict for {what}: EvalInfo's "
                f"reward/done are per-agent dicts and the EGO series must be "
                f"indexed out by the caller. A dict here becomes a 0-d object "
                f"array whose truth value is True, which silently drops every "
                f"step.")
        a = np.asarray(x)
        # REFUSE 0-d AND object dtype BY NAME. Every channel on this path is
        # a per-agent dict upstream, and `np.asarray` turns one into a 0-d
        # object array in silence -- `.astype(bool)` on which is `True`, so
        # the mask drops every step and the reduction returns 0.0. The
        # isinstance check above catches today's wiring; this catches a
        # rewiring that arrives as something else dict-shaped. The test proves
        # the wiring; the guard survives its change.
        if a.dtype == object or a.ndim == 0:
            raise TypeError(
                f"aligned_task_return got a {a.ndim}-d array of dtype "
                f"{a.dtype} for {what}. That is what `np.asarray` returns for "
                f"a per-agent dict, and its truth value is True, so the done "
                f"mask would drop every step and the return would be 0.0. "
                f"Index the ego agent out before calling.")
        a = a.astype(float)
        return a.reshape(a.shape[0], -1) if a.ndim >= 1 else a.reshape(1, 1)

    r = _2d(rewards, "rewards")
    pr = _2d(pref, "pref")
    if pr.shape != r.shape:
        pr = np.zeros_like(r)
    d = _2d(done, "done").astype(bool)
    keep = ~d if d.shape == r.shape else np.ones_like(r, bool)
    # Per EPISODE COLUMN, then averaged: a return is an episode-level
    # quantity, and summing the whole (T, n_eps) block would report
    # n_episodes times one episode's return.
    per_ep = (r * keep).sum(axis=0) - (pr * keep).sum(axis=0)
    return float(np.mean(per_ep)), int((~keep).sum())


def _prev_contact_force(metrics: Any) -> Any:
    """`prev_contact_force` off the eval rollout's logged env state.

    Walks to the `LoadAgentState` through whatever wrappers logged it
    (`LogEnvState` wraps it in the eval path), because the attribute lives on
    that state and not on the log wrapper around it.

    RAISES rather than returning None if it cannot be found. A None here is
    exactly what the preference-metric consumer refuses, so silently falling
    back to it would move the
    failure from this function -- where the traceback names the cause -- to
    the consumer, where it names only the symptom.
    """
    cf = (metrics or {}).get("bird_prev_contact_force")
    if cf is not None:
        return cf
    raise RuntimeError(
        "assistax_ppo: no `prev_contact_force` on the eval rollout's logged "
        "env state. It is carried at `aht.py:927` and `EvalInfo.env_state` is "
        "logged by default (`ippo_ff_nps.py:856`); if that has changed, the "
        "eval payload's `last_contact_force` needs a new source -- it must "
        "not become None, which the preference-metric consumer refuses.")


def _trainer_env(train_fn: Any) -> Any:
    """The env `make_train` closed over, for the post-bind assertion.

    Read out of the closure rather than rebuilt: rebuilding would construct a
    SECOND env and assert about that one, which is the shape where a guard
    passes while the thing it guards is wrong.
    """
    for cell in (getattr(train_fn, "__closure__", None) or ()):
        val = cell.cell_contents
        if hasattr(val, "agents") and hasattr(val, "step"):
            return val
    raise RuntimeError(
        "assistax_ppo: could not read the env out of make_train's closure, so "
        "the candidate-reward binding cannot be verified. Refusing rather "
        "than training on an unverified reward.")


def _last_eval_state(eval_env: Any) -> Any:
    """Best-effort handle on the eval env's last state, for `ag_idx`.

    Returns a bare object when there is none, so the caller's `getattr` is a
    miss rather than an AttributeError: a missing partner id must degrade to
    an empty list, never fail a training that otherwise succeeded.
    """
    return getattr(eval_env, "_last_state", object())


def _distinct_param_vectors(partner: Any) -> int:
    """How many distinct ten-tuples the per-episode parameters contain."""
    import numpy as np                                    # noqa: WPS433

    if not partner:
        return 0
    cols = [np.asarray(v, dtype=float).reshape(-1) for v in partner.values()]
    if not cols or cols[0].size == 0:
        return 0
    return len({tuple(float(c[i]) for c in cols) for i in range(cols[0].size)})


def _partner_params(eval_env: Any, drawn: Any, n_eps: int, split: Any) -> Any:
    """The ten preference parameters, PER EPISODE, from the stacked configs.

    The partner block must hold the arguments of `compute_preference_reward`,
    not run metadata such as `split_seed` or `index_sha256`:
    `eval_rollout_payload` keeps only `PARTNER_FIELDS`, so anything else would
    raise `EvalPayloadError` AFTER a full training -- no seed row, no
    checkpoint, `gt_return` never reaching an artifact -- because the payload
    is assembled last.

    PER EPISODE IS THE POINT, not a convenience. The AHT protocol resamples
    per episode, so a single parameter set would name the wrong partner for
    every episode after the first. The validator wants `(n_episodes,)`
    (`_check_1d(partner[name], ..., n_ep)`).

    The wrapper holds each field stacked over the loaded population; indexing
    by each episode's own partner index gives that episode's parameters.
    Missing fields are named rather than defaulted: a zero here is a
    preference the partner does not have.
    """
    import numpy as np                                    # noqa: WPS433

    from . import _ippo_backend as B                        # noqa: WPS433

    probe, configs = eval_env, None
    for _ in range(8):
        configs = getattr(probe, "pref_configs", None)
        if configs:
            break
        probe = getattr(probe, "_env", None)
        if probe is None:
            break
    if not configs:
        raise RuntimeError(
            "assistax_ppo: the eval env exposes no `pref_configs`, so the "
            "payload's ten preference parameters cannot be filled. They are "
            "the arguments of compute_preference_reward and the payload is "
            "refused without them.")

    human = configs.get("human") or next(iter(configs.values()))
    idx = np.asarray(drawn if drawn is not None else np.zeros(n_eps, int))
    idx = (idx[0] if idx.ndim == 2 else idx).reshape(-1).astype(int)
    if idx.size != n_eps:
        idx = np.resize(idx, n_eps)

    out, missing = {}, []
    for field in B.PARTNER_FIELDS:
        stacked = human.get(field)
        if stacked is None:
            missing.append(field)
            continue
        arr = np.asarray(stacked).reshape(-1)
        out[field] = arr[np.clip(idx, 0, arr.size - 1)]
    if missing:
        raise RuntimeError(
            f"assistax_ppo: the wrapper's pref_configs carry no {missing}; "
            f"the payload's partner block cannot be completed and a default "
            f"would assert a preference the partner does not have.")
    return out


def _distinct_ids(drawn: Any) -> int:
    """How many partners the rollout actually drew. 0 when unrecorded."""
    import numpy as np                                    # noqa: WPS433

    if drawn is None:
        return 0
    return int(np.unique(np.asarray(drawn)).size)


def _partner_ids(split: Any, drawn: Any) -> Any:
    """One uuid PER EPISODE, taken at each episode column's first step.

    `bird_ag_idx` arrives as `(T, n_episodes)` -- the index is recorded every
    step -- so flattening it with `reshape(-1)` would yield T x n_episodes ids
    where the payload contract wants n_episodes.

    Row 0 of each column is the partner the episode STARTED with. The wrapper
    resamples on auto-reset, so a long rollout contains later partners this
    list does not name -- `partner_distinct_ids_all_steps` records how many
    there were, so the resampling is visible rather than implied by a single
    id.
    """
    import numpy as np                                    # noqa: WPS433

    if drawn is None:
        return []
    idx = np.asarray(drawn)
    if idx.ndim == 0:
        idx = idx.reshape(1)
    first = idx[0] if idx.ndim == 2 else idx
    uuids = [u for by in split.train.values() for us in by.values() for u in us]
    return [uuids[int(i)] if 0 <= int(i) < len(uuids) else f"index:{int(i)}"
            for i in np.asarray(first).reshape(-1)]


def pre_step_contact_force(series: Any) -> Any:
    """The force carried INTO each step, from the post-step series.

    `bird_prev_contact_force` records the value AFTER the step
    (`aht.py:927` stores `new_prev_cf`), so `series[t]` is what step `t+1`
    begins with. The payload's `last_contact_force` is defined as the force
    carried into step 0 -- a DIFFERENT QUANTITY, and no shape check would
    have caught the substitution. Shifting by one and seeding the first row
    with the wrapper's own initial value (0.0, `aht.py`'s `init_prev_cf`)
    makes row 0 that quantity by construction.
    """
    import numpy as np                                    # noqa: WPS433

    arr = np.asarray(series, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return np.vstack([np.zeros((1, arr.shape[1])), arr[:-1]])



# ==========================================================================
# warm start: `train.init` on this backend
# ==========================================================================
#
# `run` resolves `train.init` and passes `store_policy` to
# `build_train_result`, so every candidate leaves a `policy_ref` for
# `warm_start_from_best` and the other warm starts to load. Without both
# halves `train.init` would be a declared-but-unread key on the one route
# whose learner is not ours. The two halves below follow `fasttd3`: the params
# go into `_POLICY_STORE` (which `build_child_payload` ships home from a
# forked candidate worker), and a resolved ref is loaded back into
# upstream's network at its own `network.init` call.
#
# THE BLOB IS A PLAIN uint8 ndarray, never a subclass defined here. It can
# cross a process boundary by pickle into a process running different code,
# which would fail to unpickle a class this module added; a bare ndarray also
# spills to a verified `.npy` sidecar in `checkpoint.encode`, like
# `_PolicyBlob` does.


def _params_blob(params: Any) -> Any:
    """Upstream's network params as flax msgpack bytes, worn as a uint8 array."""
    import numpy as np                                    # noqa: WPS433
    from flax import serialization                        # noqa: WPS433
    return np.frombuffer(serialization.to_bytes(params), dtype=np.uint8).copy()


def _params_from_blob(target: Any, blob: Any) -> Any:
    """Restore `blob` into `target`'s tree, or raise if the trees differ.

    `from_bytes` restores leaves by PATH and does not check shapes, so a blob
    from a network of another width (another task's observation) would load
    without complaint and fail later, or worse, not fail. The shape and dtype
    of every leaf are compared against the fresh init first.
    """
    import numpy as np                                    # noqa: WPS433
    import jax                                            # noqa: WPS433
    from flax import serialization                        # noqa: WPS433
    loaded = serialization.from_bytes(target, np.asarray(blob, dtype=np.uint8).tobytes())
    t_leaves, t_def = jax.tree_util.tree_flatten(target)
    l_leaves, l_def = jax.tree_util.tree_flatten(loaded)
    if t_def != l_def:
        raise ValueError(f"param tree differs: {l_def} vs fresh {t_def}")
    for a, b in zip(t_leaves, l_leaves):
        if tuple(np.shape(a)) != tuple(np.shape(b)):
            raise ValueError(f"param shape {np.shape(b)} vs fresh {np.shape(a)}")
    return jax.tree_util.tree_map(lambda t, v: jax.numpy.asarray(v, dtype=t.dtype),
                                  target, loaded)


class _WarmStartBinding:
    """Swap `trainer.MultiActorCritic` for one whose `init` returns `blob`.

    Upstream builds its network INSIDE `train()` (`ippo_ff_nps.py:303`,
    `network = MultiActorCritic(config=config)`, then `network.init` at :312)
    and exposes no argument for initial params, so the module global is the
    one seam. The fresh init still runs, for its tree and shapes; only its
    values are replaced. Bound around `make_train` + the training call only
    -- `make_evaluation` builds its own network and must not see this.

    `loaded` records whether the blob actually went into the network, which
    is what `warm_started_from` reports: a resolved ref that failed to load is
    a cold start and is recorded as one.
    """

    def __init__(self, trainer: Any, blob: Any) -> None:
        self.trainer, self.blob = trainer, blob
        self.loaded = False
        self.error = ""
        self._orig = None

    def __enter__(self) -> "_WarmStartBinding":
        if self.blob is None:
            return self
        orig = self._orig = self.trainer.MultiActorCritic
        binding = self

        class _WarmNet:
            def __init__(self, *a: Any, **k: Any) -> None:
                self._net = orig(*a, **k)

            def init(self, rng: Any, *a: Any, **k: Any) -> Any:
                fresh = self._net.init(rng, *a, **k)
                try:
                    params = _params_from_blob(fresh, binding.blob)
                except Exception as exc:  # noqa: BLE001 - a foreign/stale blob
                    binding.error = f"{type(exc).__name__}: {exc}"
                    log.warning("assistax_ppo warm start: blob did not load (%s); "
                                "training from scratch", binding.error)
                    return fresh
                binding.loaded = True
                return params

            def apply(self, *a: Any, **k: Any) -> Any:
                return self._net.apply(*a, **k)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._net, name)

        self.trainer.MultiActorCritic = _WarmNet
        return self

    def __exit__(self, *exc: Any) -> bool:
        if self._orig is not None:
            self.trainer.MultiActorCritic = self._orig
        return False


def run(ctx: Any, state: Any, candidate: Candidate, *, n_seeds: int = 1,
        env_steps: Optional[int] = None, resume_ref: Optional[str] = None,
        seed_phase: str = "",
        handoff: Optional[Any] = None, **backend_kw: Any) -> TrainResult:
    """Train `candidate` with upstream IPPO and fold the seeds into a result."""
    # THE REWARD DISPATCH FIRST, BEFORE THE JAX IMPORT. `train.reward_source`
    # must reach every learner backend, and `test_reward_source` proves it by
    # spying `_training_reward` and calling the backend. Importing jax first
    # would mean the spy never fires on a machine without the extra, so the
    # key would go unverified on this backend alone. It is also the
    # cheaper order: resolving a candidate's reward costs nothing, and a
    # ~3 GB CUDA stack should not be imported to discover the reward is
    # malformed.
    from .training import _scaled_reward, _training_reward       # noqa: WPS433

    cfg, env = ctx.cfg, ctx.env
    t_start = time.monotonic()
    reward = _scaled_reward(ctx, state, candidate,
                            _training_reward(ctx, state, candidate))

    # `train.init`, resolved once per candidate exactly as `fasttd3` does it:
    # the ref `_training_init` names, looked up in the local policy store.
    from .training import (_POLICY_STORE, _POLICY_STORE_MAX_BYTES,  # noqa: WPS433
                           _remember_code, _store, _training_init)
    init = _training_init(ctx, state, cfg, resume_ref, candidate)
    init_blob = _POLICY_STORE.get(init.ref) if init.ref else None
    if init.ref and init_blob is None:
        log.warning("train.init=%s: %s is not in _POLICY_STORE; training from scratch",
                    cfg.get("train.init", "from_scratch"), init.ref)

    def _store_policy(params: Any) -> str:
        ref = _store(_POLICY_STORE, f"policy:{candidate.cand_id}",
                     _params_blob(params), _POLICY_STORE_MAX_BYTES)
        _remember_code(candidate)
        return ref

    import jax                                            # noqa: WPS433
    from assistax.baselines.IPPO import ippo_ff_nps as trainer   # noqa: WPS433

    if not hasattr(env, "row_jax"):
        raise TypeError(
            f"train.backend assistax_ppo needs an upstream_assistax adapter "
            f"(it computes the candidate reward on device from `row_jax`); got "
            f"{type(env).__name__}. problem.env_id must be an "
            f"`upstream_assistax_*` task.")

    row_fn = reward.row()

    split = _partner_split(env, cfg)
    if split is None:
        raise ValueError(
            "assistax_ppo: no zoo configured on the adapter, so the partner "
            "would be upstream's default rather than a sampled frozen policy. "
            "That is a PASSIVE-HUMAN experiment, not the AHT protocol, and it "
            "must not be reported under this backend's name. Set `zoo_path` "
            "and `zoo_partners` on the env, or run the baseline deliberately "
            "through the adapter's own step loop.")
    # BY CONTENT, BEFORE THE COMPILE: an absent, truncated or different zoo
    # fails here in a second with a sentence, not after a trainer build.
    from ..envs.assistax_zoo import check_zoo            # noqa: WPS433
    _zoo_problem = check_zoo(str(env.zoo_path))
    if _zoo_problem:
        raise ValueError(f"assistax_ppo: {_zoo_problem}")

    config = B.upstream_ippo_config(
        env_name=str(env.task),
        # HOMOGENISATION MUST RIDE IN ENV_KWARGS. `make_train` builds its own
        # env from these alone; `homogenisation_method` is a separate class
        # attribute on the adapter, so forwarding only `upstream_env_kwargs`
        # would leave upstream's env at the constructor default (None) and a
        # 49-in zoo partner could not load into it at all. The adapter and
        # the trainer would be running DIFFERENT environments, which is the
        # one thing the ENV_KWARGS citation exists to prevent.
        env_kwargs={**env.upstream_env_kwargs,
                    "homogenisation_method": env.homogenisation_method},
        total_timesteps=int(env_steps or cfg.get("train.env_steps", 0) or 0),
        zoo_path=str(env.zoo_path),
        num_envs=backend_kw.get("num_envs"), num_steps=backend_kw.get("num_steps"))

    # The wrapper subclasses whatever the trainer currently binds, so a
    # preference-observation subclass already in place is EXTENDED rather than
    # replaced -- two sibling subclasses would mean only one of them is bound
    # at `:259` and the other silently does nothing.
    # STACKED ON THE PREFERENCE-OBSERVATION SUBCLASS, not on the bare
    # LoadAgentWrapper. Upstream's `_append_pref_to_obs` appends the 7
    # preference dims to EVERY agent's observation, robot included, so binding
    # the bare class would put the held-out preference vector into the
    # ROBOT'S TRAINING OBSERVATION -- the leak this tier is built to prevent,
    # on the one path where it matters, while the eval-path adapter stayed
    # clean and would have shown nothing. `_human_only_pref_obs_class()`
    # overrides that to append for the human only and narrows
    # `observation_spaces` to match.
    #
    # ONE STACKED CLASS, not two siblings: `make_train` binds a single name at
    # `:259`, so two independent LoadAgentWrapper subclasses would mean
    # whichever was bound second silently did nothing.
    from .assistax_ppo import _human_only_pref_obs_class   # noqa: WPS433

    wrapper_cls = RE.candidate_reward_wrapper_class(
        _human_only_pref_obs_class(), row_fn, env.row_jax, ego=env.EGO)

    # TRAIN ENV == EVAL ENV, asserted rather than intended. A run that trains
    # at `ctrl_cost_weight: 0` and evaluates at the constructor's 1e-6
    # produces every `gt_return` from a different reward function than the
    # policy optimised -- small in magnitude, invisible in the artifact, and
    # fatal to the headline number.
    # Both sides are built from `config["ENV_KWARGS"]` here, so this compares
    # what was actually used rather than what was meant.
    _adapter_kwargs = {**env.upstream_env_kwargs,
                       "homogenisation_method": env.homogenisation_method}
    # OVER THE UNION OF BOTH KEY SETS, with a sentinel for "absent on one
    # side". Iterating only the adapter's keys would make the check
    # ONE-DIRECTIONAL: a key set on the TRAINER and silent on the adapter
    # would pass without a word, and the two likeliest such keys are
    # `disability` and `preference_rewards` -- precisely the two this adapter
    # singles out as commented out in upstream's ippo.yaml, i.e. the two most
    # likely to be switched on by one side alone.
    _MISSING = object()
    _diff = {}
    for k in set(_adapter_kwargs) | set(config["ENV_KWARGS"]):
        a = _adapter_kwargs.get(k, _MISSING)
        t = config["ENV_KWARGS"].get(k, _MISSING)
        if a is not t and a != t:
            _diff[k] = ("<absent>" if a is _MISSING else a,
                        "<absent>" if t is _MISSING else t)
    if _diff:
        raise RuntimeError(
            f"train and eval environments differ on {sorted(_diff)}: "
            f"{_diff} (adapter value, trainer value). `gt_return` would be "
            f"measured under a different reward function from the one "
            f"trained on.")

    # `auto_reset` IS EXEMPT, BY ARGUMENT AND BY NAME -- not merely unchecked:
    # it is the one env key this repo knowingly diverges on, so the exemption
    # is argued here and enforced below.
    #
    # The divergence is real and intended: the adapter builds with
    # `auto_reset=False` because `EnvAdapter` owns the episode boundary and an
    # auto-reset swaps the next episode's state into the terminating step,
    # while upstream's vmapped training scan REQUIRES auto-reset -- 1024 envs
    # cannot each stop and wait for a host reset inside a `lax.scan`. So the
    # two sides need different values for the same reason the two paths exist.
    #
    # What it costs, stated rather than waved away: the training rollout
    # crosses episode boundaries in-scan and the eval rollout does not, so the
    # terminal transition of each training episode pairs a terminal row with
    # the next episode's first row. At `episode_length` 1000 that is one step
    # in a thousand, and `terminal_row_is_reset` records it on the seed row.
    # AND `auto_reset=False` DOES NOT GIVE THE ADAPTER THE EPISODE BOUNDARY
    # UNDER THE WRAPPER. `LoadAgentWrapper.step` auto-resets UNCONDITIONALLY
    # -- `jax.tree.map(lambda x, y: jax.lax.select(dones["__all__"], x, y),
    # states_re, states_st)` at `aht.py:887-909` -- regardless of the kwarg
    # passed to `assistax.make`, and it resamples `ag_idx` on the same
    # select, so the terminal step gets a fresh episode AND a new partner.
    # "The adapter owns the episode boundary" is true only WITHOUT the
    # wrapper; under a zoo partner the eval path's terminal row is a reset
    # row too.
    #
    # The exemption stands, for the narrower reason: whatever the episode
    # boundary does, it is NOT a reward-function difference, which is what
    # this assertion is for.
    if "auto_reset" in config["ENV_KWARGS"]:
        raise RuntimeError(
            "assistax_ppo: `auto_reset` was passed to the trainer's ENV_KWARGS. "
            "It is the one key the adapter and the trainer deliberately differ "
            "on (adapter False, upstream's scan True), so setting it here makes "
            "the exemption above silently false. Remove it, or make it a "
            "checked key and re-argue the exemption.")

    # THE EFFECTIVE FLAG STRING, from the environment, not the one a launcher
    # meant to set -- a driver can add `--xla_gpu_autotune_level=0` to it, and
    # autotune-off alone does NOT fix the Triton refusal, so a record of the
    # intent would credit the wrong flag for a pass.
    import os                                             # noqa: WPS433
    xla_flags = os.environ.get("XLA_FLAGS", "")

    outcomes: List[B.SeedOutcome] = []
    for i in range(int(n_seeds)):
        seed = int(cfg.get("seed", 0) or 0) + i
        t0 = time.monotonic()
        with RE.candidate_reward_binding(trainer, wrapper_cls), \
             _WarmStartBinding(trainer, init_blob) as warm:
            train_fn = trainer.make_train(config, load_zoo=split.train)
            # AFTER make_train has built its env, because that is the first
            # moment the binding can be shown to have taken effect rather
            # than merely been requested.
            # The env from `make_train`'s CLOSURE, and only that: an env cached
            # on the trainer module could be a stale one from a previous seed,
            # and the guard would assert about it and pass.
            RE.assert_candidate_rewarded(_trainer_env(train_fn))
            t_compiled = time.monotonic()
            out = jax.block_until_ready(
                train_fn(jax.random.PRNGKey(seed), config["LR"],
                         config["ENT_COEF"], config["CLIP_EPS"]))
        wall = time.monotonic() - t0
        trained_steps = (config["NUM_STEPS"] * config["NUM_ENVS"]
                         * (config["TOTAL_TIMESTEPS"]
                            // config["NUM_STEPS"] // config["NUM_ENVS"]))
        # THE EVALUATION IS RUN UNBOUND, which is the whole point of doing it
        # separately rather than reading the training rollout: `make_evaluation`
        # builds its own env and must see UPSTREAM's reward, because that is
        # where `gt_return` comes from. Binding the candidate wrapper here too
        # would make the run score itself. The binding above has already been
        # restored by the `with` block, so this is outside it by construction
        # rather than by remembering to be.
        # ONE CHECKPOINT ROW PER SEED, AT THE FINAL STEP.
        #
        # `evaluation.py:911-922`'s `fitness_native_reward` reads
        # `NATIVE_REWARD_KEYS = ("gt_return",)` out of
        # `seed_metrics[i]["checkpoints"]` -- NOT off the seed row. So the
        # seed-row `gt_return` is real and correct and sits where that reader
        # does not look: with no checkpoints emitted,
        # `evaluate.fitness.source: native` (as `era_s` uses) would score
        # nothing at all, on a backend that trained fine.
        #
        # One row, not a curve: this backend has no intermediate evaluation
        # to report, and a fabricated curve would be worse than a short one.
        eval_payload, gt_return, r_pref, r_task, n_dropped, trajectories = _evaluate(
            trainer, config, split, env, out, seed)

        # NO PAYLOAD MEANS NO CHECKPOINT ROW, not a row scoring 0.0. An
        # unscored seed and a seed that scored zero are different facts, and
        # an `else 0.0` would make them the same one.
        #
        # NOT "pass None into the row" either: `checkpoint_row` coerces
        # `fitness` into FOUR aliases with `float()`, so a None there raises,
        # and widening it would change a contract every consumer of those
        # aliases shares. A checkpoint row IS a scored point; if nothing was
        # scored there is no point to record. `gt_return` still rides the
        # seed row as None, so the seed reads unscored rather than zero.
        checkpoints = []
        if isinstance(eval_payload, dict):
            checkpoints = [B.checkpoint_row(
                step=float(trained_steps), round_=0, seed=seed,
                # The payload carries no "fitness" key, so a
                # `.get("fitness", 0.0)` would score every checkpoint 0.0 --
                # the None-vs-0.0 rule one hop over. The task metric is what a
                # checkpoint's fitness means here, and `_evaluate` does not
                # compute one, so this is the candidate's own return: the
                # criterion `train.checkpoint_selection` is allowed to use.
                fitness=float(r_task) if r_task is not None else 0.0,
                success_rate=0.0,
                # `is None`, not `or`: `r_task or 0.0` cannot tell a measured
                # 0.0 from an absent one, and a measured zero is a result.
                reward_return=float(r_task) if r_task is not None else 0.0,
                gt_return=gt_return)]

        outcomes.append(B.SeedOutcome(
            seed=seed,
            policy_params=jax.device_get(out["runner_state"].train_state.params),
            env_steps=trained_steps,
            train_steps=trained_steps,
            train_steps_requested=int(config["TOTAL_TIMESTEPS"]),
            wallclock_s=wall,
            # `t_compiled` is the trace/compile boundary of `make_train`, not
            # of the jitted step; reported as measured rather than inferred.
            compile_s=t_compiled - t0,
            sps=(trained_steps / wall) if wall > 0 else 0.0,
            num_envs=int(config["NUM_ENVS"]),
            horizon_effective=int(getattr(env, "horizon", 0)),
            checkpoints=checkpoints,
            eval_payload=eval_payload,
            gt_return=gt_return,
            r_pref=r_pref,
            r_task=r_task,
            trajectories=trajectories,
            xla_flags=xla_flags,
            gt_return_steps_dropped=n_dropped,
            warm_started_from=(init.ref or "") if warm.loaded else "",
            extra={
                "partner_split_seed": int(split.seed),
                "partner_n_train": len(
                    [u for by in split.train.values() for us in by.values() for u in us]),
                "partner_n_heldout": len(split.heldout),
                "partner_index_sha256": split.index_sha256,
                "upstream_env_kwargs": dict(env.upstream_env_kwargs),
                "num_steps": int(config["NUM_STEPS"]),
            }))

    return B.build_train_result(candidate, outcomes, algorithm_cited="ppo",
                                init_source=init.source,
                                init_from_cand_id=init.from_cand_id,
                                store_policy=_store_policy,
                                wallclock_s=time.monotonic() - t_start)
