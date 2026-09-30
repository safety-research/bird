"""The native-signal rule (N): which signal a *supervised* search is allowed to read.

One rule, shared by every search-time reader (evaluate.fitness.source and the
handful of allocators/screens/priors that also read a ground-truth channel), so
"the native signal" means the same thing everywhere.

The four signals, by origin x shape:

  native_success  the benchmark's SHIPPED binary success. A task carries one iff
                  `discrete_success.kind` is `discrete` -- and `kind: discrete`
                  holds ONLY when the spec pins a VERIFIED faithful
                  reimplementation of the vendor's own success
                  (`discrete_success.shipped.reimplementation`, sound against the
                  vendor's `info["success"]`). Every one of the catalogue's
                  discrete specs carries that block; the BIRD-authored proxies
                  (`task_metric >= threshold`, HumanoidBench geometry) live only
                  on `continuous_only` tasks and are NEVER native. `native_authored`
                  is the repo-owned-env case (this repo authored the whole env, so
                  the repo IS the vendor and its own `success()` is native by
                  construction) -- carried by the toy family (toy_reacher,
                  toy_gridworld, toy_hungry_thirsty).
                  assistax stays continuous_only (no success check to promote).
  native_reward   the benchmark's SHIPPED reward and its return (`reference_reward`,
                  `gt_return`). Present unless the spec declares
                  `reward.human.kind: none` -- `reference_reward` is absent on
                  exactly those tasks.
  custom_metric   the BIRD-authored continuous score (`task_metric`, `fitness`).
                  NEVER a search signal on a bird-origin task -- optimising toward
                  a number we wrote is not the published method.
  custom_success  a BIRD-authored threshold on `custom_metric`. Provisional; not a
                  search signal either.

`native` resolves to native_success where the task ships one, else the
native_reward return, else it REFUSES -- it never degrades to custom_metric.
The refuse set is exactly the five gymnasium tasks whose spec says
`reward.human.kind: none` (half_cheetah_backward, half_cheetah_target_speed,
hopper_hop_in_place, reacher_hold, swimmer_heading): no shipped success and no
shipped reward, so a supervised method has nothing admissible to read.

Pure and import-safe: the resolver takes the two spec facts as strings (no spec
object, no adapter, no cycle), so both `bird/config.py` (load time) and the
evaluate/allocator/screen readers (read time) resolve identically. `spec_kinds`
and `channel_for_env` are conveniences that read those two facts off a TaskSpec
(the `tasks` import is lazy, so this module has no import-time dependency on it).
"""

from typing import Any, Optional, Tuple

NATIVE_SUCCESS = "native_success"
NATIVE_REWARD = "native_reward"
REFUSE = "refuse"

#: `discrete_success.kind` values that ship a benchmark-native binary success.
#: `discrete` is provenance-safe: it holds only where the spec pins a verified
#: reimplementation of the vendor's own success (see the module docstring).
NATIVE_SUCCESS_KINDS = frozenset({"discrete", "native_authored"})

#: Curve-row keys that carry the genuine env success FLAG (its per-checkpoint
#: mean). Deliberately NOT `task_success`/`score`/`consecutive_successes`/
#: `fitness`: those are ALIASES of `custom_metric` in both backends' curve rows
#: (`training.py`), so a reader that fell through to them would grade a
#: supervised search on the number we wrote -- the exact bug N exists to stop.
NATIVE_SUCCESS_KEYS = ("success_rate", "success", "successes", "is_success")

#: The reference-reward RETURN, per checkpoint. None on a task with no reference,
#: which `_num`/`_series` skip -- so a native_reward read on such a task yields
#: nothing and the candidate takes the failure sentinel (and coherence refuses
#: the pin at load time, so this is belt-and-suspenders).
NATIVE_REWARD_KEYS = ("gt_return",)


def has_native_success(discrete_success_kind: Optional[str]) -> bool:
    """Does the task ship a benchmark-native binary success?"""
    return str(discrete_success_kind or "") in NATIVE_SUCCESS_KINDS


def has_native_reward(reward_human_kind: Optional[str]) -> bool:
    """Does the task ship a reference reward (so `gt_return` is real)?

    False exactly when `reward.human.kind` is `none` (or absent) -- the
    unsupervised-only tasks."""
    return str(reward_human_kind or "") not in ("", "none")


def resolve_channel(discrete_success_kind: Optional[str],
                    reward_human_kind: Optional[str]) -> str:
    """`native`'s resolution: native_success, else native_reward, else refuse.

    Never returns a custom channel. The two arguments are the only spec facts it
    reads, so load-time and read-time resolve the same string."""
    if has_native_success(discrete_success_kind):
        return NATIVE_SUCCESS
    if has_native_reward(reward_human_kind):
        return NATIVE_REWARD
    return REFUSE


def spec_kinds(spec: Any) -> Tuple[Optional[str], Optional[str]]:
    """`(discrete_success.kind, reward.human.kind)` off a TaskSpec, or `(None,
    None)` when there is no spec (an env-id-only run) -- which resolves to
    REFUSE for a native source, the honest answer when nativeness cannot be
    verified."""
    if spec is None:
        return None, None
    ds = (spec.discrete_success or {}).get("kind")
    rw = ((spec.reward or {}).get("human") or {}).get("kind")
    return ds, rw


def kinds_for_env(env_id: Optional[str] = None,
                  task_id: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """`(discrete_success.kind, reward.human.kind)` for a run identified by
    env_id/task_id. Loads the spec lazily (no import cycle); `(None, None)` when
    no spec can be found -- which resolves to REFUSE for a native source."""
    spec = None
    try:
        from .tasks import by_env_id as _by_env_id, load as _load  # lazy
        spec = _load(str(task_id)) if task_id else _by_env_id(str(env_id or ""))
    except Exception:  # noqa: BLE001 -- no spec means nativeness is unverifiable -> refuse
        spec = None
    return spec_kinds(spec)


def channel_for_env(env_id: Optional[str] = None,
                    task_id: Optional[str] = None) -> str:
    """Resolve the native channel for a run identified by env_id/task_id.

    Returns REFUSE when no spec can be found -- the same read every search-time
    reader makes, so they agree."""
    ds, rw = kinds_for_env(env_id, task_id)
    return resolve_channel(ds, rw)


def native_curve_keys(env_id: Optional[str] = None,
                      task_id: Optional[str] = None) -> Tuple[Optional[Tuple[str, ...]], str]:
    """`(curve keys, channel)` for reading the native signal off a checkpoint
    curve for this run, or `(None, REFUSE)`. The shared entry point for every
    search-time reader beyond `evaluate.fitness.source` (feedback, pruning,
    the Singh grid, CARD's partition) so "the native signal" is one definition
    everywhere: native_success → the success-flag keys, native_reward → the
    shipped-reward return, neither → refuse (never `task_metric`)."""
    ch = channel_for_env(env_id, task_id)
    if ch == NATIVE_SUCCESS:
        return NATIVE_SUCCESS_KEYS, ch
    if ch == NATIVE_REWARD:
        return NATIVE_REWARD_KEYS, ch
    return None, REFUSE
