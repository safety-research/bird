"""The uniform call surface over ``policies/`` -- one signature, two shapes.

Every campaign policy, whatever its native convention, is exposed as::

    pol = load_policy("mt10_suite/reach")
    env = registry.get("env", pol.env_id)(None)
    s = env.reset(rng); pol.reset(rng)
    a = pol.act(s, t=t, env=env)     # ALWAYS the adapter's normalised units

Two wrappers adapt the two campaign shapes (``policies/README.md`` has the
table) **without touching the verbatim code**:

- ``ClosurePolicy``   MT10 style: ``make_policy(p) -> act(s)``, action already
  normalised.  Handles both sub-shapes found in the wild -- ``act(s)`` owning
  its state in a closure (pegctl/pushmw) and ``pol(s, st)`` with a
  caller-owned dict (mtsuite) -- and an ``act`` that returns ``(a, info)``.
- ``RigPolicy``       humanoid style: ``fn(rig, t, ctrl) -> ctrl`` in RAW
  actuator units over a mujoco Rig/Shim.  The wrapper builds the rig lazily
  over the adapter the episode actually runs in, threads ``ctrl``, and
  normalises against ``actuator_ctrlrange`` once, here.  A rig built on
  anything but the adapter's own env reads a simulator the episode is not
  stepping.

``t`` and ``env`` are in the signature deliberately: an MT10 policy genuinely
needs nothing but the observation, and a humanoid policy genuinely cannot work
without the model.  A uniform signature that hid that difference would be how
it gets forgotten.

Trust model: ``entry.file``/``entry.symbol`` are dynamically imported.  That is
safe here for one reason only -- ``policies/`` is a committed repo artifact,
the same trust level as ``bird/`` itself.  ``_contained`` refuses any path
that escapes ``policies/``, and nothing in this module ever reads a manifest
out of a run directory (run dirs on shared mounts may be world-writable; the
``checkpoint._DECODABLE`` reasoning).

This module needs numpy and the stdlib.  Importing a *policy* additionally
needs its family's runtime (``metaworld`` and HumanoidBench pin incompatible
mujoco versions; no interpreter runs both -- ``family:`` in the manifest says
which venv you need).
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from bird.policies import PolicyError, PolicyRecord, get as get_record, \
    policies_root

__all__ = [
    "Policy",
    "gather_params",
    "ClosurePolicy",
    "RigPolicy",
    "import_beside",
    "load_policy",
    "run_episode",
]


class Policy:
    """One policy behind the uniform signature.  Stateful across an episode
    (phase machines, threaded ctrl); ``reset`` starts a fresh episode."""

    def __init__(self, record: Optional[PolicyRecord] = None) -> None:
        self.record = record
        self.id = record.id if record else type(self).__name__
        self.env_id = record.env_id if record else None

    def reset(self, rng: Any = None) -> None:  # noqa: ARG002 - part of the contract
        """Called once per episode, before the first ``act``."""

    def act(self, obs: np.ndarray, *, t: int, env: Any = None) -> np.ndarray:
        raise NotImplementedError


class ClosurePolicy(Policy):
    """Wraps ``factory(params) -> act`` where ``act`` is either ``act(s)``
    (closure-held state) or ``pol(s, st)`` (caller-owned dict), returning a
    normalised action or ``(action, info)``."""

    def __init__(self, factory: Callable[..., Callable],
                 params: Optional[Dict[str, Any]] = None,
                 record: Optional[PolicyRecord] = None) -> None:
        super().__init__(record)
        self._factory = factory
        self._params = dict(params or {})
        self._act: Optional[Callable] = None
        self._st: Dict[str, Any] = {}
        self.info: Dict[str, Any] = {}

    def reset(self, rng: Any = None) -> None:
        self._act = _call_factory(self._factory, self._params)
        self._st = {}
        self.info = {}
        n_pos = _positional_arity(self._act)
        self._pass_state = n_pos >= 2

    def act(self, obs: np.ndarray, *, t: int, env: Any = None) -> np.ndarray:
        if self._act is None:
            self.reset()
        out = self._act(obs, self._st) if self._pass_state else self._act(obs)
        if isinstance(out, tuple):
            out, self.info = out[0], (out[1] if len(out) > 1 else {})
        return np.asarray(out, dtype=float)


class RigPolicy(Policy):
    """Wraps ``fn(rig, t, ctrl) -> ctrl`` (raw actuator units).

    ``rig_factory(env)`` builds the campaign's Rig/Shim over the adapter the
    episode runs in; a factory written to take the adapter itself is tried
    first, then its underlying ``_env``.  ``ctrl`` starts from the rig's
    ``key_ctrl`` where it has one and zeros otherwise.
    """

    def __init__(self, fn_factory: Callable[..., Callable],
                 params: Optional[Dict[str, Any]] = None,
                 rig_factory: Optional[Callable] = None,
                 record: Optional[PolicyRecord] = None) -> None:
        super().__init__(record)
        self._fn_factory = fn_factory
        self._params = dict(params or {})
        self._rig_factory = rig_factory
        self._fn: Optional[Callable] = None
        self._rig: Any = None
        self._ctrl: Optional[np.ndarray] = None
        self._lo: Optional[np.ndarray] = None
        self._hi: Optional[np.ndarray] = None

    def reset(self, rng: Any = None) -> None:
        # A controller from the previous episode may hold resources (it says
        # so with a `close()` method); release them before the new one is
        # built, so the old controller never touches the data the new episode
        # is stepping.
        old = self._fn
        if old is not None and hasattr(old, "close"):
            old.close()
        self._fn = _call_factory(self._fn_factory, self._params)
        self._rig = None          # rebuilt lazily over the episode's env
        self._ctrl = None

    def _build_rig(self, env: Any) -> None:
        if self._rig_factory is None:
            raise PolicyError(
                f"{self.id}: pattern `rig` with no rig factory -- the "
                "manifest's rig: block names the campaign's Rig/Shim class")
        # A rig reads the MuJoCo model off the object it is handed.  A
        # wrapper may wrap the adapter (its `_env` IS the adapter) and the
        # adapter wraps the gym env (whose `.model` the rig wants), so one level of unwrapping would hand the rig the adapter and
        # fail with "'H1HandWalk' object has no attribute 'model'".  Walk the
        # `_env` chain until the factory accepts a layer or the chain ends;
        # re-raise the LAST error.
        cur: Any = env
        while True:
            try:
                self._rig = self._rig_factory(cur)
                break
            except Exception:
                nxt = getattr(cur, "_env", None)
                if nxt is None or nxt is cur:
                    raise
                cur = nxt
        lo = getattr(self._rig, "lo", None)
        hi = getattr(self._rig, "hi", None)
        if lo is None or hi is None:
            m = getattr(self._rig, "m", None)
            if m is None:
                raise PolicyError(
                    f"{self.id}: rig exposes neither lo/hi nor a model `m` "
                    "to read actuator_ctrlrange from")
            lo = m.actuator_ctrlrange[:, 0]
            hi = m.actuator_ctrlrange[:, 1]
        self._lo = np.asarray(lo, dtype=float)
        self._hi = np.asarray(hi, dtype=float)
        key = getattr(self._rig, "key_ctrl", None)
        self._ctrl = (np.asarray(key, dtype=float).copy() if key is not None
                      else np.zeros_like(self._lo))

    def act(self, obs: np.ndarray, *, t: int, env: Any = None) -> np.ndarray:
        if self._fn is None:
            self.reset()
        if self._rig is None:
            if env is None:
                raise PolicyError(
                    f"{self.id}: a rig policy needs `env` -- it reads the "
                    "mujoco model/data, not the observation vector")
            self._build_rig(env)
        out = np.asarray(self._fn(self._rig, t, self._ctrl.copy()), dtype=float)
        ctrl = np.clip(out, self._lo, self._hi)
        self._ctrl = ctrl
        span = self._hi - self._lo
        span = np.where(span == 0.0, 1.0, span)
        return np.clip(2.0 * (ctrl - self._lo) / span - 1.0, -1.0, 1.0)


def import_beside(here: Any, *names: str) -> Any:
    """Import a campaign's own modules by their bare names, from `here`, and
    refuse a name that already means something else in this process.

    Campaign code addresses its siblings by bare module name (`import hk`,
    `from rock import Rig`), which Python resolves through `sys.modules`
    first -- so once two campaigns that both ship a `rock.py` are loaded in one
    process, the second silently gets the first's. Nothing about a wrong
    `rock` is an error until the numbers come out different, which is the
    quiet failure this registry exists to refuse. So: `here` goes first on
    `sys.path`, and every name is checked BEFORE and AFTER the import to be
    the file beside the caller; anything else is a `PolicyError` naming both
    files. One campaign per process is the supported shape (that is how
    `scripts/eval_policy.py` runs); this is what makes the other shape loud.

    Returns the module for one name, a tuple for several.
    """
    here = Path(here).resolve()
    if here.is_file():
        here = here.parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    out = []
    for name in names:
        want = here / f"{name}.py"
        prior = sys.modules.get(name)
        if prior is not None:
            pf = getattr(prior, "__file__", None)
            if pf is None or Path(pf).resolve() != want.resolve():
                raise PolicyError(
                    f"`{name}` is already imported in this process from {pf or '<no file>'}, "
                    f"not from {want} -- two campaigns share a module name; load one "
                    "campaign per process (scripts/eval_policy.py does)")
        mod = importlib.import_module(name)
        got = getattr(mod, "__file__", None)
        if got is None or Path(got).resolve() != want.resolve():
            raise PolicyError(f"`{name}` resolved to {got or '<no file>'}, not to {want}")
        out.append(mod)
    return out[0] if len(out) == 1 else tuple(out)


# ---------------------------------------------------------------------------

def _positional_arity(fn: Callable) -> int:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return 1
    return sum(1 for p in sig.parameters.values()
               if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
               and p.default is p.empty)


def _call_factory(factory: Callable, params: Dict[str, Any]) -> Callable:
    """Call a campaign factory with its constants, tolerating both wild
    conventions: MT10 factories take ONE positional dict (``pol_reach(p)``),
    humanoid factories take keyword constants (``make_hop(**kw)``)."""
    if not params:
        # A one-positional-dict factory with nothing in the manifest gets an
        # EMPTY dict, not a bare call: `make_policy()` on `make_policy(p)` is a
        # TypeError that looks like a broken policy when it is a bare manifest.
        return factory({}) if _positional_arity(factory) >= 1 else factory()
    if _positional_arity(factory) >= 1:
        return factory(dict(params))
    try:
        return factory(**params)
    except TypeError as exc:
        raise PolicyError(
            f"{getattr(factory, '__name__', factory)}: constants do not fit "
            f"the factory signature ({exc}). The manifest's params/"
            "kwargs_from need per-policy wiring -- see the campaign's own "
            "driver for how it massaged them") from exc


def _contained(root: Path, rel: str) -> Path:
    """Resolve ``rel`` inside THIS policy's campaign dir and refuse escapes --
    including into sibling campaigns.  Each campaign dir is self-contained by
    rule (policies/README.md): a cross-campaign ``../other/`` reference would
    couple two verbatim trees so that archiving or moving one silently breaks
    the other's manifest."""
    base = Path(root).resolve()
    path = (base / rel).resolve()
    if base != path and base not in path.parents:
        raise PolicyError(
            f"`{rel}` resolves outside its campaign dir {base.name}/ -- "
            "refused (campaign dirs are self-contained; copy the file in)")
    return path


_MODULE_CACHE: Dict[Path, Any] = {}


def _evict_other_campaigns(campaign_dir: Path) -> None:
    """Drop from `sys.modules` every BARE-named module that lives under the
    policies root but outside `campaign_dir`.

    Campaign code imports its siblings by bare name (`import hk`, `from rock
    import Rig`), and Python caches those by that name for the whole process.
    A bare name can ship with DIFFERENT contents in different campaign dirs (a
    helper module copied and modified per campaign), so without this a second
    campaign loaded in
    the same process is silently handed the first campaign's module -- the
    quiet failure `import_beside` refuses for its callers. Eviction makes the
    campaign being loaded resolve its OWN files; a campaign loaded earlier keeps
    the module objects it already bound, because its code holds them by
    reference. What it cannot protect is a module imported LAZILY, inside a
    function, after another campaign was loaded: that resolves through the
    campaign loaded last (its dir is first on sys.path). One campaign per process stays the supported shape
    (`scripts/eval_policy.py` runs that way); this makes the several-policies
    shape (a caller that loads demonstrations for several tasks) correct at load
    time rather than silently wrong."""
    roots = {campaign_dir.parent}
    try:
        roots.add(Path(policies_root()).resolve())   # a `load_policy(root=...)` caller still evicts the repo's own
    except Exception:  # noqa: BLE001 -- no repo policies/ dir is not an error here
        pass
    for key, mod in list(sys.modules.items()):
        if "." in key or key.startswith("bird_policies_"):
            continue
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        try:
            fp = Path(f).resolve()
        except OSError:
            continue
        if any(r in fp.parents for r in roots) and campaign_dir not in fp.parents:
            del sys.modules[key]


def _import_file(path: Path, campaign_dir: Optional[Path] = None) -> Any:
    if path in _MODULE_CACHE:
        return _MODULE_CACHE[path]
    campaign_dir = Path(campaign_dir).resolve() if campaign_dir is not None else path.parent
    _evict_other_campaigns(campaign_dir)
    name = "bird_policies_" + "_".join(path.with_suffix("").parts[-2:])
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PolicyError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    # A verbatim campaign file may prepend its own repo guess to sys.path at
    # import (`sys.path.insert(0, <a checkout>)`).
    # In-process that insert is mostly inert -- `bird` is already in
    # sys.modules, so *its* submodules keep resolving through this checkout --
    # but any package NOT yet imported would afterwards resolve through the
    # foreign tree first, which is exactly the wrong-checkout trap. So: snapshot
    # sys.path, let the module run, then strip every entry it prepended that
    # contains a different `bird` package, and refuse outright if the loaded
    # `bird` itself moved (belt on top of the sys.modules braces).
    before = list(sys.path)
    our_bird = Path(__file__).resolve().parent
    # The file's OWN directory goes first, as it does for `python file.py`:
    # campaign code was written and measured as scripts run from their campaign
    # dir, and a verbatim file that imports a sibling by bare name without
    # inserting its own dir relies on exactly that; without it such a policy
    # registers cleanly and raises ModuleNotFoundError on every load
    # (`tests/test_policy_load_humanoid.py` is the guard). The dir is moved to
    # the FRONT even when an earlier load left it further back, so this
    # campaign's bare names resolve to its own files first (with
    # `_evict_other_campaigns`, the sys.modules half); and it is LEFT there
    # after the import, as a campaign's own insert is: a campaign may guard its
    # insert with `if here not in sys.path` and then import siblings lazily at
    # reset or step, so removing the loader's entry would strand it
    # (ModuleNotFoundError at its first step).
    here = str(path.parent)
    while here in sys.path:
        sys.path.remove(here)
    sys.path.insert(0, here)
    try:
        spec.loader.exec_module(mod)
    finally:
        added = [p for p in sys.path if p not in before]
        for entry in added:
            foreign = Path(entry, "bird", "__init__.py")
            try:
                is_foreign = (foreign.is_file()
                              and foreign.resolve().parent != our_bird)
            except OSError:
                is_foreign = True
            if is_foreign:
                sys.path.remove(entry)
    import bird as _bird_check
    if Path(_bird_check.__file__).resolve().parent != our_bird:
        raise PolicyError(
            f"importing {path} swapped the `bird` package to "
            f"{_bird_check.__file__} -- refusing to continue against a "
            "different checkout")
    _MODULE_CACHE[path] = mod
    return mod


def _resolve(record: PolicyRecord, ref: Dict[str, Any]) -> Callable:
    path = _contained(record.root, ref["file"])
    mod = _import_file(path, record.root)
    try:
        return getattr(mod, ref["symbol"])
    except AttributeError:
        raise PolicyError(
            f"{record.id}: `{ref['symbol']}` not found in {path}") from None


def gather_params(record: PolicyRecord) -> Dict[str, Any]:
    """The merged constants the policy actually runs with: kwargs_from file,
    then params_file, then inline params (later wins). Public because the
    evaluation record must carry the SAME merged set -- a record showing
    `null` for a policy that ran with a non-empty JSON-sourced set would be
    the manifest lying about its own measurement."""
    params: Dict[str, Any] = {}
    entry = record.entry or {}
    for rel in (entry.get("kwargs_from"), record.params_file):
        if rel:
            with open(_contained(record.root, rel), "r", encoding="utf-8") \
                    as fh:
                loaded = json.load(fh)
            if not isinstance(loaded, dict):
                raise PolicyError(f"{record.id}: {rel} is not a JSON object")
            params.update(loaded)
    if record.params:
        params.update(record.params)
    return params


def load_policy(policy_id: str, *, root: Optional[Path] = None) -> Policy:
    """Manifest in, uniform ``Policy`` out.

    Importing the campaign code happens here, so this call needs the policy's
    family runtime installed (``record.family``); the *loader* side
    (``bird.policies.index``) never does.

    ``root`` is the policies directory to resolve ``policy_id`` in, forwarded
    unchanged to ``bird.policies.get``, so the loader can be pointed at a
    second registry.  Keyword-only, so every positional caller is untouched,
    and it does NOT relax containment: ``_contained`` still refuses a file
    reference that escapes its own campaign dir, whichever root the campaign
    was found under.  A caller that loads policies from a directory it manages
    must pass that directory, never the repo's own ``policies/``, because
    ``index()`` raises repo-wide on one malformed manifest.
    """
    record = get_record(policy_id, root)
    factory = _resolve(record, record.entry)
    params = gather_params(record)
    if record.pattern == "closure":
        return ClosurePolicy(factory, params, record=record)
    rig_factory = _resolve(record, record.rig) if record.rig else None
    return RigPolicy(factory, params, rig_factory=rig_factory, record=record)


def episode_horizon(env: Any, max_steps: Optional[int] = None) -> int:
    """The loop bound ``run_episode`` runs to: ``max_steps`` when it is set (a
    cap -- or a lengthening; the number is what is recorded either way), else
    the adapter's own horizon.  ``0`` and ``None`` both mean the adapter's.  One
    expression, here, so the record ``scripts/eval_policy.py`` writes and the
    loop it describes cannot disagree about how long an episode was."""
    return int(max_steps or getattr(env, "horizon", 1000))


def run_episode(env: Any, policy: Policy, seed: int,
                max_steps: Optional[int] = None) -> Dict[str, Any]:
    """One episode of ``policy`` on ``env`` (a ``bird`` EnvAdapter), scored by
    the adapter -- the instrument of record.  Returns the standardized row
    shape ``scripts/eval_policy.py`` writes, plus the state trajectory under
    ``"states"`` (stripped before serialisation).

    ``terminated_early`` is the ENVIRONMENT's word: True when ``env.step``
    reported done before the loop bound, False when the episode ran to the
    bound -- ``episode_horizon(env, max_steps)``, the adapter's horizon or the
    cap.  So a row a ``max_steps`` cap ended reads exactly like a full-horizon
    row here, by design: the row does not know what bounded it.  The record
    beside the rows carries ``horizon`` and ``max_steps`` (the writer does),
    and that is where a capped measurement is told from a full
    one -- a per-step fraction over 200 steps is not the task metric over 1000.

    The episode starts from the adapter's own reset under ``seed``; the row
    records ``start: "reset"`` and ``loop_calls`` (the steps and the one reset
    the loop itself made)."""
    rng = np.random.default_rng(seed)
    s = env.reset(rng)
    policy.reset(np.random.default_rng(seed))
    horizon = episode_horizon(env, max_steps)
    states = [np.asarray(s, dtype=float).copy()]
    done = False
    t = 0
    actions = []
    while not done and t < horizon:
        a = policy.act(s, t=t, env=env)
        s, done, _info = env.step(s, a)
        states.append(np.asarray(s, dtype=float).copy())
        actions.append(np.asarray(a, dtype=float).copy())
        t += 1
    traj = np.asarray(states)
    metric = float(env.task_metric(traj))
    # The environment's own reward summed over the episode, beside the task metric. On the
    # HumanoidBench tier `reference_reward` IS the benchmark's shipped reward, so this is the
    # episode return its published baselines (FastTD3 et al.) report -- the one scale on which
    # a registry policy and a published number can be compared like for like. Convention as
    # training.py's `gt_return`: the reward of the arrived-at state under the action taken.
    # Computed AFTER the rollout from the recorded states and actions, never inside the loop:
    # on a MuJoCo adapter `reference_reward` restores its own state into the simulator, and a
    # call between two steps perturbs the next step enough to move contact-rich scores, so
    # the return would have been measured on a different trajectory from the score beside it.
    # None when the adapter has no reference reward, never 0.0: an unmeasured return must stay
    # distinguishable from a measured zero (the same rule `.fitness` follows for CARD).
    ref_return: Optional[float] = 0.0
    try:
        for k, a_k in enumerate(actions):
            ref_return += float(env.reference_reward(traj[k + 1], a_k))
    except NotImplementedError:
        ref_return = None
    return {
        "policy": policy.id,
        "seed": int(seed),
        "start": "reset",
        # Steps and resets this loop made, for the record.
        "loop_calls": {"steps": int(t), "resets": 1},
        "steps": int(t),
        "terminated_early": bool(done),
        "metric": metric,
        "success": bool(env.success(traj)),
        "reference_return": ref_return,
        "states": traj,
    }
