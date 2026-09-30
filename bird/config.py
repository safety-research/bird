"""Config loading, `extends:` resolution, and validation.

Three rules:

1. **Lists replace, they do not append.** Appending `verify.dynamic_checks`
   down an `extends` chain silently changes the method being run, which is
   exactly the failure this project exists to prevent.
2. **Unknown keys are a hard error.** A typo'd key that silently falls back to
   its default quietly invalidates an ablation.
3. **The resolved config is an artifact.** It is written out in full and
   hashed; the hash is the run ID.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
from pathlib import Path
from typing import (Any, Dict, FrozenSet, List, Mapping, Optional, Sequence,
                    Tuple)

from . import paths

import yaml

from .schema import SCHEMA, Field

#: `configs/`, resolved on USE rather than at import -- see the same note on
#: `tasks._tasks_root`. There is deliberately no fallback to `parent.parent /
#: "configs"`: under a non-editable install that is `site-packages/configs`,
#: a plausible wrong path. A fallback would be worse here than in tasks.py,
#: because `configs/` IS shipped -- so a broken symlink or a dropped
#: `package-data` glob would be hidden behind a path that merely does not
#: exist, instead of failing where the packaging broke.
def _config_root() -> Path:
    root = paths.data_dir("configs", required=False)
    if root is None:
        raise ConfigError(
            f"cannot find the `configs/` directory. It ships with the wheel "
            f"as package data, so this means the install is broken rather "
            f"than incomplete -- check that `bird/_data/configs` is present. "
            f"Failing that, set {paths.DATA_ROOT_ENV} to a directory "
            f"containing `configs/`, or install with `pip install -e`.")
    return root


def __getattr__(name: str):
    if name == "CONFIG_ROOT":
        return _config_root()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

#: THE LAZY NAME IS NOT VISIBLE TO INTROSPECTION WITHOUT THIS. PEP 562's
#: `__getattr__` is consulted only after the module dict misses, so
#: `dir(module)` -- which reads the dict -- omits `CONFIG_ROOT` entirely, and so do
#: tab-completion, `inspect.getmembers`, and any tool that enumerates a
#: module's surface to decide what it exports. The name works and looks
#: absent, which is the same failure shape this whole change exists to
#: remove, one level up.
#:
#: AND A WARNING FOR CALLERS: `hasattr(module, "CONFIG_ROOT")` is NOT a safe probe
#: here. `hasattr` swallows only `AttributeError`, and the resolver raises
#: `DataRootMissing` when the directory cannot be found -- so on the machine this
#: matters on, `hasattr` PROPAGATES rather than returning False. That is the
#: correct behaviour (absent and unresolvable are different questions) but it
#: is the opposite of what `hasattr` is usually reached for. To ask "is there
#: a configs directory?" without raising, call
#: `paths.data_dir("configs", required=False) is not None`.
def __dir__():
    return sorted(list(globals()) + ["CONFIG_ROOT"])

DEFAULT_FILE = "_default.yaml"


#: `train.algorithm` values `train.backend: sb3` has a learner for -- the keys
#: of `training._sb3_algo_and_hyper`'s class map, kept HERE so `_check_coherence`
#: can refuse the complement without importing torch, and read by the backend
#: so the two cannot disagree (`tests/test_budget_and_stage_guards.py`).
#: `qr_sac` is GT's citation and runs as plain SAC (the quantile critic is not
#: implemented; the seed rows say `sac`).
SB3_ALGORITHMS: tuple = ("ppo", "sac", "td3", "qr_sac")


class ConfigError(ValueError):
    pass


# --------------------------------------------------------------------------
# dotted-path helpers
# --------------------------------------------------------------------------


def flatten(d: Mapping, prefix: str = "") -> Dict[str, Any]:
    """Flatten to dotted leaf paths. A dict-valued *schema leaf* (e.g.
    `train.hyperparameters`) is itself a leaf and is not descended into."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        path = f"{prefix}{k}"
        if isinstance(v, dict) and v and path not in SCHEMA:
            out.update(flatten(v, path + "."))
        else:
            out[path] = v
    return out


def unflatten(flat: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for path, value in flat.items():
        node = out
        parts = path.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return out


def deep_merge(base: Mapping, override: Mapping) -> Dict[str, Any]:
    """Recursive merge. Mappings merge; **everything else replaces**, lists
    included (rule 1)."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------
# Config object
# --------------------------------------------------------------------------


class Config:
    """Attribute+dotted access over a fully-resolved config dict."""

    def __init__(self, data: Dict[str, Any], source: str = "", lineage: Optional[List[str]] = None):
        self._data = data
        self.source = source
        self.lineage = lineage or []

    # -- access --

    def __getattr__(self, item: str) -> Any:
        try:
            v = self._data[item]
        except KeyError:
            raise AttributeError(
                f"no config key {item!r} at top level (available: {sorted(self._data)})"
            ) from None
        return Config(v) if isinstance(v, dict) else v

    def __getitem__(self, path: str) -> Any:
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                raise ConfigError(f"no config key {path!r}")
            node = node[part]
        return node

    def get(self, path: str, default: Any = None) -> Any:
        try:
            return self[path]
        except ConfigError:
            return default

    def __contains__(self, path: str) -> bool:
        try:
            self[path]
            return True
        except ConfigError:
            return False

    def __repr__(self) -> str:
        return f"Config(name={self._data.get('name')!r}, keys={len(self.flat())})"

    # -- serialisation --

    def to_dict(self) -> Dict[str, Any]:
        return json.loads(json.dumps(self._data))

    def flat(self) -> Dict[str, Any]:
        return flatten(self._data)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self._data, sort_keys=False, default_flow_style=False)

    def hash(self) -> str:
        blob = json.dumps(self._data, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def diff(self, other: "Config") -> Dict[str, Tuple[Any, Any]]:
        a, b = self.flat(), other.flat()
        return {
            k: (a.get(k), b.get(k))
            for k in sorted(set(a) | set(b))
            if a.get(k) != b.get(k)
        }


def reward_language(cfg: Any) -> str:
    """`generate.reward_language`, defaulting to `numpy`.

    THE ONE READER. The prompt's signature clause, the two exec namespaces
    (`verification._restricted_globals`, `training.compile_reward` /
    `compile_observation`), the import allowlist and the batched dynamic probe
    all ask this function and never the literal key, so a config written before
    the key existed -- or a test's bare `SimpleNamespace` cfg -- reads `numpy`
    everywhere at once rather than `numpy` in four places and `None` in a fifth.
    `cfg` is duck-typed on `.get` for that reason: a resumed run's frozen config
    and a unit test's stub both reach here.
    """
    get = getattr(cfg, "get", None)
    value = get("generate.reward_language") if callable(get) else None
    return str(value or "numpy")


def effective_backend(cfg: "Config") -> Any:
    """The learner that executes: `train.backend`.

    Every cross-key rule that asks "which learner is this?" goes through this
    one function rather than reading the literal key, so there is a single
    place to change if the learner ever stops being named by `train.backend`
    alone. The rule where the answer is load-bearing:
    `interaction_cfg.shared_buffer` is refused on fasttd3 (the backend neither
    restores nor exports a slice's replay buffer, `fasttd3_backend`).

    Returns whatever the key holds, unvalidated, so a bad value reaches the
    schema's error rather than being coerced into a plausible one here.
    """
    return cfg.get("train.backend")


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"config {path} must be a mapping at top level")
    return data


def _resolve_extends(path: Path, seen: Optional[List[Path]] = None) -> Tuple[Dict[str, Any], List[str]]:
    """Walk the single-inheritance `extends:` chain, oldest ancestor first."""
    seen = seen or []
    path = path.resolve()
    if path in seen:
        chain = " -> ".join(p.name for p in seen + [path])
        raise ConfigError(f"circular extends chain: {chain}")
    seen = seen + [path]

    raw = _read_yaml(path)
    parent_ref = raw.pop("extends", None)
    if parent_ref is None:
        return raw, [path.name]

    parent_path = (path.parent / parent_ref).resolve()
    if not parent_path.exists():
        parent_path = (_config_root() / parent_ref).resolve()
    parent_data, lineage = _resolve_extends(parent_path, seen)
    return deep_merge(parent_data, raw), lineage + [path.name]


#: The keys whose `null` means "take it from the task spec" -- i.e. the config keys that
#: are ENVIRONMENT FACTS rather than behaviour choices.
#:
#: Named once, here, because several things read it: `_inherit_from_task_spec` below
#: and the coherence checks in `_check_coherence`, each of which iterates this tuple
#: rather than a hand-maintained whitelist. Deriving them from one tuple is what keeps
#: them in step: a key added here is inherited from the spec everywhere at once, and a
#: §1-§6 behaviour key never becomes spec-supplied by someone editing a set literal.
TASK_SPEC_KEYS: Tuple[str, ...] = (
    "problem.task_description",
    "verify.forbidden_symbols",
    "rapp.parameters",
)


def forbidden_symbols_of(spec: Any) -> List[str]:
    """The leak gate a task supplies to THIS consumer.

    `in`, not truthiness. A spec that declares `forbidden_symbols_by_consumer: {bird: []}`
    is saying "this task needs no gate here", and `by_consumer.get("bird") or <upstream>`
    reads that deliberate empty list as absent and silently substitutes the benchmark's.
    This repo already treats that distinction as load-bearing everywhere else --
    `_check_coherence` documents that an empty `verify.forbidden_symbols` is a real choice
    (`singh_orp` pins `[]`), and `artifacts.py` keeps `failure: ""` distinct from a reason.

    One function because the expression was duplicated: once here and once in the
    coherence gate, which then had to agree with it about what "supplied" means.
    """
    reward = getattr(spec, "reward", None) or {}
    by_consumer = reward.get("forbidden_symbols_by_consumer") or {}
    if "bird" in by_consumer:
        return list(by_consumer["bird"] or [])
    return list(reward.get("forbidden_symbols") or [])


#: A task-spec key is only consulted when its GUARD holds. The guards live here, in
#: one place, so the RESOLVER and the VALIDATOR cannot carry diverging copies of the
#: condition: a validator missing the `rapp.enabled` guard would refuse, on an env with
#: no spec, every config whose `rapp.parameters: null` resolves to `[]` and needs no
#: spec at all.
_TASK_SPEC_KEY_GUARDS: Dict[str, Any] = {
    "rapp.parameters": lambda g: bool(g("rapp.enabled")),
}


def _flat_getter(data: Dict[str, Any]):
    """`g("a.b")` over a nested dict, so a guard reads the same either side."""
    def g(key: str) -> Any:
        cur: Any = data
        for part in key.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
        return cur
    return g


def _task_spec_key_applies(key: str, g) -> bool:
    guard = _TASK_SPEC_KEY_GUARDS.get(key)
    return guard is None or bool(guard(g))


def _inherit_from_task_spec(data: Dict[str, Any]) -> Dict[str, Any]:
    """Fill the keys whose `null` means "take it from the task spec".

    MATERIALISED here, into the resolved dict, rather than resolved lazily at each read.
    Three consequences, and each is the reason:

      * `problem.task_description` reaches THIRTEEN prompt sites -- the generator's `##
        TASK` head, the VLM judge's rubric, the decomposition seed, the preference
        comparator. A lazy accessor would be thirteen edits and one of them would be
        missed, and the miss would render the string "None" into a prompt.
      * `verify.forbidden_symbols` is read by `_check_coherence`'s anti-leakage gate as
        `g(...) or []`, which is falsy on None. Materialising means the null never
        reaches that gate, so the one check `_check_coherence` itself calls "dangerous
        rather than merely untidy" keeps working on the EFFECTIVE list.
      * the resolved config is the run's artifact. A `task_description: null` in
        `config.resolved.yaml` would record that the run had no instruction, which is
        false, and `--diff` between two methods would hide the axis GT actually varies.

    What is NOT materialised is everything the ADAPTER reads -- prose, the observation
    surface, the anchors. Those stay in the file, and `_assert_same_config` refuses to
    adopt a run directory across a change to it.
    """
    from .tasks import TaskSpecError, by_env_id, load as load_spec

    problem = data.get("problem") or {}
    task_id, env_id = problem.get("task_id"), problem.get("env_id")
    try:
        spec = load_spec(str(task_id)) if task_id else by_env_id(str(env_id))
    except TaskSpecError:
        return data                      # reported by `_check_coherence`, with its name

    if spec is None:
        return data

    out = json.loads(json.dumps(data, default=str))
    if (out.get("problem") or {}).get("task_description") is None:
        # `TaskSpec.instruction` owns the l_task -> natural_language fallback (the
        # gymnasium specs keep their originating catalogue's single-field shape, where
        # `natural_language` IS the instruction); reading the fields here instead
        # would be a second copy of that rule, free to disagree with the first.
        # Without it every `gym_*` env would materialise `task_description: null`
        # and the prompt's first line would read "None".
        out.setdefault("problem", {})["task_description"] = spec.instruction
    if (out.get("verify") or {}).get("forbidden_symbols") is None:
        # This consumer's gate first, the benchmark-symbol list second. They are different
        # SETS, not nested ones: four of BIRD's six on Meta-World name adapter internals
        # (`task_metric`, `reference_reward`, `reward_source`, `_env`), which are not
        # properties of the task -- they are how a candidate could reach the metric it is
        # scored on IN THIS REPO. Inheriting upstream's list instead would leave one
        # config rejecting a different set from configs it should be comparable to,
        # which invalidates the comparison.
        gate = forbidden_symbols_of(spec)
        if not gate:
            # The spec supplied nothing. On the gymnasium ten that emptiness is the
            # core-shaped spec declaring no gate, not a review of what a candidate can
            # reach here -- so the ADAPTER
            # advertises the gate on the factory
            # (`gym_mujoco._CONSUMER_FORBIDDEN`) and this is where it is picked up.
            # `-s problem.env_id=gym_half_cheetah` thereby inherits the same
            # four-symbol gate an `mt10_*` override inherits from its spec. An env whose factory advertises
            # nothing keeps the spec's answer, empty included -- for the natives,
            # an empty gate is the spec's own true statement. An AUTHORED list
            # (even `[]`, the `singh_orp` idiom) never reaches this branch.
            from .registry import RegistryError, get as _get_env  # lazy: cycle
            try:
                factory = _get_env("env", str(spec.bird_env_id))
            except (RegistryError, KeyError):
                factory = None
            gate = list(getattr(factory, "consumer_forbidden_symbols", None)
                        or []) or gate
        out.setdefault("verify", {})["forbidden_symbols"] = gate
        # AND ITS READER, which is the same rule the `problem.env_id` block above
        # states as "A DERIVED KEY MUST FOLLOW ITS BASE KEY". `verify.static_checks`
        # is a derived key of this one: a gate is a list of symbols and
        # `forbidden_symbols` is the check that reads it, so supplying the first
        # without the second supplies nothing.
        #
        # WHY. `_check_coherence` refuses a non-empty gate whose reader is not in
        # the list -- correctly, it is the one check in that function it calls
        # "dangerous rather than merely untidy". But every published config resolves
        # `static_checks: [signature_parse, ast_syntax]`, and every MuJoCo task spec
        # supplies a gate. The env axis is an override (`-s problem.env_id=...`)
        # with no file to add the check in, so without this `-s
        # problem.env_id=mt10_reach-v3` would be REFUSED on `card`, `rda`, `limen`
        # and `limen_reward_only`:
        #
        #   verify.forbidden_symbols is in force but 'forbidden_symbols' is not in
        #   verify.static_checks, so nothing would ever check it
        #
        # i.e. those methods could not be run on the Meta-World tier at all.
        # `eureka` is the exception and not the counter-example: it passes only
        # because `verify.enabled: false`, so its gate is out of force. The gate
        # and its reader belong together, and this pairs them at the one place
        # that knows the gate came from the spec rather than from an author.
        #
        # ONLY ON THE SPEC-SUPPLIED PATH, and that is why it lives inside this
        # branch. Reaching here means `verify.forbidden_symbols` was null, so an
        # AUTHORED list -- `[]` the `singh_orp` way, or a real list with a
        # deliberately short `static_checks` -- never gets its author's §2 edited
        # underneath them. An authored gate with no reader stays a validation error
        # with a file to fix it in, which is the correct outcome there.
        #
        # It changes the resolved config (and so the hash) only for runs that would
        # otherwise be refused, so no existing run directory can collide with it.
        # Every published config's hash is unchanged on its own env.
        if gate and out["verify"].get("enabled") and \
                "forbidden_symbols" not in (out["verify"].get("static_checks") or []):
            # Appended, never inserted. `verify.check_order: staged` short-circuits
            # in list order, so a position would be a claim about which check should
            # cost first, and that claim belongs to the method. Last is the only
            # position that leaves the author's own ordering intact.
            out["verify"]["static_checks"] = (
                list(out["verify"].get("static_checks") or []) + ["forbidden_symbols"])
    if ((out.get("rapp") or {}).get("parameters") is None
            and _task_spec_key_applies("rapp.parameters", _flat_getter(out))):
        # ONLY when the RAPP sweep will actually run. `rapp.parameters` names the axes
        # that sweep varies, so for a config with `rapp.enabled: false` the correct
        # resolved value is "none" -- filling in the environment's axes there would state
        # that a run varies six things it will never touch.
        #
        # It also keeps the published configs' resolved values stable: `configs/methods/`
        # holds the published values verbatim, and that is the whole basis on which
        # `--diff eureka rda` means "what is this paper's contribution?". Without this
        # condition every config with RAPP disabled would resolve the key to the env's
        # axes instead of `[]` -- inert, but a changed resolved artifact for no reason
        # anyone could state from the file itself.
        dr = (spec.domain_randomization or {}).get("parameters") or {}
        out.setdefault("rapp", {})["parameters"] = (
            list(dr) if (out.get("rapp") or {}).get("enabled") else [])
    return out


def _find_config(config_path: str | os.PathLike) -> Path:
    """Resolve a config REFERENCE (what `--config` / `load()` is given) to a file.

    Two rules, in order, both relative to `configs/` (`.yaml` is optional):

    1. An exact relative path: `methods/eureka`, `hillclimb/v2_verify.yaml`,
       `era_u`.
    2. For a bare name (no directory part), the unique file called
       `<name>.yaml` anywhere under `configs/`, outside underscore-prefixed
       directories (`_profiles/` holds execution profiles, which are not
       configs): `eureka` finds `methods/eureka.yaml` and `v2_verify` finds
       `hillclimb/v2_verify.yaml`.

    A bare name that matches several files is an error that lists them, never
    a silent pick: a name that resolved to whichever file a directory walk met
    first would change the method being run when someone added a file
    elsewhere. `tests/test_config.py` checks that every shipped config is
    reachable by its bare name, so in the shipped tree no name is ambiguous.
    """
    root = _config_root()
    ref = str(config_path)
    filename = ref if ref.endswith(".yaml") else ref + ".yaml"
    for rel in (ref, filename):
        if (root / rel).is_file():
            return root / rel
    if len(Path(ref).parts) == 1:
        matches = sorted(
            p for p in root.rglob("*.yaml")
            if p.name == filename
            and not any(part.startswith("_") for part in p.relative_to(root).parent.parts))
        if len(matches) == 1:
            return matches[0]
        if matches:
            listing = ", ".join(f"configs/{p.relative_to(root)}" for p in matches)
            raise ConfigError(
                f"config name {ref!r} is ambiguous: it matches {listing}. Pass the "
                f"path relative to configs/ instead (e.g. "
                f"{str(matches[0].relative_to(root).with_suffix(''))!r}).")
    raise ConfigError(f"config not found: {config_path}")


#: Where execution profiles live. A profile is the SECOND parent a method config
#: needs and `extends:` cannot give it: `extends:` is single inheritance -- one
#: chain, walked once -- so "this method" and "run it this way" cannot both be
#: parents of one file. Without that second axis every method would need one
#: copy per tier (methods x tiers files); a profile supplies it as a LAYER
#: rather than a parent: it is merged in below the whole `extends:` chain, so a method config still
#: says what the method is and always wins on any key it states.
PROFILE_DIR = "_profiles"


def _profile_root() -> Path:
    return _config_root() / PROFILE_DIR


def available_profiles() -> List[str]:
    root = _profile_root()
    if not root.is_dir():
        return []
    return sorted(f.stem for f in root.glob("*.yaml") if not f.name.startswith("_"))


def _profile_name(name: str) -> str:
    """Normalise a profile name to its bare stem.

    A bare name, never a path: the profile is recorded in the resolved config
    (and therefore in `Config.hash()` and the run directory name), so `dev`
    and `dev.yaml` reaching the hash as two strings would be two run IDs for
    one run -- the same argument `_check_coherence` makes for accepting exactly
    `auto` and not `AUTO` on `loop.max_parallel_trainings`.
    """
    stem = str(name)
    if stem.endswith(".yaml"):
        stem = stem[: -len(".yaml")]
    return stem


def _find_profile(name: str) -> Path:
    """Resolve a profile NAME to `configs/_profiles/<name>.yaml`."""
    stem = _profile_name(name)
    path = _profile_root() / f"{stem}.yaml"
    if not path.exists():
        have = available_profiles()
        listing = ", ".join(have) if have else "nothing"
        raise ConfigError(
            f"profile not found: {name} (configs/{PROFILE_DIR}/ holds: {listing})")
    return path


#: The keys a profile may state -- and therefore the keys on which a profile
#: OUTRANKS the method config.
#:
#: WHY A PROFILE WINS, when `extends:` parents lose. A profile answers "how do I
#: execute this", the method answers "what IS this". Those are different
#: questions and the second must not be able to veto the first: `eureka.yaml`
#: pins its published 393,216,000 env steps and 16x5x5 candidates, so a profile
#: layered UNDERNEATH it would be inert on exactly the keys a profile exists to
#: set: `-c eureka --profile tester` would resolve to 400 trainings at 393 M
#: steps rather than a handful at 2,000 -- a tester tier in name only.
#:
#: WHY IT IS AN ALLOWLIST AND NOT "EVERYTHING THE PROFILE SAYS". Winning is
#: dangerous in the other direction: a profile that stated `train.algorithm`
#: would silently run Eureka's PPO point as SAC, and `verify.enabled` would turn
#: the no-verification ablation into a verified one -- a file that reads like
#: a re-run and is actually a second experiment, with nothing in the method
#: config to say so. So a
#: profile may set the machine, the budget and the reporting, and NOTHING that
#: appears in a method's own paper.
#:
#: Absent on purpose: `train.algorithm`, `train.architecture`,
#: `train.hyperparameters`, every `verify.*`, `generate.*`, `evaluate.*`,
#: `select.*`, `update.*` and `problem.*`. Each is a §1-§6 behaviour key, i.e.
#: the thing being compared.
PROFILE_KEY_PREFIXES: Tuple[str, ...] = (
    "llm.",                        # which model answers, and how
    "output.",                     # tracker, video, wandb -- reporting only
    "budget.",                     # spend caps
    "train.backend",               # what actually runs: mock | tabular | sb3 | none |
                                   # fasttd3 | simba_v2 | assistax_ppo. WHICH learner
                                   # executes the method's algorithm is execution, not
                                   # method identity (`train.algorithm` is the citation)
    "train.env_steps",
    "train.n_parallel_envs",
    "train.timeout_s",
    "train.candidate_parallelism",  # a SCHEDULE, held bit-identical by tests/test_parallelism.py
    "loop.n_iterations",
    "loop.n_restarts",
    "loop.max_parallel_trainings",
    "loop.resume_from",
    # THE PHYSICS-PRIOR SWEEP'S ROLLOUT COUNT is a cost, not a method identity: DrEureka's
    # RAPP sweeps RAPP_SWEEP_POINTS values per DR axis and rolls the incumbent policy
    # `rollouts_per_value` times at each. `EnvAdapter.dr_probe` makes that sweep live on
    # every adapter, and at the paper's 100 x 9 points x 8 toy axes a `dreureka` tester
    # point would run 7,200 rollouts. The tester profile sets it, the way it sets
    # `train.env_steps`.
    "rapp.rollouts_per_value",
    # An execution CAP on the report protocol, not the protocol itself: the method
    # keeps `final_retrain.env_steps` (a paper's budget, e.g. eureka's 2.62 B) and
    # the profile says how much of it a run may actually spend; the phase records
    # both. Without it a dev/full run of a config that pins a published retrain
    # budget trains 5 x 2.62 B steps.
    "final_retrain.max_env_steps",
    # THE ONE EXCEPTION, and it is a safety gate rather than a method choice.
    # `verify.forbidden_symbols` is a TASK_SPEC_KEY: a real environment supplies
    # its own leak list, and `_check_coherence` then REFUSES a config that has
    # symbols to block and no `forbidden_symbols` check to block them with. So a
    # profile that puts real generated code in front of a real env has to be able
    # to arm the gate whether or not the paper it is running had one.
    # Deliberately NOT extended to the rest of `verify.*`: a screen, a repair
    # policy or a dedup metric is the method, and a profile setting one would be
    # running a different method under the same name.
    "verify.enabled",
    "verify.static_checks",
)


def _profile_leaves(data: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    """A profile's dotted leaf keys, flattened."""
    out: Dict[str, Any] = {}
    for k, v in (data or {}).items():
        name = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_profile_leaves(v, name))
        else:
            out[name] = v
    return out


def _check_profile_scope(name: str, leaves: Mapping[str, Any]) -> None:
    """Refuse a profile that states a method-defining key."""
    bad = sorted(k for k in leaves
                 if k not in ("name", "profile", "extends")
                 and not any(k == p or k.startswith(p) for p in PROFILE_KEY_PREFIXES))
    if bad:
        raise ConfigError(
            f"profile '{name}' states {len(bad)} key(s) it may not, because a "
            f"profile OUTRANKS the method config and these decide what the "
            f"method IS rather than how it runs:\n  "
            + "\n  ".join(bad)
            + "\n\nA profile may set: " + ", ".join(PROFILE_KEY_PREFIXES)
            + "\nMove anything else into the method config, or pass it as a "
              "sweep override.")


def load(config_path: str | os.PathLike, overrides: Optional[Mapping[str, Any]] = None,
         validate_config: bool = True, profile: Optional[str] = None) -> Config:
    """Load a config: defaults <- extends chain <- file <- profile <- CLI overrides.

    The profile layer sits ABOVE everything the method config says, directly or
    through its `extends:` chain, and BELOW the CLI overrides -- on the keys a
    profile may state, it OUTRANKS the method. That is deliberate and the
    reason is `PROFILE_KEY_PREFIXES`: a profile answers "how do I execute this"
    (budgets, backend, tracker) and the published configs pin their paper
    budgets outright, so a profile layered underneath them would be inert on
    exactly the keys it exists to set (`-c eureka --profile tester` would run
    393 M steps). The method is protected the other way round --
    `_check_profile_scope` refuses a profile that states anything outside that
    allowlist, so a method can never be silently re-pointed on what it IS by
    the tier it happens to be running in.
    `test_a_profile_outranks_the_method_on_its_keys` pins this order.

    Merged with the same `deep_merge` as every other layer, so **lists replace,
    they do not append** (rule 1). A profile that appended to
    `verify.dynamic_checks` would be changing the method being run.

    The profile NAME lands in the resolved config under the top-level `profile`
    key, so it is covered by `Config.hash()` for free -- the hash is over all of
    `_data` and nothing is excluded -- and two profiles of one method get two run
    directories rather than colliding in one.
    """
    path = Path(config_path)
    if not path.is_file():
        path = _find_config(config_path)

    defaults = _read_yaml(_config_root() / DEFAULT_FILE)
    defaults.pop("extends", None)

    lineage: List[str] = []
    merged = defaults
    profile_leaves: Dict[str, Any] = {}
    if profile is not None:
        # `_resolve_extends`, not `_read_yaml`: a profile may itself extend
        # another profile, and going through the same resolver is what keeps one
        # statement of the merge rules rather than two that can disagree.
        profile_data, profile_lineage = _resolve_extends(_find_profile(profile))
        profile_leaves = _profile_leaves(profile_data)
        _check_profile_scope(_profile_name(profile), profile_leaves)
        lineage += [f"{PROFILE_DIR}/{n}" for n in profile_lineage]

    data, config_lineage = _resolve_extends(path)
    merged = deep_merge(merged, data)
    lineage += config_lineage

    if profile_leaves:
        # ABOVE the method, not below it -- see PROFILE_KEY_PREFIXES. Applied as
        # flattened leaves so a profile's `output.video.format` replaces that one
        # key rather than the whole `output.video` mapping.
        merged = deep_merge(merged, unflatten(
            {k: v for k, v in profile_leaves.items()
             if k not in ("name", "profile", "extends")}))

    if profile is not None:
        # Set AFTER the config layers and BEFORE the overrides. A method config
        # does not get to declare which profile it was run under -- that is an
        # operator fact -- and a resolved config reloaded with no `--profile`
        # keeps whatever it already recorded, which is what makes `--resume`
        # reproduce the original hash.
        merged["profile"] = _profile_name(profile)

    if overrides:
        flat_ov = dict(overrides)
        merged = deep_merge(merged, unflatten(flat_ov))
        # A DERIVED KEY MUST FOLLOW ITS BASE KEY. `TASK_SPEC_KEYS` mean "take it
        # from the task spec" only while they are null, and every top-level
        # method config states `problem.task_description` outright -- it is
        # pendulum's, because that is the environment the published point was
        # written against. So `-s problem.env_id=mt10_window-open-v3` alone would
        # move the simulator and leave the PROSE describing a pendulum: the
        # model is told to swing up a rod while a Sawyer arm runs, and nothing
        # fails -- a silently wrong experiment rather than a crash.
        #
        # A config file avoids it by pinning `task_description: null` beside
        # the env, so the spec refills it. An override cannot pin a null it was
        # not asked for, so changing the env RE-ARMS the inheritance here: any
        # task-spec key the caller did not name itself goes back to null and is
        # refilled below from the new environment's spec. Naming one explicitly
        # still wins, which is what keeps a deliberate custom instruction
        # possible.
        if "problem.env_id" in flat_ov:
            for key in TASK_SPEC_KEYS:
                if key not in flat_ov:
                    merged = deep_merge(merged, unflatten({key: None}))

    merged = _inherit_from_task_spec(merged)

    cfg = Config(merged, source=str(path), lineage=lineage)
    if validate_config:
        validate(cfg)
    return cfg


def parse_overrides(texts: Sequence[str]) -> Dict[str, Any]:
    """Every `--set KEY=VALUE` word -> one mapping, REFUSING a repeated key.

    WHY THIS REFUSES RATHER THAN TAKING THE LAST ONE. A plain
    `dict(parse_override(s) for s in args.set)` resolves a key given twice
    last-wins, silently, with nothing printed and nothing recorded. That is a
    fine rule for a shell flag and a bad one for a key that decides what an
    experiment IS.

    Example: two `--set loop.carry=...` words, one giving
    `["best_reward","policy_checkpoint"]` and one `["best_reward"]`. In one
    order the run is the Eureka baseline; in the other it carries a policy
    checkpoint, the very component the baseline is defined by NOT having. Both
    resolve, both exit 0 and `_check_coherence` refuses neither, so word order
    alone would decide which method ran.

    A duplicate is therefore either a mistake or a composition whose author
    wants last-wins. The first must be caught; the second is better written as
    one flag, and where a caller genuinely composes layers it should resolve
    them into a mapping BEFORE it builds argv -- which is what
    `tests/test_roska_fusion.py::_run` does, and what makes its
    "extra overrides the defaults" intent visible instead of implicit in word
    order.

    No opt-out flag, deliberately: an escape hatch on a guard like this is the
    thing that gets reached for at 2am on the launch that needed the guard.
    """
    out: "Dict[str, Any]" = {}
    seen: "Dict[str, str]" = {}
    for text in texts:
        key, value = parse_override(text)
        if key in out:
            if value == out[key]:
                why = ("Both give the same value, so nothing would have changed "
                       "TODAY -- which is a property of the two layers that "
                       "happen to be composed here and not of this command. The "
                       "next edit to either makes them disagree, and a guard "
                       "taught to stay quiet on the equal case would already be "
                       "quiet for the one that matters.")
            else:
                why = (f"The second would have won, silently: you would have got "
                       f"{value!r}, not {out[key]!r}.")
            raise ConfigError(
                f"--set {key} given twice: {seen[key]!r} then {text!r}. {why} Give "
                f"the key once with the value you mean, or resolve the layers into "
                f"one mapping before building the command line. A repeated key that "
                f"decides an experiment is the defect this refuses -- see the "
                f"`parse_overrides` docstring for why.")
        out[key] = value
        seen[key] = text
    return out


def parse_override(text: str) -> Tuple[str, Any]:
    """`--set generate.n_candidates=16` -> ("generate.n_candidates", 16)."""
    if "=" not in text:
        raise ConfigError(f"override must be key=value, got {text!r}")
    key, _, raw = text.partition("=")
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError:
        value = raw
    return key.strip(), value


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def _allowed(field: Field) -> Optional[Tuple[Any, ...]]:
    if field.enum is not None:
        return field.enum
    if field.kind is not None:
        from .registry import names  # local import: registry imports components
        return tuple(names(field.kind))
    return None


def _allowed_items(field: Field) -> Optional[Tuple[Any, ...]]:
    if field.item_enum is not None:
        return field.item_enum
    if field.item_kind is not None:
        from .registry import names
        return tuple(names(field.item_kind))
    return None


def _near_miss(key: str) -> str:
    """Suggest the key the author probably meant.

    A typo'd key is fatal (rule 2), but a bare "unknown key" is a poor error
    when the intended key is one character away -- and the whole reason the key
    is fatal is that a silent fallback to the default invalidates an ablation.
    Three passes, cheapest first: an exact leaf-name match under a different
    parent, a fuzzy match on the leaf name within the same parent, and finally
    a fuzzy match on the whole dotted path.
    """
    leaf = key.rsplit(".", 1)[-1]
    parent = key.rsplit(".", 1)[0] if "." in key else ""

    exact_leaf = [k for k in SCHEMA if k.rsplit(".", 1)[-1] == leaf]
    if exact_leaf:
        return f" (did you mean {exact_leaf[0]!r}?)"

    siblings = {k.rsplit(".", 1)[-1]: k for k in SCHEMA
                if "." in k and k.rsplit(".", 1)[0] == parent}
    close = difflib.get_close_matches(leaf, list(siblings), n=1, cutoff=0.6)
    if close:
        return f" (did you mean {siblings[close[0]]!r}?)"

    close = difflib.get_close_matches(key, list(SCHEMA), n=1, cutoff=0.6)
    if close:
        return f" (did you mean {close[0]!r}?)"
    return ""


def validate(cfg: Config, strict_enums: bool = True) -> None:
    """Raise ConfigError listing *every* problem, not just the first."""
    flat = cfg.flat()
    problems: List[str] = []

    # rule 2: unknown keys are fatal
    for key in flat:
        if key not in SCHEMA:
            problems.append(f"unknown key {key!r}{_near_miss(key)}")

    for key, value in flat.items():
        field = SCHEMA.get(key)
        if field is None:
            continue
        if value is None:
            if not field.nullable:
                problems.append(f"{key}: may not be null")
            continue
        if field.types and field.types != (object,):
            if isinstance(value, bool) and bool not in field.types:
                problems.append(f"{key}: expected {_names(field.types)}, got bool")
                continue
            if not isinstance(value, field.types):
                problems.append(
                    f"{key}: expected {_names(field.types)}, got {type(value).__name__} ({value!r})")
                continue
        if strict_enums:
            allowed = _allowed(field)
            if allowed and value not in allowed:
                problems.append(f"{key}: {value!r} not in {sorted(map(str, allowed))}")
            item_allowed = _allowed_items(field)
            if item_allowed and isinstance(value, list):
                for item in value:
                    probe = item.get("name") if isinstance(item, dict) else item
                    if probe not in item_allowed:
                        problems.append(
                            f"{key}: item {probe!r} not in {sorted(map(str, item_allowed))}")

    problems.extend(_check_coherence(cfg))

    if problems:
        listing = "\n  - ".join(problems)
        raise ConfigError(f"invalid config {cfg.source}:\n  - {listing}")


def _names(types: Tuple[type, ...]) -> str:
    return "|".join(t.__name__ for t in types)


#: On-policy learners, which cannot consume a replay buffer. Named here rather
#: than inverted from an off-policy list so a new off-policy algorithm is not
#: silently treated as on-policy by omission.
_ON_POLICY: Tuple[str, ...] = ("ppo",)


def n_cand_or(g, default):
    """`generate.n_candidates` as an int, or `default` when it is not one. The
    schema's own type check reports a bad value; this only keeps the R* rules
    below from raising a TypeError before it gets the chance."""
    v = g("generate.n_candidates")
    return v if isinstance(v, int) else default


#: `bird.components.training._STORE_LIMIT`, duplicated because this module may
#: not import `bird.components` (pyyaml + stdlib only). Pinned equal
#: by `tests/test_train_init.py`.
_POLICY_STORE_LIMIT = 64

#: Env suites where a torch learner on the CPU is a spending mistake, not a
#: slow run: the paper trained both on GPUs.
#:
#: Keyed on `envs.suites.env_suite` rather than an id prefix: the HumanoidBench
#: suite has several robot prefixes, and a second copy of that list is how one
#: gets missed. The `jax` suite (`upstream_assistax_*`, `jax_toy`) is not listed
#: because its cuda rule is the adapter's own `requires_cuda` declaration,
#: checked under `generate.reward_language` below.
_GPU_SUITES = frozenset({"humanoid_bench", "assistax"})



def _env_spec_required_symbols(member: str) -> Tuple[str, ...]:
    """Which symbols an `env_spec` member shows the model that no adapter defines.

    Lazy import for the reason every other registry read in `_check_coherence`
    is lazy: `config -> registry -> components -> config` is a real cycle, and
    by the time a config is being validated the components module is loaded
    anyway. An unknown member yields `()`, so this rule is silent on every
    value that has nothing for a table to cover -- which is all of them but one.
    """
    try:
        from .components.generation import ENV_SPEC_REQUIRED_SYMBOLS
    except Exception:  # noqa: BLE001 -- a components import failure is not this rule's
        return ()
    return tuple(ENV_SPEC_REQUIRED_SYMBOLS.get(member, ()))


def _symbol_table_keys(spec: Any, g: Any) -> Optional[FrozenSet[str]]:
    """The KEYS the resolved `symbol_mapping` would rewrite, or None for no table.

    All three shapes are resolvable at load, which is what makes coverage
    checkable here rather than only at run time:

      * an inline dict is in hand;
      * `t2r_metaworld_global` is a module constant;
      * `per_task` is the task spec's own `symbol_mapping:` block, and
        `bird/tasks.py` is already loaded during config resolution.

    `None` means "no table at all" and is distinct from an empty frozenset,
    which would be a table that covers nothing -- the caller reports those
    differently because the fixes differ (pin a table, versus pin the right
    one). A spec that cannot be resolved also yields an empty set rather than
    None: unknown coverage is treated as no coverage, because the alternative is
    a rule that goes quiet exactly where the spec is missing.
    """
    if isinstance(spec, dict):
        return frozenset(str(k) for k in spec)
    if not isinstance(spec, str) or not spec.strip():
        return None
    if spec == "per_task":
        try:
            from .tasks import by_env_id as _by_env_id, load as _load
            task_id = str(g("problem.task_id") or "")
            task = _load(task_id) if task_id else _by_env_id(str(g("problem.env_id") or ""))
            return frozenset(str(k) for k in (task.symbol_mapping or {}))
        except Exception:  # noqa: BLE001 -- no spec means no coverage, never a crash
            return frozenset()
    try:
        from .components.generation import _SYMBOL_TABLE_KEYS
    except Exception:  # noqa: BLE001
        return frozenset()
    return frozenset(_SYMBOL_TABLE_KEYS.get(spec, ()))


def _signature_params(value: str) -> Tuple[str, ...]:
    """Parameter names of the signature literal `generate.output.signature` pins.

    Parsed rather than tabulated, because the literals are published text and a
    table of "which params each one has" would be a second copy of a fact that is
    already written down. The published lines are not all valid Python on their
    own -- T2R's is quoted inline in prose and carries no trailing colon -- so a
    colon is supplied before parsing and a body is appended.
    """
    text = value if value.rstrip().endswith(":") else value.rstrip() + ":"
    try:
        import ast as _ast
        tree = _ast.parse(text + "\n    pass\n")
        fn = tree.body[0]
        args = fn.args  # type: ignore[attr-defined]
        return tuple(a.arg for a in (list(args.posonlyargs) + list(args.args)
                                     + list(args.kwonlyargs)))
    except Exception:  # noqa: BLE001 -- an unparseable literal is the enum's problem
        return ()


def _symbol_table_receivers(spec: Any, g: Any) -> FrozenSet[str]:
    """Which variable(s) a rewritten program will index, read off the table's VALUES.

    `obs[4:7]` yields `obs`. A value that is not an index expression on a bare
    name contributes nothing, so a table of literals or method calls makes this
    rule silent rather than wrong.
    """
    values: Sequence[str]
    if isinstance(spec, dict):
        values = [str(v) for v in spec.values()]
    elif isinstance(spec, str) and spec.strip() and spec != "per_task":
        try:
            from .components.generation import _SYMBOL_TABLE_VALUES
        except Exception:  # noqa: BLE001
            return frozenset()
        values = list(_SYMBOL_TABLE_VALUES.get(spec, ()))
    else:
        return frozenset()
    out = set()
    for v in values:
        m = re.match(r"^([A-Za-z_]\w*)\s*\[", v.strip())
        if m:
            out.add(m.group(1))
    return frozenset(out)


def _rescaled_waves_hint(waves: Sequence[Any], steady: int, want: int) -> str:
    """The wave list `generate.n_candidates=want` implies, when the ratio survives.

    WHY THE MESSAGE CARRIES A SUGGESTION. The refusal above is correct -- a wave
    list that disagrees with `n_candidates` is an unreported budget change,
    which is the whole reason the rule exists. But rescaling a wave list is not
    one edit but THREE coupled ones (`n_candidates`, `generate.crossover.n`, and
    the list), and a message that only stated the mismatch would leave the
    reader to work out a paper's population ratio. `configs/methods/rstar.yaml` spells
    the R* case out in a header.

    Deliberately silent unless the rescale is EXACT. A suggestion that rounded
    would be a plausible wrong budget offered by the guard against plausible
    wrong budgets, which is worse than no suggestion: R*'s 12:4 at a population
    of 8 is 6:2 and divides cleanly, and a population the ratio does not divide
    is a decision for whoever owns the comparison, not for this function."""
    if steady <= 0 or want <= 0 or steady % want and want % steady:
        return ""
    scaled = []
    for w in waves:
        if not isinstance(w, dict):
            return ""
        out = {}
        for field in ("llm", "crossover"):
            n = w.get(field, 0) or 0
            if not isinstance(n, int):
                return ""
            if n * want % steady:
                return ""  # not exact for this pass: say nothing
            out[field] = n * want // steady
        out["when"] = str(w.get("when", "always"))
        scaled.append(out)
    if not scaled:
        return ""
    # `crossover.n` is the per-iteration crossover share the waves reserve, so it
    # has to move with them or the next rule down refuses the pair instead.
    x_total = sum(w["crossover"] for w in scaled if w["when"] == "always")
    as_yaml = ",".join("{llm:%d,crossover:%d,when:%s}"
                       % (w["llm"], w["crossover"], w["when"]) for w in scaled)
    return (f". Rescaled to {want} the same ratio is "
            f"-s loop.waves=[{as_yaml}]"
            + (f" -s generate.crossover.n={x_total}" if x_total else ""))


def _chose_an_env(cfg: "Config") -> bool:
    """Did this config pick `problem.env_id`, or is it sitting on the default?

    Read off `_default.yaml` rather than compared against a literal, so the day
    the repo default changes this does not quietly start (or stop) firing. Used
    by one rule -- the `problem.horizon` check, whose comment explains why the
    distinction matters -- and deliberately NOT by anything else: "the config
    did not choose" is a weak signal and the only place it is the right question
    is a check against a resource the chosen env would name.
    """
    try:
        default = _read_yaml(_config_root() / DEFAULT_FILE)
    except Exception:  # noqa: BLE001 -- no defaults file is another failure's job
        return True
    want = ((default.get("problem") or {}).get("env_id")
            if isinstance(default, dict) else None)
    return want is None or str(cfg.get("problem.env_id")) != str(want)


def _check_coherence(cfg: Config) -> List[str]:
    """Cross-key invariants a per-key schema cannot express."""
    p: List[str] = []
    g = cfg.get
    # Every rule below that asks "which learner is this?" asks THIS
    # (`effective_backend`), resolved once, here.
    eff_backend = effective_backend(cfg)

    # -- generate.context.env_spec / postprocess.symbol_mapping -------------
    #
    # `t2r_class_abstraction` shows the model seven symbols -- `self.obj1.position`
    # and siblings -- that the harness does not define: they become executable
    # only through T2R's own general->specific converter,
    # `generate.postprocess.symbol_mapping`. With no table every generated
    # program refers to attributes of nothing and dies in §2, so the pair is
    # refused here rather than discovered as a run of 100% invalid candidates.
    # The string form is validated against the `symbol_table` family in the same
    # place, because the key also accepts an inline dict and so cannot be a
    # `kind=` field (see bird/schema.py on that key).
    _symtab = g("generate.postprocess.symbol_mapping")
    if isinstance(_symtab, str) and _symtab.strip():
        from .registry import names as _names
        if _symtab not in _names("symbol_table"):
            p.append(f"generate.postprocess.symbol_mapping={_symtab!r} is not a "
                     f"symbol_table registry member; expected one of "
                     f"{sorted(_names('symbol_table'))} or an inline dict")
    # COVERAGE, not presence. A presence check (does a table exist?) would pass
    # `t2r_class_abstraction` + `symbol_mapping: per_task`, which is exactly the
    # run this check exists to stop: on this
    # repo `per_task` resolves to the TASK SPEC's block, which is BIRD's own
    # vocabulary, so six of the seven symbols the prompt shows go through
    # unrewritten -- and the seventh is GARBLED. The spec's key `goal_position`
    # matches the tail of `self.goal_position` because
    # `_apply_symbol_mapping` substitutes on word boundaries and `.` is not a
    # word character, so `self.goal_position` becomes `self.s[36:39]`: an
    # attribute access on the state array, `AttributeError` at stage 2,
    # attributed to the model. A clean 100%-invalid run would at least be legible; that one is
    # a garbled program that reads as the generator's fault.
    _required = _env_spec_required_symbols(str(g("generate.context.env_spec") or ""))
    if _required:
        _covered = _symbol_table_keys(_symtab, g)
        if _covered is None:
            p.append(f"generate.context.env_spec={g('generate.context.env_spec')!r} shows "
                     f"the model {len(_required)} general symbol names no adapter defines "
                     f"({_required[0]}, ...); it needs generate.postprocess.symbol_mapping "
                     f"to rewrite them (t2r_metaworld_global is the companion table) or "
                     f"every candidate fails verification")
        else:
            _missing = [sym for sym in _required if sym not in _covered]
            if _missing:
                p.append(
                    f"generate.postprocess.symbol_mapping={_symtab!r} does not cover "
                    f"{len(_missing)} of the {len(_required)} symbols "
                    f"generate.context.env_spec={g('generate.context.env_spec')!r} shows "
                    f"the model: {', '.join(_missing)}. Those names reach the generated "
                    f"program and nothing rewrites them, so the candidate names attributes "
                    f"of nothing -- and a key that is a SUFFIX of one of them rewrites it "
                    f"PARTIALLY (the table substitutes on word boundaries and `.` is not a "
                    f"word character), which produces a plausible wrong program rather than "
                    f"an error. t2r_metaworld_global is the table that covers this member")

    # THE SEAM BETWEEN THE ABSTRACTION AND THE CONTRACT.
    #
    # `t2r_class_abstraction` shows attributes on `self`; the table rewrites
    # exactly those `self.`-prefixed names into an index expression on a
    # variable (`obs[4:7]`). Both halves therefore depend on
    # `generate.output.signature`, which is what names the function's
    # parameters: a signature with no `self` makes the prompt's own symbols
    # ungrammatical in the function being asked for, and a signature without the
    # table's receiver makes the REWRITTEN program a NameError. Neither is
    # visible in any single key.
    #
    # Observed with the default signature: gpt-4-turbo obeyed every instruction,
    # used only the six attributes the abstraction lists, and bridged the missing
    # receiver itself -- `env = state[0]`, then `env.obj1.position`. Nothing in
    # the table matches `env.`, so the program reached an attribute of a float
    # and died in `execution_smoke`; repair resends this same contradiction and
    # cannot converge, so each candidate spends ten LLM calls before aborting. A
    # refusal at load is the cheap version of that discovery.
    _sig_value = str(g("generate.output.signature") or "")
    if _required and _sig_value:
        try:
            from .components.generation import _SIGNATURES as _SIGS
            _sig_line = _SIGS.get(_sig_value, "")
        except Exception:  # noqa: BLE001
            _sig_line = ""
        _params = _signature_params(_sig_line) if _sig_line else ()
        if _params and _params[0] != "self":
            p.append(
                f"generate.context.env_spec={g('generate.context.env_spec')!r} shows the "
                f"model attributes on `self` ({_required[0]}, ...), but "
                f"generate.output.signature={_sig_value!r} pins "
                f"`{_sig_line}`, whose first parameter is {_params[0]!r}. The model is "
                f"then asked for a function in which the prompt's own symbols do not "
                f"resolve, and it will bridge the gap itself with a receiver the table "
                f"cannot rewrite (measured: `env = state[0]`). Pin "
                f"generate.output.signature=t2r_compute_dense_reward (T2R) or "
                f"card_compute_dense_reward (CARD), whose published lines both take `self`")
        _recv = _symbol_table_receivers(_symtab, g)
        _unreachable = sorted(r for r in _recv if r not in _params) if _params else []
        if _unreachable:
            p.append(
                f"generate.postprocess.symbol_mapping={_symtab!r} rewrites the prompt's "
                f"symbols into index expressions on {', '.join(_unreachable)}, which "
                f"generate.output.signature={_sig_value!r} (`{_sig_line}`) does not "
                f"declare -- its parameters are {list(_params)}. Every rewritten program "
                f"would raise NameError on its first line, so the pair is refused here "
                f"rather than after a run of 100% invalid candidates")

    # -- train.reward_source: reference, the ORACLE arm ---------------------
    #
    # Two refusals, both of the half-configured-mechanism shape the block below
    # describes. The oracle arm trains on the environment's own reward, so an
    # adapter that has none would raise once per step inside the learner --
    # `has_reference_reward` is a per-TASK fact on a multi-task adapter, which is
    # why this reads it off the constructed env family rather than trusting the
    # env id's prefix. And an oracle "population" is one program: K identical
    # oracle cells would be K learner seeds wearing a search's clothes, and the
    # reflection prompt would be built from a reward the generator did not write.
    if g("train.reward_source") == "reference":
        _env_id = str(g("problem.env_id") or "")
        # The SAME instrument the native-signal rules use, not a second reader:
        # `kinds_for_env` resolves the task spec behind `problem.env_id` (+
        # `problem.task_id` where one env backs several tasks) and
        # `has_native_reward` is the one place the `reward.human.kind` values that
        # mean "there is a shipped reward" are enumerated. A per-task fact read
        # per task, and `None` where no spec is resolvable, which is silence
        # rather than a guess.
        from .native_signal import has_native_reward as _hnr2, kinds_for_env as _kfe2
        _, _rw2 = _kfe2(env_id=_env_id, task_id=str(g("problem.task_id") or "") or None)
        if _rw2 is not None and not _hnr2(_rw2):
            p.append(f"train.reward_source=reference trains on the environment's own "
                     f"shipped reward and problem.env_id={_env_id!r} reports none "
                     f"(reward.human.kind={_rw2!r} -- on a multi-task adapter that is a "
                     f"per-task fact, so the five gymnasium tasks whose specs disown the "
                     f"base environment's forward-velocity reward are refused where the "
                     f"other five are not). Use train.reward_source=candidate, or an env "
                     f"whose task claims its shipped reward")
        _k = n_cand_or(g, None)
        if isinstance(_k, int) and _k != 1:
            p.append(f"train.reward_source=reference with generate.n_candidates={_k}: an "
                     "oracle arm trains ONE program -- the environment's -- so K cells "
                     "would be K learner seeds of the same reward wearing a search's "
                     "clothes, and the reflection prompt would be built from a reward "
                     "the generator did not write. Set generate.n_candidates=1")

    # -- llm.generator.provider: fixed, the one-program control ---------------
    #
    # A knob, not a method: the generator returns one authored program verbatim
    # and stages 2-6 run untouched (bird/llm/fixed.py). Two refusals keep it a
    # control. The file is checked HERE, at load, with the same resolution the
    # provider uses (`paths.program_path`), so a typo'd path is a ConfigError
    # and not a run spent on nothing. `generate.n_candidates` is LEFT TO THE
    # CONFIG: K candidates are K identical
    # programs, the same reward under K learner seeds, a learner-variance control
    # whose run notes state K. And the judge cannot be a program: `fixed` on the
    # evaluator role is refused outright.
    if g("llm.generator.provider") == "fixed":
        raw = g("llm.generator.program")
        if not str(raw or "").strip():
            p.append("llm.generator.provider=fixed returns one authored program and needs "
                     "llm.generator.program to name it")
        else:
            from .paths import program_path as _program_path
            _pp = _program_path(raw)
            if not _pp.is_file():
                p.append(f"llm.generator.program={raw!r} resolves to {_pp}, which is not "
                         "a file (absolute, or relative to the checkout)")
    elif str(g("llm.generator.program") or "").strip():
        p.append("llm.generator.program is set but llm.generator.provider=%r never reads "
                 "it -- a pin nothing honours; set provider fixed or drop the path"
                 % (g("llm.generator.provider"),))
    if g("llm.evaluator.provider") == "fixed":
        p.append("llm.evaluator.provider=fixed: a judge is not a program; the fixed provider "
                 "is the GENERATOR-side control only (keep the evaluator mock or real)")

    # -- an env_spec that OFFERS identifiers needs something that binds them --
    #
    # `pythonic_class_abstraction` and `state_action_api_stub` render the state
    # as code: typed class attributes, `@dataclass` fields, `def helper(s): ...`
    # -- under headings that say "you may use" -- while the OUTPUT CONTRACT pins
    # a free function whose `state` is an ndarray and `training._reward_namespace`
    # binds `{np, numpy, math}` and nothing else. With no
    # `generate.postprocess.symbol_mapping` to rewrite them, every name the
    # prompt offers is a name the candidate cannot use.
    #
    # NOT hypothetical: `revolve` on `gym_humanoid_run` with this setup produced
    # 16 real gpt-4-turbo candidates that each wrote `state.x_velocity`, and 0
    # of 16 trained.
    #
    # WHAT THIS RULE DOES NOT CATCH, said plainly so nobody reads it as more
    # than it is. A non-empty mapping satisfies it, and a mapping can still miss
    # almost everything -- the four Text2Reward-lineage points carry a five-entry
    # table overlapping their eleven rendered names by ONE. Coverage is a
    # property of (adapter, member) and cannot be seen from the config alone
    # (`_check_coherence` has no env), so it is checked by
    # `tests/test_env_spec_rendered_resolves.py` instead. This is the cheap half:
    # it makes the no-mapping-at-all case impossible to launch.
    #
    # `t2r_class_abstraction` IS DELIBERATELY NOT IN THIS TUPLE, and the reason is
    # that it has a STRICTER rule of its own rather than none. The coverage check
    # above (`_env_spec_required_symbols` / `ENV_SPEC_REQUIRED_SYMBOLS`) refuses that
    # member unless the table actually COVERS the seven symbols it shows, and even
    # names the partial-rewrite case where a key that is a suffix of one produces a
    # plausible wrong program. Adding it here would be a weaker duplicate: presence
    # where a coverage rule already applies, reported twice for one config.
    #
    # The two members here are the ones whose symbols are NOT knowable at load.
    # `ENV_SPEC_REQUIRED_SYMBOLS` can hold a fixed list for `t2r_class_abstraction`
    # because that prompt is a published CONSTANT; what
    # `pythonic_class_abstraction` and `state_action_api_stub` render depends on the
    # ADAPTER, and `_check_coherence` has no env. So presence is the most this rule
    # can ask for them, and coverage is left to
    # `tests/test_env_spec_rendered_resolves.py`.
    #
    # `tests/test_env_spec_offers_need_a_binder.py` holds a second copy of this
    # tuple and asserts the two are identical.
    _OFFERS_IDENTIFIERS = ("pythonic_class_abstraction", "state_action_api_stub")
    # Named exemptions, not a blanket: three published points this rule would
    # otherwise refuse, pending repair. Enumerated so a FOURTH cannot join them
    # silently, and so the list shrinks visibly as each is repaired.
    #
    # KEYED ON THE CONFIG FILE, not on `name`. `name` is an ordinary overridable
    # key: the published GT point is `gt.yaml` loaded with `name: gt_reward_design`
    # (`tests/conftest.py::GT_PUBLISHED_OVERRIDES`), so an exemption keyed on
    # `name` would miss it and refuse a published point -- and, the other way, any
    # caller could set `name: gt` to walk through the rule. `cfg.source` is
    # the file the point came from and no override can move it.
    _UNGUARDED_LEGACY = ("gt", "limen", "limen_reward_only")
    # The file's stem, not its path: the exemption names a published point by
    # file, wherever under `configs/` it lives.
    _src = os.path.basename(str(getattr(cfg, "source", "") or ""))
    _origin = _src[:-len(".yaml")] if _src.endswith(".yaml") else _src
    if (g("generate.context.env_spec") in _OFFERS_IDENTIFIERS
            and not g("generate.postprocess.symbol_mapping")
            and _origin not in _UNGUARDED_LEGACY):
        p.append(
            "generate.context.env_spec=%r renders the state as code -- typed attributes "
            "and callable helpers the prompt says the model may use -- but "
            "generate.postprocess.symbol_mapping is empty, so nothing rewrites those "
            "names and the reward namespace ({np, numpy, math}) does not bind them. A "
            "candidate that uses what the prompt offers dies on AttributeError/NameError "
            "at execution_smoke or, where dynamic_checks is empty, inside training. "
            "Either set generate.postprocess.symbol_mapping, or choose an env_spec that "
            "presents each name as a LABEL beside its index rather than as an identifier "
            "(`natural_language_only` is the published instance -- it is what REvolve's "
            "own prompts/env_input does)."
            % (g("generate.context.env_spec"),))

    # -- fasttd3: num_updates > 1 with policy_frequency == 1 freezes the actor --
    #
    # `i % policy_frequency == 1` is UNSATISFIABLE FOR EVERY i WHEN
    # policy_frequency IS 1, so with num_updates > 1 the actor update never
    # runs: the critic trains, the run reports normally, and the policy that
    # ships is the initialisation. A silent null result, not a crash.
    #
    # THE SHAPE IS UPSTREAM'S AND THE REFUSAL IS OURS, deliberately. FastTD3's
    # own loop has the identical branch (`refs/code/FastTD3/fast_td3/train.py`
    # :669-674), and upstream cannot reach it by default either -- its
    # `hyperparams.py:63` declares `policy_frequency: int = 2` once and no task
    # preset overrides it, so only a CLI flag gets you there. "Faithful to
    # upstream" and "cannot silently freeze the actor" are different
    # requirements and this repo separates them: the port stays faithful, and
    # the unreachable-by-default combination becomes unreachable by override
    # too. Refusing at load changes no published behaviour: every config x
    # profile that resolves to fasttd3 uses (num_updates=2,
    # policy_frequency=2), and nothing in configs/ or scripts/ sets either key.
    if eff_backend == "fasttd3":
        _hp = g("train.hyperparameters") or {}
        from .components.fasttd3 import _FASTTD3_DEFAULTS as _FD
        # DEFAULTS FROM THE BACKEND, not from the resolved dict alone: an
        # absent key is not an absent value here -- the backend fills both of
        # these when `train.hyperparameters` omits them, so reading only what
        # the config states would let the pair through whenever one half is
        # inherited.
        try:
            _nu = int(_hp.get("num_updates", _FD["num_updates"]))
            _pf = int(_hp.get("policy_frequency", _FD["policy_frequency"]))
        except (TypeError, ValueError):
            _nu = _pf = None      # a non-integer is the schema's complaint, not this rule's
        if _nu is not None and _nu > 1 and _pf == 1:
            p.append(
                "train.hyperparameters.num_updates=%d with policy_frequency=1 "
                "never updates the actor: the inner loop's condition is "
                "`i %% policy_frequency == 1`, which no i satisfies when the "
                "frequency is 1, so only the critic trains and the run ships "
                "its initial policy while every counter reads normal. The "
                "branch is upstream's (FastTD3 train.py:669-674) and upstream "
                "reaches it no more than we do (its own default is 2); set "
                "policy_frequency>=2, or num_updates=1 if you meant one "
                "gradient step per iteration" % _nu)

    # -- problem.horizon truncates; it never extends -----------------------
    #
    # The runtime guard is `envs.base.apply_horizon`, which sees the ADAPTER and
    # is therefore the authority. This is the half that can be checked at LOAD,
    # and it is worth having separately for the reason every load-time rule in
    # this function is: on a spec-backed env the environment's horizon is
    # readable from `tasks/<id>/shared_spec.yaml` without constructing a MuJoCo
    # model, so a config that asks for a longer episode than the benchmark ships
    # fails in the second it takes to load rather than after a stage-[1] LLM
    # call and an adapter construction on a GPU.
    #
    # SCOPED TO SPEC-BACKED ENVS, and silent elsewhere rather than guessing: a
    # class-attribute horizon (toy, pendulum, the control family) is only
    # knowable by constructing the adapter, which `_check_coherence` must not
    # do. Those are caught by `apply_horizon` at construction, which is the
    # complete check; this one is the early one.
    _hz = g("problem.horizon")
    if _hz is not None:
        try:
            _hz = int(_hz)
        except (TypeError, ValueError):
            _hz = None                    # the schema's complaint, not this rule's
    if _hz is not None:
        if _hz < 1:
            p.append("problem.horizon=%d is not an episode length; leave it null for "
                     "the environment's own" % _hz)
        elif _chose_an_env(cfg):
            # ONLY WHEN THE CONFIG CHOSE AN ENVIRONMENT, and that condition is
            # the published configs' own idiom rather than a convenience. A paper config
            # here names no env: `configs/methods/rda.yaml`'s header says so in as many
            # words ("`problem.env_id` stays at its default ... A paper config
            # is a citation, not a launchable job"), so it resolves to
            # `_default.yaml`'s `toy_reacher`, whose spec horizon is 25 -- and
            # this rule would then refuse `rda_humanoidbench`'s own 500 against
            # an environment that file never meant, breaking
            # `bird.py --validate-all` for the configs it exists to validate.
            #
            # NOTHING IS LOST by deferring it there. A launch that DOES choose an
            # env (`-s problem.env_id=...`, a sweep job) resolves through
            # `config.load`, so a wrong horizon still fails before anything
            # runs, which is where this rule earns its keep. And a run on the
            # default env is caught at env construction by
            # `envs.base.apply_horizon`, which sees the adapter and is the
            # authority -- before stage [1] and before any LLM call.
            from . import tasks as _tasks
            _spec = None
            try:
                _tid = g("problem.task_id")
                _spec = (_tasks.load(str(_tid)) if _tid
                         else _tasks.by_env_id(str(g("problem.env_id"))))
            except Exception:  # noqa: BLE001 -- a missing/ambiguous spec is another
                _spec = None   # rule's complaint; this one simply cannot check
            _own = None
            if _spec is not None:
                try:
                    _own = int((_spec.env or {})["horizon"])
                except Exception:  # noqa: BLE001
                    _own = None
            if _own is not None and _hz > _own:
                p.append(
                    "problem.horizon=%d is longer than %s's own horizon (%d, "
                    "tasks/%s/shared_spec.yaml::env.horizon). The key TRUNCATES "
                    "rather than extends: shortening an episode is the trainer's "
                    "choice and lengthening one is a claim about the simulator that "
                    "the adapter owns. Drop the key, or lower it"
                    % (_hz, g("problem.env_id"), _own, _spec.id))

    # -- simba_v2: the four budget keys must be able to reach the paper's --
    #
    # `train.backend: simba_v2` relates `train.env_steps`,
    # `train.n_parallel_envs`, `action_repeat` and
    # `updates_per_interaction_step` by ONE identity, which is what lets
    # `configs/methods/rda_humanoidbench.yaml` claim RDA's 625K update steps
    # (`simba_v2_backend`'s docstring derives it):
    #
    #     interaction_steps = env_steps // (n_parallel_envs * action_repeat)
    #     updates           = interaction_steps * updates_per_interaction_step
    #
    # Two ways that silently produces something else, both refused here rather
    # than warned about in a log nobody reads:
    #
    #   * `env_steps < n_parallel_envs * action_repeat` floors the first line
    #     to zero and the backend's `max(1, ...)` then trains ONE interaction
    #     step -- a run that reports `train_steps_requested` and spends
    #     `num_envs * action_repeat`. On a profile that pins a small
    #     `train.env_steps` beside a 16-env method config that is reachable by
    #     accident, which is exactly how it would arrive.
    #   * a non-positive `action_repeat` or `updates_per_interaction_step`.
    #     `max(1, ...)` in the backend would silently promote either to 1, and
    #     `action_repeat: 0` reading as 1 is a different discount horizon under
    #     the config's own gamma.
    #
    # NOT refused, deliberately: a budget that does not divide evenly. The
    # floor is upstream's own (`num_interaction_steps` is a float there and
    # `range(int(...))` truncates it), the remainder is at most
    # `num_envs * action_repeat` steps, and the seed row records
    # `interaction_steps` as executed.
    if eff_backend == "simba_v2":
        _hp = g("train.hyperparameters") or {}
        from .components.simba_v2 import _SIMBAV2_DEFAULTS as _SD
        # DEFAULTS FROM THE BACKEND, not from the resolved dict alone: an
        # absent key is not an absent value here -- the backend fills these
        # when `train.hyperparameters` omits them.
        try:
            _ar = int(_hp.get("action_repeat", _SD["action_repeat"]))
            _upi = _hp.get("updates_per_interaction_step",
                           _SD["updates_per_interaction_step"])
            _upi = _ar if _upi is None else int(_upi)
            _envs = int(g("train.n_parallel_envs") or 1)
            _steps = int(g("train.env_steps") or 0)
        except (TypeError, ValueError):
            _ar = _upi = _envs = _steps = None   # the schema's complaint, not this rule's
        if _ar is not None:
            if _ar < 1:
                p.append(
                    "train.hyperparameters.action_repeat=%d is not a number of "
                    "simulator steps per decision; the backend would floor it to 1 "
                    "and run at a decision rate the config did not ask for, which "
                    "changes the effective discount horizon under the config's own "
                    "gamma. SimbaV2's HumanoidBench value is 2 "
                    "(refs/code/SimbaV2/configs/env/hb_locomotion.yaml)" % _ar)
            # ONLY IF THE CONFIG SET IT. `_upi` defaults to `action_repeat`, so a
            # config with `action_repeat: 0` and no update-rate key of its own
            # would otherwise get TWO complaints, the second naming a key it
            # never touched and quoting `None` as the offending value.
            if _hp.get("updates_per_interaction_step") is not None and _upi < 1:
                p.append(
                    "train.hyperparameters.updates_per_interaction_step=%r is not a "
                    "number of gradient steps per decision; leave it null to get "
                    "upstream's own `${action_repeat}` "
                    "(refs/code/SimbaV2/configs/online_rl.yaml:28)"
                    % _hp.get("updates_per_interaction_step"))
            if _ar >= 1 and _steps < _envs * _ar:
                p.append(
                    "train.backend=simba_v2 with train.env_steps=%d, "
                    "train.n_parallel_envs=%d and action_repeat=%d gives "
                    "%d // %d = 0 interaction steps, so the run would train ONE "
                    "step and report the requested budget. train.env_steps counts "
                    "SIMULATOR steps on this backend (upstream's own accounting: "
                    "interaction_step * action_repeat * num_train_envs), so it must "
                    "be at least n_parallel_envs * action_repeat = %d"
                    % (_steps, _envs, _ar, _steps, _envs * _ar, _envs * _ar))

    # -- the torch learners on a GPU suite must SAY cuda, and parallelise HB --
    #
    # Both torch backends default `train.hyperparameters.device` to `"cpu"`
    # (`_FASTTD3_DEFAULTS`, SimbaV2's own defaults). A cpu learner beside an
    # idle GPU fails nothing: the downgrade also silently disables AMP and
    # `torch.compile` (both derived from `device.type == "cuda"`), and the run
    # reports normally. `learner_device` on the seed row makes that visible
    # after the fact; this refuses it before the spend, which is the half a
    # report cannot do.
    #
    # `auto` IS REFUSED HERE, and only here. `auto` means "a GPU if there is
    # one", so on a machine without CUDA it degrades to cpu silently -- the
    # failure this guards, wearing a key that looks deliberate. Everywhere else
    # (`tests/test_learner_device_provenance.py::test_auto_keeps_its_fallback`)
    # `auto` keeps its fallback, which is the point of the value on a laptop or
    # in CI.
    #
    # Scoped through `envs.suites.env_suite`, the authoritative prefix table,
    # NOT a literal `h1*` test here: the HumanoidBench suite has several robot
    # prefixes (`h1_`, `h1hand_`, `h1strong_`, ...) and a second copy of that
    # list is how one gets missed.
    if eff_backend in ("fasttd3", "simba_v2"):
        from .envs.suites import env_suite
        _suite = env_suite(g("problem.env_id"))
        if _suite in _GPU_SUITES:
            device = str((g("train.hyperparameters") or {}).get("device", "") or "")
            if not device.startswith("cuda"):
                p.append(
                    "train.backend=%s on a %s-suite env needs an explicit "
                    "train.hyperparameters.device=cuda (got %r): both torch "
                    "backends default the key to `cpu`, and a cpu learner "
                    "beside an idle GPU reports normally. `auto` is refused "
                    "here too, because a silent degrade is the failure this "
                    "guards; it stays legitimate on every other suite"
                    % (eff_backend, _suite, device or None))
        # HumanoidBench unparallelised is ENV-BOUND: one simulator instance
        # feeding a GPU learner measured ~1.39 GPU-h per 1M steps, so a run
        # that forgets `n_parallel_envs` spends most of its GPU time waiting
        # on the simulator. Refused at load, where it is one line to fix. The
        # paper's HumanoidBench runs set 24 (fasttd3); RDA's SimbaV2 recipe
        # sets 16 (`configs/_profiles/humanoid_simba.yaml`).
        if _suite == "humanoid_bench" and int(g("train.n_parallel_envs") or 1) < 8:
            p.append(
                "train.backend=%s on a HumanoidBench env needs "
                "train.n_parallel_envs >= 8 (got %s): unparallelised "
                "HumanoidBench is env-bound (~1.39 GPU-h per 1M steps measured "
                "with one env), so the learner idles on the simulator. The "
                "paper's HumanoidBench runs used 24"
                % (eff_backend, g("train.n_parallel_envs")))

    # -- generate.reward_language must match the env family ----------------
    #
    # Two refusals, and they are not symmetric.
    #
    # A `jax_*` env is GPU-batched: `_JaxVecEnvView` calls the reward under
    # `jax.jit(vmap(...))` on device rows, so a numpy reward would have to be
    # pulled back to the host per row and would serialise the batch this
    # family exists to have. Refused rather than downgraded, because the
    # downgrade is invisible in every counter and shows only as a run that is
    # inexplicably slow.
    #
    # The reverse is refused because NOTHING WOULD RUN IT: a jax reward on a
    # numpy env reaches `compile_reward`'s numpy path, and a `jnp` body either
    # raises on an import the allowlist did not grant or silently computes on
    # host arrays under a name that says device. Neither is a run anyone meant
    # to launch.
    #
    # Scoped through `envs.suites.env_suite`, never an `id.startswith("jax_")`
    # here, for the reason the HumanoidBench rules below give: the suite table
    # is the one place the prefix lives, and a second copy of it is how a
    # family (e.g. `h1strong_`) gets missed.
    lang = g("generate.reward_language")
    # `torch` IS IN THIS TUPLE FOR THE `requires_cuda` REFUSAL as well as the
    # torch rule at the end of the block: the cuda check inside is a statement
    # about the ENV's capability and has nothing to do with the reward's
    # language, so no language may skip it.
    if lang in ("numpy", "jax", "torch"):
        from .envs.suites import env_suite
        is_jax_env = env_suite(g("problem.env_id")) == "jax"

        # THE JAX FAMILY IS CUDA-ONLY, AND THIS REFUSES AT CONFIG RESOLUTION
        # RATHER THAN AT THE FIRST STEP. A run that resolves clean and then
        # dies twenty minutes into a GPU allocation -- or, worse, trains on cpu
        # beside an idle GPU with every counter reading normal -- is far more
        # expensive than a refusal at load.
        #
        # BROADER THAN THE torch-learner RULE ABOVE, WHICH DOES NOT COVER
        # THIS. That one fires only on fasttd3/simba_v2, and `_GPU_SUITES`
        # does not list `jax`, so a jax env under `assistax_ppo` would slip
        # past it entirely.
        #
        # `auto` is refused with everything else that is not `cuda`: on this
        # family a cpu fallback is not a degrade, it is a different program --
        # `_JaxVecEnvView` hands the learner device tensors through DLPack and
        # there is no host path for it to fall back to.
        # GATED ON WHAT THE ADAPTER DECLARES, NOT ON THE SUITE NAME.
        # `env_suite` maps by PREFIX (`jax_`, envs/suites.py), so keying the
        # refusal on it would make a statement about the NAME do the work of
        # one about CAPABILITY -- the no-method-branching shape. It would
        # refuse `jax_toy`, a tester-tier offline env in `jnp` with no device
        # hand-off, and with it `configs/examples/jax_reward.yaml`; and
        # it would equally miss an MJX adapter not named `jax_*`, which
        # `upstream_assistax_*` is.
        #
        # A SEPARATE NAME, because `is_jax_env` is read again below by the
        # `generate.reward_language` rule, which must still see `jax_toy` as a
        # jax env. Reassigning it would give one variable two meanings.
        #
        # Read off the REGISTERED OBJECT, never by constructing: config
        # resolution runs in venvs with no `jax`, and `jax_toy` is offline by
        # design while an MJX constructor may want a device. NO DEFAULT --
        # `getattr(..., None)` and a separate catalogue test
        # (tests/test_jax_toy.py) requires every jax-suite id to declare it.
        # Defaulting False would let an MJX adapter that forgets it train on
        # cpu beside an idle GPU, every counter normal; defaulting True would
        # wrongly refuse the next CPU-capable jax env.
        needs_cuda = None
        if is_jax_env:
            try:
                from .registry import get as _reg_get
                needs_cuda = getattr(_reg_get("env", g("problem.env_id")),
                                     "requires_cuda", None)
            except Exception:  # noqa: BLE001 -- an unresolvable id is not this rule's business
                needs_cuda = None
        # `is not False`, NOT a truthy test, and the difference is the whole
        # point of the attribute. False is the ONLY value that admits cpu;
        # None means the adapter never declared, and an undeclared jax-suite
        # env is refused by NAME here rather than quietly admitted. A truthy
        # test would make None permissive, while `xla_env.py` reads the same
        # attribute and keeps the STRICT call on None (`if requires_cuda is
        # False: return False`) -- one reader would treat an undeclared
        # adapter as cpu-capable while the other treated it as needing
        # determinism. The two must agree, and they must agree on strict: the
        # permissive side of a disagreement is the quieter failure, and config
        # load is the earlier of these two.
        # `is_jax_env and` is load-bearing: `needs_cuda` is initialised to
        # None and STAYS None for every non-jax env, so without it the refusal
        # below would fire for `toy_reacher`, `pendulum` and everything else.
        # `--validate-all` exercises that case; unit tests over jax-suite
        # stand-ins alone would not.
        if is_jax_env and needs_cuda is None:
            p.append(
                "problem.env_id=%s is a jax-suite env whose registered object "
                "declares no `requires_cuda`. That attribute is the thing this "
                "rule reads, so an adapter that omits it cannot be checked at "
                "all -- and the failure it guards is silent: an MJX adapter "
                "training on cpu beside an idle GPU with every counter "
                "reading normal. Declare it on the object the registry returns "
                "(`requires_cuda = True` for an MJX adapter that hands the "
                "learner device tensors, `False` for one that computes in jnp "
                "with no device hand-off, as jax_toy does). False is the only "
                "value that admits a cpu device; leaving it undeclared is not. "
                "`tests/test_jax_toy.py` holds the same requirement as a "
                "catalogue check, and `bird/xla_env.py` keeps the strict "
                "determinism call on the same undeclared case."
                % (g("problem.env_id"),))
        elif needs_cuda:
            jdevice = str((g("train.hyperparameters") or {}).get("device", "") or "")
            if not jdevice.startswith("cuda"):
                p.append(
                    "problem.env_id=%s declares `requires_cuda`, so it needs "
                    "train.hyperparameters.device=cuda (got %r). That adapter "
                    "steps on the GPU and hands the learner device tensors "
                    "through DLPack; there is no cpu path to fall back to, so "
                    "this is a refusal and not a downgrade. `auto` is refused "
                    "too -- a silent degrade is the failure this guards. The "
                    "refusal is keyed on the DECLARATION and not on the env "
                    "id's suite: a jax-suite env that computes in jnp without "
                    "a device hand-off (jax_toy) declares False and is not "
                    "refused, and an MJX adapter not named `jax_*` would still "
                    "be caught."
                    % (g("problem.env_id"), jdevice or None))
        # -- NO fork-parallel refusal here, deliberately -------------------
        #
        # `train.candidate_parallelism: parallel` on an env declaring
        # `batched` cannot FORK a worker once the parent has taken jax: a
        # forked fasttd3 child dies with `CUDA error: initialization error`,
        # and on CPU a fork after the parent has substantially exercised jax
        # can deadlock (children in `futex_do_wait`) -- a condition, not a
        # law, since a fresh parent under the mock learner forks and
        # completes.
        #
        # EACH FORK SITE HANDLES IT instead:
        #   - the CANDIDATE fork (`training.py`, `parallelism_parallel`) is a
        #     `posix_spawn` of a fresh interpreter whenever the live env is a
        #     `BatchedEnvAdapter`, so there is no inherited mutex and no
        #     inherited CUDA context to wait on;
        #   - the SEED fork (`training._seed_fork_workers`) refuses to fork
        #     on a batched env and runs seeds sequentially, which is
        #     bit-identical by contract;
        #   - the numpy slot worker (`fasttd3.py`'s `_VecEnvView`) is never
        #     reached on this tier, per `_JaxVecEnvView`: "no slot workers
        #     and no pipes".
        #
        # A load-time refusal of `parallel` on this tier would make the
        # fork path impossible to exercise -- the config would not load -- so
        # the guard lives at each fork, where it can be measured, rather than
        # at config load, where it could only be believed.

        if is_jax_env and lang != "jax":
            p.append(
                "problem.env_id=%s is a jax/MJX env and needs "
                "generate.reward_language=jax (got %r): the family steps a "
                "device batch and calls the reward under jax.jit(vmap(...)), "
                "so a numpy reward would be evaluated row by row on the host "
                "-- the batching this family exists for, silently undone"
                % (g("problem.env_id"), lang))
        if lang == "jax" and not is_jax_env:
            p.append(
                "generate.reward_language=jax on a non-jax env "
                "(problem.env_id=%s): nothing would run it -- the numpy "
                "backends compile the candidate with the numpy namespace, so "
                "a jnp body either fails to import or computes on host arrays "
                "under a name that says device" % g("problem.env_id"))

        # -- generate.reward_language: torch -------------------------------
        #
        # REFUSED: `torch` on a backend that cannot execute a torch reward.
        # `mock`, `tabular` and `none` compile the candidate through the numpy
        # path and call it per transition, so a body indexing `state[:, 3]`
        # either raises on the first call or -- worse -- reads the wrong
        # element of a 1-D row and computes a plausible wrong number. That is a
        # run nobody meant to launch, which is the same sentence the jax rule
        # above uses for the same reason.
        if lang == "torch" and eff_backend not in ("fasttd3", "sb3"):
            p.append(
                "generate.reward_language=torch with train.backend=%s: that "
                "backend compiles the candidate through the numpy path and "
                "calls it one transition at a time, so a batched torch body "
                "indexing `state[:, i]` either raises on the first call or "
                "reads the wrong element of a 1-D row and returns a plausible "
                "wrong number. Use train.backend: fasttd3 or sb3, or leave "
                "generate.reward_language at numpy for a surrogate tier."
                % eff_backend)

    # -- final_retrain.max_env_steps: an int, null, or the one sentinel --------
    cap = g("final_retrain.max_env_steps")
    if isinstance(cap, str) and cap != "same_as_train":
        p.append("final_retrain.max_env_steps: %r is not a step count; the only string "
                 "allowed is 'same_as_train' (cap the retrain at the resolved "
                 "train.env_steps)" % (cap,))
    if isinstance(cap, bool) or (isinstance(cap, int) and cap <= 0):
        p.append("final_retrain.max_env_steps: %r must be a positive step count, null, "
                 "or 'same_as_train'" % (cap,))

    # -- max_over_training_epochs scores the NATIVE series ------------------
    #
    # This reducer replaces the SERIES, not just the reduction: it maxes the
    # per-update training series, whose values are the native channel
    # (`bird/native_signal.py`) rather than whatever `evaluate.fitness.source`
    # names. Pinning it beside a non-native source therefore produces a report
    # whose LABEL and NUMBER disagree -- `fitness_source: ground_truth_metric`
    # against a figure that is the max of a reference-reward return on one task
    # and a success rate on the next. Nothing downstream can detect that, so it
    # is refused at load rather than warned about.
    if g("evaluate.fitness.checkpoint_aggregation") == "max_over_training_epochs":
        src = g("evaluate.fitness.source")
        if src != "native":
            # Exactly `native`, not `native_success` / `native_reward`: the series'
            # channel is whatever `native_signal.channel_for_env` resolves for the
            # task (the shipped success where the task has one, else the shipped
            # reward's return) -- the same resolution `native` makes -- so an
            # explicit `native_reward` on a task that ships a success would label
            # a success series `native_reward`, and nothing downstream could tell.
            p.append(
                "evaluate.fitness.checkpoint_aggregation: max_over_training_epochs "
                "scores the per-update TRAINING series, whose values are the task's "
                "native channel as `native` resolves it -- so evaluate.fitness.source "
                "must be `native` exactly, not %r. Otherwise the report labels the "
                "number with a source that did not produce it. Use "
                "max_over_checkpoints to score the held-out checkpoint curve under "
                "%r instead." % (src, src))

    # -- verify.on_failure: teach_next_iteration settles the slot -----------
    #
    # The policy returns a settled (dropped, taught) candidate at the FIRST
    # failure, so `run_validity` never reaches `_regenerate` under it and
    # `verify.max_repair_attempts` cannot be honoured. LIMEN's release retries a
    # crash once; written as `teach_next_iteration` + `1`, that retry would
    # never run (the message names the form that does). A declared repair that
    # cannot execute is a fabricated pin -- a key the algorithm cannot honour --
    # so refuse it.
    if g("verify.on_failure") == "teach_next_iteration" and \
            int(g("verify.max_repair_attempts") or 0) != 0:
        p.append("verify.on_failure=teach_next_iteration settles a failed slot at the "
                 "first failure, so verify.max_repair_attempts=%s can never run; set "
                 "it to 0, or express repair-then-teach as "
                 "verify.on_failure=resample_with_trace + "
                 "verify.on_exhaustion=teach_next_iteration"
                 % g("verify.max_repair_attempts"))

    # -- LaRes (train.interaction / reward_scaling / elite_constraint) --------
    #
    # Seven rules, and all seven exist because the alternative is a declared
    # mechanism that does not execute -- a fabricated pin, i.e. a key the
    # algorithm cannot honour.
    if g("train.interaction_allocator") != "uniform" and \
            g("train.interaction") != "shared_population":
        p.append("train.interaction_allocator=%s is read only by "
                 "train.interaction=shared_population; under `independent` nothing "
                 "allocates and the key would be inert"
                 % g("train.interaction_allocator"))

    # `thompson_success` counts the ENVIRONMENT's success flag once per slice
    # (LaRes_from_scratch.py:827), so it is a ground-truth channel INSIDE the
    # inner loop -- earlier and cheaper than the one at the selection boundary.
    # A config whose fitness deliberately never touches the task metric would
    # be leaking it here, invisibly, which is precisely what the GT-free line
    # exists to avoid.
    #
    # KEYED ON THE DECLARATION, `problem.fitness_access`, and not only on
    # `evaluate.fitness.source`. The source says how §4 computes a number; the
    # access key is what declares whether the run may read ground truth at all,
    # and it is the key the supervised/unsupervised column is read off. A rule
    # testing the source alone would let `rda` (access none, source vlm_score)
    # and `gt` (access none, source preference_bt) validate with
    # `thompson_success` and steer stage 3 by the env's success flag five
    # times per candidate per iteration while the resolved config still said
    # `fitness_access: none`. The sibling `train.pruning_metric`
    # rule below already keys on the access declaration.
    if g("train.interaction_allocator") == "thompson_success":
        access = g("problem.fitness_access")
        source = g("evaluate.fitness.source")
        if access not in ("ground_truth_metric", "success_indicator") or source == "none":
            p.append("train.interaction_allocator=thompson_success reads the environment's "
                     "success flag every slice, so it is ground-truth access inside the "
                     f"inner loop; it cannot be combined with problem.fitness_access="
                     f"{access!r} / evaluate.fitness.source={source!r} -- the GT-free "
                     "declaration would be false while the run read the truth (use "
                     "`uniform`, which is the paper's own Fig. 6a control)")

    # NOTE there is deliberately no rule against `candidate_parallelism:
    # parallel` here. The WAVES are sequential -- the allocator's next draw
    # needs this wave's outcomes -- but the arms WITHIN a wave are independent
    # and go through the schedule unchanged, so `parallel` executes and means
    # what it says. It also could not be refused even if we wanted to: the
    # `full` profile pins `parallel` and a profile wins over a method config
    # by design (see PROFILE_KEY_PREFIXES above), so the rule would make
    # every `shared_population` config unloadable under that profile.
    if g("train.interaction") == "shared_population":
        if g("train.hyperparameter_search") != "none":
            p.append("train.interaction=shared_population drives the backend directly and "
                     "does not wrap it in train.hyperparameter_search=%s"
                     % g("train.hyperparameter_search"))
        # A slice is ONE agent resuming ONE policy (`resume_ref` carries a
        # single policy ref from slice to slice), so the driver launches every
        # slice with `n_seeds=1` and never consults the `select.allocation`
        # plan -- both keys would resolve, hash and print while one seed
        # trained under `uniform`. LaRes runs one SAC agent per reward
        # (§5.1); a seeds ablation under this interaction mode is not
        # expressible until the hand-off carries a policy per seed, and saying
        # so at load beats a resolved config that claims three replicates it
        # never ran.
        if int(g("train.seeds_per_candidate") or 1) != 1:
            p.append("train.interaction=shared_population trains ONE agent per member, "
                     "resuming one policy per slice, so train.seeds_per_candidate=%s is "
                     "not honoured (every slice runs one seed); set it to 1, or use "
                     "train.interaction=independent for a seeds ablation"
                     % g("train.seeds_per_candidate"))
        if g("select.allocation") != "uniform":
            p.append("train.interaction=shared_population divides the pooled budget with "
                     "train.interaction_allocator and never reads the select.allocation "
                     "plan, so select.allocation=%s would be inert; leave it uniform"
                     % g("select.allocation"))
        # `interaction_cfg.shared_buffer: true` pools every arm's raw
        # transitions and pre-fills each arm's REPLAY BUFFER with them,
        # relabelled (LaRes §4.3; `training._sb3_prefill_replay`). An on-policy
        # learner has no buffer to fill and the planner has no learner, so the
        # key would read `true` while nothing was shared -- refused here rather
        # than left to a comment. The surrogates (mock, tabular) are NOT
        # refused: they are the tester tier, and they say in the seed row and
        # the journal that the buffer did not execute there.
        #
        # `fasttd3` is NOT refused: it pre-fills from the round pool and
        # exports what it added through the same `_ROUND_REPLAY` key sb3 uses
        # (`fasttd3._fasttd3_prefill_replay`). The general rule: a key that
        # reads `true` while nothing executes is refused at load, and when the
        # capability arrives the refusal goes rather than the key being
        # quietly reinterpreted.
        if g("train.interaction_cfg.shared_buffer"):
            alg = g("train.algorithm")
            if alg in ("ppo", "none"):
                p.append("train.interaction_cfg.shared_buffer=true pools raw transitions "
                         "for an off-policy learner to relabel and resume from, and "
                         "train.algorithm=%s keeps no replay buffer, so nothing would be "
                         "shared; set it false or use an off-policy train.algorithm" % alg)
            # `simba_v2` IS REFUSED, by the rule stated above: that backend
            # neither pre-fills its buffer from the round pool nor exports
            # what it added (stated as a deviation in its module docstring,
            # and its seed row carries `secondary_transitions: 0` /
            # `secondary_ratio: 0.0` to say so). So the key would read `true`
            # while nothing was shared. The refusal goes when the capability
            # lands, and not before.
            if eff_backend == "simba_v2":
                p.append("train.interaction_cfg.shared_buffer=true is not wired on "
                         "train.backend=simba_v2 (the backend neither pre-fills its "
                         "replay buffer from the round pool nor exports what it "
                         "added, unlike sb3 and fasttd3); set "
                         "it false, or use train.backend=sb3 or fasttd3")

    if g("train.reward_scaling") == "elite_moments" and \
            "replay_buffer" not in (g("loop.carry") or []):
        p.append("train.reward_scaling=elite_moments estimates both rewards' moments over "
                 "the shared replay buffer (LaRes Eq. 3), so it needs replay_buffer in "
                 "loop.carry or it can never apply")

    if g("train.elite_constraint.kind") == "l2_params":
        if g("train.init") not in ("warm_start_from_best", "bc_prior_then_warm_start"):
            p.append("train.elite_constraint.kind=l2_params constrains toward the "
                     "parameters train.init loaded, so it requires train.init="
                     "warm_start_from_best (LaRes initialises every non-elite from the "
                     "elite before constraining it to the elite)")
        if "policy_checkpoint" not in (g("loop.carry") or []):
            p.append("train.elite_constraint.kind=l2_params needs policy_checkpoint in "
                     "loop.carry, or no elite parameters survive the iteration boundary")

    # The four archive-backed topologies keep their population in
    # `state.archive`, the one carried population slot (`bird/components/
    # update.py`, module docstring), and `RunState.apply_carry` empties every
    # slot `loop.carry` does not name. So `elitist_population` without
    # `archive` in the carry is hill climbing wearing a population's name:
    # `select.n_survivors` sizes a set that is gone before anything reads it.
    # configs/methods/lares.yaml's App. B elite size of 3 depends on this: with the
    # carry omitting `archive` it would be a pin on nothing.
    if g("update.topology") in ("elitist_population", "island", "island_lineage",
                                "archive_map_elites") and \
            "archive" not in (g("loop.carry") or []):
        p.append("update.topology=%s keeps its population in state.archive, which "
                 "loop.carry empties at every iteration boundary unless it names "
                 "`archive`; add archive to loop.carry or the population (and "
                 "select.n_survivors) is dropped before the next round can read it"
                 % g("update.topology"))

    if g("evaluate.feedback.granularity") == "per_subtask" and not g("generate.decomposition.enabled"):
        p.append("evaluate.feedback.granularity=per_subtask requires "
                 "generate.decomposition.enabled=true (nothing defines the subtasks)")

    if g("evaluate.feedback.granularity") == "per_component" and \
            g("generate.output.format") == "scalar_only":
        p.append("evaluate.feedback.granularity=per_component requires a component-returning "
                 "generate.output.format (component visibility is a precondition, §1)")

    if g("verify.alignment_filter.enabled") and not g("evaluate.preferences.store_dataset"):
        p.append("verify.alignment_filter.enabled=true needs "
                 "evaluate.preferences.store_dataset=true or D_pref never fills (GT cold-start)")

    keep = g("verify.alignment_filter.keep_top_n") or 0
    sched_vals = list(g("generate.candidate_schedule_values") or [])
    # Under a schedule the filter must fit the NARROWEST round: `keep_top_n` at
    # or above the pool size makes `_rank_keep` pass every candidate and reject
    # none, with no journal event distinguishing "filtered" from "did nothing",
    # so GT's contribution would stop executing mid-run invisibly.
    min_k = min(sched_vals) if sched_vals else (g("generate.n_candidates") or 0)
    if g("verify.alignment_filter.enabled") and keep > min_k:
        p.append(f"verify.alignment_filter.keep_top_n={keep} exceeds "
                 f"the smallest candidate count per iteration ({min_k})")

    # `system_prompt` replaces THE system message; `personas` samples one PER
    # candidate. Both at once would make one of them silently inert (whichever
    # the assembler happened to prefer), so the pair is refused instead.
    if g("generate.context.system_prompt") and g("generate.sampling_mode") == "personas":
        p.append("generate.context.system_prompt cannot be combined with "
                 "generate.sampling_mode=personas (personas ARE per-candidate "
                 "system prompts; one of the two would be silently ignored)")

    # -- the per-offspring genetic operator (REvolve) ----------------------
    # A probability, so it is range-checked here rather than pinned to an enum: it
    # is a menu of nothing, and REvolve's own genetic-operator ablation (Fig. 4, p.9)
    # already walks three of its values. The two >0 rules are the fabricated-pin check --
    # `update.operator` picks ONE operator for a whole round and every existing
    # sampler draws its parents once before its loop, so outside
    # `evolutionary_operators` there is nothing that could read a per-offspring
    # rate, and a config stating one would be claiming a mechanism it does not run.
    _xrate = g("generate.crossover_rate")
    # isinstance BEFORE float(): a bare float("abc") raises ValueError and
    # float([1]) raises TypeError, and either escapes validate() as a raw
    # traceback instead of joining the collected problems -- losing the config
    # path and the report-every-problem-at-once contract. The schema's own type
    # check catches most junk first; this is the belt for anything past it.
    if _xrate is not None and not isinstance(_xrate, (int, float)):
        p.append(f"generate.crossover_rate={_xrate!r} must be a number in [0, 1]")
        _xrate = None
    if _xrate is not None and not (0.0 <= float(_xrate) <= 1.0):
        p.append(f"generate.crossover_rate={_xrate!r} is a probability (P(crossover) "
                 "per offspring) and must be in [0, 1]")
    if (_xrate or 0) > 0:
        if g("generate.sampling_mode") != "evolutionary_operators":
            p.append("generate.crossover_rate>0 but generate.sampling_mode is "
                     f"{g('generate.sampling_mode')!r}; evolutionary_operators is the "
                     "only sampler that reads the rate, so it would be a declared knob "
                     "nothing executes")
        if (g("generate.n_parents") or 1) < 2:
            p.append("generate.crossover_rate>0 needs generate.n_parents>=2; the two "
                     "keys otherwise state opposite things -- one asks for a crossover, "
                     "the other says only one parent reaches the prompt")

    # -- loop.waves: passes per iteration ----------------------------------
    # A wave list that does not add up to `generate.n_candidates` is the
    # quietest possible way to change a budget: every sizing path (worker
    # resolution, a resource request) reads n_candidates, so a mismatch trains
    # a different number of policies than the config states and nothing
    # reports it.
    _waves = g("loop.waves") or []
    if _waves:
        _totals = {}
        for i, w in enumerate(_waves):
            if not isinstance(w, dict):
                p.append(f"loop.waves[{i}]={w!r} must be a mapping with llm/crossover/when")
                continue
            unknown = set(w) - {"llm", "crossover", "when"}
            if unknown:
                p.append(f"loop.waves[{i}] has unknown field(s) {sorted(unknown)}; "
                         "allowed: llm, crossover, when")
            when = str(w.get("when", "always"))
            if when not in ("always", "no_archive"):
                p.append(f"loop.waves[{i}].when={when!r} must be 'always' or 'no_archive'")
            n_llm = w.get("llm", 0) or 0
            n_x = w.get("crossover", 0) or 0
            for label, val in (("llm", n_llm), ("crossover", n_x)):
                if not isinstance(val, int) or val < 0:
                    p.append(f"loop.waves[{i}].{label}={val!r} must be an int >= 0")
            if isinstance(n_llm, int) and isinstance(n_x, int) and n_llm + n_x <= 0:
                p.append(f"loop.waves[{i}] trains nothing (llm=0, crossover=0); a pass "
                         "that generates no candidate is a stage sequence with no effect")
            if isinstance(n_x, int) and n_x > 0 and g("generate.crossover.operator") == "none":
                p.append(f"loop.waves[{i}].crossover={n_x} but "
                         "generate.crossover.operator=none; the pass reserves candidates "
                         "nothing can build")
            if isinstance(n_llm, int) and isinstance(n_x, int):
                _totals.setdefault(when, 0)
                _totals[when] += n_llm + n_x
        # Both realisable iterations must train exactly n_candidates: the
        # steady state (`always` waves only) and a bootstrap iteration
        # (`always` + `no_archive`). R*: 12+4 always, plus a 0+4 no_archive
        # pass (6+2 / 0+2 under the paper's budget-matched n_candidates=8)
        # whose iteration suppresses wave 1's own crossover.
        want = n_cand_or(g, None)
        steady = _totals.get("always", 0)
        boot = steady + _totals.get("no_archive", 0)
        if isinstance(want, int):
            if steady != want:
                p.append(f"loop.waves `always` passes train {steady} candidate(s) but "
                         f"generate.n_candidates={want}; every sizing path reads "
                         "n_candidates, so the two must agree"
                         + _rescaled_waves_hint(_waves, steady, want))
            _boot_x = int(g("generate.crossover.n") or 0) if _totals.get("no_archive") else 0
            if boot - _boot_x != want:
                p.append(f"loop.waves on a bootstrap iteration train {boot} candidate(s) "
                         f"against generate.n_candidates={want} (the `always` passes' "
                         f"crossover share, {_boot_x}, cannot run with an empty archive); "
                         "an iteration that trains a different number than the others "
                         "silently breaks a budget-matched comparison")

    # -- R*: the crossover share, and the alignment it feeds ---------------
    # Every rule below refuses a HALF-CONFIGURED mechanism. The failure mode
    # they exist to stop is a recurring one: a key that declares a
    # capability the run then does not execute, reported nowhere, so the run
    # reads as the method and is not.
    _xn = int(g("generate.crossover.n") or 0)
    _xop = g("generate.crossover.operator")
    _xsel = g("generate.crossover.parent_selection")
    if _xn < 0:
        p.append(f"generate.crossover.n={_xn} must be >= 0")
    # 0 is legal and greedy (`parents_softmax_fitness` clamps it to 1e-9 and the
    # top-fitness parent wins every draw); below 0 the softmax ranks the pool
    # upside down, which no paper describes and no one asks for on purpose.
    # `isinstance` so a non-number is left to the schema's own type report.
    _xtemp = g("generate.crossover.temperature")
    if isinstance(_xtemp, (int, float)) and not isinstance(_xtemp, bool) and _xtemp < 0:
        p.append(f"generate.crossover.temperature={_xtemp!r} must be >= 0: a negative "
                 "softmax temperature inverts the fitness ranking; 0 is already greedy")
    if _xn > 0:
        if _xop == "none":
            p.append("generate.crossover.n>0 with generate.crossover.operator=none: the "
                     "share is reserved and nothing fills it, so the sampler is asked "
                     "for fewer candidates and the pool comes up short every iteration")
        if _xsel == "none":
            p.append("generate.crossover.n>0 needs generate.crossover.parent_selection; "
                     "`none` returns no parents, so no offspring is ever built")
        if _xn >= int(n_cand_or(g, 1)):
            p.append(f"generate.crossover.n={_xn} must be < generate.n_candidates="
                     f"{g('generate.n_candidates')!r}; at or above it the LLM is asked "
                     "for nothing and the population is recombination only")
    if _xop != "none" and _xn <= 0:
        p.append(f"generate.crossover.operator={_xop!r} with generate.crossover.n=0 is a "
                 "declared operator that never runs; set the share or the operator to none")
    if _xop != "none" and g("generate.output.format") != "component_dict_return":
        p.append(f"generate.crossover.operator={_xop!r} splits a reward on its RETURNED "
                 "COMPONENT DICT (R* App. A, p.12), so it needs "
                 "generate.output.format=component_dict_return; "
                 f"{g('generate.output.format')!r} emits no modules to recombine")

    _amethod = g("generate.alignment.method")
    _alab = g("generate.alignment.labeller")
    _acrit = int(g("generate.alignment.critics") or 0)
    _carry = list(g("loop.carry") or [])
    if _amethod != "none":
        if _alab == "none":
            p.append(f"generate.alignment.method={_amethod!r} needs a labeller; with "
                     "generate.alignment.labeller=none no segment is ever labelled and "
                     "the fit runs on an empty dataset")
        if g("generate.alignment.tunable") == "none":
            p.append(f"generate.alignment.method={_amethod!r} needs "
                     "generate.alignment.tunable to name which numbers are optimised; "
                     "`none` declares an optimiser over an empty parameter vector")
        if int(g("generate.alignment.iterations") or 0) <= 0:
            p.append(f"generate.alignment.method={_amethod!r} needs "
                     "generate.alignment.iterations>0 (R* 1000, App. A p.12)")
        if g("update.memory.trajectory_store") == "none":
            p.append(f"generate.alignment.method={_amethod!r} labels rollouts from the "
                     "trajectory store, but update.memory.trajectory_store=none never "
                     "puts one there; carrying an empty store labels nothing and "
                     "reports no error")
        if "trajectory_store" not in _carry:
            p.append(f"generate.alignment.method={_amethod!r} labels the PREVIOUS "
                     "iteration's rollouts, so loop.carry must include trajectory_store; "
                     "without it the store is emptied at every iteration boundary and no "
                     "segment can ever be labelled")
        tf = g("generate.alignment.train_fraction")
        if not isinstance(tf, (int, float)) or not (0.0 < float(tf) < 1.0):
            p.append(f"generate.alignment.train_fraction={tf!r} must be in (0, 1); the "
                     "validation split is what selects the answer (R* App. A, p.12)")
    if _alab != "none" and _amethod == "none":
        p.append(f"generate.alignment.labeller={_alab!r} with "
                 "generate.alignment.method=none labels segments nothing reads")
    if _alab == "critic_population_vote":
        if _acrit <= 0:
            p.append("generate.alignment.labeller=critic_population_vote needs "
                     "generate.alignment.critics>0 (R* c=5, App. A p.12)")
        if "critic_population" not in _carry:
            p.append("generate.alignment.labeller=critic_population_vote needs "
                     "critic_population in loop.carry; the critics are authored once "
                     "(Alg. 1 line 3) and would otherwise be re-authored -- and re-paid "
                     "for -- every iteration")
        _ladder = list(g("generate.alignment.vote_ladder") or [])
        if not _ladder:
            p.append("generate.alignment.labeller=critic_population_vote needs a "
                     "non-empty generate.alignment.vote_ladder (R* [5, 4, 3], App. A p.12)")
        for v in _ladder:
            if not isinstance(v, int) or v <= 0:
                p.append(f"generate.alignment.vote_ladder entry {v!r} must be an int > 0")
            elif _acrit and v > _acrit:
                p.append(f"generate.alignment.vote_ladder asks for {v} agreeing critics "
                         f"but generate.alignment.critics={_acrit}; that rung can never "
                         "fire, so the ladder silently starts one rung lower")
    if _acrit > 0 and _alab != "critic_population_vote":
        p.append(f"generate.alignment.critics={_acrit} but "
                 f"generate.alignment.labeller={_alab!r}: the critics would be authored "
                 "and PAID FOR and never read")

    # A candidate count of 0 empties the pool, which `run_iteration` reports as
    # "every candidate failed" and routes to `loop.on_total_failure` -- a silent
    # run-abort wearing a failure message. Refuse it at load instead.
    n_cand = g("generate.n_candidates")
    if isinstance(n_cand, int) and n_cand < 1:
        p.append(f"generate.n_candidates={n_cand} must be >= 1")

    if g("generate.candidate_schedule") == "explicit":
        n_iter = g("loop.n_iterations") or 0
        if not sched_vals:
            p.append("generate.candidate_schedule=explicit needs a non-empty "
                     "generate.candidate_schedule_values")
        else:
            if any((not isinstance(v, int)) or v < 1 for v in sched_vals):
                p.append(f"generate.candidate_schedule_values={sched_vals} must be "
                         "all ints >= 1")
            if len(sched_vals) < n_iter:
                p.append(f"generate.candidate_schedule_values has {len(sched_vals)} "
                         f"entries for loop.n_iterations={n_iter}; the tail would "
                         "silently repeat the last value")
            # `n_candidates` is what every sizing path provisions against
            # (`resolve_workers`, a resource request). A schedule
            # peaking above it splits the widest round into sequential waves
            # while the resolved config still reads `parallel`.
            if sched_vals and isinstance(n_cand, int) and max(sched_vals) > n_cand:
                p.append(f"generate.candidate_schedule_values peaks at "
                         f"{max(sched_vals)} but generate.n_candidates={n_cand}; "
                         "n_candidates must be the maximum, it is what sizing reads")
    elif sched_vals:
        p.append("generate.candidate_schedule_values is set but "
                 f"generate.candidate_schedule={g('generate.candidate_schedule')!r} "
                 "ignores it")

    # Inner-loop early stopping: the RULE (`train.pruning`) x the METRIC
    # (`train.pruning_metric`). `task_metric` is env.task_metric -- ground truth
    # -- so a method that declares no fitness access may not stop on it (a
    # `median_stop` override on RDA would otherwise stop on ground truth).
    # `pruning_cfg.{patience,min_delta,min_checkpoints}` are
    # read by `plateau` alone (the `.ceiling*` / `.rung_fraction` / `.eta` leaves
    # by the demonstration ceiling and the pool rule, checked below), so setting
    # them under another rule is the same paradox as `candidate_schedule_values`
    # under `constant`: a declared knob nothing executes.
    rule = g("train.pruning") or "none"
    if rule != "none" and g("train.pruning_metric") == "task_metric" \
            and g("problem.fitness_access") == "none":
        p.append("train.pruning_metric=task_metric stops training on the env's own "
                 "ground-truth signal but problem.fitness_access=none; use own_reward, "
                 "or declare the access")
    # Native-signal rule: `pruning_metric: task_metric` prunes on the env's
    # NATIVE signal (success_rate / gt_return), never the BIRD task_metric. On a
    # task that ships neither, there is nothing native to prune on -- refused
    # here rather than raised mid-training (the five gym reward.human.kind: none
    # tasks). Keyed per-task on `_spec`, like the fitness.source rule.
    if rule != "none" and g("train.pruning_metric") == "task_metric":
        # `_spec` is not loaded until later in this function, so resolve the
        # channel lazily via native_signal (which loads the spec itself).
        from .native_signal import (kinds_for_env as _kfe, resolve_channel as _rc,
                                     REFUSE as _REF)
        _ds, _rw = _kfe(env_id=g("problem.env_id"), task_id=g("problem.task_id"))
        if _rc(_ds, _rw) == _REF:
            p.append("train.pruning_metric=task_metric prunes on the env's native signal "
                     f"but the task for problem.env_id={g('problem.env_id')!r} ships "
                     f"neither a native success (discrete_success.kind={_ds!r}) nor a "
                     f"reference reward (reward.human.kind={_rw!r}); use own_reward")
    pc_patience = g("train.pruning_cfg.patience")
    pc_min_ck = g("train.pruning_cfg.min_checkpoints")
    pc_delta = g("train.pruning_cfg.min_delta")
    if rule == "plateau":
        if not isinstance(pc_patience, int) or pc_patience < 1:
            p.append(f"train.pruning_cfg.patience={pc_patience!r} must be an int >= 1")
        if not isinstance(pc_min_ck, int) or (isinstance(pc_patience, int)
                                              and pc_min_ck <= pc_patience):
            p.append(f"train.pruning_cfg.min_checkpoints={pc_min_ck!r} must be an int "
                     f"> patience ({pc_patience!r}); otherwise the rule can fire on "
                     "its first comparable window")
        if not isinstance(pc_delta, (int, float)) or pc_delta < 0:
            p.append(f"train.pruning_cfg.min_delta={pc_delta!r} must be >= 0")

    # The demonstration ceiling and the pool rule (not published).
    # Both roll out a SOLUTION POLICY under each candidate's reward -- the demo
    # screen's access class -- and each is refused where it would be a declared
    # knob nothing executes, or a comparison of two different quantities.
    metric = g("train.pruning_metric") or "own_reward"
    ceiling = g("train.pruning_cfg.ceiling") or "none"
    if ceiling == "demo_return":
        if rule == "none":
            p.append("train.pruning_cfg.ceiling=demo_return guards a stopping rule, but "
                     "train.pruning=none: there is nothing to guard")
        if rule == "successive_halving_pool":
            # The pool's within-training hook is `_prune_none`: the guard would
            # wrap a rule that never fires and `ceiling_decisions` would stay
            # empty forever -- a declared knob nothing executes. The pool ranks
            # on `pruning_metric: demo_fraction` instead, which reads the same
            # ceiling.
            p.append("train.pruning_cfg.ceiling=demo_return guards a WITHIN-training rule, "
                     "and train.pruning=successive_halving_pool stops nothing within a "
                     "training (it drops on the pool ranking); use a within-training rule "
                     "(plateau, median_stop, successive_halving, hyperband) with the "
                     "ceiling, or the pool with train.pruning_metric=demo_fraction and "
                     "ceiling=none")
        if g("problem.fitness_access") != "demonstrations":
            p.append("train.pruning_cfg.ceiling=demo_return rolls out a solution policy "
                     "from policies/ under each candidate reward; declare it with "
                     "problem.fitness_access=demonstrations")
        if metric == "task_metric":
            p.append("train.pruning_cfg.ceiling=demo_return compares the training's OWN "
                     "return to the expert's under the same reward, but "
                     "train.pruning_metric=task_metric watches env.task_metric -- two "
                     "different quantities; use own_reward or demo_fraction")
        cf = g("train.pruning_cfg.ceiling_fraction")
        if not isinstance(cf, (int, float)) or not (0.0 < cf <= 1.0):
            p.append(f"train.pruning_cfg.ceiling_fraction={cf!r} must be in (0, 1]")
    if metric == "demo_fraction":
        if g("problem.fitness_access") != "demonstrations":
            p.append("train.pruning_metric=demo_fraction is the training's return as a "
                     "share of the random->expert gap under its own reward, i.e. a "
                     "solution policy's rollouts; declare it with "
                     "problem.fitness_access=demonstrations")
        if rule == "none":
            p.append("train.pruning_metric=demo_fraction with train.pruning=none: a "
                     "metric no rule reads (own_reward is the inert default)")
    if rule == "successive_halving_pool":
        if metric == "own_reward":
            p.append("train.pruning=successive_halving_pool ranks candidates against each "
                     "other, and own_reward is in each candidate's own units; use "
                     "demo_fraction (or task_metric where that access is declared)")
        if int(g("train.seeds_per_candidate") or 1) != 1:
            p.append("train.pruning=successive_halving_pool continues each survivor from "
                     "its single stored policy (policy:<cand_id>); it needs "
                     "train.seeds_per_candidate=1")
        if (g("select.allocation") or "uniform") != "uniform":
            p.append("train.pruning=successive_halving_pool continues each survivor from "
                     "one stored policy, and a non-uniform select.allocation could hand "
                     f"a survivor several seeds; select.allocation={g('select.allocation')} "
                     "is refused -- the pool rule IS the allocation")
        if (g("train.hyperparameter_search") or "none") != "none":
            p.append("train.pruning=successive_halving_pool cannot continue a grid of "
                     "learner points across rungs; it needs train.hyperparameter_search=none")
        if (g("train.interaction") or "independent") != "independent":
            p.append("train.pruning=successive_halving_pool is a round driver, and so is "
                     f"train.interaction={g('train.interaction')}; a round has one driver")
        rf = g("train.pruning_cfg.rung_fraction")
        if not isinstance(rf, (int, float)) or not (0.0 < rf <= 1.0):
            p.append(f"train.pruning_cfg.rung_fraction={rf!r} must be in (0, 1]")
        eta = g("train.pruning_cfg.eta")
        if not isinstance(eta, int) or eta < 2:
            p.append(f"train.pruning_cfg.eta={eta!r} must be an int >= 2")

    # `complexity_cap` with both bounds off screens nothing: a declared screen that
    # cannot fire is a fabricated pin. And the Occam margin is read only
    # inside the incumbent rule, so a positive margin without it is a key nothing
    # honours.
    if g("verify.quality_screen") == "complexity_cap":
        caps = (int(g("verify.complexity_cap.max_components") or 0),
                int(g("verify.complexity_cap.max_ast_nodes") or 0))
        if any(c < 0 for c in caps):
            p.append("verify.complexity_cap.* bounds must be >= 0 (0 = that bound is off)")
        if not any(caps):
            p.append("verify.quality_screen=complexity_cap with max_components=0 and "
                     "max_ast_nodes=0 would screen nothing; set at least one bound")
    margin = g("select.incumbent_simplicity_margin")
    if margin is not None and float(margin) < 0:
        p.append(f"select.incumbent_simplicity_margin={margin!r} must be >= 0")
    if margin and float(margin) > 0 and not g("select.require_improvement_over_incumbent"):
        p.append("select.incumbent_simplicity_margin > 0 is read only under "
                 "select.require_improvement_over_incumbent=true; set it or drop the margin")

    # The demonstration verifier reads a SOLUTION POLICY's rollouts: privileged
    # information the config has to declare, and evidence that exists only when
    # the screen ran. Both directions are refused rather than left inert.
    dscreen = g("verify.quality_screen") == "demo_margin"
    if dscreen and g("problem.fitness_access") != "demonstrations":
        p.append("verify.quality_screen=demo_margin rolls out a solution policy from "
                 "policies/ under each candidate reward; declare it with "
                 "problem.fitness_access=demonstrations")
    arts = list(g("evaluate.artifacts") or [])
    if "demo_reward_traces" in arts and not dscreen:
        p.append("evaluate.artifacts includes demo_reward_traces but "
                 "verify.quality_screen is not demo_margin -- nothing would fill it")
    if g("evaluate.fitness.source") == "demo_margin" and not dscreen:
        p.append("evaluate.fitness.source=demo_margin needs verify.quality_screen="
                 "demo_margin to produce the score it ranks on")
    if dscreen:
        keep = g("verify.demo_screen.keep")
        kc_cap, kc_floor = g("verify.demo_screen.keep_cfg.cap"), g("verify.demo_screen.keep_cfg.floor")
        kc_pat, kc_dec = g("verify.demo_screen.keep_cfg.patience"), g("verify.demo_screen.keep_cfg.decay")
        frac = g("verify.demo_screen.keep_fraction")
        if keep == "adaptive":
            if not (isinstance(kc_floor, int) and isinstance(kc_cap, int) and 1 <= kc_floor <= kc_cap):
                p.append(f"verify.demo_screen.keep_cfg needs 1 <= floor ({kc_floor!r}) <= cap ({kc_cap!r})")
            if isinstance(kc_cap, int) and isinstance(n_cand, int) and kc_cap > n_cand:
                p.append(f"verify.demo_screen.keep_cfg.cap={kc_cap} exceeds generate.n_candidates={n_cand}")
            if not isinstance(kc_pat, int) or kc_pat < 1:
                p.append(f"verify.demo_screen.keep_cfg.patience={kc_pat!r} must be an int >= 1")
            if not isinstance(kc_dec, (int, float)) or not (0.0 < float(kc_dec) < 1.0):
                p.append(f"verify.demo_screen.keep_cfg.decay={kc_dec!r} must be in (0, 1)")
        if keep == "fraction" and (not isinstance(frac, (int, float)) or not (0.0 < float(frac) <= 1.0)):
            p.append(f"verify.demo_screen.keep_fraction={frac!r} must be in (0, 1]")
        if not list(g("verify.demo_screen.policies") or []):
            p.append("verify.demo_screen.policies is empty; the screen would have nothing to roll out")

    if g("train.reference_policy.enabled") and g("problem.fitness_access") != "demonstrations":
        p.append("train.reference_policy.enabled=true pulls the learner toward a solution "
                 "policy from policies/; declare it with problem.fitness_access=demonstrations")
    if g("train.reference_policy.enabled"):
        rb, rs = g("train.reference_policy.beta"), g("train.reference_policy.sigma")
        if not isinstance(rb, (int, float)) or rb < 0:
            p.append(f"train.reference_policy.beta={rb!r} must be >= 0")
        if not isinstance(rs, (int, float)) or rs <= 0:
            p.append(f"train.reference_policy.sigma={rs!r} must be > 0")

    # `verify.tpe.store` is §2's DECLARATION of the growth rule; the key stage 6
    # actually dispatches on is `update.memory.trajectory_store` (the screen
    # never writes to its own evidence). This rule is what makes the declared
    # key readable rather than silently ignored: flipping it alone on a
    # store-growing TPE config fails loudly here instead of doing nothing, as
    # a declared-but-unread key would.
    if g("verify.quality_screen") == "tpe" and \
            (g("update.memory.trajectory_store") or "none") != "none" and \
            g("verify.tpe.store") != g("update.memory.trajectory_store"):
        p.append(f"verify.tpe.store={g('verify.tpe.store')!r} disagrees with "
                 f"update.memory.trajectory_store="
                 f"{g('update.memory.trajectory_store')!r}: §2's declaration and "
                 "§6's growth rule must name the same policy when the TPE screen "
                 "reads a growing store (the §6 key is the one that dispatches)")

    if g("loop.curriculum.enabled"):
        carry = g("loop.carry") or []
        # The stage list without the position is not a curriculum: dropped, the
        # search restarts at stage 0 every iteration while every artifact still
        # says `curriculum: enabled`. The two travel in one slot precisely so
        # this cannot happen silently, and the slot still has to be carried.
        if "curriculum" not in carry:
            p.append("loop.curriculum.enabled=true requires 'curriculum' in loop.carry "
                     "(without it the stage list and the position are dropped every "
                     "iteration and the search restarts at stage 0)")
        # Carrying the policy across a handover is the MECHANISM, not a tuning
        # choice: a stage that starts from scratch inherits nothing from the
        # stage before it, which is the whole thing a curriculum is for.
        if "policy_checkpoint" not in carry:
            p.append("loop.curriculum.enabled=true requires 'policy_checkpoint' in "
                     "loop.carry (a stage that starts from scratch inherits nothing "
                     "from the one before it)")
        if g("train.init") == "from_scratch":
            p.append("loop.curriculum.enabled=true is inert under "
                     "train.init=from_scratch: the carried policy is never loaded, so "
                     "each stage retrains from nothing")
        if g("loop.curriculum.gate") == "vlm_ensemble" and \
                g("llm.evaluator.provider") == "none":
            p.append("loop.curriculum.gate=vlm_ensemble needs an evaluator client "
                     "(llm.evaluator.provider is none, so no stage could ever pass)")
        if g("loop.curriculum.regression_check") and \
                g("loop.curriculum.gate") == "fixed_budget":
            p.append("loop.curriculum.regression_check=true is inert under "
                     "gate=fixed_budget: that gate records no pass score, so there is "
                     "nothing for a regression to be measured against")

    if g("generate.context.include_curriculum_stage") and not g("loop.curriculum.enabled"):
        p.append("generate.context.include_curriculum_stage=true requires "
                 "loop.curriculum.enabled=true (nothing defines the stages)")

    # NO CONTEST, BUT A POOL. `select.rule: none` means there is nothing to rank
    # -- it returns `reports[:1]` and DISCARDS the rest. Asking for K > 1 under
    # it therefore pays K generations and K trainings to keep an arbitrary one,
    # which is strictly worse than K = 1 and is not the method any published
    # config with this rule describes.
    #
    # A `log.warning` inside `rule_none` would be the wrong place: it fires per
    # iteration, at runtime, after the LLM calls are already spent -- a sweep of
    # `card` at n_candidates: 16 would have every task generate 16 candidates to
    # keep one before anything objected. Here it is caught at load, before
    # anything is queued.
    #
    # Stated over KEYS, never over a method name: `select.rule` is the signal,
    # so this covers card, limen and text2reward_human alike and stays honest if
    # a new config picks the rule. `tests/test_no_method_branching.py` forbids
    # the alternative. The fix is to set K = 1, or to give the config a real
    # fitness source and a ranking rule -- at which point it is a different
    # method and should say so in its name.
    # `n_cand_or`, not `g(...) or 0`: the schema reports a non-int value, and
    # this comparison must not raise on it first (see the n_survivors rule).
    n_cand = n_cand_or(g, 0)
    if g("select.rule") == "none" and n_cand > 1:
        p.append(f"select.rule=none with generate.n_candidates={n_cand}: this rule "
                 f"ranks nothing and keeps the first candidate, so {n_cand - 1} of "
                 f"every {n_cand} would be generated, trained and thrown away. Set "
                 "generate.n_candidates=1, or choose a select.rule that ranks")

    # `loop.termination: fixed_generations` (CARD) ends the loop on a generation
    # that is validity-checked and nothing else, so its last report has no
    # fitness. Exactly one §5 pair can return that program: `rule none` is the
    # only rule that adopts a `fitness=None` report as winner (`_scored` drops
    # None, so an argmax rule would leave the final generation unadopted and
    # `chain_end` would return R_{N-2}), and `chain_end` is the only final
    # artifact whose return is defined for an unscored program
    # (`global_best`/`archive_best`/`all_survivors` read fitness-ranked slots).
    # Any other pair pays the last LLM call and discards it, so it is refused
    # here, at load, rather than discovered in the logs. With the `rule none`
    # rule above this pins the generate-only step to K = 1 chains, the only
    # shape it is defined for. Stated over KEYS, never over a method name.
    if g("loop.termination") == "fixed_generations" and \
            g("select.final_artifact") != "chain_end":
        p.append("loop.termination=fixed_generations ends the loop on a generation with "
                 "no fitness; only select.final_artifact=chain_end can return it (got "
                 f"{g('select.final_artifact')!r}) -- global_best/archive_best/"
                 "all_survivors would pay the last LLM call and discard it")
    if g("loop.termination") == "fixed_generations" and g("select.rule") != "none":
        p.append("loop.termination=fixed_generations requires select.rule=none: the "
                 "final generation is unscored and only `none` adopts an unscored report "
                 f"as chain head (got {g('select.rule')!r})")

    if g("select.rule") == "map_elites_insert" and not g("update.archive.enabled"):
        p.append("select.rule=map_elites_insert requires update.archive.enabled=true")

    if g("select.final_artifact") == "archive_best" and not g("update.archive.enabled"):
        p.append("select.final_artifact=archive_best requires update.archive.enabled=true")

    if g("update.archive.enabled"):
        d = g("update.archive.descriptors") or []
        b = g("update.archive.bins_per_descriptor") or []
        if len(d) != len(b):
            p.append(f"update.archive: {len(d)} descriptors but {len(b)} bin counts")

    # `select.n_survivors` is a nullable int and the runtime readers
    # (`selection._cfg_int`, update.py's `or 1`) treat null as 1, so null is a
    # legal spelling of the default. `Config.get(path, default)` returns the
    # STORED value when the key exists, so a present null arrives here as None
    # and `None > 1` would raise a raw TypeError out of validate() -- before the
    # collected `problems` are reported, and with the config path lost. Coerce
    # as the runtime does. `isinstance` rather
    # than `or 1` so a non-int value is left to the schema's own type report
    # instead of raising here.
    _nsurv = g("select.n_survivors")
    if (_nsurv if isinstance(_nsurv, int) and not isinstance(_nsurv, bool) else 1) > 1 \
            and g("update.topology") == "single_parent_hillclimb":
        p.append("select.n_survivors>1 is meaningless under "
                 "update.topology=single_parent_hillclimb (only one parent is ever used)")

    # `loop.max_parallel_trainings` is `int | str`, and the only legal string is
    # `auto` -- the schema cannot say that, so it is said here. Without this a
    # typo validates cleanly and then dies in `int()` deep inside stage 3, after
    # generation has already spent LLM calls. `--validate-all` is supposed to be
    # the place a bad key is caught.
    mpt = g("loop.max_parallel_trainings")
    if isinstance(mpt, str):
        # Exactly `auto`, not `AUTO` or `auto ` -- the resolved config is an
        # artifact and the run ID is its hash, so two spellings of one value
        # would be two run IDs for one run.
        if mpt != "auto":
            p.append(f"loop.max_parallel_trainings={mpt!r}: the only string accepted is "
                     "'auto' (worker count from SLURM_CPUS_PER_TASK or the CPU affinity "
                     "mask); otherwise give a positive integer")
    elif isinstance(mpt, bool) or (mpt is not None and int(mpt) < 1):
        p.append(f"loop.max_parallel_trainings={mpt!r} must be a positive integer or 'auto'")

    # `loop.resume_from` writes its checkpoint at the same iteration boundary
    # that writes `state/iterNN.json`. Deliberately the ONLY static rule: there
    # is no refusal keyed on `loop.carry` or `train.init`, because a degradation
    # is a ref that is non-null whose payload could not be persisted, which is a
    # runtime fact: `_sb3_run` writes a `_PolicyBlob` / `_ReplaySlice` per
    # candidate and `update._carry_inner_loop_refs` publishes the incumbent's,
    # and whether a ref resolves is a fact about eviction and about which
    # process wrote it, not about the config.
    if g("loop.resume_from") is not None and not g("output.save_state_every_iteration"):
        p.append("loop.resume_from requires output.save_state_every_iteration=true "
                 "(the checkpoint is written at the same boundary as the state report)")

    # `warm_start_from_best` reads `state.policy_ref`, which only survives the
    # iteration boundary if `policy_checkpoint` is in `loop.carry`
    # (`training.py`). Without it the mode resolves, is recorded in
    # `config.resolved.yaml`, contributes its bytes to the run id -- and trains
    # from scratch anyway: `warm_start_from_best` quietly becomes
    # `from_scratch`.
    # `warm_start_from_parent` / `warm_start_from_similar` (unpublished) have
    # the same requirement: their fallback IS
    # `warm_start_from_best`, and a lineage or similarity ref alone would make
    # the round after a regression cold-start with the method's name on it.
    if g("train.init") in ("warm_start_from_best", "bc_prior_then_warm_start",
                           "warm_start_from_parent", "warm_start_from_similar") and \
            "policy_checkpoint" not in (g("loop.carry") or []):
        p.append(f"train.init={g('train.init')} requires 'policy_checkpoint' in "
                 "loop.carry -- without it the policy does not survive the iteration "
                 "boundary and training silently starts from scratch")
    if g("train.init") in ("warm_start_from_parent", "warm_start_from_similar"):
        # Both read OTHER candidates' policies out of the FIFO policy store
        # (`training._STORE_LIMIT` entries; the literal below is pinned equal to
        # it by `tests/test_train_init.py`). Within that bound nothing is ever
        # evicted, so `parallel` and `sequential` read the same store; past it
        # the sequential loop's same-round writes evict the oldest donors before
        # a later candidate plans, which a forked worker never sees, and the
        # two schedules diverge. Refused rather than documented: bit-identity
        # between `parallel` and `sequential` is an invariant, not a footnote.
        cap = g("budget.max_policy_trainings")
        n_it = g("loop.n_iterations")
        if isinstance(cap, int) and not isinstance(cap, bool) and cap > 0:
            trainings: Optional[int] = int(cap)
        elif isinstance(n_it, int) and not isinstance(n_it, bool):
            vals = g("generate.candidate_schedule_values") or []
            if g("generate.candidate_schedule") == "explicit" and vals:
                slots = int(sum(int(v) for v in vals))
            else:
                slots = int(g("generate.n_candidates") or 0) * int(n_it)
            # A slot can store more than one policy: a repair (`verification.
            # _regenerate`, the runtime repair) re-mints the candidate under a
            # new id and trains again, so bound by 1 + the repair cap per slot
            # (0 = uncapped, which this estimate cannot bound).
            repairs = g("verify.max_repair_attempts") or 0
            trainings = slots * (1 + (int(repairs) if isinstance(repairs, int) and repairs > 0 else 0))
        else:
            trainings = None
        if trainings is not None and trainings > _POLICY_STORE_LIMIT:
            p.append(f"train.init={g('train.init')} reads other candidates' policies from a "
                     f"{_POLICY_STORE_LIMIT}-entry store, and this run can train {trainings} "
                     "(budget.max_policy_trainings, else n_candidates x n_iterations / the "
                     "explicit schedule's sum, x (1 + verify.max_repair_attempts) for re-minted "
                     "repairs): past the store's size a forked worker and the "
                     "sequential loop see different donors. Cap budget.max_policy_trainings at "
                     f"{_POLICY_STORE_LIMIT} or fewer, or train fewer candidates")
    if g("train.init") == "secondary_replay_buffer" and \
            "replay_buffer" not in (g("loop.carry") or []):
        p.append("train.init=secondary_replay_buffer requires 'replay_buffer' in loop.carry")

    # ROSKA. `fused_warm_start` blends the previous best policy with a fresh one,
    # so it needs a ref to blend WITH: without `policy_checkpoint` carried, the
    # ref is None every round, the blend never happens, and the run is
    # `from_scratch` wearing ROSKA's name -- the two-config-points-one-method
    # shape. Same argument as `secondary_replay_buffer` above.
    if g("train.init") == "fused_warm_start" and \
            "policy_checkpoint" not in (g("loop.carry") or []):
        p.append("train.init=fused_warm_start requires 'policy_checkpoint' in loop.carry "
                 "(nothing to fuse with otherwise)")
    # The search must be able to run. `sc_bo` scores candidate ratios by training
    # short probes, so a zero probe budget is not a cheaper search -- it is no
    # search, and it would silently deliver `train.fusion.alpha` under a config
    # that says the ratio was optimised. Refused here rather than noted in the
    # artifact, because this is statically decidable and the alternative spends
    # a full sweep before anyone reads the note.
    if g("train.fusion.ratio_search") == "sc_bo":
        if float(g("train.fusion.sc_bo.probe_fraction") or 0.0) <= 0:
            p.append("train.fusion.ratio_search=sc_bo requires "
                     "train.fusion.sc_bo.probe_fraction > 0 (a zero-length probe scores nothing)")
        if int(g("train.fusion.sc_bo.n_evaluations") or 0) <= 0:
            p.append("train.fusion.ratio_search=sc_bo requires "
                     "train.fusion.sc_bo.n_evaluations > 0")
        if not (g("train.fusion.sc_bo.init_points") or []):
            p.append("train.fusion.ratio_search=sc_bo requires a non-empty "
                     "train.fusion.sc_bo.init_points (the GP's initial design)")
    # Two schedules cannot both own a candidate's budget. `successive_halving_pool`
    # passes `env_steps` per rung; ROSKA's schedule sets it per round. Refused
    # rather than given a precedence rule, because whichever lost would be a
    # silently different budget from the one the config states.
    if g("train.init") == "fused_warm_start" and g("train.pruning") == "successive_halving_pool":
        p.append("train.init=fused_warm_start and train.pruning=successive_halving_pool both "
                 "set a candidate's env_steps; pick one")
    # The Eureka-plus-probes shape: a fused_warm_start config that never reduces
    # the per-round budget spends a FULL training per candidate and then the
    # search's probes on top, which costs more than Eureka rather than 0.89 of
    # it -- the opposite of the method's headline claim, and invisible except in
    # the total.
    if g("train.init") == "fused_warm_start" \
            and float(g("train.fusion.first_round_fraction") or 0.0) <= 0 \
            and float(g("train.fusion.sc_bo.post_probe_fraction") or 0.0) <= 0:
        p.append("train.init=fused_warm_start with neither "
                 "train.fusion.first_round_fraction nor train.fusion.sc_bo.post_probe_fraction "
                 "set: every candidate would train for a FULL train.env_steps and pay the "
                 "fusion probes on top, which is more expensive than not fusing at all")
    if g("train.fusion.ratio_search") != "fixed" and g("train.init") != "fused_warm_start":
        p.append(f"train.fusion.ratio_search={g('train.fusion.ratio_search')} has no effect "
                 "unless train.init=fused_warm_start")

    # `bc_prior` clones the scripted policy on the RAW observation; a co-designed
    # observation (LIMEN) gives every candidate its own feature space, and there
    # is no clone in it. `kl_clone` anchors to the policy the candidate STARTED
    # from, so `from_scratch` has nothing to anchor to; and the penalty lives in
    # a PPO subclass, so any other algorithm would resolve the key and ignore it
    # -- a fabricated pin.
    if g("train.init") in ("bc_prior", "bc_prior_then_warm_start") and \
            "observation" in (g("problem.search_space") or []):
        p.append("train.init=bc_prior cannot be combined with 'observation' in "
                 "problem.search_space: the clone is fit on the raw observation and a "
                 "candidate that chooses its own has no clone")
    if g("train.init") in ("bc_prior", "bc_prior_then_warm_start") and eff_backend == "sb3":
        # The clone needs a scripted policy for the env. `--validate-all` exists to
        # catch a non-runnable configuration before a launched job does, so the
        # selector is resolved here, on the backend that will actually build the
        # clone -- the surrogate backends warn and cold-start instead of building.
        from .policies import PolicyError, policy_for_env
        try:
            policy_for_env(g("problem.env_id"), g("train.bc_prior.policy") or "auto")
        except (ValueError, PolicyError) as exc:
            p.append(f"train.init=bc_prior: train.bc_prior.policy={g('train.bc_prior.policy')!r} "
                     f"does not resolve for problem.env_id={g('problem.env_id')!r}: {exc}")
    if g("evaluate.rejudge_incumbent") and g("evaluate.fitness.source") != "vlm_score":
        p.append("evaluate.rejudge_incumbent re-scores the incumbent with the VLM judge and "
                 "needs evaluate.fitness.source: vlm_score "
                 f"(got {g('evaluate.fitness.source')!r})")
    # `accept_min_metric` is read by exactly one acceptance rule. Set under another rule it
    # is a declared-but-unread pin; missing under
    # `metric_floor` the collector has no floor to apply.
    if g("train.bc_prior.accept") == "metric_floor" and g("train.bc_prior.accept_min_metric") is None:
        p.append("train.bc_prior.accept=metric_floor requires train.bc_prior.accept_min_metric")
    if g("train.bc_prior.accept") != "metric_floor" and g("train.bc_prior.accept_min_metric") is not None:
        p.append(f"train.bc_prior.accept_min_metric={g('train.bc_prior.accept_min_metric')!r} is read "
                 f"only under train.bc_prior.accept=metric_floor (got {g('train.bc_prior.accept')!r})")
    # The four `fallback_*` keys have the same contract, one pair per fallback rule:
    # `success_else_top` reads `fallback_oversample` + `fallback_keep` and nothing else does;
    # `success_else_best` reads `fallback_min_metric` + `fallback_max_episodes` likewise. So
    # each pair is required under its rule and refused under every other.
    for _rule, _keys in (("success_else_top", ("fallback_oversample", "fallback_keep")),
                         ("success_else_best", ("fallback_min_metric", "fallback_max_episodes"))):
        for _k in _keys:
            _key = f"train.bc_prior.{_k}"
            if g("train.bc_prior.accept") == _rule and g(_key) is None:
                p.append(f"train.bc_prior.accept={_rule} requires {_key}")
            if g("train.bc_prior.accept") != _rule and g(_key) is not None:
                p.append(f"{_key}={g(_key)!r} is read only under train.bc_prior.accept={_rule} "
                         f"(got {g('train.bc_prior.accept')!r})")
    if g("train.anchor.kind") in ("kl_clone", "kl_reward"):
        kind = g("train.anchor.kind")
        if g("train.init") == "from_scratch":
            p.append(f"train.anchor.kind={kind} anchors to the candidate's initial policy; "
                     "train.init=from_scratch has none -- use bc_prior or warm_start_from_best")
        if kind == "kl_clone" and g("train.algorithm") != "ppo":
            p.append("train.anchor.kind=kl_clone is implemented for train.algorithm=ppo only "
                     f"(got {g('train.algorithm')!r}); kl_reward is the algorithm-agnostic form")
        if g("train.anchor.schedule") == "adaptive" and not (g("train.anchor.target") or 0) > 0:
            p.append("train.anchor.schedule=adaptive needs train.anchor.target > 0")
        if g("train.anchor.beta_min") is not None and float(g("train.anchor.beta_min")) < 0:
            p.append("train.anchor.beta_min must be >= 0 (0 means no floor)")
    # Only this direction: `train.backend` is what executes and `train.algorithm`
    # is the paper's citation, so fasttd3 EXECUTING under some other citation is
    # the same shape as `mock` running a surrogate there (the seed rows record
    # `learner`/`backend: fasttd3` beside the citation). The reverse -- a
    # CITATION sb3 has no learner for, handed to sb3 -- is refused for EVERY
    # such algorithm, not only fasttd3. A lenient map (`{...}.get(name, PPO)`)
    # would let `l2r` (`none`) and `singh_orp` (`q_learning`) under the
    # `dev`/`full` profiles -- which pin `train.backend: sb3` and outrank the
    # method's own `backend: none`/`tabular` -- resolve with exit 0 and train
    # PPO for `train.env_steps` under a seed row citing an algorithm that
    # trains nothing. The backend's map (`training._sb3_algo_and_hyper`) is
    # strict as well; this is where the operator learns it at load rather than
    # hours into a run. `SB3_ALGORITHMS` is the one copy of the list the
    # backend reads too, so the two cannot drift.
    if eff_backend == "sb3" and g("train.algorithm") not in SB3_ALGORITHMS:
        alg = g("train.algorithm")
        way_out = {
            "fasttd3": "use train.backend: fasttd3",
            "none": ("a method that trains no policy runs under train.backend: none "
                     "(pass -s train.backend=none when a profile pins sb3)"),
            "q_learning": ("tabular Q-learning runs under train.backend: tabular "
                           "(pass -s train.backend=tabular when a profile pins sb3)"),
        }.get(alg, "pick a train.backend that implements it")
        p.append(f"train.algorithm={alg} has no sb3 implementation (sb3 implements "
                 f"{', '.join(SB3_ALGORITHMS)}); {way_out}")

    # A REPLAY BUFFER NEEDS AN OFF-POLICY ALGORITHM TO MIX IT INTO. The rule
    # above checks the buffer is CARRIED; nothing checked that the learner can
    # use it. `train.backend: sb3` answers an on-policy algorithm with
    # `log.warning("train.algorithm=%s on sb3: an on-policy algorithm has no
    # replay buffer to mix into")` and trains from scratch anyway
    # (bird/components/training.py), so `gt` -- which carries
    # `replay_buffer` and pins `train.init: secondary_replay_buffer` -- would
    # quietly lose its cross-iteration policy reuse and still report a complete
    # run.
    #
    # Same shape as the guard above: the mechanism is declared, the config
    # validates, and the thing does not execute. A warning in a log at hour
    # three is not a control.
    if g("train.init") == "secondary_replay_buffer" and \
            g("train.algorithm") in _ON_POLICY:
        p.append(f"train.init=secondary_replay_buffer with train.algorithm="
                 f"{g('train.algorithm')}: an on-policy algorithm has no replay "
                 "buffer to initialise from, so the carried buffer would be "
                 "silently discarded. Use an off-policy algorithm (sac, td3, "
                 "qr_sac) or train.init=from_scratch")

    if g("update.co_evolve.subtasks") and "subtask_list" not in (g("loop.carry") or []):
        p.append("update.co_evolve.subtasks=true requires 'subtask_list' in loop.carry")

    if g("generate.co_design.observation_fn") and \
            "observation" not in (g("problem.search_space") or []):
        p.append("generate.co_design.observation_fn=true requires 'observation' in "
                 "problem.search_space")

    # `update.co_evolve.observation_fn` is the §6 half of §1's
    # `co_design.observation_fn`: whether the co-designed observation PERSISTS
    # across iterations (`generation._inheritable_program` reads it). Persisting
    # an observation that is never generated is the fabricated-pin shape, so
    # `true` needs §1's `true`.
    if g("update.co_evolve.observation_fn") and not g("generate.co_design.observation_fn"):
        p.append("update.co_evolve.observation_fn=true requires "
                 "generate.co_design.observation_fn=true: an observation function that "
                 "is never generated cannot persist across iterations")
    # And the corner where NOT persisting leaves no way for one to arrive:
    # `weights_only` copies the parent's program verbatim and discards the
    # model's code, so from the first iteration with a parent every child would
    # fail `_apply_co_design` for defining no get_observation.
    if g("generate.co_design.observation_fn") and \
            not g("update.co_evolve.observation_fn") and \
            g("generate.output.edit_mode") == "weights_only":
        p.append("generate.output.edit_mode=weights_only with "
                 "generate.co_design.observation_fn=true requires "
                 "update.co_evolve.observation_fn=true: weights_only inherits the parent's "
                 "program verbatim, so a non-persisting observation has no way to reach a "
                 "child and every child after the first iteration would be invalid")

    if g("generate.co_design.dr_config") and "dr" not in (g("problem.search_space") or []):
        p.append("generate.co_design.dr_config=true requires 'dr' in problem.search_space")

    # -- §0's two DESCRIPTIVE axes, enforced by coherence with what dispatches --
    #
    # `problem.reward_representation` and `problem.search_space_mode` are read
    # by no stage: a reward's FORM is dispatched by `generate.output.format` and
    # `generate.generator_backend`, and STAGING by the `pre` phases. Declared
    # and unread, either could be ablated alone -- the config would validate,
    # the hash move, the run be byte-identical (e.g. `rda` with
    # `-s problem.reward_representation=free_form_code`), and `--diff` list
    # the key as a between-method contribution. The fix for a
    # declared-but-unread key is to honour it, never to delete it quietly:
    # each value is tied to the dispatching value(s) that implement it, so an
    # inconsistent pair fails here naming both keys.
    _REPRESENTATION_FORMATS = {
        "free_form_code": ("component_dict_return", "scalar_only"),
        "weighted_components": ("component_dict_plus_weights",),
        "template_dsl": ("template_params",),
    }
    rep = g("problem.reward_representation")
    fmt = g("generate.output.format")
    gen_backend = g("generate.generator_backend") or "llm"
    if rep == "tabular":
        # Singh (2009): the reward IS a table, enumerated by the backend; the
        # output format is the shape the enumerator writes the table in.
        if gen_backend != "exhaustive_enumeration":
            p.append(f"problem.reward_representation='tabular' declares an enumerated "
                     f"reward table, which only generate.generator_backend="
                     f"exhaustive_enumeration produces (got {gen_backend!r}); the two "
                     f"keys describe one reward and must agree")
    elif gen_backend == "exhaustive_enumeration":
        p.append(f"generate.generator_backend=exhaustive_enumeration enumerates a reward "
                 f"TABLE, so problem.reward_representation must be 'tabular' (got "
                 f"{rep!r}); the two keys describe one reward and must agree")
    elif rep in _REPRESENTATION_FORMATS:
        allowed = _REPRESENTATION_FORMATS[rep]
        if fmt not in allowed:
            implied = next((r for r, fs in _REPRESENTATION_FORMATS.items() if fmt in fs), None)
            p.append(f"problem.reward_representation={rep!r} is implemented by "
                     f"generate.output.format in {list(allowed)}, but this config "
                     f"dispatches generate.output.format={fmt!r}"
                     + (f" (the form of {implied!r})" if implied else "")
                     + "; the two keys describe one reward and must agree")

    # -- declared, not implemented: refuse anything but the default ----------
    #
    # Both keys have a default and a schema Field, and no reader anywhere in
    # bird/. Set, either would validate, move the hash, open a new run
    # directory and execute
    # the unmodified method -- a null result for a mechanism that never ran.
    # The precedent for the class (`verify.tpe.store`, `accept_min_metric`
    # above) is to refuse loudly here until the key is honoured; deleting it
    # would hide that the schema was right to claim the capability.
    if g("select.retrain_before_select"):
        p.append("select.retrain_before_select=true is declared but not implemented: no "
                 "stage retrains candidates before the selection comparison, so the run "
                 "would be the unmodified method under a new hash (a declared-but-unread "
                 "key). Leave it false until an implementation lands")
    if (g("update.meta.prompt_optimizer") or "none") != "none":
        p.append(f"update.meta.prompt_optimizer={g('update.meta.prompt_optimizer')!r} is "
                 "declared but not implemented: nothing rewrites the prompt or feedback "
                 "templates, so the run would be the unmodified method under a new hash "
                 "(a declared-but-unread key). Only 'none' is honoured until an "
                 "implementation lands")

    mode = g("problem.search_space_mode")
    surfaces = list(g("problem.search_space") or [])
    pre_names = [ph["name"] if isinstance(ph, dict) else ph for ph in (g("pre") or [])]
    if mode == "staged":
        # DrEureka: the reward loop first, then `dr_generation` searches the
        # DR surface as a separate program (`pre: [rapp, dr_generation]`).
        if len(surfaces) < 2:
            p.append(f"problem.search_space_mode='staged' declares surfaces searched in "
                     f"sequence, but problem.search_space={surfaces!r} names only one; "
                     f"there is nothing to stage")
        elif "dr_generation" not in pre_names:
            p.append(f"problem.search_space_mode='staged' is implemented by the "
                     f"dr_generation pre phase (DrEureka: `pre: [rapp, dr_generation]` "
                     f"searches the dr surface after the reward loop); this config's "
                     f"pre={pre_names!r} stages nothing, so the key would be inert")
    elif mode == "joint" and "dr_generation" in pre_names:
        p.append(f"problem.search_space_mode='joint' declares the surfaces are searched "
                 f"together, but pre={pre_names!r} searches the dr surface in a separate "
                 f"stage (dr_generation): the mode this config runs is 'staged'")

    if g("verify.tpe.enabled") and g("verify.quality_screen") != "tpe":
        p.append("verify.tpe.enabled=true but verify.quality_screen is "
                 f"{g('verify.quality_screen')!r}; set it to 'tpe'")

    if g("verify.alignment_filter.enabled") and g("verify.quality_screen") != "tac":
        p.append("verify.alignment_filter.enabled=true but verify.quality_screen is "
                 f"{g('verify.quality_screen')!r}; set it to 'tac'")

    frac = g("verify.cascade.short_budget_fraction")
    if frac is not None and not (0 < float(frac) <= 1):
        p.append("verify.cascade.short_budget_fraction must be in (0, 1] or null, "
                 f"got {frac!r}; it is a fraction of train.env_steps, not a step count")

    if g("verify.cascade.enabled") and g("verify.quality_screen") != "cascade":
        p.append("verify.cascade.enabled=true but verify.quality_screen is "
                 f"{g('verify.quality_screen')!r}; set it to 'cascade'")

    # `train.checkpoint_selection: best_by_reward` restores the best checkpoint's
    # PARAMETERS into the learner. The planner backend (`train.backend: none`,
    # L2R's online MPC) learns none, so on it the key could only ever be recorded
    # as "nothing to restore" -- a declared key no backend honours. Refused at
    # load rather than reported hours into a run.
    if g("train.checkpoint_selection", "final") != "final" and eff_backend == "none":
        p.append(f"train.checkpoint_selection={g('train.checkpoint_selection')} has nothing "
                 "to restore under train.backend=none (the planner learns no parameters); "
                 "use final")
    # `min_delta` is a FRACTION of the curve's range. Below 0 the guard never
    # fires (any strictly better row restores); above 1 it always fires
    # (`best_by_reward` silently becomes `final` under another reason string).
    md = g("train.checkpoint_selection_cfg.min_delta")
    if md is not None and not (0.0 <= float(md) <= 1.0):
        p.append(f"train.checkpoint_selection_cfg.min_delta={md!r} must be in [0, 1]: "
                 "it is a fraction of the reward curve's range, not a return")

    if g("train.algorithm") == "none" and g("evaluate.fitness.source") not in (
            "none", "human_score", "vlm_score"):
        p.append("train.algorithm=none (no policy learning) cannot produce "
                 f"evaluate.fitness.source={g('evaluate.fitness.source')!r}")

    if g("loop.termination") in ("fitness_plateau", "any_of") and \
            g("loop.termination_cfg.patience") is None:
        p.append(f"loop.termination={g('loop.termination')!r} requires "
                 "loop.termination_cfg.patience")

    # -- per-stage history (L2R) ------------------------------------------
    stage_hist = g("generate.stage_history_modes") or []
    if stage_hist and not g("generate.output.two_stage_nl_then_code"):
        p.append("generate.stage_history_modes is per-LLM-call, so it needs more than one "
                 "call: set generate.output.two_stage_nl_then_code=true or leave it []")
    if stage_hist and len(stage_hist) != 2:
        p.append(f"generate.stage_history_modes has {len(stage_hist)} entries but the "
                 "two-stage generator makes exactly 2 calls (thinker, coder)")

    # -- per-candidate hyperparameter search (Singh) ----------------------
    if g("train.hyperparameter_search") != "none" and not (g("train.hyperparameter_grid") or {}):
        p.append(f"train.hyperparameter_search={g('train.hyperparameter_search')!r} "
                 "requires a non-empty train.hyperparameter_grid")
    if g("train.hyperparameter_search") == "none" and (g("train.hyperparameter_grid") or {}):
        p.append("train.hyperparameter_grid is set but train.hyperparameter_search=none, "
                 "so it would never be searched")

    # -- staged validity filtering (LIMEN) --------------------------------
    known_checks = set(g("verify.static_checks") or []) | set(g("verify.dynamic_checks") or [])
    for check in (g("verify.stage_timeouts_s") or {}):
        if check not in known_checks:
            p.append(f"verify.stage_timeouts_s names {check!r}, which is in neither "
                     "verify.static_checks nor verify.dynamic_checks")
    if g("verify.check_order") == "staged" and not g("verify.enabled"):
        p.append("verify.check_order=staged has nothing to order with verify.enabled=false")

    # -- the task definition must exist, and must supply what it is asked to ---
    # The FIRST filesystem-derived rules in this function. They read a memoised in-memory
    # index (`tasks.index()` is scanned once, at registry load, where failure is already
    # fatal), so each rule stays a pure function of (cfg, index) and the disk is not
    # touched per config. The argument for having them at all is the one written above
    # `loop.max_parallel_trainings`: `--validate-all` is supposed to be the place a bad
    # key is caught. A task id naming no file otherwise dies at `registry.get("env", ...)`
    # 20-47 s into a launched job, leaving a run directory that holds a config and an
    # empty journal -- one per job in a sweep.
    from .tasks import TaskSpecError as _TaskSpecError, available as _tasks_available
    from .tasks import by_env_id as _by_env_id, load as _load_spec

    _env_id = str(g("problem.env_id") or "")
    _task_id = g("problem.task_id")
    _spec = None
    if _task_id:
        try:
            _spec = _load_spec(str(_task_id))
        except _TaskSpecError:
            p.append(f"problem.task_id={_task_id!r} names no task spec; "
                     f"tasks/ holds {list(_tasks_available())}")
        else:
            if _spec.bird_env_id != _env_id:
                p.append(
                    f"problem.task_id={_task_id!r} defines {_spec.bird_env_id!r} but "
                    f"problem.env_id={_env_id!r}. Changing the task means changing BOTH "
                    "or neither: an env id on its own leaves the run describing the "
                    "previous environment to the model, which is a silently wrong "
                    "experiment rather than a crash")
    else:
        try:
            _spec = _by_env_id(_env_id)
        except _TaskSpecError as exc:
            p.append(str(exc))

    # NOT `if _spec is None` -- that asks whether a spec EXISTS, and what matters is
    # whether one SUPPLIED a value. `load()` materialises the inheritance before
    # `validate()` runs, so a key still None here was filled by nobody: either no spec at
    # all, or a spec whose own field is null. Gating on the spec's existence would
    # suppress the second case exactly (a catalogue spec may carry a null
    # `description.l_task`), and the symptom would be the prompt's first line reading
    # `None`, silently.
    # ASK THE RESOLVER what it would supply, rather than restating its rules. Each PROXY
    # for "nothing would supply it" is wrong in a different direction: `_spec is None`
    # asks whether a spec EXISTS (silent when one exists and its own field is null), and
    # a bare `g(_key) is None` assumes materialisation has already run (it has under
    # `load()`, but `validate()` is also called directly on a config that never went
    # through it -- `_default.yaml`, say, whose spec supplies both keys perfectly well).
    # Running `_inherit_from_task_spec` is the only statement of the rule that cannot
    # disagree with the rule.
    _supplied = _inherit_from_task_spec({
        "problem": {"env_id": _env_id, "task_id": _task_id, "task_description": None},
        "verify": {"forbidden_symbols": None},
        "rapp": {"parameters": None, "enabled": g("rapp.enabled")},
    })
    for _key in TASK_SPEC_KEYS:
        _section, _leaf = _key.split(".", 1)
        if (g(_key) is None and _task_spec_key_applies(_key, g)
                and (_supplied.get(_section) or {}).get(_leaf) is None):
            _src = (f"task spec {_spec.id!r} leaves it null" if _spec is not None
                    else f"environment {_env_id!r} has no spec to inherit from")
            p.append(
                f"{_key} is null, meaning 'inherit from the task spec', but {_src}. "
                "Nothing would supply it"
                + (" and the prompt's first line would read 'None'"
                   if _key == "problem.task_description" else ""))

    if g("rapp.enabled") and not (g("rapp.parameters") or []):
        p.append("rapp.enabled=true but rapp.parameters resolved to an empty list, so the "
                 "RAPP sweep would randomise nothing. DrEureka's stage-1 mechanism IS "
                 "those axis names; an unknown one is dropped silently by "
                 "EnvAdapter.set_dr, which is how a sweep measures nothing and logs "
                 "`degenerate`")

    # -- the anti-leakage gate must actually be wired up ------------------
    # A `forbidden_symbols` list that no check ever reads is a fabricated pin
    # -- and the one case where the fabrication is dangerous rather than
    # merely untidy, because the key's whole job is to stop a candidate
    # reaching for the metric it is scored on.
    # TWO CASES, and they are not the same question.
    #
    # Because `verify.forbidden_symbols` inherits by default, a list is present
    # whether or not the config asked for one. Keying the WHOLE check on "is a list
    # present" would fire on `-s problem.env_id=mt10_reach-v3` over a tester config,
    # where nothing was fabricated. Keying the whole check on "did the config author
    # it" is worse: inheriting is the DEFAULT, so both cases would stop covering the
    # common path, and a check that only fires where nobody goes is in effect deleted.
    #
    # So: authorship decides case ONE, and it alone. Why:
    #
    #   Case two is an inconsistency WITHIN the verification config. Verification is
    #   running, it is checking things, it is not checking this one. Nobody chooses that,
    #   whatever the list's provenance, so origin is noise there.
    #
    #   Case one is a coherent global opt-out. `verify.enabled: false` makes EVERY
    #   `verify.*` key inert, not just this one -- singling out `forbidden_symbols` would
    #   mean equally refusing an inherited `static_checks` or an inherited quality screen.
    #   What makes authorship the discriminator here is that an AUTHORED list is a
    #   positive statement by that file -- "these symbols matter" -- sitting next to "I
    #   run no verification". That is a self-contradiction inside one config, which is
    #   exactly what a coherence check exists to catch. An INHERITED list is not a
    #   statement by that config at all, so there is nothing for it to contradict.
    #
    # Refuse when the config contradicts itself; separately refuse when verification is
    # running but incompletely wired.
    #
    # KNOWN RESIDUAL, named rather than fixed: a NON-tester config that sets
    # `verify.enabled: false` on an env supplying a real gate is silent here. No shipped
    # config is in that state. If it ever matters, set `verify.enabled=True` in whatever
    # trips over it rather than weaken the rule.
    _in_force = list(g("verify.forbidden_symbols") or [])
    # What INHERITANCE would supply, read off the resolver's answer above rather
    # than restated as `forbidden_symbols_of(_spec)`: the resolver falls back to
    # the FACTORY's consumer gate when the spec supplies nothing, so a spec-only
    # restatement would misclassify every inherited gym gate as authored and
    # reject `-s problem.env_id=gym_*` over the tester tier (verify off + an
    # "authored" list it never wrote). Same argument as the
    # TASK_SPEC_KEYS check above: running the resolver is the only statement of
    # the rule that cannot disagree with the rule.
    _inherited = list((_supplied.get("verify") or {}).get("forbidden_symbols") or [])
    _authored = _in_force and _in_force != _inherited

    if _authored and not g("verify.enabled"):
        # A PIN NOTHING HONOURS -- the case the rule was written for. The config wrote a
        # list it cannot use. An INHERITED list here is different in kind: the method has
        # opted out of verification entirely (the tester tier does, for speed), and the
        # environment's gate simply does not apply. That is coherent, and reporting it as
        # an error trains people to ignore the one check in this function that is
        # dangerous rather than merely untidy.
        p.append("verify.forbidden_symbols is set but verify.enabled=false, so nothing "
                 "would ever check it")
    elif _in_force and g("verify.enabled") and \
            "forbidden_symbols" not in (g("verify.static_checks") or []):
        # INCOHERENT WHATEVER ITS ORIGIN, so this case ignores authorship. Verification IS
        # running and a non-empty gate IS in force, and the check that would read it is
        # not in the list -- so a candidate reaching for the metric it is scored on gets
        # through a gate the config believes it is running. Inherited or authored makes no
        # difference to that.
        p.append("verify.forbidden_symbols is in force but 'forbidden_symbols' is not in "
                 "verify.static_checks, so nothing would ever check it")

    # -- the fitness reduction must be one the environment actually applies ---
    # A §4 key an adapter silently ignores is a fabricated pin, and this is the
    # dangerous kind rather than the untidy kind: the run
    # completes, the dashboard fills, and the number reported is the OTHER reduction's.
    # Asked of the registered factory rather than a constructed env -- building a
    # Meta-World adapter to answer it would import mujoco and cost 1.2 s per config on a
    # path whose whole point is being cheap.
    # local imports: `registry` imports components, which import this module
    from .registry import RegistryError, get as _get_component, names as _component_names
    from .tasks import PER_STEP

    try:
        _env_factory = _get_component("env", str(g("problem.env_id")))
    except (RegistryError, KeyError):          # an unknown env is already reported above
        _env_factory = None

    # No `!= PER_STEP` shortcut. The shortcut would assume every env honours the
    # dense default, and an env that terminates at first success does not: there
    # per_step degenerates to reciprocal time-to-first-success and the adapter
    # declares ANY_STEP only -- under the shortcut, a config carrying the DEFAULT
    # reduction on such an env would validate cleanly and then run the other
    # quantity, the exact silent reinterpretation this check names.
    # (Consequence for the enum's meaning: `supported_reductions` is "the
    # values a config may carry for this env", so an adapter whose metric is no
    # reduction of anything -- the gym_* tier -- still lists PER_STEP, the
    # schema's inert default, and documents that it moves nothing.)
    reduction = g("evaluate.fitness.reduction")
    if reduction and _env_factory is not None:
        supported = getattr(_env_factory, "supported_reductions", (PER_STEP,))
        if reduction not in supported:
            p.append(
                f"evaluate.fitness.reduction={reduction!r} but environment "
                f"{g('problem.env_id')!r} only applies {list(supported)}; its "
                "task_metric would be computed the other way and nothing would say so")

    # -- a success-rate fitness needs a success bar to exist ----------------
    # Same shape as the reduction rule, and the same fabricated-pin argument: the gymnasium
    # tier's `success()` is False by construction (no expert anchor, no bar -- the
    # factory says so via `defines_success = False`), so `fitness.source:
    # success_rate` there assigns every candidate fitness 0.0. That is a REAL value,
    # distinct from the failure sentinel, so the pool is an all-tie -- and under
    # LIMEN's map-elites insert the strict `>` means the first occupant of every
    # archive cell keeps it forever, with nothing in the artifact saying the metric
    # itself was constant. Refused here, where the config is read, rather than
    # discovered in a finished sweep.
    if (_env_factory is not None
            and getattr(_env_factory, "defines_success", True) is False
            and g("evaluate.fitness.source") == "success_rate"):
        p.append(
            f"evaluate.fitness.source='success_rate' but environment "
            f"{g('problem.env_id')!r} defines no success bar (its spec's expert "
            "anchor is null, so success() is False by construction): every "
            "candidate would score exactly 0.0 and selection would be an all-tie")

    # -- the native-signal rule: a native source needs a native signal --------
    # `evaluate.fitness.source` in {native, native_success, native_reward} reads
    # the env's OWN signal and must NEVER fall back on the BIRD `task_metric`
    # (`custom_metric`). So a native source is refused here when the task ships
    # nothing it can read: `native_success` needs a shipped binary success
    # (`discrete_success.kind: discrete`/`native_authored` -- the kinds that pin a
    # VERIFIED reimplementation of the vendor's own success); `native_reward`
    # needs a shipped reward (`reward.human.kind` != none); bare `native` needs at
    # least one of the two. The bare-`native` refuse set is exactly the
    # five gymnasium tasks with `reward.human.kind: none` and no success. Refused
    # here rather than discovered when a finished sweep has scored a supervised
    # method on a number we wrote. Keyed on `_spec` (per task), not the factory
    # `defines_success` (per family), so the split is the right 5, not all 10 gym.
    _fsrc = g("evaluate.fitness.source")
    if _fsrc in ("native", "native_success", "native_reward") and _spec is not None:
        from .native_signal import has_native_success as _hns, has_native_reward as _hnr
        _ds = (_spec.discrete_success or {}).get("kind")
        _rw = ((_spec.reward or {}).get("human") or {}).get("kind")
        _ns, _nr = _hns(_ds), _hnr(_rw)
        if _fsrc == "native_success" and not _ns:
            p.append(
                "evaluate.fitness.source='native_success' but the task for "
                f"problem.env_id={g('problem.env_id')!r} ships no native success "
                f"(discrete_success.kind={_ds!r}, not 'discrete'/'native_authored'); "
                "the native-signal rule refuses to score it on the BIRD task_metric")
        elif _fsrc == "native_reward" and not _nr:
            p.append(
                "evaluate.fitness.source='native_reward' but the task for "
                f"problem.env_id={g('problem.env_id')!r} ships no reference reward "
                f"(reward.human.kind={_rw!r}): there is no native reward to score against")
        elif _fsrc == "native" and not (_ns or _nr):
            p.append(
                "evaluate.fitness.source='native' but the task for "
                f"problem.env_id={g('problem.env_id')!r} ships neither a native success "
                f"(discrete_success.kind={_ds!r}) nor a reference reward "
                f"(reward.human.kind={_rw!r}), so there is no native signal to select on; "
                "it must not fall back on the BIRD task_metric (one of the five gymnasium "
                "unsupervised-only tasks)")

    # -- confirm-before-execute (L2R) -------------------------------------
    if g("verify.human_confirm_before_execute") and not g("verify.enabled"):
        p.append("verify.human_confirm_before_execute=true but verify.enabled=false, "
                 "so no generated code is executed inside the gate")

    # -- archive sizing (LIMEN) -------------------------------------------
    if (g("update.archive.population_size") or 1) > 1 and \
            g("update.topology") == "single_parent_hillclimb":
        p.append("update.archive.population_size>1 is meaningless under "
                 "update.topology=single_parent_hillclimb")
    # Same fabricated-pin argument as `crossover_rate` above, and the same shape as the
    # migration rule below: the TOPOLOGY declares that it consults the gate
    # (`honours_admission = True` on the registered function), because a rule that
    # names the reader by string is a list that goes stale the next time one is
    # written. Absent means False, with no grandfather clause -- `truncate` is what a
    # topology without the marker does, and the other gates are silently inert under
    # any such topology.
    _adm = g("update.archive.admission")
    if _adm and _adm != "truncate":
        _adm_topo = g("update.topology")
        try:
            _adm_fn = _get_component("topology", str(_adm_topo))
        except (RegistryError, KeyError):      # unknown topology, reported elsewhere
            _adm_fn = None
        if not getattr(_adm_fn, "honours_admission", False):
            _readers = sorted(n for n in _component_names("topology")
                              if getattr(_get_component("topology", n),
                                         "honours_admission", False))
            p.append(f"update.archive.admission={_adm!r} but update.topology is "
                     f"{_adm_topo!r}, which does not consult it; the gate would never "
                     "refuse anything. Topologies that read it: "
                     f"{', '.join(_readers) or 'none registered'}")
    # Migration needs DEMES, not the `island` topology specifically. LIMEN is the
    # case that proves it: its topology is `archive_map_elites` and it migrates at
    # 0.1 between three islands *inside* the archive. `_maybe_migrate` is reached
    # from `insert_into_archive` as well as from `topo_island`, and its own guard is
    # `n_islands < 2 or interval <= 0` -- nothing about the topology. Requiring
    # `topology == "island"` would make MAP-Elites + islands + migration mutually
    # exclusive and leave LIMEN unable to state its own published migration rate.
    # Nor is the check a literal tuple. A rule that names its members has to be
    # edited by every method that adds a deme-holding topology (REvolve's
    # `island_lineage`, say), and the edit is in a file no component author is
    # looking at -- so the topology DECLARES the capability instead, in the same
    # shape the env factory declares `supported_reductions` above: `has_demes = True`
    # on the registered function. The getattr DEFAULT covers the pair that does not
    # carry the marker, and each drops out of `_predates_marker` once its own
    # function carries the attribute; a topology with no demes still needs to say
    # nothing at all.
    if (g("update.archive.migration_rate") or 0) > 0:
        _predates_marker = ("island", "archive_map_elites")
        _topo = g("update.topology")
        try:
            _topo_fn = _get_component("topology", str(_topo))
        except (RegistryError, KeyError):      # unknown topology, already reported above
            _topo_fn = None
        deme_topology = bool(getattr(_topo_fn, "has_demes", _topo in _predates_marker))
        if not deme_topology:
            # The list is RESOLVED rather than written out. An enumeration in an error
            # message goes stale exactly as quietly as the one in the condition did --
            # and this message is what a reader consults to learn what to put instead,
            # so a stale one sends them to a value that no longer has the capability.
            _with_demes = sorted(
                n for n in _component_names("topology")
                if getattr(_get_component("topology", n), "has_demes",
                           n in _predates_marker))
            p.append("update.archive.migration_rate>0 requires an update.topology that "
                     f"maintains demes ({', '.join(_with_demes)}); {_topo!r} has none to "
                     "migrate between")
        elif (g("update.archive.n_islands") or 1) < 2:
            p.append("update.archive.migration_rate>0 needs update.archive.n_islands>1; "
                     "with one island there is nowhere to migrate to")
        elif (g("update.archive.migration_interval") or 0) <= 0:
            p.append("update.archive.migration_rate>0 needs a positive "
                     "update.archive.migration_interval or migration never fires")

    # -- archive parent-sampling branch names ------------------------------
    # Validated HERE and not only at runtime. `update.py::_draw_branch` raises on an
    # unknown key, but only in stage 1 of the first iteration that finds a non-empty
    # archive -- so a typo costs a generate call, a verify pass and a full training
    # round before it surfaces, which can be hours. The table is the
    # authority (plus its spelling aliases); this rule just reads it at load.
    _ps = g("update.archive.parent_sampling") or {}
    if isinstance(_ps, dict) and _ps:
        from .components.update import _BRANCH_ALIASES, _PARENT_BRANCHES
        _known = set(_PARENT_BRANCHES) | set(_BRANCH_ALIASES)
        for _k in sorted(_ps):
            if _k not in _known:
                p.append(f"update.archive.parent_sampling has no branch {_k!r}; "
                         f"known: {', '.join(sorted(_known))}")
        if all(float(v or 0) <= 0 for v in _ps.values()):
            p.append("update.archive.parent_sampling weights are all zero or negative; "
                     "no branch could ever be drawn")

    # -- a self-filing topology must not ALSO run an inserting winner action --
    # `island_lineage` (and any future topology that declares the marker) files
    # every survivor into a deme itself; a `winner_action` that inserts then files
    # the WINNER a second time -- a second cell under descriptor coords rather
    # than its own deme, a second `archive_inserts` tick that desynchronises the
    # migration cadence, and a second `_maybe_migrate` at the floor capacity
    # instead of the deme's. On `revolve` that shows as archive occupancy
    # 17 / 39 / 51 over three iterations against 16 admissions each, with the
    # winner present twice in two different islands. Nothing fails
    # -- it just reads as a richer archive than the run produced, which is why it
    # is refused at load. RF-Agent's `search_tree` leaves the default for the same
    # reason; LIMEN's `archive_map_elites` does NOT self-file, so `insert_archive`
    # is correct there and the marker is what tells the two apart.
    _wact = g("update.winner.action") or "become_parent"
    if _wact in ("insert_archive", "both"):
        try:
            _wtopo_fn = _get_component("topology", str(g("update.topology")))
        except (RegistryError, KeyError):
            _wtopo_fn = None
        if getattr(_wtopo_fn, "files_survivors", False):
            p.append(f"update.winner.action={_wact!r} under "
                     f"update.topology={g('update.topology')!r}, which files every "
                     "survivor into a deme itself; the winner would be archived "
                     "twice. Use become_parent")

    # -- the preference channel needs a pool and a judge (REvolve) --------
    # `cumulative` refits the aggregator over every individual still held in state
    # rather than over this round's, which needs the store to exist AND to survive
    # the iteration boundary. Either half missing leaves the key selectable and an
    # identity -- the `loop.resume_from` shape: a declared capability with no reader.
    if g("evaluate.preferences.enabled") and \
            g("evaluate.preferences.scope") == "cumulative":
        if not g("evaluate.preferences.store_dataset"):
            p.append("evaluate.preferences.scope=cumulative refits over "
                     "state.preferences, which only grows when "
                     "evaluate.preferences.store_dataset is true")
        if "preference_dataset" not in (g("loop.carry") or []):
            p.append("evaluate.preferences.scope=cumulative needs 'preference_dataset' "
                     "in loop.carry, or the store is dropped at every iteration boundary "
                     "and the pool is this round's again")

    # NOT a crash, which is the whole reason this is refused at load. Chain read in
    # this tree: `build_human_oracle` returns `_NullHumanOracle` (phases.py:964-966)
    # -> `.compare` -> `_refuse` raises RuntimeError (phases.py:793-796) ->
    # `comparator_human` tries "compare" first (preferences.py:1205) and catches it in
    # `except Exception` (preferences.py:1216, carrying `# pragma: no cover`, which is
    # why no test has ever seen this path) -> `answer = None` -> `_decide_offline`
    # (preferences.py:1227, :831) ranks the pair by (success, mean_per_step_return,
    # cand_id). So the run COMPLETES with a full preference dataset no person labelled.
    # Three things stop that being visible: the log line is DEBUG against a default
    # `output.log_level: INFO`; `ctx.budget.record_human()` already fired at
    # preferences.py:1194, before the oracle was consulted, so the artifact reports
    # human labour nobody supplied; and the only surviving trace is `judge: "none"` in
    # the preference log.
    if g("evaluate.preferences.enabled") and \
            g("evaluate.preferences.comparator") == "human" and \
            (g("evaluate.human.mode") or "none") == "none":
        p.append("evaluate.preferences.comparator=human with evaluate.human.mode=none: "
                 "the null oracle raises on every query, comparator_human swallows it at "
                 "log.debug and falls through to _decide_offline, and the round is still "
                 "billed to budget.human_queries. The run would complete with a full "
                 "preference dataset no human labelled")

    # -- search tree (RF-Agent) -------------------------------------------
    # Three values make one method point and none of them works alone: `uct_leaf`
    # reads a tree only `search_tree` writes, `tree_actions` expands the leaf only
    # `uct_leaf` returns, and a tree not named in `loop.carry` is dropped at the
    # iteration boundary -- after which every round re-initialises under a
    # resolved config that still reads as a tree search. Each rule refuses the
    # half-configured point instead of letting it run as a silent zero-shot.
    # `puct_leaf` is the same descent under a different child score
    # (tree.py::parent_puct_leaf) and stands wherever `uct_leaf` does here.
    _carry = g("loop.carry") or []
    _TREE_SOURCES = ("uct_leaf", "puct_leaf")
    _src_name = g("generate.parent_source")
    _tree_src = _src_name in _TREE_SOURCES
    _tree_topo = g("update.topology") == "search_tree"
    _tree_acts = g("generate.sampling_mode") == "tree_actions"
    if _tree_src:
        if not _tree_topo:
            p.append(f"generate.parent_source={_src_name} requires update.topology=search_tree "
                     "(nothing else writes the tree it descends)")
        if not _tree_acts:
            p.append(f"generate.parent_source={_src_name} requires "
                     "generate.sampling_mode=tree_actions (the leaf it returns is expanded "
                     "by the tree actions and read by nothing else)")
        if "search_tree" not in _carry:
            p.append(f"generate.parent_source={_src_name} requires 'search_tree' in loop.carry "
                     "(a tree dropped at the boundary re-initialises every round)")
        if g("generate.tree.horizon_trainings") is None:
            p.append(f"generate.parent_source={_src_name} requires "
                     "generate.tree.horizon_trainings (Alg. 1's N, the denominator of the "
                     "lambda schedule and of the backup decay)")
    if _tree_topo:
        if not _tree_src:
            p.append("update.topology=search_tree requires generate.parent_source in {uct_leaf, puct_leaf} "
                     "(a tree nothing descends is written and never read)")
        if "search_tree" not in _carry:
            p.append("update.topology=search_tree requires 'search_tree' in loop.carry "
                     "(the tree is the population; dropped, the search restarts every round)")
        # A failed round's reports stay in `state.all_reports` and `tree._catch_up`
        # inserts them under their iteration index before the next selection. A
        # RETRIED round then produces a second batch under the same index, so the
        # tree would hold two rounds' nodes as one -- and the release has no
        # retry: a failed child is backed up at 0 and the search moves on.
        if g("loop.on_total_failure") == "retry_iteration":
            p.append("update.topology=search_tree cannot run under "
                     "loop.on_total_failure=retry_iteration (the failed round's nodes "
                     "join the tree under their iteration index; a retry would file a "
                     "second round under the same index)")
        # Eq. 2's lambda is linear in t/horizon with no clamp (the release's
        # `c_param`, rfagent_algo.py:778). A schedule that outlives the horizon
        # by more than lambda_final/(lambda0-lambda_final) drives it NEGATIVE,
        # at which point UCT prefers the most-visited, lowest-self-verify child
        # and nothing in the run says so. Refuse it at load: the last selection
        # happens with t = sum of every round but the last already inserted.
        _l0 = g("generate.tree.uct_lambda0")
        _lf = g("generate.tree.uct_lambda_final")
        _hz = g("generate.tree.horizon_trainings")
        _n_it = g("loop.n_iterations")
        if isinstance(_l0, (int, float)) and isinstance(_lf, (int, float)):
            if _l0 < 0 or _lf < 0:
                p.append(f"generate.tree.uct_lambda0={_l0} and uct_lambda_final={_lf} must "
                         "both be >= 0")
            elif isinstance(_hz, int) and isinstance(_n_it, int) and _n_it >= 2:
                _vals = list(sched_vals) or [n_cand] * _n_it
                _t_last = sum(int(v) for v in _vals[:_n_it - 1])
                _lam_last = (_l0 - _lf) * (1.0 - _t_last / float(_hz)) + _lf
                if _lam_last < 0:
                    p.append(f"generate.tree.horizon_trainings={_hz} is outlived by the "
                             f"schedule: the last selection happens at t={_t_last} where "
                             f"lambda = {_lam_last:.4f} < 0 and UCT would prefer the "
                             "most-visited, lowest-self-verify child; raise the horizon or "
                             "shorten loop.n_iterations")
        for _key, _floor in (("generate.tree.elite_size", 1), ("generate.tree.path_window", 1),
                             ("generate.tree.max_depth", 1)):
            _v = g(_key)
            if _v is not None and (not isinstance(_v, int) or isinstance(_v, bool) or _v < _floor):
                p.append(f"{_key}={_v!r} must be an int >= {_floor} (the code would run it as "
                         f"{_floor} while the config says otherwise -- a fabricated pin)")

    _acts = g("generate.actions") or {}
    _ACTIONS = ("mutation_mechanism", "mutation_param", "crossover_elite",
                "path_reasoning", "different_thought")
    if _tree_acts:
        if not _acts:
            p.append("generate.sampling_mode=tree_actions needs a non-empty "
                     "generate.actions (the per-expansion count of each of the five actions)")
        else:
            _bad = sorted(k for k in _acts if k not in _ACTIONS)
            if _bad:
                p.append(f"generate.actions names unknown action(s) {_bad}; the five are "
                         f"{list(_ACTIONS)}")
            if any((not isinstance(v, int)) or isinstance(v, bool) or v < 0
                   for v in _acts.values()):
                p.append(f"generate.actions={_acts} must be all ints >= 0")
            else:
                # The counts ARE the round's candidate pool, so the two declarations
                # of its size have to agree -- with the one exception of round 0,
                # which has no tree yet and samples the zero-shot init nodes
                # (the release's N_I=6 against 8 children per expansion).
                _total = sum(_acts.values())
                if isinstance(n_cand, int) and _total != n_cand:
                    p.append(f"generate.actions sums to {_total} but "
                             f"generate.n_candidates={n_cand}; under tree_actions the "
                             "action counts are the round's whole candidate pool")
                _off = [(i, v) for i, v in enumerate(sched_vals) if i >= 1 and v != _total]
                if _off:
                    p.append(f"generate.candidate_schedule_values disagrees with the "
                             f"generate.actions sum ({_total}) at (iteration, value) "
                             f"{_off}; only iteration 0, the zero-shot init round, may "
                             "differ")
        if (g("generate.n_parents") or 1) != 1:
            p.append("generate.sampling_mode=tree_actions requires generate.n_parents=1 "
                     "(the group each prompt shows is chosen by the action, not by "
                     "n_parents)")
    elif _acts:
        p.append("generate.actions is set but "
                 f"generate.sampling_mode={g('generate.sampling_mode')!r} ignores it")
    if not _tree_src:
        # A tree knob with no tree is a declared pin nothing reads.
        _tree_defaults = {"generate.tree.horizon_trainings": None,
                          "generate.tree.uct_lambda0": 0.0,
                          "generate.tree.uct_lambda_final": 0.0,
                          "generate.tree.max_depth": None,
                          "generate.tree.group_size": [2, 2],
                          "generate.tree.elite_size": 1,
                          "generate.tree.path_window": 1,
                          "update.tree.best_child_weight": 1.0,
                          "update.tree.mean_weight": 0.0}
        _moved = sorted(k for k, d in _tree_defaults.items() if g(k) != d)
        if _moved:
            p.append(f"{_moved} are set but generate.parent_source="
                     f"{g('generate.parent_source')!r} runs no tree, so nothing reads them")

    if g("generate.output.design_thought") == "inline_brace" and \
            g("generate.output.two_stage_nl_then_code"):
        p.append("generate.output.design_thought=inline_brace cannot be combined with "
                 "generate.output.two_stage_nl_then_code=true (both write "
                 "Candidate.nl_spec; one would silently overwrite the other)")

    if g("update.prompt.assistant_content") == "nl_spec" and not (
            g("generate.output.two_stage_nl_then_code")
            or g("generate.output.design_thought") == "inline_brace"):
        p.append("update.prompt.assistant_content=nl_spec needs a writer of "
                 "Candidate.nl_spec: set generate.output.two_stage_nl_then_code=true "
                 "(GT, App. B) or generate.output.design_thought=inline_brace "
                 "(RF-Agent); otherwise every carried turn would silently fall back "
                 "to the code")

    # Eq. 3's old-Q weight is 1 - best_child_weight - mean_weight*decay; the pair
    # is bounded so that weight cannot go negative at decay 1.
    _bcw, _mw = g("update.tree.best_child_weight"), g("update.tree.mean_weight")
    if isinstance(_bcw, (int, float)) and isinstance(_mw, (int, float)):
        if _bcw < 0 or _mw < 0:
            p.append(f"update.tree.best_child_weight={_bcw} and mean_weight={_mw} must "
                     "both be >= 0")
        elif _bcw + _mw > 1:
            p.append(f"update.tree.best_child_weight + mean_weight = {_bcw + _mw} > 1; "
                     "the old-Q weight 1 - best_child_weight - mean_weight*decay would "
                     "go negative")

    _rng = g("evaluate.self_verify.range")
    if _rng is not None and not (
            isinstance(_rng, list) and len(_rng) == 2
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in _rng)
            and _rng[0] < _rng[1]):
        p.append(f"evaluate.self_verify.range={_rng!r} must be [lo, hi], two numbers "
                 "with lo < hi")
    _gs = g("generate.tree.group_size")
    if _gs is not None and not (
            isinstance(_gs, list) and len(_gs) == 2
            and all(isinstance(x, int) and not isinstance(x, bool) for x in _gs)
            and 2 <= _gs[0] <= _gs[1]):
        p.append(f"generate.tree.group_size={_gs!r} must be [lo, hi], two ints with "
                 "2 <= lo <= hi (the parent is always one of the nodes shown)")

    # -- recording --------------------------------------------------------
    if "videos" in (g("evaluate.artifacts") or []) and not g("output.video.enabled"):
        p.append("evaluate.artifacts includes 'videos' but output.video.enabled=false; "
                 "the method needs rollout frames the run would not record")
    # You cannot upload frames that were never rendered. `best` < `best_and_worst`
    # < `all` is a containment order, so the upload mode must sit at or below the
    # record mode; `record: best` with `video_record: all` would upload one clip
    # and label it the whole population.
    _BREADTH = {"none": 0, "best": 1, "best_and_worst": 2, "all": 3}
    _rec, _up = g("output.video.record") or "none", g("output.wandb.video_record") or "none"
    if _BREADTH.get(_up, 0) > _BREADTH.get(_rec, 0):
        p.append(f"output.wandb.video_record={_up!r} is broader than "
                 f"output.video.record={_rec!r}; nothing renders the extra rollouts")
    if _up != "none" and not g("output.video.enabled"):
        p.append(f"output.wandb.video_record={_up!r} but output.video.enabled=false, "
                 "so there is nothing to upload")

    if g("evaluate.preferences.enabled") and \
            g("evaluate.preferences.comparator") in ("vlm", "llm_on_vlm_captions") and \
            not g("output.video.enabled"):
        p.append(f"evaluate.preferences.comparator={g('evaluate.preferences.comparator')!r} "
                 "watches rollouts, so output.video.enabled must be true")
    _nv = g("output.video.n_views")
    if _nv is not None and (not isinstance(_nv, int) or isinstance(_nv, bool) or _nv < 1):
        p.append(f"output.video.n_views={_nv!r}; it counts viewpoints per frame and must "
                 "be an integer >= 1 (1 = the primary camera only)")

    return p
