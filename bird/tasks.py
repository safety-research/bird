"""The base task definition: `tasks/<id>/shared_spec.yaml`, loaded and indexed.

A task is a file (the catalogue's conventions are recorded in `tasks/SOURCE.md`), and
this module is the only thing that reads it.

WHAT THIS MODULE MAY IMPORT, and why it matters. `yaml`, `hashlib`, `json`, `pathlib` and
the standard library -- nothing else. Not `numpy`, not `bird.registry`, not `bird.schema`:

  * `bird/config.py` imports this for its coherence rules, and `config -> registry ->
    components -> config` is a real cycle;
  * every CLI path that does not execute a search (`--validate-all`, `--list-configs`,
    `--diff`, `--print-config`, `--dry-run`) and the whole test suite must run on
    pyyaml + numpy alone, so `jsonschema` cannot appear here. Full 2020-12 validation
    lives in `tests/test_task_specs.py` behind an `importorskip`; what runs HERE is the
    structural subset the algorithm actually depends on.

`tasks/` is a hard repo artifact, like `configs/_default.yaml`. `bird/envs/metaworld.py`
reads the index at import time, so a missing or corrupt `tasks/` fails `registry.load_all()`
and therefore `--validate-all` and the suite. That is the price of a single source of truth;
what this module owes in exchange is that the failure NAMES the file and the field, rather
than surfacing a hundred lines away as ten "unknown key `problem.env_id`" errors.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import yaml

from contextlib import contextmanager

from . import paths

try:  # pragma: no cover - availability differs per install
    from yaml import CSafeLoader as _Loader
except ImportError:  # pragma: no cover
    from yaml import SafeLoader as _Loader  # type: ignore[assignment]

#: `tasks/`, resolved on USE rather than at import.
#:
#: NOT A MODULE CONSTANT, and not a constant with a fallback either. A
#: fallback such as `paths.data_dir("tasks", required=False) or
#: parent.parent / "tasks"` yields `site-packages/tasks` under a non-editable
#: install -- the exact plausible wrong path `bird.paths` exists to remove,
#: reintroduced by the fallback meant to keep imports working.
#:
#: A constant cannot be both: resolving strictly at import fails the import
#: for every caller including those that never touch a task spec, and
#: resolving leniently hands out a wrong path. Resolving on USE is the only
#: shape that is neither, so `TASKS_ROOT` is served by the module's
#: `__getattr__` (PEP 562) and every internal reader calls `_tasks_root()`.
#: `from bird.tasks import TASKS_ROOT` works and raises where the value is
#: wanted rather than where the module is loaded.
#: A redirect set by `using_root`, consulted before anything else.
#:
#: NOT a `TASKS_ROOT` entry in the module dict, and that distinction is the
#: whole design. `monkeypatch.setattr(mod, "TASKS_ROOT", x)`
#: reads the old value in its SAVE step -- which runs `__getattr__` and
#: raises where the root does not resolve -- and `undo()` does not delete the
#: name, it writes the computed value back as a real dict entry. So the first
#: caller to patch the public name FREEZES it into a snapshot for the rest of
#: the process, taken once, early, possibly before an override was set. In a
#: module whose rule is "never return a plausible wrong path", honouring that
#: snapshot would be the defect wearing a fix's clothes.
_ROOT_OVERRIDE: Optional[Path] = None


@contextmanager
def using_root(path):
    """Read the catalogue from `path` for the duration of the block.

    PROCESS-GLOBAL, NOT LEXICAL, and `with` reads as though it were the
    other way round. Inside the block EVERY reader in this process sees the
    temporary catalogue, not only the code written inside it. There is no
    thread hazard today because candidate parallelism forks processes rather
    than threading, but a caller who assumes locality is assuming something
    this does not provide.

    Nesting works: the previous override is restored, not cleared, so
    `using_root(a)` inside `using_root(b)` drops back to `b` and not to the
    resolver. Clearing on exit would break that silently.

    On exit the override is restored FIRST and the index is then
    INVALIDATED rather than rebuilt. A rebuild that raised would leave
    `_INDEX` populated from the temporary root behind a correct-looking
    override -- the worst of the three outcomes, and invisible in a green
    run. Invalidation cannot fail; the next reader pays for a correct build.
    """
    global _ROOT_OVERRIDE, _INDEX
    previous = _ROOT_OVERRIDE
    _ROOT_OVERRIDE = Path(path)
    _INDEX = None
    try:
        yield _ROOT_OVERRIDE
    finally:
        _ROOT_OVERRIDE = previous
        _INDEX = None


def _tasks_root() -> Path:
    if _ROOT_OVERRIDE is not None:
        return _ROOT_OVERRIDE
    root = paths.data_dir("tasks", required=False)
    if root is None:
        raise TaskSpecError(
            f"cannot find the `tasks/` directory. The task catalogue is a "
            f"hard repo artifact, like configs/_default.yaml -- without it no "
            f"environment has a definition. If you are running a NON-EDITABLE "
            f"install, `tasks/` is a repo data directory and is "
            f"deliberately not shipped in the wheel: set "
            f"{paths.DATA_ROOT_ENV} to a directory containing it, or install "
            f"with `pip install -e`.")
    return root


def __getattr__(name: str):
    if name == "TASKS_ROOT":
        return _tasks_root()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

#: THE LAZY NAME IS NOT VISIBLE TO INTROSPECTION WITHOUT THIS. PEP 562's
#: `__getattr__` is consulted only after the module dict misses, so
#: `dir(module)` -- which reads the dict -- omits `TASKS_ROOT` entirely, and so do
#: tab-completion, `inspect.getmembers`, and any tool that enumerates a
#: module's surface to decide what it exports. The name works and looks
#: absent, which is the same failure shape `bird.paths` exists to remove,
#: one level up.
#:
#: AND A WARNING FOR CALLERS: `hasattr(module, "TASKS_ROOT")` is NOT a safe probe
#: here. `hasattr` swallows only `AttributeError`, and the resolver raises
#: `TaskSpecError` when the directory cannot be found -- so on the machine this
#: matters on, `hasattr` PROPAGATES rather than returning False. That is the
#: correct behaviour (absent and unresolvable are different questions) but it
#: is the opposite of what `hasattr` is usually reached for. To ask "is there
#: a tasks directory?" without raising, call
#: `paths.data_dir("tasks", required=False) is not None`.
def __dir__():
    return sorted(list(globals()) + ["TASKS_ROOT"])


SPEC_FILE = "shared_spec.yaml"
LEDGER_FILE = "_no_adapter.json"

#: The schema's top-level `required`. Restated here because this check must work on a
#: machine with no jsonschema, and because a missing group should name itself.
REQUIRED_GROUPS: Tuple[str, ...] = (
    "schema_version", "id", "env", "description", "state_surface", "discrete_success",
    "continuous_success", "anchors", "budget", "reward", "judge", "provenance",
)

SCHEMA_VERSION = 1

#: BIRD env id <- spec, one rule per adapter family.
#:
#: NOT a table of ids in the spec file. `bird/envs/metaworld.py` registers `mt10_`/`mt50_`
#: + the MT50_V3 key VERBATIM precisely so that an id which resolves is an id the benchmark
#: accepts; a second spelling carried in the data would be the drift that naming scheme
#: exists to prevent. A derivation cannot drift -- it can only be wrong everywhere at
#: once, which a test catches.
#:
#: Each rule takes the SPEC, not the benchmark key, because the suites disagree about
#: which field identifies a task: Meta-World's `env.env_id` is unique across the
#: catalogue, the mujoco specs' is not (`HalfCheetah-v5` backs three -- a
#: task is an (environment, objective) pair) and only the spec id distinguishes the
#: objectives.
#: The MT10 protocol's ten task keys (`metaworld.env_dict.MT10_V3`), restated here
#: because the id rule below needs them and this module imports no simulator. The
#: prefix a Meta-World env id carries names the SMALLEST published protocol the task
#: belongs to: the ten are `mt10_<key>`, the other forty of MT50 are `mt50_<key>`.
#: MT10 is a subset of MT50, so one task never gets two ids. `bird/envs/metaworld.py`
#: imports this tuple as `_EXPECTED_MT10` and fails at import if the catalogue's
#: Meta-World specs are not exactly the fifty.
METAWORLD_MT10: Tuple[str, ...] = (
    "reach-v3", "push-v3", "pick-place-v3", "door-open-v3", "drawer-open-v3",
    "drawer-close-v3", "button-press-topdown-v3", "peg-insert-side-v3",
    "window-open-v3", "window-close-v3",
)


def _metaworld_prefix(env_id: str) -> str:
    return "mt10_" if env_id in METAWORLD_MT10 else "mt50_"


_BIRD_ID_RULES: Dict[str, Callable[["TaskSpec"], str]] = {
    # `mt10_` for the ten MT10_V3 keys, `mt50_` for the rest of MT50_V3 -- see
    # METAWORLD_MT10. The suffix is the benchmark key VERBATIM in both cases.
    "metaworld": lambda spec: _metaworld_prefix(spec.env_id) + spec.env_id,
    # An environment this repo implements itself: the spec's `env_id` IS the registry
    # name, because there is no upstream benchmark whose key it has to preserve.
    "bird": lambda spec: spec.env_id,
    # The suffix is the spec id (the objective's name) VERBATIM, for the same
    # cannot-drift reason as `mt10_`; the benchmark key cannot be the suffix because it
    # is shared. `bird/envs/gym_mujoco.py::_env_id` mirrors this rule and the coverage
    # partition test is what holds the two together.
    "mujoco": lambda spec: "gym_" + spec.id,
    # HumanoidBench: the gym id with `-v0` dropped and `-` turned into `_`. NO PREFIX is
    # added, and that is the one rule here that differs in shape from the others --
    # because the benchmark's own id already carries the robot (`h1hand-powerlift-v0`,
    # `h1strong-highbar_hard-v0`), and the robot is what a prefix would have named. It
    # is still a derivation rather than a second spelling (`h1hand-walk-v0` ->
    # `h1hand_walk`). `bird/envs/suites.py` therefore matches on the ROBOT prefixes
    # (`h1_`, `h1hand_`, `h1strong_`, ...) rather than on one family prefix.
    # A guarded slice: `-v0` is stripped only when present, so an id without the
    # suffix passes through unchanged.
    "humanoid_bench": lambda spec: (
        spec.env_id[:-len("-v0")] if spec.env_id.endswith("-v0") else spec.env_id
    ).replace("-", "_"),
    # Assistax (assistive-autonomy/assistax), its MJCF scenes driven through plain
    # `mujoco`. The suffix is upstream's own task key (`scratchitch`, `bedbathing`, ...,
    # the strings `assistax.registered_envs` holds), which is also the row key in
    # `bird/envs/assistax.py::_TASKS`; the prefix names the BENCHMARK, as `mt10_` does.
    "assistax": lambda spec: "assistax_" + spec.env_id,
    # UPSTREAM Assistax: upstream's OWN jax env classes, stepped as upstream steps them
    # (`bird/envs/upstream_assistax.py`), at the assets' pinned commit
    # (`scripts/setup_jax.sh`). A SECOND tier over the same upstream tasks, and the two
    # are separated here because nothing else separates them: `assistax_x` is upstream's
    # MJCF driven by this repo through plain `mujoco`, and `upstream_assistax_x` is
    # upstream's code running upstream's way. Both answer to the same `env.env_id` --
    # upstream's key, `scratchitch` -- so a reader pairs them by stripping one prefix,
    # and only the prefix keeps a table from averaging two different solvers.
    "assistax_upstream": lambda spec: "upstream_assistax_" + spec.env_id,
}


#: The two reductions of one per-step ground-truth success check, and the keys
#: `anchors.by_reduction` is keyed by. `task_metric` is the fraction of an episode's
#: arriving states in the goal region; `success()` is the same check reduced to "on at
#: least one step", because a `success_threshold` of `1/(2*horizon)` makes the second a
#: threshold on the first -- so they cannot disagree about WHETHER a task was done. They
#: are still different NUMBERS: a uniform-random policy scores 0.15 per-step and 0.25 any-step
#: on drawer-close (n=100, tasks/_anchors_2026-08-25_n100.json). An anchor that does not say
#: which one it measured cannot normalise
#: anything safely, which is why the core schema's single pair grew a `reduction` label.
PER_STEP = "per_step_fraction"
ANY_STEP = "any_step_episode_fraction"
REDUCTIONS = (PER_STEP, ANY_STEP)

#: `env.reset`'s closed role vocabulary -- the schema's `$defs.reset_role` enum, restated
#: so a consumer can say "the seed changes: agent pose, goal" by reading `role` off each
#: draw rather than parsing prose. `tests/test_task_specs.py` pins this tuple EQUAL to the
#: schema's enum, so adding a role is a two-file edit that fails loudly when it is one.
RESET_ROLES: Tuple[str, ...] = (
    "agent_pose", "agent_velocity", "object_pose", "object_velocity", "goal",
    "command", "mode", "scene_selection",
)

#: role -> a short label for a human reader of the spec. Keys are exactly
#: `RESET_ROLES` (a test), so a role the schema admits always has a readable
#: name.
RESET_ROLE_LABELS: Dict[str, str] = {
    "agent_pose": "agent start pose",
    "agent_velocity": "agent start velocity",
    "object_pose": "object placement",
    "object_velocity": "object velocity",
    "goal": "goal position",
    "command": "commanded reference",
    "mode": "start mode",
    "scene_selection": "which objects take part",
}


class TaskSpecError(ValueError):
    """A task spec is missing, unparseable, or fails a rule the algorithm relies on."""


@dataclass(frozen=True)
class TaskSpec:
    """One parsed `shared_spec.yaml`.

    `id` is the ONLY unique key. `env.env_id` is not: `HalfCheetah-v5` backs three specs
    and `Reacher-v5`, `Swimmer-v5` and `Hopper-v5` back two each, because a task is an
    (environment, objective) pair and the catalogue carries several objectives per
    environment. Anything that looks a task up by `env_id` must refuse an ambiguous
    match rather than pick one.
    """

    id: str
    raw: Mapping[str, Any]
    path: Path
    sha256: str

    # -- group accessors ---------------------------------------------------
    # Each names the file on a missing group, so a half-written spec fails at load with
    # a path rather than KeyError'ing four stages deep with a bare string.

    def group(self, name: str) -> Any:
        try:
            return self.raw[name]
        except KeyError:  # pragma: no cover - `index` rejects these first
            raise TaskSpecError(f"{self.path}: no `{name}` group") from None

    @property
    def env(self) -> Mapping[str, Any]:
        return self.group("env")

    @property
    def description(self) -> Mapping[str, Any]:
        return self.group("description")

    @property
    def state_surface(self) -> Mapping[str, Any]:
        return self.group("state_surface")

    @property
    def discrete_success(self) -> Mapping[str, Any]:
        return self.group("discrete_success")

    @property
    def continuous_success(self) -> Mapping[str, Any]:
        return self.group("continuous_success")

    @property
    def anchors(self) -> Mapping[str, Any]:
        return self.group("anchors")

    @property
    def budget(self) -> Mapping[str, Any]:
        return self.group("budget")

    @property
    def reward(self) -> Mapping[str, Any]:
        return self.group("reward")

    @property
    def judge(self) -> Mapping[str, Any]:
        return self.group("judge")

    @property
    def provenance(self) -> Mapping[str, Any]:
        return self.group("provenance")

    @property
    def domain_randomization(self) -> Optional[Mapping[str, Any]]:
        return self.raw.get("domain_randomization")

    @property
    def symbol_mapping(self) -> Optional[Mapping[str, str]]:
        return self.raw.get("symbol_mapping")

    @property
    def exploits(self) -> Any:
        return self.raw.get("exploits") or []

    @property
    def reset(self) -> Optional[Mapping[str, Any]]:
        """`env.reset`: what one seeded reset draws and what the model is told about it.

        EVIDENCE, not an input -- read by `tests/test_task_specs.py` and by no code
        path under `bird/` (the adapter's `_reset` is the reset; this block
        describes it). None when the spec does not carry one, which the schema allows
        and the test suite does not: a reader treats the absence as "not recorded",
        never as "nothing varies".
        """
        return self.env.get("reset")

    @property
    def library(self) -> str:
        return str(self.env["library"]["name"])

    @property
    def env_id(self) -> str:
        """The BENCHMARK's env key -- not unique across the catalogue. See the class doc."""
        return str(self.env["env_id"])

    @property
    def bird_env_id(self) -> Optional[str]:
        """The `problem.env_id` this spec backs, or None if BIRD has no rule for its suite."""
        rule = _BIRD_ID_RULES.get(self.library)
        return rule(self) if rule else None

    @property
    def instruction(self) -> Optional[str]:
        """The l_task: the instruction rendered to the model.

        `l_task` where the split exists -- it is BIRD-authored (`tasks/SCHEMA_DELTA.md`
        #5) and present only on the specs BIRD is canonical for. On the specs that keep
        the core shape, the core `natural_language` field IS the instruction: it is
        the text the task definition renders as the task, so falling through to it
        supplies the same granularity, not a substitute.

        THE ONE OWNER of that fallback, on the rule `bird/envs/spec.py` states for
        `env_prose`: a `natural_language` fallback that lives in each consumer is three
        consumers disagreeing about what the field means. `config._inherit_from_task_
        spec` reads this.

        Known cost, recorded rather than hidden: the core single field is not pure
        instruction -- `hopper_hop`'s carries the joint inventory and the termination
        mechanics, content `generate.context.env_spec` exists to gate -- so on those
        specs the env_spec axis is partially flattened through `problem.task_
        description`, which no env_spec value withholds. That is a property of the
        core `natural_language` field's shape; splitting it is a schema change, not a
        fallback this accessor should silently refuse to perform.
        """
        d = self.description
        return d.get("l_task") or d.get("natural_language")


# --------------------------------------------------------------------------
# structural validation
# --------------------------------------------------------------------------


def _check(doc: Any, path: Path) -> None:
    """The subset of the schema the algorithm leans on, in pure Python.

    Deliberately NOT a reimplementation of the 1300-line schema. These are the rules a
    BIRD code path would otherwise trip over silently:

      * the twelve groups, `schema_version`, `id` == directory -- without these every
        accessor below is a guess;
      * "absence is explicit" on anchors -- `SpecEnvAdapter` turns a null anchor into
        `baselines = None`, which makes `_normalise_pool` leave fitness RAW rather than
        invent a scale. A null that lost its reason becomes an unexplained unnormalised
        number instead of a stated gap;
      * `continuous_only` and `pending` needing their reasons, for the same argument;
      * one source of truth for the normalisation, because `anchors` and `formula` both
        present is two scales for one number.
    """
    if not isinstance(doc, dict):
        raise TaskSpecError(f"{path}: not a mapping")

    missing = [g for g in REQUIRED_GROUPS if g not in doc]
    if missing:
        raise TaskSpecError(f"{path}: missing required group(s) {missing}")

    if doc["schema_version"] != SCHEMA_VERSION:
        raise TaskSpecError(
            f"{path}: schema_version {doc['schema_version']!r}, expected {SCHEMA_VERSION}")

    if doc["id"] != path.parent.name:
        raise TaskSpecError(
            f"{path}: id {doc['id']!r} does not match its directory "
            f"{path.parent.name!r}; the id is the catalogue's only unique key")

    for which in ("random", "expert"):
        anchor = doc["anchors"].get(which) or {}
        if anchor.get("value") is None and not str(anchor.get("reason") or "").strip():
            raise TaskSpecError(
                f"{path}: anchors.{which}.value is null with no reason. A value nobody "
                "obtained is recorded as absent, never omitted and never zero-filled")

    ds = doc["discrete_success"]
    shipped = ds.get("shipped") or {}
    reimpl = shipped.get("reimplementation")
    if reimpl is not None:
        missing_keys = [k for k in ("repo", "module", "symbol") if not reimpl.get(k)]
        if missing_keys:
            raise TaskSpecError(
                f"{path}: discrete_success.shipped.reimplementation is missing "
                f"{missing_keys}. It is the only field that names EXECUTABLE code, so a "
                "half-written one must fail here with the file named rather than as a "
                "bare KeyError from inside an adapter constructor")
    if ds.get("kind") in ("continuous_only", "native_authored"):
        # native_authored (the toy family): this repo authored the env, so its
        # own success() IS the native success and there is no vendor-shipped
        # block -- shipped/hardened null, and `no_discrete_success_because`
        # states why (the same shape as continuous_only).
        if not str(ds.get("no_discrete_success_because") or "").strip():
            raise TaskSpecError(
                f"{path}: discrete_success.kind is {ds.get('kind')} with no reason")
    else:
        hardened = ds.get("hardened") or {}
        if hardened.get("status") == "pending" and \
                not str(hardened.get("pending_plan") or "").strip():
            raise TaskSpecError(
                f"{path}: hardened oracle is pending with no plan. An oracle that does "
                "not exist yet is a state, not an omission")
        if hardened.get("tightened") is False and \
                not str(hardened.get("not_tightened_because") or "").strip():
            raise TaskSpecError(
                f"{path}: shipped success check left untightened with no reason")

    norm = doc["continuous_success"].get("normalized") or {}
    if norm.get("method") == "anchors" and norm.get("formula") is not None:
        raise TaskSpecError(
            f"{path}: normalized.method is `anchors` but a formula is also given; "
            "that is two sources of truth for one scale")
    if norm.get("method") == "explicit" and not str(norm.get("formula") or "").strip():
        raise TaskSpecError(f"{path}: normalized.method is `explicit` with no formula")

    _check_views(doc, path)


#: `judge.extra_views[].pose` keys. Mirrors `bird/envs/cameras.py::POSE_KEYS` -- kept
#: here as a literal because this module imports pyyaml and the standard library only
#: (see the module docstring), and `tests/test_task_specs.py` pins the two equal.
VIEW_POSE_KEYS = ("track_body", "lookat", "distance", "azimuth", "elevation")
VIEW_MODES = ("fixed", "tracking")


def _check_views(doc: Any, path: Path) -> None:
    """`judge.extra_views`: the rules the recorder would otherwise trip over silently.

    A view is drawn beside the primary and DESCRIBED to the judge by name, so two views
    with one name, or an extra view that repeats the primary camera, would put two
    panels in a frame the judge is told shows two viewpoints. A `pose` with an unknown
    key would leave a camera at MuJoCo's default pose with no error at all -- the
    typo'd key is simply never read -- which is the fabricated-pin failure the config
    loader refuses for config keys, refused here for the same reason.
    """
    judge = doc.get("judge") or {}
    primary = (judge.get("camera") or {}).get("name")
    views = judge.get("extra_views")
    if views is None:
        return
    if not isinstance(views, list):
        raise TaskSpecError(f"{path}: judge.extra_views must be a list")
    seen = set()
    for i, view in enumerate(views):
        where = f"{path}: judge.extra_views[{i}]"
        if not isinstance(view, dict):
            raise TaskSpecError(f"{where}: not a mapping")
        name = view.get("name")
        if not isinstance(name, str) or not name.strip():
            raise TaskSpecError(f"{where}: needs a non-empty `name`")
        if name == primary:
            raise TaskSpecError(
                f"{where}: {name!r} is the primary camera (judge.camera.name); an extra "
                "view that repeats it would draw the same camera twice")
        if name in seen:
            raise TaskSpecError(f"{where}: duplicate view name {name!r}")
        seen.add(name)
        if view.get("mode") not in VIEW_MODES:
            raise TaskSpecError(f"{where}: mode must be one of {list(VIEW_MODES)}, "
                                f"got {view.get('mode')!r}")
        pose = view.get("pose")
        if pose is not None:
            if not isinstance(pose, dict) or not pose:
                raise TaskSpecError(f"{where}: pose must be a non-empty mapping")
            unknown = sorted(set(pose) - set(VIEW_POSE_KEYS))
            if unknown:
                raise TaskSpecError(f"{where}: unknown pose key(s) {unknown}; "
                                    f"allowed: {list(VIEW_POSE_KEYS)}")
            has_track = pose.get("track_body") is not None
            has_lookat = pose.get("lookat") is not None
            if has_track and has_lookat:
                raise TaskSpecError(f"{where}: pose names both track_body and lookat; a "
                                    "tracking camera follows its body and a free camera "
                                    "looks at a point, not both")
            if not has_track and not has_lookat:
                raise TaskSpecError(f"{where}: pose names neither track_body nor lookat; "
                                    "distance, azimuth and elevation alone would aim the "
                                    "camera at MuJoCo's default look-at point, the world "
                                    "origin, with nothing to say so")


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


_INDEX: Optional[Dict[str, TaskSpec]] = None


def _read(path: Path) -> TaskSpec:
    blob = path.read_bytes()
    try:
        doc = yaml.load(blob.decode("utf-8"), Loader=_Loader)
    except yaml.YAMLError as exc:
        raise TaskSpecError(f"{path}: {exc}") from None
    _check(doc, path)
    return TaskSpec(id=doc["id"], raw=doc, path=path,
                    sha256=hashlib.sha256(blob).hexdigest())


def index(refresh: bool = False) -> Mapping[str, TaskSpec]:
    """Every spec under `TASKS_ROOT`, keyed by `id`. Memoised.

    Memoised because `bird/envs/metaworld.py` needs it at import time and
    `_check_coherence` needs it per validated config -- one per config on
    `--validate-all`. A spec is written once and read many times, so caching is safe,
    as it is for `config.resolved.yaml`; `refresh=True` exists for tests that write
    specs into a temporary root.
    """
    global _INDEX
    if _INDEX is not None and not refresh:
        return _INDEX
    root = _tasks_root()
    if not root.is_dir():
        raise TaskSpecError(
            f"{root} does not exist. The task catalogue is a hard repo "
            f"artifact, like configs/_default.yaml -- without it no "
            f"environment has a definition. If this path is inside "
            f"site-packages you are running a NON-EDITABLE install: `tasks/` "
            f"is a repo data directory and is deliberately not shipped in "
            f"the wheel, so set {paths.DATA_ROOT_ENV} to a directory "
            f"containing it (before importing bird -- this constant resolves "
            f"at import time) or install with `pip install -e`.")
    out: Dict[str, TaskSpec] = {}
    for path in sorted(root.glob(f"*/{SPEC_FILE}")):
        spec = _read(path)
        if spec.id in out:  # pragma: no cover - `_check` pins id to the directory name
            raise TaskSpecError(f"{path}: duplicate task id {spec.id!r}")
        out[spec.id] = spec
    _INDEX = out
    return _INDEX


def available() -> Tuple[str, ...]:
    """Sorted task ids, for an error message that suggests what the author meant."""
    return tuple(sorted(index()))


def load(task_id: str) -> TaskSpec:
    """One spec by id.

    Guarded against traversal: `task_id` reaches this from a config file and from `-s`
    on a batch-job command line, and run directories on a shared mount may be
    world-writable.
    The guard is a containment check on the resolved path rather than a blocklist of
    `..` spellings, for the same reason `checkpoint._DECODABLE` is an allow-list.
    """
    if not isinstance(task_id, str) or not task_id.strip():
        raise TaskSpecError(f"task id must be a non-empty string, got {task_id!r}")
    try:
        candidate = (_tasks_root() / task_id / SPEC_FILE).resolve()
    except (ValueError, OSError) as exc:
        # A NUL byte, an over-long name, an undecodable surrogate: `pathlib` raises
        # ValueError/OSError, not our error, and `_inherit_from_task_spec` catches only
        # TaskSpecError -- so unconverted, a bad `-s problem.task_id=...` would escape
        # config validation entirely and surface as a raw traceback from inside `load()`.
        raise TaskSpecError(f"task id {task_id!r} is not a usable path component: {exc}") from None
    if _tasks_root().resolve() not in candidate.parents:
        raise TaskSpecError(
            f"task id {task_id!r} resolves outside {_tasks_root()}")
    try:
        return index()[task_id]
    except KeyError:
        raise TaskSpecError(
            f"unknown task {task_id!r}; {_tasks_root().name}/ holds "
            f"{list(available())}") from None


def by_env_id(bird_env_id: str) -> Optional[TaskSpec]:
    """The spec backing a `problem.env_id`, or None.

    Raises on ambiguity instead of picking. Today no two specs derive one BIRD id, but
    the catalogue carries several objectives per environment (three on `HalfCheetah-v5`),
    so the day a suite's rule maps them together is the day a silent pick would hand a
    run the wrong task description and nothing would say so.
    """
    matches = [s for s in index().values() if s.bird_env_id == bird_env_id]
    if len(matches) > 1:
        raise TaskSpecError(
            f"env id {bird_env_id!r} is claimed by {sorted(s.id for s in matches)}; "
            "a task must be named explicitly (`problem.task_id`) rather than derived")
    return matches[0] if matches else None


def no_adapter_ledger() -> Mapping[str, Mapping[str, Any]]:
    """Specs carried as data for which this repo has no environment.

    Absence is explicit here too: the registry is the `problem.env_id` enum, so a spec
    with no adapter can never become a config value with no implementation behind it --
    but it CAN sit in the tree unexplained, and this is what stops that.
    """
    path = _tasks_root() / LEDGER_FILE
    if not path.exists():
        raise TaskSpecError(f"{path} is missing; the coverage partition is unstated")
    return json.loads(path.read_text())["specs"]
