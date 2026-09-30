"""`SpecEnvAdapter` -- an `EnvAdapter` whose DESCRIPTION comes from a task spec.

The split this class draws is the schema's own. A task spec carries what a task *is*:
the prose, the observation surface, the helper vocabulary, the symbol table, the
randomisation axes, the anchors. It does not carry dynamics, and a file that claimed to
would be source code with none of Python's tooling -- which is exactly why the schema
references the human reward rather than vendoring it, and why the shipped success check is
pinned as `{module, symbol}` rather than transcribed as an evaluable string.

So:

    from the spec   _prose, _state_fields, _action_fields, _helpers, horizon,
                    obs_dim, action_dim, symbol_mapping, dr_parameters, _dr_nominal
    from Python     _step, _reset, _build_action_set, _bounds, discretise,
                    random_state, reference_reward, and the success check

This class registers no environment. It is a base, and `registry._MODULES` imports it so
that a subclass in another module can rely on it having been loaded.
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

# ANY_STEP/PER_STEP/REDUCTIONS are re-exported: an adapter reasons about reductions
# and should not have to know they are declared in the task loader.
from ..tasks import (ANY_STEP, PER_STEP, REDUCTIONS,  # noqa: F401
                     TaskSpec, TaskSpecError, by_env_id)
from .base import EnvAdapter, View


def state_fields_of(spec: TaskSpec) -> Tuple[Tuple[str, str], ...]:
    """`flat_fields` -> the `(name, doc)` pairs the four `describe()` renderings want.

    `flat_fields`, not `fields`: BIRD's reward contract is
    `compute_reward(state, action=None, next_state=None)` over a flat array, so the
    positional table is the one a candidate is actually writing against. `fields` is the
    same observation as a named namespace, for a consumer whose contract is that.
    """
    flat = spec.state_surface.get("flat_fields")
    if not flat:
        raise TaskSpecError(
            f"{spec.path}: no `state_surface.flat_fields`. An empty state surface renders "
            "a syntactically valid, information-free prompt with no error and no warning, "
            "so `generate.context.env_spec` would appear to be an axis while moving nothing")
    ordered = sorted(flat, key=lambda e: e["index"])
    if [e["index"] for e in ordered] != list(range(len(ordered))):
        raise TaskSpecError(f"{spec.path}: flat_fields indices are not 0..n-1")
    return tuple((str(e["name"]), str(e["description"])) for e in ordered)


def state_groups_of(spec: TaskSpec) -> Tuple[Tuple[str, int, int], ...]:
    """`state_surface.fields` -> `(name, start, stop)` slices over the flat row.

    THE GROUPS ARE A SECOND VIEW OF THE SAME ROW, not a second layout: `fields`
    names contiguous runs of `flat_fields` in the same order, so the slice
    bounds are a running sum of the declared widths. Rendered beside the
    positional table so a prompt can say `tool_pose_velocity_force: s[65:78]`
    as well as `tool_x: s[65]` -- a candidate that wants the tool's position
    should not have to re-derive the span from thirteen scalar names.

    RETURNS EMPTY RATHER THAN RAISING when the groups do not tile the row, and
    that is deliberate. This feeds a PROMPT: a spec whose widths disagree with
    its own `flat_fields` would otherwise take the whole search down, and the
    positional table alone is a complete interface. The disagreement is a spec
    defect, but it is `tests/test_task_specs.py`'s to refuse, not this
    renderer's -- refusing in two places means the second one fires during a
    run instead of in CI.
    """
    fields = spec.state_surface.get("fields") or ()
    n_flat = len(spec.state_surface.get("flat_fields") or ())
    out, at = [], 0
    for f in fields:
        m = re.match(r"\s*\((\d+),", str(f.get("shape", "")))
        if not m:
            return ()
        width = int(m.group(1))
        out.append((str(f.get("name", "")), at, at + width))
        at += width
    return tuple(out) if at == n_flat and out else ()


def action_fields_of(spec: TaskSpec) -> Tuple[Tuple[str, str], ...]:
    fields = (spec.env.get("spaces") or {}).get("action_fields") or ()
    return tuple((str(f["name"]), str(f["description"])) for f in fields)


def helpers_of(spec: TaskSpec) -> Tuple[Tuple[str, str], ...]:
    """`(signature, doc)` pairs, rendered only into `pythonic_class_abstraction`.

    Every entry must be `kind: inline_expression`. A helper advertised as a real method
    would let a `self.hand_pos(s)` candidate pass verification -- which binds a proxy
    `self` -- and then die in training, which binds `self=None`: a wasted policy run
    charged to the reward that did not cause it.
    """
    out = []
    for helper in spec.state_surface.get("helpers") or ():
        if helper.get("kind") != "inline_expression":
            raise TaskSpecError(
                f"{spec.path}: helper {helper.get('name')!r} is kind "
                f"{helper.get('kind')!r}; this adapter advertises expressions and "
                "implements no helper methods")
        out.append((str(helper["signature"]), str(helper["description"])))
    return tuple(out)


def dr_of(spec: TaskSpec) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, float]]:
    """`(ranges, nominal)` for `train.domain_randomization` / DrEureka's RAPP.

    An axis name the adapter does not know is dropped silently by `EnvAdapter.set_dr`,
    so a spec and an adapter that disagree here produce a run that randomises nothing
    and says so only as `degenerate` in a log.
    """
    block = spec.domain_randomization
    if not block:
        return {}, {}
    ranges = {str(k): (float(v[0]), float(v[1]))
              for k, v in (block.get("parameters") or {}).items()}
    nominal = {str(k): float(v) for k, v in (block.get("nominal") or {}).items()}
    missing = sorted(set(ranges) - set(nominal))
    if missing:
        raise TaskSpecError(f"{spec.path}: domain_randomization axes with no nominal: {missing}")
    return ranges, nominal



def views_of(spec: TaskSpec) -> Tuple[View, Tuple[View, ...]]:
    """`(primary, extras)` from the spec's `judge` block.

    `judge.camera` is required by the schema and is the camera `render` shows; it is
    returned as a `View` so the recorder can NAME the primary panel to the judge.
    `judge.extra_views` is optional and ordered: `output.video.n_views: N` takes the
    first `N - 1`, so an author puts the most informative view first. Shape rules
    (unique names, a `pose` with only known keys) are enforced by `tasks._check`, so
    by the time a spec reaches an adapter this is a straight read.
    """
    judge = spec.judge or {}
    cam = judge.get("camera") or {}
    primary = View.from_mapping({"name": cam.get("name") or "primary",
                                 "mode": cam.get("mode") or "fixed",
                                 "note": cam.get("note") or ""})
    extras = tuple(View.from_mapping(v) for v in (judge.get("extra_views") or ()))
    return primary, extras


def baselines_of(spec: TaskSpec, reduction: str) -> Optional[Dict[str, float]]:
    """`{random, expert, human}` for one reduction, or None if the spec has no pair.

    None rather than a default, and this is the useful half. `evaluation._env_baselines`
    returns None in that case and `_normalise_pool` leaves fitness RAW -- which is the
    schema's "absence is explicit" rule reaching all the way through to a number a human
    reads. A zero-filled anchor would silently rescale every fitness on the tier against
    a measurement nobody took.
    """
    anchors = spec.anchors or {}
    by_reduction = anchors.get("by_reduction")
    if not by_reduction:
        # THE FLAT SHAPE, refused DELIBERATELY rather than fallen through. Specs that
        # have no `by_reduction` may carry a MEASURED `random` (`half_cheetah` -4.58,
        # `swimmer_forward` 0.39) and no reduction label -- and an anchor that cannot
        # say which reduction it measured
        # cannot normalise anything: 0.15 per-step against 0.25 any-step on
        # drawer-close (uniform-random, n=100) is the same policy on the same task.
        #
        # Spelled out because the number being PRESENT is exactly what makes a fallback
        # tempting. From inside this function the flat anchors look overlooked, so the
        # obvious "fix" is to read them -- which would silently normalise every gym-tier
        # fitness against a possibly-any-step baseline. Do not add that fallback without
        # a reduction label on the anchor.
        if anchors.get("random") is not None or anchors.get("expert") is not None:
            return None
        raise TaskSpecError(
            f"{spec.path}: the anchors group has neither `by_reduction` nor a flat "
            "random/expert pair, so it states nothing at all -- which the schema's "
            "absence-is-explicit rule does not allow")
    pair = by_reduction.get(reduction)
    if not pair:
        return None
    lo, hi = pair["random"].get("value"), pair["expert"].get("value")
    if lo is None or hi is None:
        return None
    if lo == hi:
        raise TaskSpecError(
            f"{spec.path}: {reduction} anchors are both {lo}; normalising against them "
            "divides by zero")
    return {"random": float(lo), "expert": float(hi), "human": float(hi)}


class SpecEnvAdapter(EnvAdapter):
    """Base for adapters defined by `tasks/<id>/shared_spec.yaml`.

    `_apply_spec` is called by the subclass BEFORE `EnvAdapter.__init__`, because that
    constructor reads `_bounds()` and `_build_action_set()` and a subclass may size those
    off `obs_dim` / `action_dim`.
    """

    #: The task spec backing this instance. Named `task_spec` and not `spec` because a
    #: subclass may already call its own per-task row `spec`. Set by `_apply_spec`;
    #: `None` means the adapter
    #: still carries its own description, which is a state to migrate out of, not an
    #: error -- `_render_full_source` and friends work either way.
    task_spec: Optional[TaskSpec] = None

    def __init__(self) -> None:
        """Resolve and apply this env's spec, then build the adapter.

        Before `EnvAdapter.__init__` because that reads `_bounds()` and
        `_build_action_set()`, which a subclass may size off `obs_dim`/`action_dim`.

        A subclass that has already called `_apply_spec` itself -- `MetaWorld` does, so it
        can raise its own error naming the catalogue -- is left alone.

        A missing spec RAISES rather than falling back to whatever the class happens to
        declare. Inheriting this class is opting in, and the silent fallback is precisely
        the failure this prevents: an adapter with empty `_state_fields` renders a
        syntactically valid, information-free prompt with no error and no warning, so
        `generate.context.env_spec` looks like an axis while moving nothing.
        """
        if self.task_spec is None:
            spec = by_env_id(self.name)
            if spec is None:
                raise TaskSpecError(
                    f"{type(self).__name__} is a SpecEnvAdapter but no task spec backs "
                    f"env id {self.name!r}. Add tasks/<id>/shared_spec.yaml, or do not "
                    "inherit this class -- an adapter without a description renders an "
                    "empty prompt rather than failing")
            self._apply_spec(spec)
        super().__init__()

    def _apply_spec(self, spec: TaskSpec) -> None:
        self.task_spec = spec
        prose = spec.description.get("env_prose")
        if not str(prose or "").strip():
            raise TaskSpecError(
                f"{spec.path}: no `description.env_prose`. Falling back to "
                "`natural_language` is exactly the bug that field exists to fix -- "
                "upstream writes it at environment granularity on some specs and at "
                "instruction granularity on others, and a silent fallback would put a "
                "120-word instruction where a class docstring belongs, with no error")
        self._prose = str(prose).strip()
        self._success_prose = str(spec.description.get("success_criterion_prose") or "").strip()
        self._state_fields = state_fields_of(spec)
        #: `(name, start, stop)` per declared group, or empty when the spec's
        #: groups do not tile its own flat row. Rendered into the prompt's
        #: interface paragraph beside the positional table.
        self._state_groups = state_groups_of(spec)
        self._action_fields = action_fields_of(spec)
        self._helpers = helpers_of(spec)

        # SHAPE: assert, do not silently overwrite. These three legitimately live in
        # both places -- the class needs them to build its action set and bounds, and the
        # spec needs them to render an API stub -- so the honest contract is that the two
        # AGREE, and a disagreement is a bug in one of them rather than something to
        # paper over by preferring one.
        #
        # Silently preferring the spec would make a spec edit invisible: three
        # `describe()` renderings would follow the new value while `full_source`, which is
        # `inspect.getsource(type(self))`, kept shipping the class's literal. Two answers
        # to one question, in one prompt, with nothing to say which was current.
        #
        # Read off the INSTANCE, which falls back to the class through the MRO. `Assistax`
        # holds the row's horizon/obs_dim on the instance and `None` in the class body
        # (the base class's `horizon = 1` / `obs_dim = 0` would otherwise collide), and
        # assigns them two lines before calling this. Reading `type(self)` instead would
        # find `None`, skip, and let the spec overwrite the row -- the silent preference
        # described above, on exactly the adapters whose shapes vary per row. A class
        # that declares `None` and sets nothing still adopts the spec's value;
        # `tests/test_task_specs.py` holds all three cases.
        spaces = spec.env.get("spaces") or {}
        for attr, want in (("horizon", int(spec.env["horizon"])),
                           ("obs_dim", int(spaces["obs_dim"])),
                           ("action_dim", int(spaces["action_dim"]))):
            have = getattr(self, attr, None)
            if have is not None and int(have) != want:
                raise TaskSpecError(
                    f"{spec.path}: {attr}={want} but {type(self).__name__}.{attr}={have}. "
                    "These must agree: the class builds its action set and bounds from "
                    "its value and the spec renders the API stub from its own, so a "
                    "disagreement puts two answers in one prompt")
            setattr(self, attr, want)

        if len(self._state_fields) != self.obs_dim:
            raise TaskSpecError(
                f"{spec.path}: {len(self._state_fields)} flat_fields for an "
                f"{self.obs_dim}-D observation")
        if self._action_fields and len(self._action_fields) != self.action_dim:
            raise TaskSpecError(
                f"{spec.path}: {len(self._action_fields)} action_fields for a "
                f"{self.action_dim}-D action")

        if spec.symbol_mapping:
            self.symbol_mapping = dict(spec.symbol_mapping)
        ranges, nominal = dr_of(spec)
        if ranges:
            self.dr_parameters = ranges
            self._dr_nominal = nominal
        # The viewpoints, primary named and extras ordered (`views_of`). Set on the
        # INSTANCE so an adapter class shared by several tasks -- `MetaWorld` backs fifty
        # -- carries each task's own list rather than the last one applied.
        self.primary_view, self.extra_views = views_of(spec)

    # -- provenance the run artifact records --------------------------------

    @property
    def task_spec_id(self) -> Optional[str]:
        return self.task_spec.id if self.task_spec is not None else None

    @property
    def task_spec_sha256(self) -> Optional[str]:
        return self.task_spec.sha256 if self.task_spec is not None else None
