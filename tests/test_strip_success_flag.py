"""`strip_existing_reward` must withhold every per-step success predicate, on every adapter.

WHY THIS FILE EXISTS. Under `generate.context.env_spec: full_source` -- the default in
`configs/_default.yaml` and Eureka's own setting -- the generator is shown
`inspect.getsource(type(env))`, the whole adapter class, with `generation._strip_reward`
as the only thing between it and the ground truth on an adapter that hands over no
`reward_source` span. A stripper regex that matched `def \\w*reward\\w*(` by stem but
`task_metric`, `success` and `reference_reward` only as WHOLE names would let
`def _success_flag(` -- the per-step predicate `step()` writes to `info["success"]`, and
on the HumanoidBench family the metric's own per-step term -- survive on every h1hand
class that declares one. Powerlift's would ship verbatim as
`return bool(s[78] > s[2] and s[2] >= _POWERLIFT_STANDING_Z)`; hurdle's as
`return bool(s[0] > _HURDLE_WALLS_X[0])`. A candidate that reads that line does not
design a reward, it restates the number it is scored on -- the tautology the stripper's
own docstring calls worse than leaking the reward.

The regex matches `def \\w*success\\w*(`, the way it matches `reward`.
These tests hold it there SIM-FREE: every adapter class is read as TEXT out of
`bird/envs/*.py` with `ast`, never imported, so they run in the `--extra test` job where
no simulator is installed and the HumanoidBench tier is otherwise deselected whole.
Checking the REAL render path (`describe("full_source")` on a constructed adapter) is
`tests/test_env_spec_leak.py`'s job, and `_success_flag` is on its list too.
"""
import ast
import re
from pathlib import Path

import pytest

from bird.components import generation

ENVS = Path(__file__).resolve().parents[1] / "bird" / "envs"

#: A `def` whose name contains `success`, however it is spelled around it. Every per-step
#: predicate in the tree takes this shape -- `_success_flag` (HumanoidBench),
#: `_success_of` (Meta-World), `success` itself (`EnvAdapter`) -- and it is
#: exactly the shape the stripper matched only as the bare word.
SUCCESS_DEF = re.compile(r"^(\s*)def\s+(\w*success\w*)\s*\(", re.IGNORECASE)
_SUCCESS_NAME = re.compile("success", re.IGNORECASE)


class _Env:
    """No `reward_source`, `reward_code` or `reference_reward_code`. The name regex is
    then the ONLY mechanism `_strip_reward` has, which is the case this file guards: the
    HumanoidBench adapters offer no span, so the regex is the whole defence."""


class _Ctx:
    env = _Env()


def _success_defs(class_node: ast.ClassDef):
    """The success-named methods declared DIRECTLY in a class body -- what
    `inspect.getsource(cls)` would show beside the class's other methods."""
    return [n for n in class_node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and _SUCCESS_NAME.search(n.name)]


def _cases():
    """`(class source, [predicate sources])` for every class under `bird/envs/` that
    declares a success-named method, read as text.

    `ast.get_source_segment` on a `ClassDef` is what `inspect.getsource(cls)` returns for
    the same class, minus decorators -- and it needs no import, so `humanoid_hand.py`
    (humanoid_bench) and `gym_mujoco.py` (mujoco) are read on an install that has
    neither. EVERY class, not only the registered leaves: Stair, Slide and the two
    Balance tasks define `_success_flag` on an intermediate parent that
    `getsource(leaf)` does not render, so a count over leaves alone misses exactly
    those four -- the parent's body is what a leaf that overrides nothing ships when
    the render is changed to include it.
    """
    out = []
    for path in sorted(ENVS.glob("*.py")):
        source = path.read_text()
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.ClassDef):
                continue
            defs = _success_defs(node)
            if not defs:
                continue
            out.append(pytest.param(
                ast.get_source_segment(source, node),
                [ast.get_source_segment(source, d) for d in defs],
                id=f"{path.name}::{node.name}"))
    return out


CASES = _cases()


def test_the_catalogue_is_not_empty_and_names_the_classes_it_guards():
    """A parametrisation over a glob passes vacuously the day the glob matches nothing,
    and this one is derived from file contents rather than from the registry. So the
    classes this guard exists for are named: `_H1HandBase` declares the abstract
    `_success_flag`, and the three below are ones whose predicates restate their metric
    most directly. Renaming the predicate is fine -- update this list, do not drop the
    guard."""
    ids = {p.id for p in CASES}
    for must in ("base.py::EnvAdapter", "humanoid_hand.py::_H1HandBase",
                 "humanoid_hand.py::H1HandPowerlift", "humanoid_hand.py::H1HandHurdle",
                 "humanoid_hand.py::H1StrongHighBarHard"):
        assert must in ids, (
            f"{must} declares no success-named method any more, so a class this "
            "guard exists for is no longer covered. If the predicate was renamed, "
            "name the new spelling here.")
    # 29 at release (24 h1hand classes, EnvAdapter, and one each on assistax,
    # upstream_assistax, gym_mujoco, metaworld). A floor, not an equality: an
    # adapter may legitimately come or go, and the named ids above are the real guard.
    assert len(ids) >= 25, f"only {len(ids)} classes declare a success-named method"


@pytest.mark.parametrize("text,predicates", CASES)
def test_no_success_def_survives_the_strip(text, predicates):
    """Three assertions, each a different failure of the same shape.

    (a) No `def *success*(` survives -- the strip fired at all.
    (b) No body line UNIQUE to a predicate survives -- the strip took the whole suite,
        not just the signature (`tests/test_strip_existing_reward.py` records the
        multi-line-signature bug where the marker was written and the body followed it).
        Uniqueness within the class source is what makes a surviving line mean "this
        body survived" rather than "this is boilerplate every method shares".
    (c) At least one withheld marker per predicate -- the marker is what a reader greps
        for, and the earlier failure mode was marker-present-and-body-present.
    """
    stripped = generation._strip_reward(_Ctx(), text)

    survived = [m.group(2) for m in map(SUCCESS_DEF.match, stripped.splitlines()) if m]
    assert not survived, (
        f"{survived} survived `strip_existing_reward`, so under `env_spec: full_source` "
        "the generator is shown the per-step ground truth it is scored on.")

    stripped_lines = {ln.strip() for ln in stripped.splitlines()}
    for pred in predicates:
        body = [ln.strip() for ln in pred.splitlines()[1:]
                if ln.strip() and not ln.strip().startswith(("#", '"', "'"))]
        unique = [ln for ln in body if text.count(ln) == 1]
        leaked = [ln for ln in unique if ln in stripped_lines]
        assert not leaked, (
            "a success predicate's body survived `strip_existing_reward`:\n"
            + "\n".join(leaked[:5]))

    markers = stripped.count(generation._WITHHELD)
    assert markers >= len(predicates), (
        f"{len(predicates)} success predicate(s) but only {markers} withheld marker(s): "
        "something was cut without leaving the marker a reader checks for.")


#: The module docstring's Powerlift example, reduced to the lines that matter, with two
#: neighbours that must NOT be cut: a helper with a neutral name and a class attribute.
POWERLIFT = '''class H1HandPowerlift(_H1HandBase):
    """Lift the barbell to a standing lockout."""

    horizon = 1000

    def _foot_height(self, s):
        return float(s[2])

    def _success_flag(self, s: np.ndarray) -> bool:
        return bool(s[78] > s[2] and s[2] >= _POWERLIFT_STANDING_Z)

    def task_metric(self, traj):
        return float(np.mean([self._success_flag(s) for s in traj.states]))
'''


def test_the_powerlift_predicate_is_cut_and_its_neighbours_are_not():
    out = generation._strip_reward(_Ctx(), POWERLIFT)
    assert "def _success_flag" not in out
    assert "_POWERLIFT_STANDING_Z" not in out, "the predicate's body survived the strip"
    assert "s[78] > s[2]" not in out
    assert "def task_metric" not in out
    assert "def _foot_height" in out and "return float(s[2])" in out, (
        "stripping ate a helper that is not ground truth")
    assert "horizon = 1000" in out, "stripping ate a class attribute"
    assert out.count(generation._WITHHELD) == 2, out


def test_the_regex_matches_success_wherever_it_sits_in_the_name():
    """`success` is matched as a STEM, like `reward` always was -- not as a whole word.
    The negatives guard against the regex being widened further than that: a helper
    with a neutral name, and `succeed`, which shares a prefix and nothing else."""
    for name in ("success", "_success_flag", "_success_of", "is_success", "Success",
                 "compute_success_rate", "describe_success"):
        assert generation._REWARD_DEF.match(f"    def {name}(self, s):"), (
            f"`def {name}(` is not matched by _REWARD_DEF")
    for name in ("_foot_height", "step", "_observe", "discretise", "render_view",
                 "succeed", "process"):
        assert generation._REWARD_DEF.match(f"    def {name}(self, s):") is None, (
            f"`def {name}(` is matched by _REWARD_DEF -- the stem is too wide")
