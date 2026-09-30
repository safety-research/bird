"""An adapter's solution policy must never reach the generator's prompt.

`generate.context.env_spec: full_source` ships `inspect.getsource(type(env))`
verbatim, after `generation._strip_reward`. `EnvAdapter.expert_policy()` is
where an adapter ships the analytic law `bird.demos` rolls out for the demo
screen and the demo ceiling -- a SOLUTION to the task, which in the
prompt is the same leak as the metric. Two lines of defence, both held here:

1. Convention: the expert is bound OUTSIDE the class body (`bird/envs/toy.py`
   binds `ToyReacher.expert_policy` at the foot of the file), so the class
   render never contains it. An expert in the class body would put the PD law
   into every tester prompt.
2. Stripper: `expert` is a `_GROUND_TRUTH_STEMS` stem, so a `def *expert*(` in
   a class body is withheld like `reward` and `success`.

Why a test: a leaked expert renders as a plausible, slightly better prompt and
nothing else.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from bird.components import generation

REPO = Path(__file__).resolve().parents[1]
ENVS = sorted((REPO / "bird" / "envs").glob("*.py"))

#: The one class-body `expert` allowed: the contract's declaration on the base,
#: which returns None and is never rendered (`full_source` renders the SUBCLASS).
_ALLOWED = {("base.py", "EnvAdapter", "expert_policy")}


def _class_methods():
    for path in ENVS:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        yield path.name, node.name, item.name


def test_no_adapter_class_body_defines_an_expert():
    found = {(f, c, m) for f, c, m in _class_methods() if "expert" in m.lower()}
    assert found == _ALLOWED, (
        f"{sorted(found - _ALLOWED)} put an expert in a class body; bind it outside "
        "the class (see the foot of bird/envs/toy.py) so `env_spec: full_source` "
        "never renders it")


def test_toy_reacher_ships_an_expert_and_its_render_does_not():
    import inspect
    from bird.envs.toy import ToyReacher
    env = ToyReacher()
    assert callable(env.expert_policy()), "the demo screen's third source is gone"
    src = inspect.getsource(ToyReacher)
    assert "expert" not in src.lower(), "the expert is back in the class body"


@pytest.mark.parametrize("name", ["expert_policy", "_expert", "toy_reacher_expert",
                                  "expertPolicy", "get_expert_action"])
def test_the_regex_matches_expert_wherever_it_sits_in_the_name(name):
    assert generation._REWARD_DEF.match(f"    def {name}(self, s):"), name


def test_a_class_body_expert_is_withheld_by_the_strip():
    text = """class Env:
    horizon = 50

    def expert_policy(self):
        def law(s, t):
            return 3.0 * (s[4:6] - s[0:2]) - 1.0 * s[2:4]
        return law

    def _observe(self):
        return self.s
"""
    class _Ctx:
        cfg = {"generate.context.strip_existing_reward": True}
        env = None
    out = generation._strip_reward(_Ctx(), text)
    assert "def expert_policy" not in out and "3.0 * (s[4:6]" not in out, out
    assert "def _observe" in out and "horizon = 50" in out, out
    assert out.count(generation._WITHHELD) == 1, out
