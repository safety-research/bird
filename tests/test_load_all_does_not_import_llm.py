"""The component modules do not import `bird.llm` at module scope.

WHY. The dependency between the component layer and the LLM clients points
one way only: `bird/llm/` may import from `bird/parsing.py` and from the
components, never the reverse. `registry.load_all()` imports every component
module, so a module-scope `bird.llm` import in any one of them would make the
LLM clients a transitive import of the whole registry, and of every process
that only trains, evaluates or replays a policy.

The route by which that happens is ACCIDENTAL. A component module that imports
a helper such as `extract_json` from `bird/llm/base.py` at module scope pulls
`bird.llm` in. `extract_json` is a string scanner -- text in, value out -- that
never knew what a model was; it lives in `bird/parsing.py`, and the tests below
pin both the placement and the absence of a re-export.

THE OTHER DIRECTION IS PINNED TOO. Every assertion about what the components
do NOT import would be satisfied by deleting the llm family from the registry
outright, so the default `load_all()` is asserted to register every provider.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _in_fresh_interpreter(body: str) -> str:
    """Run `body` in a new process and return its stdout.

    A SUBPROCESS AND NOT AN IMPORT, because the pytest process has already
    imported half the tree, and `load_all()` memoises through `_LOADED`, so
    the second call in one process is a no-op and would silently test nothing.
    """
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        cwd=REPO, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, (
        f"probe exited {out.returncode}\nstdout:\n{out.stdout}\n"
        f"stderr:\n{out.stderr}")
    return out.stdout.strip()


def test_the_default_registers_every_provider():
    """Not a formality.

    Every assertion about what the components do NOT import is satisfied by
    deleting the llm family from the registry entirely. This is the test that
    fails if someone does, and it names the three rather than counting them so
    that losing one is not absorbed by gaining another.
    """
    got = _in_fresh_interpreter('''
        from bird import registry
        registry.load_all()
        print(",".join(registry.names("llm")))
    ''')
    names = set(got.split(",")) if got else set()
    for provider in ("mock", "anthropic", "openai"):
        assert provider in names, (
            f"load_all() no longer registers llm:{provider}; "
            f"got {sorted(names)}")


def test_no_component_module_imports_bird_llm_at_module_scope():
    """The one-way dependency, guarded at the source.

    This names the file and the line, which is the difference between a
    five-minute fix and an afternoon's search.
    """
    offenders = []
    for path in sorted((REPO / "bird").rglob("*.py")):
        if path.parts[-2] == "llm":
            continue  # bird/llm may import itself
        for n, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if (stripped.startswith(("from ..llm", "from .llm",
                                     "from bird.llm", "import bird.llm"))
                    and not line.startswith((" ", "\t"))):
                offenders.append(f"{path.relative_to(REPO)}:{n}: {stripped}")
    assert not offenders, (
        "module-scope imports of bird.llm outside bird/llm/:\n  "
        + "\n  ".join(offenders)
        + "\n\nregistry.load_all() imports every component module, so an "
          "import here makes the LLM clients a transitive import of the whole "
          "registry. If the symbol is not model-specific it belongs in "
          "bird/parsing.py (that is where extract_json went); if it is, "
          "import it inside the function that needs it.")


def test_extract_json_is_in_parsing_and_not_re_exported_from_llm():
    """No re-export, deliberately.

    A compatibility alias in `bird/llm/base.py` would keep old call sites
    working -- and would keep alive the exact import path whose existence is
    the defect, so the next module wanting a JSON helper finds it there
    again. The move is only worth making if the old address is gone.
    """
    from bird import parsing
    from bird.llm import base

    assert parsing.extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert not hasattr(base, "extract_json"), (
        "bird.llm.base re-exports extract_json; the old import path is what "
        "made bird.llm a transitive import of the component modules")
