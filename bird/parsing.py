"""The one code extractor stage 1 and stage 2 share.

`generate.parse.patterns` is config, so both stages always read the same pattern
list. They also share the code around it: the repair path in
`components/verification.py` calls this extractor rather than keeping its own copy.
A copy drifts -- missing the language-tag strip, stripping only newlines, or
returning the whole reply instead of None -- and each drift turns a well-formed
reply into a harness failure charged to the model (a ```py fence repairing into a
program whose first line is `py`, an indented fence body into an IndentationError,
a prose-only reply into a SyntaxError at `signature_parse`).

Pure and config-free so both stages can import it without either importing the
other: the caller reads the pattern list, this applies it. Stdlib only.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, List, Optional, Sequence, Tuple

#: A fence's language tag on its own first line -- ```python / ```py / ```json
#: -- captured by the bare ```(.*?)``` pattern and never part of the program.
_LANG_TAG = re.compile(r"^\s*(python|py|json)\s*\n")

#: The `pattern_used` reported when a reply IS the program, with no fence.
BARE_CODE = "<bare-code fallback>"


def extract_code(patterns: Sequence[str], raw: str) -> Tuple[Optional[str], str]:
    """Apply `patterns` in order; returns `(code, pattern_used)`, or `(None, "")`
    when the reply holds no code.

    Syntax is NOT checked here: an extracted block that does not compile is a
    §2 `ast_syntax` failure, and stealing it into §1 would move a candidate
    between two failure populations the artifact keeps apart.

    The one exception is the leniency floor at the end -- a reply that IS the
    program, with no fence at all. It is admitted only if it parses as Python
    and defines something, which is exactly the evidence that no fence was
    needed.
    """
    raw = raw or ""
    for pattern in patterns or []:
        match = re.search(str(pattern), raw, re.DOTALL)
        if not match:
            continue
        block = match.group(1) if match.groups() else match.group(0)
        block = _LANG_TAG.sub("", block)
        if block.strip():
            return block.strip(), str(pattern)
    stripped = raw.strip()
    if "def " in stripped:
        try:
            ast.parse(stripped)
        except SyntaxError:
            return None, ""
        return stripped, BARE_CODE
    return None, ""


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Optional[Any]:
    """Pull the first JSON value out of a model response.

    Tries, in order: a ```json fence, any fence, then a balanced-brace scan.
    Returns None rather than raising -- a malformed structured answer is a
    normal event that the caller degrades on, not an exception.

    LIVES HERE RATHER THAN IN `bird/llm/`. Nothing about it is model-specific:
    text in, value out. Defining it under `bird/llm/` would make `bird.llm` a
    transitive import of two component modules and therefore of every
    training-only process. The dependency must only ever point this way --
    `bird/llm/` may import from here, never the reverse -- and
    `tests/test_load_all_does_not_import_llm.py` pins it.
    """
    if not text:
        return None
    candidates: List[str] = []
    for m in _JSON_FENCE.finditer(text):
        candidates.append(m.group(1).strip())
    stripped = text.strip()
    if stripped[:1] in ("{", "["):
        candidates.append(stripped)
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start >= 0:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == opener:
                    depth += 1
                elif text[i] == closer:
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[start:i + 1])
                        break
    for blob in candidates:
        try:
            return json.loads(blob)
        except (ValueError, TypeError):
            continue
    return None
