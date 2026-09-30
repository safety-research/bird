"""Reward design as Monte Carlo tree search (RF-Agent, NeurIPS 2025).

Six registry values, one module, because they are one mechanism read at five
points of the loop: a `parent_source` that descends the tree by UCT
(`uct_leaf`; `puct_leaf` is the same descent under AlphaZero's PUCT score,
ours, not RF-Agent's), a `sampling_mode` that expands the chosen leaf with the five LLM
"actions" (`tree_actions`), a `topology` that inserts the round's candidates as
children and backs their scores up the path (`search_tree`), and two §4
`phase`s that make the extra LLM calls the method spends per trained candidate
-- a design-thought rewrite (`thought_alignment`) and a self-verify score that
enters UCT through a softmax over siblings (`self_verify`).

Sources, and the locator style used throughout: `rfagent_algo.py:LINE` is the
release's driver, the one module of the `*_algo` package under
`refs/code/RF-Agent/RF_Agent/` at the vendored clone; `tex:LINE` is the paper's
arXiv source `neurips_2025_arxiv.tex` under `refs/tex/`; prompt files are
`RF_Agent/utils/prompts_rfagent/`. As elsewhere:
**dagger** paper and released code disagree, **double-dagger** the paper is
silent and the released code supplies the value.

WHERE THE TREE LIVES. `RunState.tree` is `{node_id: TreeNode}` under the carry
slot `search_tree`; a node holds statistics only (`bird/state.py::TreeNode`)
and the report it summarises is resolved from `state.all_reports` by id at
read time (`report_index`). Nothing here reads a rollout off a carried report
-- the invariant `state.py`'s docstring states -- and the tree never mutates
outside `search_tree`. `uct_leaf` in particular draws nothing from `ctx.rng`
and writes nothing: two parent selections over the same state are the same
selection, which is what lets `tests/test_resume.py` hold this method to a
byte-identical resumed run directory.

WHAT IS NOT REPRODUCED. Three release bugs the config does not pin:
self-verify scoring the PREVIOUS attempt's design/code
(`rfagent_algo.py:592` reads fields assigned at `:655-656`), `epoch_freq=None`
in every mutation prompt (`:273`, `:286`), and the negative decay of the
overshoot batch (`:694`, clamped at 0 here). And one loop property this module
works around rather than reproduces exactly: a round in which every candidate
failed never reaches §6 (`bird.py::run_iteration` returns before `update`), so
`_catch_up` inserts those failures at `select.failure_value` one round late --
before the next selection reads the tree -- where the release inserts them at
once (`:673`, `:800-805`). Same tree, same next selection; the journal marks
the late rows `catch_up: true`. An all-failed INIT round would otherwise
leave the tree empty, and the next round would be a second zero-shot init of 8
root children rather than the release's 8 actions on a failed root child
(`:781-790`); `_catch_up` therefore creates the root itself, and
`configs/methods/rf_agent.yaml` pins `loop.on_total_failure: continue_after_update`,
which files a failed round in its own iteration so the backstop finds nothing
missing.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..config import ConfigError
from ..context import Context
from ..registry import get as registry_get
from ..registry import register
from ..state import RunState, TreeNode
from ..types import Candidate, CandidateReport, Selection
from .evaluation import _parse_score
from .generation import (
    _SECTION_ORDER,
    _SYSTEM_TIPS,
    _build_candidate,
    _build_messages,
    _call_llm,
    _n_this_iteration,
    _n_llm_this_iteration,
    _parents_for,
)
from .update import _dispatch_operator, _dispatch_winner, _maintain_memory, _update_global_best

log = logging.getLogger("bird")

#: The virtual root's id (`rfagent_algo.py:108`, `MCTSNode(path="root")`).
ROOT = "root"

#: The five expansion actions, in the release's order (`rfagent_algo.py:123`,
#: `action_list = ['0i', '1m', '2m', '3e', '4r', '5d']` minus the init action).
#: `generate.actions` is a dict keyed by these names; iteration order of that
#: dict is the expansion order.
ACTIONS = ("mutation_mechanism", "mutation_param", "crossover_elite",
           "path_reasoning", "different_thought")

#: The release's `eps = 1e-8` on the Q normaliser (`rfagent_algo.py:75`).
_EPS = 1e-8


# ==========================================================================
# prompt texts -- VERBATIM from utils/prompts_rfagent/, minus the slots
# ==========================================================================
#
# Each action file opens with a sentence naming what is shown, then the
# `{design_idea}` / `{reward_function}` / `{trained_results}` /
# `{trained_result_analysis_tip}` slots, then the instruction paragraph. The
# slots are rendered by BIRD's own sections -- PARENT REWARD carries the idea,
# code and trained result of every group member, `generate.context.
# reflection_guidance` carries the analysis tips -- so the SEARCH ACTION section
# carries the opening sentence (placeholders dropped, `{nums}` filled) and the
# instruction paragraph, nothing else.

#: action_1_mutation_mechanism.txt:1 and :9.
_MUTATION_MECHANISM = (
    "I have one reward function with its design idea and code as follows.\n\n"
    "Please create a new reward function that has a different form but can be a "
    "modified version of the provided reward function. The new reward function "
    "should have a higher task score. Try to introduce more novel idea and add or "
    "remove some basic reward components from the environment."
)

#: action_2_mutation_param.txt:1 and :9.
_MUTATION_PARAM = (
    "I have one reward function with its design idea and code as follows.\n\n"
    "Please identify the reward function parameters and create a new reward "
    "function that has a different parameter settings compared to the provided "
    "version. The new reward function should have a higher task score."
)

#: action_3_crossover_elite.txt:1 and :8. `{nums}` is the group size.
_CROSSOVER_ELITE = (
    "I have {nums} existing reward functions with their design ideas and codes as "
    "follows.\n\n"
    "Please create a new reward function inspired by those reward functions. Try "
    "to list the common ideas in those high score reward functions and combine "
    "high-performing reward components from different reward functions based on "
    "the data tracked. The new reward function should have a higher task score."
)

#: action_4_tree_reasoning.txt:1 and :8.
_PATH_REASONING = (
    "I have {nums} existing reward functions related to evolution in sequence with "
    "their design ideas and codes as follows.\n\n"
    "Please create a new reward function inspired by all the above reward "
    "functions. Try to reason the evolution path and list some ideas in those "
    "reward functions that are clearly helpful to a better improvement. The new "
    "reward function should have a higher task score than any of them."
)

#: action_5_different_thought.txt:1 and :8-14.
_DIFFERENT_THOUGHT = (
    "I have {nums} existing reward functions with their design ideas and codes as "
    "follows.\n\n"
    "Please create a new reward function that has a totally different form from "
    "the given algorithms. Try generating codes with different structures, flows "
    "or algorithms.\n"
    "Here are some advices may helpful:\n"
    "Select observation components that are more relevant to the task for reward "
    "calculation, adopting or replacing some components from the previous reward "
    "function;\n"
    "Design a hierarchical reward system (if it is necessary) where the agent "
    "first completes one subtask and then proceeds to the next;\n"
    "Use nested rewards, such as applying an exponential function the distance "
    "between two objects or using cosine similarity.\n\n"
    "Remember the new reward function should have a higher task score."
)

#: action_1_mutation_mechanism.txt:4 (and action_3/4/5 :2, "Additionally, we
#: trained ... respectively"), with `{epoch_freq}` rendered as BIRD's own
#: stride: the reflection shows ~10 evenly strided points of the learner's
#: checkpoints (`train.checkpoint_interval`, `evaluation._REFLECT_POINTS`); the
#: release's `epoch_freq = max_iterations // 10` (`rfagent_algo.py:627`) strides
#: only its displayed curve, its max/mean/min running over every epoch
#: (`:637-641`). The release's own prompt reads
#: "after every None epochs" (a release bug).
_TRAINED_FRAMING = (
    "We trained a RL policy using the provided reward function code and tracked "
    "the values of the individual components in the reward function as well as "
    "global policy metrics such as success rates and episode lengths at each "
    "training checkpoint and the maximum, mean, minimum values encountered:"
)


def _split_action_text(text: str) -> Tuple[str, str]:
    """(opening sentence, instruction): the action files' first line and the
    paragraph after the evidence slots, kept apart so each lands where the
    release puts it."""
    head, _, tail = text.partition("\n\n")
    return head.strip(), tail.strip()


ACTION_TEXT: Dict[str, str] = {
    "mutation_mechanism": _MUTATION_MECHANISM,
    "mutation_param": _MUTATION_PARAM,
    "crossover_elite": _CROSSOVER_ELITE,
    "path_reasoning": _PATH_REASONING,
    "different_thought": _DIFFERENT_THOUGHT,
}

#: initial_thought_alignment.txt:1-5, verbatim, slots included: they are
#: filled with the candidate's own idea and code, which is the file's whole
#: content (`rfagent_algo.py:567`).
ALIGN_TEXT = (
    "Following is the Design Idea of a reward function for the problem and the "
    "code for implementing the reward function.\n"
    "Design Idea: {design_idea}\n"
    "Code: {reward_function}\n"
    "The content of the Design Idea cannot fully represent what the reward "
    "function has done informative. So, now you should re-describe the reward "
    "function using less than 4 sentences.\n"
    "Hint: You should reference the given Design Idea and highlight the most "
    "critical design ideas of the code. You can analyze the code to describe "
    "which reward components it contains, which observations they are calculated "
    "based on, what are the parameters and structure of the reward coupling "
    "process, and what special ideas are used."
)

#: initial_system_verify.txt:1-3, verbatim (`rfagent_algo.py:343`).
VERIFY_SYSTEM = (
    "You are a reward engineer trying to evaluate reward functions to solve "
    "reinforcement learning tasks as effective as possible.\n"
    "Your goal is to evaluate a reward function for the environment that will "
    "help the agent learn the task described in text.\n"
    "A good reward function should use useful variables from the environment as "
    "inputs."
)

#: self_node_value_verify_single.txt:1-8, verbatim except that the two places
#: the file states its range -- "limited to [-1,1]" (:6) and the worked example
#: "such as [0.5]" (:8) -- are rendered from `evaluate.self_verify.range`, so a
#: config that moves the range moves the instruction with it (a range the
#: prompt did not state would be a fabricated pin). At the published
#: [-1, 1] the rendering is the file byte for byte: `{lo},{hi}` -> `-1,1` and
#: the example is the range's 0.75 quantile, which is 0.5 there.
VERIFY_TEXT = (
    "Be sure to remember the definition of the task. Additionally, I have an "
    "existing reward function with its design idea and code as follows.\n\n"
    "Design Idea: {design_idea}\n"
    "Code: {reward_function}\n\n"
    "Now, based on the task definition, imagine how an expert-level strategy "
    "completes the task, describe the execution planning and motion process of "
    "the expert strategy for the task, and then evaluate the given reward "
    "functions according to your imagined description process, and judge the "
    "similarity of the given reward function with the imagined process, with the "
    "numerical value limited to [{lo},{hi}].\n\n"
    "Finally, return the value enclosed in square brackets, such as [{example}]."
)


# ==========================================================================
# helpers over `state.tree`
# ==========================================================================


def report_index(state: RunState) -> Dict[str, CandidateReport]:
    """`cand_id -> CandidateReport` over `state.all_reports`, last wins.

    The tree stores no report (`TreeNode`'s docstring); this is how a node id
    becomes the parent a prompt is built from. `all_reports` is encoded in
    full by the checkpoint, so a resumed leg resolves the same objects.
    """
    return {r.cand_id: r for r in state.all_reports}


def non_root_count(state: RunState) -> int:
    """Nodes that are reward functions -- the release's `sim_times`
    (`rfagent_algo.py:804`), i.e. trainings so far: Alg. 1's `t`."""
    return sum(1 for n in state.tree.values() if n.node_id != ROOT)


def q_bounds(state: RunState) -> Tuple[float, float]:
    """`(Q_min, Q_max)` for Eq. 2's normaliser (`tex:190`).

    Double-dagger: the paper does not say over which set. The release keeps a
    GLOBAL pair seeded at 0 and fed each new leaf's score
    (`rfagent_algo.py:113-114`, `:684-685`) -- never a backed-up Q, never
    reset -- so this is min/max over every non-root node's `score`, with 0
    always in the set.
    """
    scores = [n.score for n in state.tree.values()
              if n.node_id != ROOT and n.score is not None]
    return min([0.0] + scores), max([0.0] + scores)


def elite_ids(state: RunState, k: int) -> List[str]:
    """Top-`k` node ids by score, descending; ties by insertion order.

    The release appends every trained node -- failures at score 0 included --
    to `elite_set`, stable-sorts by `reward_cur` desc and truncates to
    `elite_max_length` (`rfagent_algo.py:687-690`). Appending then stable
    sorting is exactly "score desc, then `sim_index` asc", so the set is
    derived from the tree at read time rather than carried.
    """
    nodes = [n for n in state.tree.values() if n.node_id != ROOT and n.score is not None]
    nodes.sort(key=lambda n: (-float(n.score), n.sim_index))  # type: ignore[arg-type]
    return [n.node_id for n in nodes[:max(0, int(k))]]


def path_to_root(state: RunState, node_id: str) -> List[TreeNode]:
    """`[node, parent, ..., root]`."""
    out: List[TreeNode] = []
    cur: Optional[str] = node_id
    while cur is not None:
        node = state.tree[cur]
        out.append(node)
        cur = node.parent_id
    return out


def init_ancestor(state: RunState, node_id: str) -> str:
    """The root child whose subtree holds `node_id` (the release's
    `child_sub_tree_root`, `rfagent_algo.py:226-228`)."""
    for node in path_to_root(state, node_id):
        if node.parent_id == ROOT:
            return node.node_id
    raise KeyError(f"tree: {node_id!r} has no init ancestor")


def subtree_height(state: RunState, node_id: str) -> int:
    """Levels below `node_id` (`rfagent_algo.py:166-177`, `get_max_depth`):
    0 for a leaf."""
    node = state.tree[node_id]
    if not node.children:
        return 0
    return 1 + max(subtree_height(state, c) for c in node.children)


def lambda_now(ctx: Context, state: RunState) -> float:
    """Eq. 2's lambda for THIS round, linear in trainings done.

    Dagger: Alg. 1 line 8 decays `lambda_0 * (N - t) / N`, i.e. to 0
    (`tex:1038`); the release decays to `c_param_final = 0.1`
    (`rfagent_algo.py:117`, `:778`). `uct_lambda_final` holds whichever the
    config pins. `t` is `non_root_count` -- the release evaluates `c_param`
    before the batch, with `sim_times` = trainings already backed up. Not
    clamped past the horizon, like the release: with `horizon_trainings: 80`
    the eleventh and last expansion selects at t = 78 (lambda 0.1075); t
    reaches 86 only after that batch is backed up, when nothing selects again.
    `horizon_trainings: null` freezes lambda at `uct_lambda0`.
    """
    cfg = ctx.cfg
    lam0 = float(cfg.get("generate.tree.uct_lambda0", 0.0) or 0.0)
    lam_final = float(cfg.get("generate.tree.uct_lambda_final", 0.0) or 0.0)
    horizon = cfg.get("generate.tree.horizon_trainings")
    if horizon is None:
        return lam0
    t = non_root_count(state)
    return (lam0 - lam_final) * (1.0 - t / float(horizon)) + lam_final


def _verify_floor(ctx: Context) -> float:
    rng = ctx.cfg.get("evaluate.self_verify.range") or [-1.0, 1.0]
    return float(rng[0])


def _softmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    z = sum(exps)
    return [e / z for e in exps]


# ==========================================================================
# parent_source: uct_leaf / puct_leaf -- one descent, two child scores
# ==========================================================================
#
# Both values walk the same path: catch up a failed round, start at the root,
# step to the argmax child while the node has children and sits above
# `generate.tree.max_depth`, journal every comparison. They differ only in the
# number each child is scored by, so the walk is `_descend` and the score is a
# `_ChildScore`; anything the two shared by copy would be the thing that drifts.


class _DescentInputs:
    """Per-descent constants every child score reads (computed once)."""

    __slots__ = ("lam", "q_min", "q_max", "verify_on", "floor")

    def __init__(self, ctx: Context, state: RunState) -> None:
        self.lam = lambda_now(ctx, state)
        self.q_min, self.q_max = q_bounds(state)
        self.verify_on = bool(ctx.cfg.get("evaluate.self_verify.enabled", False))
        self.floor = _verify_floor(ctx)

    def q_norm(self, children: Sequence[TreeNode]) -> List[float]:
        return [(c.q - self.q_min) / (self.q_max - self.q_min + _EPS) for c in children]

    def sibling_softmax(self, children: Sequence[TreeNode]) -> List[float]:
        """softmax over the SIBLINGS' self-verify scores, a never-judged node at
        `evaluate.self_verify.range[0]` (the release's initial -1)."""
        return _softmax([self.floor if c.self_verify is None else float(c.self_verify)
                         for c in children])


#: `(node, children, inputs) -> (scores, rows)`: one score and one journal row
#: per child, rows without `parent`/`id` (the descent adds those).
_ChildScore = Callable[[TreeNode, Sequence[TreeNode], _DescentInputs],
                       Tuple[List[float], List[Dict[str, Any]]]]


def _score_uct(node: TreeNode, children: Sequence[TreeNode],
               d: _DescentInputs) -> Tuple[List[float], List[Dict[str, Any]]]:
    """Eq. 2 (`rfagent_algo.py:70-89`, `uct_select_with_verify`), see `parent_uct_leaf`."""
    q_norm = d.q_norm(children)
    explore = [math.sqrt(2.0 * math.log(node.visits + 1) / max(c.visits, 1))
               for c in children]
    verify = d.sibling_softmax(children) if d.verify_on else [0.0] * len(children)
    uct = [qn + d.lam * ex + d.lam * ve for qn, ex, ve in zip(q_norm, explore, verify)]
    rows = [{"q_norm": qn, "explore": ex, "verify": ve, "uct": u}
            for qn, ex, ve, u in zip(q_norm, explore, verify, uct)]
    return uct, rows


def _score_puct(node: TreeNode, children: Sequence[TreeNode],
                d: _DescentInputs) -> Tuple[List[float], List[Dict[str, Any]]]:
    """AlphaZero's PUCT (Silver et al. 2017, Methods "Select"), see `parent_puct_leaf`."""
    q_norm = d.q_norm(children)
    n_parent = sum(c.visits for c in children)
    explore = [math.sqrt(n_parent) / (1.0 + c.visits) for c in children]
    if d.verify_on:
        prior = d.sibling_softmax(children)
    else:
        prior = [1.0 / len(children)] * len(children)
    puct = [qn + d.lam * pr * ex for qn, pr, ex in zip(q_norm, prior, explore)]
    rows = [{"q_norm": qn, "prior": pr, "explore": ex, "puct": u}
            for qn, pr, ex, u in zip(q_norm, prior, explore, puct)]
    return puct, rows


def _descend(ctx: Context, state: RunState, *, rule: str,
             score: _ChildScore) -> List[CandidateReport]:
    """The shared walk. `[]` while the tree is empty -- the initialisation round,
    which `tree_actions` answers with zero-shot "init" children of the root
    (`rfagent_algo.py:781-784`). Otherwise the release's `selection`
    (`rfagent_algo.py:144-152`): argmax child, strict `>` so ties keep the
    lowest index (`np.argmax`, `:89`), until a leaf or `generate.tree.max_depth`.
    Draws nothing from `ctx.rng` and mutates nothing; the whole decision is
    journaled as `tree_select`, one row per child compared, `rule` naming the
    score."""
    cfg = ctx.cfg
    tree = state.tree
    # A round in which every candidate failed never reached §6 under
    # `loop.on_total_failure: continue` (see `_catch_up`); it joins here, before
    # this round's selection reads the tree, so the release's "failures back up
    # at 0" holds one round late. Deliberately not guarded on `if tree:`: an
    # all-failed INITIALISATION round leaves the tree empty, such a guard would
    # skip the catch-up, and the next round would re-issue zero-shot init calls
    # instead of expanding a failed child -- a different sampling mode than the
    # release's. `_catch_up` creates the root itself when it has something to add.
    _catch_up(ctx, state)
    root = tree.get(ROOT)
    if root is None or not root.children:
        return []

    reports = report_index(state)
    d = _DescentInputs(ctx, state)
    max_depth = cfg.get("generate.tree.max_depth")
    max_depth = None if max_depth is None else int(max_depth)

    node = root
    path: List[str] = []
    compared: List[Dict[str, Any]] = []
    while node.children and (max_depth is None or node.depth < max_depth):
        children = [tree[c] for c in node.children]
        scores, rows = score(node, children, d)
        best = 0
        for i in range(1, len(scores)):
            if scores[i] > scores[best]:  # strict: ties keep the lowest index
                best = i
        for c, row in zip(children, rows):
            compared.append({"parent": node.node_id, "id": c.node_id, **row})
        node = children[best]
        path.append(node.node_id)

    depth_capped = bool(node.children) and max_depth is not None and node.depth >= max_depth
    ctx.event("tree_select", rule=rule, leaf=node.node_id, path=path, depth=node.depth,
              lam=d.lam, q_min=d.q_min, q_max=d.q_max, depth_capped=depth_capped,
              children=compared)
    log.info("      %s -> %s (depth %d, lambda %.4f%s)", rule, node.node_id, node.depth,
             d.lam, ", at max_depth" if depth_capped else "")

    report = reports.get(node.node_id)
    if report is None:
        # Every node was created from a report in `all_reports` in the same
        # iteration; a miss means the tree and the history disagree, which is a
        # corrupt carry, not a case to paper over with a different parent.
        raise RuntimeError(
            f"{rule}: tree node {node.node_id!r} has no report in state.all_reports")
    return [report]


@register("parent_source", "uct_leaf")
def parent_uct_leaf(ctx: Context, state: RunState) -> List[CandidateReport]:
    """RF-Agent: descend from the root by Eq. 2's UCT to the node to expand.

    Double-dagger on the depth cap: the paper states none (`tex:174`,
    `tex:1037`); the release reads `tree_max_depth: 16` and, since
    `is_fully_expanded` is set and never read, a capped node with children is
    simply expanded again -- stopping at the cap and returning that node is the
    same behaviour.

    Per child (`rfagent_algo.py:70-89`, `uct_select_with_verify`):

        q_norm  = (q - Q_min) / (Q_max - Q_min + 1e-8)
        explore = sqrt(2 ln(N_parent + 1) / N_child)
        verify  = softmax over the SIBLINGS of self_verify        (double-dagger:
                  the paper names no set, tex:190-193; the release softmaxes
                  the children being chosen among, :78)
        uct     = q_norm + lambda * explore + lambda * verify

    and the argmax with ties to the lowest index (`np.argmax`, `:89`). A
    never-judged sibling (invalid or errored, so the phase skipped it) enters
    the softmax at `evaluate.self_verify.range[0]` (the release initialises
    `self_verify_score = -1`, `:47`); a judged-but-unparsed one carries the
    release's fallback 0 (`_verify_for_tree`); with
    `evaluate.self_verify.enabled: false` the verify term is 0 everywhere.

    Worked example, hand-checked. Root with N=2 and two children, lambda 0.4,
    Q_min 0, Q_max 0.5 (so A's q_norm is 0.5/0.50000001 = 1.0 to 7 places):

        A: q 0.5, N 1, self_verify  0.5      B: q 0.0, N 1, self_verify -0.5
        explore (both) = sqrt(2 ln 3 / 1) = sqrt(2.197225) = 1.482304
        softmax(0.5, -0.5) = (1.648721, 0.606531) / 2.255252 = (0.731059, 0.268941)
        uct A = 1.0 + 0.4*1.482304 + 0.4*0.731059 = 1.0 + 0.592922 + 0.292424 = 1.885345
        uct B = 0.0 + 0.592922 + 0.4*0.268941  = 0.592922 + 0.107576          = 0.700498
        -> A

    The walk itself is `_descend`; the journal rows carry `q_norm`, `explore`,
    `verify`, `uct`.
    """
    return _descend(ctx, state, rule="uct_leaf", score=_score_uct)


@register("parent_source", "puct_leaf")
def parent_puct_leaf(ctx: Context, state: RunState) -> List[CandidateReport]:
    """NOT A PUBLISHED REWARD-DESIGN POINT. AlphaZero's PUCT on RF-Agent's tree:
    the same descent as `uct_leaf` with the child score replaced by

        q_norm  = (q - Q_min) / (Q_max - Q_min + 1e-8)          (as uct_leaf)
        explore = sqrt(N_parent) / (1 + N_child),  N_parent = sum over the
                  siblings of N_child (AlphaZero's sum_b N(s, b))
        prior   = softmax over the SIBLINGS of self_verify      (the same set
                  and the same floor/fallback readings as uct_leaf's verify
                  term) -- or the uniform 1/|children| when
                  `evaluate.self_verify.enabled` is false
        puct    = q_norm + c_puct * prior * explore

    with `c_puct` = `lambda_now` -- the SAME `generate.tree.uct_lambda0 ->
    uct_lambda_final` schedule uct_leaf reads, so `--diff` between the two
    values is the score's form and nothing else. Silver et al. 2017 (Nature
    550, Methods, "Select") write Q(s,a) + c_puct P(s,a) sqrt(sum_b N(s,b)) /
    (1 + N(s,a)); AlphaZero's c_puct is a constant, and reusing the decaying
    lambda in its place is a choice this docstring makes, not the paper's.

    Where the prior comes from. `evaluate.self_verify` scores a candidate on
    `evaluate.self_verify.range` (RF-Agent [-1, 1]); softmax over the siblings
    turns that into a distribution over the children being chosen among, so a
    +1 sibling is weighted e^2 = 7.39x a -1 sibling and equal scores give the
    uniform. That is the reading RF-Agent's own Eq. 2 already gives the score
    (additive there, multiplicative on the exploration term here); the range is
    NOT rescaled first, so widening `range` sharpens the prior. With self-verify
    off the prior is exactly uniform and this value is a plain PUCT with
    P = 1/|children| -- the key is inert everywhere self-verify is.

    Worked example, hand-checked. Same tree as uct_leaf's: root with two
    children, lambda 0.4, Q_min 0, Q_max 0.5:

        A: q 0.5, N 1, self_verify  0.5      B: q 0.0, N 1, self_verify -0.5
        N_parent = 1 + 1 = 2 ; explore (both) = sqrt(2) / (1 + 1) = 0.707107
        prior = softmax(0.5, -0.5) = (0.731059, 0.268941)
        puct A = 1.0 + 0.4*0.731059*0.707107 = 1.0 + 0.206775 = 1.206775
        puct B = 0.0 + 0.4*0.268941*0.707107 =       0.076068
        -> A

    And the case the prior decides -- equal Q, equal visits, self-verify on:

        A: q 0.5, N 1, self_verify -0.5      B: q 0.5, N 1, self_verify  0.5
        puct A = 1.0 + 0.4*0.268941*0.707107 = 1.076068
        puct B = 1.0 + 0.4*0.731059*0.707107 = 1.206775
        -> B   (self-verify off: both 1.0 + 0.4*0.5*0.707107 = 1.141421 -> A,
                the lowest index, as `select.tie_break: first` would)

    Journal rows carry `q_norm`, `prior`, `explore`, `puct`.
    """
    return _descend(ctx, state, rule="puct_leaf", score=_score_puct)


# ==========================================================================
# sampling_mode: tree_actions
# ==========================================================================


def _group_size(ctx: Context) -> Tuple[int, int]:
    lo, hi = ctx.cfg.get("generate.tree.group_size") or [2, 2]
    return int(lo), int(hi)


def _group_for(ctx: Context, state: RunState, action: str, leaf: CandidateReport,
               reports: Dict[str, CandidateReport]) -> List[CandidateReport]:
    """The nodes one action prompt shows, LEAF LAST (`rfagent_algo.py:205`,
    `:249`: `nodes.append(child.parent)` after the sampled ones; the path action
    ends on the leaf by construction).

    Every random draw is `ctx.rng`, in the order the release makes it
    (`random.randint` for k, then the sample), so the same state yields the
    same group on a resumed leg.
    """
    cfg = ctx.cfg
    tree = state.tree
    if action in ("mutation_mechanism", "mutation_param"):
        # `rfagent_algo.py:191-198`: the parent alone.
        return [leaf]
    lo, hi = _group_size(ctx)
    if action == "crossover_elite":
        # `rfagent_algo.py:199-205`: k-1 elites weighted 1/(rank + 1 + bias),
        # bias 1 (`:130`), WITH replacement (`random.choices`), plus the parent.
        # The paper's main text says "weighted by their scores" (`tex:199`);
        # App. B.3 says reciprocal of the sorting (`tex:726`) and the code
        # agrees with the appendix.
        k = ctx.rng.randint(lo, hi)
        elites = elite_ids(state, int(cfg.get("generate.tree.elite_size", 1) or 1))
        weights = [1.0 / (rank + 2) for rank in range(len(elites))]
        picks = ctx.rng.choices(elites, weights=weights, k=k - 1) if elites else []
        return [reports[p] for p in picks] + [leaf]
    if action == "path_reasoning":
        # `rfagent_algo.py:209-217`: root-to-leaf path, root excluded, oldest
        # first, truncated to the LAST `tree_reasoning_max_length` = 4 (`:131`).
        # Dagger: the paper says a k-length path with k ~ [2, 4] (`tex:201`,
        # `tex:829`); the code is a fixed 4. `generate.tree.path_window` pins.
        window = max(1, int(cfg.get("generate.tree.path_window", 1) or 1))
        chain = [n for n in reversed(path_to_root(state, leaf.cand_id)) if n.node_id != ROOT]
        return [reports[n.node_id] for n in chain[-window:]]
    if action == "different_thought":
        # `rfagent_algo.py:221-249`: k-1 OTHER init subtrees (root children not
        # holding the leaf), sampled without replacement; in each, a uniform
        # depth in [0, height] and a greedy descent by score for that many
        # levels (`best_child_step_reward`, `:91-97`: `max`, ties to the first).
        k = ctx.rng.randint(lo, hi)
        own = init_ancestor(state, leaf.cand_id)
        available = [c for c in tree[ROOT].children if c != own]
        chosen = ctx.rng.sample(available, min(k - 1, len(available)))
        group: List[CandidateReport] = []
        for sub in chosen:
            target = ctx.rng.randint(0, subtree_height(state, sub))
            node = tree[sub]
            for _ in range(target):
                if not node.children:
                    break
                kids = [tree[c] for c in node.children]
                node = max(kids, key=lambda n: float(n.score) if n.score is not None
                           else float("-inf"))
            group.append(reports[node.node_id])
        return group + [leaf]
    raise ConfigError(f"generate.actions: unknown action {action!r}; "
                      f"expected one of {list(ACTIONS)}")


@register("sampling_mode", "tree_actions")
def sample_tree_actions(ctx: Context, state: RunState,
                        backend: Callable[..., List[str]]) -> List[Candidate]:
    """RF-Agent: expand the UCT leaf with the five LLM actions, each its own prompt.

    Round 0 (no parent): `n` zero-shot "init" candidates, each from its own
    call to the plain prompt -- the release submits `initial_size` separate
    `0i` expansions (`rfagent_algo.py:783-784`), not one prompt sampled n
    times, and the mock salts each call off the shared stream so they differ.

    Later rounds: for each action in `generate.actions`, `count` times, build
    the group `_group_for` says that action shows, render it as PARENT REWARD
    (numbered blocks with design idea, code and trained result -- the release's
    `base_thought_code.txt`), put the action's instruction in a SEARCH ACTION
    section, and make ONE call. Counts pinned at {2, 2, 2, 1, 1} = 8
    (`tex:829`, `rfagent_algo.py:123`); the coherence rule makes their sum the
    round's candidate count, and a mismatch here is refused rather than
    silently sizing the pool off the schedule.

    Every action prompt, the single-parent mutations included, renders the
    release's `base_thought_code` shape: `_parent_section` with an explicit
    lead gives numbered blocks (design idea, code, that parent's own trained
    result), `_build_messages` then appends `reflection_guidance` once after
    the blocks and renders no separate NUMERIC REFLECTION, and SEARCH ACTION
    carries the instruction -- the order of `action_1_mutation_mechanism.txt`
    lines 1-9.
    """
    cfg = ctx.cfg
    n = _n_llm_this_iteration(ctx, state)
    parents = _parents_for(ctx, state)

    if not parents:
        out: List[Candidate] = []
        for i in range(n):
            messages = _build_messages(ctx, state, [])
            got = backend(ctx, state, messages, 1)
            raw = got[0] if got else None
            out.append(_build_candidate(
                ctx, state, backend, messages, raw=raw, parent_id=None, parent_code="",
                meta={"sampling_mode": "tree_actions", "action": "init", "sample_index": i}))
        return out

    leaf = parents[0]
    reports = report_index(state)
    if "SEARCH ACTION" not in _SECTION_ORDER:
        # `_build_messages` renders `extra` titles only when `_SECTION_ORDER`
        # names them and DROPS the rest without a word -- so without this the
        # five actions would collapse into one plain parent prompt and every
        # artifact would still say `action: crossover_elite`.
        raise RuntimeError("tree_actions: generation._SECTION_ORDER lacks 'SEARCH ACTION'; "
                           "the action instruction would be silently dropped")
    actions: Dict[str, int] = dict(cfg.get("generate.actions") or {})
    total = sum(int(c) for c in actions.values())
    if total != n:
        raise ConfigError(
            f"generate.actions sums to {total} but the candidate schedule asks for {n} "
            f"this iteration; the two must agree (generate.n_candidates)")
    leaf_depth = state.tree[leaf.cand_id].depth if leaf.cand_id in state.tree else None

    out = []
    i = 0
    for action, count in actions.items():
        for _ in range(int(count)):
            group = _group_for(ctx, state, action, leaf, reports)
            lead, instruction = _split_action_text(ACTION_TEXT[action].format(nums=len(group)))
            # The release's user turn is: opening sentence -> idea / code /
            # trained results -> analysis tips -> instruction
            # (`rfagent_algo.py:270-276`, `:303-305`). The opening sentence and
            # the "We trained a RL policy ..." framing lead PARENT REWARD, whose
            # numbered blocks carry idea, code and each parent's own trained
            # result (`base_thought_code.txt`); the instruction is SEARCH ACTION.
            messages = _build_messages(ctx, state, group, extra={"SEARCH ACTION": instruction},
                                       parent_lead=lead + "\n" + _TRAINED_FRAMING)
            got = backend(ctx, state, messages, 1)
            raw = got[0] if got else None
            cand = _build_candidate(
                ctx, state, backend, messages, raw=raw, parent_id=leaf.cand_id,
                parent_code=leaf.candidate.reward_code or "",
                meta={"sampling_mode": "tree_actions", "action": action, "sample_index": i,
                      "group_ids": [r.cand_id for r in group], "k": len(group),
                      "leaf_id": leaf.cand_id, "leaf_depth": leaf_depth})
            ctx.event("tree_expand", cand_id=cand.cand_id, action=action, leaf=leaf.cand_id,
                      group=[r.cand_id for r in group])
            out.append(cand)
            i += 1
    return out


# ==========================================================================
# topology: search_tree
# ==========================================================================


@register("topology", "search_tree")
def topo_search_tree(ctx: Context, state: RunState, selection: Selection) -> RunState:
    """RF-Agent: the round's candidates become children of the expanded node,
    and each score is backed up its path (`tex:207-214`, Eq. 3).

    The incumbent bookkeeping is `single_parent_hillclimb`'s -- `iteration_best`,
    the trajectory store, `_update_global_best`, the winner and operator
    dispatch -- WITHOUT the promotion gate: the tree decides who parents next
    (`uct_leaf`), so `update.rollback_if_worse` has nothing to roll back.
    `select.final_artifact: global_best` is then the release's `best_cur_node`
    (`rfagent_algo.py:741-766`).

    Insertion (`rfagent_algo.py:679-690`): one node per report, winners and
    losers together in candidate-index order (the release backs up in thread
    completion order, `:802-805`; BIRD reassembles by index so the batch order
    is fixed). The recurrence is NOT order-independent: `best` is the max over
    children inserted so far and the blend is applied per child, so the same
    eight scores in two orders leave the parent at two different Q -- which is
    exactly why the order has to be deterministic here, and why a
    thread-completion order could never be reproduced bit for bit. `q = total = score`, `visits = 1`; a failed
    candidate arrives with `fitness` already at `select.failure_value` and
    joins like any other node (`reward_fail_bound = 0`, `:386`) -- a `None`
    fitness is refused, because it would mean a report §4 did not score
    reaching a topology that has no rule for it.

    Backup (`rfagent_algo.py:59-68` and `:692-699`), for every ancestor `a`
    below the root, with `decay = max(0, 1 - sim_index / horizon)`:

        a.visits += 1 ; a.total += score ; mean = a.total / a.visits
        best = max(child.q for child in a.children)
        a.q = (1 - eta - mu*decay) * a.q + eta * best + mu*decay * mean

    `eta` is `update.tree.best_child_weight` (Eq. 3's update rate, 0.7,
    `tex:829`); `mu` is `update.tree.mean_weight`. Dagger: Eq. 3 has no mean
    term at all; the release's `update` blends `update_mean_gamma = 0.15 *
    decay` of the running mean of backed-up scores (`:44`, `:67-68`). The
    release's decay goes negative on the overshoot batch (`t = 81..86` against
    `N = 80`, `:694`) and is clamped at 0 here; `horizon_trainings: null`
    holds decay at 1. The root only counts visits (`:698-699`).

    Worked example, hand-checked. Parent P was created at score 0.2 (q 0.2,
    visits 1, total 0.2). Its child C scores 0.6 as the 10th training of an
    80-training horizon, eta 0.7, mu 0.15:

        decay = 1 - 10/80 = 0.875 ; mu*decay = 0.13125
        P.visits = 2 ; P.total = 0.8 ; mean = 0.4 ; best = max(0.6) = 0.6
        P.q = (1 - 0.7 - 0.13125)*0.2 + 0.7*0.6 + 0.13125*0.4
            = 0.16875*0.2 + 0.42 + 0.0525 = 0.03375 + 0.42 + 0.0525 = 0.50625

    (Eq. 3 alone, mu = 0: 0.3*0.2 + 0.7*0.6 = 0.48.)
    """
    cfg = ctx.cfg
    winner = selection.winner
    state.iteration_best = winner
    _maintain_memory(ctx, state, selection)
    if winner is not None:
        _update_global_best(ctx, state, winner)
        _dispatch_winner(ctx, state, winner)
        _dispatch_operator(ctx, state, selection)

    _ensure_root(state)
    _catch_up(ctx, state)
    horizon = cfg.get("generate.tree.horizon_trainings")
    eta = float(cfg.get("update.tree.best_child_weight", 1.0) or 0.0)
    mu = float(cfg.get("update.tree.mean_weight", 0.0) or 0.0)

    for report in _in_slot_order(list(selection.winners) + list(selection.losers)):
        _insert_report(ctx, state, report, horizon, eta, mu, catch_up=False)

    _tree_summary(ctx, state)
    return state


def _ensure_root(state: RunState) -> None:
    if ROOT not in state.tree:
        state.tree[ROOT] = TreeNode(node_id=ROOT, parent_id=None, depth=0, action="root",
                                    iteration=-1, score=None, sim_index=0)


def _in_slot_order(reports: Sequence[CandidateReport]) -> List[CandidateReport]:
    """Candidate-INDEX order. `meta["sample_index"]` is the slot a candidate was
    generated into and survives repair (`verification._regenerate` copies meta),
    where a repaired id `r0000` would sort after every `c####` sibling and move
    its `sim_index` -- and with it the decay -- to the end of the batch."""
    def _slot(r: CandidateReport) -> Tuple[int, int, str]:
        idx = r.candidate.meta.get("sample_index")
        return (int(r.candidate.iteration),
                int(idx) if isinstance(idx, int) else 10 ** 9, r.cand_id)
    return sorted(reports, key=_slot)


def _catch_up(ctx: Context, state: RunState) -> None:
    """Insert every scored report of an EARLIER iteration the tree does not hold.

    Under `loop.on_total_failure: continue`, `bird.py::run_iteration` returns
    before §6 when every candidate of a round failed, so a round of eight
    non-compiling rewards never reached this topology -- and `uct_leaf`,
    deterministic over an unchanged tree, would descend to the same leaf and
    re-issue the same eight prompts for the rest of the run while `status.json`
    read `ok`. The release has no such branch: each failed child returns
    `reward_fail_bound = 0` and is backed up like any other
    (`rfagent_algo.py:673`, `:800-805`), moving `sim_times`, the parent's
    visits and Q, and therefore the next selection. `state.all_reports` is
    extended BEFORE that early return (`bird.py::run_iteration`) and is never
    carry-gated, so the failed round is recoverable here, one iteration late
    and in the same slot order; the journal marks these rows `catch_up: true`.
    Only reports of PAST iterations qualify -- the current round's arrive
    through `topo_search_tree` after selection.

    This is the BACKSTOP. `loop.on_total_failure: continue_after_update` (what
    `configs/methods/rf_agent.yaml` pins) runs §6 on the failed round itself, with no
    winner, so the round joins the tree in its own iteration -- including the
    final round, which no later descent would ever catch up -- and this
    function finds nothing missing. Works into an empty tree too: an all-failed
    initialisation round creates the root here.
    """
    cfg = ctx.cfg
    tree = state.tree
    missing = [r for r in state.all_reports
               if r.cand_id not in tree and int(r.candidate.iteration) < int(state.iteration)]
    if not missing:
        return
    _ensure_root(state)
    horizon = cfg.get("generate.tree.horizon_trainings")
    eta = float(cfg.get("update.tree.best_child_weight", 1.0) or 0.0)
    mu = float(cfg.get("update.tree.mean_weight", 0.0) or 0.0)
    for report in _in_slot_order(missing):
        _insert_report(ctx, state, report, horizon, eta, mu, catch_up=True)
    log.info("      search_tree: %d node(s) from an all-failed round joined late", len(missing))


def _insert_report(ctx: Context, state: RunState, report: CandidateReport,
                   horizon: Any, eta: float, mu: float, *, catch_up: bool) -> None:
    """One report -> one node, then Eq. 3 up the path (`rfagent_algo.py:679-699`)."""
    tree = state.tree
    if report.fitness is None:
        raise RuntimeError(
            f"search_tree: {report.cand_id} reached §6 with fitness None; the tree "
            "needs a scalar for every node (select.failure_value for failures)")
    if report.cand_id in tree:
        raise RuntimeError(f"search_tree: {report.cand_id} is already in the tree")
    score = float(report.fitness)
    parent_id = report.candidate.parent_id
    parent_id = parent_id if parent_id in tree else ROOT
    parent = tree[parent_id]
    sim_index = non_root_count(state) + 1
    node = TreeNode(
        node_id=report.cand_id, parent_id=parent_id, depth=parent.depth + 1,
        action=str(report.candidate.meta.get("action", "init")),
        iteration=int(report.candidate.iteration), score=score, q=score, visits=1,
        total=score, self_verify=_verify_for_tree(ctx, report), sim_index=sim_index)
    tree[node.node_id] = node
    parent.children.append(node.node_id)

    decay = 1.0 if horizon is None else max(0.0, 1.0 - sim_index / float(horizon))
    backed: List[Dict[str, Any]] = []
    anc_id = parent_id
    while anc_id != ROOT:
        a = tree[anc_id]
        q_before = a.q
        a.visits += 1
        a.total += score
        mean = a.total / a.visits
        best = max(tree[c].q for c in a.children)
        a.q = (1.0 - eta - mu * decay) * a.q + eta * best + mu * decay * mean
        backed.append({"id": a.node_id, "q_before": q_before, "q_after": a.q,
                       "visits": a.visits})
        anc_id = a.parent_id  # type: ignore[assignment]
    tree[ROOT].visits += 1
    ctx.event("tree_backup", node=node.node_id, parent=parent_id, score=score,
              sim_index=sim_index, decay=decay, path=backed, catch_up=catch_up)


def _verify_for_tree(ctx: Context, report: CandidateReport) -> Optional[float]:
    """The self-verify value a node carries into Eq. 2's softmax.

    Three cases, two of them the release's (`rfagent_algo.py:47`, `:611-616`):
    a parsed score is itself; a call that was made and did not parse is the
    release's `except` fallback, 0 -- rendered as the midpoint of
    `evaluate.self_verify.range` so a moved range keeps the same reading; a
    candidate never judged (invalid or errored, so the phase skipped it) is
    `None`, which `uct_leaf` reads as `range[0]`, the release's `-1` initial
    value. `report.meta["self_verify"]` keeps `None` for the unparsed case so
    the artifact says "unparsed" rather than "middling" -- only the tree's
    reading is the release's.
    """
    sv = report.meta.get("self_verify")
    if sv is not None:
        return float(sv)
    if report.meta.get("self_verify_parsed") is False:
        lo, hi = [float(x) for x in (ctx.cfg.get("evaluate.self_verify.range") or [-1.0, 1.0])]
        return (lo + hi) / 2.0
    return None


def _tree_summary(ctx: Context, state: RunState) -> None:
    cfg = ctx.cfg
    tree = state.tree
    q_min, q_max = q_bounds(state)
    elite_size = int(cfg.get("generate.tree.elite_size", 1) or 1)
    ctx.event("tree", n_nodes=non_root_count(state),
              max_depth=max((n.depth for n in tree.values()), default=0),
              elites=elite_ids(state, elite_size), q_min=q_min, q_max=q_max)


# ==========================================================================
# phases: thought_alignment, self_verify
# ==========================================================================


def _trained(report: CandidateReport) -> bool:
    """The release's gate: `traceback_msg == ''` after the RL run
    (`rfagent_algo.py:563`) -- a candidate that compiled and trained."""
    result = report.result
    return bool(report.candidate.valid) and bool(getattr(result, "trained", True)) \
        and not getattr(result, "error", "")


def _task_and_env(ctx: Context, state: RunState) -> str:
    """The release's `initial_user` -- task, then the environment source
    (`initial_user.txt:1`) -- in BIRD's section shape."""
    env_spec = registry_get("env_spec", ctx.cfg["generate.context.env_spec"])(ctx, state)
    return ("## TASK\n" + str(ctx.cfg["problem.task_description"]).strip()
            + "\n\n## ENVIRONMENT\n" + str(env_spec).strip())


def _num(v: float) -> str:
    """`-1` not `-1.0`, `0.5` as is -- the file's own spelling of its numbers."""
    return f"{float(v):g}"


@register("phase", "thought_alignment")
def phase_thought_alignment(ctx: Context, state: RunState,
                            reports: List[CandidateReport]) -> List[CandidateReport]:
    """RF-Agent's thought alignment: once a reward has trained, re-describe it
    from its idea AND its code (`tex:205`; `rfagent_algo.py:566-588`).

    One generator call per trained candidate. The release's system prompt is
    Eureka's `initial_system` ALONE -- no `code_output_tip` (`:566`). RECORDED
    gap: this phase sends `generation._SYSTEM_TIPS`, the generic persona, not
    `initial_system` and not `generate.context.system_prompt`. The text lands
    in `report.meta["design_thought"]`, which `_parent_section` prefers over
    `candidate.nl_spec` when `include_parent_thought` is on, and which
    `self_verify` reads next -- the release assigns the aligned thought before
    verifying (`:587`, then `:592`, though see the module docstring for what
    `:592` actually reads).
    """
    temperature = ctx.cfg.get("llm.generator.temperature")
    for report in reports:
        if not _trained(report):
            continue
        cand = report.candidate
        user = (_task_and_env(ctx, state) + "\n\n"
                + ALIGN_TEXT.format(design_idea=cand.nl_spec or "(none)",
                                    reward_function=(cand.reward_code or "").strip()))
        messages = [{"role": "system", "content": _SYSTEM_TIPS},
                    {"role": "user", "content": user}]
        got = _call_llm(ctx, ctx.generator, messages, 1, temperature,
                        tag="describe_reward_thought")
        text = (got[0] if got else "").strip()
        report.meta["design_thought"] = text
        # The question is kept beside the answer (`report.json` via meta), for the
        # same reason `judgments/` exists: a judgment whose prompt was never
        # written down cannot be audited or re-asked.
        report.meta["thought_alignment_prompt"] = user
        ctx.event("thought_alignment", cand_id=report.cand_id, chars=len(text))
    return reports


_BRACKET_RE = re.compile(r"\[(.*?)\]", re.DOTALL)


@register("phase", "self_verify")
def phase_self_verify(ctx: Context, state: RunState,
                      reports: List[CandidateReport]) -> List[CandidateReport]:
    """RF-Agent's self-verify score: how much the reward resembles an imagined
    expert strategy, on `evaluate.self_verify.range` (`tex:188`, App. B.4
    `tex:812-820`; `rfagent_algo.py:590-617`).

    One evaluator call per trained candidate; the design idea shown is the
    aligned thought when `thought_alignment` ran, else `nl_spec`. Parsing is
    the release's (`:611-613`): the LAST `[...]` group as a float -- UNCLAMPED,
    so a model that answers `[1.5]` on [-1, 1] is recorded at 1.5 with
    `self_verify_in_range: false` rather than silently pulled to the bound.
    A bracket that is not a bare number goes through `evaluation._parse_score`
    on the group; no bracket at all goes through it on the whole text. What
    nothing parses (no bracket, no `a/b` fraction, no bare number inside the
    range) is `None` in the artifact; `_verify_for_tree` reads a
    judged-but-unparsed `None` as the midpoint of `range` (the release's
    fallback 0, `:616`) and only a never-judged node as `range[0]` --
    `self_verify_parsed` is the flag.
    """
    if ctx.evaluator is None:
        raise RuntimeError("self_verify: no LLM client on ctx.evaluator; "
                           "set llm.evaluator.provider to a registered client (e.g. 'mock')")
    cfg = ctx.cfg
    lo, hi = [float(x) for x in (cfg.get("evaluate.self_verify.range") or [-1.0, 1.0])]
    example = lo + 0.75 * (hi - lo)
    temperature = cfg.get("llm.evaluator.temperature")
    for report in reports:
        if not _trained(report):
            continue
        cand = report.candidate
        idea = report.meta.get("design_thought") or cand.nl_spec or "(none)"
        user = (_task_and_env(ctx, state) + "\n\n"
                + VERIFY_TEXT.format(design_idea=idea,
                                     reward_function=(cand.reward_code or "").strip(),
                                     lo=_num(lo), hi=_num(hi), example=_num(example)))
        messages = [{"role": "system", "content": VERIFY_SYSTEM},
                    {"role": "user", "content": user}]
        got = _call_llm(ctx, ctx.evaluator, messages, 1, temperature, tag="judge_self_verify")
        text = got[0] if got else ""
        groups = _BRACKET_RE.findall(text)
        score: Optional[float] = None
        if groups:
            try:
                score = float(groups[-1].strip())
            except ValueError:
                score = _parse_score(groups[-1], lo, hi)
        else:
            score = _parse_score(text, lo, hi)
        if score is not None and not math.isfinite(score):
            score = None
        report.meta["self_verify"] = score
        report.meta["self_verify_parsed"] = score is not None
        report.meta["self_verify_prompt"] = user
        report.meta["self_verify_response"] = text
        report.meta["self_verify_in_range"] = score is not None and lo <= score <= hi
        ctx.event("self_verify", cand_id=report.cand_id, score=score, parsed=score is not None)
    return reports
