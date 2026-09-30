"""Cross-iteration state.

Everything that survives an iteration lives here. `loop.carry` (§0) governs
the MEMORY slots -- the `CARRY_SLOTS` below, with their
`CARRY_COMPANIONS`: a memory slot not named there is cleared by `apply_carry` at
the iteration boundary, so a stage that wants a previous iteration's incumbent,
dialogue, archive or checkpoint must have its config name the slot.

TWO ATTRIBUTES ARE NOT MEMORY SLOTS AND SURVIVE EVERY `loop.carry`, `[]`
INCLUDED: `iteration_best` (this round's winner, written by every topology) and
`latest` (the chain head, written by `update.winner.action: become_parent`).
They are the PARENT-CHAIN slots, governed by `generate.parent_source`,
`update.winner.action` and `select.final_artifact`, not by `loop.carry`. That is
by design, not by omission: every method configuration -- Eureka with
`[best_reward, dialogue]`, CARD with `[dialogue, trajectory_store]` -- reads
`iteration_best` or `latest` with no carry slot for it, so gating them would
break every declared method rather than fix one. The documented way to say "no
parent" is `generate.parent_source: none` or `update.operator: restart`, never
`loop.carry: []`, which drops memory and keeps the chain.
`tests/test_carry_contract.py` pins both lists.

A THIRD ATTRIBUTE, `returned`, IS A RECORD RATHER THAN A SLOT: the run's return
as `select.final_artifact` resolved it at the end of the last completed
iteration, written by `update()` BEFORE the boundary carry. `final_artifact`
reads a memory slot (`best`, `archive`) or the chain head, and the carry can
have dropped the slot by the time the run ends -- `zeroshot` (`loop.carry: []`,
one iteration, inherited `global_best`) would otherwise return
`returned_cand_id: None` with its candidate sitting in the artifact. What a search PRODUCED
cannot depend on what its config forgets between iterations, so the return is
recorded when it is known and read back through `RunState.final_artifact`.

That indirection is deliberate. `loop.carry` is how a config says what kind of
search it is running -- Eureka carries `[best_reward, dialogue]`, LIMEN carries
`[archive, failure_memory]`, GT carries a preference dataset and a replay
buffer -- without any stage needing to know which method it is part of.

THE INVARIANT A CARRIED REPORT RESTS ON, and it is a norm rather than a guard:
**a carried report is read for its scalar, its prose and its code -- never for
its rollouts.** `best`, `latest`, `iteration_best`, `all_reports` and every
`ArchiveCell.report` are consumed for `.fitness`, `.feedback` and
`.candidate.reward_code`. Every read of `.result.trajectories` in the tree
operates on THIS iteration's reports, never on one pulled back out of
`all_reports` (the readers in `observability.py`, `phases.py`, `update.py`,
`preferences.py` and `evaluation.py` all obey this).

Two places already depend on it: `RunDir.save_train_result` drops trajectories
from the artifact, and `bird/checkpoint.py` drops them from the checkpoint (and
COUNTS them in `dropped`). `state.trajectory_store` and `state.preferences` keep
theirs -- those are carry slots and they are the method. A future component that
reads rollouts off a carried report would make a resumed run silently divergent,
and no static test can catch it; this paragraph is the whole defence.

`summary()` and `to_json()` are the REPORT (`state/iterNN.json`); they are
deliberately lossy and are NOT the checkpoint.
The checkpoint is a separate artifact under `checkpoints/`. Do not merge them.
The objects the summary reduces to counts are written beside it by
`RunDir.save_records` (`records/iterNN.json`) -- still a report a
reader consumes, still not the checkpoint; the counts stay here because a
count that disagrees with the records it summarises is itself an alarm.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional

from .types import Candidate, CandidateReport, Preference, Trajectory

#: The values `select.final_artifact` takes; `RunState.resolve_final_artifact`
#: implements each and `bird.final_artifact` validates against this.
FINAL_ARTIFACT_RULES = ("global_best", "chain_end", "archive_best", "all_survivors")

#: Legal members of `loop.carry`, each mapping to a RunState attribute.
CARRY_SLOTS = {
    "best_reward": "best",
    "dialogue": "dialogue",
    "preference_dataset": "preferences",
    "trajectory_store": "trajectory_store",
    "subtask_list": "subtasks",
    "replay_buffer": "replay_ref",
    "policy_checkpoint": "policy_ref",
    "failure_memory": "failure_memory",
    "archive": "archive",
    "curriculum": "curriculum",
    "search_tree": "tree",
    "critic_population": "critics",
}

#: Attributes that travel WITH a carry slot rather than having a slot of their
#: own. `descriptor_stats` is archive state -- LIMEN persists `_feature_stats`
#: in the same metadata.json as the grid (`refs/code/LIMEN/limen/database.py`
#: L650, L679) -- so `loop.carry: [archive, ...]`, the published value, carries
#: it without a config change, and a config that drops the archive drops the
#: stats that give its coordinates meaning. NOT a `CARRY_SLOTS` entry: the
#: schema's `loop.carry` enum names things a config may choose independently,
#: and running stats without their grid (or vice versa) is not a choosable
#: state, it is a corrupt one.
CARRY_COMPANIONS = {
    "archive": ("descriptor_stats",),
}


@dataclass
class ArchiveCell:
    """One MAP-Elites cell (LIMEN)."""

    coords: tuple
    island: int
    report: Optional[CandidateReport] = None
    fitness: float = float("-inf")


@dataclass
class CurriculumState:
    """An ordered progression of skills, and where in it the search is.

    RDA already decomposes a task; a curriculum is not that.
    `state.subtasks` is J CONCURRENT ASPECTS of one task, all scored every
    iteration while the policy attempts the whole thing. This is K SEQUENTIAL
    STAGES with a gate: the search works on `stages[index]` and nothing later
    until something says stage `index` is done. Same J, different loop --
    "how is each aspect going?" against "which aspect are we allowed to work on
    yet?".

    ONE dataclass in ONE carry slot, deliberately. The stage list without the
    index is not a curriculum: carried separately, a config could keep the
    stages and drop the position, and the search would restart at stage 0 every
    iteration while every artifact still said "curriculum". Slots are cleared
    independently by `apply_carry`, so the only way to make that unrepresentable
    is for the two to be one object.

    `checkpoints` maps a stage index (as a string, because this is JSON in a
    checkpoint) to the `policy_ref` of the report whose rollout PASSED the gate
    for that stage -- the judged winner's own, never `state.policy_ref`, which
    under elitism is the incumbent's and may never have been judged. It
    is what a regression rollback restores, and it is why passing a stage is
    recorded rather than merely counted.
    """

    stages: List[str] = field(default_factory=list)
    index: int = 0
    entered_at: int = 0  # the iteration the current stage began
    #: One entry per event that moved the curriculum -- authored, passed,
    #: stalled, split, regressed. The recorded per-stage trace of what the
    #: policy actually did at each handover: a stage that passed and a
    #: stage that was advanced past because it stalled are the same `index += 1`
    #: and must never be the same artifact.
    history: List[Dict[str, Any]] = field(default_factory=list)
    #: stage index (str) -> the gated report's own `policy_ref` when it passed
    checkpoints: Dict[str, str] = field(default_factory=dict)
    #: stage index (str) -> the gate score it passed with, for the regression
    #: check to compare against. A pass is not a number the gate keeps
    #: otherwise, and "still working" is a comparison, not an absolute.
    pass_scores: Dict[str, float] = field(default_factory=dict)
    #: What the generator is told beyond the stage text -- a regression warning,
    #: a re-split notice. Empty on the ordinary path.
    note: str = ""
    splits: int = 0
    #: Iterations one stage may spend, FROZEN at authoring time when the config
    #: leaves `stage_patience` null. Frozen because the derived value is
    #: `n_iterations // n_stages` and `on_stall: resplit` grows `n_stages`: left
    #: live, the remedy for a stalled stage shrinks the budget of the stages it
    #: creates, so each of them stalls sooner, splits again, and the search
    #: re-splits every iteration without ever training anything (observed:
    #: 7 -> 9 -> 13 stages in three iterations, zero stages passed). 0 means
    #: "not yet derived" -- and it
    #: STAYS 0 when the config named `stage_patience` explicitly, because
    #: freezing exists only for the derived value. A reader wanting the
    #: EFFECTIVE patience takes it from the `authored` history event, which
    #: records what the run actually used; this field answers the narrower
    #: question "was it derived, and to what".
    patience: int = 0

    @property
    def current(self) -> str:
        """The stage the search is working on, or `""` when there is none."""
        if 0 <= self.index < len(self.stages):
            return self.stages[self.index]
        return ""

    @property
    def complete(self) -> bool:
        return bool(self.stages) and self.index >= len(self.stages)

    def summary(self) -> Dict[str, Any]:
        return {
            "n_stages": len(self.stages),
            "index": self.index,
            "current": self.current,
            "complete": self.complete,
            "passed": sorted(int(k) for k in self.checkpoints),
            "splits": self.splits,
            "patience": self.patience,
            "history": self.history,
        }


@dataclass
class TreeNode:
    """One node of a Monte Carlo search tree over reward functions (RF-Agent).

    A virtual root plus one node per trained candidate (`neurips_2025_arxiv.tex:183`,
    `s = [z, R, F, l_feedback]`). The node carries the SEARCH STATISTICS only --
    the report it summarises is resolved from `RunState.all_reports` at read
    time (`bird/components/tree.py::report_index`), never duplicated here, so a
    checkpoint encodes each report once and the tree stays a few hundred bytes
    per node.

    `score` is the node's own `report.fitness` -- for a failed candidate that is
    already `select.failure_value` (the release's `reward_fail_bound = 0`,
    `rfagent_algo.py:386`), so a failure sits in the tree and in the elite set
    like any other node (`rfagent_algo.py:687-690`). `q` starts equal to
    `score` (`rfagent_algo.py:680-681`, leaf Q at creation) and moves only by
    backup; `visits`/`total` are the release's `visits`/`total_reward`
    (`rfagent_algo.py:63-64`), `total` feeding the running mean the release
    blends into Q. `sim_index` is the release's `sim_time` (`rfagent_algo.py:805`):
    1-based insertion order over non-root nodes, and the `t` of Alg. 1's
    lambda schedule and of the backup decay.
    """

    node_id: str  # cand_id, or "root"
    parent_id: Optional[str]  # None for root
    children: List[str] = field(default_factory=list)  # insertion order
    depth: int = 0  # root 0, init nodes 1
    action: str = "root"  # "root" | "init" | one of the five action names
    iteration: int = -1
    score: Optional[float] = None  # report.fitness; None for root
    q: float = 0.0
    visits: int = 0
    total: float = 0.0  # sum of scores backed up through this node (for the mean term)
    self_verify: Optional[float] = None
    sim_index: int = 0  # 1-based insertion order over non-root nodes; root 0


@dataclass
class RunState:
    """The mutable state threaded through the loop by `update()`."""

    # --- bookkeeping ---
    iteration: int = 0
    restart: int = 0
    run_id: str = ""
    rng_seed: int = 0

    # --- carry slots (see CARRY_SLOTS) ---
    best: Optional[CandidateReport] = None  # global incumbent (`best_reward`)
    # --- parent-chain slots: NOT carry slots. Written every round by the
    # topology / `update.winner.action`, they survive every `loop.carry`, `[]`
    # included (module docstring). Declared here, between two carry slots, only
    # because dataclass field order is positional and moving them is a break.
    iteration_best: Optional[CandidateReport] = None
    latest: Optional[CandidateReport] = None  # CARD's chain head
    # --- carry slots, continued ---
    dialogue: List[Dict[str, str]] = field(default_factory=list)
    preferences: List[Preference] = field(default_factory=list)
    trajectory_store: List[Trajectory] = field(default_factory=list)
    subtasks: List[str] = field(default_factory=list)
    #: R*'s `PCritic`: the source of each LLM-written critic PROGRAM, not a
    #: compiled function. Code survives a checkpoint round trip and a fork;
    #: a closure does neither, and a critic population silently emptied by a
    #: resume would drop the vote to zero and read as "no segments found".
    critics: List[str] = field(default_factory=list)
    replay_ref: Optional[str] = None
    policy_ref: Optional[str] = None
    failure_memory: List[Dict[str, str]] = field(default_factory=list)
    archive: Dict[tuple, ArchiveCell] = field(default_factory=dict)
    #: Running min/max per archive descriptor, `{name: {"min": .., "max": ..}}`.
    #: LIMEN's adaptive binning (`database.py::_update_feature_stats` L715-723):
    #: NEVER reset within a run, updated once per report at insert, and the
    #: coords it produced are frozen on the report -- the stats moving later
    #: re-keys nothing. Travels with the `archive` carry slot (CARRY_COMPANIONS)
    #: and checkpoints through the generic dataclass path (plain str/float
    #: dict; older checkpoints decode it to {}).
    descriptor_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    curriculum: Optional[CurriculumState] = None
    #: RF-Agent's search tree, keyed by node id (`"root"` plus one cand_id per
    #: trained candidate). Carry slot `search_tree`. Statistics only -- see
    #: `TreeNode`; the reports live in `all_reports`.
    tree: Dict[str, TreeNode] = field(default_factory=dict)

    # --- transient, per-iteration ---
    #: Which `loop.waves` pass is executing. 0 with no waves configured, which
    #: is every method but R*. Transient by design: it is a position inside an
    #: iteration, not state that survives one, and a checkpoint restores at an
    #: iteration boundary where it is always 0.
    wave: int = 0
    #: Did this iteration BEGIN with fewer than two evaluated candidates? Set
    #: once by `run_iteration` before any wave runs, and read by
    #: `loop.waves`' `when: no_archive`. Recomputing it per wave would be
    #: wrong in the one case it exists for: wave 1 fills the archive.
    wave_bootstrap: bool = False
    #: Is THIS iteration the generate-only last step of a
    #: `loop.termination: fixed_generations` run (CARD)? Set by `run_search`
    #: right before `run_iteration`, from the configured rule's
    #: `ends_on_generation` attribute and `it + 1 >= loop.n_iterations`; read by
    #: stages 2-4 (validity only, no training, no evaluation). Transient like
    #: `wave`: recomputed from `it` every pass, so a resumed leg re-entering the
    #: last iteration gets it from the loop counter and never from the file --
    #: the checkpoint codec round-trips it generically and an older
    #: checkpoint decodes it to False. NOT a carry slot.
    generation_only: bool = False
    human_feedback: str = ""
    last_selection_notes: str = ""
    feedback_channel: str = "none"
    consecutive_failures: int = 0
    stagnation: int = 0

    # --- history for termination rules and reporting ---
    fitness_history: List[Optional[float]] = field(default_factory=list)
    execute_rate_history: List[float] = field(default_factory=list)
    all_reports: List[CandidateReport] = field(default_factory=list)

    # --- the run's return, recorded before the boundary carry (module docstring) ---
    #: What `select.final_artifact` resolved to at the end of the last completed
    #: iteration, set by `update()` before `apply_carry`. Not a carry slot: the
    #: carry must not be able to drop it, that is its whole purpose. Checkpointed
    #: with every other field, so a resumed leg returns what its predecessor
    #: had; None on a state that never completed an iteration, in which case
    #: `final_artifact` reads the live slots as it always did.
    returned: Optional[CandidateReport] = None

    def resolve_final_artifact(self, rule: str) -> Optional[CandidateReport]:
        """`select.final_artifact` applied to the LIVE slots, right now.

        The one place the four rules are spelled out (`FINAL_ARTIFACT_RULES`),
        used by `bird.final_artifact`, by `update()` to record `returned`, and
        by the post phases, which cannot import the repo-root `bird.py` and
        would otherwise need their own copy of this table.
        Unknown rule -> None; the caller validates against the enum.
        """
        if rule in ("global_best", "all_survivors"):
            return self.best
        if rule == "chain_end":
            return self.latest
        if rule == "archive_best":
            cells = [c for c in self.archive.values() if c.report is not None]
            return max(cells, key=lambda c: c.fitness).report if cells else None
        return None

    def final_artifact(self, rule: str) -> Optional[CandidateReport]:
        """What the run returns: the record `update()` wrote before the boundary
        carry when there is one, else the live slots (a state built by hand, or
        one that never completed an iteration). The two agree whenever the slot
        the rule reads survived the carry; when it did not, the record is the
        only correct answer."""
        if self.returned is not None:
            return self.returned
        return self.resolve_final_artifact(rule)

    def apply_carry(self, carry: List[str]) -> None:
        """Clear every carry slot NOT named in `loop.carry`.

        Called at EVERY iteration boundary -- the end of `update()` on a round
        that produced a selection, and `run_iteration`'s total-failure return
        on one that did not -- so a config that omits a slot truly does not
        have it. This is
        what makes `history_mode: none` mean none.
        """
        keep = {CARRY_SLOTS[c] for c in carry if c in CARRY_SLOTS}
        for slot, attr in CARRY_SLOTS.items():
            if attr in keep:
                continue
            # Companions clear with their owner: dropping the archive drops the
            # running descriptor stats its coordinates were binned against.
            for name in (attr,) + CARRY_COMPANIONS.get(slot, ()):
                current = getattr(self, name)
                if isinstance(current, list):
                    setattr(self, name, [])
                elif isinstance(current, dict):
                    setattr(self, name, {})
                else:
                    setattr(self, name, None)

    # -- serialisation (checkpoint every iteration; inner-loop runs are dear) --

    def summary(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "restart": self.restart,
            "best_id": self.best.cand_id if self.best else None,
            "best_fitness": self.best.fitness if self.best else None,
            "latest_id": self.latest.cand_id if self.latest else None,
            "returned_id": self.returned.cand_id if self.returned else None,
            "n_dialogue": len(self.dialogue),
            "n_preferences": len(self.preferences),
            "n_trajectories": len(self.trajectory_store),
            "n_subtasks": len(self.subtasks),
            "n_failures_remembered": len(self.failure_memory),
            "archive_occupied": len(self.archive),
            # The WHOLE curriculum, not a count. Every other slot here is
            # summarised to a length because the thing itself is large and is
            # written elsewhere; the curriculum is written nowhere else, and its
            # per-stage trace is the evidence the axis exists to produce.
            "curriculum": self.curriculum.summary() if self.curriculum else None,
            # Non-root nodes, i.e. the release's `sim_times`: reward functions
            # in the tree. `records/iterNN.json` carries the root as a row too,
            # so its `tree` list is one longer than this count when non-empty.
            "n_tree_nodes": sum(1 for n in self.tree.values() if n.parent_id is not None),
            "tree_depth": max((n.depth for n in self.tree.values()), default=0),
            "fitness_history": self.fitness_history,
            "execute_rate_history": self.execute_rate_history,
        }

    def to_json(self) -> str:
        def _enc(o: Any) -> Any:
            if is_dataclass(o):
                return asdict(o)
            if isinstance(o, tuple):
                return list(o)
            return str(o)

        return json.dumps(self.summary(), indent=2, default=_enc)
