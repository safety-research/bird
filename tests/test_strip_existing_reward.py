"""`generate.context.strip_existing_reward` must actually strip the reward.

WHY THIS FILE EXISTS. On `mt10_window-open-v3` a stripper that scans by indent
from the `def` line writes its `[reference reward withheld ...]` marker and then
emits the entire reward body immediately below it -- `TARGET_RADIUS = 0.05` and
`reward_utils.hamacher_product(...)` verbatim, the 5 cm tolerance that is
supposed to be withheld as the metric. A false assurance is worse than no
stripping, because the marker is the thing a reader greps for.

The cause: a MULTI-LINE signature closes on a line indented to the same level
as the `def`, so an indent scan stops at `) -> tuple[...]:` and never reaches
the body. Pendulum and Acrobot write single-line signatures, so tests on them
alone pass. That asymmetry is the point: the bug is invisible on the simple
envs and active on Meta-World.

It also matters ASYMMETRICALLY ACROSS METHODS, which is what makes it a
comparison bug rather than a leak. `generate.context.env_spec: full_source`
(eureka, eureka_no_evolution, rda, zeroshot) would receive the reward;
`pythonic_class_abstraction` (card, both text2reward) helper names only;
`state_action_api_stub` (gt, limen) nothing. A cross-method table built on
that would rank how much of the answer each prompt handed over.
"""
import pytest

from bird.components import generation


class _Env:
    pass


class _Ctx:
    env = _Env()


MULTILINE = '''class SawyerWindowOpenEnvV3:
    def compute_reward(
        self,
        actions: np.ndarray,
        obs: np.ndarray,
    ) -> tuple[float, float, float]:
        TARGET_RADIUS: float = 0.05
        reward = 10 * reward_utils.hamacher_product(reach, in_place)
        return reward, tcp_to_obj, in_place

    def evaluate_state(self):
        return 1
'''

SINGLELINE = '''class Pendulum:
    def compute_reward(self, s, a, s2):
        return -abs(s[0]) - 0.1 * a[0] ** 2

    def keep_me(self):
        return 2
'''

#: Tokens that must never survive stripping. These are the specific strings a
#: failed strip puts in front of the generator.
FORBIDDEN = ("hamacher_product", "TARGET_RADIUS", "reward_utils.tolerance",
             "-abs(s[0])")


@pytest.mark.parametrize("src,neighbour", [(MULTILINE, "def evaluate_state"),
                                           (SINGLELINE, "def keep_me")],
                         ids=["multiline_signature", "single_line_signature"])
def test_the_reward_body_does_not_survive(src, neighbour):
    out = generation._strip_reward(_Ctx(), src)
    leaked = [t for t in FORBIDDEN if t in out]
    assert not leaked, f"reward body leaked through the stripper: {leaked}"
    assert generation._WITHHELD.strip() in out, "no withheld marker was written"
    assert neighbour in out, "stripping ate a method that is not the reward"


def test_the_marker_is_never_written_without_the_strip():
    """The failure mode is marker-present-and-body-present, so assert the pair.

    A test that only checked for the marker would pass with the bug in place.
    """
    out = generation._strip_reward(_Ctx(), MULTILINE)
    marker_at = out.index(generation._WITHHELD.strip())
    after = out[marker_at:]
    assert "hamacher_product" not in after, (
        "the withheld marker is immediately followed by the withheld reward")
