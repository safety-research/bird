"""The byte caps must fit the policy blob a LARGE learner network produces,
not the one the surrogate backends produce.

WHY A TEST AND NOT A COMMENT. A cap sized on mt10 (obs 39, act 4: a 3.30 MB
policy blob, at which 64 entries is 211 MB and `_STORE_LIMIT` is the binding
bound) can carry a comment that is true in every word and still be sized for
the wrong shape: a larger network's policy blob is 304.48 MiB, 92x larger, and
at that size a 512 MiB cap holds ONE. Eviction is oldest-first, so the entry
lost first is the INCUMBENT -- the one the run cannot lose. The failure is
silent: the payload still ships and the worker still trains, from the wrong
policy.

A comment cannot fail. This can, and it fails on the thing that actually
matters -- the ratio between the cap and a REAL blob -- rather than on the
constant's value, so re-tuning the cap for a new shape means changing the
measurement here and saying where it came from.

THE BLOB SIZE IS MEASURED: a K=8 FastTD3/SimbaV2 policy blob, serialised, is
319,265,762 B. Measure it rather than estimate it: a 284.45 MiB figure taken
from a code comment gets the SIGN of the margin wrong.
"""

from __future__ import annotations

from bird.components import training

#: Measured, not assumed. See the module docstring.
LARGE_BLOB_BYTES = 319_265_762

#: Worst case for one run: 8 candidates x 5 iterations = 40 candidate blobs,
#: plus the incumbent that must survive all of them.
WORST_CASE_BLOBS = 41


def test_the_policy_cap_holds_a_whole_run_of_large_blobs():
    """41 blobs must fit, so the cap never binds and the incumbent never goes.

    Asserted against the cap as a RATIO to a measured blob rather than against
    its literal value: a cap that merely "looks big" can survive a 92x change
    in what it holds.
    """
    need = LARGE_BLOB_BYTES * WORST_CASE_BLOBS
    assert training._POLICY_STORE_MAX_BYTES >= need, (
        f"_POLICY_STORE_MAX_BYTES={training._POLICY_STORE_MAX_BYTES} holds "
        f"{training._POLICY_STORE_MAX_BYTES // LARGE_BLOB_BYTES} large blobs; a run "
        f"produces {WORST_CASE_BLOBS} (40 candidates + the incumbent) and eviction is "
        f"oldest-first, so the incumbent is what goes. Need >= {need}.")
    assert training._REPLAY_STORE_MAX_BYTES >= need, (
        "the replay cap is sized for the same run and must not be the tighter of the two")


def test_the_entry_cap_does_not_bind_before_the_byte_cap():
    """Two bounds, and neither may be the one that quietly evicts.

    `_STORE_LIMIT` is a count and the byte cap is a size; a run that fits the
    bytes and not the count evicts just as silently.
    """
    assert training._STORE_LIMIT > WORST_CASE_BLOBS, (
        f"_STORE_LIMIT={training._STORE_LIMIT} binds before {WORST_CASE_BLOBS} blobs are "
        "stored, so the byte cap is not the operative bound and raising it alone is not enough")


def test_the_old_cap_would_have_failed_this():
    """The negative control: the test must reject the values it replaces.

    A guard that passes on both the old and the new constant is not testing
    the change.

    BOTH rejected values are pinned, and the SECOND is the one that matters.
    512 MiB is obviously sized for mt10, where it is right. 2689597440 is THE
    PLAUSIBLE WRONG ANSWER: it looks as though it was chosen for this workload,
    and it is right against the figure it was sized on -- but that figure
    (284.45 MiB) CAME FROM A COMMENT rather than from a measurement. A reader is
    far likelier to mistake that value for sufficient than to mistake 512 MiB,
    so it is the one a control has to name.

    Driving a superseded value by hand once protects no future reader. Both
    are encoded here for the same reason the caps are pinned by ratio rather
    than by literal.
    """
    need = LARGE_BLOB_BYTES * WORST_CASE_BLOBS
    for label, superseded, holds in (("a cap sized for mt10", 512 * 1024 * 1024, 1),
                                     ("a cap sized from a comment's figure", 2_689_597_440, 8)):
        assert superseded < need, (
            f"{label} ({superseded}) is not actually superseded by this change: it already "
            f"holds the {WORST_CASE_BLOBS} blobs a run produces, so the premise here is wrong")
        assert superseded // LARGE_BLOB_BYTES == holds, (
            f"{label} holds {superseded // LARGE_BLOB_BYTES} large blobs, not {holds}; "
            "the blob measurement has moved and every number in this file needs re-deriving")
