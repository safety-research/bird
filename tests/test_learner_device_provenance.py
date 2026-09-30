"""A GPU job that silently became a CPU job must not reach a report.

THE FAILURE MODE. A GPU job can train entirely on the CPU with a GPU idle
beside it when two things line up:

* `bird/components/fasttd3.py`'s `train.hyperparameters.device` defaults to
  `"cpu"`, and nothing in the config sets it;
* when `cuda` *is* asked for and is unavailable, the backend downgrades to
  cpu with a `log.warning` -- and the downgrade also disables AMP and
  `torch.compile`, both derived from `device.type == "cuda"` two lines later.

THE PART WORTH REMEMBERING is that the evidence already exists: the seed row
carries `device` -- the `torch.device` the networks are constructed on. The
guard is mostly *comparing that field against intent*, and only secondarily
adding new fields.
"""

from __future__ import annotations

import pytest

from bird.components import training as T
from bird.types import Candidate, TrainResult


class _Cfg:
    def __init__(self, device):
        self._d = {"train.hyperparameters": {"device": device} if device else {}}

    def get(self, key, default=None):
        return self._d.get(key, default)


def _result(*devices):
    cand = Candidate(cand_id="c0000", iteration=0, reward_code="")
    r = TrainResult(cand_id="c0000", candidate=cand)
    r.seed_metrics = [{"seed": i, "device": d} for i, d in enumerate(devices)]
    return r


def test_an_explicit_cuda_request_that_ran_on_cpu_is_a_mismatch():
    reason = T.learner_device_mismatch(_Cfg("cuda"), _result("cpu"))
    assert reason, "cuda was asked for and the learner ran on cpu: the exact failure shape"
    assert "cpu" in reason and "cuda" in reason
    # The reason must say WHY it matters, not just that it happened: whoever
    # reads it is deciding whether the run must be repeated.
    assert "AMP" in reason or "compile" in reason


def test_a_cuda_request_honoured_is_no_mismatch():
    assert T.learner_device_mismatch(_Cfg("cuda"), _result("cuda:0")) == ""
    assert T.learner_device_mismatch(_Cfg("cuda:1"), _result("cuda:1")) == ""


def test_auto_keeps_its_fallback():
    """`auto` means "a GPU if there is one", so cpu is its correct answer.

    Refusing here would fail every laptop and every CI run, and would make
    the key useless for the thing it exists for.
    """
    assert T.learner_device_mismatch(_Cfg("auto"), _result("cpu")) == ""


def test_the_default_cpu_config_is_not_a_mismatch():
    """Only an EXPLICIT cuda request is checked. A config that asked for cpu
    and got cpu is not a defect -- it is the repo default, and most of the
    suite runs that way."""
    assert T.learner_device_mismatch(_Cfg("cpu"), _result("cpu")) == ""
    assert T.learner_device_mismatch(_Cfg(None), _result("cpu")) == ""


def test_a_backend_that_records_no_device_is_not_accused():
    """Silence is not evidence of a mismatch.

    The surrogate backends write no `device` on their seed rows. Refusing on
    absence would fail every mock run, which is the whole tester tier.
    """
    cand = Candidate(cand_id="c0000", iteration=0, reward_code="")
    r = TrainResult(cand_id="c0000", candidate=cand)
    r.seed_metrics = [{"seed": 0}]
    assert T.learner_device_mismatch(_Cfg("cuda"), r) == ""
    r.seed_metrics = []
    assert T.learner_device_mismatch(_Cfg("cuda"), r) == ""


def test_one_cuda_seed_among_several_is_enough_to_pass():
    """A mixed result is not the failure this guards. The defect is a job
    that ran ENTIRELY on cpu while claiming a GPU; a mix means something
    stranger and is not silently a cpu run."""
    assert T.learner_device_mismatch(_Cfg("cuda"), _result("cpu", "cuda:0")) == ""


def test_a_stub_cfg_does_not_raise():
    """Provenance may never be the thing that fails a training."""
    class _Bad:
        def get(self, *a, **k):
            raise RuntimeError("no config here")

    assert T.learner_device_mismatch(_Bad(), _result("cpu")) == ""


def test_the_gpu_fields_name_their_instrument_and_need_no_threshold():
    """Two instruments answer different questions; only one is evidence.

    `torch.cuda.max_memory_allocated` counts TENSORS and reads exactly 0 on
    cpu, so non-zero is the whole test. `nvidia-smi`'s memory figure shows the
    ~600 MiB CUDA context even when nothing trains, so non-zero there proves
    nothing -- which is why the field names the instrument rather than saying
    `gpu_peak_mib` and leaving a reader to guess.

    NO MiB FLOOR, deliberately: peak allocation scales with steps -- the
    replay buffer caps at `iters + 1` rows -- so a short job that trained
    genuinely on a GPU can sit below any floor set from a long one. A floor
    here is strictly worse than none: it fails real GPU jobs and catches
    nothing a non-zero test misses.
    """
    import inspect
    from bird.components import fasttd3 as FT

    src = inspect.getsource(FT)
    assert '"gpu_peak_alloc_torch_mib"' in src, (
        "the field must name its instrument: an unqualified gpu_peak_mib "
        "reads as nvidia-smi memory, which is ~600 MiB of CUDA context on an "
        "idle card and therefore not evidence of anything")
    assert "max_memory_allocated" in src
    # AND THE UTILISATION FIELDS MUST NAME THEIR INSTRUMENT TOO -- the same
    # contract. One instantaneous sample is not a summary: a real GPU job can
    # read 0% on eight consecutive samples, and on a cold JIT backend a single
    # mid-chunk sample can land inside the compile window -- 0% recorded on a
    # run that used the GPU for its whole stepping phase.
    #
    # So the module takes several samples and writes three fields that name
    # their instrument completely: which statistic (`max`/`mean`) and over HOW
    # MANY readings (`n_samples`), so a max over one sample cannot be read as a
    # max over four. A singular point-sample field must NOT appear alongside
    # them: one row carrying both a point sample and an aggregate under two
    # names for the same quantity is the hazard, not the cure.
    for field in ('"gpu_utilization_max_pct"', '"gpu_utilization_mean_pct"',
                  '"gpu_utilization_n_samples"'):
        assert field in src, (
            f"{field} is missing: the utilisation fields must say which "
            "statistic over how many samples, or a reader takes one "
            "instantaneous reading for a summary of the run")
    assert '"gpu_utilization_point_pct"' not in src, (
        "a singular point-sample field is in fasttd3 beside the "
        "aggregates; one row must not carry two names for one quantity")
    doc = inspect.getdoc(FT._gpu_utilization_percent) or ""
    assert "NOT evidence of idleness" in doc
    # THE DOCSTRING MUST NAME THE INSTRUMENT AND THE CADENCE, not just warn
    # about the reading: a reader who finds `gpu_utilization_max_pct` on a row
    # needs to know it is `nvidia-smi` sampled several times DURING the run,
    # because that is what makes a max meaningful and a zero uninformative.
    assert "nvidia-smi" in doc, "the docstring must name the instrument"
    assert "_GPU_UTIL_SAMPLES" in doc, (
        "the docstring must say how many samples a training takes; a max over "
        "an unstated number of readings is not a summary of anything")
    # AND THE DENOMINATOR IS CHECKED ON THE RETURNED DICT, NOT THE SOURCE TEXT.
    # The literal appears TWICE in `row_fields`'s source -- once in the
    # empty-path early return and once in the real one -- so a scan of the
    # source would stay green with the key renamed in the statistics that are
    # actually reported: the empty-path copy satisfies it. Source-based
    # assertions fail quietly; the NAMING half of a guard can stay green while
    # the property is gone.
    #
    # Two behavioural cases, because they are two properties: a max must
    # arrive with the count it is over, and a sampler that read nothing must
    # say so rather than report statistics over zero readings.
    sampler = FT._UtilSampler.__new__(FT._UtilSampler)
    sampler.samples = [33, 58, 41]
    row = sampler.row_fields()
    assert row["gpu_utilization_n_samples"] == 3, (
        "the denominator must travel with the statistic IN THE ROW: a max "
        "over one sample and a max over four are different claims")
    assert row["gpu_utilization_max_pct"] == 58 and row["gpu_utilization_mean_pct"] == 44.0, row

    empty = FT._UtilSampler.__new__(FT._UtilSampler)
    empty.samples = []
    erow = empty.row_fields()
    assert erow["gpu_utilization_n_samples"] == 0, erow
    assert erow["gpu_utilization_max_pct"] is None and erow["gpu_utilization_mean_pct"] is None, (
        "a sampler that read nothing must report None statistics, never 0 -- "
        "'could not tell' and 'the GPU was idle' are the two facts this "
        "family exists to keep apart")


def test_nothing_decides_anything_from_the_utilisation_sample():
    """The fold's evidence is learner_device and tensor allocation, never
    utilisation -- a 0% reading is "no evidence", and a rule that treated it
    as idleness would fail exactly the jobs that were working."""
    import inspect
    from bird.components import training as TT

    body = inspect.getsource(TT.learner_device_mismatch)
    assert "utilization" not in body and "utilisation" not in body


def test_the_utilisation_query_is_scoped_to_this_jobs_gpu(monkeypatch):
    """GPU 0 on a shared node is very often somebody else's job.

    Reporting their utilisation beside our seed row is worse than reporting
    none: it is a number about the wrong machine wearing our run's name.
    """
    from bird.components import fasttd3 as FT

    seen = {}

    class _Out:
        returncode = 0
        stdout = "42\n"

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return _Out()

    monkeypatch.setattr(FT.subprocess, "run", fake_run)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,4")
    assert FT._gpu_utilization_percent() == 42
    assert "-i" in seen["cmd"] and "3" in seen["cmd"], (
        f"the query is not scoped to this job's device: {seen['cmd']}")

    # Unset means we are not scoped and GPU 0 really is ours.
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    FT._gpu_utilization_percent()
    assert "-i" not in seen["cmd"]


def test_an_unusable_nvidia_smi_reads_as_no_evidence_not_as_zero(monkeypatch):
    """None, never 0. "Could not tell" and "was idle" are different facts and
    a reader of the seed row must not read the first as the second."""
    from bird.components import fasttd3 as FT

    class _Fail:
        returncode = 9
        stdout = ""

    monkeypatch.setattr(FT.subprocess, "run", lambda *a, **k: _Fail())
    assert FT._gpu_utilization_percent() is None

    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(FT.subprocess, "run", boom)
    assert FT._gpu_utilization_percent() is None

    class _Junk:
        returncode = 0
        stdout = "not a number\n"

    monkeypatch.setattr(FT.subprocess, "run", lambda *a, **k: _Junk())
    assert FT._gpu_utilization_percent() is None


def test_the_param_device_helper_never_raises():
    """Provenance may not be the thing that fails a training."""
    from bird.components import fasttd3 as FT

    assert FT._param_device({}) == "unknown"
    assert FT._param_device({"actor": object()}) == "unknown"


def test_param_device_reports_the_WHOLE_set_not_the_first_parameter():
    """A critic left on cpu must be visible, not averaged away.

    `fasttd3` builds the actor and the critics in separate calls, each with
    its own `device=`, so there are at least two independent chances to get
    it wrong -- and reading the actor's FIRST parameter looks at neither the
    critic nor the rest of the actor. The production field reports the device
    SET over all parameters, so it is at least as strict as any bench that
    checks the learner's placement.
    """
    from bird.components import fasttd3 as FT

    class _P:
        def __init__(self, dev):
            self.device = dev

    class _Net:
        def __init__(self, *devs):
            self._p = [_P(d) for d in devs]

        def parameters(self):
            return iter(self._p)

    # Everything agrees: one device, reported plainly.
    agent = {"actor": _Net("cuda:0"), "qnet": _Net("cuda:0"),
             "qnet_target": _Net("cuda:0")}
    assert FT._param_device(agent) == "cuda:0"

    # A critic left behind: VISIBLE, not hidden by the actor agreeing.
    split = {"actor": _Net("cuda:0"), "qnet": _Net("cpu"),
             "qnet_target": _Net("cuda:0")}
    assert FT._param_device(split) == "cpu, cuda:0", (
        "a critic on cpu passed as a clean cuda run -- exactly what reading "
        "the actor's first parameter cannot see")

    # A later parameter of the ACTOR itself, too.
    inner = {"actor": _Net("cuda:0", "cpu")}
    assert FT._param_device(inner) == "cpu, cuda:0"

    # Provenance never raises.
    assert FT._param_device({}) == "unknown"
    assert FT._param_device({"actor": object()}) == "unknown"


def test_unknown_provenance_is_not_treated_as_a_cpu_run():
    """Neither pass nor fail. Refusing on `unknown` would fail a healthy run
    whose reporting broke -- the mirror of the defect this guards, and the
    worse trade, since a real cpu run at least leaves `cpu` in the artifact.
    """
    cand = Candidate(cand_id="c0000", iteration=0, reward_code="")
    r = TrainResult(cand_id="c0000", candidate=cand)
    r.seed_metrics = [{"seed": 0, "device": "unknown"}]
    assert T.learner_device_mismatch(_Cfg("cuda"), r) == ""

    # But a row that genuinely says cpu still fails, alongside an unknown one.
    r.seed_metrics = [{"seed": 0, "device": "unknown"}, {"seed": 1, "device": "cpu"}]
    assert T.learner_device_mismatch(_Cfg("cuda"), r)
