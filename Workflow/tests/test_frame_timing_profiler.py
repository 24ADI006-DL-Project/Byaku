import pytest

from Backend.frame_timing_profiler import FrameProfiler, profile_stage


def test_profiler_summarizes_samples_and_metadata():
    profiler = FrameProfiler()
    profiler.record("decode", 3)
    profiler.record("decode", 1)
    profiler.record("decode", 2)
    profiler.add_metadata("processor_shape", [1, 3, 364, 364])

    result = profiler.summary()

    assert result == {
        "stages": {
            "decode": {
                "count": 3,
                "mean_ms": 2,
                "median_ms": 2,
                "min_ms": 1,
                "max_ms": 3,
            }
        },
        "metadata": {"processor_shape": [1, 3, 364, 364]},
    }


def test_profiler_returns_independent_copies_of_stage_samples():
    profiler = FrameProfiler()
    profiler.record("depth", 4)
    profiler.record("depth", 7)

    samples = profiler.samples()
    samples["depth"].append(99)

    assert profiler.samples() == {"depth": [4.0, 7.0]}


@pytest.mark.parametrize("stage", ["", "  "])
def test_profiler_rejects_empty_stage_name(stage):
    with pytest.raises(ValueError, match="stage"):
        FrameProfiler().record(stage, 1)


@pytest.mark.parametrize("elapsed_ms", [-1, float("inf"), float("nan")])
def test_profiler_rejects_invalid_duration(elapsed_ms):
    with pytest.raises(ValueError, match="finite and non-negative"):
        FrameProfiler().record("decode", elapsed_ms)


def test_measure_records_even_when_stage_raises_and_clear_resets():
    profiler = FrameProfiler()

    with pytest.raises(RuntimeError, match="stage failed"):
        with profiler.measure("inference"):
            raise RuntimeError("stage failed")

    assert profiler.summary()["stages"]["inference"]["count"] == 1
    profiler.clear()
    assert profiler.summary() == {"stages": {}, "metadata": {}}


def test_disabled_profile_stage_is_a_no_op_context():
    with profile_stage(None, "disabled"):
        pass
