"""Tests for the DisturbancePlan helper."""
import pytest

from cubli_mpc.sim.disturbance import DisturbancePlan, StepDisturbance


def test_step_disturbance_active_within_window():
    s = StepDisturbance(magnitude=0.05, start_t=1.0, duration=0.5)
    assert s.at(0.9) == 0.0
    assert s.at(1.0) == 0.05
    assert s.at(1.4) == 0.05
    assert s.at(1.5) == 0.0  # exclusive at end
    assert s.at(2.0) == 0.0


def test_plan_sums_multiple_steps():
    plan = DisturbancePlan(steps=[
        StepDisturbance(magnitude=0.05, start_t=1.0, duration=0.5),
        StepDisturbance(magnitude=-0.02, start_t=1.2, duration=0.5),
    ])
    # 1.0–1.2: only the first step active
    assert plan.at(1.1, dt=0.01) == pytest.approx(0.05)
    # 1.2–1.5: both overlap, sum to 0.03
    assert plan.at(1.3, dt=0.01) == pytest.approx(0.03)
    # 1.5–1.7: only the second
    assert plan.at(1.6, dt=0.01) == pytest.approx(-0.02)


def test_empty_plan_returns_zero():
    plan = DisturbancePlan()
    assert plan.at(0.0, dt=0.01) == 0.0
    assert plan.at(123.0, dt=0.01) == 0.0
    assert plan.is_active is False


def test_is_active_flag():
    assert DisturbancePlan().is_active is False
    assert DisturbancePlan(
        steps=[StepDisturbance(0.0, 0.0, 0.5)]).is_active is False
    assert DisturbancePlan(
        steps=[StepDisturbance(0.1, 0.0, 0.0)]).is_active is False
    assert DisturbancePlan(
        steps=[StepDisturbance(0.1, 0.0, 0.5)]).is_active is True
    assert DisturbancePlan(
        impulse_magnitude=0.1, impulse_rate_per_s=0.5).is_active is True
    assert DisturbancePlan(
        impulse_magnitude=0.1, impulse_rate_per_s=0.0).is_active is False


def test_impulse_stream_fires_at_expected_rate():
    """With rate=10/s and dt=0.01, p_fire = 0.1 per tick. Over 10000 ticks
    we expect ~1000 impulses; tolerate ±20%."""
    plan = DisturbancePlan(
        impulse_magnitude=0.1, impulse_rate_per_s=10.0, seed=0,
    )
    n_fires = 0
    for i in range(10_000):
        tau = plan.at(t=i * 0.01, dt=0.01)
        if tau != 0.0:
            n_fires += 1
    assert 800 <= n_fires <= 1200


def test_impulse_stream_is_seeded():
    plan_a = DisturbancePlan(
        impulse_magnitude=0.1, impulse_rate_per_s=10.0, seed=42,
    )
    plan_b = DisturbancePlan(
        impulse_magnitude=0.1, impulse_rate_per_s=10.0, seed=42,
    )
    seq_a = [plan_a.at(i * 0.01, 0.01) for i in range(200)]
    seq_b = [plan_b.at(i * 0.01, 0.01) for i in range(200)]
    assert seq_a == seq_b


def test_impulse_signs_are_balanced():
    """Over a long run the impulse stream should fire roughly equal numbers
    of positive and negative impulses."""
    plan = DisturbancePlan(
        impulse_magnitude=0.1, impulse_rate_per_s=10.0, seed=7,
    )
    pos = neg = 0
    for i in range(10_000):
        tau = plan.at(i * 0.01, 0.01)
        if tau > 0:
            pos += 1
        elif tau < 0:
            neg += 1
    assert abs(pos - neg) < 0.2 * (pos + neg)


def test_reset_resyncs_impulse_rng():
    plan = DisturbancePlan(
        impulse_magnitude=0.1, impulse_rate_per_s=10.0, seed=3,
    )
    seq_a = [plan.at(i * 0.01, 0.01) for i in range(100)]
    plan.reset()
    seq_b = [plan.at(i * 0.01, 0.01) for i in range(100)]
    assert seq_a == seq_b


def test_steps_and_impulses_compose():
    plan = DisturbancePlan(
        steps=[StepDisturbance(0.05, 0.0, 1.0)],
        impulse_magnitude=0.1, impulse_rate_per_s=10.0, seed=1,
    )
    # Inside the step window, output is step_value + (0 or +/-impulse).
    seen_step_only = False
    seen_combined = False
    for i in range(100):
        tau = plan.at(t=i * 0.005, dt=0.01)
        if tau == 0.05:
            seen_step_only = True
        elif abs(tau - 0.05) >= 0.09:
            # |tau| close to 0.05 + 0.1 or 0.05 - 0.1 (i.e. 0.15 or -0.05)
            seen_combined = True
    assert seen_step_only
    assert seen_combined
