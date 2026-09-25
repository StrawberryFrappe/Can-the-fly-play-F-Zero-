import numpy as np
import pytest

from flyzero.pilot import RAM_HEADING, RAM_SPEED, RAM_X, RAM_Y, TURN, Pilot, PilotParams


def fake_ram(x, y, heading, speed):
    """Work RAM with the car at (x, y), heading in radians (atan2 convention), speed."""
    ram = np.zeros(0x20000, np.uint8)
    raw = int(round((heading + np.pi / 2) / (2 * np.pi) * TURN)) % TURN
    for addr, v in ((RAM_X, int(x)), (RAM_Y, int(y)), (RAM_HEADING, raw), (RAM_SPEED, int(speed))):
        ram[addr], ram[addr + 1] = v & 0xFF, v >> 8
    ram[0x00C9], ram[0x00CA] = 0x00, 0x08   # full energy
    return ram


def circle_line(n=400, r=2000.0, c=3000.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[c + r * np.cos(a), c + r * np.sin(a)], np.full(n, 1800.0)


def test_pursuit_gas_is_unchanged_by_the_gas_duty_cycle():
    """Style "pursuit": gas is 0 or 1 every frame, so B must be exactly gas > 0.5, as before."""
    pts, sp = circle_line()
    p = Pilot(pts, sp)
    rng = np.random.default_rng(0)
    for g in rng.integers(0, 2, 200).astype(float):
        b = p.buttons(None, np.array([0.0, 0.0, g, 0.0, 0.0], np.float32))
        assert b["B"] == (g > 0.5)


def test_owner_style_pulses_gas_at_the_requested_duty():
    pts, sp = circle_line()
    p = Pilot(pts, sp, PilotParams(style="owner"))
    presses = [p.buttons(None, np.array([0.0, 0.0, 0.6, 0.0, 0.0], np.float32))["B"] for _ in range(1000)]
    assert abs(np.mean(presses) - 0.6) < 0.01


def test_owner_style_corrects_small_errors_by_leaning_only():
    pts, sp = circle_line()
    p = Pilot(pts, sp, PilotParams(style="owner", kd=0.0))
    x, y = pts[0]
    tangent = np.arctan2(pts[1, 1] - pts[0, 1], pts[1, 0] - pts[0, 0])
    # almost on course: a small turn command -> lean, no D-pad
    s, lean, gas, _, _ = p.intent(fake_ram(x, y, tangent + 0.02, 1800))
    assert s == 0.0 and lean != 0.0 and gas == 1.0


def test_line_files_choose_the_style():
    pts, sp = circle_line()
    line = {"points": pts, "speed": sp, "style": np.array("owner"), "kp": np.array(5.0)}
    p = Pilot.from_line(line)
    assert p.p.style == "owner" and p.p.kp == 5.0
    with pytest.raises(AssertionError):
        Pilot(pts, sp, PilotParams(style="nonsense"))


def test_owner_clone_loads_from_a_line_file_and_drives():
    from flyzero.teacher import N_IN, OwnerClone

    pts, sp = circle_line()
    rng = np.random.default_rng(1)
    W = {"w0": rng.normal(0, 0.1, (128, N_IN)), "b0": np.zeros(128), "w1": rng.normal(0, 0.1, (128, 128)),
         "b1": np.zeros(128), "w2": rng.normal(0, 0.1, (8, 128)), "b2": np.zeros(8)}
    line = {"points": pts, "speed": sp, "style": np.array("clone"), **{f"clone_{k}": v for k, v in W.items()}}
    t = Pilot.from_line(line)
    assert isinstance(t, OwnerClone)
    x, y = pts[10]
    for i in range(10):
        b = t.buttons(fake_ram(x + i, y, 0.3, 1500))
    assert set(b) >= {"B", "LEFT", "RIGHT", "L", "R", "A"}
    x5 = t.intent(fake_ram(x, y, 0.3, 1500))
    assert x5.shape == (5,) and -1 <= x5[0] <= 1 and 0 <= x5[2] <= 1


def test_track_offset_sign_and_heading_error():
    from flyzero.teacher import Track

    pts = np.c_[np.arange(0, 6400, 16.0), np.full(400, 1000.0)]   # a straight line along +x
    tr = Track(pts)
    left, _ = tr.locate(fake_ram(800, 1100, 0.0, 1000))   # 100 units to the +y side
    right, _ = tr.locate(fake_ram(800, 900, 0.2, 1000))
    assert left[0] == pytest.approx(1.0, abs=0.05) and right[0] == pytest.approx(-1.0, abs=0.05)
    assert right[1] == pytest.approx(0.2, abs=0.01)


def test_soft_targets_are_distributions():
    from flyzero.cnn_dagger import soft_targets
    from flyzero.teacher import soft_intent

    x = np.array([[0.4, -1.0, 0.7, 0.0, 1.0], [-0.2, 0.0, 1.0, 0.0, 0.0]], np.float32)
    for t in (soft_targets(x), soft_intent(x)):
        assert np.allclose(t[:, 0:3].sum(1), 1) and np.allclose(t[:, 3:6].sum(1), 1)
        assert t[0, 2] == pytest.approx(0.4) and t[0, 4] == pytest.approx(1.0)


def test_cnn_mirror_swaps_left_and_right():
    torch = pytest.importorskip("torch")
    from flyzero.cnn_dagger import mirror

    xb = torch.arange(2 * 6 * 4 * 5, dtype=torch.float32).reshape(2, 6, 4, 5)
    tb = torch.tensor([[0.2, 0.8, 0.0, 0.5, 0.0, 0.5, 1.0, 0.0]] * 2)
    xm, tm = mirror(xb, tb)
    assert torch.equal(xm[..., 0], xb[..., -1])
    assert tm[0, 1] == 0.0 and tm[0, 2] == pytest.approx(0.8) and tm[0, 6] == 1.0


def test_teacher_features_online_match_the_dataset_rates():
    """The clone must see the same features when driving as when it was trained."""
    from flyzero.teacher import LAG, OwnerClone, Track, with_rates

    pts, sp = circle_line()
    rng = np.random.default_rng(2)
    rams = [fake_ram(pts[i, 0] + rng.normal(0, 20), pts[i, 1] + rng.normal(0, 20), rng.normal(0, 1), 1500)
            for i in range(0, 40)]
    tr, k, F = Track(pts), None, []
    for ram in rams:
        f, k = tr.locate(ram, k)
        F.append(f)
    offline = with_rates(np.array(F), np.zeros(len(F), int))
    c = OwnerClone(pts, {})
    online = np.array([c.features(ram) for ram in rams])
    assert np.allclose(offline, online, atol=1e-5)
    assert np.all(offline[:LAG, -2:] == 0)


def test_fitted_clone_matches_its_numpy_copy():
    torch = pytest.importorskip("torch")
    from flyzero.teacher import N_IN, build_net, fit, mlp_forward

    rng = np.random.default_rng(3)
    X = rng.normal(size=(64, N_IN)).astype(np.float32)
    T = np.zeros((64, 8), np.float32)
    T[:, 0] = T[:, 3] = 1
    W = fit(X, T, epochs=1)
    m = build_net()
    lin = [layer for layer in m.net if hasattr(layer, "weight")]
    with torch.no_grad():
        for i, layer in enumerate(lin):
            layer.weight.copy_(torch.tensor(W[f"w{i}"]))
            layer.bias.copy_(torch.tensor(W[f"b{i}"]))
        ref = m(torch.tensor(X)).numpy()
    assert np.allclose(mlp_forward(W, X), ref, atol=1e-4)
