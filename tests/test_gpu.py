import numpy as np
import pytest
import scipy.sparse as sp

from flyzero.brain import Brain, LIFParams

cp = pytest.importorskip("cupy")
try:
    cp.cuda.runtime.getDeviceCount()
except Exception:  # pragma: no cover
    pytest.skip("no CUDA device", allow_module_level=True)

from flyzero.gpu import BatchBrain  # noqa: E402


def _weights(n=300, seed=2):
    rng = np.random.default_rng(seed)
    w = sp.random(n, n, density=0.05, random_state=seed, format="csr")
    w.data = rng.normal(40, 30, w.nnz)
    return w


def _counts_cpu(w, data=None, dt=0.25):
    b = Brain(w, LIFParams(dt=dt), backend="numba" if _has_numba() else "numpy")
    if data is not None:
        b.data[:] = data
    b.v[:40] = -40.0
    return b.run(50)


def _has_numba():
    try:
        import numba  # noqa: F401
        return True
    except ImportError:
        return False


def test_gpu_matches_cpu_without_noise():
    w = _weights()
    ref = _counts_cpu(w)
    gb = BatchBrain(w, LIFParams(dt=0.25), batch=3)
    gb.v[:, :40] = -40.0
    out = cp.asnumpy(gb.run(50))
    assert ref.sum() > 50
    for k in range(3):
        np.testing.assert_array_equal(out[k], ref)


def test_each_fly_has_its_own_plastic_synapses():
    w = _weights()
    cpu = Brain(w, LIFParams(dt=0.25))
    pos = np.flatnonzero(cpu.indices < 30)          # synapses onto neurons 0..29
    gb = BatchBrain(w, LIFParams(dt=0.25), batch=2, plastic_pos=pos)
    changed = cpu.data.copy()
    changed[pos] = np.abs(changed[pos]) * 3        # fly 1: much stronger excitation onto 0..29
    gb.pw[1] = cp.asarray(changed[pos])
    gb.v[:, :40] = -40.0
    out = cp.asnumpy(gb.run(50))
    np.testing.assert_array_equal(out[0], _counts_cpu(w))
    np.testing.assert_array_equal(out[1], _counts_cpu(w, changed))
    assert not np.array_equal(out[0], out[1])


def test_poisson_input_rate():
    w = sp.csr_matrix((10, 10), dtype=np.float32)
    gb = BatchBrain(w, LIFParams(dt=0.25), batch=2, seed=5)
    gb.set_input(np.arange(10), 100.0)
    c = cp.asnumpy(gb.run(1000))
    # each kick (68.75 mV) fires the neuron unless it's refractory: ~100 Hz minus refractory losses
    assert 70 < c.mean() < 105
    assert not np.array_equal(c[0], c[1])   # flies get different noise


def test_deep_update_matches_plain_formula():
    from flyzero import connectome as cx
    from flyzero.batch import BatchInstructDeep, deep_positions, plastic_positions

    conn = cx.synthetic()
    p1 = plastic_positions(conn)
    pos = np.union1d(p1, deep_positions(conn, p1))
    gb = BatchBrain(conn.weights, LIFParams(dt=0.25), batch=4, plastic_pos=pos)
    bi = BatchInstructDeep(gb, conn, fly_of_slot=[0, 0, 1, 1], eta_deep=1e-3)
    assert bi.n_deep > 0
    rng = np.random.default_rng(0)
    bi.trace[...] = cp.asarray(rng.uniform(0, 5, bi.trace.shape).astype(np.float32))
    bi.rate[...] = cp.asarray(rng.uniform(0, 60, bi.rate.shape).astype(np.float32))
    tgt = rng.uniform(0, 80, (4, len(bi.names))).astype(np.float32)
    learn = np.array([True, False, True, True])
    w0 = cp.asnumpy(gb.pw).copy()
    # expected: plain formula for the deep synapses (DN synapses checked elsewhere)
    sel, dsel = cp.asnumpy(bi.dn_sel), cp.asnumpy(bi.deep_sel)
    err = tgt[:, cp.asnumpy(bi.post_group)[sel]] - cp.asnumpy(bi.rate)[:, cp.asnumpy(bi.post_k)[sel]]
    delta = np.zeros((4, len(bi.l1)), np.float32)
    np.add.at(delta.T, cp.asnumpy(bi.dn_pre_l1), (w0[:, sel] / gb.p.w_syn * err).T)
    change = bi.eta_deep * cp.asnumpy(bi.trace)[:, cp.asnumpy(bi.deep_pre_k)] * delta[:, cp.asnumpy(bi.deep_post_l1)]
    per_fly = np.stack([change[0], change[2:].mean(0)])
    sign, wmax = cp.asnumpy(bi.sign)[dsel], cp.asnumpy(bi.wmax)[dsel]
    expected = sign * np.clip(sign * (w0[:, dsel] + per_fly[[0, 0, 1, 1]]), 0, wmax)
    bi._learn(cp.asarray(tgt), learn, bi.p.eta)
    np.testing.assert_allclose(cp.asnumpy(gb.pw)[:, dsel], expected, rtol=1e-4, atol=1e-5)
    assert np.abs(expected - w0[:, dsel]).max() > 1e-4


def test_intrinsic_plasticity_moves_bias_towards_target_and_round_trips():
    from flyzero import connectome as cx
    from flyzero.batch import BatchInstruct, plastic_positions

    conn = cx.synthetic()
    gb = BatchBrain(conn.weights, LIFParams(dt=0.25), batch=2, plastic_pos=plastic_positions(conn))
    gb.set_bias(conn.find("DNp09"), 7.8)
    bi = BatchInstruct(gb, conn, fly_of_slot=[0, 0], eta_bias=0.01, bias_limit=2.0)
    j = gb.slots(bi.dn)
    before = cp.asnumpy(gb.bias_ext[:, j]).copy()
    bi.rate[...] = 100.0                              # every DN far above its target
    tgt = np.zeros((2, len(bi.names)), np.float32)
    for _ in range(50):
        bi.step(cp.zeros((2, conn.n), cp.int32), tgt, np.array([True, False]), 16.6)
        bi.rate[...] = 100.0
    after = cp.asnumpy(gb.bias_ext[:, j])
    assert (after[0] < before[0]).all() and (after[0] >= before[0] - 2.0 - 1e-5).all()   # down, bounded
    np.testing.assert_allclose(after[1], after[0])    # same fly: every slot changes together
    st = bi.state(0)
    gb2 = BatchBrain(conn.weights, LIFParams(dt=0.25), batch=1, plastic_pos=plastic_positions(conn))
    gb2.set_bias(conn.find("DNp09"), 7.8)
    BatchInstruct(gb2, conn).load(st)
    np.testing.assert_allclose(cp.asnumpy(gb2.bias_ext[0, gb2.slots(bi.dn)]), after[0], atol=1e-5)


def test_consolidated_weights_average_and_swap_back():
    from flyzero import connectome as cx
    from flyzero.batch import BatchInstruct, plastic_positions

    conn = cx.synthetic()
    gb = BatchBrain(conn.weights, LIFParams(dt=0.25), batch=1, plastic_pos=plastic_positions(conn))
    bi = BatchInstruct(gb, conn, eta_bias=0.01)
    w0, b0 = cp.asnumpy(gb.pw).copy(), cp.asnumpy(gb.bias_ext).copy()
    bi.consolidate(0.5)                        # slow copy starts at the current weights
    gb.pw[...] *= 0.5
    bi.ib += 1.0
    gb.bias_ext[:, bi.dn_slots] += 1.0
    bi.consolidate(0.5)                        # slow = halfway between old and new
    fast_w, fast_b, fast_ib = cp.asnumpy(gb.pw).copy(), cp.asnumpy(gb.bias_ext).copy(), cp.asnumpy(bi.ib).copy()
    bi.swap_slow()
    np.testing.assert_allclose(cp.asnumpy(gb.pw), 0.75 * w0, rtol=1e-6)
    np.testing.assert_allclose(cp.asnumpy(bi.ib), fast_ib - 0.5, atol=1e-6)
    bi.swap_slow()
    np.testing.assert_allclose(cp.asnumpy(gb.pw), fast_w)
    np.testing.assert_allclose(cp.asnumpy(gb.bias_ext), fast_b)
    np.testing.assert_allclose(cp.asnumpy(bi.ib), fast_ib, atol=1e-6)


def test_tap_readout_duty_cycle():
    from flyzero import connectome as cx
    from flyzero.batch import BatchMotor
    from flyzero.motor import MotorParams

    m = BatchMotor(cx.synthetic(), 1, MotorParams(taps=True))
    def presses(steer_hz, n=200):
        m.reset()
        r = np.zeros(9, np.float32); r[0] = steer_hz   # DNa02 left above right: steer left
        return sum(m.buttons(r)["LEFT"] for _ in range(n)) / n
    assert presses(5.0) == 0.0                          # inside the threshold: no taps
    assert abs(presses(85.0) - 0.5) < 0.02              # (85 - 10) / 150 = half the frames
    assert presses(400.0) == 1.0                        # beyond the span: held
