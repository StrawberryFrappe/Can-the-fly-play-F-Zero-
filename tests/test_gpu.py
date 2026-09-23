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
    tgt = rng.uniform(0, 80, (4, 8)).astype(np.float32)
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
