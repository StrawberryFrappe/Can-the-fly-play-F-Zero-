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
