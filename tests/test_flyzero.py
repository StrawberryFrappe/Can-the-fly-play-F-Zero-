import os

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from flyzero import connectome as cx
from flyzero.brain import Brain
from flyzero.cli import main
from flyzero.connectome import Connectome
from flyzero.motor import MotorReadout
from flyzero.vision import MotionEye


def test_driven_neuron_excites_its_target_after_the_delay():
    w = np.zeros((2, 2)); w[0, 1] = 100  # 100 synapses -> 27.5 mV kick
    b = Brain(sp.csr_matrix(w))
    b.set_input([0], 200.0)
    counts = b.run(200)
    assert counts[0] > 10 and counts[1] > 5
    # an inhibitory connection silences instead
    b = Brain(sp.csr_matrix(-w)); b.set_input([0], 200.0)
    assert b.run(200)[1] == 0


def test_spike_propagation_matches_dense_matrix():
    w = sp.random(50, 50, density=0.2, random_state=1, format="csr") * 10
    b = Brain(w)
    spiking = np.array([3, 7, 20, 49])
    out = np.zeros(50, np.float32)
    b._propagate(spiking, out)
    expected = np.asarray(w[spiking].sum(0)).ravel() * b.p.w_syn
    np.testing.assert_allclose(out, expected, rtol=1e-5)


def test_synthetic_connectome_has_everything_the_readout_needs():
    conn = cx.synthetic()
    motor = MotorReadout(conn)
    assert all(len(v) > 0 for v in motor.groups.values())
    assert len(conn.photoreceptors()) > 0


def motion_connectome():
    """T4/T5 a-d on both sides laid out on a 10x10 retinotopic grid."""
    rows = []
    gy, gx = np.mgrid[0:10, 0:10]
    for kind in ("T4", "T5"):
        for sub in "abcd":
            for side, sgn in (("left", 1), ("right", -1)):
                for x, y in zip(gx.ravel(), gy.ravel()):
                    rows.append(("optic", kind + sub, side, 100 + sgn * (10 + x), float(y), 0.0))
    neurons = pd.DataFrame(rows, columns=["super_class", "cell_type", "side", "pos_x", "pos_y", "pos_z"])
    neurons.insert(0, "root_id", np.arange(len(neurons)))
    neurons["cell_class"] = pd.NA
    neurons["hemibrain_type"] = pd.NA
    for col in ["super_class", "cell_class", "cell_type", "hemibrain_type", "side"]:
        neurons[col] = neurons[col].astype("string")
    return Connectome(sp.csr_matrix((len(neurons), len(neurons)), dtype=np.float32), neurons)


@pytest.mark.parametrize("direction", [1, -1])
def test_motion_eye_is_direction_selective(direction):
    conn = motion_connectome()
    eye = MotionEye(conn, (224, 256))
    t4 = {sub: conn.find("T4" + sub) for sub in "abcd"}
    total = {sub: 0.0 for sub in "abcd"}
    pos = {k: i for i, k in enumerate(eye.idx)}
    right_eye = conn.neurons["side"].to_numpy() == "right"
    for t in range(40):  # a bright bar sweeping horizontally over a dark screen
        frame = np.zeros((224, 256, 3), np.uint8)
        x = 20 + 5 * t if direction > 0 else 235 - 5 * t
        frame[:, x:x + 6] = 255
        rates = eye.see(frame)
        for sub, idx in t4.items():
            sel = [pos[i] for i in idx if right_eye[i]]
            total[sub] += rates[sel].sum()
    # on the right eye, rightward = front-to-back (a), leftward = back-to-front (b)
    if direction > 0:
        assert total["a"] > 3 * total["b"]
    else:
        assert total["b"] > 3 * total["a"]
    assert total["c"] < total["a"] + total["b"]


def test_play_mock_with_synthetic_brain(tmp_path):
    video = tmp_path / "mock.mp4"
    main(["play", "--game", "mock", "--connectome", "synthetic", "--frames", "30",
          "--video", str(video), "--log-every", "10"])
    assert video.stat().st_size > 0


@pytest.mark.skipif(not os.environ.get("FLYZERO_ROM"), reason="set FLYZERO_ROM to an F-Zero ROM")
def test_fzero_boots_and_reaches_the_race():
    from flyzero.games import FZero

    g = FZero(os.environ["FLYZERO_ROM"])
    frame = g.reset()
    for _ in range(200):
        frame = g.step({"B": True})
    assert frame.shape == (224, 256, 3) and frame.std() > 10


def test_numba_and_numpy_backends_agree_without_noise():
    pytest.importorskip("numba")
    rng = np.random.default_rng(3)
    w = sp.random(300, 300, density=0.05, random_state=2, format="csr")
    w.data = rng.normal(20, 15, w.nnz)
    out = []
    for backend in ("numpy", "numba"):
        b = Brain(w, backend=backend)
        b.v[:40] = -40.0  # kick-start a few neurons deterministically, no Poisson input
        out.append(b.run(50))
    np.testing.assert_array_equal(out[0], out[1])
