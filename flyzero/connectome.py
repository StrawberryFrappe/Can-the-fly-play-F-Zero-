"""Loading the FlyWire whole-brain connectome (or a small synthetic stand-in).

The real data are the files used by Shiu et al. 2024, "A Drosophila computational
brain model reveals sensorimotor processing" (Nature), built on FlyWire
materialization 783:

* ``Completeness_783.csv``   - one row per neuron, index = FlyWire root id.
  Row order defines the neuron index used everywhere else.
* ``Connectivity_783.parquet`` - one row per connected pair, with
  ``Presynaptic_Index``, ``Postsynaptic_Index`` and ``Excitatory x Connectivity``
  (synapse count, signed by the presynaptic neurotransmitter).

plus the FlyWire neuron annotations (Schlegel et al. 2024), which give us cell
types, hemisphere and a 3D position for every neuron:

* ``Supplemental_file1_neuron_annotations.tsv``

``flyzero download`` fetches all three into the data directory.
"""

from __future__ import annotations

import os
import re
import shutil
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

DEFAULT_DATA_DIR = Path(os.environ.get("FLYZERO_DATA", Path.home() / ".cache" / "flyzero"))

SHIU = "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/"
ANNOT = (
    "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/"
    "supplemental_files/Supplemental_file1_neuron_annotations.tsv"
)
FILES = {
    "Completeness_783.csv": SHIU + "Completeness_783.csv",
    "Connectivity_783.parquet": SHIU + "Connectivity_783.parquet",
    "neuron_annotations.tsv": ANNOT,
}

# Photoreceptor cell types in FlyWire: R1-6 (lamina), R7/R8 (medulla) and their
# pale/yellow/dorsal-rim subtypes (R7p, R8y, R7d, ...).
PHOTORECEPTOR_RE = re.compile(r"^R[1-8]([a-z]|-6)?$|^R1-6$", re.IGNORECASE)


def download(data_dir: Path = DEFAULT_DATA_DIR, force: bool = False) -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = data_dir / name
        if dest.exists() and not force:
            print(f"  have {dest}")
            continue
        print(f"  fetching {url}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.rename(dest)
    return data_dir


@dataclass
class Connectome:
    """A connectome ready for simulation.

    ``weights`` is a CSR matrix (pre x post) of signed synapse counts.
    ``neurons`` has one row per neuron (in index order) with at least the
    columns ``root_id``, ``cell_type``, ``super_class``, ``side``,
    ``pos_x``, ``pos_y``, ``pos_z``.
    """

    weights: sp.csr_matrix
    neurons: pd.DataFrame
    name: str = "connectome"
    _type_cache: dict = field(default_factory=dict, repr=False)

    @property
    def n(self) -> int:
        return self.weights.shape[0]

    def find(self, cell_type: str, side: str | None = None) -> np.ndarray:
        """Indices of neurons whose cell type (or hemibrain type) matches exactly.

        A trailing ``*`` matches a prefix, e.g. ``DNg02*`` for DNg02_a ... DNg02_h."""
        key = cell_type.lower()
        if key not in self._type_cache:
            if key.endswith("*"):
                m = self.neurons["cell_type"].str.lower().str.startswith(key[:-1])
            else:
                m = self.neurons["cell_type"].str.lower().eq(key)
                if "hemibrain_type" in self.neurons:
                    m |= self.neurons["hemibrain_type"].str.lower().eq(key)
            self._type_cache[key] = m.to_numpy(dtype=bool, na_value=False)
        m = self._type_cache[key]
        if side is not None:
            m = m & self.neurons["side"].eq(side).to_numpy(dtype=bool, na_value=False)
        return np.flatnonzero(m)

    def photoreceptors(self) -> np.ndarray:
        types = self.neurons["cell_type"].fillna("")
        m = types.str.match(PHOTORECEPTOR_RE)
        if m.sum() == 0:  # annotation release without per-type labels for them
            cls = self.neurons["cell_class"].fillna("").str.lower()
            m = self.neurons["super_class"].eq("sensory") & cls.str.contains(
                "photoreceptor|retin|visual")
        return np.flatnonzero(m.to_numpy(dtype=bool, na_value=False))

    def summary(self) -> str:
        sc = self.neurons["super_class"].value_counts().head(8)
        return (
            f"{self.name}: {self.n:,} neurons, {self.weights.nnz:,} connections, "
            f"{int(abs(self.weights).sum()):,} synapses\n"
            + "\n".join(f"  {k:<16}{v:>8,}" for k, v in sc.items())
        )


def load_flywire(data_dir: Path = DEFAULT_DATA_DIR) -> Connectome:
    data_dir = Path(data_dir)
    missing = [f for f in FILES if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"missing {missing} in {data_dir}; run `flyzero download` first "
            "(or pass --connectome synthetic to try things out)"
        )

    comp = pd.read_csv(data_dir / "Completeness_783.csv", index_col=0)
    root_ids = comp.index.to_numpy(dtype=np.int64)
    n = len(root_ids)

    con = pd.read_parquet(
        data_dir / "Connectivity_783.parquet",
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
    )
    weights = sp.csr_matrix(
        (
            con["Excitatory x Connectivity"].to_numpy(np.float32),
            (con["Presynaptic_Index"].to_numpy(), con["Postsynaptic_Index"].to_numpy()),
        ),
        shape=(n, n),
    )
    del con

    ann = pd.read_csv(data_dir / "neuron_annotations.tsv", sep="\t", low_memory=False)
    # The annotation release carries ids from several materializations; use the
    # column that best matches the 783 ids of the connectivity data.
    id_cols = [c for c in ann.columns if c == "root_id" or c.startswith("root_")]
    best = max(id_cols, key=lambda c: pd.Index(ann[c]).isin(root_ids).sum())
    ann = ann.drop_duplicates(best).set_index(best)

    neurons = pd.DataFrame({"root_id": root_ids})
    for col in ["super_class", "cell_class", "cell_type", "hemibrain_type", "side", "top_nt",
                "pos_x", "pos_y", "pos_z"]:
        if col in ann:
            neurons[col] = ann[col].reindex(root_ids).to_numpy()
        else:
            neurons[col] = np.nan
    for col in ["super_class", "cell_class", "cell_type", "hemibrain_type", "side", "top_nt"]:
        neurons[col] = neurons[col].astype("string")
    return Connectome(weights, neurons, name="FlyWire v783")


# --------------------------------------------------------------------------
# Synthetic stand-in
# --------------------------------------------------------------------------

def synthetic(seed: int = 0, eye_size: int = 24, n_optic: int = 1200, n_central: int = 1500) -> Connectome:
    """A small, made-up "brain" with the same schema as the FlyWire data.

    It is NOT a fly brain: it exists so the whole pipeline can run offline and in
    tests. It has two retinotopic eyes, a random optic/central network in which
    each eye's lower lateral field excites the same side's turning neurons (so
    it steers towards the busier side, a crude stripe-fixation reflex) and the
    same descending neuron types that the motor readout looks for.
    """
    rng = np.random.default_rng(seed)
    rows = []

    def add(n, super_class, cell_type, side, pos):
        for i in range(n):
            rows.append((super_class, cell_type, side, *pos[i]))

    # eyes: a grid of photoreceptors per side; pos encodes the retinotopy
    gy, gx = np.mgrid[0:eye_size, 0:eye_size]
    for side, sgn in (("left", 1), ("right", -1)):
        pos = np.stack([100_000 + sgn * (20_000 + 400 * gx.ravel()),
                        40_000 + 400 * gy.ravel(),
                        np.full(gx.size, 2_000)], 1)
        add(gx.size, "sensory", "R7", side, pos)
    for side, sgn in (("left", 1), ("right", -1)):
        add(n_optic // 2, "optic", "Tm", side,
            np.c_[100_000 + sgn * rng.uniform(15_000, 35_000, n_optic // 2),
                  rng.uniform(40_000, 50_000, n_optic // 2), np.zeros(n_optic // 2)])
    add(n_central, "central", "CX", "center",
        np.c_[rng.normal(100_000, 5_000, n_central), rng.uniform(40_000, 50_000, n_central),
              np.zeros(n_central)])
    dn_types = {"DNa02": 4, "DNa01": 3, "DNg02_a": 2, "DNp09": 3, "MDN": 2, "DNp01": 1}
    for t, k in dn_types.items():
        for side, sgn in (("left", 1), ("right", -1)):
            add(k, "descending", t, side, np.c_[np.full(k, 100_000 + sgn * 3_000),
                                                np.full(k, 60_000), np.zeros(k)])

    neurons = pd.DataFrame(rows, columns=["super_class", "cell_type", "side", "pos_x", "pos_y", "pos_z"])
    neurons.insert(0, "root_id", 720575940600000000 + np.arange(len(neurons)))
    neurons["cell_class"] = pd.NA
    neurons["hemibrain_type"] = pd.NA
    for col in ["super_class", "cell_class", "cell_type", "hemibrain_type", "side"]:
        neurons[col] = neurons[col].astype("string")
    n = len(neurons)

    sc = neurons["super_class"].to_numpy()
    side = neurons["side"].to_numpy()
    ctype = neurons["cell_type"].to_numpy()
    idx = lambda m: np.flatnonzero(m)  # noqa: E731

    pre, post, w = [], [], []

    def connect(src, dst, p, mean, sign_p=1.0):
        """Random connections; ``p`` may be a scalar or a (len(src), len(dst)) array."""
        if len(src) == 0 or len(dst) == 0:
            return
        m = rng.random((len(src), len(dst))) < p
        s, d = np.nonzero(m)
        sign = np.where(rng.random(len(s)) < sign_p, 1, -1)
        pre.append(src[s]); post.append(dst[d])
        w.append(sign * rng.poisson(mean, len(s)).clip(1))

    # retinal coordinates of the photoreceptor grid: u = lateral, v = ventral (0..1)
    uv = np.c_[gx.ravel(), gy.ravel()] / (eye_size - 1)
    for s in ("left", "right"):
        # same random draws for both hemispheres -> a mirror-symmetric brain, so
        # any left/right asymmetry in behaviour comes from what the eyes see
        rng = np.random.default_rng(seed + 1)
        other = "right" if s == "left" else "left"
        eye = idx((ctype == "R7") & (side == s))
        optic = idx((sc == "optic") & (side == s))
        # optic neurons pool a small retinotopic patch (columnar organisation)
        rf = rng.random((len(optic), 2))
        dist = np.linalg.norm(uv[:, None, :] - rf[None, :, :], axis=2)
        connect(eye, optic, 0.6 * (dist < 0.12), 18)
        connect(optic, optic, 0.01, 3, sign_p=0.6)
        # turning pathway fed by the lower lateral field: turn towards the side
        # with more going on (a crude version of a fly's stripe fixation)
        low_lat = ((rf[:, 0] > 0.5) & (rf[:, 1] > 0.5)).astype(float)[:, None]
        connect(optic, idx((ctype == "DNa02") & (side == s)), 0.25 * low_lat, 4)
        connect(optic, idx((ctype == "DNa01") & (side == s)), 0.15 * low_lat, 4)
        connect(optic, idx((ctype == "DNa02") & (side == other)), 0.02, 3, sign_p=0.0)
        connect(optic, idx(sc == "central"), 0.01, 3, sign_p=0.8)
    rng = np.random.default_rng(seed + 2)
    central = idx(sc == "central")
    connect(central, central, 0.004, 3, sign_p=0.7)
    connect(central, idx(sc == "descending"), 0.02, 3, sign_p=0.75)
    # forward (P9) and backward (MDN) walking inhibit each other
    connect(idx(ctype == "DNp09"), idx(ctype == "MDN"), 1.0, 6, sign_p=0.0)
    connect(idx(ctype == "MDN"), idx(ctype == "DNp09"), 1.0, 6, sign_p=0.0)

    pre, post, w = map(np.concatenate, (pre, post, w))
    weights = sp.coo_matrix((w.astype(np.float32), (pre, post)), shape=(n, n)).tocsr()
    weights.sum_duplicates()
    return Connectome(weights, neurons, name="synthetic toy brain")


def load(kind: str = "flywire", data_dir: Path = DEFAULT_DATA_DIR, seed: int = 0) -> Connectome:
    if kind == "synthetic":
        return synthetic(seed)
    if kind == "flywire":
        return load_flywire(data_dir)
    raise ValueError(f"unknown connectome {kind!r}")
