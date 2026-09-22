"""Biological corrections to the plain count-weighted connectome model.

The Shiu et al. 2024 model signs every synapse from the *predicted* neurotransmitter of the
presynaptic neuron: acetylcholine excitatory, GABA and glutamate inhibitory, and everything else
(dopamine, serotonin, octopamine) also counted as fast excitation. That works for the small,
local stimulations used in the paper. Under whole-field visual input it tips the whole brain
into a self-sustaining "seizure" (tens of thousands of spikes per frame that persist after the
input is switched off). Three facts from the literature account for most of it:

``modulatory``
    Dopamine, serotonin and octopamine act through G-protein-coupled receptors in Drosophila;
    there are no known fast ionotropic receptors for them. Their ~2 M synapses should not
    act as fast excitation, so they get zero fast weight. Dopamine's real role, gating
    plasticity in the mushroom body, is modelled explicitly in ``learning.py``.

``kenyon_cholinergic``
    The neurotransmitter predictor labels 5,172 of the 5,177 Kenyon cells dopaminergic, but
    Kenyon cells are cholinergic (Barnstedt et al. 2016, Neuron 89:1237). Their outputs are
    restored as excitatory, except KC -> KC synapses, which act through the inhibitory
    muscarinic receptor mAChR-B (Manoim et al. 2022, Curr Biol 32:4490), so those are
    inhibitory.

Nothing is added or removed: only the signs of synapses that exist in FlyWire change, or their
fast weights are set to zero.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .connectome import Connectome

MODULATORY = ("dopamine", "serotonin", "octopamine")
ALL = ("modulatory", "kenyon_cholinergic")


def corrected(conn: Connectome, fixes=ALL) -> Connectome:
    if not fixes:
        return conn
    w = conn.weights.tocoo()
    data = w.data.copy()
    nt = conn.neurons["top_nt"].fillna("").to_numpy().astype(str)
    kc = conn.neurons["cell_class"].eq("Kenyon_Cell").to_numpy(dtype=bool, na_value=False)
    pre_kc = kc[w.row]
    if "modulatory" in fixes:
        data[np.isin(nt[w.row], MODULATORY) & ~pre_kc] = 0.0
    if "kenyon_cholinergic" in fixes:
        mag = np.abs(data[pre_kc])
        data[pre_kc] = np.where(kc[w.col[pre_kc]], -mag, mag)
    elif "modulatory" in fixes:
        data[pre_kc] = 0.0  # still labelled dopaminergic
    weights = sp.csr_matrix((data, (w.row, w.col)), shape=w.shape)
    weights.eliminate_zeros()
    return Connectome(weights, conn.neurons, name=conn.name + " + corrections(" + ",".join(fixes) + ")")
