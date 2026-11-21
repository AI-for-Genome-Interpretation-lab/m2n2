"""
Reference simulation, run manually: python tests/reference/simulation.py
270 individuals, 3 crossed specificities A, B, C of 3 levels each, 10 individuals per combination.
y = Z u_G + sum_k Z_Sk u_Sk + e, with Z the unit-sphere embedding of 1000 random features by a
freshly initialized ToyNN, and the variances G: 10, A: 5, B: 3, C: 1, R: 1.
m2n2 is then fitted on the same features, with a freshly initialized ToyNN each time:
without specificity, with A, and with A, B and C.
Writes tests/reference/simulation.json: simulated variances, y and feature value per individual.
"""
# %%
import json
from itertools import product
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from m2n2 import MixedModelNN, ToyNN

VARIANCES = {"G": 10, "A": 5, "B": 3, "C": 1, "R": 1}

torch.manual_seed(42)
z = torch.randn(270, 1000)
data = pd.DataFrame(
    [(f"A{a}", f"B{b}", f"C{c}") for a, b, c in product(range(3), repeat=3)] * 10,
    columns=["A", "B", "C"],
)


class Embedder(nn.Module):
    def __init__(self, z):
        super().__init__()
        self.nn = ToyNN(z.shape[1])
        self.register_buffer("z", z)

    def forward(self, rows=None):
        return self.nn.embed(self.z if rows is None else self.z[rows])


# %%
# simulation, embeddings of a freshly initialized network projected on the unit sphere
with torch.no_grad():
    Z = F.normalize(Embedder(z).eval()(), dim=1)
    d = Z.shape[1]

    feature = Z @ (VARIANCES["G"] ** 0.5 * torch.randn(d))
    for s in "ABC":
        codes, levels = pd.factorize(data[s])
        u = VARIANCES[s] ** 0.5 * torch.randn(len(levels), d)
        feature += (Z * u[codes]).sum(dim=1)
    data["y"] = (feature + VARIANCES["R"] ** 0.5 * torch.randn(len(data))).numpy()


def fit(specificity):
    torch.manual_seed(0)
    model = MixedModelNN.from_dataframe(
        data,
        response="y",
        Z=Embedder(z),
        specificity=specificity,
        lr_beta=1e-2,
        lr_var=1e-2,
    ).fit(n_epoch=1000, validation=0.2, patience=50, verbose=False)
    model.plot()
    print(model.varcomp())
    return model


# %%
# general only
general = fit(None)

# %%
# one specificity
specific = fit("A")

# %%
# three specificities
specific3 = fit(["A", "B", "C"])

# %%
with open(Path(__file__).parent / "simulation.json", "w") as f:
    json.dump({
        "variances": VARIANCES,
        "y": [round(float(v), 4) for v in data["y"]],
        "feature": [round(float(v), 4) for v in feature],
    }, f, indent=1)
