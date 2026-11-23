# %%
import json
from functools import partial
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import to_hex
from meiosim import Arabidopsis
from meiosim.Population import MISSING
from m2n2 import MixedModelNN, Embedding, RandomNN, ToyNN
from pyreml import Residual

pop = Arabidopsis(n_SNPs=0, seed=42)

missing = pop.genotypes == MISSING
Z = pop.genotypes.astype(np.float32) - 1.0
Z[missing] = 0.0
C = (Z @ Z.T) / Z.shape[1]

# remove clones / quasi-clones
dup = (np.triu(C, 1) >= 0.90).any(axis=0)
pop = pop.subset(list(np.where(~dup)[0]))

data = pop.phenotypes[["individual", "DTF1"]].merge(
    pop.metadata[["individual", "country"]], on="individual"
)

z = pop.genotypes.astype(np.float32)
z[z == MISSING] = np.nan
z = torch.tensor(np.where(np.isnan(z), np.nanmean(z, axis=0), z))

# 3D widgets: one JSON per PCA, read by doc/www/script.js
codes, levels = pd.factorize(data["country"])  # same order as model.levels
palette = plt.cm.turbo(np.random.default_rng(0).permutation(np.linspace(0, 1, len(levels))))  # same as model.plot()
Path("../widgets").mkdir(exist_ok=True)

def pca(P):
    Pc = (P - P.mean(dim=0)).double()
    eigval, eigvec = torch.linalg.eigh(Pc @ Pc.T)
    eigval, eigvec = eigval.flip(0).clamp(min=0), eigvec.flip(1)
    return (eigvec * eigval.sqrt()).numpy(), (100 * eigval / eigval.sum()).numpy()

def save_cloud(coords, inertia, name):
    with open(f"../widgets/{name}.json", "w") as f:
        json.dump({
            "inertia": np.round(inertia[:3], 2).tolist(),
            "levels": list(levels),
            "colors": [to_hex(c) for c in palette],
            "group": codes.tolist(),
            "coords": np.round(coords[:, :3], 4).tolist(),
        }, f, indent=1)

# %%
# genomic kinship (VanRaden) and its PCA
import seaborn as sns
from pyreml import A_genomic

K = A_genomic(z.numpy().astype(np.float64) - 1)  # dosages recoded to -1 / 0 / 1

order = data["country"].argsort(kind="stable").to_numpy()  # accessions grouped by country
fig, ax = plt.subplots(figsize=(3, 2.5))
sns.heatmap(K[np.ix_(order, order)], cmap="coolwarm", center=0, square=True, xticklabels=False, yticklabels=False, ax=ax)
ax.set_title("G (additive, genomic)", fontsize=10)
plt.tight_layout()
fig.savefig("../img/kinship.png", dpi=300, bbox_inches="tight")
plt.close(fig)

eigval, eigvec = np.linalg.eigh(K)
eigval, eigvec = eigval[::-1], eigvec[:, ::-1]
inertia = 100 * eigval / eigval.sum()
pcs = eigvec[:, :3] * np.sqrt(eigval[:3])

save_cloud(pcs, inertia, "kinship")

# %%
# M2N2

class Network(nn.Module):
    """ToyNN embedding of the accessions, rows in the order of z; rows is an optional boolean mask."""

    def __init__(self, z):
        super().__init__()
        self.nn = ToyNN(z.shape[1])
        self.register_buffer("z", z)

    def forward(self, rows=None):
        return self.nn.embed(self.z if rows is None else self.z[rows])

torch.manual_seed(42)
network = Network(z) # /!\ z and data must be perfectly aligned

model = MixedModelNN.from_dataframe(
    data,
    response = "DTF1",
    fixed = "1 + country",
    embedding = {
        "geno": Embedding(
            network,
            optimizer = partial(
                torch.optim.AdamW,
                lr=1e-3,
                weight_decay=1e-1
            )
        )
    },
    random = [
        RandomNN(
            embedding = "geno",
        ),
        RandomNN(
            embedding = "geno",
            group = "country",
        )
    ],
    residual = Residual(),
    lr_beta=1e-2,
    lr_var=1e-2,
)

model.fit(
    n_epoch=1000,
    batch_size=256,
    validation=0.2,
    patience=50,
    verbose=False,
)


fig = model.plot()
fig.savefig("../img/m2n2_training.png", dpi=150, bbox_inches="tight")

model.random[0].plot(group_by = "country")
model.random[1].plot()

model.varcomp()

# embedding kinship, VanRaden-like: embeddings on the unit sphere, centered, scaled by 1 / sum_j var(z_j)
network.eval()
with torch.no_grad():
    Z = F.normalize(network(), dim=1)
Z_c = Z - Z.mean(dim=0)
K_Z = (Z_c @ Z_c.T / Z.var(dim=0).sum()).cpu().numpy()

fig, ax = plt.subplots(figsize=(3, 2.5))
sns.heatmap(K_Z[np.ix_(order, order)], cmap="coolwarm", center=0, square=True, xticklabels=False, yticklabels=False, ax=ax)
ax.set_title("K (embeddings, VanRaden-like)", fontsize=10)
plt.tight_layout()
fig.savefig("../img/m2n2_kinship.png", dpi=300, bbox_inches="tight")
plt.close(fig)

for term, title, name in [
    (model.random[0], "Z", "m2n2_embedding"),
    (model.random[0], "Z*u", "m2n2_general"),
    (model.random[1], "Z*u", "m2n2_specific"),
]:
    save_cloud(term.pca[title]["coords"], term.pca[title]["inertia"], name)

# %%
# NN alone
labelled = data["DTF1"].notna().to_numpy()

torch.manual_seed(42)
nn_alone = ToyNN(z.shape[1]).fit(
    z[labelled],
    torch.tensor(data.loc[labelled, "DTF1"].to_numpy(), dtype=torch.float32),
    n_epoch=1000,
    batch_size=256,
    validation=0.2,
    patience=50,
    verbose=False,
)

fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(10, 3.5))
ax_loss.plot(nn_alone.loss)
ax_loss.set_xlabel("epoch")
ax_loss.set_ylabel("train MSE")
ax_score.plot(nn_alone.score)
ax_score.axvline(np.argmin(nn_alone.score), color="grey", linestyle="--")
ax_score.set_xlabel("epoch")
ax_score.set_ylabel("validation MSE")
plt.tight_layout()
plt.show()

with torch.no_grad():
    nn_embedding = nn_alone.embed(z)            # embedding of every accession
    nn_prediction = nn_alone.head(nn_embedding)  # missing DTF1 included

save_cloud(*pca(nn_embedding), "nn_embedding")

# %%
# GBLUP: y = 1 mu + g + e, g ~ N(0, s2_g K), K shrunk to be positive definite
from pyreml import MixedModel, Random

K_s = A_genomic(z.numpy().astype(np.float64) - 1, shrink=True)
gblup = MixedModel.from_dataframe(
    data=data[labelled],
    response="DTF1",
    fixed="1",
    random=Random(
        unit="individual",
        right_hand="str",
        covariance=K_s,
        matrix_index=data["individual"].tolist(),  # every accession, missing DTF1 included
    ),
    residual=Residual(),
).fit()
gblup.random[0].variance["sigma"], gblup.residual.variance["sigma"]

table = gblup.random[0].table.set_index("unit")
gblup_prediction = torch.tensor(table.loc[data["individual"], "prediction"].to_numpy())

# marker effects back-solved from the GEBVs: minimal-norm least squares, M a ≈ g (exact without shrinkage)
M = (z - z.mean(dim=0)).double()  # centered markers
a_hat = torch.linalg.pinv(M) @ gblup_prediction
ridge_weighted = M * a_hat  # row sums ≈ gblup_prediction
save_cloud(*pca(ridge_weighted), "ridge_weighted")
