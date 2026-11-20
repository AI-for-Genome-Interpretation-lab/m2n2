# %%
import json
from functools import partial
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib.colors import to_hex
from meiosim import Arabidopsis
from meiosim.Population import MISSING
from m2n2 import MixedModelNN, ToyNN as NN

pop = Arabidopsis(n_SNPs=10_000, seed=42)
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
        }, f)

# %%
# genomic kinship (VanRaden) and its PCA
import seaborn as sns
from pyreml import A_genomic

K = A_genomic(z.numpy().astype(np.float64) - 1)  # dosages recoded to -1 / 0 / 1

order = data["country"].argsort(kind="stable").to_numpy()  # accessions grouped by country
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(K[np.ix_(order, order)], cmap="coolwarm", center=0, square=True, xticklabels=False, yticklabels=False, ax=ax)
ax.set_title("G (additive, genomic)")
plt.tight_layout()
fig.savefig("../img/kinship.png", dpi=150, bbox_inches="tight")
plt.close(fig)

eigval, eigvec = np.linalg.eigh(K)
eigval, eigvec = eigval[::-1], eigvec[:, ::-1]
inertia = 100 * eigval / eigval.sum()
pcs = eigvec[:, :3] * np.sqrt(eigval[:3])

save_cloud(pcs, inertia, "kinship")

# %%
# M2N2

class Network(nn.Module):
    def __init__(self, z):
        super().__init__()
        self.register_buffer("z", z)
        self.body = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(z.shape[1], 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(128, 64),
        )

    def forward(self, rows=None):
        return self.body(self.z if rows is None else self.z[rows])

torch.manual_seed(42)
network = Network(z) # /!\ z and data must be perfectly aligned

model = MixedModelNN.from_dataframe(
    data,
    response="DTF1",
    Z=network,
    specificity="country",
    optimizer=partial(
        torch.optim.AdamW,
        lr=1e-3,
        weight_decay=1e-1
    ),
    lr_beta=1e-2,
    lr_var=1e-2,
)

model.fit(
    n_epoch=1000,
    batch_size=256,
    validation=0.5,
    patience=50,
    verbose=False,
)


fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(10, 3.5))
ax_loss.plot(model.loss)
ax_loss.set_xlabel("epoch")
ax_loss.set_ylabel("REML -2logLik")
ax_score.plot(model.score)
ax_score.axvline(np.argmin(model.score), color="grey", linestyle="--")
ax_score.set_xlabel("epoch")
ax_score.set_ylabel("validation MSE")
fig.suptitle(f"REML polish converged: {model.opti_REML.converged}")
plt.tight_layout()
fig.savefig("../img/m2n2_training.png", dpi=150, bbox_inches="tight")
plt.close(fig)

model.plot()

model.varcomp()

# embedding kinship, VanRaden-like: centered embeddings scaled by 1 / sum_j var(z_j)
model.network.eval()
with torch.no_grad():
    Z_G = model.embed()[:, :model.d]
Z_c = Z_G - Z_G.mean(dim=0)
K_Z = (Z_c @ Z_c.T * model.var_ratio).cpu().numpy()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(K_Z[np.ix_(order, order)], cmap="coolwarm", center=0, square=True, xticklabels=False, yticklabels=False, ax=ax)
ax.set_title("K (embeddings, VanRaden-like)")
plt.tight_layout()
fig.savefig("../img/m2n2_kinship.png", dpi=150, bbox_inches="tight")
plt.close(fig)

for title, name in [
    ("Raw embeddings", "m2n2_embedding"),
    ("General weighted embeddings", "m2n2_general"),
    ("Specific weighted embeddings", "m2n2_specific"),
]:
    save_cloud(model.pca[title]["coords"], model.pca[title]["inertia"], name)

# %%
# NN alone
labelled = data["DTF1"].notna().to_numpy()

torch.manual_seed(42)
nn_alone = NN(z.shape[1]).fit(
    z[labelled],
    torch.tensor(data.loc[labelled, "DTF1"].to_numpy(), dtype=torch.float32),
    n_epoch=1000,
    batch_size=256,
    validation=0.5,
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
fig.savefig("../img/nn_training.png", dpi=150, bbox_inches="tight")
plt.close(fig)

with torch.no_grad():
    nn_embedding = nn_alone.embed(z)            # embedding of every accession
    nn_prediction = nn_alone.head(nn_embedding)  # missing DTF1 included
    nn_weighted = nn_embedding * nn_alone.head[0].weight.flatten()  # row sums + bias = prediction

save_cloud(*pca(nn_weighted), "nn_weighted")

# %%
# RR-BLUP: ridge regression on the markers, y = 1 mu + M a + e, a ~ N(0, s2_a I)
from pyreml import MixedModel

M = (z - z.mean(dim=0)).double()  # centered markers
y = torch.tensor(data.loc[labelled, "DTF1"].to_numpy(), dtype=torch.double).reshape(-1, 1)
X = torch.ones_like(y)

log_sa = torch.tensor(np.log(y.var().item() / M.shape[1]), dtype=torch.double, requires_grad=True)
log_se = torch.tensor(np.log(y.var().item()), dtype=torch.double, requires_grad=True)

def varmeth(self):
    G = torch.exp(log_sa) * torch.eye(self.q, dtype=self.dtype, device=self.device)
    R = torch.exp(log_se) * torch.eye(self.n, dtype=self.dtype, device=self.device)
    return G, R

rrblup = MixedModel(y=y, X=X, Z=M[labelled], params=[log_sa, log_se], varmeth=varmeth)
rrblup.fit()

rrblup_prediction = (M @ rrblup.uhat).flatten()  # M a of every accession, missing DTF1 included
log_sa.exp().item(), log_se.exp().item()

ridge_weighted = M * rrblup.uhat.detach().flatten()  # row sums = rrblup_prediction
save_cloud(*pca(ridge_weighted), "ridge_weighted")
