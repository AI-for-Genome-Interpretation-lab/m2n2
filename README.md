<h1 align="center">
<img width="750" alt="m2n2" src="https://raw.githubusercontent.com/AI-for-Genome-Interpretation-lab/m2n2/main/doc/img/banner.png" />
</h1>

<p>
  Mixed modeling with neural networks, built on
  <a href="https://github.com/ai-for-genome-interpretation-lab/pyreml">
    <img width="110" alt="pyreml"
         src="https://raw.githubusercontent.com/AI-for-Genome-Interpretation-lab/pyreml/main/doc/img/logo/pyreml.png"
         align="absmiddle" /></a>.
</p>

The random design of a linear mixed model is the embedding of a neural network, trained jointly with the fixed effects and the variance components on the REML loss.

## Installation

Install **m2n2** with `pip`:

```bash
pip install m2n2
```

## The model

**m2n2** fits a linear mixed model built upon the neural network $\nu$ of parameters $\theta$:

$$
y = X\beta + Zu + r
\qquad
z_i \mapsto \nu(z_i, \theta) \in \mathbb{R}^d
$$

- $y$: a quantitative continuous label.
- $X$: the fixed effects design matrix, provided from a `patsy` formula.
- $\beta$: the fixed effects.
- $z_i$: features of individual $i$.
- $Z = \bigoplus_{k=1}^{q} \nu(z_{(k)}, \theta), Z \in \mathbb{R}^{n \times qd}$: the random effects design matrix, or matrix of embeddings, for all $n$ individuals from $q$ distinct groups.
- $u \sim N(0, \sigma^2_u I), u \in \mathbb{R}^{qd}$: the embedding effects, with an embedding effect vector $u_k$ for each of the $q$ groups.
- $r \sim N(0, \sigma^2_R I)$: the residual.

Several random effects based on neural networks can be provided, either referring to the same neural network, the same features, or not.

Additional random effects following the **pyreml** specification system can also be provided.

## Example

```python
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from meiosim import Arabidopsis
from meiosim.Population import MISSING
from m2n2 import MixedModelNN, Embedding, RandomNN
from pyreml import Random, Residual

pop = Arabidopsis(n_SNPs=10_000, seed=42)
data = pop.phenotypes[["individual", "DTF1"]].merge(
    pop.metadata[["individual", "country"]], on="individual"
)

z = pop.genotypes.astype(np.float32)
z[z == MISSING] = np.nan
z = torch.tensor(np.where(np.isnan(z), np.nanmean(z, axis=0), z))

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


fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(10, 3.5))
ax_loss.plot(model.loss)
ax_loss.set_xlabel("epoch")
ax_loss.set_ylabel("REML -2logLik")
ax_score.plot(model.score)
ax_score.axvline(np.argmin(model.score), color="grey", linestyle="--")
ax_score.set_xlabel("epoch")
ax_score.set_ylabel("validation MSE")
plt.show()

model.random[0].plot(group_by = "country")
model.random[1].plot()

model.varcomp()
```

## Documentation

The documentation is available at [this address](https://ai-for-genome-interpretation-lab.github.io/m2n2).

## Code

The code is available at [this address](https://github.com/ai-for-genome-interpretation-lab/m2n2).

## License

Copyright © 2026 CNRS, University of Montpellier

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
