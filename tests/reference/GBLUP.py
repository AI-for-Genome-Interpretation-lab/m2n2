"""
Reference GEBVs for tests/test_arabido.py, run manually: python tests/reference/GBLUP.py
GBLUP on DTF1, every phenotyped accession of the meiosim Arabidopsis panel (10k SNPs), on CPU.
Writes tests/reference/GBLUP.json: REML convergence, variances and GEBV per individual.
"""
import json
from pathlib import Path

import numpy as np
from meiosim import Arabidopsis
from meiosim.Population import MISSING
from pyreml import MixedModel, Random, Residual, A_genomic

pop = Arabidopsis(n_SNPs=10_000, seed=42)
data = pop.phenotypes[["individual", "DTF1"]]
labelled = data["DTF1"].notna().to_numpy()
data = data[labelled].reset_index(drop=True)

# dosages mean-imputed over the whole panel, as in the m2n2 test, then recoded to -1 / 0 / 1
z = pop.genotypes.astype(np.float64)
z[z == MISSING] = np.nan
z = np.where(np.isnan(z), np.nanmean(z, axis=0), z)
K = A_genomic(z[labelled] - 1, shrink = True)

model = MixedModel.from_dataframe(
    data=data,
    response="DTF1",
    fixed="1",
    random=Random(
        unit="individual",
        right_hand="str",
        covariance=K,
        matrix_index=data["individual"].tolist()
    ),
    residual=Residual(),
    device="cpu",
).fit()

var_G = float(np.squeeze(model.random[0].variance["sigma"]))
var_R = float(np.squeeze(model.residual.variance["sigma"]))
print("REML converged:", model.opti_REML.converged)
mean_diag_K = float(np.diag(K).mean())  # var_G * mean_diag_K: genetic variance per individual, as m2n2 var_G_unit
print("var_G:", var_G, "var_R:", var_R, "mean diag K:", mean_diag_K)

gebv = model.random[0].table.set_index("unit")["prediction"]
gebv.index = gebv.index.astype(str)

with open(Path(__file__).parent / "GBLUP.json", "w") as f:
    json.dump({
        "converged": bool(model.opti_REML.converged),
        "var_G": var_G,
        "var_R": var_R,
        "mean_diag_K": mean_diag_K,
        "gebv": {
            i: round(float(gebv[i]), 2) for i in data["individual"]
        },
    }, f, indent=1)
