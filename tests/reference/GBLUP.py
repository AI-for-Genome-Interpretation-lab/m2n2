"""
Reference GEBVs for the tests, run manually: python tests/reference/GBLUP.py
Trivariate GBLUP on DTF1, DTF2 and DTF3, every accession of the meiosim Arabidopsis panel (10k SNPs)
phenotyped for at least one of them, on CPU: genomic kinship (str), full covariances across the traits.
Writes tests/reference/GBLUP.json: REML convergence, covariances and GEBV per trait and individual.
"""
import json
from pathlib import Path

import numpy as np
from meiosim import Arabidopsis
from meiosim.Population import MISSING
from pyreml import MixedModel, Random, Residual, A_genomic

TRAITS = ["DTF1", "DTF2", "DTF3"]

pop = Arabidopsis(n_SNPs=10_000, seed=42)
data = pop.phenotypes[["individual", *TRAITS]]
labelled = data[TRAITS].notna().any(axis=1).to_numpy()
data = data[labelled].reset_index(drop=True)

# dosages mean-imputed over the whole panel, as in the m2n2 tests, then recoded to -1 / 0 / 1
z = pop.genotypes.astype(np.float64)
z[z == MISSING] = np.nan
z = np.where(np.isnan(z), np.nanmean(z, axis=0), z)
K = A_genomic(z[labelled] - 1, shrink = True)

model = MixedModel.from_dataframe(
    data=data,
    response=TRAITS,
    fixed="1",
    random=Random(
        unit="individual",
        left_hand="full",
        right_hand="str",
        covariance=K,
        matrix_index=data["individual"].tolist()
    ),
    residual=Residual(left_hand="full"),
    device="cpu",
).fit()

G = np.asarray(model.random[0].variance["sigma"]).tolist()
R = np.asarray(model.residual.variance["sigma"]).tolist()
mean_diag_K = float(np.diag(K).mean())  # G * mean_diag_K: genetic covariance per individual, as m2n2 unit variances
print("REML converged:", model.opti_REML.converged)
print("G:", G, "R:", R, "mean diag K:", mean_diag_K)

table = model.random[0].table
gebv = {
    trait: {
        str(unit): round(float(prediction), 2)
        for unit, prediction in table.loc[table["response"] == trait, ["unit", "prediction"]].to_numpy()
    }
    for trait in TRAITS
}

with open(Path(__file__).parent / "GBLUP.json", "w") as f:
    json.dump({
        "converged": bool(model.opti_REML.converged),
        "traits": TRAITS,
        "G": G,
        "R": R,
        "mean_diag_K": mean_diag_K,
        "gebv": gebv,
    }, f, indent=1)
