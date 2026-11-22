import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from pyreml import Random


class RandomNN(Random):
    """
    Random effect Z_j u_j upon an embedding Z, rows on the unit sphere:
    Z_j = [Z * 1{group A} | Z * 1{group B} | ...], one effect vector per group and per response.
    Without group, a single group: Z_j = Z, the general effect.

    A pyreml Random whose design is the embedding: k responses, c = 1 component, and its
    L = q_j * dim levels are the (group, embedding dimension) pairs, so that u_j ~ N(0, S ⊗ I_L),
    S structured by left_hand across the responses. design is overridden, the network replacing the
    incidence of `unit`; the variance structure (init_varparams, varmeth, varmeth_inv,
    format_variance) is pyreml's.
    """

    def __init__(
        self,
        embedding: str,
        group: str | None = None,
        left_hand: str = "iid",
        n_axes: int | None = None,
    ):
        self.embedding = embedding
        self.group = group
        # design is overridden, so unit is never read as a column: its only role left is effect_label,
        # the name of the effect in the pyreml outputs and in varcomp; None is not accepted by pyreml
        super().__init__(
            unit=embedding if group is None else f"{embedding}:{group}",
            left_hand=left_hand,
            n_axes=n_axes,
        )

    def design(self, data: pd.DataFrame, responses: list[str], network, dim: int, device, scale):
        if self.group is None:
            codes, levels = np.zeros(len(data), dtype=int), [None]
        else:
            if pd.api.types.is_numeric_dtype(data[self.group]):
                raise ValueError(f"group column '{self.group}' must be qualitative, got {data[self.group].dtype}")
            if data[self.group].isna().any():
                raise ValueError(f"group column '{self.group}' has {data[self.group].isna().sum()} missing values")
            codes, levels = pd.factorize(data[self.group])

        self.data = data
        self.network = network
        self.dim = dim
        self.levels = list(levels)
        self.codes = torch.as_tensor(codes, device=device)
        self.groups = F.one_hot(self.codes, len(levels)).double()

        # pyreml layout: k responses, c = 1 component, L = (group, dimension) levels, d = k * c
        self.device = device
        self.responses = list(responses)
        self.scale = scale
        self.colnames = ["embedding"]
        self.k, self.c, self.L = len(self.responses), 1, len(levels) * dim
        self.d = self.k * self.c
        self.init_varparams()  # -> log_S, varparams, components; S = diag(var(y_r)) in natural units
        self.log_S0 = self.log_S.detach().clone()  # restored by fit

    def migrate(self, dtype: torch.dtype = torch.double):
        super().migrate(dtype)
        self.groups = self.groups.to(dtype)

    def block(self, Z: torch.Tensor, rows: torch.Tensor, responses: torch.Tensor | None = None) -> torch.Tensor:
        """
        Z_j of the observations: Z their embeddings, rows their data rows, responses their response
        (None: the design of the rows, regardless of the response).
        """
        B = (self.groups[rows][:, :, None] * Z[:, None, :]).flatten(1)
        if responses is None:
            return B
        return (F.one_hot(responses, self.k).to(B.dtype)[:, :, None] * B[:, None, :]).flatten(1)

    def embed(self) -> torch.Tensor:
        """Z_j of every row of data, regardless of the response."""
        return self.block(F.normalize(self.network(), dim=1), torch.arange(len(self.data), device=self.codes.device))

    def format_pred(self, uhat: torch.Tensor, pev: torch.Tensor):
        uhat = uhat.detach().reshape(self.k, len(self.levels), self.dim)
        self.uhat = uhat[0] if self.k == 1 else uhat
        self.PEV = pev

    def summarize(self, Z: torch.Tensor):
        """Prediction, variance ratio and PCA, from Z the embeddings of every row of data."""
        # variance ratio lambda = var(embedding dimension) / var(unit), on every row of data:
        # 1 / sum_j var(z_j), the embedding analog of 1 / sum_i 2 p_i (1 - p_i) for SNPs
        self.var_ratio = 1 / Z.var(dim=0).sum().item()

        uhat = self.uhat.to(Z.dtype).reshape(self.k, len(self.levels), self.dim)
        weighted = [Z * u[self.codes] for u in uhat]  # one per response
        prediction = torch.stack([w.sum(dim=1) for w in weighted], dim=1)
        self.prediction = prediction[:, 0] if self.k == 1 else prediction

        def pca(P):
            Pc = P - P.mean(dim=0)
            eigval, eigvec = torch.linalg.eigh(Pc.T @ Pc / (len(Pc) - 1))
            eigval, eigvec = eigval.flip(0), eigvec.flip(1)
            return {
                "coords": (Pc @ eigvec).cpu().numpy(),
                "inertia": (100 * eigval / eigval.sum()).cpu().numpy(),
                "eigvec": eigvec.cpu().numpy(),
            }

        self.pca = {
            "Z": pca(Z),
            "Z*u": pca(weighted[0]) if self.k == 1 else {r: pca(w) for r, w in zip(self.responses, weighted)},
        }

    def plot(self, plan: tuple[int, int] = (0, 1), group_by: str | None = None, response: str | None = None):
        i, j = plan
        group_by = self.group if group_by is None else group_by
        if group_by is None:
            codes, levels = np.zeros(len(self.data), dtype=int), [None]
        else:
            codes, levels = pd.factorize(self.data[group_by])

        if self.k == 1:
            panels = self.pca
        else:
            response = self.responses[0] if response is None else response
            panels = {"Z": self.pca["Z"], f"Z*u, {response}": self.pca["Z*u"][response]}

        # one PCA panel per embedding, plus a dedicated legend panel when colored by group
        legend = group_by is not None
        fig, axes = plt.subplots(
            1, 2 + legend,
            figsize=(12 + 3 * legend, 6),
            width_ratios=[6, 6] + [3] * legend,
        )
        # continuous palette: one distinct color per level, however many levels
        colors = plt.cm.turbo(np.random.default_rng(0).permutation(np.linspace(0, 1, len(levels))))
        for ax, (title, pca) in zip(axes, panels.items()):
            for g, level in enumerate(levels):
                ax.scatter(pca["coords"][codes == g, i], pca["coords"][codes == g, j], color=colors[g], label=level, alpha=0.7)
            ax.axhline(0, color="grey", lw=0.5)
            ax.axvline(0, color="grey", lw=0.5)
            ax.set_xlabel(f"PC{i + 1} ({pca['inertia'][i]:.1f} %)")
            ax.set_ylabel(f"PC{j + 1} ({pca['inertia'][j]:.1f} %)")
            ax.set_title(f"Individual projections, {self.effect_label}\n{title}")
        if legend:
            axes[-1].axis("off")
            axes[-1].legend(
                *axes[0].get_legend_handles_labels(),
                loc="center left",
                ncol=1 + len(levels) // 25,
                fontsize=8,
                frameon=False,
            )
        plt.tight_layout()
        plt.show()
