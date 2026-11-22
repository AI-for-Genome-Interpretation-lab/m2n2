import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


class RandomNN:
    """
    Random effect Z_j u_j upon an embedding Z, rows on the unit sphere:
    Z_j = [Z * 1{group A} | Z * 1{group B} | ...], u_j ~ N(0, s2_j I), one effect vector per group.
    Without group, a single group: Z_j = Z, the general effect.

    Exposes the interface of the pyreml components (varmeth, varmeth_inv, migrate, format_pred,
    format_variance, k * c * L columns), so that MixedModel handles it as any random effect.
    """

    def __init__(self, embedding: str, group: str | None = None):
        self.embedding = embedding
        self.group = group
        self.label = embedding if group is None else f"{embedding}:{group}"

    def design(self, data: pd.DataFrame, network, d: int, device):
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
        self.d = d
        self.levels = list(levels)
        self.codes = torch.as_tensor(codes, device=device)
        self.groups = F.one_hot(self.codes, len(levels)).double()
        self.k, self.c, self.L = 1, d, len(levels)  # read by pyreml to slice uhat and PEV
        self.log_var = torch.tensor(0.0, dtype=torch.double, device=device, requires_grad=True)  # set by fit
        self.dtype = torch.double

    def migrate(self, dtype: torch.dtype = torch.double):
        self.dtype = dtype
        self.groups = self.groups.to(dtype)

    def block(self, Z: torch.Tensor, rows: torch.Tensor | None = None) -> torch.Tensor:
        """Z_j of the rows of Z, the embeddings of the data rows `rows` (boolean mask, None = all)."""
        g = self.groups if rows is None else self.groups[rows]
        return (g[:, :, None] * Z[:, None, :]).flatten(1)

    def embed(self) -> torch.Tensor:
        """Z_j of every row of data."""
        return self.block(F.normalize(self.network(), dim=1))

    def varmeth(self):
        def block():
            eye = torch.eye(self.L * self.d, dtype=self.dtype, device=self.log_var.device)
            return torch.exp(self.log_var).to(self.dtype) * eye
        return block

    def varmeth_inv(self):
        def block_inv():
            eye = torch.eye(self.L * self.d, dtype=self.dtype, device=self.log_var.device)
            return torch.exp(-self.log_var).to(self.dtype) * eye, self.L * self.d * self.log_var
        return block_inv

    def format_pred(self, uhat: torch.Tensor, pev: torch.Tensor):
        self.uhat = uhat.detach().reshape(self.L, self.d)
        self.PEV = pev

    def format_variance(self):
        self.var = self.log_var.exp().item()

    def summarize(self, Z: torch.Tensor):
        """Prediction, unit variance and PCA, from Z the embeddings of every row of data."""
        # variance ratio lambda = var(embedding dimension) / var(unit), on every row of data:
        # 1 / sum_j var(z_j), the embedding analog of 1 / sum_i 2 p_i (1 - p_i) for SNPs
        self.var_ratio = 1 / Z.var(dim=0).sum().item()
        self.var_unit = self.var / self.var_ratio

        weighted = Z * self.uhat.to(Z.dtype)[self.codes]
        self.prediction = weighted.sum(dim=1)

        self.pca = {}
        for title, P in {"Raw embeddings": Z, "Weighted embeddings": weighted}.items():
            Pc = P - P.mean(dim=0)
            eigval, eigvec = torch.linalg.eigh(Pc.T @ Pc / (len(Pc) - 1))
            eigval, eigvec = eigval.flip(0), eigvec.flip(1)
            self.pca[title] = {
                "coords": (Pc @ eigvec).cpu().numpy(),
                "inertia": (100 * eigval / eigval.sum()).cpu().numpy(),
                "eigvec": eigvec.cpu().numpy(),
            }

    def plot(self, plan: tuple[int, int] = (0, 1), group_by: str | None = None):
        i, j = plan
        group_by = self.group if group_by is None else group_by
        if group_by is None:
            codes, levels = np.zeros(len(self.data), dtype=int), [None]
        else:
            codes, levels = pd.factorize(self.data[group_by])

        # one PCA panel per embedding, plus a dedicated legend panel when colored by group
        legend = group_by is not None
        fig, axes = plt.subplots(
            1, 2 + legend,
            figsize=(12 + 3 * legend, 6),
            width_ratios=[6, 6] + [3] * legend,
        )
        # continuous palette: one distinct color per level, however many levels
        colors = plt.cm.turbo(np.random.default_rng(0).permutation(np.linspace(0, 1, len(levels))))
        for ax, (title, pca) in zip(axes, self.pca.items()):
            for g, level in enumerate(levels):
                ax.scatter(pca["coords"][codes == g, i], pca["coords"][codes == g, j], color=colors[g], label=level, alpha=0.7)
            ax.axhline(0, color="grey", lw=0.5)
            ax.axvline(0, color="grey", lw=0.5)
            ax.set_xlabel(f"PC{i + 1} ({pca['inertia'][i]:.1f} %)")
            ax.set_ylabel(f"PC{j + 1} ({pca['inertia'][j]:.1f} %)")
            ax.set_title(f"Individual projections, {self.label}\n{title}")
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
