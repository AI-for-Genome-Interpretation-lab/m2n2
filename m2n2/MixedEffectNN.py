import copy
from functools import partial

import numpy as np
import pandas as pd
import patsy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyreml import MixedModel


class MixedModelNN(MixedModel):
    """
    y = X beta + Z u_G + sum_k Z_Sk u_Sk + e, with u_G ~ N(0, s2_G I),
    u_Sk ~ N(0, s2_Sk I), e ~ N(0, s2_R I). Z is the embedding returned by the
    network Z(), rows projected on the unit sphere (||z_i||^2 = 1) so that the
    variances are identifiable. Z_Sk = [Z * 1{group A} | Z * 1{group B} | ...],
    with the groups given by the k-th qualitative column of `specificity`, each
    specificity independent of the others; without any, the model reduces to
    y = X beta + Z u_G + e. The network is trained jointly with beta and the
    variances on the REML loss.

    Z() must return one embedding per row of data, in the same order: aligning
    the two is the caller's responsibility. Z(rows), with rows a boolean mask
    over the rows of data, must return the embeddings of the selected rows only:
    training feeds the labelled rows alone, so that batch statistics (e.g.
    BatchNorm) never see the others. Rows with a missing response are left out
    of the fit but still predicted.
    Everything runs in the dtype and on the device of Z()'s output.
    """

    @classmethod
    def from_dataframe(
        cls,
        data: pd.DataFrame,
        response: str,
        Z: nn.Module,
        specificity: str | list[str] | None = None,
        fixed: str = "1",
        optimizer=partial(torch.optim.AdamW, lr=1e-3, weight_decay=1e-1),
        lr_beta: float | None = None,
        lr_var: float | None = None,
    ):
        data = data.reset_index(drop=True)

        # eval mode: this shape check must not update the BatchNorm running stats
        Z.eval()
        with torch.no_grad():
            Z_all = Z()
        
        device, dtype = Z_all.device, Z_all.dtype

        if Z_all.shape[0] != len(data):
            raise ValueError(f"Z() returns {Z_all.shape[0]} rows, data has {len(data)}")

        labelled = data[response].notna()
        specificity = [specificity] if isinstance(specificity, str) else list(specificity or [])
        if len(set(specificity)) < len(specificity):
            raise ValueError(f"duplicated specificity columns: {specificity}")
        codes, levels, groups = {}, {}, {}
        for s in specificity:
            if pd.api.types.is_numeric_dtype(data[s]):
                raise ValueError(f"specificity column '{s}' must be qualitative, got {data[s].dtype}")
            if data[s].isna().any():
                raise ValueError(f"specificity column '{s}' has {data[s].isna().sum()} missing values")
            codes[s], levels[s] = pd.factorize(data[s])
            levels[s] = list(levels[s])
            groups[s] = torch.tensor(codes[s][:, None] == np.arange(len(levels[s])), dtype=dtype, device=device)

        dm = patsy.dmatrix(fixed, data=data[labelled], NA_action="drop", return_type="dataframe")
        keep = torch.as_tensor(data.index.isin(dm.index), device=device)

        y = torch.tensor(data.loc[dm.index, response].to_numpy(), dtype=dtype, device=device).reshape(-1, 1)
        X = torch.tensor(dm.to_numpy(), dtype=dtype, device=device)

        d = Z_all.shape[1]
        n_groups = [len(levels[s]) for s in specificity]

        log_v0 = float(np.log(y.var().item()))
        log_sG = torch.tensor(log_v0, dtype=dtype, device=device, requires_grad=True)
        log_sR = torch.tensor(log_v0, dtype=dtype, device=device, requires_grad=True)
        log_sS = [torch.tensor(log_v0, dtype=dtype, device=device, requires_grad=True) for _ in specificity]
        varparams = [log_sG, log_sR] + log_sS

        # pyreml picks the path: Woodbury (varmeth_inv) if q < n, direct (varmeth) otherwise
        def varmeth(self):
            g = torch.cat([torch.exp(log_sG).repeat(d)] + [torch.exp(log_s).repeat(n * d) for log_s, n in zip(log_sS, n_groups)])
            R = torch.exp(log_sR) * torch.eye(self.n, dtype=self.dtype, device=self.device)
            return torch.diag(g), R

        def varmeth_inv(self):
            ginv = torch.cat([torch.exp(-log_sG).repeat(d)] + [torch.exp(-log_s).repeat(n * d) for log_s, n in zip(log_sS, n_groups)])
            Rinv = torch.exp(-log_sR) * torch.eye(self.n, dtype=self.dtype, device=self.device)
            logdetG = d * log_sG + sum(n * d * log_s for log_s, n in zip(log_sS, n_groups))
            return torch.diag(ginv), Rinv, logdetG, self.n * log_sR

        mm = cls(
            y = y,
            X = X,
            q = d * (1 + sum(n_groups)),
            params      = varparams,
            varmeth     = varmeth,
            varmeth_inv = varmeth_inv,
            device      = device
        )

        mm.migrate(dtype)
        mm.OLS(terminate=False)

        mm.response = [response]
        mm.fixed_names = dm.design_info.column_names
        mm.keep = keep
        mm.network = Z
        mm.specificity = specificity
        mm.groups = groups
        mm.codes = codes
        mm.levels = levels
        mm.d = d
        mm.optimizer = optimizer([
            {"params": # network weights (matrices, kernels): shrunk
                [p for p in Z.parameters() if p.ndim >= 2],
            },
            {"params": # network biases and normalization parameters (vectors): free
                [p for p in Z.parameters() if p.ndim < 2],
                "weight_decay": 0.0,
            },
            {"params": # beta: free
                mm.beta,
                "weight_decay": 0.0,
            },
            {"params": # variances: free
                varparams,
                "weight_decay": 0.0,
            },
        ])
        for group, lr in zip(
            mm.optimizer.param_groups[2:],
            (lr_beta, lr_var),
        ):
            if lr is not None:
                group["lr"] = lr

        mm.loss = []
        mm.score = []

        return mm

    def embed(self, rows=None):
        Z = F.normalize(self.network(rows), dim=1)
        blocks = [Z]
        for g in self.groups.values():
            g = g if rows is None else g[rows]
            blocks.append((g[:, :, None] * Z[:, None, :]).flatten(1))
        return torch.cat(blocks, dim=1)

    def fit(
        self,
        n_epoch: int = 1000,
        batch_size: int | None = None,
        validation: float | None = None,
        patience: int | None = 50,
        verbose: bool = True,
    ):
        """
        batch_size: rows per minibatch, None = full batch; the incomplete last batch is dropped.
        validation: fraction of the labelled rows held out (None or 0 = none) to score each epoch by the
        MSE of their BLUP from the training rows; the best state is restored at the end, and the training
        stops after `patience` epochs without improvement (never if patience is None).
        """
        # full labelled data; minibatches swap _y, _X, _Z and n, restored after each epoch
        y, X, n = self._y, self._X, self.n
        labelled = self.keep.nonzero().flatten()  # data row of each labelled row

        perm = torch.randperm(n, device=self.device)
        n_valid = 0 if validation is None else round(validation * n)
        valid, train = perm[:n_valid], perm[n_valid:]
        size = len(train) if batch_size is None else min(batch_size, len(train))

        best_mse, best_epoch = float("inf"), 0
        for epoch in range(n_epoch):

            self.network.train()
            losses = []
            for b in train[torch.randperm(len(train), device=self.device)].split(size):
                if len(b) < size:
                    continue
                b = b.sort().values
                rows = torch.zeros_like(self.keep)
                rows[labelled[b]] = True
                self._y, self._X, self.n = y[b], X[b], len(b)
                self._Z = self.embed(rows)
                self.optimizer.zero_grad()
                loss = self.REML_loss()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            self._y, self._X, self.n = y, X, n
            self.loss.append(float(np.mean(losses)))
            message = f"epoch {epoch + 1}/{n_epoch}  REML -2logLik {self.loss[-1]:.4f}"

            if n_valid:
                # BLUP of the validation rows from the training rows, network frozen
                self.network.eval()
                with torch.no_grad():
                    Z = self.embed(self.keep)
                    G, R = self.varmeth()
                    g = G.diagonal()
                    V = (Z[train] * g) @ Z[train].T + R[train][:, train]
                    r = y[train] - X[train] @ self.beta.to(self.dtype)
                    pred = X[valid] @ self.beta.to(self.dtype) + (Z[valid] * g) @ Z[train].T @ torch.linalg.solve(V, r)
                    self.score.append(((y[valid] - pred) ** 2).mean().item())
                message += f"  validation MSE {self.score[-1]:.4f}"

                if self.score[-1] < best_mse:
                    best_mse, best_epoch = self.score[-1], epoch
                    best = (
                        copy.deepcopy(self.network.state_dict()),
                        self.beta.detach().clone(),
                        [p.detach().clone() for p in self.params],
                    )

            if verbose:
                print(message)
            if n_valid and patience is not None and epoch - best_epoch >= patience:
                break

        if n_valid:
            state, beta, params = best
            self.network.load_state_dict(state)
            self.beta.data.copy_(beta)
            for p, value in zip(self.params, params):
                p.data.copy_(value)

        # final embedding with the network frozen, then BLUE / BLUP / PEV
        self.network.eval()
        with torch.no_grad():
            Z = self.embed()
        Z_G = Z[:, :self.d]
        self._Z = Z[self.keep]

        # polish
        self.REML()
        self.HMME()

        uhat = self.uhat.detach().to(self.dtype)
        self.uhat_general = uhat[:self.d].flatten()
        self.var_G = self.params[0].exp().item()
        self.var_R = self.params[1].exp().item()
        # variance ratio lambda = var(embedding dimension) / var(unit), on every row of data:
        # 1 / sum_j var(z_j), the embedding analog of 1 / sum_i 2 p_i (1 - p_i) for SNPs
        self.var_ratio = 1 / Z_G.var(dim=0).sum().item()
        self.var_G_unit = self.var_G / self.var_ratio
        self.var_R_unit = self.var_R
        # genetic value of every row of data, missing responses included
        self.prediction = (Z @ uhat).flatten()

        # PCA of the raw and weighted embeddings
        weighted = {
            "Raw embeddings": Z_G,
            "General weighted embeddings": Z_G * self.uhat_general,
        }
        # one block of uhat, one variance per specificity, in the order of self.specificity
        self.uhat_specific, self.var_S, self.var_S_unit = {}, {}, {}
        start = self.d
        for k, s in enumerate(self.specificity):
            n = len(self.levels[s])
            self.uhat_specific[s] = uhat[start:start + n * self.d].reshape(n, self.d)
            start += n * self.d
            self.var_S[s] = self.params[2 + k].exp().item()
            self.var_S_unit[s] = self.var_S[s] / self.var_ratio
            weighted[f"Specific weighted embeddings ({s})"] = Z_G * self.uhat_specific[s][self.codes[s]]

        self.pca = {}
        for title, P in weighted.items():
            Pc = P - P.mean(dim=0)
            eigval, eigvec = torch.linalg.eigh(Pc.T @ Pc / (len(Pc) - 1))
            eigval, eigvec = eigval.flip(0), eigvec.flip(1)
            self.pca[title] = {
                "coords": (Pc @ eigvec).cpu().numpy(),
                "inertia": (100 * eigval / eigval.sum()).cpu().numpy(),
                "eigvec": eigvec.cpu().numpy(),
            }

        return self

    def varcomp(self):
        """Raw variances (per embedding dimension) and unit variances (per individual, GBLUP-like)."""
        rows = {"G": [self.var_G, self.var_G_unit]}
        rows |= {f"S:{s}": [self.var_S[s], self.var_S_unit[s]] for s in self.specificity}
        rows["R"] = [self.var_R, self.var_R_unit]
        return pd.DataFrame.from_dict(rows, orient="index", columns=["raw", "unit"])

    def plot(self, plan: tuple[int, int] = (0, 1), focus: int | str | None = None):
        """
        Without specificity: raw and general weighted embeddings, monochrome.
        One specificity, or `focus` (index or name of a specificity): raw, general and specific
        weighted embeddings colored by its groups, and its legend.
        Several specificities without `focus`: raw and general weighted embeddings monochrome,
        then one row per specificity, its specific weighted embeddings and its legend.
        """
        i, j = plan

        if focus is None and len(self.specificity) == 1:
            focus = 0
        if isinstance(focus, int):
            if not -len(self.specificity) <= focus < len(self.specificity):
                raise ValueError(f"focus {focus} out of range, {len(self.specificity)} specificities")
            focus = self.specificity[focus]
        if focus is not None and focus not in self.specificity:
            raise ValueError(f"focus '{focus}' is not a specificity: {self.specificity}")

        def scatter(ax, title, s=None):
            coords = self.pca[title]["coords"]
            if s is None:
                ax.scatter(coords[:, i], coords[:, j], color="C0", alpha=0.7)
            else:
                # continuous palette: one distinct color per level, however many levels
                colors = plt.cm.turbo(np.random.default_rng(0).permutation(np.linspace(0, 1, len(self.levels[s]))))
                for g, level in enumerate(self.levels[s]):
                    ax.scatter(coords[self.codes[s] == g, i], coords[self.codes[s] == g, j], color=colors[g], label=level, alpha=0.7)
            ax.axhline(0, color="grey", lw=0.5)
            ax.axvline(0, color="grey", lw=0.5)
            ax.set_xlabel(f"PC{i + 1} ({self.pca[title]['inertia'][i]:.1f} %)")
            ax.set_ylabel(f"PC{j + 1} ({self.pca[title]['inertia'][j]:.1f} %)")
            ax.set_title(f"Individual projections\n{title}")

        def legend(ax, source, s):
            ax.axis("off")
            ax.legend(
                *source.get_legend_handles_labels(),
                loc="center left",
                ncol=1 + len(self.levels[s]) // 25,
                fontsize=8,
                frameon=False,
            )

        general = ["Raw embeddings", "General weighted embeddings"]
        if focus is not None:
            fig, axes = plt.subplots(1, 4, figsize=(21, 6), width_ratios=[6, 6, 6, 3])
            for ax, title in zip(axes, general + [f"Specific weighted embeddings ({focus})"]):
                scatter(ax, title, focus)
            legend(axes[3], axes[0], focus)
        else:
            n_rows = 1 + len(self.specificity)
            fig, axes = plt.subplots(n_rows, 2, figsize=(12, 6 * n_rows), squeeze=False)
            for ax, title in zip(axes[0], general):
                scatter(ax, title)
            for (ax, ax_legend), s in zip(axes[1:], self.specificity):
                scatter(ax, f"Specific weighted embeddings ({s})", s)
                legend(ax_legend, ax, s)
        plt.tight_layout()
        plt.show()
