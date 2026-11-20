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
    y = X beta + Z u_G + Z_S u_S + e, with u_G ~ N(0, s2_G I), u_S ~ N(0, s2_S I),
    e ~ N(0, s2_R I). Z is the embedding returned by the network Z(), rows
    projected on the unit sphere (||z_i||^2 = 1) so that the variances are
    identifiable. Z_S = [Z * 1{group A} | Z * 1{group B} | ...], with the groups
    given by the qualitative column `specificity`; without it, the model reduces
    to y = X beta + Z u_G + e. The network is trained jointly with beta and the
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
        specificity: str | None = None,
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
        if specificity is None:
            codes, levels, groups = np.zeros(len(data), dtype=int), [None], None
        else:
            if pd.api.types.is_numeric_dtype(data[specificity]):
                raise ValueError(f"specificity column '{specificity}' must be qualitative, got {data[specificity].dtype}")
            if data[specificity].isna().any():
                raise ValueError(f"specificity column '{specificity}' has {data[specificity].isna().sum()} missing values")
            codes, levels = pd.factorize(data[specificity])
            groups = torch.tensor(codes[:, None] == np.arange(len(levels)), dtype=dtype, device=device)

        dm = patsy.dmatrix(fixed, data=data[labelled], NA_action="drop", return_type="dataframe")
        keep = torch.as_tensor(data.index.isin(dm.index), device=device)

        y = torch.tensor(data.loc[dm.index, response].to_numpy(), dtype=dtype, device=device).reshape(-1, 1)
        X = torch.tensor(dm.to_numpy(), dtype=dtype, device=device)

        d = Z_all.shape[1]
        n_groups = 0 if groups is None else groups.shape[1]

        log_v0 = float(np.log(y.var().item()))
        log_sG = torch.tensor(log_v0, dtype=dtype, device=device, requires_grad=True)
        log_sR = torch.tensor(log_v0, dtype=dtype, device=device, requires_grad=True)
        log_sS = torch.tensor(log_v0, dtype=dtype, device=device, requires_grad=True)
        varparams = [log_sG, log_sR] + ([log_sS] if n_groups else [])

        # pyreml picks the path: Woodbury (varmeth_inv) if q < n, direct (varmeth) otherwise
        def varmeth(self):
            g = torch.cat([torch.exp(log_sG).repeat(d), torch.exp(log_sS).repeat(n_groups * d)])
            R = torch.exp(log_sR) * torch.eye(self.n, dtype=self.dtype, device=self.device)
            return torch.diag(g), R

        def varmeth_inv(self):
            ginv = torch.cat([torch.exp(-log_sG).repeat(d), torch.exp(-log_sS).repeat(n_groups * d)])
            Rinv = torch.exp(-log_sR) * torch.eye(self.n, dtype=self.dtype, device=self.device)
            return torch.diag(ginv), Rinv, d * log_sG + n_groups * d * log_sS, self.n * log_sR

        mm = cls(
            y = y,
            X = X,
            q = d * (1 + n_groups),
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
        mm.levels = list(levels)
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
        if self.groups is not None:
            g = self.groups if rows is None else self.groups[rows]
            Z = torch.cat([Z, (g[:, :, None] * Z[:, None, :]).flatten(1)], dim=1)
        return Z

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

            # one REML step on the training rows, network frozen: beta and variances recalibrated
            t = train.sort().values
            rows = torch.zeros_like(self.keep)
            rows[labelled[t]] = True
            self.network.eval()
            with torch.no_grad():
                self._Z = self.embed(rows)
            self._y, self._X, self.n = y[t], X[t], len(t)
            self.opti_REML.set_lbfgs()
            self.REML(n_epoch=1)
            self._y, self._X, self.n = y, X, n


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
        if self.groups is not None:
            self.uhat_specific = uhat[self.d:].reshape(len(self.levels), self.d)
            self.var_S = self.params[2].exp().item()
            self.var_S_unit = self.var_S / self.var_ratio
            weighted["Specific weighted embeddings"] = Z_G * self.uhat_specific[self.codes]

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
        varcomp = pd.DataFrame(
            {"raw": [self.var_G, self.var_R], "unit": [self.var_G_unit, self.var_R_unit]},
            index=["G", "R"],
        )
        if self.groups is not None:
            varcomp.loc["S"] = [self.var_S, self.var_S_unit]
        return varcomp

    def plot(self, plan: tuple[int, int] = (0, 1)):
        i, j = plan

        # one PCA panel per embedding, plus a dedicated legend panel when there are groups
        n_pca, legend = len(self.pca), self.specificity is not None
        fig, axes = plt.subplots(
            1, n_pca + legend,
            figsize=(6 * n_pca + 3 * legend, 6),
            width_ratios=[6] * n_pca + [3] * legend,
            squeeze=False,
        )
        # continuous palette: one distinct color per level, however many levels
        colors = plt.cm.turbo(np.random.default_rng(0).permutation(np.linspace(0, 1, len(self.levels))))
        for ax, (title, pca) in zip(axes[0], self.pca.items()):
            for g, level in enumerate(self.levels):
                ax.scatter(pca["coords"][self.codes == g, i], pca["coords"][self.codes == g, j], color=colors[g], label=level, alpha=0.7)
            ax.axhline(0, color="grey", lw=0.5)
            ax.axvline(0, color="grey", lw=0.5)
            ax.set_xlabel(f"PC{i + 1} ({pca['inertia'][i]:.1f} %)")
            ax.set_ylabel(f"PC{j + 1} ({pca['inertia'][j]:.1f} %)")
            ax.set_title(f"Individual projections\n{title}")
        if legend:
            axes[0, -1].axis("off")
            axes[0, -1].legend(
                *axes[0, 0].get_legend_handles_labels(),
                loc="center left",
                ncol=1 + len(self.levels) // 25,
                fontsize=8,
                frameon=False,
            )
        plt.tight_layout()
        plt.show()
