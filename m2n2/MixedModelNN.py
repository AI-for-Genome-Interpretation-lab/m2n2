import copy

import numpy as np
import pandas as pd
import patsy
import scipy.linalg
import torch
import torch.nn.functional as F

from pyreml import MixedModel, Random, Residual
from .Embedding import Embedding
from .RandomNN import RandomNN


class MixedModelNN(MixedModel):
    """
    y = X beta + sum_j Z_j u_j + e. Each random effect is either a RandomNN, built upon one of the
    networks of `embedding`, or a pyreml Random; e follows the pyreml Residual. The networks are
    trained jointly with beta and the variances on the REML loss.

    The model is first built by pyreml without the RandomNN terms: it designs the Random terms and
    the residual, and is kept to initialize beta and their variances at fit. It is then rebuilt
    by the low-level constructor, with varmeth and varmeth_inv stacking every term, in the order of
    `random`. Rows with a missing response are left out of the fit, but the RandomNN terms still
    predict them. The mixed model runs in double, on the device of the networks output; the networks
    keep their own dtype.
    """

    @classmethod
    def from_dataframe(
        cls,
        data: pd.DataFrame,
        response: str,
        embedding: dict[str, Embedding],
        random: list[RandomNN | Random],
        fixed: str = "1",
        residual: Residual | None = None,
        lr_beta: float = 1e-3,
        lr_var: float = 1e-3,
    ):
        data = data.reset_index(drop=True)
        residual = Residual() if residual is None else residual

        # eval mode: these shape checks must not update the BatchNorm running stats
        outputs = {}
        for name, emb in embedding.items():
            emb.network.eval()
            with torch.no_grad():
                outputs[name] = emb.network()
            if outputs[name].shape[0] != len(data):
                raise ValueError(f"embedding '{name}' returns {outputs[name].shape[0]} rows, data has {len(data)}")
        if len({o.device for o in outputs.values()}) > 1:
            raise ValueError(f"the networks must share their device, got {[o.device for o in outputs.values()]}")
        device = next(iter(outputs.values())).device

        # rows of the fit, as pyreml filters them: response, fixed and Random covariates complete
        randoms = [term for term in random if not isinstance(term, RandomNN)]
        keep = data[response].notna()
        for formula in [fixed] + [term.formula for term in randoms]:
            keep &= data.index.isin(patsy.dmatrix(formula, data=data, NA_action="drop", return_type="dataframe").index)
        for term in randoms:
            keep &= data[[term.unit] if isinstance(term.unit, str) else list(term.unit)].notna().all(axis=1)

        pm = MixedModel.from_dataframe(data[keep], response, fixed=fixed, random=randoms, residual=residual, device=str(device))
        if pm.n != keep.sum():
            raise ValueError(f"pyreml kept {pm.n} rows, {keep.sum()} expected")

        for term in random:
            if isinstance(term, RandomNN):
                term.design(data, embedding[term.embedding].network, outputs[term.embedding].shape[1], device)

        # pyreml picks the path: Woodbury (varmeth_inv) if q < n, direct (varmeth) otherwise
        def varmeth(self):
            G = torch.block_diag(*[term.varmeth()() for term in self.random])
            R = self.residual.varmeth()()[self.w_grid][:, self.w_grid]
            return G, R

        def varmeth_inv(self):
            inv_logdets = [term.varmeth_inv()() for term in self.random]
            Ginv = torch.block_diag(*[gi for gi, _ in inv_logdets])
            Rinv, logdet_R = self.residual.varmeth_inv()(self.w_grid)
            return Ginv, Rinv, sum(ld for _, ld in inv_logdets), logdet_R

        mm = cls(
            y = pm.y,
            X = pm.X,
            q = sum(term.k * term.c * term.L for term in random),
            params      = pm.params + [term.log_var for term in random if isinstance(term, RandomNN)],
            varmeth     = varmeth,
            varmeth_inv = varmeth_inv,
            device      = str(device),
        )

        mm.response = [response]
        mm.fixed_names = pm.fixed_names
        mm.w_grid = pm.w_grid
        mm.random = random
        mm.residual = residual
        mm.embedding = embedding
        mm.pyreml = pm  # shares the Random terms, the residual and their variances
        mm.keep = torch.tensor(keep.to_numpy(), device=device)
        # the design of the Random terms is fixed: their dense incidence on the rows of the fit
        mm.Z_random = {
            i: torch.as_tensor(term.Z, dtype=torch.double, device=device)
            for i, term in enumerate(random) if not isinstance(term, RandomNN)
        }
        mm.migrate(torch.double)

        mm.optimizers = [
            emb.optimizer([
                {"params": # network weights (matrices, kernels): shrunk
                    [p for p in emb.network.parameters() if p.ndim >= 2],
                },
                {"params": # network biases and normalization parameters (vectors): free
                    [p for p in emb.network.parameters() if p.ndim < 2],
                    "weight_decay": 0.0,
                },
            ])
            for emb in embedding.values()
        ] + [
            torch.optim.Adam([
                {"params": [mm.beta], "lr": lr_beta},
                {"params": mm.params, "lr": lr_var},
            ])
        ]

        mm.loss = []
        mm.score = []

        return mm

    def design(self, b: torch.Tensor | None = None) -> torch.Tensor:
        """
        Random design [Z_1 | Z_2 | ...] of the labelled rows at positions b (sorted, None = all).
        Each network is called once, on these rows alone.
        """
        labelled = self.keep.nonzero().flatten()  # data row of each labelled row
        rows = self.keep.clone()
        if b is not None:
            rows[:] = False
            rows[labelled[b]] = True

        embedded = {name: F.normalize(emb.network(rows), dim=1).to(self.dtype) for name, emb in self.embedding.items()}
        return torch.cat([
            term.block(embedded[term.embedding], rows) if isinstance(term, RandomNN)
            else (self.Z_random[i] if b is None else self.Z_random[i][b])
            for i, term in enumerate(self.random)
        ], dim=1)

    def fit(
        self,
        n_epoch: int = 1000,
        batch_size: int | None = None,
        validation: float | None = None,
        patience: int | None = 50,
        verbose: bool = True,
        full_REML: bool = True,
    ):
        """
        full_REML: initialization by pyreml and REML polish; False skips both, for datasets too large for them.
        batch_size: rows per minibatch, None = full batch; the incomplete last batch is dropped.
        validation: fraction of the labelled rows held out (None or 0 = none) to score each epoch by the
        MSE of their BLUP from the training rows; the best state is restored at the end, and the training
        stops after `patience` epochs without improvement (never if patience is None).
        """
        # initialization: beta and the variances of the Random terms and of the residual by pyreml,
        # without the RandomNN terms (full_REML), or by OLS and their pyreml initial values
        if full_REML:
            self.pyreml.fit(verbose=False)
            self.beta.data.copy_(self.pyreml.beta.data)
        else:
            self.OLS(terminate=False)
        for term in self.random:
            if isinstance(term, RandomNN):
                term.log_var.data.fill_(np.log(self.y.var().item()))

        # full labelled data; minibatches swap _y, _X, _Z, n, w_grid, beta and p, restored after each epoch
        y, X, n, w_grid, beta_full, p = self._y, self._X, self.n, self.w_grid, self.beta, self.p
        networks = [emb.network for emb in self.embedding.values()]

        perm = torch.randperm(n, device=self.keep.device)
        n_valid = 0 if validation is None else round(validation * n)
        valid, train = perm[:n_valid], perm[n_valid:]
        size = len(train) if batch_size is None else min(batch_size, len(train))

        best_mse, best_epoch = float("inf"), 0
        for epoch in range(n_epoch):

            for network in networks:
                network.train()
            losses = []
            for b in train[torch.randperm(len(train), device=train.device)].split(size):
                if len(b) < size:
                    continue
                b = b.sort().values
                # X_b loses rank when a level is absent from the batch (an empty column, or the reference
                # level of a qualitative effect, the intercept being then the sum of the present dummies):
                # its X'V^-1 X is singular. A basis of its columns is kept, by pivoted QR. Indexing beta
                # keeps the gradient on the full parameter
                R, piv = scipy.linalg.qr(X[b].cpu().numpy(), mode="r", pivoting=True)
                d = np.abs(np.diag(R))
                rank = int((d > d[0] * max(R.shape) * np.finfo(R.dtype).eps).sum())
                cols = torch.as_tensor(np.sort(piv[:rank]), device=X.device)
                self._y, self._X, self.n, self.w_grid = y[b], X[b][:, cols], len(b), w_grid[b]
                self.beta, self.p = beta_full[cols], rank
                self._Z = self.design(b)
                for optimizer in self.optimizers:
                    optimizer.zero_grad()
                loss = self.REML_loss()
                loss.backward()
                for optimizer in self.optimizers:
                    optimizer.step()
                losses.append(loss.item())
            self._y, self._X, self.n, self.w_grid = y, X, n, w_grid
            self.beta, self.p = beta_full, p
            self.loss.append(float(np.mean(losses)))
            message = f"epoch {epoch + 1}/{n_epoch}  REML -2logLik {self.loss[-1]:.4f}"

            if n_valid:
                # BLUP of the validation rows from the training rows, networks frozen
                for network in networks:
                    network.eval()
                with torch.no_grad():
                    Z = self.design()
                    G, R = self.varmeth()
                    V = Z[train] @ G @ Z[train].T + R[train][:, train]
                    beta = self.beta.to(self.dtype)
                    r = y[train] - X[train] @ beta
                    pred = X[valid] @ beta + Z[valid] @ G @ Z[train].T @ torch.linalg.solve(V, r)
                    self.score.append(((y[valid] - pred) ** 2).mean().item())
                message += f"  validation MSE {self.score[-1]:.4f}"

                if self.score[-1] < best_mse:
                    best_mse, best_epoch = self.score[-1], epoch
                    best = (
                        [copy.deepcopy(network.state_dict()) for network in networks],
                        self.beta.detach().clone(),
                        [p.detach().clone() for p in self.params],
                    )

            if verbose:
                print(message)
            if n_valid and patience is not None and epoch - best_epoch >= patience:
                break

        if n_valid:
            states, beta, params = best
            for network, state in zip(networks, states):
                network.load_state_dict(state)
            self.beta.data.copy_(beta)
            for p, value in zip(self.params, params):
                p.data.copy_(value)

        # networks frozen, then polish: REML if full_REML, BLUE / BLUP / PEV
        for network in networks:
            network.eval()
        with torch.no_grad():
            self._Z = self.design()
            embedded = {name: F.normalize(emb.network(), dim=1).to(self.dtype) for name, emb in self.embedding.items()}

        if full_REML:
            self.REML()
        self.HMME()

        for term in self.random:
            term.format_variance()
        self.residual.format_variance()
        for term in self.random:
            if isinstance(term, RandomNN):
                term.summarize(embedded[term.embedding])

        return self

    def compute_AIC(self, REML=True):
        # called by HMME: no AIC, the networks are not fitted by maximum likelihood
        pass

    def varcomp(self):
        """Raw variances (per embedding dimension for a RandomNN) and unit variances (per individual, GBLUP-like)."""
        rows = {}
        for term in self.random:
            if isinstance(term, RandomNN):
                rows[term.label] = [term.var, term.var_unit]
            else:
                var = float(np.squeeze(term.variance["sigma"]))
                rows[term.effect_label] = [var, var]
        var_R = float(np.squeeze(self.residual.variance["sigma"]))
        rows["R"] = [var_R, var_R]
        return pd.DataFrame.from_dict(rows, orient="index", columns=["raw", "unit"])
