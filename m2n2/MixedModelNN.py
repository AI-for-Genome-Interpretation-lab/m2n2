import copy
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    `random`. With several responses, y stacks them response by response, as in pyreml: each
    observation is a (response, individual) pair. Individuals with no response are left out of the
    fit, but the RandomNN terms still predict them. The mixed model runs in double, on the device of the networks output; the networks
    keep their own dtype.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        response: str | list[str],
        embedding: dict[str, Embedding],
        random: list[RandomNN | Random],
        fixed: str = "1",
        residual: Residual | None = None,
    ):
        data = data.reset_index(drop=True)
        responses = [response] if isinstance(response, str) else list(response)
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

        # rows of the fit, as pyreml filters them: a response, fixed and Random covariates complete
        randoms = [term for term in random if not isinstance(term, RandomNN)]
        keep = data[responses].notna().any(axis=1)
        for formula in [fixed] + [term.formula for term in randoms]:
            keep &= data.index.isin(patsy.dmatrix(formula, data=data, NA_action="drop", return_type="dataframe").index)
        for term in randoms:
            keep &= data[[term.unit] if isinstance(term.unit, str) else list(term.unit)].notna().all(axis=1)

        pm = MixedModel.from_dataframe(data[keep], responses, fixed=fixed, random=randoms, residual=residual, device=str(device))
        masks = [data.loc[keep, r].notna().to_numpy() for r in responses]  # observed rows of each response
        if pm.n != sum(m.sum() for m in masks):
            raise ValueError(f"pyreml kept {pm.n} observations, {sum(m.sum() for m in masks)} expected")

        for term in random:
            if isinstance(term, RandomNN):
                network, dim = embedding[term.embedding].network, outputs[term.embedding].shape[1]
                term.design(data, responses, network, dim, device, residual.scale)

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

        super().__init__(
            y = pm.y,
            X = pm.X,
            q = sum(term.k * term.c * term.L for term in random),
            params      = pm.params + [vp["tensor"] for term in random if isinstance(term, RandomNN) for vp in term.varparams],
            varmeth     = varmeth,
            varmeth_inv = varmeth_inv,
            device      = str(device),
        )

        self.response = responses
        self.fixed_names = pm.fixed_names
        self.w_grid = pm.w_grid
        # each observation of y: its individual (residual level), its data row, its response
        self.obs_unit = pm.w_grid % residual.L
        self.obs_row = torch.as_tensor(np.flatnonzero(keep.to_numpy()), device=device)[self.obs_unit]
        self.obs_resp = pm.w_grid // residual.L
        self.n_data = len(data)
        self.random = random
        self.residual = residual
        self.embedding = embedding
        self.pyreml = pm  # shares the Random terms, the residual and their variances
        # the design of the Random terms is fixed: their dense incidence on the observations,
        # block diagonal over the responses, as in pyreml
        self.Z_random = {
            i: torch.as_tensor(scipy.linalg.block_diag(*[term.Z[m] for m in masks]), dtype=torch.double, device=device)
            for i, term in enumerate(random) if not isinstance(term, RandomNN)
        }
        self.migrate(torch.double)

        self.optimizers = [
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
                {"params": [self.beta], "lr": lr_beta},
                {"params": self.params, "lr": lr_var},
            ])
        ]
        self.schedulers = [
            factory(optimizer)
            for factory, optimizer in zip([emb.scheduler for emb in embedding.values()] + [scheduler], self.optimizers)
            if factory is not None
        ]

        self.loss = []
        self.score = []

    def reset(self, embedding: str, seed: int = 42):
        self.embedding[embedding].reset(seed=seed)

    def set_optimizer(self, embedding: str, optimizer: Callable):
        self.embedding[embedding].optimizer = optimizer

    def set_scheduler(self, embedding: str, scheduler: Callable | None):
        self.embedding[embedding].scheduler = scheduler

    def get_optimizer(self, embedding: str) -> Callable:
        return self.embedding[embedding].optimizer

    def get_scheduler(self, embedding: str) -> Callable | None:
        return self.embedding[embedding].scheduler

    def design(self, b: torch.Tensor | None = None) -> torch.Tensor:
        """
        Random design [Z_1 | Z_2 | ...] of the observations at positions b (sorted, None = all).
        Each network is called once, on the data rows of these observations alone.
        """
        row = self.obs_row if b is None else self.obs_row[b]
        resp = self.obs_resp if b is None else self.obs_resp[b]
        rows = torch.zeros(self.n_data, dtype=torch.bool, device=row.device)
        rows[row] = True
        pos = rows.cumsum(0) - 1  # position of each data row among the selected ones

        embedded = {
            name: F.normalize(emb.network(rows), dim=1).to(self.dtype)[pos[row]]
            for name, emb in self.embedding.items()
        }
        return torch.cat([
            term.block(embedded[term.embedding], row, resp) if isinstance(term, RandomNN)
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
        lr_beta: float = 1e-3,
        lr_var: float = 1e-3,
        scheduler: Callable | None = None,
    ):
        """
        full_REML: initialization by pyreml and REML polish; False skips both, for datasets too large for them.
        batch_size: individuals per minibatch, with all their responses, None = full batch; the incomplete
        last batch is dropped.
        validation: fraction of the labelled individuals held out (None or 0 = none) to score each epoch by
        the MSE of their BLUP from the training ones, standardized by the variance of each response with
        several responses; the best state is restored at the end, and the training stops after `patience`
        epochs without improvement (never if patience is None).
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
                term.log_S.data.copy_(term.log_S0)

        # full labelled data; minibatches swap _y, _X, _Z, n, w_grid, beta and p, restored after each epoch
        y, X, n, w_grid, beta_full, p = self._y, self._X, self.n, self.w_grid, self.beta, self.p
        networks = [emb.network for emb in self.embedding.values()]

        # individuals are split and batched, each with all its observations
        units = self.obs_unit
        perm = torch.randperm(int(units.max()) + 1, device=units.device)
        n_valid = 0 if validation is None else round(validation * len(perm))
        valid_units, train_units = perm[:n_valid], perm[n_valid:]
        valid = torch.isin(units, valid_units).nonzero().flatten()
        train = torch.isin(units, train_units).nonzero().flatten()
        size = len(train_units) if batch_size is None else min(batch_size, len(train_units))
        var_y = torch.stack([y[self.obs_resp == r].var() for r in range(len(self.response))])

        best_mse, best_epoch = float("inf"), 0
        for epoch in range(n_epoch):

            for network in networks:
                network.train()
            losses = []
            for batch in train_units[torch.randperm(len(train_units), device=units.device)].split(size):
                if len(batch) < size:
                    continue
                b = torch.isin(units, batch).nonzero().flatten()  # its observations, sorted
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
                    err, resp = ((y[valid] - pred) ** 2).flatten(), self.obs_resp[valid]
                    mse = torch.stack([err[resp == r].mean() for r in range(len(self.response))])
                    self.score.append((mse[0] if len(self.response) == 1 else (mse / var_y).mean()).item())
                message += f"  validation MSE {self.score[-1]:.4f}"

                if self.score[-1] < best_mse:
                    best_mse, best_epoch = self.score[-1], epoch
                    best = (
                        [copy.deepcopy(network.state_dict()) for network in networks],
                        self.beta.detach().clone(),
                        [p.detach().clone() for p in self.params],
                    )

            # once per epoch; ReduceLROnPlateau on the validation score, or on the loss without validation
            for scheduler in self.schedulers:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(self.score[-1] if n_valid else self.loss[-1])
                else:
                    scheduler.step()

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
        self.convergence = self.opti_REML.converged if full_REML else None
        self.HMME()

        for term in self.random:
            term.format_variance()
        self.residual.format_variance()
        for term in self.random:
            if isinstance(term, RandomNN):
                term.summarize(embedded[term.embedding])

        return self

    def plot(self):
        """Training curves: mean minibatch REML loss and validation MSE per epoch, the restored epoch dashed."""
        fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(10, 3.5))
        ax_loss.plot(self.loss)
        ax_loss.set_xlabel("epoch")
        ax_loss.set_ylabel("REML -2logLik")
        ax_score.plot(self.score)
        if self.score:
            ax_score.axvline(np.argmin(self.score), color="grey", linestyle="--")
        ax_score.set_xlabel("epoch")
        ax_score.set_ylabel("validation MSE")
        plt.tight_layout()
        plt.show()
        return fig

    def compute_AIC(self, REML=True):
        # called by HMME: no AIC, the networks are not fitted by maximum likelihood
        pass

    def varcomp(self):
        """
        Raw variances (per embedding dimension for a RandomNN) and unit variances (per individual,
        GBLUP-like), one row per term and per response, the residual last. The covariances across
        the responses are in the .variance of each term.
        """
        rows = {}
        for term in [*self.random, self.residual]:
            with torch.no_grad():
                raw = term.build_S_full().diagonal().cpu().numpy()  # natural units, one per component
            unit = raw / term.var_ratio if isinstance(term, RandomNN) else raw
            label = "R" if term is self.residual else term.effect_label
            for (resp, comp), r, u in zip(term.components, raw, unit):
                if term.d == 1:
                    rows[label] = [r, u]
                else:
                    rows[f"{label}[{resp}]" if term.c == 1 else f"{label}[{resp}, {comp}]"] = [r, u]
        return pd.DataFrame.from_dict(rows, orient="index", columns=["raw", "unit"])
