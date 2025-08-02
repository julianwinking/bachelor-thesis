from typing import Any, Optional

import torch
from linear_operator.operators import DiagLinearOperator
from torch import Tensor

from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.priors import Prior
from gpytorch.likelihoods.noise_models import _HomoskedasticNoiseBase
from gpytorch.likelihoods import _GaussianLikelihoodBase


class MultiplicativeNoise(_HomoskedasticNoiseBase):
    def __init__(self, factors, noise_prior=None, noise_constraint=None, batch_shape=torch.Size(), preserve_factors_grad=False):
        super().__init__(
            noise_prior=noise_prior, noise_constraint=noise_constraint, batch_shape=batch_shape, num_tasks=1
        )

        # ensure factors is a tensor
        if not torch.is_tensor(factors):
            factors = torch.tensor(factors)

        # ensure factors is positive
        if (factors < 0).any():
            raise ValueError("factors should be positive")

        # ensure factors is 1D
        factors = factors.reshape(-1)

        # Handle gradient preservation properly
        if preserve_factors_grad and factors.requires_grad:
            # Keep gradients if requested and available
            self.factors = factors
        else:
            # Detach and disable gradients for non-gradient cases
            self.factors = factors.detach().requires_grad_(False)

    def set_factors(self, factors, preserve_grad=False):
        if not torch.is_tensor(factors):
            factors = torch.tensor(factors)

        if (factors < 0).any():
            raise ValueError("factors should be positive")

        factors = factors.reshape(-1)
        
        if preserve_grad and factors.requires_grad:
            self.factors = factors
        else:
            self.factors = factors.detach().requires_grad_(False)

    def forward(self, *params: Any, shape: Optional[torch.Size] = None, **kwargs: Any) -> DiagLinearOperator:
        if shape is None:
            p = params[0] if torch.is_tensor(params[0]) else params[0][0]
            shape = p.shape if len(p.shape) == 1 else p.shape[:-1]
        noise = self.noise
        *batch_shape, n = shape
        noise_batch_shape = noise.shape[:-1] if noise.dim() > 1 else torch.Size()
        num_tasks = noise.shape[-1]
        batch_shape = torch.broadcast_shapes(noise_batch_shape, batch_shape)

        noise = noise.unsqueeze(-2)
        noise_diag = noise.expand(*batch_shape, 1, num_tasks).contiguous()
        if num_tasks == 1:
            noise_diag = noise_diag.view(*batch_shape, 1)
        if noise_diag.shape[-1] != 1:
            noise_diag = noise_diag.unsqueeze(-1)

        return DiagLinearOperator(noise_diag * self.factors)


class MultiplicativeGaussianLikelihood(_GaussianLikelihoodBase):
    r"""
    The standard likelihood for regression.
    Assumes a standard homoskedastic noise model:

    .. math::
        p(y \mid f) = f + \epsilon, \quad \epsilon \sim \mathcal N (0, \sigma^2)

    where :math:`\sigma^2` is a noise parameter.

    .. note::
        This likelihood can be used for exact or approximate inference.

    .. note::
        GaussianLikelihood has an analytic marginal distribution.

    :param noise_prior: Prior for noise parameter :math:`\sigma^2`.
    :param noise_constraint: Constraint for noise parameter :math:`\sigma^2`.
    :param batch_shape: The batch shape of the learned noise parameter (default: []).
    :param kwargs:

    :ivar torch.Tensor noise: :math:`\sigma^2` parameter (noise)
    """

    def __init__(
        self,
        factors,
        noise_prior: Optional[Prior] = None,
        noise_constraint: Optional[Interval] = None,
        batch_shape: torch.Size = torch.Size(),
        preserve_factors_grad: bool = False,
        **kwargs: Any,
    ) -> None:
        noise_covar = MultiplicativeNoise(
            factors=factors,
            noise_prior=noise_prior,
            noise_constraint=noise_constraint,
            batch_shape=batch_shape,
            preserve_factors_grad=preserve_factors_grad
        )
        super().__init__(noise_covar=noise_covar)

    @property
    def noise(self) -> Tensor:
        return self.noise_covar.noise

    @noise.setter
    def noise(self, value: Tensor) -> None:
        self.noise_covar.initialize(noise=value)

    @property
    def raw_noise(self) -> Tensor:
        return self.noise_covar.raw_noise

    @raw_noise.setter
    def raw_noise(self, value: Tensor) -> None:
        self.noise_covar.initialize(raw_noise=value)

    def marginal(self, function_dist: MultivariateNormal, *args: Any, **kwargs: Any) -> MultivariateNormal:
        r"""
        :return: Analytic marginal :math:`p(\mathbf y)`.
        """
        return super().marginal(function_dist, *args, **kwargs)