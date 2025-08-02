from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from botorch.models import SingleTaskGP


class VanillaGP(SingleTaskGP):
    """
    A wrapper around SingleTaskGP that always uses ScaleKernel(RBFKernel(ard_num_dims=dim)).
    """
    def __init__(
        self,
        train_X,
        train_Y,
        covar_module=None, # Optional: Receive pre-fitted covariance module
        mean_module=None,  # Optional: Receive pre-fitted mean module
        outcome_transform=None, # None ensures no transformation is applied
        input_transform=None,
        likelihood= GaussianLikelihood(),
    ):
        # Create covariance and mean modules with ScaleKernel for consistency
        ard_dims = train_X.shape[-1]
        if covar_module is None:
            covar_module = ScaleKernel(RBFKernel(ard_num_dims=ard_dims))
        if mean_module is None:
            mean_module = ConstantMean()

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            covar_module=covar_module,
            mean_module=mean_module,
            likelihood=likelihood,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )