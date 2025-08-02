from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean

from .base_noiseinjection_gp import BaseNoiseInjectionGP
from .noise_optimizers import (
    NaiveOptimizer,
    IteratedLocalSearchOptimizer,
    BeamSearchOptimizer,
    GradientOptimizer,
    BinaryGradientOptimizer,
)


class HeuristicNoiseInjectionGP(BaseNoiseInjectionGP):
    """
    A heuristic noise injection GP model that uses a heuristic to determine the noise factors.
    This is useful for models that use noise injection during training.
    """
    def __init__(
        self,
        train_X,
        train_Y,
        outcome_transform=None,
        initial_injection="zeros",
        heuristic_algorithm=None,
        refit_gp=False,
        heuristic_hyperparameters=None,
    ):
        """
        Initialize the HeuristicNoiseInjectionGP model.
        
        Args:
            train_X: Training input data
            train_Y: Training target data
            outcome_transform: Optional outcome transform
            input_transform: Optional input transform
            persistent_model: Whether to keep the model in memory after training
            initial_injection: Initial noise injection strategy ("zeros" or "ones")
        """

        self.heuristic_algorithm = heuristic_algorithm
        self.refit_gp = refit_gp
        self.heuristic_hyperparameters = heuristic_hyperparameters or {}

        # Same covar and mean modules for noise optimization as in VanillaGP
        # Create modules as local variables first, then pass to parent
        covar_module = ScaleKernel(RBFKernel(ard_num_dims=train_X.shape[-1]))
        mean_module = ConstantMean()

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            covar_module=covar_module,
            mean_module=mean_module,
            outcome_transform=outcome_transform,
            initial_injection=initial_injection,
        )

    def _optimize_noise(self, covar_module, mean_module):
        """
        Optimize the noise factors using the specified heuristic strategy.
        """
        # Create optimizer based on selected algorithm
        if self.heuristic_algorithm == "naive_mll":
            optimizer = NaiveOptimizer(
                self.train_X,
                self.train_Y,
                covar_module,
                mean_module,
                self.initial_injection,
                self.refit_gp,
            )
        elif self.heuristic_algorithm == "ils_mll":
            optimizer = IteratedLocalSearchOptimizer(
                self.train_X,
                self.train_Y,
                covar_module,
                mean_module,
                self.initial_injection,
                self.refit_gp,
                **self.heuristic_hyperparameters,
            )
        elif self.heuristic_algorithm == "bs_mll":
            optimizer = BeamSearchOptimizer(
                self.train_X,
                self.train_Y,
                covar_module,
                mean_module,
                self.initial_injection,
                self.refit_gp,
                **self.heuristic_hyperparameters,
            )
        elif self.heuristic_algorithm == "gradient_mll":
            optimizer = GradientOptimizer(
                self.train_X,
                self.train_Y,
                covar_module,
                mean_module,
                self.initial_injection,
                **self.heuristic_hyperparameters,
            )
        elif self.heuristic_algorithm == "binarygradient_mll":
            optimizer = BinaryGradientOptimizer(
                self.train_X,
                self.train_Y,
                covar_module,
                mean_module,
                self.initial_injection,
                **self.heuristic_hyperparameters,
            )
        else:
            raise ValueError(f"Unknown heuristic algorithm: {self.heuristic_algorithm}")

        # Optimize and return factors
        return optimizer.optimize()