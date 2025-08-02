import time

import torch
from .vanilla_gp import VanillaGP

from ..multiplicative_gaussian_likelihood import MultiplicativeGaussianLikelihood
from ..utils.constants import NOISE_JITTER


class BaseNoiseInjectionGP(VanillaGP):
    """
    A wrapper around VanillaGP that adds noise injection to the training data.
    This is useful for models that use noise injection during training.
    """
    
    def __init__(
            self,
            train_X,
            train_Y,
            covar_module=None,
            mean_module=None,
            outcome_transform=None,
            initial_injection="no_noise",
        ):
        # Store training data for use in noise optimization
        self.train_X = train_X
        self.train_Y = train_Y

        self.dim = train_X.shape[-1]
        self.num_train = self.train_X.shape[0]
        self.initial_injection = initial_injection
        
        # Start timing the noise optimization
        noise_optimization_start = time.time()

        # Optimize the noise factors and get fitted modules
        self.noise_factors = self._optimize_noise(covar_module, mean_module)

        # End noise optimization time
        self.noise_optimization_time = time.time() - noise_optimization_start

        # Extract fitted modules from noise optimization (if available)
        fitted_covar_module, fitted_mean_module = self._get_fitted_modules()

        # Use fitted modules if available, otherwise use original modules
        final_covar_module = fitted_covar_module if fitted_covar_module is not None else covar_module
        final_mean_module = fitted_mean_module if fitted_mean_module is not None else mean_module

        # Create a VanillaGP with the fitted modules (not a MultiplicativeGaussianLikelihood)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=MultiplicativeGaussianLikelihood(factors=self.noise_factors),
            outcome_transform=outcome_transform,
            covar_module=final_covar_module,
            mean_module=final_mean_module,
        )

    def _optimize_noise(self, covar_module=None, mean_module=None):
        """
        Optimize the noise factors for the model.
        This method can be overridden to implement specific noise optimization strategies.
        """
        
        # Initialize factors to standard noise injection value for each training point
        if self.initial_injection == "no_noise":
            injection_value = NOISE_JITTER
        elif self.initial_injection == "noise":
            injection_value = 1.0
        else:
            raise ValueError(f"Unknown initial injection type: {self.initial_injection}")
        
        noise_factors = torch.full((self.num_train,), injection_value)

        return noise_factors

    def _get_fitted_modules(self):
        """
        Extract fitted covariance and mean modules from the noise optimization process.
        Should be overridden by subclasses that perform hyperparameter fitting during noise optimization.
        
        Returns:
            tuple: (fitted_covar_module, fitted_mean_module) or (None, None) if no fitted modules available
        """
        return None, None