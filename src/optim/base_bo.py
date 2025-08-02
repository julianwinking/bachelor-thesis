import time
import torch
from typing import Dict, Any, Optional

import gpytorch
from torch.quasirandom import SobolEngine
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import normalize, unnormalize
from botorch.exceptions.errors import ModelFittingError

from abc import ABC, abstractmethod
from src.utils.utils import log


class BaseBO(ABC):
    """
    Base class for Bayesian Optimization implementations.
    
    Contains common functionality shared between different BO variants like VanillaBO and TrustRegionBO.
    Subclasses should implement the abstract methods to define their specific optimization behavior.
    """

    def __init__(
        self,
        objective_function,
        model_class,
        acqf_class,
        model_kwargs: Dict[str, Any] = None,
        acqf_kwargs: Dict[str, Any] = None,
        preprocessing_transform=None,
        verbose: bool = True,
    ):
        """
        Initialize the base Bayesian optimizer.
        
        Args:
            objective_function: The objective function to optimize (must have dim and bounds attributes)
            model_class: The GP model class to use
            acqf_class: The acquisition function class to use
            model_kwargs: Keyword arguments for model initialization
            acqf_kwargs: Keyword arguments for acquisition function
            preprocessing_transform: Optional preprocessing transform to apply to the data
            verbose: Whether to print progress information
        """
        self.objective_function = objective_function
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.acqf_class = acqf_class
        self.acqf_kwargs = acqf_kwargs or {}
        self.verbose = verbose
        self.preprocessing_transform = preprocessing_transform
        
        # Extract dimensions and bounds from objective function
        self.dim = objective_function.dim
        self.bounds = torch.tensor(objective_function.bounds, dtype=torch.float64)
        
        # Initialize counters and history
        self.n_evals = 0
        self.train_X = None
        self.train_Y = None
        self.train_Y_transformed = None
        self.best_x = None
        self.best_value = float('inf')
        self.best_value_transformed = float('inf')
        self.history_x = []
        self.history_y = []
        self.exploration_metrics = []
        self.noise_vectors = []  # Final noise vectors for each iteration (if applicable)
        self.best_mlls = []  # Best MLL per iteration
        
        # GP hyperparameter tracking
        self.lengthscales = []  # Lengthscale parameters for each iteration
        self.signal_variances = []  # Signal variance (outputscale) for each iteration
        self.noise_variances = []  # Noise variance for each iteration
        
        # Runtime tracking
        self.start_time = None
        self.iteration_times = []  # Individual iteration times
        self.cumulative_times = []  # Cumulative runtime at each iteration
        self.gp_fitting_times = []  # Time for GP hyperparameter optimization
        self.noise_optimization_times = []  # Time for noise vector optimization (if applicable)
        self.acqf_optimization_times = []  # Time for acquisition function optimization
        
        # Check if this is a trust region model and if it's beam search variant
        self._is_trust_region_model = 'TrustRegion' in str(self.model_class)
        self._is_beam_search_model = 'BeamSearch' in str(self.model_class)
        
        # Initialize beam search state for beam search models
        if self._is_beam_search_model:
            self.beam_search_state = None

    def _generate_initial_points(self, n_initial: int, x0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate initial points using Sobol sequence and include x0 if provided.
        
        Args:
            n_initial: Number of initial points
            x0: Optional initial point to include
            
        Returns:
            Initial points tensor
        """
        # Generate Sobol points
        sobol = SobolEngine(dimension=self.dim, scramble=True)
        X_sobol = sobol.draw(n_initial).to(dtype=torch.float64)
        
        # Rescale to bounds
        X_init = unnormalize(X_sobol, self.bounds)
        
        # Replace first point with x0 if provided
        if x0 is not None and n_initial > 0:
            X_init[0] = x0
            
        return X_init

    def _evaluate_objective(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate objective function and update counters and history.
        
        Args:
            x: Input tensor (batch functionality supported)
            
        Returns:
            Function values
        """
        y = self.objective_function(x)
        self.n_evals += x.shape[0]
        
        # Convert to tensor if needed
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float64)
        
        # Reshape to ensure correct dimensions
        if y.ndim == 0:
            y = y.view(1, 1)
        elif y.ndim == 1:
            y = y.view(-1, 1)
            
        # Update best value and history
        for i, yi in enumerate(y):
            yi_value = yi.item()
            if yi_value < self.best_value:
                self.best_value = yi_value
                self.best_x = x[i].clone()
                
            self.history_x.append(x[i].clone())
            self.history_y.append(yi_value)
        
        return y

    def _fit_model(self):
        """
        Fit the GP model to the current data.
        
        Returns:
            Fitted GP model, gp_fitting_time, noise_optimization_time, final_noise_vector, best_mll, hyperparameters
        """
        # Start timing for overall GP fitting (including model initialization)
        gp_fitting_start = time.time()

        # Apply preprocessing transformation if provided
        if self.preprocessing_transform is not None:
            self.train_Y_transformed = self.preprocessing_transform(self.train_Y)
            self.best_value_transformed = self.train_Y_transformed.min()
        else:
            self.train_Y_transformed = self.train_Y
            self.best_value_transformed = self.best_value if hasattr(self, 'best_value') else self.train_Y.min()
        
        # Handle trust region models
        if self._is_trust_region_model:
            # Create fresh model each time with trust region bounds
            model_kwargs = dict(self.model_kwargs)
            
            # Calculate trust region bounds if we have trust region state
            if hasattr(self, 'trust_region_center') and hasattr(self, 'trust_region_length') and self.trust_region_center is not None:
                # Calculate trust region bounds
                tr_bounds = self._calculate_trust_region_bounds()
                model_kwargs['trustregion_bounds'] = tr_bounds
                model_kwargs['trustregion_center'] = self.trust_region_center
            
            # Handle beam search state if this is a beam search model
            if self._is_beam_search_model and hasattr(self, 'beam_search_state'):
                model_kwargs['existing_beams'] = self.beam_search_state
            
            # Create new model instance
            model = self.model_class(
                train_X=self.train_X,
                train_Y=self.train_Y_transformed,
                **model_kwargs
            )
            
            # Extract beam state for next iteration if this is a beam search model
            if self._is_beam_search_model and hasattr(model, 'get_updated_beams'):
                self.beam_search_state = model.get_updated_beams()
        else:
            # Create new model instance for non-trust region models
            model = self.model_class(
                train_X=self.train_X,
                train_Y=self.train_Y_transformed,
                **self.model_kwargs
            )

        # Extract noise optimization time and noise vector if this is a NoiseInjectionGP
        noise_optimization_time = 0.0
        final_noise_vector = None
        best_mll = None
        
        if hasattr(model, 'noise_optimization_time'):
            noise_optimization_time = model.noise_optimization_time
            final_noise_vector = model.likelihood.noise_covar.factors.detach().clone()

        # Set initial lengthscale to sqrt(d) (https://arxiv.org/pdf/2402.02746)
        with torch.no_grad():
            if hasattr(model.covar_module, 'base_kernel') and hasattr(model.covar_module.base_kernel, 'lengthscale'):
                init_lengthscale = torch.sqrt(torch.tensor(self.dim, dtype=torch.float64))
                model.covar_module.base_kernel.lengthscale.copy_(init_lengthscale)
        
        # Fit model hyperparameters
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        
        # Make fitting more robust
        try:
            fit_gpytorch_mll(mll, max_attempts=10)
        except ModelFittingError as e:
            log(self.verbose, f"WARNING: Model fitting failed after 10 attempts: {e}. Proceeding with prior hyperparameters.")

        model.train()
        # Calculate final MLL with fitted hyperparameters
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            output = model(*model.train_inputs)
            best_mll = mll(output, model.train_targets).item()
        model.eval()

        # Calculate total GP fitting time (includes model initialization and hyperparameter optimization)
        total_fitting_time = time.time() - gp_fitting_start
        
        # For NoiseInjectionGP, separate pure hyperparameter optimization time from noise optimization time
        if noise_optimization_time > 0:
            gp_fitting_time = total_fitting_time - noise_optimization_time
        else:
            gp_fitting_time = total_fitting_time
        
        # Extract GP hyperparameters after fitting
        hyperparameters = self._extract_gp_hyperparameters(model)
        
        # Extract and store lengthscales for trust region models
        if (hasattr(model, 'covar_module') and 
            hasattr(model.covar_module, 'base_kernel') and 
            hasattr(model.covar_module.base_kernel, 'lengthscale')):
            self.current_lengthscales = model.covar_module.base_kernel.lengthscale.detach().clone().cpu().squeeze()
        
        return model, gp_fitting_time, noise_optimization_time, final_noise_vector, best_mll, hyperparameters

    def _extract_gp_hyperparameters(self, model):
        """
        Extract GP hyperparameters from the fitted model.
        Expected model structure: ScaleKernel(RBFKernel) with standard Gaussian likelihood.
        
        Args:
            model: Fitted GP model
            
        Returns:
            Dict containing lengthscale, signal_variance, and noise_variance
        """
        # Extract lengthscale from ScaleKernel(RBFKernel) structure
        lengthscale = model.covar_module.base_kernel.lengthscale.detach().clone().cpu()
        
        # Extract signal variance (outputscale) from ScaleKernel
        signal_variance = model.covar_module.outputscale.detach().clone().cpu()
        
        # Extract noise variance from Gaussian likelihood
        noise_variance = model.likelihood.noise.detach().clone().cpu()
        
        return {
            'lengthscale': lengthscale,
            'signal_variance': signal_variance,
            'noise_variance': noise_variance
        }

    def _calculate_exploration_metric(self, X_train, X_next):
        """
        Calculate the exploration metric as the minimum distance between new 
        candidate points and existing training points.
        
        Args:
            X_train: Existing training points (normalized)
            X_next: New candidate points (normalized)
            
        Returns:
            Average minimum distance
        """
        # Reshape X_next to ensure it's treated as a batch
        if X_next.ndim == 1:
            X_next = X_next.unsqueeze(0)
            
        # Calculate minimum distance from each new point to any training point
        min_distances = []
        for x_next_sample in X_next:
            # Calculate Euclidean distances from this point to all training points
            distances = torch.sqrt(torch.sum((X_train - x_next_sample)**2, dim=1))
            # Find the minimum distance
            min_dist = torch.min(distances).item()
            min_distances.append(min_dist)
        
        # Return the average minimum distance
        return sum(min_distances) / len(min_distances)

    def _prepare_common_results(self, total_runtime: float, additional_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Prepare common results dictionary that all BO variants share.
        
        Args:
            total_runtime: Total optimization runtime
            additional_results: Additional results specific to the BO variant
            
        Returns:
            Dictionary with optimization results
        """
        results = {
            "x": self.best_x.detach().cpu(),
            "fun": self.best_value,
            "nfev": self.n_evals,
            "history_x": torch.stack(self.history_x) if self.history_x else torch.tensor([]),
            "history_y": torch.tensor(self.history_y) if self.history_y else torch.tensor([]),
            "exploration_metrics": torch.tensor(self.exploration_metrics) if self.exploration_metrics else torch.tensor([]),
            
            # Runtime metrics
            "total_runtime": total_runtime,
            "iteration_times": torch.tensor(self.iteration_times) if self.iteration_times else torch.tensor([]),
            "cumulative_times": torch.tensor(self.cumulative_times) if self.cumulative_times else torch.tensor([]),
            
            # Detailed timing metrics for each BO step
            "gp_fitting_times": torch.tensor(self.gp_fitting_times) if self.gp_fitting_times else torch.tensor([]),
            "noise_optimization_times": torch.tensor(self.noise_optimization_times) if self.noise_optimization_times else torch.tensor([]),
            "acqf_optimization_times": torch.tensor(self.acqf_optimization_times) if self.acqf_optimization_times else torch.tensor([]),
            
            # GP hyperparameters for each iteration
            "lengthscales": self.lengthscales if self.lengthscales else [],
            "signal_variances": self.signal_variances if self.signal_variances else [],
            "noise_variances": self.noise_variances if self.noise_variances else [],
            "noise_vectors": self.noise_vectors if self.noise_vectors else [],
            "best_mlls": self.best_mlls if self.best_mlls else [],
        }
        
        # Add additional results if provided
        if additional_results:
            results.update(additional_results)
            
        return results

    @abstractmethod
    def _optimize_acquisition(self, model, bounds: torch.Tensor):
        """
        Optimize the acquisition function.
        
        Args:
            model: GP model
            bounds: Bounds for optimization
            
        Returns:
            Candidate points, acquisition_optimization_time
        """
        pass

    @abstractmethod
    def minimize(self, x0: Optional[torch.Tensor] = None, n_initial: int = 5, max_evals: int = 30) -> Dict[str, Any]:
        """
        Run the Bayesian optimization procedure.
        
        Args:
            x0: Initial point (optional)
            n_initial: Number of initial points
            max_evals: Maximum number of function evaluations
            
        Returns:
            Dictionary with optimization results
        """
        pass

    def _calculate_trust_region_bounds(self):
        """
        Calculate trust region bounds for the new trust region noise injection GP.
        This method should be overridden by TrustRegionBO to provide the actual bounds calculation.
        
        Returns:
            tuple: (lower_bounds, upper_bounds) for trust region
        """
        raise NotImplementedError("Must implement _calculate_trust_region_bounds to provide actual bounds calculation.")