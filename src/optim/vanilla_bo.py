import time
import torch
from typing import Dict, Any, Optional

from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize

from src.utils.utils import log
from .base_bo import BaseBO
    
class VanillaBO(BaseBO):
    """
    Vanilla Bayesian Optimization implementation with support for various GP models, acquisition functions and objective functions.
    
    Inherits common functionality from BaseBO and implements standard BO without trust regions.
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
        Initialize the Vanilla Bayesian optimizer.
        
        Args:
            objective_function: The objective function to optimize (must have dim and bounds attributes)
            model_class: The GP model class to use
            model_kwargs: Keyword arguments for model initialization
            acqf_kwargs: Keyword arguments for acquisition function
            preprocessing_transform: Optional preprocessing transform to apply to the data (w/o untransform)
            verbose: Whether to print progress information
        """
        super().__init__(
            objective_function=objective_function,
            model_class=model_class,
            acqf_class=acqf_class,
            model_kwargs=model_kwargs,
            acqf_kwargs=acqf_kwargs,
            preprocessing_transform=preprocessing_transform,
            verbose=verbose,
        )
    
    def _optimize_acquisition(self, model, bounds: torch.Tensor):
        """
        Optimize the acquisition function.
        
        Args:
            model: GP model
            bounds: Bounds for optimization
            
        Returns:
            Candidate points, acquisition_optimization_time
        """
        acqf_optimization_start = time.time()
        
        # Assemble acquisition function kwargs
        # For acquisition functions that require best_f (like EI, LogEI), we dynamically
        # set it to the current best observed value. The YAML value is just a placeholder.
        # UCB and similar functions don't need best_f, so they can omit it entirely.
        acq_kwargs = dict(self.acqf_kwargs)
        if "best_f" in acq_kwargs:
            acq_kwargs["best_f"] = self.train_Y_transformed.min().item()

        candidates, _ = optimize_acqf(
            acq_function=self.acqf_class(model=model, **acq_kwargs),
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
            )
        
        acqf_optimization_time = time.time() - acqf_optimization_start
        
        return candidates, acqf_optimization_time

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
        # Start timing
        self.start_time = time.time()
        
        # Generate and evaluate initial points
        init_start_time = time.time()
        X_init_orig = self._generate_initial_points(n_initial, x0)
        Y_init = self._evaluate_objective(X_init_orig)
        init_time = time.time() - init_start_time
        
        # Record initialization time
        self.iteration_times.append(init_time)
        self.cumulative_times.append(init_time)
        
        # Initialize detailed timing lists with placeholder for initialization
        # (no GP fitting, noise optimization, or acquisition optimization in init phase)
        self.gp_fitting_times.append(0.0)
        self.noise_optimization_times.append(0.0) 
        self.acqf_optimization_times.append(0.0)
        self.noise_vectors.append(None)
        self.best_mlls.append(None)
        
        # Initialize hyperparameter tracking with placeholders for initialization
        self.lengthscales.append(None)
        self.signal_variances.append(None)
        self.noise_variances.append(None)
        
        # Initialize training data
        self.train_X = normalize(X_init_orig, self.bounds)
        self.train_Y = Y_init
        
        # Main optimization loop
        total_iterations = max_evals - n_initial
        for i in range(total_iterations):
            if self.n_evals >= max_evals:
                break
            
            # Start timing this iteration
            iteration_start_time = time.time()
            
            log(self.verbose, "=" * 60)
            log(self.verbose, f"Iteration {i+1}/{total_iterations}, Best value: {self.best_value:.6f}, Best value transformed: {self.best_value_transformed:.6f}")
            
            # Fit model and track detailed timings
            model, gp_fitting_time, noise_optimization_time, final_noise_vector, best_mll, hyperparameters = self._fit_model()
            
            # Store timing and noise vector information
            self.gp_fitting_times.append(gp_fitting_time)
            self.noise_optimization_times.append(noise_optimization_time)
            
            if final_noise_vector is not None:
                self.noise_vectors.append(final_noise_vector.cpu())
            else:
                self.noise_vectors.append(None)
                
            # Store best MLL information
            self.best_mlls.append(best_mll)
            
            # Store GP hyperparameters
            self.lengthscales.append(hyperparameters['lengthscale'])
            self.signal_variances.append(hyperparameters['signal_variance'])
            self.noise_variances.append(hyperparameters['noise_variance'])
            
            # Get next candidate point and track acquisition optimization time
            bounds_normalized = torch.stack([torch.zeros(self.dim), torch.ones(self.dim)])
            
            x_new_norm, acqf_optimization_time = self._optimize_acquisition(model, bounds_normalized)
            
            # Store acquisition function optimization time
            self.acqf_optimization_times.append(acqf_optimization_time)
            
            # Calculate exploration metric before updating training data
            exploration_value = self._calculate_exploration_metric(self.train_X, x_new_norm)
            self.exploration_metrics.append(exploration_value)   
            
            # Convert from [0, 1] space back to original space
            x_new_orig = unnormalize(x_new_norm, self.bounds)
            
            # Evaluate new point
            y_new = self._evaluate_objective(x_new_orig)
            
            # Update training data (x stays in normalized space)
            self.train_X = torch.cat([self.train_X, x_new_norm])
            self.train_Y = torch.cat([self.train_Y, y_new])
            
            # Record iteration time
            iteration_time = time.time() - iteration_start_time
            self.iteration_times.append(iteration_time)
            
            # Update cumulative time
            current_cumulative_time = time.time() - self.start_time
            self.cumulative_times.append(current_cumulative_time)
            
            log(self.verbose, f"Exploration metric (avg min distance): {exploration_value:.6f}")
            log(self.verbose, f"Iteration time: {iteration_time:.3f}s, Cumulative time: {current_cumulative_time:.3f}s")
            log(self.verbose, f"  GP fitting time: {gp_fitting_time:.3f}s")
            log(self.verbose, f"  Noise optimization time: {noise_optimization_time:.3f}s")
            log(self.verbose, f"  Acquisition optimization time: {acqf_optimization_time:.3f}s")
            lengthscales_str = [f"{ls:.6f}" for ls in hyperparameters['lengthscale'].flatten().tolist()]
            log(self.verbose, f"GP Lengthscales: [{', '.join(lengthscales_str)}]")
            log(self.verbose, f"GP Signal Variance: {hyperparameters['signal_variance'].item():.10f}")
            log(self.verbose, f"GP Noise Variance: {hyperparameters['noise_variance'].item():.6f}")
            log(self.verbose, f"Best MLL: {best_mll:.6f}")
            
        # Prepare results
        total_runtime = time.time() - self.start_time
        
        # VanillaBO specific results
        vanilla_results = {
            "batch_size": 1,  # Vanilla BO always uses batch size 1
        }
        
        return self._prepare_common_results(total_runtime, vanilla_results)
