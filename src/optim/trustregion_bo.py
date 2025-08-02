import time
import torch
import math
from typing import Dict, Any, Optional

from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize

from src.utils.utils import log
from .base_bo import BaseBO
    
class TrustRegionBO(BaseBO):
    """
    Trust Region Bayesian Optimization implementation following the TURBO algorithm.
    This implementation uses lengthscale-weighted trust regions with adaptive expansion and shrinkage
    based on success/failure counters like the original TURBO implementation.
    Compatible with various GP models and acquisition functions through Hydra configuration.
    
    Inherits common functionality from BaseBO and adds trust region specific behavior.
    """

    def __init__(
        self,
        objective_function,
        model_class,
        acqf_class,
        model_kwargs: Dict[str, Any] = None,
        acqf_kwargs: Dict[str, Any] = None,
        preprocessing_transform=None,
        # Trust region specific parameters
        trust_region_length: float = 0.8,
        min_trust_region_length: float = 0.5**7,  # = 0.0078, following TURBO
        max_trust_region_length: float = 1.6,     # Following TURBO
        trust_region_success_tolerance: int = 3,
        trust_region_failure_tolerance: Optional[int] = None,  # Auto-computed if None
        batch_size: int = 1,  # Number of points to evaluate per iteration
        verbose: bool = True,
    ):
        """
        Initialize the Trust Region Bayesian optimizer.
        
        Args:
            objective_function: The objective function to optimize (must have dim and bounds attributes)
            model_class: The GP model class to use
            acqf_class: The acquisition function class to use
            model_kwargs: Keyword arguments for model initialization
            acqf_kwargs: Keyword arguments for acquisition function
            preprocessing_transform: Optional preprocessing transform to apply to the data
            trust_region_length: Initial trust region length (normalized coordinates)
            min_trust_region_length: Minimum trust region length
            max_trust_region_length: Maximum trust region length
            trust_region_success_tolerance: Shrink trust region after this many successes
            trust_region_failure_tolerance: Expand trust region after this many failures (auto-computed if None)
            batch_size: Number of points to evaluate per iteration
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
        
        # Trust region parameters (following TURBO paper)
        self.trust_region_length = trust_region_length
        self.min_trust_region_length = min_trust_region_length
        self.max_trust_region_length = max_trust_region_length
        self.trust_region_success_tolerance = trust_region_success_tolerance
        
        # Compute failure tolerance like in TURBO (based on batch size and dimension)
        if trust_region_failure_tolerance is None:
            self.trust_region_failure_tolerance = math.ceil(
                max([4.0 / batch_size, float(self.dim) / batch_size])
            )
        else:
            self.trust_region_failure_tolerance = trust_region_failure_tolerance
        
        # Classical TURBO options
        self.batch_size = batch_size

        # Trust region state
        self.trust_region_center = None
        self.success_counter = 0
        self.failure_counter = 0
        self.last_best_y = float('inf')
        
        # Trust region specific tracking
        self.trust_region_history = []  # Track trust region evolution
        
        log(self.verbose, f"TrustRegionBO initialized with dim={self.dim}, TR_length={self.trust_region_length:.4f}")
        log(self.verbose, f"TR tolerances: success={self.trust_region_success_tolerance}, failure={self.trust_region_failure_tolerance}")



    def _get_trust_region_bounds(self, global_bounds: torch.Tensor) -> torch.Tensor:
        """
        Get trust region bounds clipped to global bounds.
        Uses lengthscale-weighted trust region (following original TURBO implementation).
        
        Args:
            global_bounds: Global bounds tensor of shape (2, dim)
            
        Returns:
            Trust region bounds tensor of shape (2, dim)
        """
        if self.trust_region_center is None:
            return global_bounds
        
        # Use stored lengthscales from the fitted model
        if hasattr(self, 'current_lengthscales') and self.current_lengthscales is not None:
            # Extract and process lengthscales as weights (following TURBO implementation)
            weights = self.current_lengthscales
            
            # Normalize weights following original implementation
            weights = weights / weights.mean()  # This will make the next line more stable
            
            # Ensure product of weights is 1 (following TURBO implementation)
            weights = weights / torch.prod(torch.pow(weights, 1.0 / len(weights)))
            
            log(self.verbose, f"Using lengthscale-weighted trust region with weights: {weights.tolist()}")
            
            # Apply weighted trust region bounds
            half_length = self.trust_region_length / 2.0
            tr_lower = self.trust_region_center - weights * half_length
            tr_upper = self.trust_region_center + weights * half_length
        else:
            # Simple uniform trust region if no lengthscales available
            log(self.verbose, "No lengthscales available, using uniform trust region")
            half_length = self.trust_region_length / 2.0
            tr_lower = self.trust_region_center - half_length
            tr_upper = self.trust_region_center + half_length
        
        # Clip to global bounds [0, 1] in normalized space
        tr_lower = torch.clamp(tr_lower, 0.0, 1.0)
        tr_upper = torch.clamp(tr_upper, 0.0, 1.0)
        
        log(self.verbose, f"TR bounds: [{tr_lower.tolist()}, {tr_upper.tolist()}]")
        
        return torch.stack([tr_lower, tr_upper])

    def _calculate_trust_region_bounds(self):
        """
        Calculate trust region bounds for the noise injection GP models.
        Uses stored lengthscales from the fitted model.
        
        Returns:
            tuple: (lower_bounds, upper_bounds) for trust region
        """
        # Return full bounds if no trust region info available
        if self.trust_region_center is None:
            return (torch.zeros(self.dim), torch.ones(self.dim))
        
        # Use stored lengthscales if available
        if hasattr(self, 'current_lengthscales') and self.current_lengthscales is not None:
            # Extract and process lengthscales as weights (following TURBO implementation)
            weights = self.current_lengthscales
            
            # Normalize weights following original implementation
            weights = weights / weights.mean()  # This will make the next line more stable
            
            # Ensure product of weights is 1 (following TURBO implementation)
            weights = weights / torch.prod(torch.pow(weights, 1.0 / len(weights)))
            
            # Apply weighted trust region bounds
            half_length = self.trust_region_length / 2.0
            tr_lower = self.trust_region_center - weights * half_length
            tr_upper = self.trust_region_center + weights * half_length
        else:
            # Simple uniform trust region if no lengthscales available (first iteration)
            half_length = self.trust_region_length / 2.0
            tr_lower = self.trust_region_center - half_length
            tr_upper = self.trust_region_center + half_length
        
        # Clip to global bounds [0, 1] in normalized space
        tr_lower = torch.clamp(tr_lower, 0.0, 1.0)
        tr_upper = torch.clamp(tr_upper, 0.0, 1.0)
        
        return (tr_lower, tr_upper)

    def _optimize_acquisition(self, model, bounds: torch.Tensor):
        """
        Optimize the acquisition function within trust region bounds.
        
        Args:
            model: GP model
            bounds: Bounds for optimization (either global or trust region bounds)
            
        Returns:
            Candidate points, acquisition_optimization_time
        """
        acqf_optimization_start = time.time()
        
        # Traditional acquisition function optimization
        # Assemble acquisition function kwargs
        acq_kwargs = dict(self.acqf_kwargs)
        if "best_f" in acq_kwargs:
            acq_kwargs["best_f"] = self.train_Y_transformed.min().item()

        candidates, _ = optimize_acqf(
            acq_function=self.acqf_class(model=model, **acq_kwargs),
            bounds=bounds,
            q=self.batch_size,
            num_restarts=10,
            raw_samples=512,
        )
        
        acqf_optimization_time = time.time() - acqf_optimization_start
        
        return candidates, acqf_optimization_time

    def _initialize_trust_region(self):
        """
        Initialize trust region center based on current best point.
        """
        if self.train_Y is None or len(self.train_Y) == 0:
            return
            
        # Initialize trust region center as the best observed point (in normalized space)
        best_idx = torch.argmin(self.train_Y)
        self.trust_region_center = self.train_X[best_idx].clone()
        self.last_best_y = self.train_Y[best_idx].item()
        
        log(self.verbose, f"Trust region initialized at best point: {self.trust_region_center.tolist()}")
        log(self.verbose, f"Initial best value: {self.last_best_y:.6f}")

    def _update_trust_region(self, new_x_norm: torch.Tensor, new_y: float):
        """
        Update trust region based on new observation using TURBO logic.
        Following the reference implementation more closely.
        
        Args:
            new_x_norm: New observation point (normalized coordinates)
            new_y: New observation value
        """
        # Store previous state for logging
        prev_length = self.trust_region_length
        prev_success = self.success_counter
        prev_failure = self.failure_counter
        
        # Check if new point improved (following TURBO: improvement threshold)
        improvement_threshold = 1e-3 * abs(self.last_best_y) if self.last_best_y != 0 else 1e-3
        improved = new_y < (self.last_best_y - improvement_threshold)
        
        log(self.verbose, f"TR Update: new_y={new_y:.6f}, threshold={self.last_best_y - improvement_threshold:.6f}, improved={improved}")
        
        if improved:
            # Success: increment success counter and update best value
            self.success_counter += 1
            self.failure_counter = 0
            self.last_best_y = new_y
            
            log(self.verbose, f"SUCCESS! New best: {new_y:.6f}, Success counter: {self.success_counter}")
            
            # Expand trust region after enough successes
            if self.success_counter >= self.trust_region_success_tolerance:
                old_length = self.trust_region_length
                self.trust_region_length = min(
                    2.0 * self.trust_region_length,  # TURBO uses 2x expansion
                    self.max_trust_region_length
                )
                self.success_counter = 0
                
                log(self.verbose, f"EXPANDING trust region: {old_length:.6f} -> {self.trust_region_length:.6f}")
        else:
            # Failure: increment failure counter
            self.failure_counter += 1
            self.success_counter = 0
            
            log(self.verbose, f"FAILURE.")
            
            # Shrink trust region after enough failures
            if self.failure_counter >= self.trust_region_failure_tolerance:
                old_length = self.trust_region_length
                self.trust_region_length = max(
                    self.trust_region_length / 2.0,  # TURBO uses 0.5x shrinkage
                    self.min_trust_region_length
                )
                self.failure_counter = 0
                
                log(self.verbose, f"SHRINKING trust region: {old_length:.6f} -> {self.trust_region_length:.6f}")
        
        # Check for restart conditions
        self._check_restart_conditions()
        
        # Record trust region state
        tr_state = {
            'iteration': len(self.history_y),
            'center': self.trust_region_center.tolist() if self.trust_region_center is not None else None,
            'length': self.trust_region_length,
            'success_counter': self.success_counter,
            'failure_counter': self.failure_counter,
            'improved': improved,
            'best_y': self.last_best_y
        }
        self.trust_region_history.append(tr_state)
        
        log(self.verbose, f"TR State: length={self.trust_region_length:.6f}, S/F={self.success_counter}/{self.failure_counter}")

    def _check_restart_conditions(self):
        """
        Check if trust region should be restarted and restart if necessary.
        """
        # Condition 1: Trust region at absolute minimum size with enough failures
        absolute_minimum_restart = (self.trust_region_length <= self.min_trust_region_length and 
                                  self.failure_counter >= self.trust_region_failure_tolerance // 2)
        
        # Condition 2: Trust region very small and potentially stuck
        stuck_restart = (self.failure_counter >= self.trust_region_failure_tolerance and
                        self.trust_region_length <= 0.2)
        
        restart_triggered = absolute_minimum_restart or stuck_restart
        
        if restart_triggered:
            restart_reason = ("absolute minimum" if absolute_minimum_restart else "stuck trust region")
            log(self.verbose, f"RESTART triggered due to: {restart_reason}")
            self._restart_trust_region()

    def _restart_trust_region(self):
        """
        Restart the trust region when it gets stuck.
        Following TURBO approach: randomly relocate center and reset to initial length.
        """
        # Store old state for logging
        old_center = self.trust_region_center.clone() if self.trust_region_center is not None else None
        old_length = self.trust_region_length
        
        # Find the best point from training data for reference
        if self.train_Y is not None and len(self.train_Y) > 0:
            best_idx = torch.argmin(self.train_Y)
            best_point = self.train_X[best_idx]
        else:
            # Fallback to current center or random point
            best_point = self.trust_region_center if self.trust_region_center is not None else torch.rand(self.dim)
        
        # Randomly perturb the best point to create new center
        # Use moderate random perturbation in normalized space
        perturbation_scale = 0.1  # 10% standard perturbation
        random_perturbation = torch.randn_like(best_point) * perturbation_scale
        new_center = best_point + random_perturbation
        
        # Clip to [0, 1] bounds (normalized space)
        new_center = torch.clamp(new_center, 0.0, 1.0)
        
        self.trust_region_center = new_center
        
        # Reset trust region length to initial value (original TURBO)
        self.trust_region_length = 0.8
        
        # Reset counters
        self.success_counter = 0
        self.failure_counter = 0
        
        log(self.verbose, f"RESTARTED trust region:")
        log(self.verbose, f"  Length: {old_length:.6f} -> {self.trust_region_length:.6f}")
        log(self.verbose, f"  Center: {old_center.tolist() if old_center is not None else None} -> {self.trust_region_center.tolist()}")
        log(self.verbose, f"  Counters reset: Success=0, Failure=0")

    def get_trust_region_info(self):
        """
        Get information about current trust region state.
        
        Returns:
            Dictionary with trust region information
        """
        return {
            'center': self.trust_region_center.tolist() if self.trust_region_center is not None else None,
            'trust_region_length': self.trust_region_length,
            'success_counter': self.success_counter,
            'failure_counter': self.failure_counter,
            'last_best_y': self.last_best_y,
            'success_tolerance': self.trust_region_success_tolerance,
            'failure_tolerance': self.trust_region_failure_tolerance,
            'min_length': self.min_trust_region_length,
            'max_length': self.max_trust_region_length,
            'dim': self.dim,
        }

    def minimize(self, x0: Optional[torch.Tensor] = None, n_initial: int = 5, max_evals: int = 30) -> Dict[str, Any]:
        """
        Run the Trust Region Bayesian optimization procedure.
        
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
        
        # Initialize training data (normalize inputs)
        self.train_X = normalize(X_init_orig, self.bounds)
        self.train_Y = Y_init
        
        # Initialize trust region
        self._initialize_trust_region()
        
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
            
            # Update trust region center to current best point at start of iteration
            best_idx = torch.argmin(self.train_Y)
            self.trust_region_center = self.train_X[best_idx].clone()
            
            # Show trust region info
            tr_info = self.get_trust_region_info()
            log(self.verbose, f"Trust Region: length={tr_info['trust_region_length']:.4f}, S/F={tr_info['success_counter']}/{tr_info['failure_counter']}")
            
            # Get trust region bounds (in normalized space)
            bounds_normalized = torch.stack([torch.zeros(self.dim), torch.ones(self.dim)])
            tr_bounds = self._get_trust_region_bounds(bounds_normalized)
            
            # Get next candidate point within trust region
            x_new_norm, acqf_optimization_time = self._optimize_acquisition(model, tr_bounds)
            
            # Store acquisition function optimization time
            self.acqf_optimization_times.append(acqf_optimization_time)
            
            # Calculate exploration metric before updating training data
            exploration_value = self._calculate_exploration_metric(self.train_X, x_new_norm)
            self.exploration_metrics.append(exploration_value)
            
            # Convert from normalized space back to original space
            x_new_orig = unnormalize(x_new_norm, self.bounds)
            
            # Evaluate new point
            y_new = self._evaluate_objective(x_new_orig)
            
            # Update trust region based on new observation
            self._update_trust_region(x_new_norm, y_new.item())
            
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
        
        # Trust region specific results
        tr_results = {
            "batch_size": self.batch_size,
            "trust_region_history": self.trust_region_history,
            "final_trust_region_info": self.get_trust_region_info(),
        }
        
        return self._prepare_common_results(total_runtime, tr_results)
