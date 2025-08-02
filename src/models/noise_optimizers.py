import torch

from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.exceptions.errors import ModelFittingError

from src.utils.utils import mll_for_factors, mll_with_refit
from src.utils.constants import NOISE_JITTER
from src.beam_search import BeamSearchEngine, HeuristicBeamSearchStrategy, BeamNode


class BaseNoiseOptimizer:
    """Base class for noise optimization strategies"""

    def __init__(self, train_X, train_Y, covar_module, mean_module, initial_injection, refit_gp=False):
        self.train_X = train_X
        self.train_Y = train_Y
        self.covar_module = covar_module
        self.mean_module = mean_module
        self.initial_injection = initial_injection
        self.refit_gp = refit_gp

        self.num_train = train_X.shape[0]
        self.dim = train_X.shape[-1]

        # Choose MLL evaluation function
        self.mll_eval_func = mll_with_refit if self.refit_gp else mll_for_factors
        if self.refit_gp:
            print("Using MLL evaluation with per-iteration GP hyperparameter refitting.")

    def optimize(self):
        """Override in subclasses to implement specific optimization logic."""
        raise NotImplementedError

    def _prefit_hyperparameters(self):
        """Pre-fit GP hyperparameters for heuristic methods."""
        try:
            print("Pre-fitting GP hyperparameters for heuristic noise optimization.")
            temp_likelihood = GaussianLikelihood()
            temp_model = SingleTaskGP(
                self.train_X,
                self.train_Y,
                likelihood=temp_likelihood,
                covar_module=self.covar_module,
                mean_module=self.mean_module,
            )
            mll = ExactMarginalLogLikelihood(temp_model.likelihood, temp_model)
            fit_gpytorch_mll(mll, max_attempts=10)
        except ModelFittingError:
            print("Warning: Model fitting failed during pre-fit for heuristic noise optimization. Noisy factors may not be optimal and optimized on prior covariance module.")


class NaiveOptimizer(BaseNoiseOptimizer):
    """Naive MLL optimization strategy."""

    def optimize(self):
        factor_no_noise = NOISE_JITTER
        factor_with_noise = 1.0

        # Pre-fit GP hyperparameters
        self._prefit_hyperparameters()

        # Set initial factors based on initial_injection
        if self.initial_injection == "no_noise":
            factors = torch.full((self.num_train,), factor_no_noise)
        elif self.initial_injection == "noise":
            factors = torch.full((self.num_train,), factor_with_noise)
        else:
            raise ValueError(f"Unknown initial injection type: {self.initial_injection}")

        # Greedily optimize factors point-by-point
        for i in range(self.num_train):
            # Try with noise
            factors[i] = factor_with_noise
            mll_with_noise = self.mll_eval_func(
                self.train_X, self.train_Y, factors, self.covar_module, self.mean_module
            )

            # Try without noise
            factors[i] = factor_no_noise
            mll_without_noise = self.mll_eval_func(
                self.train_X, self.train_Y, factors, self.covar_module, self.mean_module
            )

            # Keep the change if it improves MLL
            if mll_with_noise > mll_without_noise:
                factors[i] = factor_with_noise
            else:
                factors[i] = factor_no_noise

        print(
            f"Optimized factors via brute-force MLL (start: {self.initial_injection}): {[round(float(f), 1) for f in factors.tolist()]}"
        )
        return factors


class IteratedLocalSearchOptimizer(BaseNoiseOptimizer):
    """Iterated Local Search optimization strategy."""

    def __init__(
        self,
        train_X,
        train_Y,
        covar_module,
        mean_module,
        initial_injection,
        refit_gp=False,
        perturbation_size=0.20,
    ):
        super().__init__(train_X, train_Y, covar_module, mean_module, initial_injection, refit_gp)
        self.perturbation_size = perturbation_size


    def optimize(self):
        factor_no_noise = NOISE_JITTER
        factor_with_noise = 1.0
        
        # Pre-fit GP hyperparameters
        self._prefit_hyperparameters()

        # Set initial factors based on initial_injection
        if self.initial_injection == "no_noise":
            factors = torch.full((self.num_train,), factor_no_noise)
        elif self.initial_injection == "noise":
            factors = torch.full((self.num_train,), factor_with_noise)
        else:
            raise ValueError(f"Unknown initial injection type: {self.initial_injection}")

        best_factors = factors.clone()
        best_mll = self.mll_eval_func(
            self.train_X, self.train_Y, best_factors, self.covar_module, self.mean_module
        )
        n_trials = 80
        flip_frac = self.perturbation_size

        # Early stopping parameters
        stagnation_limit = 10
        plateau_count = 0

        for i in range(n_trials):
            # Create a trial vector by perturbing the current best solution
            trial_factors = best_factors.clone()
            n_to_flip = max(1, int(self.num_train * flip_frac))
            perturbation_indices = torch.randperm(self.num_train)[:n_to_flip].tolist()

            # Local search improvement
            for idx in perturbation_indices:
                # Try with noise
                trial_factors[idx] = factor_with_noise
                mll_with_noise = self.mll_eval_func(
                    self.train_X, self.train_Y, trial_factors, self.covar_module, self.mean_module
                )

                # Try without noise
                trial_factors[idx] = factor_no_noise
                mll_without_noise = self.mll_eval_func(
                    self.train_X, self.train_Y, trial_factors, self.covar_module, self.mean_module
                )

                # Keep the change if it improves MLL for the trial vector
                if mll_with_noise > mll_without_noise:
                    trial_factors[idx] = factor_with_noise
                else:
                    trial_factors[idx] = factor_no_noise

            # Check if trial is better than global best
            trial_mll = self.mll_eval_func(
                self.train_X, self.train_Y, trial_factors, self.covar_module, self.mean_module
            )

            if trial_mll > best_mll + 1e-6:  # Use a small tolerance for improvement
                best_factors = trial_factors.clone()
                best_mll = trial_mll
                plateau_count = 0  # Reset counter on improvement
            else:
                plateau_count += 1  # Increment counter if no improvement

            print(f"Iteration: {i + 1}, ILS iteration: best MLL = {best_mll:.4f}")

            # Check for early stopping
            if plateau_count >= stagnation_limit:
                print(f"Early stopping at iteration {i + 1} (plateau detected)")
                break

        print(
            f"Optimized factors via Iterated Local Search (start: {self.initial_injection}): {best_factors.tolist()}"
        )
        return best_factors


class BeamSearchOptimizer(BaseNoiseOptimizer):
    """Beam Search optimization strategy using centralized beam search engine."""

    def __init__(
        self,
        train_X,
        train_Y,
        covar_module,
        mean_module,
        initial_injection,
        refit_gp=False,
        beam_width=5,
    ):
        super().__init__(train_X, train_Y, covar_module, mean_module, initial_injection, refit_gp)
        self.beam_width = beam_width

    def optimize(self):
        """
        Optimize noise factors using centralized beam search engine.
        """
        # Pre-fit GP hyperparameters
        self._prefit_hyperparameters()
        
        # Determine search direction based on initial_injection
        if self.initial_injection == "noise":
            search_mode = "remove_noise"
            initial_noise_vector = torch.full((self.num_train,), 1.0)
        else:  # initial_injection == "no_noise"
            search_mode = "add_noise"
            initial_noise_vector = torch.full((self.num_train,), NOISE_JITTER)
        
        # Create initial beam node
        initial_beam = BeamNode(
            noise_vector=initial_noise_vector,
            decision_history=[]
        )
        
        # Create beam search strategy
        strategy = HeuristicBeamSearchStrategy(
            mll_eval_func=self.mll_eval_func,
            factor_no_noise=NOISE_JITTER,
            factor_with_noise=1.0
        )
        
        # Create beam search engine
        engine = BeamSearchEngine(
            strategy=strategy,
            beam_width=self.beam_width,
            max_iterations=self.num_train,
            convergence_threshold=1e-6,
            stagnation_limit=5,
            verbose=True
        )
        
        # Initialize search
        engine.initialize_search([initial_beam])
        
        # Run beam search
        best_beam = engine.search(
            search_mode=search_mode,
            num_train=self.num_train,
            train_X=self.train_X,
            train_Y=self.train_Y,
            covar_module=self.covar_module,
            mean_module=self.mean_module
        )
        
        # Extract results
        if best_beam is not None:
            factors = best_beam.noise_vector
            print(f"Beam Search completed ({search_mode}): Best MLL = {best_beam.mll_score:.4f}")
            
            # Log final configuration
            no_noise_count = (factors <= NOISE_JITTER + 1e-6).sum().item()
            noise_count = self.num_train - no_noise_count
            print(f"Final configuration: {noise_count} noise, {no_noise_count} no-noise points")
            
            # Get search statistics
            stats = engine.get_search_stats()
            print(f"Search completed in {stats['iterations']} iterations")
        else:
            # Fallback to simple configuration if beam search fails
            print("Beam search failed, using fallback configuration")
            factors = torch.full((self.num_train,), NOISE_JITTER)
        
        return factors


class GradientOptimizer(BaseNoiseOptimizer):
    """Gradient-based optimization for continuous noise factors."""
    # TODO: Check simplify gradient implementation

    def __init__(
            self,
            train_X,
            train_Y,
            covar_module,
            mean_module,
            initial_injection,
            refit_gp=False,
            learning_rate=0.01,
            n_iterations=300,
            scheduler_factor=0.8,
            scheduler_patience=10,
            early_stop_patience=20,
        ):
            super().__init__(train_X, train_Y, covar_module, mean_module, initial_injection, refit_gp)
            self.learning_rate = learning_rate
            self.n_iterations = n_iterations
            self.scheduler_factor = scheduler_factor
            self.scheduler_patience = scheduler_patience
            self.early_stop_patience = early_stop_patience

    def optimize(self):
        jitter_value = NOISE_JITTER
        max_noise_value = 1.0
        
        # Dimension-adaptive hyperparameters
        dim = self.train_X.shape[-1]
        learning_rate = self.learning_rate / max(1.0, dim / 5.0)

        n_iterations = self.n_iterations
        
        # Initialize noise factors around midpoint to avoid bias
        # Assumptions: (1) midpoint 0.5 between jitter and full noise is unbiased
        #              (2) small random perturbations break symmetry without bias
        initial_mean = 0.5
        initial_std = 0.05
        factors = torch.normal(initial_mean, initial_std, (self.num_train,))
        factors = torch.clamp(factors, jitter_value, max_noise_value)
        factors.requires_grad_(True)
        
        # Combine parameters for joint optimization
        joint_params = [{'params': factors}]
        joint_params.append({'params': self.covar_module.parameters()})
        joint_params.append({'params': self.mean_module.parameters()})

        # Use Adam without weight decay (no regularization needed)
        optimizer = torch.optim.Adam(joint_params, lr=learning_rate)
        
        # Learning rate scheduling
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            verbose=False,
            min_lr=1e-6,
        )
        
        print(f"Starting gradient-based optimization: {n_iterations} iterations, lr={learning_rate:.6f}, dim={dim}")
        
        best_mll = -float('inf')
        best_factors = None
        plateau_count = 0
        
        for iteration in range(n_iterations):
            optimizer.zero_grad()
            
            # Use gradient-friendly MLL function
            current_mll = mll_for_factors(self.train_X, self.train_Y, factors, self.covar_module, self.mean_module, preserve_grad=True)
            
            loss = -current_mll
            
            loss.backward()

            # More aggressive gradient clipping for high-dim
            max_grad_norm = 1.0 / max(1.0, dim / 5.0)
            torch.nn.utils.clip_grad_norm_([factors], max_norm=max_grad_norm)
            
            optimizer.step()
            
            # Clamp factors
            with torch.no_grad():
                factors.clamp_(jitter_value, max_noise_value)
            
            # Track best solution
            current_mll_val = current_mll.item()
            
            # Update scheduler
            scheduler.step(current_mll_val)
            
            # Early stopping with plateau detection
            if current_mll_val > best_mll + 1e-6:
                best_mll = current_mll_val
                best_factors = factors.detach().clone()
                plateau_count = 0
            else:
                plateau_count += 1
            
            # Stop if stuck on plateau
            if plateau_count >= self.early_stop_patience:
                print(f"Early stopping at iteration {iteration + 1} (plateau detected)")
                break
            
            # Progress logging (less frequent for high-dim)
            log_freq = max(10, n_iterations // 10)
            if iteration % log_freq == 0:
                avg_factor = factors.mean().item()
                min_factor = factors.min().item()
                max_factor = factors.max().item()
                grad_norm = factors.grad.norm().item() if factors.grad is not None else 0.0
                current_lr = optimizer.param_groups[0]['lr']
                
                # Extract GP hyperparameters for logging
                outputscale = self.covar_module.outputscale.item()
                lengthscale_mean = self.covar_module.base_kernel.lengthscale.mean().item()

                print(f"Iteration {iteration + 1}: MLL = {current_mll_val:.4f}, "
                        f"Factors: avg={avg_factor:.4f}, min={min_factor:.4f}, max={max_factor:.4f}, "
                        f"Grad norm = {grad_norm:.2f}, LR = {current_lr:.6f}, "
                        f"Signal: {outputscale:.4f}, LS (avg): {lengthscale_mean:.4f}")
        
        # Use best factors
        factors = best_factors if best_factors is not None else factors
        print(f"Gradient optimization completed: Best MLL = {best_mll:.4f}")
        print(f"Final factors summary: mean={factors.mean().item():.4f}, "
                f"std={factors.std().item():.4f}, "
                f"min={factors.min().item():.4f}, "
                f"max={factors.max().item():.4f}")
        
        return factors


class BinaryGradientOptimizer(BaseNoiseOptimizer):
    """Gradient-based optimization for binary noise decisions using sigmoid."""

    def __init__(
            self,
            train_X,
            train_Y,
            covar_module,
            mean_module,
            initial_injection,
            refit_gp=False,
            learning_rate=0.01,
            n_iterations=300,
            scheduler_factor=0.8,
            scheduler_patience=10,
            early_stop_patience=20,
        ):
            super().__init__(train_X, train_Y, covar_module, mean_module, initial_injection, refit_gp)
            self.learning_rate = learning_rate
            self.n_iterations = n_iterations
            self.scheduler_factor = scheduler_factor
            self.scheduler_patience = scheduler_patience
            self.early_stop_patience = early_stop_patience

    def optimize(self):
        jitter_value = NOISE_JITTER
        noise_value = 1.0
        
        # Dimension-adaptive hyperparameters
        dim = self.train_X.shape[-1]
        learning_rate = self.learning_rate / max(1.0, dim / 5.0)

        n_iterations = self.n_iterations
        
        # Initialize logits with random values to break symmetry
        # Small random values around 0 ensure unbiased start while breaking symmetry
        initial_std = 0.1
        logits = torch.normal(0.0, initial_std, (self.num_train,), requires_grad=True)
        
        # Use Adam optimizer (no regularization needed)
        optimizer = torch.optim.Adam([logits], lr=learning_rate)
        
        # Learning rate scheduling
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            verbose=False,
            min_lr=1e-6,
        )
        
        print(f"Starting gradient-based binary optimization: {n_iterations} iterations, lr={learning_rate:.6f}, dim={dim}")
        
        best_mll = -float('inf')
        best_logits = None
        plateau_count = 0
        
        for iteration in range(n_iterations):
            optimizer.zero_grad()
            
            # Convert logits to probabilities using sigmoid
            probs = torch.sigmoid(logits)
            
            # Create continuous factors for gradient computation
            factors = probs * noise_value + (1 - probs) * jitter_value
            
            # Compute MLL
            current_mll = mll_for_factors(self.train_X, self.train_Y, factors, self.covar_module, self.mean_module, preserve_grad=True)
            
            loss = -current_mll
            
            loss.backward()
            
            # Gradient clipping
            max_grad_norm = 1.0 / max(1.0, dim / 5.0)
            torch.nn.utils.clip_grad_norm_([logits], max_norm=max_grad_norm)
            
            optimizer.step()
            
            # Track best solution
            current_mll_val = current_mll.item()
            
            # Update scheduler
            scheduler.step(current_mll_val)
            
            # Early stopping with plateau detection
            if current_mll_val > best_mll + 1e-6:
                best_mll = current_mll_val
                best_logits = logits.detach().clone()
                plateau_count = 0
            else:
                plateau_count += 1
            
            # Stop if stuck on plateau
            if plateau_count >= self.early_stop_patience:
                print(f"Early stopping at iteration {iteration + 1} (plateau detected)")
                break
            
            # Progress logging
            log_freq = max(15, n_iterations // 15)
            if iteration % log_freq == 0:
                with torch.no_grad():
                    current_probs = torch.sigmoid(logits)
                    noise_decisions = (current_probs > 0.5).sum().item()
                    jitter_decisions = self.num_train - noise_decisions
                    avg_noise_prob = current_probs.mean().item()
                    grad_norm = logits.grad.norm().item() if logits.grad is not None else 0.0
                    current_lr = optimizer.param_groups[0]['lr']
                
                print(f"Iteration {iteration + 1}: MLL = {current_mll_val:.4f}, "
                      f"Noise/Jitter: {noise_decisions}/{jitter_decisions}, "
                      f"Avg noise prob = {avg_noise_prob:.3f}, "
                      f"Grad norm = {grad_norm:.2f}, LR = {current_lr:.6f}")
        
        # Make final binary decisions
        final_logits = best_logits if best_logits is not None else logits
        
        with torch.no_grad():
            final_probs = torch.sigmoid(final_logits)
            hard_decisions = (final_probs > 0.5).float()  # 1 for noise, 0 for jitter
            factors = hard_decisions * noise_value + (1 - hard_decisions) * jitter_value
        
        noise_count = (factors == noise_value).sum().item()
        jitter_count = self.num_train - noise_count
        
        print(f"Gradient binary optimization completed: Best MLL = {best_mll:.4f}")
        print(f"Final binary decisions: {noise_count} noise points, {jitter_count} jitter points")
        print(f"Final factors: {factors.tolist()}")
        
        return factors