import torch

from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean

from .base_noiseinjection_gp import BaseNoiseInjectionGP
from src.utils.utils import mll_for_factors, mll_with_refit
from src.utils.constants import NOISE_JITTER
from src.beam_search import BeamSearchEngine, TrustRegionBeamSearchStrategy, BeamNode


class TrustRegionNoiseInjectionGP(BaseNoiseInjectionGP):
    """
    A trust region noise injection GP model that uses a trust region approach to determine the noise factors.
    """
    def __init__(
        self,
        train_X,
        train_Y,
        outcome_transform=None,
        trustregion_center=None,
        trustregion_bounds=None,
        noise_optional=False # Noise only injected outside the trust region if improving mll
    ):
        """
        Initialize the TrustRegionNoiseInjectionGP model.

        Args:
            train_X: Training input data
            train_Y: Training target data
            outcome_transform: Optional outcome transform
            trustregion_center: Center of trust region (for reference/debugging)
            trustregion_bounds: Tuple of (lower_bounds, upper_bounds) for trust region
            noise_optional: If True, compare MLL with and without noise injection and choose the better option.
                           If False, always inject noise outside trust region.
        """
        # Trust region center is the index of the point with minimum target value by default or can be specified
        if trustregion_center is None:
            self.trustregion_center = torch.argmin(train_Y)
        else:
            self.trustregion_center = trustregion_center

        self.trustregion_bounds = trustregion_bounds
        self.noise_optional = noise_optional

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
        )

    def _get_trust_region_mask(self):
        """
        Get a boolean mask indicating which points are outside the trust region.
        
        Returns:
            torch.Tensor: Boolean mask where True indicates points outside trust region
        """
        # If no trust region bounds are provided, treat all points as inside (no noise injection)
        if self.trustregion_bounds is None:
            return torch.zeros(self.num_train, dtype=torch.bool)
        
        lower_bounds, upper_bounds = self.trustregion_bounds
        
        # Check if points are outside the trust region bounds
        outside_lower = torch.any(self.train_X < lower_bounds, dim=-1)
        outside_upper = torch.any(self.train_X > upper_bounds, dim=-1)
        outside_mask = outside_lower | outside_upper
        
        return outside_mask

    def _optimize_noise(self, covar_module, mean_module):
        """
        Configures the noise factors so that noise is injected outside the trust region.
        
        If noise_optional=True, compares MLL with and without noise injection and chooses the better option.
        """
        # Initialize fitted modules tuple to None
        self._best_fitted_modules = None
        
        noise_outside = 1.0
        noise_inside = NOISE_JITTER

        # Get trust region mask
        outside_mask = self._get_trust_region_mask()

        if self.noise_optional:
            # Compare MLL with and without noise injection outside trust region
            
            # Configuration 1: No noise injection (jitter everywhere)
            noise_factors_no_injection = torch.full((self.num_train,), NOISE_JITTER)
            
            # Configuration 2: Noise injection outside trust region
            noise_factors_with_injection = torch.where(outside_mask, noise_outside, noise_inside)
            
            # Evaluate both configurations
            try:
                mll_no_injection, fitted_covar_no_injection, fitted_mean_no_injection = mll_with_refit(
                    self.train_X, 
                    self.train_Y, 
                    noise_factors_no_injection, 
                    covar_module,
                    mean_module,
                    return_fitted_modules=True
                )
            except Exception as e:
                mll_no_injection = float('-inf')
                print(f"Error during MLL evaluation without noise injection: {str(e)}")
                fitted_covar_no_injection = None
                fitted_mean_no_injection = None
            
            try:
                mll_with_injection, fitted_covar_with_injection, fitted_mean_with_injection = mll_with_refit(
                    self.train_X, 
                    self.train_Y, 
                    noise_factors_with_injection, 
                    covar_module,
                    mean_module,
                    return_fitted_modules=True
                )
            except Exception as e:
                mll_with_injection = float('-inf')
                print(f"Error during MLL evaluation with noise injection: {str(e)}")
                fitted_covar_with_injection = None
                fitted_mean_with_injection = None
            
            # Choose the better configuration
            if mll_with_injection > mll_no_injection:
                # Use noise injection configuration
                noise_factors = noise_factors_with_injection
                # Store fitted modules as tuple (not as direct attributes to avoid PyTorch module assignment error)
                self._best_fitted_modules = (fitted_covar_with_injection, fitted_mean_with_injection)
                print(f"Trust region noise injection: Noise injection chosen (MLL: {mll_with_injection:.4f} > {mll_no_injection:.4f})")
            else:
                # Use no noise injection configuration
                noise_factors = noise_factors_no_injection
                # Store fitted modules as tuple (not as direct attributes to avoid PyTorch module assignment error)
                self._best_fitted_modules = (fitted_covar_no_injection, fitted_mean_no_injection)
                print(f"Trust region noise injection: No noise injection chosen (MLL: {mll_no_injection:.4f} > {mll_with_injection:.4f})")
        else:
            # Always use noise injection outside trust region
            noise_factors = torch.where(outside_mask, noise_outside, noise_inside)
            # No fitted modules for this case
            self._best_fitted_modules = None

        return noise_factors

    def _get_fitted_modules(self):
        """
        Get the fitted covariance and mean modules from the noise optimization.
        These modules have been fitted with the optimal noise configuration.
        
        Returns:
            tuple: (fitted_covar_module, fitted_mean_module) or (None, None) if fitting failed
        """
        if hasattr(self, '_best_fitted_modules') and self._best_fitted_modules is not None:
            return self._best_fitted_modules
        return None, None


class TrustRegionNoiseInjectionBeamSearchGP(TrustRegionNoiseInjectionGP):
    """
    Trust Region Noise Injection GP with Beam Search.
    This is a stateless model that gets recreated each iteration.
    The beam search state is managed externally and passed in during initialization.
    """
    
    def __init__(
        self,
        train_X,
        train_Y,
        outcome_transform=None,
        trustregion_center=None,
        trustregion_bounds=None,
        beam_width=5,
        noise_increment=1.0,
        existing_beams=None,
        normalize_noise=False,
        force_noise=False,
        verbose=True
    ):
        """
        Initialize the beam search GP model.
        
        Args:
            train_X: Training input data
            train_Y: Training target data
            outcome_transform: Optional outcome transform
            trustregion_center: Center of trust region (for reference/debugging)
            trustregion_bounds: Tuple of (lower_bounds, upper_bounds) for trust region
            beam_width: Number of beams to maintain
            noise_increment: Amount of noise to add when decision is True
            existing_beams: List of existing BeamNode objects from previous iteration
            normalize_noise: Whether to normalize noise vectors after creating candidates
            force_noise: If True, skip MLL calculation and always use additive noise injection
            verbose: Whether to print detailed logging information
        """
        self.beam_width = beam_width
        self.noise_increment = noise_increment
        self.existing_beams = existing_beams or []
        self.normalize_noise = normalize_noise
        self.force_noise = force_noise
        self.verbose = verbose
        
        # Log initialization
        if self.verbose:
            print(f"\n==== Beam Search Initialization ====")
            print(f"Training points: {len(train_X)}")
            print(f"Beam width: {beam_width}")
            print(f"Noise increment: {noise_increment}")
            print(f"Existing beams: {len(self.existing_beams)}")
            print(f"Normalize noise: {normalize_noise}")
            print(f"Force noise: {force_noise}")
            if trustregion_bounds is not None:
                lower_bounds, upper_bounds = trustregion_bounds
                print(f"Trust region bounds: lower={lower_bounds.tolist()}, upper={upper_bounds.tolist()}")
            else:
                print("Trust region bounds: None")
            print(f"===================================\n")
        
        # Initialize parent class
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            outcome_transform=outcome_transform,
            trustregion_center=trustregion_center,
            trustregion_bounds=trustregion_bounds,
        )
    

    
    def _optimize_noise(self, covar_module, mean_module):
        """
        Override to use centralized beam search for noise optimization.
        """
        if self.verbose:
            print(f"\n==== Beam Search Noise Optimization ====")
        
        # Get trust region mask
        outside_mask = self._get_trust_region_mask()
        current_size = len(self.train_X)
        
        # Create beam search strategy
        strategy = TrustRegionBeamSearchStrategy(
            mll_eval_func=mll_with_refit,  # Use the MLL function from utils
            noise_increment=self.noise_increment,
            force_noise=self.force_noise,
            verbose=self.verbose
        )
        
        # Create beam search engine
        engine = BeamSearchEngine(
            strategy=strategy,
            beam_width=self.beam_width,
            max_iterations=10,  # Reasonable limit for trust region beam search
            convergence_threshold=1e-6,
            stagnation_limit=3,
            verbose=self.verbose
        )
        
        # Initialize with existing beams or start fresh
        if self.existing_beams:
            engine.initialize_search(self.existing_beams)
        else:
            engine.initialize_search([])
        
        # Run beam search
        best_beam = engine.search(
            outside_mask=outside_mask,
            current_size=current_size,
            train_X=self.train_X,
            train_Y=self.train_Y,
            covar_module=covar_module,
            mean_module=mean_module
        )
        
        # Store results
        if best_beam is not None:

            # Save normalized noise vectors for all beams
            if self.normalize_noise:
                # Normalize noise vectors for all beams before storing
                for beam in engine.current_beams:
                    beam.noise_vector = self._normalize_noise_vector(beam.noise_vector)
                best_beam.noise_vector = self._normalize_noise_vector(best_beam.noise_vector)


            self.updated_beams = engine.current_beams
            self.best_beam = best_beam
            noise_factors = best_beam.noise_vector
            
            # Apply normalization if requested
            # Only normalizing the output noise vector, not the noise factors in the beam
            # if self.normalize_noise:
            #     noise_factors = self._normalize_noise_vector(noise_factors)
            
            if self.verbose:
                print(f"Best beam MLL: {best_beam.mll_score:.4f}")
                print(f"Best beam decisions: {best_beam.decision_history}")
                final_noise_stats = {
                    'min': noise_factors.min().item(),
                    'max': noise_factors.max().item(),
                    'mean': noise_factors.mean().item(),
                    'std': noise_factors.std().item(),
                    'high_noise_points': (noise_factors > 0.1).sum().item(),
                    'total_points': len(noise_factors)
                }
                print(f"Final noise vector statistics: {final_noise_stats}")
                print(f"=====================================\n")
        else:
            # Fallback configuration
            noise_factors = torch.full((current_size,), NOISE_JITTER)
            self.updated_beams = []
            self.best_beam = None
            if self.verbose:
                print("Beam search failed, using fallback configuration")
        
        return noise_factors
           
    def get_updated_beams(self):
        """
        Get the updated beams for the next iteration.
        This should be called after model fitting to get the beams to store externally.
        """
        if hasattr(self, 'updated_beams'):
            if self.verbose:
                print(f"Returning {len(self.updated_beams)} updated beams for next iteration")
            return self.updated_beams
        else:
            if self.verbose:
                print("No updated beams available - returning empty list")
            return []
    
    def _normalize_noise_vector(self, noise_vector):
        """
        Normalize a noise vector so that the maximum value becomes 1.0,
        while ensuring no value goes below NOISE_JITTER.
        
        Args:
            noise_vector: Tensor of noise values
            
        Returns:
            Normalized noise vector
        """
        max_noise = torch.max(noise_vector)
        
        if max_noise > 0:
            # Normalize to make max value = 1.0
            normalized_vector = noise_vector / max_noise
            # Clamp to ensure minimum value is NOISE_JITTER
            return torch.clamp(normalized_vector, min=NOISE_JITTER)
        else:
            # If all values are 0, set to NOISE_JITTER
            return torch.full_like(noise_vector, NOISE_JITTER)

    def _get_fitted_modules(self):
        """
        Get the best-fitted covariance and mean modules from the beam search.
        These modules have been fitted with the optimal noise configuration.
        
        Returns:
            tuple: (fitted_covar_module, fitted_mean_module) or (None, None) if no valid beam
        """
        if hasattr(self, 'best_beam') and self.best_beam is not None:
            if self.verbose:
                print(f"Returning fitted modules from best beam (MLL: {self.best_beam.mll_score:.4f})")
            return self.best_beam.fitted_covar_module, self.best_beam.fitted_mean_module
        else:
            if self.verbose:
                print("No best beam available - returning None fitted modules")
            return None, None