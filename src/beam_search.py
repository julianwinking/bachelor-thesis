import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Callable, Tuple, Dict, Any
from abc import ABC, abstractmethod

from src.utils.constants import NOISE_JITTER


@dataclass
class BeamNode:
    """
    Represents a single beam node in the beam search tree.
    Contains the noise configuration and associated metadata.
    """
    noise_vector: torch.Tensor  # Noise configuration for each training point
    decision_history: List[bool]  # History of decisions that led to this configuration
    mll_score: float = 0.0  # Marginal likelihood score for this configuration
    fitted_covar_module: Optional[object] = None  # Fitted covariance module
    fitted_mean_module: Optional[object] = None  # Fitted mean module
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def copy(self):
        """Create a deep copy of this beam node."""
        return BeamNode(
            noise_vector=self.noise_vector.clone(),
            decision_history=self.decision_history.copy(),
            mll_score=self.mll_score,
            fitted_covar_module=self.fitted_covar_module,
            fitted_mean_module=self.fitted_mean_module,
            metadata=self.metadata.copy() if self.metadata else {}
        )
    
    def set_metadata(self, key: str, value: Any):
        """Set metadata key-value pair, ensuring metadata dict exists."""
        if self.metadata is None:
            self.metadata = {}
        self.metadata[key] = value


class BeamSearchStrategy(ABC):
    """
    Abstract base class for beam search strategies.
    Defines the interface for different beam search approaches.
    """
    
    @abstractmethod
    def generate_candidates(self, beams: List[BeamNode], **kwargs) -> List[BeamNode]:
        """
        Generate candidate beam nodes from existing beams.
        
        Args:
            beams: List of existing beam nodes
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            List of candidate beam nodes
        """
        pass
    
    @abstractmethod
    def evaluate_beam(self, beam: BeamNode, **kwargs) -> Tuple[float, Optional[object], Optional[object]]:
        """
        Evaluate a beam node and return its score and fitted modules.
        
        Args:
            beam: BeamNode to evaluate
            **kwargs: Additional evaluation parameters
            
        Returns:
            Tuple of (mll_score, fitted_covar_module, fitted_mean_module)
        """
        pass


class HeuristicBeamSearchStrategy(BeamSearchStrategy):
    """
    Heuristic beam search strategy for general noise optimization.
    
    This strategy systematically explores adding/removing noise from training points.
    """
    
    def __init__(self, mll_eval_func: Callable, factor_no_noise: float = NOISE_JITTER, 
                 factor_with_noise: float = 1.0):
        self.mll_eval_func = mll_eval_func
        self.factor_no_noise = factor_no_noise
        self.factor_with_noise = factor_with_noise
    
    def generate_candidates(self, beams: List[BeamNode], **kwargs) -> List[BeamNode]:
        """
        Generate candidates by adding/removing noise from training points.
        
        Args:
            beams: Existing beam nodes
            **kwargs: Should contain 'search_mode' ('add_noise' or 'remove_noise')
                     and 'num_train' (number of training points)
        """
        search_mode = kwargs.get('search_mode', 'add_noise')
        num_train = kwargs.get('num_train')
        
        if num_train is None:
            raise ValueError("num_train must be provided in kwargs")
        
        candidates = []
        candidates_seen = set()
        
        for beam_idx, beam in enumerate(beams):
            # Always add "no action" candidate
            noise_tuple = tuple(torch.round(beam.noise_vector, decimals=4).tolist())
            if noise_tuple not in candidates_seen:
                candidates_seen.add(noise_tuple)
                no_action_candidate = beam.copy()
                no_action_candidate.set_metadata('parent_beam_idx', beam_idx)
                candidates.append(no_action_candidate)
            
            # Generate action candidates based on search mode
            if search_mode == "remove_noise":
                # Add points to no-noise set (remove noise)
                no_noise_indices = (beam.noise_vector <= self.factor_no_noise + 1e-6).nonzero(as_tuple=True)[0]
                no_noise_set = set(no_noise_indices.tolist())
                remaining_indices = [i for i in range(num_train) if i not in no_noise_set]
                
                for point_idx in remaining_indices:
                    new_candidate = beam.copy()
                    new_candidate.noise_vector[point_idx] = self.factor_no_noise
                    new_candidate.decision_history.append(True)
                    new_candidate.set_metadata('parent_beam_idx', beam_idx)
                    new_candidate.set_metadata('modified_point', point_idx)
                    
                    noise_tuple = tuple(torch.round(new_candidate.noise_vector, decimals=4).tolist())
                    if noise_tuple not in candidates_seen:
                        candidates_seen.add(noise_tuple)
                        candidates.append(new_candidate)
            
            else:  # search_mode == "add_noise"
                # Remove points from no-noise set (add noise)
                no_noise_indices = (beam.noise_vector <= self.factor_no_noise + 1e-6).nonzero(as_tuple=True)[0]
                
                for point_idx in no_noise_indices:
                    new_candidate = beam.copy()
                    new_candidate.noise_vector[point_idx] = self.factor_with_noise
                    new_candidate.decision_history.append(True)
                    new_candidate.set_metadata('parent_beam_idx', beam_idx)
                    new_candidate.set_metadata('modified_point', point_idx)
                    
                    noise_tuple = tuple(torch.round(new_candidate.noise_vector, decimals=4).tolist())
                    if noise_tuple not in candidates_seen:
                        candidates_seen.add(noise_tuple)
                        candidates.append(new_candidate)
        
        return candidates
    
    def evaluate_beam(self, beam: BeamNode, **kwargs) -> Tuple[float, Optional[object], Optional[object]]:
        """
        Evaluate a beam using the provided MLL evaluation function.
        
        Args:
            beam: BeamNode to evaluate
            **kwargs: Should contain 'train_X', 'train_Y', 'covar_module', 'mean_module'
        """
        train_X = kwargs['train_X']
        train_Y = kwargs['train_Y']
        covar_module = kwargs['covar_module']
        mean_module = kwargs['mean_module']
        
        try:
            mll_score = self.mll_eval_func(train_X, train_Y, beam.noise_vector, covar_module, mean_module)
            return mll_score, None, None
        except Exception as e:
            return float('-inf'), None, None


class TrustRegionBeamSearchStrategy(BeamSearchStrategy):
    """
    Trust region beam search strategy for noise optimization.
    
    This strategy focuses on noise injection outside trust regions.
    """
    
    def __init__(self, mll_eval_func: Callable, noise_increment: float = 1.0, 
                 force_noise: bool = False, verbose: bool = True):
        self.mll_eval_func = mll_eval_func
        self.noise_increment = noise_increment
        self.force_noise = force_noise
        self.verbose = verbose
    
    def generate_candidates(self, beams: List[BeamNode], **kwargs) -> List[BeamNode]:
        """
        Generate candidates by adding noise outside trust regions.
        
        Args:
            beams: Existing beam nodes
            **kwargs: Should contain 'outside_mask' (boolean mask for points outside trust region)
                     and 'current_size' (current training set size)
        """
        outside_mask = kwargs.get('outside_mask')
        current_size = kwargs.get('current_size')
        
        if outside_mask is None or current_size is None:
            raise ValueError("Fehler: outside_mask and current_size must be provided in kwargs")
        
        candidates = []
        
        if not beams:
            # Initialize first beam with noise jitter
            initial_beam = BeamNode(
                noise_vector=torch.full((current_size,), NOISE_JITTER),
                decision_history=[]
            )
            beams = [initial_beam]
        
        for beam_idx, beam in enumerate(beams):
            # Handle noise vector size extension if new points were added
            if len(beam.noise_vector) < current_size:
                num_new_points = current_size - len(beam.noise_vector)
                extension = torch.full((num_new_points,), NOISE_JITTER)
                beam.noise_vector = torch.cat([beam.noise_vector, extension])
            
            # Branch A: Add noise to points outside trust region
            branch_a = beam.copy()
            branch_a.noise_vector = branch_a.noise_vector + outside_mask.float() * self.noise_increment
            branch_a.noise_vector = torch.clamp(branch_a.noise_vector, min=NOISE_JITTER)
            branch_a.decision_history.append(True)
            branch_a.set_metadata('parent_beam_idx', beam_idx)
            branch_a.set_metadata('action', 'add_noise')
            candidates.append(branch_a)
            
            # Branch B: Don't add noise (keep current noise vector)
            if not self.force_noise:
                branch_b = beam.copy()
                branch_b.noise_vector = torch.clamp(branch_b.noise_vector, min=NOISE_JITTER)
                branch_b.decision_history.append(False)
                branch_b.set_metadata('parent_beam_idx', beam_idx)
                branch_b.set_metadata('action', 'no_noise')
                candidates.append(branch_b)
        
        return candidates
    
    def evaluate_beam(self, beam: BeamNode, **kwargs) -> Tuple[float, Optional[object], Optional[object]]:
        """
        Evaluate a beam using MLL with refitting.
        
        Args:
            beam: BeamNode to evaluate
            **kwargs: Should contain 'train_X', 'train_Y', 'covar_module', 'mean_module'
        """
        train_X = kwargs['train_X']
        train_Y = kwargs['train_Y']
        covar_module = kwargs['covar_module']
        mean_module = kwargs['mean_module']
        
        # If force_noise is True, skip MLL calculation and return -inf
        if self.force_noise:
            return float('-inf'), None, None
        
        try:
            # Import here to avoid circular imports
            from src.utils.utils import mll_with_refit
            
            mll_score, fitted_covar, fitted_mean = mll_with_refit(
                train_X, train_Y, beam.noise_vector, covar_module, mean_module,
                return_fitted_modules=True
            )
            return mll_score, fitted_covar, fitted_mean
        except Exception as e:
            return float('-inf'), None, None


class BeamSearchEngine:
    """
    Core beam search engine that coordinates the search process.
    
    This engine is strategy-agnostic and can work with different beam search strategies.
    """
    
    def __init__(self, strategy: BeamSearchStrategy, beam_width: int = 5, 
                 max_iterations: Optional[int] = None, convergence_threshold: float = 1e-6,
                 stagnation_limit: int = 5, verbose: bool = True):
        self.strategy = strategy
        self.beam_width = beam_width
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.stagnation_limit = stagnation_limit
        self.verbose = verbose
        
        # Search state
        self.current_beams: List[BeamNode] = []
        self.best_beam: Optional[BeamNode] = None
        self.iteration_count = 0
        self.converged = False
        self.best_mll_history: List[float] = []
    
    def initialize_search(self, initial_beams: Optional[List[BeamNode]] = None):
        """
        Initialize the search with initial beam configurations.
        
        Args:
            initial_beams: List of initial beam nodes. If None, strategy will create initial beams.
        """
        self.current_beams = initial_beams or []
        self.best_beam = None
        self.iteration_count = 0
        self.converged = False
        self.best_mll_history = []
    
    def step(self, **kwargs) -> bool:
        """
        Perform one step of beam search.
        
        Args:
            **kwargs: Strategy-specific parameters
            
        Returns:
            bool: True if search should continue, False if converged
        """
        self.iteration_count += 1
        
        # Generate candidates
        candidates = self.strategy.generate_candidates(self.current_beams, **kwargs)
        
        # Deduplicate candidates
        candidates = self._deduplicate_candidates(candidates)
        
        # Evaluate candidates
        for candidate in candidates:
            mll_score, fitted_covar, fitted_mean = self.strategy.evaluate_beam(candidate, **kwargs)
            candidate.mll_score = mll_score
            candidate.fitted_covar_module = fitted_covar
            candidate.fitted_mean_module = fitted_mean
        
        # Select best beams
        candidates.sort(key=lambda x: x.mll_score, reverse=True)
        self.current_beams = candidates[:self.beam_width]
        
        # Update best beam
        if self.current_beams:
            current_best = self.current_beams[0]
            if self.best_beam is None or current_best.mll_score > self.best_beam.mll_score:
                self.best_beam = current_best
        
        # Check convergence
        if self.current_beams:
            current_best_mll = self.current_beams[0].mll_score
            self.best_mll_history.append(current_best_mll)
            
            if self._check_convergence():
                self.converged = True
                if self.verbose:
                    print(f"Beam search converged at iteration {self.iteration_count}")
                return False
        
        # Check max iterations
        if self.max_iterations and self.iteration_count >= self.max_iterations:
            if self.verbose:
                print(f"Beam search reached max iterations ({self.max_iterations})")
            return False
        
        return True
    
    def search(self, **kwargs) -> Optional[BeamNode]:
        """
        Run the complete beam search process.
        
        Args:
            **kwargs: Strategy-specific parameters
            
        Returns:
            BeamNode: The best beam found, or None if no valid beam found
        """
        if self.verbose:
            print(f"Starting beam search with width={self.beam_width}")
        
        while self.step(**kwargs):
            if self.verbose and self.iteration_count % 10 == 0:
                current_best_mll = self.current_beams[0].mll_score if self.current_beams else float('-inf')
                print(f"Iteration {self.iteration_count}: Best MLL = {current_best_mll:.4f}")
        
        if self.verbose:
            final_mll = self.best_beam.mll_score if self.best_beam else float('-inf')
            print(f"Beam search completed: Best MLL = {final_mll:.4f}")
        
        return self.best_beam
    
    def _deduplicate_candidates(self, candidates: List[BeamNode]) -> List[BeamNode]:
        """Remove candidates with identical noise patterns."""
        unique_candidates = []
        seen_patterns = set()
        
        for candidate in candidates:
            noise_hash = tuple(torch.round(candidate.noise_vector, decimals=4).tolist())
            if noise_hash not in seen_patterns:
                seen_patterns.add(noise_hash)
                unique_candidates.append(candidate)
        
        return unique_candidates
    
    def _check_convergence(self) -> bool:
        """Check if the search has converged."""
        if len(self.best_mll_history) < self.stagnation_limit + 1:
            return False
        
        recent_improvements = [
            self.best_mll_history[-i] - self.best_mll_history[-(i+1)]
            for i in range(1, self.stagnation_limit + 1)
        ]
        
        return max(recent_improvements) < self.convergence_threshold
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get statistics about the search process."""
        return {
            'iterations': self.iteration_count,
            'converged': self.converged,
            'best_mll': self.best_beam.mll_score if self.best_beam else None,
            'beam_width': self.beam_width,
            'mll_history': self.best_mll_history.copy()
        }


def find_optimal_beam_indices(scores: np.ndarray, parents: np.ndarray, 
                            beam_size: int, discard_fraction: float = 1.0/3.0) -> np.ndarray:
    """
    Find optimal beam indices with diversity consideration using greedy selection.
    
    Args:
        scores: Array of MLL scores for candidates
        parents: Array of parent beam indices for candidates
        beam_size: Number of beams to select
        discard_fraction: Fraction of candidates to discard from bottom
        
    Returns:
        Array of selected candidate indices
    """
    if beam_size >= len(scores):
        return np.arange(len(scores))
    
    # Discard bottom fraction of candidates
    num_keep = min(len(scores), round((1.0 - discard_fraction) * (2 * beam_size)))
    candidate_indices = np.argsort(-scores)[:num_keep]
    candidate_scores = scores[candidate_indices]
    candidate_parents = parents[candidate_indices]
    
    # Greedy selection for diversity
    selected_indices = []
    used_parents = set()
    
    # First pass: select best candidate from each unique parent
    for idx in range(len(candidate_indices)):
        parent = candidate_parents[idx]
        if parent not in used_parents and len(selected_indices) < beam_size:
            selected_indices.append(candidate_indices[idx])
            used_parents.add(parent)
    
    # Second pass: fill remaining slots with highest scoring candidates
    for idx in range(len(candidate_indices)):
        if len(selected_indices) >= beam_size:
            break
        if candidate_indices[idx] not in selected_indices:
            selected_indices.append(candidate_indices[idx])
    
    return np.array(selected_indices[:beam_size])