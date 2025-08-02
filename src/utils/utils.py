import os
import re
import random
import pickle
import torch
import numpy as np
from copy import deepcopy

import gpytorch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from ..multiplicative_gaussian_likelihood import MultiplicativeGaussianLikelihood


def log(verbose, msg: str):
        if verbose:
            print(msg)


def numpy_wrapper(func):
    def wrapper(x):
        if isinstance(x, np.ndarray):
            return func(torch.from_numpy(x)).numpy()
        else:
            return func(x).squeeze().numpy()

    return wrapper


def convert_to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().squeeze().numpy()
    if isinstance(x, list):
        # Convert each element in the list to numpy if possible
        return np.array(
            [convert_to_numpy(item) for item in x]
        )  # Keep object dtype to handle mixed types safely
    if isinstance(x, dict):
        return {key: convert_to_numpy(value) for key, value in x.items()}
    return x


def data2pickle(data, name):
    with open(name, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle2data(name):
    with open(name, "rb") as handle:
        data = pickle.load(
            handle,
        )
    return data


def seed_everything(seed=0):
    """
    Sets the random seed for various libraries to ensure reproducibility.

    Args:
    - seed (int): Seed value. Default is 0.
    """

    # Set seed for the Python standard library's random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)


def find_matching_folders(base_name, directory="."):
    """
    Find all folders matching the pattern '{base_name}_{number}' in the specified directory.

    Args:
        base_name (str): The base name to search for (e.g., 'sqpbo_toy2D')
        directory (str): Directory to search in (defaults to current directory)

    Returns:
        list: Sorted list of matching folder names
    """
    # Get all items in the directory
    all_items = os.listdir(directory)

    # Filter only directories
    folders = [
        item for item in all_items if os.path.isdir(os.path.join(directory, item))
    ]

    # Create regex pattern for "base_name_{number}"
    pattern = f"^{re.escape(base_name)}_(\d+)$"

    # Filter folders that match the pattern
    matching_folders = [folder for folder in folders if re.match(pattern, folder)]

    # Sort folders numerically by their number
    matching_folders.sort(key=lambda x: int(re.match(pattern, x).group(1)))

    return matching_folders


def mll_with_refit(
    train_X,
    train_Y,
    factors,
    covar_module,
    mean_module,
    return_fitted_modules=False,
):
    """
    Return Exact MLL after refitting GP hyperparameters for given per-point noise factors.
    Uses deepcopies of modules to avoid side-effects.
    Optionally return fitted modules.
    
    Args:
        train_X: Training inputs
        train_Y: Training targets
        factors: Noise factors for each training point
        covar_module: Covariance module
        mean_module: Mean module
        preserve_grad: Dummy arg for compatibility
        return_fitted_modules: If True, return (mll_value, fitted_covar, fitted_mean)
        
    Returns:
        float or tuple: MLL value, or (MLL value, fitted_covar_module, fitted_mean_module)
    """
    # Use a temporary GP model for fitting. We deepcopy the modules to avoid
    # modifying the originals during this evaluation.
    temp_covar_module = deepcopy(covar_module)
    temp_mean_module = deepcopy(mean_module)

    _likelihood = MultiplicativeGaussianLikelihood(factors=factors)

    # Use SingleTaskGP for consistency with the main model in the BO loop.
    # This is more robust than a minimal gpytorch.models.ExactGP.
    temp_model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,  # This train_Y is expected to be already transformed
        likelihood=_likelihood,
        covar_module=temp_covar_module,
        mean_module=temp_mean_module,
    )
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(temp_model.likelihood, temp_model)

    try:
        # Fit the temporary model
        fit_gpytorch_mll(mll, max_retries=1)  # Only 1 retry to avoid long waits

        # After fitting, set model to train() mode to avoid GPInputWarning when
        # computing MLL on training data.
        temp_model.train()

        # Return the MLL of the *fitted* model
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            output = temp_model(*temp_model.train_inputs)
            mll_value = mll(output, temp_model.train_targets).item()
        
        if return_fitted_modules:
            return mll_value, temp_covar_module, temp_mean_module
        else:
            return mll_value
            
    except Exception:
        # If fitting fails for some reason, return -inf
        if return_fitted_modules:
            return -float("inf"), None, None
        else:
            return -float("inf")


def mll_for_factors(
    train_X,
    train_Y,
    factors,
    covar_module,
    mean_module,
    preserve_grad=False,
):
    """
    Return Exact MLL for given per‑point noise factors.
    
    Args:
        train_X: Training inputs
        train_Y: Training targets (expected to have shape (n, 1); will be squeezed)
        factors: Noise factors for each training point
        covar_module: Covariance module for the GP
        mean_module: Mean module for the GP
        preserve_grad: If True, return tensor with gradients. If False, return Python float.
        
    Returns:
        float or torch.Tensor: MLL value
    """
    _likelihood = MultiplicativeGaussianLikelihood(factors=factors, preserve_factors_grad=preserve_grad)
    class _TempExactGP(gpytorch.models.ExactGP):
        # Temporary Exact GP model for MLL computation
        def __init__(self, X, Y, likelihood, c_module, m_module):
            super().__init__(X, Y, likelihood)
            self.mean_module = m_module
            self.covar_module = c_module

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x)
            )

    _temp = _TempExactGP(
        train_X, train_Y.squeeze(-1), _likelihood, covar_module, mean_module
    )
    _temp.train(); _likelihood.train()
    mll_obj = gpytorch.mlls.ExactMarginalLogLikelihood(_likelihood, _temp)
    try:
        mll_tensor = mll_obj(_temp(train_X), train_Y.squeeze(-1))
        return mll_tensor if preserve_grad else mll_tensor.item()
    except Exception:
        if preserve_grad:
            return torch.tensor(-float("inf"), dtype=train_X.dtype, device=train_X.device)
        else:
            return -float("inf")