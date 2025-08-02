"""
Toy problem visualization demonstrating trust region and noise injection mechanisms in GPs.
Chart types: multi-panel plots showing GP models, uncertainty quantification, and noise injection effects.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import torch
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood

torch.set_default_dtype(torch.float64)

import os
import sys
import copy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from multiplicative_gaussian_likelihood import MultiplicativeGaussianLikelihood

import warnings
warnings.filterwarnings('ignore')

##########################################################################
# Configuration and Setup
##########################################################################

script_dir = os.path.dirname(os.path.abspath(__file__))
plt.style.use(os.path.join(script_dir, 'paper.mplstyle'))

# RWTH Colors
rwth_colors = {
    'blue': ["#00549F", "#407FB7", "#8EBAE5", "#C7DDF2", "#E8F1FA"],
    'black': ["#000000", "#646567", "#9C9E9F", "#CFD1D2", "#ECEDED"],
    'magenta': ["#E30066", "#E96088", "#F19EB1", "#F9D2DA", "#FDEEF0"],
    'yellow': ["#FFED00", "#FFF055", "#FFF59B", "#FFFAD1", "#FFFDEE"],
    'petrol': ["#006165", "#2D7F83", "#7DA4A7", "#BFD0D1", "#E6ECEC"],
    'tuerkis': ["#0098A1", "#00B1B7", "#89CCCF", "#CAE7E7", "#EBF6F6"],
    'green': ["#57AB27", "#8DC060", "#B8D698", "#DDEBCE", "#F2F7EC"],
    'maigreen': ["#BDCD00", "#D0D95C", "#E0E69A", "#F0F3D0", "#F9FAED"],
    'orange': ["#F6A800", "#FABE50", "#FDD48F", "#FEEAC9", "#FFF7EA"],
    'red': ["#CC071E", "#D85C41", "#E69679", "#F3CDBB", "#FAEBE3"],
    'bordeaux': ["#A11035", "#B65256", "#CD8B87", "#E5C0C0", "#F5E8E5"],
    'violet': ["#612158", "#834E75", "#A8859E", "#D2C0CD", "#EDE5EA"],
    'lila': ["#7A6FAC", "#9B91C1", "#BCB5D7", "#DEDAEB", "#F2F0F7"],
}

# Figure dimensions
pt = 1./72.27  # 72.27 points to an inch
text_width = 398.33862
fig_width = text_width * pt
golden = (1 + 5 ** 0.5) / 2
fig_height = fig_width / golden

# Output path
path = '/Users/julian/Library/CloudStorage/OneDrive-StudentsRWTHAachenUniversity/03 Research/02 BA/figures/thesis/background/'

# Configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(os.path.dirname(script_dir), "notebooks/test_functions/standardized_function_2.csv")
area_of_interest = (10.65, 19.40)

# Design points for Model 2
design_points = torch.tensor(
    [1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 11.0, 14.5, 18.0, 
     19.0, 21.0, 22.0, 25.0, 27.0, 28.0, 29.0], 
    dtype=torch.float64
)

##########################################################################
# Core Functions
##########################################################################

def f(x):
    """Interpolate function values from CSV file data"""
    # Read data from CSV file
    data = pd.read_csv(csv_path, header=None, names=['x', 'y'])
    x_data = data['x'].values
    y_data = data['y'].values
    
    # Create interpolation function
    interpolation_function = interp1d(x_data, y_data, kind='linear', fill_value="extrapolate")
    
    # Handle both scalar and array inputs
    if isinstance(x, (float, int, np.number, torch.Tensor)) and not hasattr(x, "__len__"):
        return float(interpolation_function(float(x)))
    else:
        return interpolation_function(x)

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_gp_model(x_train, y_train, noise_factors, training_iterations=120):
    print(f"Training GP model with {len(x_train)} points...")
    noise_factors = noise_factors.view(-1, 1)

    likelihood = MultiplicativeGaussianLikelihood(factors=noise_factors)
    model = ExactGPModel(x_train, y_train, likelihood)

    model.train()
    likelihood.train()
    
    # Set up optimizer
    model_params = list(model.parameters())
    lik_params = [p for p in likelihood.parameters() if p not in model_params]
    optimizer = torch.optim.Adam(model_params + lik_params, lr=0.1)
    
    mll = ExactMarginalLogLikelihood(likelihood, model)
    
    # Training loop
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(x_train)
        loss = -mll(output, y_train).sum()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 30 == 0:
            lengthscale = model.covar_module.base_kernel.lengthscale.item()
            output_var = model.covar_module.outputscale.item()
            noise_var = likelihood.noise.item()
            print(f"Iter {i+1:3d}/{training_iterations} | Loss: {loss.item():8.3f} | "
                  f"l: {lengthscale:6.3f} | o_f²: {output_var:6.3f} | o_n²: {noise_var:6.3f}")

    return model, likelihood

def get_predictions(model, likelihood, x_test):
    model.eval()
    likelihood.eval()

    likelihood_copy = copy.deepcopy(likelihood)
    
    with torch.no_grad():
        # Start with all-ones noise factors for the test grid
        new_factors = torch.ones(x_test.shape[0], dtype=torch.float64, device=x_test.device)

        # Check for factors to make normal likelihood still trainable
        if hasattr(likelihood_copy.noise_covar, 'factors'):
            train_x = model.train_inputs[0].view(-1)
            train_factors = likelihood_copy.noise_covar.factors.view(-1)

            # For each training point, find matching indices in x_test
            for tx, tf in zip(train_x, train_factors):
                match = torch.isclose(x_test.view(-1), tx, atol=2e-1)
                if match.any():
                    new_factors[match] = tf

        # Set new factors to the likelihood copy
        likelihood_copy.noise_covar.factors = new_factors

        predictions = model(x_test)
        mean = predictions.mean
        lower, upper = predictions.confidence_region()

    return mean, lower, upper

def calculate_expected_improvement(model, likelihood, x_test, y_train):
    mean, lower, upper = get_predictions(model, likelihood, x_test)
    
    with torch.no_grad():
        best_y = y_train.max()
        sigma = (upper - lower) / 3.92
        sigma = torch.clamp(sigma, min=1e-9)
        improvement = mean - best_y
        Z = improvement / sigma
        normal = torch.distributions.Normal(0, 1)
        ei = improvement * normal.cdf(Z) + sigma * normal.log_prob(Z).exp()
        ei = torch.clamp(ei, min=0.0)
    
    return ei

def highlight_area_of_interest(ax, area_of_interest, color='grey', alpha=0.2):
    if area_of_interest is not None and len(area_of_interest) == 2:
        start_x, end_x = area_of_interest
        y_min, y_max = ax.get_ylim()
        ax.axvspan(start_x, end_x, alpha=alpha, color=color)
    return ax

def create_comparison_subplot(ax1, ax2, ax3, model, likelihood, x_train, y_train, x_test, 
                            title_prefix, area_of_interest, show_ground_truth=True):
    
    # Get predictions
    mean, lower, upper = get_predictions(model, likelihood, x_test)
    ei = calculate_expected_improvement(model, likelihood, x_test, y_train)
    
    # Convert to numpy for plotting
    x_test_np = x_test.flatten().numpy()
    x_train_np = x_train.flatten().numpy()
    
    # UPPER SUBPLOT - GP fit and predictions
    ax1.plot(x_test_np, mean.numpy(), color=rwth_colors['blue'][0], linewidth=2)
    ax1.fill_between(x_test_np, lower.numpy(), upper.numpy(), 
                     alpha=0.8, color=rwth_colors['blue'][2])
    
    # Plot the ground truth function
    if show_ground_truth:
        x_dense = np.linspace(x_test_np.min(), x_test_np.max(), 500)
        y_dense = f(x_dense)
        ax1.plot(x_dense, y_dense, linestyle='--', color='k', linewidth=2, alpha=0.7)
    
    highlight_area_of_interest(ax1, area_of_interest)
    
    # Plot training data
    ax1.scatter(x_train_np, y_train.numpy(), color=rwth_colors['orange'][0], 
               marker='o', s=20, zorder=5)
    
    ax1.set_xlim(0, 30)
    
    # MIDDLE SUBPLOT - Noise factors
    if hasattr(likelihood.noise_covar, 'factors'):
        noise_factors_tensor = likelihood.noise_covar.factors.view(-1)
        min_length = min(len(noise_factors_tensor), len(x_train_np))
        
        bars = ax2.bar(x_train_np[:min_length], 
                      noise_factors_tensor[:min_length].numpy(), 
                      width=0.5, color=rwth_colors['tuerkis'][0])
        
        highlight_area_of_interest(ax2, area_of_interest)
    
    ax2.set_ylim(0, 1.05)
    ax2.set_xlim(0, 30)
    
    # LOWER SUBPLOT - Expected Improvement
    ax3.plot(x_test_np, ei.numpy(), color=rwth_colors['green'][0], linewidth=2)
    
    # Mark the maximizer
    max_idx = torch.argmax(ei)
    max_x = x_test[max_idx]
    max_ei = ei[max_idx]
    ax3.axvline(max_x.item(), color=rwth_colors['red'][0], linestyle=':', alpha=0.6)
    ax3.plot(max_x.numpy(), max_ei.numpy(), color=rwth_colors['red'][0], marker='o', markersize=np.sqrt(20))
    ax3.set_ylim(-0.005, 0.125)  # Set y-limits for EI plot

    highlight_area_of_interest(ax3, area_of_interest)
    
    ax3.set_xlim(0, 30)
    
    ax3.set_xlabel(r'$x$')

##########################################################################
# Main Execution
##########################################################################

def main():
    np.random.seed(0)
    torch.manual_seed(0) # Necessary?
    
    # Prepare data
    x_train = design_points.clone()
    y_train = torch.tensor(f(x_train.numpy()), dtype=torch.float64)
    x_test = torch.linspace(0, 30, 400, dtype=torch.float64)
    
    # Create noise factor configurations
    # Configuration 1: All zeros (no noise)
    noise_factors_no_noise = torch.zeros_like(x_train)
    
    # Configuration 2: Zero inside AoI, ones outside
    area_start, area_end = area_of_interest
    noise_factors_aoi_zero = torch.where(
        (x_train >= area_start) & (x_train <= area_end),
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64)
    )
    
    # Train both models
    print("Training Model 1: No noise at all points")
    model1, likelihood1 = train_gp_model(x_train, y_train, noise_factors_no_noise)
    
    print("\nTraining Model 2: No noise in AoI, noise outside")
    model2, likelihood2 = train_gp_model(x_train, y_train, noise_factors_aoi_zero)
    
    # Create the comparison plot
    fig, axes = plt.subplots(3, 2, figsize=(fig_width, 0.8*fig_height), 
                            sharex=True, sharey='row',
                            gridspec_kw={'height_ratios': [3, 1, 2]})
    
    # Use original font size since figure width matches text width
    plt.rcParams.update({'font.size': 10})
    
    # Left column: No noise configuration
    create_comparison_subplot(axes[0,0], axes[1,0], axes[2,0], 
                             model1, likelihood1, x_train, y_train, x_test,
                             "No Noise", area_of_interest)
    
    # Right column: AoI zero noise configuration
    create_comparison_subplot(axes[0,1], axes[1,1], axes[2,1], 
                             model2, likelihood2, x_train, y_train, x_test,
                             "AoI Zero Noise", area_of_interest)
    
    # Set shared labels
    axes[2,0].set_xlabel(r'$x$')
    axes[2,1].set_xlabel(r'$x$')
    
    # Set y-labels for left column with consistent padding
    axes[0,0].set_ylabel(r'$f(x)$', labelpad=6)
    axes[1,0].set_ylabel(r'$\lambda_i$', labelpad=6)
    axes[2,0].set_ylabel(r'$\alpha_{\text{EI}}(x)$', labelpad=6)

    # Align y-labels to the same horizontal position
    fig.align_ylabels([axes[0,0], axes[1,0], axes[2,0]])
    
    plt.tight_layout()
    
    output_filename = path + 'noise_injection_comparison.pdf'
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"\nChart saved to: {output_filename}")
    
    plt.show()

if __name__ == "__main__":
    main()
