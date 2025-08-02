"""
Streamlit app for interactive noise injection via the multiplicative Gaussian likelihood into a GP model.

Run app with: streamlit run noise_injection.py
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import torch
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood

# Use higher precision throughout for better numerical stability (didn't fix spike error but is better anyway)
torch.set_default_dtype(torch.float64)

import os
import sys
import copy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from multiplicative_gaussian_likelihood import MultiplicativeGaussianLikelihood

import streamlit as st

##########################################################################
# 1. GP
##########################################################################

np.random.seed(0)

x = torch.linspace(0, 30, 400, dtype=torch.float64)

# Cache for function data to avoid repeated file reads
function_cache = {}

def f(x):
    """Interpolate function values from CSV file data"""
    # Use cached interpolation if available
    csv_path = st.session_state.csv_path
    if csv_path not in function_cache:
        # Read data from a CSV file
        data = pd.read_csv(csv_path, header=None, names=['x', 'y'])
        x_data = data['x'].values
        y_data = data['y'].values
        
        # Cache the interpolation function
        function_cache[csv_path] = interp1d(x_data, y_data, kind='linear', fill_value="extrapolate")
    
    # Use the cached interpolation function
    interpolation_function = function_cache[csv_path]

    # Handle both scalar and array inputs
    if isinstance(x, (float, int, np.number, torch.Tensor)) and not hasattr(x, "__len__"):
        # For scalar inputs
        return float(interpolation_function(float(x)))
    else:
        # For array-like inputs
        return interpolation_function(x)


# Define GP model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

# Function to train a GP model with specified noise factors
def train_gp_model(x_train, y_train, noise_factors, training_iterations=120):
    print("Training GP model with noise factors")
    noise_factors = noise_factors.view(-1, 1)

    likelihood = MultiplicativeGaussianLikelihood(factors=noise_factors)
    #likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(x_train, y_train, likelihood)

    model.train()
    likelihood.train()
    
    # Some params from likelihood are also in the model, avoid duplicate optimization
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
        
        if (i + 1) % 10 == 0:
            lengthscale = model.covar_module.base_kernel.lengthscale.item()
            output_var = model.covar_module.outputscale.item()
            noise_var = likelihood.noise.item()
            print(
                f"Iter {i+1:3d}/{training_iterations} | "
                f"Loss: {loss.item():8.3f} | "
                f"l: {lengthscale:6.3f} | "
                f"o_f²: {output_var:6.3f} | "
                f"o_n²: {noise_var:6.3f} | "
                f"mll: {mll(output, y_train).item():8.3f}"
            )

    return model, likelihood


# Function to get predictions from the fitted GP model
def get_predictions(model, likelihood, x_test=x):
    model.eval()
    likelihood.eval()

    likelihood_copy = copy.deepcopy(likelihood)
    
    with torch.no_grad():
        # Start with all‑ones noise factors for the test grid
        new_factors = torch.ones(x_test.shape[0], dtype=torch.float64, device=x_test.device)

        # Check for factors to make normal likelihood still trainable
        if hasattr(likelihood_copy.noise_covar, 'factors'):
            train_x = model.train_inputs[0].view(-1)            # shape (n_train,)
            train_factors = likelihood_copy.noise_covar.factors.view(-1)

            # For each training point, find matching indices in x_test
            for tx, tf in zip(train_x, train_factors):
                match = torch.isclose(x_test.view(-1), tx, atol=2e-1)
                if match.any():
                    new_factors[match] = tf

        # Set new factors to the likelihood copy
        likelihood_copy.noise_covar.factors = new_factors

        # predictions = likelihood_copy(model(x_test))
        predictions = model(x_test)
        mean = predictions.mean
        lower, upper = predictions.confidence_region()

        print(new_factors)

    return mean, lower, upper


# Function to calculate the marginal log likelihood (MLL) in a subset of the data
def calculate_mll(model, likelihood, x, y, factors=None):
    model.eval()
    likelihood.eval()

    # Ensure x has correct shape
    x = x.view(-1, 1) if x.dim() == 1 else x

    # A GP needs at least two points
    if x.shape[0] < 2:
        return None

    lh = copy.deepcopy(likelihood) # Make a deep copy of the likelihood to avoid modifying the original

    # Align noise factors to the subset length
    if hasattr(lh.noise_covar, "factors"):
        if factors is not None:
            lh.noise_covar.factors = factors.view(-1).requires_grad_(False)
        else:
            lh.noise_covar.factors = torch.ones(x.shape[0], device=x.device).requires_grad_(False)

    with torch.no_grad():
        mll = ExactMarginalLogLikelihood(lh, model)
        return mll(model(x), y).item()


# Helper function to highlight an area of interest on a plot
def highlight_area_of_interest(ax, area_of_interest, color='grey', alpha=0.2, label_text=None):
    if area_of_interest is not None and len(area_of_interest) == 2:
        start_x, end_x = area_of_interest
        y_min, y_max = ax.get_ylim()
        ax.axvspan(start_x, end_x, alpha=alpha, color=color)
        
        # Add an optional label to the area
        if label_text:
            # Place label at a fixed fraction above the bottom (e.g., 5% above y_min)
            y_label = y_min + 0.05 * (y_max - y_min)
            ax.text((start_x + end_x) / 2, y_label, label_text, 
                    horizontalalignment='center', color=color)
    return ax


# Function to predict and visualize GP model results
def predict_and_visualize(model, likelihood, x_train, y_train, x_test=x, title="GP Fit", show_ground_truth=True, area_of_interest=None):
    mean, lower, upper = get_predictions(model, likelihood, x_test)

    # ------------------------------------------------------------------
    # Acquisition Function: one‑point Expected Improvement (EI, maximisation)
    # ------------------------------------------------------------------
    with torch.no_grad():
        best_y = y_train.max()  # current best value (maximisation)
        sigma = (upper - lower) / 3.92  # 95% band -> 2*1.96*sigma
        sigma = torch.clamp(sigma, min=1e-9)
        improvement = mean - best_y
        Z = improvement / sigma
        normal = torch.distributions.Normal(0, 1)
        ei = improvement * normal.cdf(Z) + sigma * normal.log_prob(Z).exp()
        ei = torch.clamp(ei, min=0.0)

    # Calculate the marginal log likelihood for the overall dataset
    overall_mll = calculate_mll(
        model, likelihood,
        x_train, y_train,
        factors=likelihood.noise_covar.factors
    )
    
    # Calculate area of interest MLL if area_of_interest is specified
    area_mll = None
    if area_of_interest is not None and len(area_of_interest) == 2:
        start_x, end_x = area_of_interest
        area_mask = (x_train >= start_x) & (x_train <= end_x)
        num_points = torch.sum(area_mask).item()
        if num_points >= 1:
            area_x = x_train[area_mask]
            area_y = y_train[area_mask]
            area_factors = likelihood.noise_covar.factors[area_mask]
            # Marginal log‑likelihood for AoI points
            area_mll = calculate_mll(model, likelihood, area_x, area_y, factors=area_factors)

    # Create a figure with three subplots sharing the same x-axis
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=(12, 12),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1, 2]}
    )

    # Set the overall font size for the plots
    plt.rcParams.update({'font.size': 18})
    # Set size for axes labels and titles
    ax1.tick_params(labelsize=14)
    ax2.tick_params(labelsize=14)
    ax3.tick_params(labelsize=14)
    # Enforce 3 decimal places for y‑axis ticks
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))

    
    # Convert back to 1D for plotting if needed
    x_test_flat = x_test.flatten() if x_test.dim() > 1 else x_test
    x_train_flat = x_train.flatten() if x_train.dim() > 1 else x_train
    
    # UPPER SUBPLOT - GP fit and predictions
    
    # Plot predictive mean as blue line
    ax1.plot(x_test_flat.numpy(), mean.numpy(), color="#00549F", label='Mean', linewidth=4)
    
    # Shade between the lower and upper confidence bounds
    ax1.fill_between(x_test_flat.numpy(), lower.numpy(), upper.numpy(), alpha=0.8, label='Confidence', color="#8EBAE5")
    
    # Plot the ground truth function if requested
    if show_ground_truth:
        # Create dense points for the ground truth
        x_dense = np.linspace(x_test_flat.min().item(), x_test_flat.max().item(), 500)
        y_dense = f(x_dense)  # Using the f function defined earlier
        ax1.plot(x_dense, y_dense, linestyle='--', color="#612158", label='Ground Truth', linewidth=4)

    # Highlight area of interest if provided
    highlight_area_of_interest(ax1, area_of_interest, label_text='Area of Interest')
    
    # Plot training data as black stars
    ax1.scatter(x_train_flat.numpy(), y_train.numpy(), color='red', marker='*', label='Training Data', s=200, zorder=5)

    ax1.legend(loc='upper right')
    ax1.set_title(title)
    ax1.grid(True)
    
    # LOWER SUBPLOT - Noise factors
    # Get the noise factors from the likelihood's noise_covar
    if hasattr(likelihood.noise_covar, 'factors'):
        noise_factors_tensor = likelihood.noise_covar.factors
        
        # Make sure noise_factors_tensor is 1D
        noise_factors_tensor = noise_factors_tensor.view(-1) if noise_factors_tensor.dim() > 1 else noise_factors_tensor
        
        # If the number of factors doesn't match the number of training points, 
        # we'll plot only the ones we have (first n points)
        min_length = min(len(noise_factors_tensor), len(x_train_flat))
        
        # Create bar plot for noise factors
        bars = ax2.bar(x_train_flat[:min_length].numpy(), 
                       noise_factors_tensor[:min_length].numpy(), 
                       width=0.2, 
                       color='orange', 
                       label='Noise Factors')
        
        # Highlight area of interest in the noise factors subplot too
        highlight_area_of_interest(ax2, area_of_interest)

    # Scale the y axis alyways from 0 to 1
    ax2.set_ylim(0, 1.05)
    ax2.set_title('Noise Factors for Each Training Point')
    ax2.grid(True)
    ax2.legend()
    
    # ACQUISITION subplot
    ax3.plot(x_test_flat.numpy(), ei.numpy(), color="#0C8000", label='Expected Improvement', linewidth=4)
    
    # Mark the maximiser
    max_idx = torch.argmax(ei)
    max_x = x_test_flat[max_idx]
    max_ei = ei[max_idx]
    ax3.axvline(max_x.item(), color='gray', linestyle='--', alpha=0.6)
    ax3.plot(max_x.numpy(), max_ei.numpy(), 'ro', label='EI max', markersize=8)
    
    # Highlight area of interest for context
    highlight_area_of_interest(ax3, area_of_interest)
    
    ax3.set_xlabel('x', fontsize=14)
    ax3.set_title('Acquisition Function')
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    
    # Return the MLL values along with the figure
    return fig, overall_mll, area_mll



##########################################################################
# 2. Noise factor optimization
##########################################################################


# Function to optimize noise factors for area of interest using random search
def optimize_noise_factors(
        x_train,
        y_train,
        active_mask,
        area_start,
        area_end,
        initial_noise_factors
    ):

    print(f"TODO Implement optimization")
    return initial_noise_factors



##########################################################################
# 2. Streamlit app
##########################################################################

st.title("Interactive GP Regression with Noise Injection")

# ------------------------------------------------------------------
# Model selection
# ------------------------------------------------------------------
if "csv_path" not in st.session_state:
    st.session_state.csv_path = "test_functions/standardized_function_2.csv"

# Define available test functions
model_map = {
    "Model 1 (function 1)": "test_functions/standardized_function_1.csv",
    "Model 2 (function 2)": "test_functions/standardized_function_2.csv",
}

# Pre-defined design points for each model to ensure good coverage
design_points = {
    "Model 1 (function 1)": torch.tensor(
        [1.0, 5.0, 6.0, 10.0, 11.0, 11.5, 12.0, 12.5,
         13.0, 13.5, 14.0, 14.5, 17.0, 20.0, 21.0, 24.0, 30.0],
        dtype=torch.float64
    ),
    "Model 2 (function 2)": torch.tensor(
        [1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 11.0, 14.5, 18.0, 
         19.0, 21.0, 22.0, 25.0, 27.0, 28.0, 30.0], 
        dtype=torch.float64
    ),
}

# Show a dropdown to select the test function
selected_label = st.selectbox("Select test function", model_map.keys())
st.session_state.csv_path = model_map[selected_label]

# ---------------------------------------------------------------
# Reset training state if the selected model has changed
# ---------------------------------------------------------------
if "last_selected_model" not in st.session_state:
    st.session_state.last_selected_model = selected_label
elif selected_label != st.session_state.last_selected_model:
    # Clear training‑dependent session keys so the app starts fresh
    keys_to_reset = [
        "x_train", "y_train", "noise_factors", 
        "active_mask", "area_start", "area_end",
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    # Update memory and rerun to re‑initiate everything
    st.session_state.last_selected_model = selected_label
    st.rerun()

# Get initial design points for the selected model
initial_x_points = design_points[selected_label]

# Recompute initial y and noise vectors every run (safe if state exists already)
initial_y_points = torch.tensor(f(initial_x_points.numpy()), dtype=torch.float64)
noise_factors = torch.ones_like(initial_x_points)


# Function to sort tensors (and corresponding masks) by x values so that sliders are in order
def sort_tensors_by_x(x_tensor, y_tensor, noise_tensor, active_tensor):
    _, indices = torch.sort(x_tensor)
    sorted_x = x_tensor[indices]
    sorted_y = y_tensor[indices]
    sorted_noise = noise_tensor[indices]
    sorted_active = active_tensor[indices]
    return sorted_x, sorted_y, sorted_noise, sorted_active

# Store session state
if "x_train" not in st.session_state:
    st.session_state.x_train = initial_x_points.clone()
    st.session_state.y_train = initial_y_points.clone()
    st.session_state.noise_factors = noise_factors.clone()
    st.session_state.active_mask = torch.ones_like(st.session_state.x_train, dtype=torch.bool)
    # Sort initial tensors
    (st.session_state.x_train,
     st.session_state.y_train,
     st.session_state.noise_factors,
     st.session_state.active_mask) = sort_tensors_by_x(
        st.session_state.x_train,
        st.session_state.y_train,
        st.session_state.noise_factors,
        st.session_state.active_mask
    )
    # Initialize area of interest
    x_min, x_max = initial_x_points.min().item(), initial_x_points.max().item()
    st.session_state.area_start = x_min + (x_max - x_min) * 0.35
    st.session_state.area_end = x_min + (x_max - x_min) * 0.60

# ------------------------------------------------------------------
# Area of Interest section
# ------------------------------------------------------------------
st.subheader("Area of Interest")
min_x = float(st.session_state.x_train.min().item())
max_x = float(st.session_state.x_train.max().item())

col1, col2 = st.columns(2)
with col1:
    area_start = st.number_input(
        "Start X",
        min_value=min_x,
        max_value=max_x,
        value=float(st.session_state.area_start),
        step=0.5,
    )
with col2:
    area_end = st.number_input(
        "End X",
        min_value=min_x,
        max_value=max_x,
        value=float(st.session_state.area_end),
        step=0.5,
    )

# Update session state
st.session_state.area_start = area_start
st.session_state.area_end = area_end

# Ensure start is less than or equal to end
if st.session_state.area_start > st.session_state.area_end:
    st.session_state.area_start = st.session_state.area_end


# ------------------------------------------------------------------
# Noise‑factor controls
# ------------------------------------------------------------------
st.subheader("Noise Factor Controls")

# Buttons to change noise factors at once
col1, col2, col3, col4 = st.columns([1, 1, 1.5, 1.5])

# Set all noise factors to zero
with col1:
    if st.button("All Zero"):
        st.session_state.noise_factors = torch.zeros_like(st.session_state.noise_factors)
        st.rerun()

# Set all noise factors to one
with col2:
    if st.button("All One"):
        st.session_state.noise_factors = torch.ones_like(st.session_state.noise_factors)
        st.rerun()

# Set AoI noise to zero, outside noise to one
with col3:
    if st.button("AoI Zero / Outside One"):
        a_start = st.session_state.area_start
        a_end = st.session_state.area_end
        x_vals = st.session_state.x_train
        new_factors = torch.where(
            (x_vals >= a_start) & (x_vals <= a_end),
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(1.0, dtype=torch.float64),
        )
        st.session_state.noise_factors = new_factors
        st.rerun()
        
# Optimize noise factors for area of interest
with col4:
    if st.button("Optimize Noise"):
        a_start = st.session_state.area_start
        a_end = st.session_state.area_end
        # Run optimization
        optimized_factors = optimize_noise_factors(
            st.session_state.x_train,
            st.session_state.y_train,
            st.session_state.active_mask,
            a_start, a_end,
            st.session_state.noise_factors
        )
        # Update session state with optimized noise factors
        st.session_state.noise_factors = optimized_factors
        st.rerun()

with st.expander("Modify Individual Noise Factors", expanded=False):
    st.caption("Adjust point‑wise noise or toggle observations on/off.")

    # Create sliders for each noise factor and active toggle
    for i, x_val in enumerate(st.session_state.x_train):
        col_slider, col_toggle = st.columns([4, 1])
        with col_slider:
            st.session_state.noise_factors[i] = st.slider(
                f"Noise factor for x = {x_val.item():.2f}",
                min_value=0.0,
                max_value=5.0,
                value=float(st.session_state.noise_factors[i]),
                step=0.1,
                key=f"slider_noise_{i}_{x_val.item():.2f}"
            )
        with col_toggle:
            st.session_state.active_mask[i] = st.checkbox(
                "Observed",
                value=bool(st.session_state.active_mask[i]),
                key=f"active_{i}_{x_val.item():.2f}"
            )

# Filter active points
active_mask = st.session_state.active_mask
x_active = st.session_state.x_train[active_mask]
y_active = st.session_state.y_train[active_mask]
noise_active = st.session_state.noise_factors[active_mask]

# Need at least two active points to fit a GP
if x_active.shape[0] < 2:
    st.warning("Please keep at least two points active to train the GP.")
    st.stop()

# Train and visualize model
model, likelihood = train_gp_model(x_active, y_active, noise_active)
st.subheader("GP Model Fit")
fig_gp, overall_mll, area_mll = predict_and_visualize(
    model, likelihood,
    x_active,
    y_active,
    title="GP Fit with Current Training Points",
    show_ground_truth=True,
    area_of_interest=(st.session_state.area_start, st.session_state.area_end)
)
st.pyplot(fig_gp, dpi=300)

# ---------------------------------------------------------------
# Summary metrics table
# ---------------------------------------------------------------
lengthscale_val = model.covar_module.base_kernel.lengthscale.item()
output_var_val = model.covar_module.outputscale.item()
noise_var_val = likelihood.noise.item()

metric_df = pd.DataFrame({
    "Metric": [
        "Overall MLL",
        "AoI MLL",
        "Length‑scale",
        "Output variance",
        "Noise variance"
    ],
    "Value": [
        f"{overall_mll:.4f}" if overall_mll is not None else "n/a",
        f"{area_mll:.4f}"   if area_mll   is not None else "n/a",
        f"{lengthscale_val:.4f}",
        f"{output_var_val:.4f}",
        f"{noise_var_val:.4f}",
    ]
})
st.subheader("Model Diagnostics")
st.table(metric_df)

# ---------------------------------------------------------------
# Compute current EI maximiser for convenience button
# ---------------------------------------------------------------
x_test_grid = torch.linspace(st.session_state.x_train.min(),
                           st.session_state.x_train.max(),
                           200)
mean_test, lower_test, upper_test = get_predictions(model, likelihood, x_test_grid)

# Calculate Expected Improvement using the same logic as in predict_and_visualize
with torch.no_grad():
    best_y_current = y_active.max()
    sigma_test = torch.clamp((upper_test - lower_test) / 3.92, min=1e-9)  # 95% confidence band
    improv_test = mean_test - best_y_current
    Z_test = improv_test / sigma_test
    normal_dist = torch.distributions.Normal(0, 1)
    ei_test = improv_test * normal_dist.cdf(Z_test) + sigma_test * normal_dist.log_prob(Z_test).exp()
    ei_test = torch.clamp(ei_test, min=0.0)
    max_idx_ei = torch.argmax(ei_test)
    x_ei_max = x_test_grid[max_idx_ei].item()

# ---------------------------------------------------------------
# Add a new evaluation point section
# ---------------------------------------------------------------
st.subheader("Add New Point")

# User can either manually enter a point or add the EI maximizer
new_x_val = st.number_input("New x value", value=0.0, step=0.1)
col_add1, col_add2 = st.columns(2)

with col_add1:
    add_point = st.button("Add point to training data")
    
with col_add2:
    add_ei_point = st.button(f"Add EI max ({x_ei_max:.2f})")

# Handle point addition logic
if add_point or add_ei_point:
    # Determine which x value to add
    x_to_add = x_ei_max if add_ei_point else new_x_val
    
    # Evaluate the function at the new point
    new_y_val = f(x_to_add)
    
    # Add the new point to our training data
    st.session_state.x_train = torch.cat([st.session_state.x_train, torch.tensor([x_to_add], dtype=torch.float64)])
    st.session_state.y_train = torch.cat([st.session_state.y_train, torch.tensor([new_y_val], dtype=torch.float64)])
    st.session_state.noise_factors = torch.cat([st.session_state.noise_factors, torch.tensor([1.0], dtype=torch.float64)])
    st.session_state.active_mask = torch.cat([st.session_state.active_mask, torch.tensor([True], dtype=torch.bool)])

    # Sort tensors after adding a new point to keep UI elements in order
    (st.session_state.x_train,
     st.session_state.y_train,
     st.session_state.noise_factors,
     st.session_state.active_mask) = sort_tensors_by_x(
        st.session_state.x_train,
        st.session_state.y_train,
        st.session_state.noise_factors,
        st.session_state.active_mask
    )

    # Redraw the UI with updated data
    st.rerun()


