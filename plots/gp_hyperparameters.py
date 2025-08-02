"""
Analyzes and visualizes Gaussian process hyperparameter evolution during optimization.
Chart types: Line plots showing lengthscale, signal variance, and noise variance trajectories.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from utils import (
    setup_project_path, 
    extract_info_from_result_dir, 
    get_acquisition_function_display_names,
    get_method_color_map,
    get_default_markers,
    get_method_display_name,
    get_objective_display_name,
    get_sorted_method_names,
    filter_results_by_criteria,
    group_results_by_dimension,
    ensure_output_dir,
    save_plot_figure,
    convert_to_numpy,
    group_methods_by_type,
    load_and_filter_results,
    create_grouped_legend,
    setup_plot_style,
)


def main(objective=None, dim=None, seed=0, methods=None, acquisition_function=None, save_plots=False, output_dir=None, sweep="debug", method_alpha=None, hyperparameter_type="all"):
    
    # Base path for results
    base_path = os.path.join(project_root, "results", sweep)
    
    if not os.path.exists(base_path):
        print(f"Results directory not found at {base_path}")
        return

    # Create output directory if saving plots
    if save_plots:
        prefix = f"{objective}_{dim}" if objective and dim else "gp_hyperparameters"
        output_dir = ensure_output_dir(output_dir, prefix, acquisition_function, seed)

    # Load and filter results using common function
    results = load_and_filter_results(
        base_path, objective, dim, seed, acquisition_function, methods
    )
    
    if results[0] is None:  # Check if loading failed
        return
        
    all_results, all_objectives, all_dimensions, all_acquisition_functions = results
    
    print(f"Found acquisition functions: {all_acquisition_functions}")
    print(f"Found objectives: {all_objectives}")
    print(f"Found dimensions: {all_dimensions}")
    
    # Simplified plot generation logic
    objectives_to_plot = [objective] if objective else all_objectives
    acq_funcs_to_plot = [acquisition_function] if acquisition_function else all_acquisition_functions
    
    for obj in objectives_to_plot:
        for acq_func in acq_funcs_to_plot:
            print(f"\nGenerating GP hyperparameter plots for objective: {obj}, acquisition function: {acq_func}")
            generate_hyperparameter_plots(
                all_results=all_results, 
                objective=obj, 
                acquisition_function=acq_func, 
                save_plots=save_plots, 
                output_dir=output_dir,
                methods=methods,
                method_alpha=method_alpha,
                hyperparameter_type=hyperparameter_type
            )


def generate_hyperparameter_plots(all_results, objective, acquisition_function, save_plots=False, 
                                 output_dir=None, methods=None, method_alpha=None, 
                                 hyperparameter_type="all"):
  
    # Filter results for the specified objective and acquisition function
    filtered_results = filter_results_by_criteria(
        all_results, 
        objective=objective,
        acquisition_function=acquisition_function
    )
    
    if not filtered_results:
        print(f"No results found for objective: {objective}, acquisition function: {acquisition_function}")
        return
    
    # Group results by dimension
    dimension_groups = group_results_by_dimension(filtered_results)
    
    # Set plot style once for all plots
    setup_plot_style(PLOT_STYLE)
    
    # Create plots for each dimension
    for dim, dim_results in dimension_groups.items():
        # Extract hyperparameter data
        hyperparameter_data = extract_hyperparameter_data(dim_results, dim)
        
        if not hyperparameter_data:
            print(f"No valid hyperparameter data found for objective: {objective}, "
                  f"acquisition function: {acquisition_function}, dimension: {dim}")
            continue
        
        # Filter out methods with no valid data
        valid_methods = {}
        for method_name, data in hyperparameter_data.items():
            has_valid_data = (
                (data['lengthscales'] and len(data['lengthscales']) > 0) or
                (data['signal_variances'] and len(data['signal_variances']) > 0) or
                (data['noise_variances'] and len(data['noise_variances']) > 0)
            )
            if has_valid_data:
                valid_methods[method_name] = data
        
        if not valid_methods:
            print(f"No methods with valid hyperparameter data for objective: {objective}, "
                  f"acquisition function: {acquisition_function}, dimension: {dim}")
            continue
        
        hyperparameter_data = valid_methods
        
        # Generate plots based on hyperparameter_type
        if hyperparameter_type in ["lengthscales", "all"]:
            generate_lengthscale_plot(
                hyperparameter_data, objective, acquisition_function, dim, 
                save_plots, output_dir, method_alpha
            )
        
        if hyperparameter_type in ["signal_variance", "all"]:
            generate_signal_variance_plot(
                hyperparameter_data, objective, acquisition_function, dim,
                save_plots, output_dir, method_alpha
            )
        
        if hyperparameter_type in ["noise_variance", "all"]:
            generate_noise_variance_plot(
                hyperparameter_data, objective, acquisition_function, dim,
                save_plots, output_dir, method_alpha
            )


def extract_hyperparameter_data(dim_results, problem_dim):
    hyperparameter_data = defaultdict(lambda: {
        'lengthscales': [],
        'signal_variances': [],
        'noise_variances': []
    })
    
    for result_dir, results in dim_results.items():
        dir_info = extract_info_from_result_dir(result_dir)
        method_name = dir_info['method']
        
        # Check if hyperparameter data is available
        has_lengthscales = "lengthscales" in results and results["lengthscales"] and len(results["lengthscales"]) > 0
        has_signal_variances = "signal_variances" in results and results["signal_variances"] and len(results["signal_variances"]) > 0
        has_noise_variances = "noise_variances" in results and results["noise_variances"] and len(results["noise_variances"]) > 0
        
        if not (has_lengthscales or has_signal_variances or has_noise_variances):
            print(f"Warning: No hyperparameter data found for {method_name} in {result_dir}")
            continue
        
        # Process lengthscales (may be multi-dimensional)
        if has_lengthscales:
            lengthscales_list = results["lengthscales"]
            # Convert each iteration's lengthscales to numpy array
            lengthscales_per_iteration = []
            
            # First pass: determine the maximum dimension across all iterations
            max_dim = 1
            for ls in lengthscales_list:
                if ls is not None:
                    ls_array = convert_to_numpy(ls)
                    if ls_array.ndim == 0:
                        max_dim = max(max_dim, 1)
                    else:
                        max_dim = max(max_dim, ls_array.shape[0] if ls_array.ndim == 1 else ls_array.size)
            
            # Use max_dim instead of problem_dim for consistent shape
            effective_dim = max(max_dim, problem_dim)
            
            # Second pass: convert to consistent shape
            for ls in lengthscales_list:
                if ls is not None:
                    ls_array = convert_to_numpy(ls)
                    if ls_array.ndim == 0:  # Scalar lengthscale
                        ls_array = np.array([ls_array])
                    elif ls_array.ndim == 2:  # 2D array with batch dimension (e.g., shape (1, 2))
                        ls_array = ls_array.flatten()  # Convert (1, 2) -> (2,)
                    
                    # Ensure consistent dimensionality
                    if len(ls_array) < effective_dim:
                        # Pad with the last value if dimensions don't match
                        last_val = ls_array[-1] if len(ls_array) > 0 else 1.0
                        ls_array = np.pad(ls_array, (0, effective_dim - len(ls_array)), 
                                        mode='constant', constant_values=last_val)
                    elif len(ls_array) > effective_dim:
                        # Truncate if too many dimensions
                        ls_array = ls_array[:effective_dim]
                    
                    lengthscales_per_iteration.append(ls_array)
                else:
                    # Handle None values with NaN array of appropriate shape
                    lengthscales_per_iteration.append(np.full(effective_dim, np.nan))
            
            if lengthscales_per_iteration:
                # Convert to numpy array - should now have consistent shapes
                try:
                    lengthscales_array = np.array(lengthscales_per_iteration)
                    hyperparameter_data[method_name]['lengthscales'].append(lengthscales_array)
                except ValueError:
                    # Store as list if array conversion fails
                    hyperparameter_data[method_name]['lengthscales'].append(lengthscales_per_iteration)
        
        # Process signal variances
        if has_signal_variances:
            signal_variances_list = results["signal_variances"]
            signal_variances_array = []
            for sv in signal_variances_list:
                if sv is not None:
                    sv_value = convert_to_numpy(sv)
                    if sv_value.ndim > 0:
                        sv_value = sv_value.item()
                    signal_variances_array.append(sv_value)
                else:
                    signal_variances_array.append(np.nan)
            
            if signal_variances_array:
                hyperparameter_data[method_name]['signal_variances'].append(
                    np.array(signal_variances_array)
                )
        
        # Process noise variances
        if has_noise_variances:
            noise_variances_list = results["noise_variances"]
            noise_variances_array = []
            for nv in noise_variances_list:
                if nv is not None:
                    nv_value = convert_to_numpy(nv)
                    if nv_value.ndim > 0:
                        nv_value = nv_value.item()
                    noise_variances_array.append(nv_value)
                else:
                    noise_variances_array.append(np.nan)
            
            if noise_variances_array:
                hyperparameter_data[method_name]['noise_variances'].append(
                    np.array(noise_variances_array)
                )
    
    return hyperparameter_data


def generate_lengthscale_plot(hyperparameter_data, objective, acquisition_function, dim, 
                            save_plots, output_dir, method_alpha):
    # Get method names and sort them
    method_names = [name for name in hyperparameter_data.keys() 
                   if hyperparameter_data[name]['lengthscales']]
    
    if not method_names:
        print("No lengthscale data available for plotting")
        return
    
    sorted_method_names = get_sorted_method_names(method_names)
    color_map = get_method_color_map(sorted_method_names)
    acq_display_names = get_acquisition_function_display_names()
    
    # Determine problem dimension from data
    sample_lengthscales = hyperparameter_data[method_names[0]]['lengthscales'][0]
    if isinstance(sample_lengthscales, np.ndarray):
        if sample_lengthscales.ndim > 1:
            problem_dim = sample_lengthscales.shape[1]
        else:
            problem_dim = 1
    else:
        # Handle list case
        if len(sample_lengthscales) > 0:
            first_entry = sample_lengthscales[0]
            if isinstance(first_entry, np.ndarray) and first_entry.ndim > 0:
                problem_dim = len(first_entry)
            else:
                problem_dim = 1
        else:
            problem_dim = 1
    
    # Create subplots for each dimension
    if problem_dim == 1:
        fig, ax = plt.subplots(figsize=(12, 7))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, problem_dim, figsize=(12 * problem_dim, 7))
        if problem_dim == 1:
            axes = [axes]
    
    for dim_idx in range(problem_dim):
        ax = axes[dim_idx] if problem_dim > 1 else axes[0]
        
        for method_name in sorted_method_names:
            if not hyperparameter_data[method_name]['lengthscales']:
                continue
                
            color = color_map[method_name]
            display_name = get_method_display_name(method_name)
            
            # Get alpha values
            line_alpha = method_alpha.get(method_name, 0.8) if method_alpha else 0.8
            band_alpha = method_alpha.get(method_name, 0.3) if method_alpha else 0.3
            
            # Process all seeds for this method
            all_lengthscales = []
            for seed_data in hyperparameter_data[method_name]['lengthscales']:
                if isinstance(seed_data, np.ndarray):
                    # Handle numpy array case
                    if problem_dim == 1:
                        lengthscales_dim = seed_data[:, 0] if seed_data.ndim > 1 else seed_data
                    else:
                        if seed_data.ndim > 1 and seed_data.shape[1] > dim_idx:
                            lengthscales_dim = seed_data[:, dim_idx]
                        else:
                            # Handle case where we don't have enough dimensions
                            lengthscales_dim = np.full(seed_data.shape[0], np.nan)
                else:
                    # Handle list case (fallback from failed array conversion)
                    lengthscales_dim = []
                    for ls_entry in seed_data:
                        if isinstance(ls_entry, np.ndarray):
                            # Handle numpy array entries
                            if ls_entry.ndim == 0:  # Scalar
                                if problem_dim == 1:
                                    lengthscales_dim.append(float(ls_entry))
                                else:
                                    lengthscales_dim.append(float(ls_entry) if dim_idx == 0 else np.nan)
                            elif ls_entry.ndim == 1:  # 1D array
                                if len(ls_entry) > dim_idx:
                                    lengthscales_dim.append(float(ls_entry[dim_idx]))
                                else:
                                    lengthscales_dim.append(np.nan)
                            elif ls_entry.ndim == 2:  # 2D array (batch dimension)
                                # Handle case like torch.tensor([[43.96, 41.85]]) -> shape (1, 2)
                                if ls_entry.shape[0] > 0 and ls_entry.shape[1] > dim_idx:
                                    lengthscales_dim.append(float(ls_entry[0, dim_idx]))
                                else:
                                    lengthscales_dim.append(np.nan)
                            else:
                                lengthscales_dim.append(np.nan)
                        elif isinstance(ls_entry, (int, float)):
                            # Handle scalar entries
                            if problem_dim == 1:
                                lengthscales_dim.append(float(ls_entry))
                            else:
                                lengthscales_dim.append(float(ls_entry) if dim_idx == 0 else np.nan)
                        else:
                            # Handle other cases (None, etc.)
                            lengthscales_dim.append(np.nan)
                    
                    # Convert to numpy array - should now be all floats
                    try:
                        lengthscales_dim = np.array(lengthscales_dim, dtype=float)
                    except (ValueError, TypeError):
                        # Fallback to NaN array if conversion fails
                        lengthscales_dim = np.full(len(lengthscales_dim), np.nan)
                
                all_lengthscales.append(lengthscales_dim)
            
            # Pad to same length and compute statistics
            max_len = max(len(ls) for ls in all_lengthscales)
            padded_lengthscales = []
            
            for ls in all_lengthscales:
                if len(ls) < max_len:
                    # Forward fill the last valid value
                    last_valid = ls[~np.isnan(ls)][-1] if len(ls[~np.isnan(ls)]) > 0 else np.nan
                    padded = np.pad(ls, (0, max_len - len(ls)), mode='constant', constant_values=last_valid)
                else:
                    padded = ls
                padded_lengthscales.append(padded)
            
            lengthscales_array = np.array(padded_lengthscales)
            
            # Compute statistics
            median_lengthscales = np.nanmedian(lengthscales_array, axis=0)
            lower_bound = np.nanpercentile(lengthscales_array, 25, axis=0)
            upper_bound = np.nanpercentile(lengthscales_array, 75, axis=0)
            
            # Plot
            x_values = np.arange(len(median_lengthscales))
            
            ax.plot(
                x_values, median_lengthscales,
                color=color, linewidth=2.5, alpha=line_alpha,
                label=f"{display_name} (median)", marker='o', markersize=4,
                markevery=max(1, len(x_values) // 20)
            )
            
            ax.fill_between(
                x_values, lower_bound, upper_bound,
                color=color, alpha=band_alpha
            )
        
        # Style subplot
        ax.set_xlabel("Iteration", fontweight="bold")
        if problem_dim == 1:
            ax.set_ylabel("Lengthscale", fontweight="bold")
        else:
            ax.set_ylabel(f"Lengthscale (Dim {dim_idx + 1})", fontweight="bold")
        
        ax.grid(True, linestyle="--", alpha=0.6)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
    
    # Main title
    objective_display_name = get_objective_display_name(objective)
    acq_display_name = acq_display_names.get(acquisition_function, acquisition_function)
    n_seeds = len(hyperparameter_data[sorted_method_names[0]]['lengthscales']) if sorted_method_names else 0
    
    fig.suptitle(f"GP Lengthscale Evolution - {objective_display_name} with {acq_display_name} ({n_seeds} seeds)", 
                fontweight="bold", fontsize=16)
    
    # Create legend only on the first subplot
    if sorted_method_names:
        ax = axes[0] if problem_dim > 1 else axes[0]
        handles, labels = ax.get_legend_handles_labels()
        
        # Extract method names from labels
        method_labels = {}
        for h, l in zip(handles, labels):
            if "(median)" in l:
                method_name = l.replace(" (median)", "")
                for method_key in sorted_method_names:
                    if get_method_display_name(method_key) == method_name:
                        method_labels[method_key] = (h, method_name)
                        break
        
        # Create grouped legend
        method_groups = group_methods_by_type(list(method_labels.keys()))
        max_methods_per_group = max(len(methods) for methods in method_groups.values()) if method_groups else 0
        
        create_grouped_legend(
            fig, ax, method_labels, sorted_method_names, max_methods_per_group,
            "Lines show median lengthscale values with IQR shading"
        )
    
    fig.tight_layout(rect=[0, 0.25, 1, 0.95])
    
    if save_plots:
        acq_name = acq_display_name.replace(' ', '_')
        filename = f"lengthscales_{objective}_{acq_name}.png"
        save_plot_figure(fig, output_dir, filename)
        
    plt.show()


def generate_signal_variance_plot(hyperparameter_data, objective, acquisition_function, dim,
                                save_plots, output_dir, method_alpha):
    # Get method names and sort them
    method_names = [name for name in hyperparameter_data.keys() 
                   if hyperparameter_data[name]['signal_variances']]
    
    if not method_names:
        print("No signal variance data available for plotting")
        return
    
    sorted_method_names = get_sorted_method_names(method_names)
    color_map = get_method_color_map(sorted_method_names)
    acq_display_names = get_acquisition_function_display_names()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for method_name in sorted_method_names:
        if not hyperparameter_data[method_name]['signal_variances']:
            continue
            
        color = color_map[method_name]
        display_name = get_method_display_name(method_name)
        
        # Get alpha values
        line_alpha = method_alpha.get(method_name, 0.8) if method_alpha else 0.8
        band_alpha = method_alpha.get(method_name, 0.3) if method_alpha else 0.3
        
        # Process all seeds for this method
        all_signal_variances = hyperparameter_data[method_name]['signal_variances']
        
        # Pad to same length and compute statistics
        max_len = max(len(sv) for sv in all_signal_variances)
        padded_signal_variances = []
        
        for sv in all_signal_variances:
            if len(sv) < max_len:
                # Forward fill the last valid value
                last_valid = sv[~np.isnan(sv)][-1] if len(sv[~np.isnan(sv)]) > 0 else np.nan
                padded = np.pad(sv, (0, max_len - len(sv)), mode='constant', constant_values=last_valid)
            else:
                padded = sv
            padded_signal_variances.append(padded)
        
        signal_variances_array = np.array(padded_signal_variances)
        
        # Compute statistics
        median_signal_variances = np.nanmedian(signal_variances_array, axis=0)
        lower_bound = np.nanpercentile(signal_variances_array, 25, axis=0)
        upper_bound = np.nanpercentile(signal_variances_array, 75, axis=0)
        
        # Plot
        x_values = np.arange(len(median_signal_variances))
        
        ax.plot(
            x_values, median_signal_variances,
            color=color, linewidth=2.5, alpha=line_alpha,
            label=f"{display_name} (median)", marker='o', markersize=4,
            markevery=max(1, len(x_values) // 20)
        )
        
        ax.fill_between(
            x_values, lower_bound, upper_bound,
            color=color, alpha=band_alpha
        )
    
    # Style plot
    ax.set_xlabel("Iteration", fontweight="bold")
    ax.set_ylabel("Signal Variance (Outputscale)", fontweight="bold")
    ax.set_yscale("log")  # Log scale often better for variance values
    
    # Add title
    objective_display_name = get_objective_display_name(objective)
    acq_display_name = acq_display_names.get(acquisition_function, acquisition_function)
    n_seeds = len(hyperparameter_data[sorted_method_names[0]]['signal_variances']) if sorted_method_names else 0
    ax.set_title(f"GP Signal Variance Evolution - {objective_display_name} with {acq_display_name} ({n_seeds} seeds)", 
                fontweight="bold", pad=12)
    
    ax.grid(True, linestyle="--", alpha=0.6)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
    
    # Create legend
    if sorted_method_names:
        handles, labels = ax.get_legend_handles_labels()
        
        # Extract method names from labels
        method_labels = {}
        for h, l in zip(handles, labels):
            if "(median)" in l:
                method_name = l.replace(" (median)", "")
                for method_key in sorted_method_names:
                    if get_method_display_name(method_key) == method_name:
                        method_labels[method_key] = (h, method_name)
                        break
        
        # Create grouped legend
        method_groups = group_methods_by_type(list(method_labels.keys()))
        max_methods_per_group = max(len(methods) for methods in method_groups.values()) if method_groups else 0
        
        create_grouped_legend(
            fig, ax, method_labels, sorted_method_names, max_methods_per_group,
            "Lines show median signal variance values with IQR shading"
        )
    
    fig.tight_layout(rect=[0, 0.25, 1, 1.0])
    
    if save_plots:
        acq_name = acq_display_name.replace(' ', '_')
        filename = f"signal_variance_{objective}_{acq_name}.png"
        save_plot_figure(fig, output_dir, filename)
        
    plt.show()


def generate_noise_variance_plot(hyperparameter_data, objective, acquisition_function, dim,
                               save_plots, output_dir, method_alpha):
    # Get method names and sort them
    method_names = [name for name in hyperparameter_data.keys() 
                   if hyperparameter_data[name]['noise_variances']]
    
    if not method_names:
        print("No noise variance data available for plotting")
        return
    
    sorted_method_names = get_sorted_method_names(method_names)
    color_map = get_method_color_map(sorted_method_names)
    acq_display_names = get_acquisition_function_display_names()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for method_name in sorted_method_names:
        if not hyperparameter_data[method_name]['noise_variances']:
            continue
            
        color = color_map[method_name]
        display_name = get_method_display_name(method_name)
        
        # Get alpha values
        line_alpha = method_alpha.get(method_name, 0.8) if method_alpha else 0.8
        band_alpha = method_alpha.get(method_name, 0.3) if method_alpha else 0.3
        
        # Process all seeds for this method
        all_noise_variances = hyperparameter_data[method_name]['noise_variances']
        
        # Pad to same length and compute statistics
        max_len = max(len(nv) for nv in all_noise_variances)
        padded_noise_variances = []
        
        for nv in all_noise_variances:
            if len(nv) < max_len:
                # Forward fill the last valid value
                last_valid = nv[~np.isnan(nv)][-1] if len(nv[~np.isnan(nv)]) > 0 else np.nan
                padded = np.pad(nv, (0, max_len - len(nv)), mode='constant', constant_values=last_valid)
            else:
                padded = nv
            padded_noise_variances.append(padded)
        
        noise_variances_array = np.array(padded_noise_variances)
        
        # Compute statistics
        median_noise_variances = np.nanmedian(noise_variances_array, axis=0)
        lower_bound = np.nanpercentile(noise_variances_array, 25, axis=0)
        upper_bound = np.nanpercentile(noise_variances_array, 75, axis=0)
        
        # Plot
        x_values = np.arange(len(median_noise_variances))
        
        ax.plot(
            x_values, median_noise_variances,
            color=color, linewidth=2.5, alpha=line_alpha,
            label=f"{display_name} (median)", marker='o', markersize=4,
            markevery=max(1, len(x_values) // 20)
        )
        
        ax.fill_between(
            x_values, lower_bound, upper_bound,
            color=color, alpha=band_alpha
        )
    
    # Style plot
    ax.set_xlabel("Iteration", fontweight="bold")
    ax.set_ylabel("Noise Variance", fontweight="bold")
    ax.set_yscale("log")  # Log scale often better for variance values
    
    # Add title
    objective_display_name = get_objective_display_name(objective)
    acq_display_name = acq_display_names.get(acquisition_function, acquisition_function)
    n_seeds = len(hyperparameter_data[sorted_method_names[0]]['noise_variances']) if sorted_method_names else 0
    ax.set_title(f"GP Noise Variance Evolution - {objective_display_name} with {acq_display_name} ({n_seeds} seeds)", 
                fontweight="bold", pad=12)
    
    ax.grid(True, linestyle="--", alpha=0.6)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
    
    # Create legend
    if sorted_method_names:
        handles, labels = ax.get_legend_handles_labels()
        
        # Extract method names from labels
        method_labels = {}
        for h, l in zip(handles, labels):
            if "(median)" in l:
                method_name = l.replace(" (median)", "")
                for method_key in sorted_method_names:
                    if get_method_display_name(method_key) == method_name:
                        method_labels[method_key] = (h, method_name)
                        break
        
        # Create grouped legend
        method_groups = group_methods_by_type(list(method_labels.keys()))
        max_methods_per_group = max(len(methods) for methods in method_groups.values()) if method_groups else 0
        
        create_grouped_legend(
            fig, ax, method_labels, sorted_method_names, max_methods_per_group,
            "Lines show median noise variance values with IQR shading"
        )
    
    fig.tight_layout(rect=[0, 0.25, 1, 1.0])
    
    if save_plots:
        acq_name = acq_display_name.replace(' ', '_')
        filename = f"noise_variance_{objective}_{acq_name}.png"
        save_plot_figure(fig, output_dir, filename)
        
    plt.show()


##################################################################
# Configuration
##################################################################

if __name__ == "__main__":

    project_root = setup_project_path()

    PLOT_STYLE = {
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 16,
        "figure.figsize": (13, 20),
        "figure.dpi": 80,
    }

    main(
        objective=None,     # Set to None to auto-detect all objectives
        dim=None,           # Set to None to auto-detect all dimensions
        seed=None,      # Set to None to auto-detect all seeds
        methods=[
            "bo_plain", 
            "boot_standardize", "bopt_standardize", 
            "boot_log", "bopt_log",
            "bopt_bilog",
            # "boni_plain", "boni_plainnonoise",
            # "boni_standardize", "boni_standardizenonoise",
            # "boni_standardizenonoiserefit",
            # "boni_standardizeones", "boni_standardizezeros",
            # "boni_ilsstandardize", "boni_ilsstandardizenonoise",
            # "boni_ilsstandardiznonoiserefit",
            # "boni_bsnoise",
            # "boni_bsnonoise",  
            # "boni_standardizegradient", "boni_standardizegradientbinary",
            # "boni_tr", "boni_trbs",
            # "turbo_plain", "turbo_standardize",
            # "turboni_standardize", "turboni_tr", "turboni_trbs",
        ],
        acquisition_function=None, # Set to None to auto-detect all acquisition functions
        save_plots=True,
        output_dir="figures/xseed_gp_hyperparameters",
        sweep="final",
        hyperparameter_type="all",  # "lengthscales", "signal_variance", "noise_variance", or "all"
        method_alpha={  # Control transparency/blending of specific methods
            # "bo_plain": 0.5,
            # "boot_standardize": 0.3,
        }
    )
