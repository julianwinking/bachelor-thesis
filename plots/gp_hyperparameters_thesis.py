"""
Comprehensive GP hyperparameter visualization module for thesis plots with multiple configurations.
Chart types: individual hyperparameter trajectories, overview grids, and comparative evolution plots.
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
    convert_to_numpy,
    load_and_filter_results,
    get_thesis_figure_size,
    create_thesis_legend,
    setup_thesis_style,
    sort_objectives_by_name_and_dimension,
    create_thesis_output_dir,
    save_thesis_plot,
)


def main(objective=None, dim=None, seed=0, methods=None, acquisition_function=None, save_plots=False, output_dir=None, sweep="debug", method_alpha=None, hyperparameter_type="all", chart_type="individual", overview_acquisition_functions=None, method_column_order=None, sobol_offset=10,
):
    # Base path for results
    base_path = os.path.join(project_root, "results", sweep)
    
    if not os.path.exists(base_path):
        print(f"Results directory not found at {base_path}")
        return

    # Create output directory if saving plots (thesis directory)
    if save_plots:
        output_dir = create_thesis_output_dir(output_dir, "thesis/gp_hyperparameters")

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
    if chart_type == "overview":
        # Handle overview charts
        if overview_acquisition_functions is None or len(overview_acquisition_functions) != 2:
            print("Error: overview chart type requires exactly two acquisition functions in overview_acquisition_functions parameter")
            print("Example: overview_acquisition_functions=['UpperConfidenceBound', 'LogExpectedImprovement']")
            return
            
        objectives_to_plot = [objective] if objective else all_objectives
        
        # Sort objectives for consistent ordering (by function name, then dimension)
        objectives_to_plot = sort_objectives_by_name_and_dimension(objectives_to_plot)
        
        # Generate overview charts for each hyperparameter type
        if hyperparameter_type == "all":
            hyperparameter_types = ["lengthscales", "signal_variance", "noise_variance"]
        else:
            hyperparameter_types = [hyperparameter_type]
            
        for hp_type in hyperparameter_types:
            print(f"\nGenerating overview chart for hyperparameter type: {hp_type}")
            generate_overview_chart(
                all_results=all_results,
                objectives=objectives_to_plot,
                acquisition_functions=overview_acquisition_functions,
                hyperparameter_type=hp_type,
                save_plots=save_plots,
                output_dir=output_dir,
                methods=methods,
                method_alpha=method_alpha,
                method_column_order=method_column_order
            )
    elif chart_type == "comparison":
        # Handle comparison charts (3 subplots: lengthscales, signal variance, noise variance)
        if objective is None:
            print("Error: comparison chart type requires a specific objective function")
            print("Please set objective parameter to a specific objective function")
            return
            
        if acquisition_function is None:
            print("Error: comparison chart type requires a specific acquisition function")
            print("Please set acquisition_function parameter to a specific acquisition function")
            return
            
        print(f"\nGenerating comparison chart for objective: {objective}, acquisition function: {acquisition_function}")
        generate_comparison_chart(
            all_results=all_results,
            objective=objective,
            acquisition_function=acquisition_function,
            save_plots=save_plots,
            output_dir=output_dir,
            methods=methods,
            method_alpha=method_alpha,
            sobol_offset=sobol_offset
        )
    else:
        # Original individual chart logic
        objectives_to_plot = [objective] if objective else all_objectives
        # Sort objectives for consistent ordering (by function name, then dimension)
        objectives_to_plot = sort_objectives_by_name_and_dimension(objectives_to_plot)
        acq_funcs_to_plot = [acquisition_function] if acquisition_function else all_acquisition_functions
        
        for obj in objectives_to_plot:
            for acq_func in acq_funcs_to_plot:
                print(f"\nGenerating thesis-style GP hyperparameter plots for objective: {obj}, acquisition function: {acq_func}")
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
        has_lengthscales = data['lengthscales'] and len(data['lengthscales']) > 0
        has_signal_variances = data['signal_variances'] and len(data['signal_variances']) > 0
        has_noise_variances = data['noise_variances'] and len(data['noise_variances']) > 0
        
        has_valid_data = has_lengthscales or has_signal_variances or has_noise_variances
        
        # Debug print for noise variance data
        if has_noise_variances:
            print(f"DEBUG: {method_name} has noise variance data with {len(data['noise_variances'])} seeds")
        else:
            print(f"DEBUG: {method_name} has NO noise variance data")
            
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
    """Extract and organize hyperparameter data from results."""
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
    """Generate thesis-style lengthscale evolution plot."""
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
    
    # Get thesis figure size
    fig_width, fig_height = get_thesis_figure_size()
    
    # Create subplots for each dimension with thesis sizing
    if problem_dim == 1:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, problem_dim, figsize=(fig_width * problem_dim, fig_height))
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
                label=f"{display_name} (median)",
                markevery=max(1, len(x_values) // 20)
            )
            
            ax.fill_between(
                x_values, lower_bound, upper_bound,
                color=color, alpha=band_alpha
            )
        
        # Style subplot
        ax.set_xlabel(r'Iteration')
        if problem_dim == 1:
            ax.set_ylabel(r'Lengthscale')
        else:
            ax.set_ylabel(fr'Lengthscale (Dim {dim_idx + 1})')
        
        # Show all spines for full outline
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
    
    # Create thesis-style legend
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
        
        # Create thesis legend using custom function
        create_thesis_legend(fig, ax, method_labels, sorted_method_names)
    
    plt.tight_layout(rect=[0, 0.1, 1, 1.0])
    
    if save_plots:
        acq_display_name = acq_display_names.get(acquisition_function, acquisition_function)
        acq_name = acq_display_name.replace(' ', '_')
        filename = f"lengthscales_{objective}_{acq_name}_thesis.pdf"
        save_thesis_plot(fig, output_dir, filename)
        
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
    
    # Get thesis figure size
    fig_width, fig_height = get_thesis_figure_size()
    
    # Create plot with thesis sizing
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
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
            label=f"{display_name} (median)"
        )
        
        ax.fill_between(
            x_values, lower_bound, upper_bound,
            color=color, alpha=band_alpha
        )
    
    # Style plot
    ax.set_xlabel(r'Iteration')
    ax.set_ylabel(r'$\sigma_f^2$')
    ax.set_yscale("log")
    
    # Show all spines for full outline
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
    
    # Create thesis-style legend
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
        
        # Create thesis legend using custom function
        create_thesis_legend(fig, ax, method_labels, sorted_method_names)
    
    plt.tight_layout(rect=[0, 0.25, 1, 1.0])
    
    if save_plots:
        acq_display_name = acq_display_names.get(acquisition_function, acquisition_function)
        acq_name = acq_display_name.replace(' ', '_')
        filename = f"signal_variance_{objective}_{acq_name}_thesis.pdf"
        save_thesis_plot(fig, output_dir, filename)
        
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
    
    # Get thesis figure size
    fig_width, fig_height = get_thesis_figure_size()
    
    # Create plot with thesis sizing
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
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
        
        # Debug: Print first few values for this method
        if len(all_noise_variances) > 0:
            first_seed = all_noise_variances[0]
            print(f"DEBUG: {method_name} noise variance - first seed has {len(first_seed)} iterations")
            print(f"DEBUG: {method_name} noise variance - first 5 values: {first_seed[:5]}")
        
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
            label=f"{display_name} (median)"
        )
        
        ax.fill_between(
            x_values, lower_bound, upper_bound,
            color=color, alpha=band_alpha
        )
    
    # Style plot
    ax.set_xlabel(r'Iteration')
    ax.set_ylabel(r'$\sigma_n^2$')
    ax.set_yscale("log")
    
    # Show all spines for full outline
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
    
    # Create thesis-style legend
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
        
        # Create thesis legend using custom function
        create_thesis_legend(fig, ax, method_labels, sorted_method_names)
    
    plt.tight_layout(rect=[0, 0.25, 1, 1.0])
    
    if save_plots:
        acq_display_name = acq_display_names.get(acquisition_function, acquisition_function)
        acq_name = acq_display_name.replace(' ', '_')
        filename = f"noise_variance_{objective}_{acq_name}_thesis.pdf"
        save_thesis_plot(fig, output_dir, filename)
        
    plt.show()


def generate_overview_chart(all_results, objectives, acquisition_functions, hyperparameter_type, 
                          save_plots=False, output_dir=None, methods=None, method_alpha=None,
                          method_column_order=None):

    if len(acquisition_functions) != 2:
        print("Error: Overview chart requires exactly 2 acquisition functions")
        return
        
    if len(objectives) == 0:
        print("Error: No objectives found for overview chart")
        return
    
    # Get thesis figure size and scale for overview
    base_fig_width, base_fig_height = get_thesis_figure_size()
    
    # Calculate figure size for overview (2 columns, n rows)
    n_rows = len(objectives)
    n_cols = 2
    
    # Scale figure size appropriately
    fig_width = base_fig_width * 1.8
    fig_height = base_fig_height * n_rows * 0.5
    
    # Create subplot grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    
    # Handle case where we have only one row
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Store all method labels for unified legend
    all_method_labels = {}
    
    # Process each objective (row) and acquisition function (column)
    for row_idx, objective in enumerate(objectives):
        for col_idx, acq_func in enumerate(acquisition_functions):
            ax = axes[row_idx, col_idx]
            
            print(f"Processing: {objective} with {acq_func}")
            
            # Get data for this objective and acquisition function
            hyperparameter_data = get_overview_data(
                all_results, objective, acq_func, methods
            )
            
            if not hyperparameter_data:
                # If no data, create empty plot with label
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_xlabel(r'Iteration')
                set_overview_ylabel(ax, hyperparameter_type)
                continue
            
            # Plot based on hyperparameter type
            method_labels = plot_overview_subplot(
                ax, hyperparameter_data, hyperparameter_type, method_alpha
            )
            
            # Collect method labels for unified legend
            all_method_labels.update(method_labels)
            
            # Set labels and title
            # Only show x-axis label on bottom row
            if row_idx == n_rows - 1:
                ax.set_xlabel(r'Iteration')
            else:
                ax.set_xlabel('')
            
            # Only show y-axis label on left column
            if col_idx == 0:
                set_overview_ylabel(ax, hyperparameter_type)
            else:
                ax.set_ylabel('')
            
            # Add title for top row only
            if row_idx == 0:
                acq_display_names = get_acquisition_function_display_names()
                acq_display_name = acq_display_names.get(acq_func, acq_func)
                ax.set_title(acq_display_name)
            
            # Add objective label on left column only
            if col_idx == 0:
                objective_display_name = get_objective_display_name(objective)
                ax.text(-0.25, 0.5, objective_display_name, rotation=90, 
                       ha='center', va='center', transform=ax.transAxes, fontweight='bold')
            
            # Show all spines for full outline
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
    
    # Create unified legend below all charts
    if all_method_labels:
        # Get sorted method names for consistent ordering
        sorted_method_names = get_sorted_method_names(list(all_method_labels.keys()))
        
        # Filter method labels to only include sorted methods
        filtered_method_labels = {
            method: all_method_labels[method] 
            for method in sorted_method_names 
            if method in all_method_labels
        }
        
        # Temporarily override legend fontsize for overview charts (different from thesis style)
        original_fontsize = plt.rcParams.get('legend.fontsize', 8)
        plt.rcParams['legend.fontsize'] = 10
        
        # Custom legend ordering if provided
        if method_column_order:
            # Use custom column ordering for overview charts
            from matplotlib.lines import Line2D
            
            # Get list of columns in order
            column_names = list(method_column_order.keys())
            num_columns = len(column_names)
            
            if num_columns > 0:
                # Create grid layout: organize by columns
                all_handles = []
                all_labels = []
                
                # Find the maximum number of methods in any column
                max_methods_per_column = max(len(methods) for methods in method_column_order.values())
                
                # Fill column by column
                for col, column_name in enumerate(column_names):
                    methods_in_column = method_column_order[column_name]
                    
                    # Add methods for this column
                    for method in methods_in_column:
                        if method in filtered_method_labels:
                            handle, display_name = filtered_method_labels[method]
                            all_handles.append(handle)
                            all_labels.append(display_name)
                    
                    # Add padding to make all columns the same length
                    methods_in_this_column = len(methods_in_column)
                    padding_needed = max_methods_per_column - methods_in_this_column
                    for _ in range(padding_needed):
                        all_handles.append(Line2D([0], [0], alpha=0))  # Invisible handle
                        all_labels.append("")  # Empty label
                
                # Create legend with custom handles and labels
                legend = fig.legend(all_handles, all_labels, 
                                  loc='upper center', bbox_to_anchor=(0.5, -0.02),
                                  ncol=num_columns, frameon=False, fontsize=10)
                fig.add_artist(legend)
        else:
            # Fallback to default legend creation if no custom order
            create_thesis_legend(fig, axes[0, 0], filtered_method_labels, sorted_method_names)
        
        # Restore original fontsize
        plt.rcParams['legend.fontsize'] = original_fontsize
    
    # Adjust layout with reduced margins, closer legend, and reduced vertical spacing between rows
    plt.subplots_adjust(
        left=0.1,      # left margin
        right=0.95,    # right margin
        bottom=0.01,   # bottom margin (for legend)
        top=0.95,      # top margin
        hspace=0.22     # height space between subplots (vertical spacing)
    )
    
    if save_plots:
        acq1_name = acquisition_functions[0].replace(' ', '_')
        acq2_name = acquisition_functions[1].replace(' ', '_')
        filename = f"overview_{hyperparameter_type}_{acq1_name}_vs_{acq2_name}_thesis.pdf"
        save_thesis_plot(fig, output_dir, filename)
        
    plt.show()


def get_overview_data(all_results, objective, acquisition_function, methods):
    # Filter results for the specified objective and acquisition function
    filtered_results = filter_results_by_criteria(
        all_results, 
        objective=objective,
        acquisition_function=acquisition_function
    )
    
    if not filtered_results:
        return None
    
    # Group results by dimension (assuming single dimension for overview)
    dimension_groups = group_results_by_dimension(filtered_results)
    
    # Get data from the first (and likely only) dimension group
    if not dimension_groups:
        return None
        
    dim, dim_results = next(iter(dimension_groups.items()))
    
    # Extract hyperparameter data
    hyperparameter_data = extract_hyperparameter_data(dim_results, dim)
    
    if not hyperparameter_data:
        return None
    
    # Filter methods if specified
    if methods:
        filtered_data = {}
        for method_name, data in hyperparameter_data.items():
            if method_name in methods:
                filtered_data[method_name] = data
        hyperparameter_data = filtered_data
    
    return hyperparameter_data


def plot_overview_subplot(ax, hyperparameter_data, hyperparameter_type, method_alpha):
    method_labels = {}
    
    if hyperparameter_type == "lengthscales":
        method_labels = plot_overview_lengthscales(ax, hyperparameter_data, method_alpha)
    elif hyperparameter_type == "signal_variance":
        method_labels = plot_overview_variance(ax, hyperparameter_data, "signal_variances", method_alpha)
    elif hyperparameter_type == "noise_variance":
        method_labels = plot_overview_variance(ax, hyperparameter_data, "noise_variances", method_alpha)
    
    return method_labels


def plot_overview_lengthscales(ax, hyperparameter_data, method_alpha):
    method_labels = {}
    
    # Get method names and sort them
    method_names = [name for name in hyperparameter_data.keys() 
                   if hyperparameter_data[name]['lengthscales']]
    
    if not method_names:
        return method_labels
    
    sorted_method_names = get_sorted_method_names(method_names)
    color_map = get_method_color_map(sorted_method_names)
    
    for method_name in sorted_method_names:
        if not hyperparameter_data[method_name]['lengthscales']:
            continue
            
        color = color_map[method_name]
        display_name = get_method_display_name(method_name)
        
        # Get alpha values
        line_alpha = method_alpha.get(method_name, 0.8) if method_alpha else 0.8
        band_alpha = method_alpha.get(method_name, 0.3) if method_alpha else 0.3
        
        # Process all seeds and compute mean across dimensions
        all_mean_lengthscales = []
        
        for seed_data in hyperparameter_data[method_name]['lengthscales']:
            if isinstance(seed_data, np.ndarray):
                # For each iteration, compute mean across dimensions
                if seed_data.ndim > 1:
                    mean_lengthscales = np.nanmean(seed_data, axis=1)
                else:
                    mean_lengthscales = seed_data
            else:
                # Handle list case - compute mean for each iteration
                mean_lengthscales = []
                for ls_entry in seed_data:
                    if isinstance(ls_entry, np.ndarray):
                        if ls_entry.ndim == 0:
                            mean_lengthscales.append(float(ls_entry))
                        else:
                            mean_lengthscales.append(np.nanmean(ls_entry))
                    elif isinstance(ls_entry, (int, float)):
                        mean_lengthscales.append(float(ls_entry))
                    else:
                        mean_lengthscales.append(np.nan)
                mean_lengthscales = np.array(mean_lengthscales)
            
            all_mean_lengthscales.append(mean_lengthscales)
        
        # Pad to same length and compute statistics
        max_len = max(len(ls) for ls in all_mean_lengthscales)
        padded_lengthscales = []
        
        for ls in all_mean_lengthscales:
            if len(ls) < max_len:
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
        
        line = ax.plot(
            x_values, median_lengthscales,
            color=color, linewidth=2.0, alpha=line_alpha,
            label=display_name
        )[0]
        
        ax.fill_between(
            x_values, lower_bound, upper_bound,
            color=color, alpha=band_alpha
        )
        
        # Store for legend
        method_labels[method_name] = (line, display_name)
    
    # Set log scale for lengthscales plots
    ax.set_yscale("log")
    
    return method_labels


def plot_overview_variance(ax, hyperparameter_data, variance_type, method_alpha):
    method_labels = {}
    
    # Get method names and sort them
    method_names = [name for name in hyperparameter_data.keys() 
                   if hyperparameter_data[name][variance_type]]
    
    if not method_names:
        return method_labels
    
    sorted_method_names = get_sorted_method_names(method_names)
    color_map = get_method_color_map(sorted_method_names)
    
    for method_name in sorted_method_names:
        if not hyperparameter_data[method_name][variance_type]:
            continue
            
        color = color_map[method_name]
        display_name = get_method_display_name(method_name)
        
        # Get alpha values
        line_alpha = method_alpha.get(method_name, 0.8) if method_alpha else 0.8
        band_alpha = method_alpha.get(method_name, 0.3) if method_alpha else 0.3
        
        # Process all seeds for this method
        all_variances = hyperparameter_data[method_name][variance_type]
        
        # Pad to same length and compute statistics
        max_len = max(len(var) for var in all_variances)
        padded_variances = []
        
        for var in all_variances:
            if len(var) < max_len:
                last_valid = var[~np.isnan(var)][-1] if len(var[~np.isnan(var)]) > 0 else np.nan
                padded = np.pad(var, (0, max_len - len(var)), mode='constant', constant_values=last_valid)
            else:
                padded = var
            padded_variances.append(padded)
        
        variances_array = np.array(padded_variances)
        
        # Compute statistics
        median_variances = np.nanmedian(variances_array, axis=0)
        lower_bound = np.nanpercentile(variances_array, 25, axis=0)
        upper_bound = np.nanpercentile(variances_array, 75, axis=0)
        
        # Plot
        x_values = np.arange(len(median_variances))
        
        line = ax.plot(
            x_values, median_variances,
            color=color, linewidth=2.0, alpha=line_alpha,
            label=display_name
        )[0]
        
        ax.fill_between(
            x_values, lower_bound, upper_bound,
            color=color, alpha=band_alpha
        )
        
        # Store for legend
        method_labels[method_name] = (line, display_name)
    
    # Set log scale for variance plots
    ax.set_yscale("log")
    
    return method_labels


def set_overview_ylabel(ax, hyperparameter_type):
    if hyperparameter_type == "lengthscales":
        ax.set_ylabel(r'$\bar{\ell}$')
    elif hyperparameter_type == "signal_variance":
        ax.set_ylabel(r'$\sigma_f^2$')
    elif hyperparameter_type == "noise_variance":
        ax.set_ylabel(r'$\sigma_n^2$')


def generate_comparison_chart(all_results, objective, acquisition_function, save_plots=False, 
                            output_dir=None, methods=None, method_alpha=None, sobol_offset=10):
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
    
    # Use the first dimension group (assuming single dimension for comparison)
    if not dimension_groups:
        print(f"No dimension groups found for objective: {objective}, acquisition function: {acquisition_function}")
        return
        
    dim, dim_results = next(iter(dimension_groups.items()))
    
    # Extract hyperparameter data
    hyperparameter_data = extract_hyperparameter_data(dim_results, dim)
    
    if not hyperparameter_data:
        print(f"No valid hyperparameter data found for objective: {objective}, "
              f"acquisition function: {acquisition_function}, dimension: {dim}")
        return
    
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
        return
    
    hyperparameter_data = valid_methods
    
    # Get method names and sort them
    method_names = list(valid_methods.keys())
    sorted_method_names = get_sorted_method_names(method_names)
    color_map = get_method_color_map(sorted_method_names)
    acq_display_names = get_acquisition_function_display_names()
    
    # Get thesis figure size and scale for comparison (3 subplots in a row)
    base_fig_width, base_fig_height = get_thesis_figure_size()
    
    # Calculate figure size for comparison (3 columns, 1 row)
    fig_width = base_fig_width * 1.05  # Wider for 3 columns to fit letter height of thesis (experimented)
    fig_height = base_fig_height * 0.6 
    
    # Create subplot grid
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height))
    
    # Store all method labels for unified legend
    all_method_labels = {}
    
    # Plot lengthscales (mean across dimensions)
    print("Plotting lengthscales subplot...")
    method_labels_ls = plot_comparison_lengthscales(axes[0], hyperparameter_data, method_alpha, color_map, sorted_method_names, sobol_offset)
    all_method_labels.update(method_labels_ls)
    
    # Plot signal variance
    print("Plotting signal variance subplot...")
    method_labels_sv = plot_comparison_variance(axes[1], hyperparameter_data, "signal_variances", method_alpha, color_map, sorted_method_names, sobol_offset)
    all_method_labels.update(method_labels_sv)
    
    # Plot noise variance
    print("Plotting noise variance subplot...")
    method_labels_nv = plot_comparison_variance(axes[2], hyperparameter_data, "noise_variances", method_alpha, color_map, sorted_method_names, sobol_offset)
    all_method_labels.update(method_labels_nv)
    
    # Set labels and customize x-axis for each subplot
    for i, ax in enumerate(axes):
        # Set x-axis limits to account for Sobol offset (start at sobol_offset, end at 400)
        ax.set_xlim(sobol_offset, 500)
        
        # Set manual x-axis ticks: sobol_offset, middle, 400
        middle_tick = sobol_offset + (500 - sobol_offset) // 2
        ticks = [sobol_offset, middle_tick, 500]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        
        # Set labels
        ax.set_xlabel(r'Iteration')
        if i == 0:
            ax.set_ylabel(r'$\bar{\ell}$', labelpad=1)
            ax.set_yscale("log")
        elif i == 1:
            ax.set_ylabel(r'$\sigma_f^2$', labelpad=1)
            ax.set_yscale("log")
        elif i == 2:
            ax.set_ylabel(r'$\sigma_n^2$', labelpad=1)
            ax.set_yscale("log")
    
    # Create unified legend below all charts
    if all_method_labels:
        # Get sorted method names for consistent ordering
        sorted_method_names = get_sorted_method_names(list(all_method_labels.keys()))
        
        # Filter method labels to only include sorted methods
        filtered_method_labels = {
            method: all_method_labels[method] 
            for method in sorted_method_names 
            if method in all_method_labels
        }
        
        create_thesis_legend(fig, axes[0], filtered_method_labels, sorted_method_names)
    
    # Alternative approach with more control
    plt.subplots_adjust(
        left=0.08,      # left margin
        right=0.95,     # right margin  
        bottom=0.50,    # bottom margin (for legend)
        top=0.95,       # top margin
        wspace=0.52     # width space between subplots (horizontal spacing)
    )
    
    if save_plots:
        acq_display_name = acq_display_names.get(acquisition_function, acquisition_function)
        acq_name = acq_display_name.replace(' ', '_')
        filename = f"comparison_{objective}_{acq_name}_thesis.pdf"
        save_thesis_plot(fig, output_dir, filename)
        
    plt.show()


def plot_comparison_lengthscales(ax, hyperparameter_data, method_alpha, color_map, sorted_method_names, sobol_offset=10):
    method_labels = {}
    
    for method_name in sorted_method_names:
        if not hyperparameter_data[method_name]['lengthscales']:
            continue
            
        color = color_map[method_name]
        display_name = get_method_display_name(method_name)
        
        # Get alpha values
        line_alpha = method_alpha.get(method_name, 0.8) if method_alpha else 0.8
        band_alpha = method_alpha.get(method_name, 0.3) if method_alpha else 0.3
        
        # Process all seeds and compute mean across dimensions
        all_mean_lengthscales = []
        
        for seed_data in hyperparameter_data[method_name]['lengthscales']:
            if isinstance(seed_data, np.ndarray):
                # For each iteration, compute mean across dimensions
                if seed_data.ndim > 1:
                    mean_lengthscales = np.nanmean(seed_data, axis=1)
                else:
                    mean_lengthscales = seed_data
            else:
                # Handle list case - compute mean for each iteration
                mean_lengthscales = []
                for ls_entry in seed_data:
                    if isinstance(ls_entry, np.ndarray):
                        if ls_entry.ndim == 0:
                            mean_lengthscales.append(float(ls_entry))
                        else:
                            mean_lengthscales.append(np.nanmean(ls_entry))
                    elif isinstance(ls_entry, (int, float)):
                        mean_lengthscales.append(float(ls_entry))
                    else:
                        mean_lengthscales.append(np.nan)
                mean_lengthscales = np.array(mean_lengthscales)
            
            all_mean_lengthscales.append(mean_lengthscales)
        
        # Pad to same length and compute statistics
        max_len = max(len(ls) for ls in all_mean_lengthscales)
        padded_lengthscales = []
        
        for ls in all_mean_lengthscales:
            if len(ls) < max_len:
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
        
        # Plot with Sobol offset
        x_values = np.arange(sobol_offset, sobol_offset + len(median_lengthscales))
        
        line = ax.plot(
            x_values, median_lengthscales,
            color=color, linewidth=1.0, alpha=line_alpha,
            label=display_name
        )[0]
        
        ax.fill_between(
            x_values, lower_bound, upper_bound,
            color=color, alpha=band_alpha
        )
        
        # Store for legend
        method_labels[method_name] = (line, display_name)
    
    return method_labels


def plot_comparison_variance(ax, hyperparameter_data, variance_type, method_alpha, color_map, sorted_method_names, sobol_offset=10):
    method_labels = {}
    
    for method_name in sorted_method_names:
        if not hyperparameter_data[method_name][variance_type]:
            continue
            
        color = color_map[method_name]
        display_name = get_method_display_name(method_name)
        
        # Get alpha values
        line_alpha = method_alpha.get(method_name, 0.8) if method_alpha else 0.8
        band_alpha = method_alpha.get(method_name, 0.3) if method_alpha else 0.3
        
        # Process all seeds for this method
        all_variances = hyperparameter_data[method_name][variance_type]
        
        # Pad to same length and compute statistics
        max_len = max(len(var) for var in all_variances)
        padded_variances = []
        
        for var in all_variances:
            if len(var) < max_len:
                last_valid = var[~np.isnan(var)][-1] if len(var[~np.isnan(var)]) > 0 else np.nan
                padded = np.pad(var, (0, max_len - len(var)), mode='constant', constant_values=last_valid)
            else:
                padded = var
            padded_variances.append(padded)
        
        variances_array = np.array(padded_variances)
        
        # Compute statistics
        median_variances = np.nanmedian(variances_array, axis=0)
        lower_bound = np.nanpercentile(variances_array, 25, axis=0)
        upper_bound = np.nanpercentile(variances_array, 75, axis=0)
        
        # Plot with Sobol offset
        x_values = np.arange(sobol_offset, sobol_offset + len(median_variances))
        
        line = ax.plot(
            x_values, median_variances,
            color=color, linewidth=2.0, alpha=line_alpha,
            label=display_name
        )[0]
        
        ax.fill_between(
            x_values, lower_bound, upper_bound,
            color=color, alpha=band_alpha
        )
        
        # Store for legend
        method_labels[method_name] = (line, display_name)
    
    return method_labels

##################################################################
# Configuration
##################################################################

if __name__ == "__main__":
    project_root = setup_project_path()
    
    setup_thesis_style()

    main(
        objective=None,  # Set to specific objective for comparison chart
        dim=None,              # Set to None to auto-detect all dimensions
        seed=None,             # Set to None to auto-detect all seeds
        methods=[
            "bo_plain", 
            #"bopt_standardize", #"boot_standardize",
            #"boot_log", "bopt_log",
            #"bopt_bilog",
            #"boni_plainnonoise",#"boni_plain", 
            #"boni_standardizenonoise",# "boni_standardize", 
            #"boni_standardizenonoiserefit",
            # "boni_standardizeones", "boni_standardizezeros",
            #"boni_ilsstandardizenonoise",# "boni_ilsstandardize", 
            #"boni_ilsstandardiznonoiserefit",
            #"boni_bsnoise",
            #"boni_bsnonoise",  
            #"boni_standardizegradient", "boni_standardizegradientbinary",
            # "turbo_standardize", "turbo_plain",
            # "turboni_standardize", 
            # "turboni_tr", "turboni_trbinary",
            # "turboni_tradditivenorm",# "turboni_tradditive",
            # "turboni_trbsnorm",# "turboni_trbs",
        ],
        acquisition_function=None, # Set to specific acquisition function for comparison chart
        save_plots=True,
        output_dir="figures/thesis/hyperparameters/",
        sweep="final",
        hyperparameter_type="all",  # "lengthscales", "signal_variance", "noise_variance", or "all"
        chart_type="overview",  # "individual", "overview", or "comparison"
        overview_acquisition_functions=["UpperConfidenceBound", "LogExpectedImprovement"],  # For overview charts
        method_column_order={  # Control method ordering in columns for combined charts
            "Column1": ["turbo_plain", "turbo_standardize"],
            "Column2": ["turboni_standardize", "turboni_tr", "turboni_trbinary"],
            "Column3": ["turboni_tradditivenorm", "turboni_trbsnorm"],
        }, 
        method_alpha={  # Control transparency/blending of specific methods
            # Add method-specific alpha values if needed
        },
        sobol_offset=20  # Set the number of Sobol samples here
    )

"""
Configs

PoT
methods=[
            "bo_plain", 
            "bopt_standardize", "boot_standardize",
            "boot_log", "bopt_log",
            "bopt_bilog",
            #"boni_plainnonoise",# "boni_plain", 
            #"boni_standardizenonoise",# "boni_standardize", 
            # "boni_standardizenonoiserefit",
            # "boni_standardizeones", "boni_standardizezeros",
            #"boni_ilsstandardize", #"boni_ilsstandardizenonoise",
            # "boni_ilsstandardiznonoiserefit",
            # "boni_bsnoise",
            #"boni_bsnonoise",  
            #"boni_standardizegradient", #"boni_standardizegradientbinary",
            # "boni_tr", "boni_trbs",
            #"turbo_plain", "turbo_standardize",
            #"turboni_standardize", "turboni_tr", "turboni_trbs",
        ],
bottom=0.5,
        
LogEI!

HNI
methods=[
            #"bo_plain", 
            "bopt_standardize", #"boot_standardize",
            #"boot_log", "bopt_log",
            #"bopt_bilog",
            "boni_plainnonoise",# "boni_plain", 
            "boni_standardizenonoise",# "boni_standardize", 
            # "boni_standardizenonoiserefit",
            # "boni_standardizeones", "boni_standardizezeros",
            #"boni_ilsstandardize", #"boni_ilsstandardizenonoise",
            # "boni_ilsstandardiznonoiserefit",
            # "boni_bsnoise",
            #"boni_bsnonoise",  
            #"boni_standardizegradient", #"boni_standardizegradientbinary",
            # "boni_tr", "boni_trbs",
            #"turbo_plain", "turbo_standardize",
            #"turboni_standardize", "turboni_tr", "turboni_trbs",
        ],
bottom=0.45,

UCB!

"""

'''

Additional results

PoT
methods=[
            "bo_plain", 
            "bopt_standardize", "boot_standardize",
            "boot_log", "bopt_log",
            "bopt_bilog",
            #"boni_plainnonoise",#"boni_plain", 
            #"boni_standardizenonoise",# "boni_standardize", 
            #"boni_standardizenonoiserefit",
            # "boni_standardizeones", "boni_standardizezeros",
            #"boni_ilsstandardizenonoise",# "boni_ilsstandardize", 
            # "boni_ilsstandardiznonoiserefit",
            #"boni_bsnoise",
            # "boni_bsnonoise",  
            #"boni_standardizegradient", "boni_standardizegradientbinary",
            #"turbo_standardize", "turbo_plain",
            #"turboni_standardize", 
            #"turboni_tr", "turboni_trbinary",
            #"turboni_tradditivenorm",# "turboni_tradditive",
            #"turboni_trbsnorm",# "turboni_trbs",
        ],

method_column_order={  # Control method ordering in columns for combined charts
            "Column1": ["bo_plain"],
            "Column2": ["boot_standardize", "boot_log"],
            "Column3": ["bopt_standardize", "bopt_log", "bopt_bilog"],
        }, 

bottom=0.25

'''