"""
Visualizes Gaussian process marginal log-likelihood evolution during optimization iterations.
Chart types: line plots showing MLL convergence and model fitting quality over iterations.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
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
from collections import defaultdict

def main(objective=None, dim=None, seed=0, methods=None, acquisition_function=None, save_plots=False, output_dir=None, sweep="debug", method_alpha=None):
    
    # Base path for results - point directly to the debug folder where your results are stored
    base_path = os.path.join(project_root, "results", sweep)
    
    if not os.path.exists(base_path):
        print(f"Results directory not found at {base_path}")
        return

    # Create output directory if saving plots
    if save_plots:
        prefix = f"{objective}_{dim}" if objective and dim else "mll_plots"
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
            print(f"\nGenerating MLL plot for objective: {obj}, acquisition function: {acq_func}")
            generate_mll_plot(
                all_results=all_results, 
                objective=obj, 
                acquisition_function=acq_func, 
                save_plots=save_plots, 
                output_dir=output_dir,
                methods=methods,
                method_alpha=method_alpha
            )


def generate_mll_plot(all_results, objective, acquisition_function, save_plots=False, output_dir=None, methods=None, method_alpha=None):
    
    # Get utility elements
    default_markers = get_default_markers()
    acq_display_names = get_acquisition_function_display_names()
    
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
    
    # Create a separate plot for each dimension
    for dim, dim_results in dimension_groups.items():
        # Prepare data for MLL plotting
        method_mll_histories = defaultdict(list)

        for result_dir, results in dim_results.items():
            dir_info = extract_info_from_result_dir(result_dir)
            method_name = dir_info['method']
            
            # Check if MLL data is available
            if "best_mlls" in results:
                best_mlls = convert_to_numpy(results["best_mlls"])
                
                # Filter out None values and convert to float
                # best_mlls contains MLL for each iteration, with None for initialization
                valid_mlls = []
                for mll in best_mlls:
                    if mll is not None:
                        valid_mlls.append(float(mll))
                    else:
                        # For initialization (None values), we'll skip or use NaN
                        valid_mlls.append(np.nan)
                
                if len(valid_mlls) > 1:  # Only include if we have actual MLL data
                    method_mll_histories[method_name].append(np.array(valid_mlls))
                else:
                    print(f"Warning: No valid MLL data found for {method_name} in {result_dir}")
            else:
                print(f"Warning: No 'best_mlls' found in results for {result_dir}")

        if not method_mll_histories:
            print(f"No valid MLL data found for objective: {objective}, acquisition function: {acquisition_function}, dimension: {dim}")
            continue
        
        # Get method names and sort them
        method_names = list(method_mll_histories.keys())
        sorted_method_names = get_sorted_method_names(method_names)
        
        # Get consistent color mapping
        color_map = get_method_color_map(sorted_method_names)

        # Prepare data for plotting
        max_len = 0
        for histories in method_mll_histories.values():
            for history in histories:
                max_len = max(max_len, len(history))

        # Process data for each method
        method_data = {}
        for method_name in sorted_method_names:
            histories = method_mll_histories[method_name]
            
            # Pad histories to the same length using forward fill for MLL values
            padded_histories = []
            for h in histories:
                if len(h) < max_len:
                    # Forward fill the last valid value
                    last_valid = h[~np.isnan(h)][-1] if len(h[~np.isnan(h)]) > 0 else np.nan
                    padded = np.pad(h, (0, max_len - len(h)), mode='constant', constant_values=last_valid)
                else:
                    padded = h
                padded_histories.append(padded)
            
            padded_histories = np.array(padded_histories)
            
            # Handle NaN values by interpolating or using median of other seeds at that iteration
            for i in range(padded_histories.shape[1]):
                # For each iteration, if some seeds have NaN, use median of valid values
                col = padded_histories[:, i]
                if np.isnan(col).any() and not np.isnan(col).all():
                    valid_values = col[~np.isnan(col)]
                    if len(valid_values) > 0:
                        padded_histories[np.isnan(col), i] = np.median(valid_values)
            
            # Store processed data
            method_data[method_name] = {
                'mll_histories': padded_histories,
                'median': np.nanmedian(padded_histories, axis=0),
                'lower_bound': np.nanpercentile(padded_histories, 25, axis=0),
                'upper_bound': np.nanpercentile(padded_histories, 75, axis=0),
            }

        n_seeds = len(method_data[sorted_method_names[0]]['mll_histories']) if sorted_method_names else 0
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 7))

        # Global y-axis limits for consistency
        ymin_global, ymax_global = float("inf"), float("-inf")
        
        # Calculate global y-limits from all data
        for method_name in sorted_method_names:
            data = method_data[method_name]
            # Only consider valid (non-NaN) values for y-limits
            valid_lower = data['lower_bound'][~np.isnan(data['lower_bound'])]
            valid_upper = data['upper_bound'][~np.isnan(data['upper_bound'])]
            
            if len(valid_lower) > 0 and len(valid_upper) > 0:
                ymin = np.min(valid_lower)
                ymax = np.max(valid_upper)
                ymin_global = min(ymin_global, ymin)
                ymax_global = max(ymax_global, ymax)

        # Get display name for acquisition function
        acq_display_name = acq_display_names.get(acquisition_function, acquisition_function) if acquisition_function else "All"

        # Plot MLL curves
        for i, method_name in enumerate(sorted_method_names):
            data = method_data[method_name]
            color = color_map[method_name]
            display_name = get_method_display_name(method_name)

            # Get alpha values for this method (default values if not specified)
            line_alpha = method_alpha.get(method_name, 0.8) if method_alpha else 0.8
            band_alpha = method_alpha.get(method_name, 0.3) if method_alpha else 0.3

            # Create x-axis (iteration numbers, starting from 0)
            x_values = np.arange(len(data['median']))
            
            # Plot median MLL line
            ax.plot(
                x_values,
                data['median'],
                color=color,
                linewidth=2.5,
                alpha=line_alpha,
                label=f"{display_name} (median MLL)",
                marker='o',
                markersize=4,
                markevery=max(1, len(x_values) // 20)  # Show markers every ~20th point
            )

            # Plot IQR band around the median
            ax.fill_between(
                x_values,
                data['lower_bound'],
                data['upper_bound'],
                color=color,
                alpha=band_alpha
            )

        # Set y-axis limits with padding
        if ymin_global != float("inf") and ymax_global != float("-inf"):
            padding = (ymax_global - ymin_global) * 0.05
            ax.set_ylim(ymin_global - padding, ymax_global + padding)

        # Style plot
        ax.set_xlabel("Iteration", fontweight="bold")
        ax.set_ylabel("Marginal Log Likelihood (MLL)", fontweight="bold")
        
        # Add title
        objective_display_name = get_objective_display_name(objective)
        title = f"MLL Evolution - {objective_display_name} with {acq_display_name} ({n_seeds} seeds)"
        ax.set_title(title, fontweight="bold", pad=12)

        # Add grid and borders
        ax.grid(True, linestyle="--", alpha=0.6)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)

        # Create legend
        handles, labels = ax.get_legend_handles_labels()
        
        # Extract method names from labels (remove " (median MLL)" suffix)
        method_labels = {}
        for h, l in zip(handles, labels):
            if "(median MLL)" in l:
                method_name = l.replace(" (median MLL)", "")
                # Find the original method key by matching display name
                for method_key in sorted_method_names:
                    if get_method_display_name(method_key) == method_name:
                        method_labels[method_key] = (h, method_name)
                        break
        
        # Group methods by type
        method_groups = group_methods_by_type(list(method_labels.keys()))
        max_methods_per_group = max(len(methods) for methods in method_groups.values()) if method_groups else 0
        
        # Create grouped legend using common function
        create_grouped_legend(
            fig, ax, method_labels, sorted_method_names, max_methods_per_group,
            "Lines show median MLL values with IQR shading"
        )

        fig.tight_layout(rect=[0, 0.35, 1, 1.0])

        if save_plots:
            filename = f"mll_{objective}_{acq_display_name.replace(' ', '_')}.png"
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
        objective="ackley2D",     # Set to None to auto-detect all objectives
        dim=None,           # Set to None to auto-detect all dimensions
        seed=None,      # Set to None to auto-detect all seeds
        methods=[
            "bo_plain", 
            "boot_standardize", "bopt_standardize", 
            "boot_log", "bopt_log",
            "bopt_bilog",
            "boni_plain", "boni_plainnonoise",
            "boni_standardize", "boni_standardizenonoise",
            "boni_standardizenonoiserefit",
            "boni_standardizeones", "boni_standardizezeros",
            "boni_ilsstandardize", "boni_ilsstandardizenonoise",
            "boni_ilsstandardiznonoiserefit",
            "boni_bsnoise",
            "boni_bsnonoise",  
            "boni_standardizegradient", "boni_standardizegradientbinary",
            "boni_tr", "boni_trbs",
            "turbo_plain", "turbo_standardize",
            "turboni_standardize", "turboni_tr", "turboni_trbs",
        ],
        acquisition_function=None, # Set to None to auto-detect all acquisition functions
        save_plots=True,
        output_dir="figures/xseed_mll",
        sweep="final",
        method_alpha={  # Control transparency/blending of specific methods
            # "bo_plain": 0.3,
            # "boot_log": 0.3,
            # "boot_standardize": 0.3,
        }
    )
