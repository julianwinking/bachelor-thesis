"""
Visualizes exploration behavior and search space coverage of optimization methods.
Chart types: Line plots showing exploration patterns and search space utilization.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from utils import (
    setup_project_path, 
    extract_info_from_result_dir, 
    get_acquisition_function_display_names,
    get_method_color_map,
    get_method_display_name,
    get_objective_display_name,
    filter_results_by_criteria,
    group_results_by_dimension,
    ensure_output_dir,
    save_plot_figure,
    convert_to_numpy,
    sort_methods_within_group,
    group_methods_by_type,
    load_and_filter_results,
    create_grouped_legend,
    sort_methods_within_group,
    setup_plot_style,
    get_default_markers
)


def generate_exploration_plot(all_results, objective, acquisition_function, save_plots=False, output_dir=None):
    # Get utility elements
    acq_display_names = get_acquisition_function_display_names()
    default_markers = get_default_markers()
    
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
        fig, ax = plt.subplots()
        ymin_global, ymax_global = float("inf"), float("-inf")

        # Group results by method across seeds
        method_exploration = defaultdict(list)

        for result_dir, results in dim_results.items():
            dir_info = extract_info_from_result_dir(result_dir)
            method_name = dir_info['method']
            # Check if exploration metrics are available
            if "exploration_metrics" in results:
                exploration_data = convert_to_numpy(results["exploration_metrics"])
                method_exploration[method_name].append(exploration_data)
        
        if not any(method_exploration.values()):
            print(f"No exploration metrics found for dimension: {dim}")
            continue
            
        # Get method names and sort them using consistent method grouping
        method_names = list(method_exploration.keys())
        
        # Group methods by type and sort within groups for consistent organization
        method_groups = group_methods_by_type(method_names)
        sorted_method_names = []
        for group_name, methods in method_groups.items():
            sorted_method_names.extend(sort_methods_within_group(methods))
        
        # Get consistent color mapping based on sorted order
        color_map = get_method_color_map(sorted_method_names)

        for i, method_name in enumerate(sorted_method_names): # Iterate over sorted_method_names
            explorations = method_exploration[method_name] # Get data for the current method
            color = color_map[method_name]
            marker = default_markers[i % len(default_markers)]
            display_name = get_method_display_name(method_name)

            # Pad exploration data to the same length
            max_len = max(len(e) for e in explorations if len(e) > 0)
            if max_len == 0:
                continue
                
            padded_explorations = np.array([np.pad(e, (0, max_len - len(e)), 
                                     mode='constant', constant_values=np.nan) 
                            for e in explorations if len(e) > 0])
            
            if len(padded_explorations) == 0:
                continue

            # Number of seeds used
            n_seeds = len(padded_explorations)

            # Compute mean and confidence intervals
            mean_exploration = np.nanmean(padded_explorations, axis=0)
            lower_bound = np.nanpercentile(padded_explorations, 5, axis=0)
            upper_bound = np.nanpercentile(padded_explorations, 95, axis=0)

            # Plot mean exploration line
            ax.plot(
                np.arange(len(mean_exploration)),
                mean_exploration,
                color=color,
                marker=marker,
                markersize=5,
                linewidth=2.5,
                alpha=0.9,
                label=f"{display_name} (mean exploration)"
            )

            # Plot 90% confidence interval band around mean exploration
            ax.fill_between(
                np.arange(len(mean_exploration)),
                lower_bound,
                upper_bound,
                color=color,
                alpha=0.2
            )

            ymin, ymax = np.nanmin(lower_bound), np.nanmax(upper_bound)
            ymin_global = min(ymin_global, ymin)
            ymax_global = max(ymax_global, ymax)

        # Set y-axis limits with a small padding
        if ymin_global != float("inf") and ymax_global != float("-inf"):
            padding = (ymax_global - ymin_global) * 0.05
            ax.set_ylim(ymin_global - padding, ymax_global + padding)

        # Create a title that includes the objective, dimension, and acquisition function
        acq_display_name = acq_display_names.get(acquisition_function, acquisition_function) if acquisition_function else "All"
        objective_display_name = get_objective_display_name(objective)
        dim_str = f"{dim}D" if dim else ""
        title = f"Exploration Behavior - {objective_display_name} with {acq_display_name} ({n_seeds} seeds)"
        
        # Add plot styling
        ax.set_xlabel("Iteration", fontweight="bold")
        ax.set_ylabel("Average Minimum Distance (Exploration Metric)", fontweight="bold")
        
        if title:
            ax.set_title(title, fontweight="bold", pad=15)

        # Add a light horizontal line at y=0 if relevant
        ymin, ymax = ax.get_ylim()
        if ymin < 0 < ymax:
            ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

        # Add a subtle grid
        ax.grid(True, linestyle="--", alpha=0.6)

        # Add a border to the plot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
        
        fig.tight_layout(rect=[0, 0.2, 1, 1])
        
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            # Create method_labels_dict for the grouped legend
            method_labels_dict = {}
            for handle, label in zip(handles, labels):
                # Extract method name from label (remove " (mean exploration)")
                display_name = label.replace(" (mean exploration)", "")
                # Find corresponding method name
                for method in sorted_method_names:
                    if get_method_display_name(method) == display_name:
                        method_labels_dict[method] = (handle, display_name)
                        break
            
            # Get the maximum number of methods in any group for proper spacing
            method_groups = group_methods_by_type(list(method_labels_dict.keys()))
            max_methods_per_group = max(len(methods) for methods in method_groups.values()) if method_groups else 0
            
            # Use the shared grouped legend function
            create_grouped_legend(
                fig, ax, method_labels_dict, sorted_method_names, max_methods_per_group,
                info_text="Lines show mean exploration with 90% confidence intervals"
            )

        # Save plot if requested
        if save_plots:
            filename = f"exploration_{objective}_{dim_str}_{acq_display_name.replace(' ', '_')}.png"
            save_plot_figure(fig, output_dir, filename)
            
        plt.show()


def main(
    objective=None, dim=None, seed=None, methods=None, acquisition_function=None, save_plots=False, output_dir=None, sweep="debug"
):
    project_root = setup_project_path()
    
    # Base path for results
    base_path = os.path.join(project_root, "results", sweep)
    
    if not os.path.exists(base_path):
        print(f"Results directory not found at {base_path}")
        return

    # Create output directory if saving plots
    if save_plots:
        prefix = f"exploration_{objective}_{dim}" if objective and dim else "exploration_plots"
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
    
    # Generate exploration plots
    objectives_to_plot = [objective] if objective else all_objectives
    acq_funcs_to_plot = [acquisition_function] if acquisition_function else all_acquisition_functions
    
    for obj in objectives_to_plot:
        for acq_func in acq_funcs_to_plot:
            print(f"\nGenerating exploration plot for objective: {obj}, acquisition function: {acq_func}")
            generate_exploration_plot(
                all_results=all_results, 
                objective=obj, 
                acquisition_function=acq_func, 
                save_plots=save_plots, 
                output_dir=output_dir
            )

##################################################################
# Configuration
##################################################################

if __name__ == "__main__":

    project_root = setup_project_path()

    PLOT_STYLE = {
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.figsize": (18, 10),  # Increased height to accommodate legend below
        "figure.dpi":70,
    }

    main(
        objective="ackley10D",     # Set to None to auto-detect all objectives
        dim=None,                  # Set to None to auto-detect all dimensions
        seed=None,                 # Set to None to auto-detect all seeds
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
        output_dir="figures/xseed_exploration",
        sweep="final"
    )
