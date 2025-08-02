"""
Thesis visualization of exploration behavior and search space coverage analysis.
Chart types: Line plots with custom legends showing exploration patterns across optimization iterations.
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
    get_method_display_name,
    filter_results_by_criteria,
    group_results_by_dimension,
    convert_to_numpy,
    sort_methods_within_group,
    group_methods_by_type,
    load_and_filter_results,
    get_thesis_figure_size,
    create_thesis_legend,
    setup_thesis_style,
    create_thesis_output_dir,
    save_thesis_plot
)

def generate_exploration_plot(all_results, objective, acquisition_function, save_plots=False, output_dir=None, method_column_order=None):
    # Get utility elements
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
    
    # Create a separate plot for each dimension
    for dim, dim_results in dimension_groups.items():
        # Get thesis figure size
        fig_width, fig_height = get_thesis_figure_size()
        fig, ax = plt.subplots(figsize=(fig_width * 1.0, fig_height * 0.8))
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
            
        # Get method names and sort them
        method_names = list(method_exploration.keys())
        method_groups = group_methods_by_type(method_names)
        sorted_method_names = []
        for group_name, methods in method_groups.items():
            sorted_method_names.extend(sort_methods_within_group(methods))
        
        # Get RWTH color mapping - use same as regret chart
        color_map = get_method_color_map(sorted_method_names)

        for i, method_name in enumerate(sorted_method_names):
            explorations = method_exploration[method_name]
            color = color_map[method_name]
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

            # Compute mean and confidence intervals
            mean_exploration = np.nanmean(padded_explorations, axis=0)
            lower_bound = np.nanpercentile(padded_explorations, 25, axis=0)
            upper_bound = np.nanpercentile(padded_explorations, 75, axis=0)

            # Plot mean exploration line
            zorder = 1 if method_name == "boni_bsnonoise" else 2  # Manual fix: Put boni_bsnonoise in the back
            ax.plot(
                np.arange(len(mean_exploration)),
                mean_exploration,
                color=color,
                linewidth=2,
                alpha=1.0,
                label=display_name,
                zorder=zorder
            )

            # Plot IQR confidence interval band
            ax.fill_between(
                np.arange(len(mean_exploration)),
                lower_bound,
                upper_bound,
                color=color,
                alpha=0.3
            )

            ymin, ymax = np.nanmin(lower_bound), np.nanmax(upper_bound)
            ymin_global = min(ymin_global, ymin)
            ymax_global = max(ymax_global, ymax)

        # Set y-axis limits with a small padding
        if ymin_global != float("inf") and ymax_global != float("-inf"):
            padding = (ymax_global - ymin_global) * 0.05
            ax.set_ylim(ymin_global - padding, ymax_global + padding)

        # Set x-axis limits with no margins (start at 0, end at max iteration)
        max_iterations = 0
        for method_name in sorted_method_names:
            explorations = method_exploration[method_name]
            for exploration in explorations:
                if len(exploration) > 0:
                    max_iterations = max(max_iterations, len(exploration))
        if max_iterations > 0:
            ax.set_xlim(0, max_iterations - 1)

        ax.set_xlabel(r'Iteration')
        ax.set_ylabel(r'Exploration Metric')

        handles, labels = ax.get_legend_handles_labels()
        
        # Extract method names from labels and create proper method_labels dict
        method_labels = {}
        for h, l in zip(handles, labels):
            # Find the original method key by matching display name
            for method_key in sorted_method_names:
                if get_method_display_name(method_key) == l:
                    method_labels[method_key] = (h, l)
                    break
        
        # Create legend with custom column ordering if specified
        if method_column_order is not None:
            # Use custom column ordering for exploration charts
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
                        if method in method_labels:
                            handle, display_name = method_labels[method]
                            all_handles.append(handle)
                            all_labels.append(display_name)
                    
                    # Add padding to make all columns the same length
                    methods_in_this_column = len(methods_in_column)
                    padding_needed = max_methods_per_column - methods_in_this_column
                    for _ in range(padding_needed):
                        all_handles.append(Line2D([0], [0], alpha=0))  # Invisible handle
                        all_labels.append("")  # Empty label                 
                # Create the unified legend with proper grid layout
                legend = fig.legend(
                    all_handles,
                    all_labels,
                    loc='lower center',
                    bbox_to_anchor=(0.5, 0.02),  # Close to plots
                    ncol=num_columns,  # One column per specified column
                    frameon=False,  # No background frame
                    columnspacing=2.0,
                    handletextpad=0.5
                )
        else:
            # Use standard thesis legend for other chart types
            create_thesis_legend(
                fig, ax, method_labels, sorted_method_names
            )

        # Adjust layout for legend below - similar to regret thesis chart
        plt.tight_layout(rect=[0, 0.2, 1, 1.0])  # Make room for legend below

        # Save plot if requested
        if save_plots:
            acq_display_name = acq_display_names.get(acquisition_function, acquisition_function) if acquisition_function else "All"
            dim_str = f"{dim}D" if dim else ""
            filename = f"exploration_{objective}_{dim_str}_{acq_display_name.replace(' ', '_')}_thesis.pdf"
            save_thesis_plot(fig, output_dir, filename)
            
        plt.show()


def main(objective=None, dim=None, seed=None, methods=None, acquisition_function=None, save_plots=False, method_column_order=None, output_dir=None, sweep="final"):
    project_root = setup_project_path()
    
    # Base path for results
    base_path = os.path.join(project_root, "results", sweep)
    
    if not os.path.exists(base_path):
        print(f"Results directory not found at {base_path}")
        return

    # Create output directory for thesis plots
    if save_plots:
        output_dir = create_thesis_output_dir(output_dir, "thesis")

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
            print(f"\nGenerating thesis-style exploration plot for objective: {obj}, acquisition function: {acq_func}")
            generate_exploration_plot(
                all_results=all_results, 
                objective=obj, 
                acquisition_function=acq_func, 
                save_plots=save_plots, 
                output_dir=output_dir,
                method_column_order=method_column_order
            )

##################################################################
# Configuration
##################################################################

if __name__ == "__main__":
    project_root = setup_project_path()
    
    setup_thesis_style()

    main(
        objective="ackley10D",     # Set to None to auto-detect all objectives
        dim=None,                  # Set to None to auto-detect all dimensions
        seed=None,                 # Set to None to auto-detect all seeds
        methods=[
            #"bo_plain", 
            "bopt_standardize",#"boot_standardize",  
            #"boot_log", "bopt_log",
            #"bopt_bilog",
            "boni_plainnonoise",# "boni_plain", 
            "boni_standardizenonoise",# "boni_standardize", 
            # "boni_standardizenonoiserefit",
            # "boni_standardizeones", "boni_standardizezeros",
            "boni_ilsstandardize", #"boni_ilsstandardizenonoise",
            # "boni_ilsstandardiznonoiserefit",
            #"boni_bsnoise",
            "boni_bsnonoise",  
            "boni_standardizegradient", #"boni_standardizegradientbinary",
            #"turbo_standardize", "turbo_plain",
            #"turboni_standardize", 
            #"turboni_tr", "turboni_trbinary",
            #"turboni_tradditivenorm",# "turboni_tradditive",
            #"turboni_trbsnorm",# "turboni_trbs",
        ],
        acquisition_function="UpperConfidenceBound", # Set to None to auto-detect all acquisition functions
        save_plots=True,
        output_dir="figures/thesis/",
        sweep="final",
        method_column_order={  # Control method ordering in columns for exploration charts
                "Column1": ["bopt_standardize"],
                "Column2": ["boni_plainnonoise", "boni_standardizenonoise"],
                "Column3": ["boni_ilsstandardize", "boni_bsnonoise", "boni_standardizegradient"]
            }
    )

"""
Configs for thesis plots

PoT
bottom margin: 0.15


HNI
methods=[
            #"bo_plain", 
            "bopt_standardize",#"boot_standardize",  
            #"boot_log", "bopt_log",
            #"bopt_bilog",
            "boni_plainnonoise",# "boni_plain", 
            "boni_standardizenonoise",# "boni_standardize", 
            # "boni_standardizenonoiserefit",
            # "boni_standardizeones", "boni_standardizezeros",
            "boni_ilsstandardize", #"boni_ilsstandardizenonoise",
            # "boni_ilsstandardiznonoiserefit",
            #"boni_bsnoise",
            "boni_bsnonoise",  
            "boni_standardizegradient", #"boni_standardizegradientbinary",
            #"turbo_standardize", "turbo_plain",
            #"turboni_standardize", 
            #"turboni_tr", "turboni_trbinary",
            #"turboni_tradditivenorm",# "turboni_tradditive",
            #"turboni_trbsnorm",# "turboni_trbs",
        ],

bottom margin: 0.2

method_column_order={  # Control method ordering in columns for exploration charts
                "Column1": ["bopt_standardize"],
                "Column2": ["boni_plainnonoise", "boni_standardizenonoise"],
                "Column3": ["boni_ilsstandardize", "boni_bsnonoise", "boni_standardizegradient"]
            }

            
TRNI
methods=[
    #"bo_plain", 
    #"bopt_standardize",#"boot_standardize",  
    #"boot_log", "bopt_log",
    #"bopt_bilog",
    #"boni_plainnonoise",# "boni_plain", 
    #"boni_standardizenonoise",# "boni_standardize", 
    # "boni_standardizenonoiserefit",
    # "boni_standardizeones", "boni_standardizezeros",
    #"boni_ilsstandardize", #"boni_ilsstandardizenonoise",
    # "boni_ilsstandardiznonoiserefit",
    #"boni_bsnoise",
    #"boni_bsnonoise",  
    #"boni_standardizegradient", #"boni_standardizegradientbinary",
    "turbo_standardize", "turbo_plain",
    "turboni_standardize", 
    "turboni_tr", #"turboni_trbinary",
    "turboni_tradditivenorm",# "turboni_tradditive",
    "turboni_trbsnorm",# "turboni_trbs",
],

bottom margin: 0.15

method_column_order={  # Control method ordering in columns for exploration charts
                "Column1": ["turbo_standardize", "turbo_plain"],
                "Column2": ["turboni_standardize", "turboni_tr"],
                "Column3": ["turboni_tradditivenorm", "turboni_trbsnorm"],
            }
"""