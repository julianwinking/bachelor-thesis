"""
Sensitivity analysis for model refit frequency impact on optimization performance.
Chart types: comparison plots with error bars showing refit strategy effects across different methods.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from utils import (
    setup_project_path, 
    get_objective_display_name,
    convert_to_numpy,
    RWTH_COLORS,
    get_thesis_figure_size,
    setup_thesis_style,
    save_thesis_plot,
)

def main(
    save_plots=True,
    output_dir="figures/thesis/sensitivity",
    sweep="final"
):
    
    # Setup project path
    project_root = setup_project_path()
    
    # Base path for results
    base_path = os.path.join(project_root, "results", sweep)
    
    if not os.path.exists(base_path):
        print(f"Results directory not found at {base_path}")
        return

    # Create output directory if saving plots
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)

    # Load all results
    all_results = load_all_refit_results(base_path)
    
    if not all_results:
        print("No results found")
        return
    
    # Generate comparison plot
    print("Generating refit comparison plot with error charts")
    generate_refit_comparison_plot(
        all_results=all_results,
        save_plots=save_plots,
        output_dir=output_dir
    )


def load_all_refit_results(base_path):
    all_results = {}
    
    if not os.path.exists(base_path):
        print(f"Base path does not exist: {base_path}")
        return all_results
    
    # Define the methods to compare (refit vs non-refit)
    methods_to_compare = [
        "boni_standardizenonoise", "boni_standardizenonoiserefit",  # Naive
        "boni_ilsstandardizenonoise", "boni_ilsstandardizenonoiserefit"  # ILS
    ]
    
    # Get all result directories
    result_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    # Debug: count methods found
    method_counts = {method: 0 for method in methods_to_compare}
    
    for result_dir in result_dirs:
        result_path = os.path.join(base_path, result_dir)
        
        # Extract information from directory name
        # Format: boni_standardizenonoiserefit_ackley2D_2_UpperConfidenceBound_0
        parts = result_dir.split('_')
        
        if len(parts) < 5:
            continue
            
        # Extract method name (first parts until we hit the objective)
        # Handle both refit and non-refit methods
        if parts[0] == "boni":
            if parts[1] == "standardizenonoiserefit":
                method_name = "boni_standardizenonoiserefit"
                objective_start_idx = 2
            elif parts[1] == "standardizenonoise":
                method_name = "boni_standardizenonoise"
                objective_start_idx = 2
            elif parts[1] == "ilsstandardizenonoiserefit":
                method_name = "boni_ilsstandardizenonoiserefit"
                objective_start_idx = 2
            elif parts[1] == "ilsstandardizenonoise":
                method_name = "boni_ilsstandardizenonoise"
                objective_start_idx = 2
            else:
                continue
        else:
            continue
        
        # Check if this is one of the methods we want to compare
        if method_name not in methods_to_compare:
            continue
            
        # Count this method
        method_counts[method_name] += 1
            
        # Extract objective and dimension
        objective_name = parts[objective_start_idx]  # ackley2D
        dim = int(parts[objective_start_idx + 1])    # 2
        
        # Extract acquisition function
        acq_func = parts[objective_start_idx + 2]    # UpperConfidenceBound
        
        # Extract seed
        seed = int(parts[objective_start_idx + 3])   # 0
        
        # Load results
        results_file = os.path.join(result_path, "results.pkl")
        if os.path.exists(results_file):
            try:
                with open(results_file, 'rb') as f:
                    results = pickle.load(f)
                
                # Get final iteration result (best so far)
                if "history_y" in results:
                    history_y = convert_to_numpy(results["history_y"])
                    final_result = np.min(history_y)  # Assuming minimization
                    
                    # Store result
                    key = (objective_name, dim, acq_func, method_name, seed)
                    all_results[key] = final_result
                    
            except Exception as e:
                print(f"Error loading {results_file}: {e}")
                continue
    
    # Debug: print summary of loaded results
    print(f"\nMethod counts found:")
    for method, count in method_counts.items():
        print(f"  {method}: {count} directories")
    
    print(f"\nTotal results loaded: {len(all_results)}")
    
    # Debug: show breakdown by method
    method_result_counts = {}
    for key in all_results.keys():
        method = key[3]  # method_name is at index 3
        method_result_counts[method] = method_result_counts.get(method, 0) + 1
    
    print(f"Results by method:")
    for method, count in method_result_counts.items():
        print(f"  {method}: {count} results")
    
    return all_results


def generate_refit_comparison_plot(all_results, save_plots=False, output_dir=None):

    setup_thesis_style()
    
    # Define objectives to plot (only 2D)
    objectives = [
        ("ackley2D", 2),
        ("rastrigin2D", 2)
    ]
    
    # Define methods to compare (grouped by heuristic type)
    method_groups = {
        "Naive": ["boni_standardizenonoise", "boni_standardizenonoiserefit"],
        "ILS": ["boni_ilsstandardizenonoise", "boni_ilsstandardizenonoiserefit"]
    }
    
    # Define acquisition functions
    acquisition_functions = ["UpperConfidenceBound", "LogExpectedImprovement"]
    
    # Create figure with 1x4 subplots (1 row: 2 objectives x 2 acquisition functions)
    height, width = get_thesis_figure_size()
    width = width * 1.6
    height = height * 0.35

    fig, axes = plt.subplots(1, 4, figsize=(width, height))
    
    # Colors for refit vs non-refit variants
    refit_colors = {
        "norefit": RWTH_COLORS['blue'][0],
        "refit": RWTH_COLORS['orange'][0]
    }
    
    # Markers for different heuristic types
    heuristic_markers = {
        "Naive": "o",
        "ILS": "s"
    }
    
    # Process each acquisition function and objective combination
    # Layout: [UCB_Ackley2D, UCB_Rastrigin2D, LogEI_Ackley2D, LogEI_Rastrigin2D]
    plot_idx = 0
    for acq_func in acquisition_functions:
        for objective_name, dim in objectives:
            ax = axes[plot_idx]
            plot_idx += 1
            
            # Get objective display name
            obj_display_name = get_objective_display_name(objective_name)
            
            # Collect data for this objective and acquisition function
            all_method_data = []
            method_labels = []
            method_colors = []
            method_markers = []
            
            # Process methods in pairs to ensure alignment
            for group_name, methods in method_groups.items():
                norefit_method = methods[0]  # First method (no refit variant)
                refit_method = methods[1]    # Second method (refit variant)
                
                # Check if we have data for both variants
                norefit_results = []
                refit_results = []
                
                # Collect results for no refit variant
                for seed in range(20):  # 0-19
                    key = (objective_name, dim, acq_func, norefit_method, seed)
                    if key in all_results:
                        norefit_results.append(all_results[key])
                
                # Collect results for refit variant
                for seed in range(20):  # 0-19
                    key = (objective_name, dim, acq_func, refit_method, seed)
                    if key in all_results:
                        refit_results.append(all_results[key])
                
                # Add no refit variant data (or placeholder)
                if norefit_results:
                    all_method_data.append(norefit_results)
                    method_colors.append(refit_colors["norefit"])
                    method_markers.append(heuristic_markers[group_name])
                else:
                    # Add placeholder with NaN values
                    all_method_data.append([np.nan])
                    method_colors.append('white')
                    method_markers.append(heuristic_markers[group_name])
                
                # Add refit variant data (or placeholder)
                if refit_results:
                    all_method_data.append(refit_results)
                    method_colors.append(refit_colors["refit"])
                    method_markers.append(heuristic_markers[group_name])
                else:
                    # Add placeholder with NaN values
                    all_method_data.append([np.nan])
                    method_colors.append('white')
                    method_markers.append(heuristic_markers[group_name])
                
                # Add group name label (will be centered between the two variants)
                method_labels.append(group_name)
            
            # Plot error charts for this objective and acquisition function
            if all_method_data:
                plot_method_comparison(
                    ax, all_method_data, method_labels, method_colors, method_markers
                )
            
            # Customize subplot
            if plot_idx == 1:  # Only show y-label on leftmost two columns (UCB)
                ax.set_ylabel("Best value")
            
            # Set title for each subplot
            ax.set_title(f"{obj_display_name}")
            
            # Show x-axis ticks and labels for all subplots
            ax.tick_params(axis='x', rotation=45)
    
    # Create legend below the chart
    legend_handles = []
    legend_labels = []
    
    # Add refit variant legend entries
    for variant, color in refit_colors.items():
        variant_name = "No refit" if variant == "norefit" else "Refit"
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                       markersize=8, label=variant_name))
        legend_labels.append(variant_name)
    
    # Add legend below the chart
    fig.legend(legend_handles, legend_labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), 
              ncol=2, frameon=False)
    
    # Adjust layout for 1x4 subplot grid
    plt.subplots_adjust(
        left=0.10,      # Left margin
        right=0.95,     # Right margin  
        top=0.90,       # Top margin
        bottom=0.45,    # Bottom margin
        wspace=0.30      # Width spacing between subplots
    )
    
    # Save plot
    if save_plots and output_dir:
        filename = "refit_sensitivity.pdf"
        save_thesis_plot(fig, output_dir, filename)
        print(f"Saved comparison plot to {os.path.join(output_dir, filename)}")
    
    plt.show()


def plot_method_comparison(ax, results_list, labels, colors, markers):
    
    x_positions = np.arange(1, len(results_list) + 1)
    
    for i, (results, x_pos) in enumerate(zip(results_list, x_positions)):
        # Skip plotting if this is a placeholder (NaN values)
        if len(results) == 1 and np.isnan(results[0]):
            continue
        
        # Calculate median and error
        median_val = np.median(results)
        
        # Calculate error as interquartile range (IQR)
        q25 = np.percentile(results, 25)
        q75 = np.percentile(results, 75)
        error_lower = median_val - q25
        error_upper = q75 - median_val
        
        # Plot median value as a dot with asymmetric error bars showing full IQR
        ax.errorbar(x_pos, median_val, yerr=[[error_lower], [error_upper]],
                   c=colors[i], marker=markers[i], markersize=3, capsize=3, capthick=1,
                   alpha=1, linewidth=0.5)
    
    # Set x-axis ticks and labels - center labels between pairs of error bars
    # We have 2 groups (Naive, ILS), each with 2 variants
    # So positions 1.5, 3.5 for the centered labels
    group_positions = [1.5, 3.5]
    ax.set_xticks(group_positions)
    ax.set_xticklabels(labels)


if __name__ == "__main__":
    project_root = setup_project_path()
    
    setup_thesis_style()

    main(
        save_plots=True,
        output_dir="figures/thesis/sensitivity",
        sweep="final"
    ) 