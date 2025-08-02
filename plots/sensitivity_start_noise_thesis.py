"""
Sensitivity analysis for initial noise configuration effects on heuristic optimization methods.
Chart types: comparison plots with error bars showing heuristic performance across different noise initializations.
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
    all_results = load_all_heuristic_results(base_path)
    
    if not all_results:
        print("No results found")
        return
    
    # Generate comparison plot
    print("Generating heuristic noise comparison plot with error charts")
    generate_heuristic_comparison_plot(
        all_results=all_results,
        save_plots=save_plots,
        output_dir=output_dir
    )


def load_all_heuristic_results(base_path):
    all_results = {}
    
    if not os.path.exists(base_path):
        print(f"Base path does not exist: {base_path}")
        return all_results
    
    # Define the methods to compare
    methods_to_compare = [
        "boni_standardize", "boni_standardizenonoise",  # Naive
        "boni_ilsstandardize", "boni_ilsstandardizenonoise",  # ILS
        "boni_bsnoise", "boni_bsnonoise"  # Beam Search
    ]
    
    # Get all result directories
    result_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    for result_dir in result_dirs:
        result_path = os.path.join(base_path, result_dir)
        
        # Extract information from directory name
        # Format: boni_standardize_ackley2D_2_UpperConfidenceBound_0
        parts = result_dir.split('_')
        
        if len(parts) < 5:
            continue
            
        # Extract method name (first two parts)
        method_name = f"{parts[0]}_{parts[1]}"
        
        # Check if this is one of the methods we want to compare
        if method_name not in methods_to_compare:
            continue
            
        # Extract objective and dimension
        objective_name = parts[2]  # ackley2D
        dim = int(parts[3])        # 2
        
        # Extract acquisition function
        acq_func = parts[4]        # UpperConfidenceBound
        
        # Extract seed
        seed = int(parts[5])       # 0
        
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
    
    return all_results


def generate_heuristic_comparison_plot(all_results, save_plots=False, output_dir=None):
    
    setup_thesis_style()

    objectives = [
        ("ackley2D", 2),
        ("ackley10D", 10), 
        ("rastrigin2D", 2),
        ("rastrigin10D", 10)
    ]
    
    method_groups = {
        "Naive": ["boni_standardize", "boni_standardizenonoise"],
        "ILS": ["boni_ilsstandardize", "boni_ilsstandardizenonoise"],
        "BS": ["boni_bsnoise", "boni_bsnonoise"]
    }
    
    acquisition_functions = ["UpperConfidenceBound", "LogExpectedImprovement"]
    
    # Create figure with 2x4 subplots (2 rows for acquisition functions, 4 columns for objectives)
    # Use GridSpec to create different widths for 2D vs 10D objectives
    height, width = get_thesis_figure_size()
    height = height * 0.45 # Increase height for 2 rows
    width = width * 1.6

    fig = plt.figure(figsize=(width, height))
    
    # Create GridSpec with different column widths
    # 2D objectives (columns 0, 2) get more width, 10D objectives (columns 1, 3) get less width
    gs = fig.add_gridspec(2, 4, width_ratios=[1.4, 1.0, 1.4, 1.0], 
                          height_ratios=[1, 1],
                          wspace=0.4, hspace=0.2)
    
    # Create axes using GridSpec
    axes = np.empty((2, 4), dtype=object)
    for i in range(2):
        for j in range(4):
            axes[i, j] = fig.add_subplot(gs[i, j])
    
    # Colors for noise vs nonoise variants
    noise_colors = {
        "noise": RWTH_COLORS['blue'][0],
        "nonoise": RWTH_COLORS['orange'][0]
    }
    
    # Markers for different heuristic types
    heuristic_markers = {
        "Naive": "o",
        "ILS": "s", 
        "BS": "^"
    }
    
    # Process each acquisition function (row) and objective (column)
    for acq_idx, acq_func in enumerate(acquisition_functions):
        for obj_idx, (objective_name, dim) in enumerate(objectives):
            ax = axes[acq_idx, obj_idx]
            
            # Get objective display name
            obj_display_name = get_objective_display_name(objective_name)
            
            # Collect data for this objective and acquisition function
            all_method_data = []
            method_labels = []
            method_colors = []
            method_markers = []
            
            # Process methods in pairs to ensure alignment
            for group_name, methods in method_groups.items():
                # Manual: Skip beam search for 10D objectives
                if group_name == "BS" and dim == 10:
                    continue
                    
                noise_method = methods[0]  # First method (noise variant)
                nonoise_method = methods[1]  # Second method (nonoise variant)
                
                # Check if we have data for both variants
                noise_results = []
                nonoise_results = []
                
                # Collect results for noise variant
                for seed in range(20):  # 0-19
                    key = (objective_name, dim, acq_func, noise_method, seed)
                    if key in all_results:
                        noise_results.append(all_results[key])
                
                # Collect results for nonoise variant
                for seed in range(20):  # 0-19
                    key = (objective_name, dim, acq_func, nonoise_method, seed)
                    if key in all_results:
                        nonoise_results.append(all_results[key])
                
                # Add noise variant data (or placeholder)
                if noise_results:
                    all_method_data.append(noise_results)
                    method_colors.append(noise_colors["noise"])
                    method_markers.append(heuristic_markers[group_name])
                else:
                    # Add placeholder with NaN values
                    all_method_data.append([np.nan])
                    method_colors.append('white')
                    method_markers.append(heuristic_markers[group_name])
                
                # Add nonoise variant data (or placeholder)
                if nonoise_results:
                    all_method_data.append(nonoise_results)
                    method_colors.append(noise_colors["nonoise"])
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
            if obj_idx == 0:  # Only show y-label on leftmost column
                ax.set_ylabel("Best value")
            
            # Set title for top row only
            if acq_idx == 0:
                ax.set_title(f"{obj_display_name}")
            
            # Only show x-axis ticks and labels on bottom row
            if acq_idx == 1:
                ax.tick_params(axis='x', rotation=45)
            else:
                ax.set_xticklabels([])  # Hide x-axis labels on top row
            
            # Format y-axis to show decimal places only when needed
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}' if x % 1 != 0 else f'{int(x)}'))
        
    
    # Create legend below the chart
    legend_handles = []
    legend_labels = []
    
    # Add noise variant legend entries
    for variant, color in noise_colors.items():
        variant_name = "Noise First" if variant == "noise" else "No Noise First"
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                       markersize=8, label=variant_name))
        legend_labels.append(variant_name)
    
    # Add legend below the chart
    fig.legend(legend_handles, legend_labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), 
              ncol=5, frameon=False)
    
    # Adjust layout for legend positioning
    plt.subplots_adjust(
        left=0.10,      # Left margin
        right=0.95,     # Right margin  
        top=0.95,       # Top margin
        bottom=0.33     # Bottom margin
    )
    

    
    # Save plot
    if save_plots and output_dir:
        filename = "start_noise_sensitivity.pdf"
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
    # Calculate positions dynamically based on number of groups
    # Each group has 2 variants, so positions are at 1.5, 3.5, 5.5, etc.
    num_groups = len(labels)
    group_positions = [1.5 + 2*i for i in range(num_groups)]
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