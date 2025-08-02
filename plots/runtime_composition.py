"""
Analyzes runtime composition breakdown showing time allocation across optimization components.
Chart types: stacked bar charts and area plots visualizing relative and absolute time distributions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    setup_project_path, 
    extract_info_from_result_dir, 
    get_acquisition_function_display_names,
    get_method_display_name,
    get_objective_display_name,
    get_sorted_method_names,
    filter_results_by_criteria,
    group_results_by_dimension,
    ensure_output_dir,
    save_plot_figure,
    convert_to_numpy,
    load_and_filter_results,
    setup_plot_style,
)
from collections import defaultdict

def main(
    objective=None,  # Set to None to detect all objectives
    dim=None,        # Set to None to detect all dimensions
    seed=0,
    methods=None,
    acquisition_function=None,  # Set to None to detect all acquisition functions
    save_plots=False,
    output_dir=None,
    sweep="debug",
    show_individual_runs=False,  # Whether to show individual seed runs or just median
    normalize_by_total=False,    # Whether to show relative proportions (0-100%) instead of absolute times
):
    # Base path for results
    base_path = os.path.join(project_root, "results", sweep)
    
    if not os.path.exists(base_path):
        print(f"Results directory not found at {base_path}")
        return

    # Create output directory if saving plots
    if save_plots:
        prefix = f"{objective}_{dim}" if objective and dim else "runtime_composition"
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
            print(f"\nGenerating runtime composition plot for objective: {obj}, acquisition function: {acq_func}")
            generate_runtime_composition_plot(
                all_results=all_results, 
                objective=obj, 
                acquisition_function=acq_func, 
                save_plots=save_plots, 
                output_dir=output_dir,
                methods=methods,
                show_individual_runs=show_individual_runs,
                normalize_by_total=normalize_by_total
            )


def generate_runtime_composition_plot(
    all_results, 
    objective, 
    acquisition_function, 
    save_plots=False, 
    output_dir=None, 
    methods=None,
    show_individual_runs=False,
    normalize_by_total=False
):
    
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
    
    # Set plot style once for all plots
    setup_plot_style(PLOT_STYLE)
    
    # Create a separate plot for each dimension
    for dim, dim_results in dimension_groups.items():
        # Collect timing data for each method
        method_timing_data = defaultdict(list)
        
        for result_dir, results in dim_results.items():
            dir_info = extract_info_from_result_dir(result_dir)
            method_name = dir_info['method']
            
            # Check if all required timing data is available
            required_keys = ["gp_fitting_times", "noise_optimization_times", "acqf_optimization_times"]
            if all(key in results for key in required_keys):
                # Convert to numpy arrays
                gp_times = convert_to_numpy(results["gp_fitting_times"])
                noise_times = convert_to_numpy(results["noise_optimization_times"])
                acqf_times = convert_to_numpy(results["acqf_optimization_times"])
                
                # Store timing data for this run
                method_timing_data[method_name].append({
                    'gp_times': gp_times,
                    'noise_times': noise_times,
                    'acqf_times': acqf_times
                })
            else:
                print(f"Warning: Missing timing data in {result_dir}")
        
        if not method_timing_data:
            print(f"No timing data found for dimension {dim}")
            continue
        
        # Get method names and sort them
        method_names = list(method_timing_data.keys())
        sorted_method_names = get_sorted_method_names(method_names)
        
        # Filter methods if specified
        if methods is not None:
            sorted_method_names = [m for m in sorted_method_names if m in methods]
        
        if not sorted_method_names:
            print(f"No methods found matching criteria for dimension {dim}")
            continue
        
        # Calculate subplot layout
        n_methods = len(sorted_method_names)
        n_cols = min(3, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_methods == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_methods == 1 else axes
        else:
            axes = axes.flatten()
        
        # Hide extra subplots if needed
        for i in range(n_methods, len(axes)):
            axes[i].set_visible(False)
        
        # Define colors for the stacked areas
        colors = {
            'acqf': '#1f77b4',      # Blue for acquisition function
            'gp': '#ff7f0e',        # Orange for GP fitting
            'noise': '#2ca02c'      # Green for noise optimization
        }
        
        # Track global y-limits for consistent scaling
        global_max_time = 0
        
        # Process each method
        for method_idx, method_name in enumerate(sorted_method_names):
            ax = axes[method_idx]
            timing_runs = method_timing_data[method_name]
            
            # Find the maximum length across all runs for this method
            max_len = 0
            for run_data in timing_runs:
                max_len = max(max_len, len(run_data['gp_times']))
            
            # Prepare arrays for stacking
            all_acqf_times = []
            all_gp_times = []
            all_noise_times = []
            
            for run_data in timing_runs:
                # Pad arrays to same length
                gp_padded = np.pad(run_data['gp_times'], (0, max_len - len(run_data['gp_times'])), mode='edge')
                noise_padded = np.pad(run_data['noise_times'], (0, max_len - len(run_data['noise_times'])), mode='edge')
                acqf_padded = np.pad(run_data['acqf_times'], (0, max_len - len(run_data['acqf_times'])), mode='edge')
                
                all_acqf_times.append(acqf_padded)
                all_gp_times.append(gp_padded)
                all_noise_times.append(noise_padded)
            
            # Convert to numpy arrays
            all_acqf_times = np.array(all_acqf_times)
            all_gp_times = np.array(all_gp_times)
            all_noise_times = np.array(all_noise_times)
            
            # Calculate medians for each timing component
            median_acqf = np.median(all_acqf_times, axis=0)
            median_gp = np.median(all_gp_times, axis=0)
            median_noise = np.median(all_noise_times, axis=0)
            
            # Create iteration array
            iterations = np.arange(max_len)
            
            if normalize_by_total:
                # Calculate total time and normalize to percentages
                total_time = median_acqf + median_gp + median_noise
                # Avoid division by zero
                total_time = np.where(total_time == 0, 1, total_time)
                
                median_acqf_norm = (median_acqf / total_time) * 100
                median_gp_norm = (median_gp / total_time) * 100
                median_noise_norm = (median_noise / total_time) * 100
                
                # Create stacked areas (normalized)
                ax.fill_between(iterations, 0, median_acqf_norm, 
                               color=colors['acqf'], alpha=0.7)
                ax.fill_between(iterations, median_acqf_norm, median_acqf_norm + median_gp_norm, 
                               color=colors['gp'], alpha=0.7)
                ax.fill_between(iterations, median_acqf_norm + median_gp_norm, 
                               median_acqf_norm + median_gp_norm + median_noise_norm, 
                               color=colors['noise'], alpha=0.7)
                
                ax.set_ylabel('Time Proportion (%)')
                ax.set_ylim(0, 100)
            else:
                # Create stacked areas (absolute times)
                ax.fill_between(iterations, 0, median_acqf, 
                               color=colors['acqf'], alpha=0.7)
                ax.fill_between(iterations, median_acqf, median_acqf + median_gp, 
                               color=colors['gp'], alpha=0.7)
                ax.fill_between(iterations, median_acqf + median_gp, 
                               median_acqf + median_gp + median_noise, 
                               color=colors['noise'], alpha=0.7)
                
                # Show individual runs if requested
                if show_individual_runs:
                    for i in range(len(all_acqf_times)):
                        total_individual = all_acqf_times[i] + all_gp_times[i] + all_noise_times[i]
                        ax.plot(iterations, total_individual, color='gray', alpha=0.3, linewidth=0.5)
                
                # Plot median total time as a line
                total_median = median_acqf + median_gp + median_noise
                ax.plot(iterations, total_median, color='black', linewidth=2)
                
                ax.set_ylabel('Time (seconds)')
                
                # Update global max for consistent scaling
                current_max = np.max(median_acqf + median_gp + median_noise)
                global_max_time = max(global_max_time, current_max)
            
            # Formatting
            ax.set_xlabel('Iteration')
            ax.set_title(get_method_display_name(method_name), fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Set consistent y-limits for absolute time plots
        if not normalize_by_total and global_max_time > 0:
            for i in range(n_methods):
                axes[i].set_ylim(0, global_max_time * 1.1)
        
        # Create centralized legend below all subplots
        handles = []
        labels = []
        
        # Collect unique legend entries
        if normalize_by_total:
            # For normalized plots
            handles = [
                plt.Rectangle((0,0),1,1, color=colors['acqf'], alpha=0.7),
                plt.Rectangle((0,0),1,1, color=colors['gp'], alpha=0.7),
                plt.Rectangle((0,0),1,1, color=colors['noise'], alpha=0.7)
            ]
            labels = ['Acquisition Optimization', 'GP Fitting', 'Noise Optimization']
        else:
            # For absolute time plots
            handles = [
                plt.Rectangle((0,0),1,1, color=colors['acqf'], alpha=0.7),
                plt.Rectangle((0,0),1,1, color=colors['gp'], alpha=0.7),
                plt.Rectangle((0,0),1,1, color=colors['noise'], alpha=0.7),
                plt.Line2D([0], [0], color='black', linewidth=2)
            ]
            labels = ['Acquisition Optimization', 'GP Fitting', 'Noise Optimization', 'Total Time (median)']
        
        # Create the centralized legend
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), 
                  ncol=len(labels), fontsize=12, frameon=False)
        
        # Create overall title
        objective_display_name = get_objective_display_name(objective)
        acq_display_name = acq_display_names.get(acquisition_function, acquisition_function) if acquisition_function else "All"
        
        title_prefix = "Runtime Composition per Iteration"
        if normalize_by_total:
            title_prefix = "Runtime Composition (Normalized)"
        
        fig.suptitle(f"{title_prefix}: {objective_display_name} - {acq_display_name}", 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.80, bottom=0.25)
        
        # Save plot if requested
        if save_plots:
            suffix = "_normalized" if normalize_by_total else ""
            suffix += "_individual" if show_individual_runs else ""
            filename = f"runtime_composition_{objective}_{dim}D_{acquisition_function}{suffix}.png"
            save_plot_figure(fig, output_dir, filename)
        
        plt.show()


##################################################################
# Configuration
##################################################################

if __name__ == "__main__":

    project_root = setup_project_path()

    PLOT_STYLE = {
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 12,
        "figure.figsize": (15, 12),
        "figure.dpi": 100,
    }

    main(
        objective="ackley2D",     # Set to None to auto-detect all objectives
        dim=None,           # Set to None to auto-detect all dimensions
        seed=None,          # Set to None to auto-detect all seeds
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
        output_dir="figures/xseed_runtime",
        sweep="final",
        show_individual_runs=False,  # Set to True to show individual seed runs as thin lines
        normalize_by_total=False,    # Set to True to show relative proportions instead of absolute times
    )
