"""
Analyzes and visualizes computational runtime performance of optimization methods.
Chart types: cumulative runtime curves, runtime distribution boxplots, and combined runtime-performance plots.
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
    apply_method_alpha_to_boxplot
)

def get_runtime_data(results, method_name, seed, fun_name):
    # Check if actual runtime data is available
    if 'cumulative_times' in results and len(results['cumulative_times']) > 0:
        # Use actual runtime data
        cumulative_times = results['cumulative_times']
        if hasattr(cumulative_times, 'numpy'):
            return cumulative_times.numpy()
        elif hasattr(cumulative_times, 'detach'):
            return cumulative_times.detach().cpu().numpy()
        else:
            return convert_to_numpy(cumulative_times)
    else:
        # No runtime data available
        print(f"Warning: No runtime data found for {method_name} (seed {seed}, function {fun_name})")
        return None


def generate_runtime_plot(all_results, objective, acquisition_function, save_plots=False, output_dir=None, chart_type="combined", methods=None, method_alpha=None):
    
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
        # Group results by method
        method_results = defaultdict(list)
        
        for result_dir, results in dim_results.items():
            dir_info = extract_info_from_result_dir(result_dir)
            method_name = dir_info['method']
            
            # Create run data structure
            run_data = {
                'results': results,
                'seed': dir_info['seed'],
                'fun_name': dir_info['objective']
            }
            method_results[method_name].append(run_data)
        
        if not method_results:
            print(f"No method results to plot for dimension {dim}")
            continue
        
        # Get sorted methods for consistent ordering
        sorted_methods = get_sorted_method_names(list(method_results.keys()))
        color_map = get_method_color_map(sorted_methods)
        
        # Generate plots based on chart type
        if chart_type == "combined":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [4, 1]})
        elif chart_type == "runtime":
            fig, ax1 = plt.subplots(figsize=(12, 8))
            ax2 = None
        elif chart_type == "boxplot":
            fig, ax2 = plt.subplots(figsize=(6, 8))
            ax1 = None
        else:
            raise ValueError(f"Invalid chart_type: {chart_type}. Must be 'runtime', 'boxplot', or 'combined'")
        
        # Collect final runtimes for boxplot
        final_runtimes = []
        method_labels = []
        n_seeds = 0
        
        # Plot runtime curves
        if ax1 is not None:
            for method_name in sorted_methods:
                method_data = method_results[method_name]
                
                all_runtimes = []
                max_iterations = 0
                method_final_runtimes = []
                
                # Process all runs for this method
                for run_data in method_data:
                    results = run_data['results']
                    seed = run_data['seed']
                    fun_name = run_data.get('fun_name', '')
                    
                    # Get actual runtime data
                    cumulative_runtime = get_runtime_data(results, method_name, seed, fun_name)
                    
                    # Skip this run if no runtime data is available
                    if cumulative_runtime is None:
                        continue
                        
                    all_runtimes.append(cumulative_runtime)
                    method_final_runtimes.append(cumulative_runtime[-1])
                    max_iterations = max(max_iterations, len(cumulative_runtime))
                
                if not all_runtimes:
                    print(f"Warning: No runtime data available for method {method_name}, skipping...")
                    continue
                
                # Store final runtimes for boxplot
                if method_final_runtimes:
                    final_runtimes.append(method_final_runtimes)
                    method_labels.append(method_name)
                    n_seeds = max(n_seeds, len(method_final_runtimes))
                
                # Pad shorter runs and compute statistics
                padded_runtimes = []
                for runtime in all_runtimes:
                    if len(runtime) < max_iterations:
                        # Extend with the last value (assume no more computation)
                        padded = np.concatenate([runtime, np.full(max_iterations - len(runtime), runtime[-1])])
                    else:
                        padded = runtime[:max_iterations]
                    padded_runtimes.append(padded)
                
                padded_runtimes = np.array(padded_runtimes)
                
                # Compute median and quantiles for consistency with regret plots
                median_runtime = np.median(padded_runtimes, axis=0)
                q25_runtime = np.percentile(padded_runtimes, 25, axis=0)
                q75_runtime = np.percentile(padded_runtimes, 75, axis=0)
                
                x = np.arange(1, len(median_runtime) + 1)
                color = color_map[method_name]
                display_name = get_method_display_name(method_name)
                
                # Get alpha value for this method
                alpha_value = method_alpha.get(method_name, 1.0) if method_alpha else 1.0
                
                # Plot median line
                ax1.plot(x, median_runtime, color=color, label=f"{display_name} (median runtime)", 
                        linewidth=2, alpha=alpha_value)
                
                # Add IQR shading
                shading_alpha = method_alpha.get(method_name, 0.2) if method_alpha else 0.2
                ax1.fill_between(x, q25_runtime, q75_runtime, color=color, alpha=shading_alpha)
            
            # Configure runtime curves subplot
            ax1.set_xlabel('Iteration', fontsize=14)
            ax1.set_ylabel('Cumulative Runtime (seconds)', fontsize=14)
            
            # Set title based on chart type
            objective_display_name = get_objective_display_name(objective)
            acq_display_name = acq_display_names.get(acquisition_function, acquisition_function) if acquisition_function else "All"
            if chart_type == "runtime":
                ax1.set_title(f'Cumulative Runtime - {objective_display_name} with {acq_display_name} ({n_seeds} seeds)', 
                            fontsize=16, fontweight='bold', pad=12)
            elif chart_type == "combined":
                ax1.set_title(f'Cumulative Runtime - {objective_display_name} with {acq_display_name} ({n_seeds} seeds)', 
                            fontsize=16, fontweight='bold', pad=18)
            
            ax1.grid(True, alpha=0.3)
        
        # Create boxplot
        if ax2 is not None and final_runtimes:
            # Use consistent method order - always apply the same sorting as the runtime plot
            if methods is not None:
                # Filter methods that have data, then apply proper sorting
                available_methods = [m for m in methods if m in method_labels]
                # Group and sort the available methods to match the runtime plot ordering
                method_groups = group_methods_by_type(available_methods)
                sorted_boxplot_methods = []
                for group_name, group_methods in method_groups.items():
                    sorted_boxplot_methods.extend(group_methods)
            else:
                sorted_boxplot_methods = sorted_methods
                
            # Filter and reorder data
            boxplot_data = []
            boxplot_method_names = []
            for method in sorted_boxplot_methods:
                if method in method_labels:
                    idx = method_labels.index(method)
                    boxplot_data.append(final_runtimes[idx])
                    boxplot_method_names.append(method)
            
            display_names = [get_method_display_name(m) for m in boxplot_method_names]
            
            # Create boxplot
            bp = ax2.boxplot(
                boxplot_data, 
                tick_labels=display_names if chart_type == "boxplot" else [""] * len(boxplot_data),
                patch_artist=True, 
                showmeans=False,
                meanline=False, 
                boxprops=dict(linewidth=0.), 
                medianprops=dict(visible=True, linewidth=2, linestyle='-'),
                showfliers=False
            )
            
            # Apply consistent colors and alpha values using common function
            apply_method_alpha_to_boxplot(bp, boxplot_method_names, color_map, method_alpha)
            
            # Configure boxplot
            if chart_type == "boxplot":
                ax2.set_title(f'Final Runtime Distribution - {objective_display_name} with {acq_display_name} ({n_seeds} seeds)', 
                            fontsize=16, fontweight='bold')
                ax2.set_xticklabels(display_names, rotation=45)
                ax2.set_ylabel('Final Runtime (seconds)', fontsize=14)
                ax2.tick_params(left=True, labelleft=True)
            elif chart_type == "combined":
                ax2.set_title('Final Runtime', fontweight='bold', pad=18)
                ax2.set_xticklabels([])
                ax2.set_ylabel("")
                ax2.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
            
            ax2.grid(True, alpha=0.3, axis='y')
            for spine in ax2.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
            
            # For combined charts, synchronize y-axis with runtime plot
            if chart_type == "combined" and ax1 is not None:
                # Get the y-axis limits from the runtime plot
                runtime_ylim = ax1.get_ylim()
                ax2.set_ylim(runtime_ylim)
                
        elif ax2 is not None:
            # No data to plot - show message
            ax2.text(0.5, 0.5, 'No runtime data available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes, fontsize=14)
            ax2.set_title('Final Runtime Distribution', fontsize=16, fontweight='bold')
        
        # Check if we have any data to plot
        if not final_runtimes:
            print("Warning: No runtime data available for any methods. Ensure results contain 'cumulative_times' data.")
            if ax1 is not None:
                ax1.text(0.5, 0.5, 'No runtime data available\nEnsure optimization results contain "cumulative_times" data', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax1.transAxes, fontsize=14)
                ax1.set_title('Cumulative Runtime vs Iteration', fontsize=16, fontweight='bold')
        
        # Add grouped legend below both charts for combined view
        if chart_type == "combined" and ax1 is not None:
            handles, labels = ax1.get_legend_handles_labels()
            
            # Extract method names from labels (remove " (median runtime)" suffix)
            method_labels_dict = {}
            for h, l in zip(handles, labels):
                if "(median runtime)" in l:
                    method_name = l.replace(" (median runtime)", "")
                    # Find the original method key by matching display name
                    for method_key in sorted_methods:
                        if get_method_display_name(method_key) == method_name:
                            method_labels_dict[method_key] = (h, method_name)
                            break
            
            # Group methods by type and get max methods per group
            method_groups = group_methods_by_type(list(method_labels_dict.keys()))
            max_methods_per_group = max(len(methods) for methods in method_groups.values()) if method_groups else 0
            
            # Create grouped legend using common function
            create_grouped_legend(
                fig, ax1, method_labels_dict, sorted_methods, max_methods_per_group,
                "Lines show median runtime with IQR shading"
            )
        elif chart_type == "runtime" and ax1 is not None:
            handles, labels = ax1.get_legend_handles_labels()
            if handles:
                ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        fig.tight_layout()
        
        # Adjust layout for combined plot
        if chart_type == "combined":
            fig.tight_layout(rect=[0, 0.35, 1, 1.0])
        
        # Save plot if requested
        if save_plots and output_dir:
            acq_display_name = acq_display_names.get(acquisition_function, acquisition_function or "all_acq_funcs")
            if chart_type == "combined":
                filename = f"runtime_combined_{objective}_{acq_display_name.replace(' ', '_')}.png"
            elif chart_type == "boxplot":
                filename = f"runtime_boxplot_{objective}_{acq_display_name.replace(' ', '_')}.png"
            else:
                filename = f"runtime_{objective}_{acq_display_name.replace(' ', '_')}.png"
            save_plot_figure(fig, output_dir, filename)
        
        plt.show()


def main(
    objective=None,  # Set to None to detect all objectives
    dim=None,        # Set to None to detect all dimensions
    seed=0,
    methods=None,
    acquisition_function=None,  # Set to None to detect all acquisition functions
    save_plots=False,
    output_dir=None,
    sweep="debug",
    chart_type="runtime",  # Options: "runtime", "boxplot", "combined"
    method_alpha=None,  # Dict to control alpha/transparency of specific methods e.g., {"bo_plain": 0.3, "boot_log": 0.5}
):
    # Base path for results - point directly to the debug folder where your results are stored
    base_path = os.path.join(project_root, "results", sweep)
    
    if not os.path.exists(base_path):
        print(f"Results directory not found at {base_path}")
        return

    # Create output directory if saving plots
    if save_plots:
        prefix = f"{objective}_{dim}" if objective and dim else "runtime_plots"
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
            print(f"\nGenerating runtime plot for objective: {obj}, acquisition function: {acq_func}")
            generate_runtime_plot(
                all_results=all_results, 
                objective=obj, 
                acquisition_function=acq_func, 
                save_plots=save_plots, 
                output_dir=output_dir,
                chart_type=chart_type,
                methods=methods,
                method_alpha=method_alpha
            )

    print("Runtime plotting completed!")


##################################################################
# Configuration
##################################################################

if __name__ == '__main__':
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
        objective="ackley2D",
        dim=None,
        seed=None,
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
        acquisition_function=None,
        save_plots=True,
        output_dir="figures/xseed_runtime",
        sweep="final",
        chart_type="combined",
        method_alpha={
            #"bo_plain": 0.1,
            #"boot_standardize": 0.1,
            #"bopt_standardize": 0.1,
            #"boni_plain": 0.1,
            #"boni_standardize": 0.1,
        }
    )