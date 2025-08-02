"""
Generates optimization regret analysis plots for Bayesian optimization methods.
Chart types: convergence curves with IQR bands, final performance boxplots, and combined visualizations.
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
    apply_method_alpha_to_boxplot
)
from collections import defaultdict

"""
Types of plots:
1. "regret" - Shows regret/convergence curves over iterations with IQR bands
2. "boxplot" - Shows final best-so-far values as boxplots for comparison
3. "combined" - Shows both regret curves and boxplots side by side with shared y-axis
"""

def main(
    objective=None,  # Set to None to detect all objectives
    dim=None,        # Set to None to detect all dimensions
    seed=0,
    methods=None,
    acquisition_function=None,  # Set to None to detect all acquisition functions
    save_plots=False,
    output_dir=None,
    sweep="debug",
    chart_type="regret",  # Options: "regret", "boxplot", "combined"
    method_alpha=None,  # Dict to control alpha/transparency of specific methods e.g., {"bo_plain": 0.3, "boot_log": 0.5}
):
    # Base path for results
    base_path = os.path.join(project_root, "results", sweep)
    
    if not os.path.exists(base_path):
        print(f"Results directory not found at {base_path}")
        return

    # Create output directory if saving plots
    if save_plots:
        prefix = f"{objective}_{dim}" if objective and dim else "comparison_plots"
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
            print(f"\nGenerating plot for objective: {obj}, acquisition function: {acq_func}")
            generate_plot(
                all_results=all_results, 
                objective=obj, 
                acquisition_function=acq_func, 
                save_plots=save_plots, 
                output_dir=output_dir,
                chart_type=chart_type,
                methods=methods,
                method_alpha=method_alpha
            )


def generate_plot(all_results, objective, acquisition_function, save_plots=False, output_dir=None, chart_type="regret", methods=None, method_alpha=None):
    
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
        # Prepare data for both regret and boxplot
        method_histories = defaultdict(list)
        labels_final = defaultdict(list)

        for result_dir, results in dim_results.items():
            dir_info = extract_info_from_result_dir(result_dir)
            method_name = dir_info['method']
            if "history_y" in results:
                history_y = convert_to_numpy(results["history_y"])
                method_histories[method_name].append(history_y)
        
        # Get method names and sort them
        method_names = list(method_histories.keys())
        sorted_method_names = get_sorted_method_names(method_names)
        
        # Get consistent color mapping
        color_map = get_method_color_map(sorted_method_names)

        # Prepare data for plotting
        max_len = 0
        for histories in method_histories.values():
            for history in histories:
                max_len = max(max_len, len(history))

        # Process data for each method
        method_data = {}
        for method_name in sorted_method_names:
            histories = method_histories[method_name]
            
            # Pad histories to the same length
            padded_histories = np.array([np.pad(h, (0, max_len - len(h)), mode='edge') for h in histories])
            
            # Compute best-so-far for each seed
            best_so_far_histories = []
            for h in padded_histories:
                best_so_far = np.minimum.accumulate(h)
                best_so_far_histories.append(best_so_far)
                # Store final value for boxplot
                labels_final[method_name].append(best_so_far[-1])
            
            best_so_far_histories = np.array(best_so_far_histories)
            
            # Store processed data
            method_data[method_name] = {
                'best_so_far_histories': best_so_far_histories,
                'median': np.median(best_so_far_histories, axis=0),
                'lower_bound': np.percentile(best_so_far_histories, 25, axis=0),
                'upper_bound': np.percentile(best_so_far_histories, 75, axis=0),
                'final_values': labels_final[method_name]
            }

        n_seeds = len(method_data[sorted_method_names[0]]['best_so_far_histories']) if sorted_method_names else 0
        
        # Create subplot layout based on chart type
        if chart_type == "combined":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 13), gridspec_kw={'width_ratios': [4, 1]})
        elif chart_type == "regret":
            fig, ax1 = plt.subplots(figsize=(12, 7))
            ax2 = None
        elif chart_type == "boxplot":
            fig, ax2 = plt.subplots(figsize=(6, 7))
            ax1 = None
        else:
            raise ValueError(f"Invalid chart_type: {chart_type}. Must be 'regret', 'boxplot', or 'combined'")

        # Global y-axis limits for consistency
        ymin_global, ymax_global = float("inf"), float("-inf")
        
        # Calculate global y-limits from all data
        for method_name in sorted_method_names:
            data = method_data[method_name]
            ymin = np.min(data['lower_bound'])
            ymax = np.max(data['upper_bound'])
            ymin_global = min(ymin_global, ymin)
            ymax_global = max(ymax_global, ymax)

        # Get display name for acquisition function (used in titles and filenames)
        acq_display_name = acq_display_names.get(acquisition_function, acquisition_function) if acquisition_function else "All"

        # Plot regret chart
        if ax1 is not None:
            for i, method_name in enumerate(sorted_method_names):
                data = method_data[method_name]
                color = color_map[method_name]
                display_name = get_method_display_name(method_name)

                # Get alpha values for this method (default values if not specified)
                line_alpha = method_alpha.get(method_name, 0.7) if method_alpha else 0.7
                band_alpha = method_alpha.get(method_name, 0.2) if method_alpha else 0.2

                # Plot median best-so-far line
                ax1.step(
                    np.arange(len(data['median'])),
                    data['median'],
                    color=color,
                    linewidth=2.5,
                    where="post",
                    alpha=line_alpha,
                    label=f"{display_name} (median best so far)"
                )

                # Plot IQR band around the median
                ax1.fill_between(
                    np.arange(len(data['median'])),
                    data['lower_bound'],
                    data['upper_bound'],
                    color=color,
                    alpha=band_alpha
                )

            # Set y-axis limits with padding
            if ymin_global != float("inf") and ymax_global != float("-inf"):
                padding = (ymax_global - ymin_global) * 0.05
                ax1.set_ylim(ymin_global - padding, ymax_global + padding)

            # Style regret plot
            ax1.set_xlabel("Iteration", fontweight="bold")
            ax1.set_ylabel("Objective Value", fontweight="bold")
            
            # Add title for regret plot
            objective_display_name = get_objective_display_name(objective)
            if chart_type == "regret":
                title = f"Bayesian Optimization Performance on {objective_display_name} with {acq_display_name} ({n_seeds} seeds)"
                ax1.set_title(title, fontweight="bold", pad=12)
            elif chart_type == "combined":
                # Include the main comparison info as subtitle for the left chart
                title = f"Bayesian Optimization Performance on {objective_display_name} with {acq_display_name} ({n_seeds} seeds)"
                ax1.set_title(title, fontweight="bold", pad=18)

            # Add horizontal line at y=0 if relevant
            ymin, ymax = ax1.get_ylim()
            if ymin < 0 < ymax:
                ax1.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

            # Add grid and borders
            ax1.grid(True, linestyle="--", alpha=0.6)
            for spine in ax1.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)

            # Legend for regret plot (will be handled later for combined plots)
            if chart_type == "regret":
                handles, labels = ax1.get_legend_handles_labels()
                
                # Extract method names from labels (remove " (median best so far)" suffix)
                method_labels = {}
                for h, l in zip(handles, labels):
                    if "(median best so far)" in l:
                        method_name = l.replace(" (median best so far)", "")
                        # Find the original method key by matching display name
                        for method_key in sorted_method_names:
                            if get_method_display_name(method_key) == method_name:
                                method_labels[method_key] = (h, method_name)
                                break
                
                # Group methods by type
                method_groups = group_methods_by_type(list(method_labels.keys()))
                
                # Create simple legend for standalone regret plot
                legend_handles = []
                legend_labels = []
                
                for group_name, methods in method_groups.items():
                    if methods:
                        # Add group header
                        from matplotlib.lines import Line2D
                        header_handle = Line2D([0], [0], color='none', label=f"{group_name}")
                        legend_handles.append(header_handle)
                        legend_labels.append(f"{group_name}")
                        
                        # Add methods in this group
                        for method in methods:
                            if method in method_labels:
                                handle, display_name = method_labels[method]
                                legend_handles.append(handle)
                                legend_labels.append(display_name)
                
                if legend_handles:
                    legend = ax1.legend(
                        legend_handles,
                        legend_labels,
                        loc="best",
                        frameon=True,
                        fancybox=True,
                        framealpha=0.9,
                        shadow=True
                    )
                    
                    # Style group headers
                    for i, text in enumerate(legend.get_texts()):
                        label_text = text.get_text()
                        # Check if this is a group header by checking if it's in the group names
                        method_groups = group_methods_by_type(list(method_labels.keys()))
                        if label_text in method_groups.keys():
                            text.set_fontweight('bold')
                            text.set_fontsize(text.get_fontsize() + 1)

        # Plot boxplot
        if ax2 is not None:
            # Use consistent method order - always apply the same sorting as the regret plot
            if methods is not None:
                # Filter methods that have data, then apply proper sorting
                available_methods = [m for m in methods if m in labels_final]
                # Group and sort the available methods to match the regret plot ordering
                method_groups = group_methods_by_type(available_methods)
                sorted_methods = []
                for group_name, group_methods in method_groups.items():
                    sorted_methods.extend(group_methods)
            else:
                sorted_methods = sorted_method_names

            data_final = [labels_final[m] for m in sorted_methods]
            display_names = [get_method_display_name(m) for m in sorted_methods]

            # Create boxplot
            bp = ax2.boxplot(
                data_final, 
                tick_labels=display_names if chart_type == "boxplot" else [""] * len(data_final),  # No labels for combined plot
                patch_artist=True, 
                showmeans=False,  # Hide mean
                meanline=False, 
                boxprops=dict(linewidth=0.), 
                medianprops=dict(visible=True, linewidth=2, linestyle='-'),  # Show median as solid line
                showfliers=False
            )
            
            # Apply consistent colors and alpha values using common function
            apply_method_alpha_to_boxplot(bp, sorted_methods, color_map, method_alpha)

            # Set same y-axis limits as regret plot
            if ymin_global != float("inf") and ymax_global != float("-inf"):
                padding = (ymax_global - ymin_global) * 0.05
                ax2.set_ylim(ymin_global - padding, ymax_global + padding)

            # Style boxplot
            if chart_type == "boxplot":
                # Standalone boxplot keeps labels and y-axis
                ax2.set_xticklabels(display_names, rotation=45)
                objective_display_name = get_objective_display_name(objective)
                ax2.set_title(f"Final best-so-far value - {objective_display_name} ({n_seeds} seeds)", fontweight="bold", pad=5)
                ax2.set_ylabel("Objective Value", fontweight="bold")
                ax2.tick_params(left=True, labelleft=True)
            else:
                # Combined plot: no labels, no y-axis
                ax2.set_xticklabels([])  # Remove x-axis labels
                ax2.set_ylabel("")  # No y-label for boxplot
                ax2.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)  # Hide all ticks and labels
                ax2.set_title("Final Iteration", fontweight="bold", pad=18)

            # Add grid and borders for boxplot
            ax2.grid(True, linestyle="--", alpha=0.6)
            for spine in ax2.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)

        # Add grouped legend below both charts for combined view
        if chart_type == "combined" and ax1 is not None:
            handles, labels = ax1.get_legend_handles_labels()
            
            # Extract method names from labels (remove " (median best so far)" suffix)
            method_labels = {}
            for h, l in zip(handles, labels):
                if "(median best so far)" in l:
                    method_name = l.replace(" (median best so far)", "")
                    # Find the original method key by matching display name
                    for method_key in sorted_method_names:
                        if get_method_display_name(method_key) == method_name:
                            method_labels[method_key] = (h, method_name)
                            break
            
            # Group methods by type and get max methods per group
            method_groups = group_methods_by_type(list(method_labels.keys()))
            max_methods_per_group = max(len(methods) for methods in method_groups.values()) if method_groups else 0
            
            # Create grouped legend using common function
            create_grouped_legend(
                fig, ax1, method_labels, sorted_method_names, max_methods_per_group,
                "Lines show median best-so-far values with IQR shading"
            )

        fig.tight_layout()
        
        if chart_type == "combined":
            fig.tight_layout(rect=[0, 0.35, 1, 1.0])

        if save_plots:
            if chart_type == "combined":
                filename = f"combined_{objective}_{acq_display_name.replace(' ', '_')}.png"
            elif chart_type == "boxplot":
                filename = f"boxplot_{objective}_{acq_display_name.replace(' ', '_')}.png"
            else:
                filename = f"comparison_{objective}_{acq_display_name.replace(' ', '_')}.png"
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
        objective="ackley10D",     # Set to None to auto-detect all objectives
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
            "turboni_standardize", 
            "turboni_tr", "turboni_trbinary",
            "turboni_tradditive", "turboni_tradditivenorm",
            "turboni_trbs", "turboni_trbsnorm",
        ],
        acquisition_function=None, # Set to None to auto-detect all acquisition functions
        save_plots=True,
        output_dir="figures/xseed_regret",
        sweep="final",
        chart_type="combined",  # Options: "regret", "boxplot", "combined"
        method_alpha={  # Control transparency/blending of specific methods
            # "bo_plain": 0.1,
            # "boot_log": 0.1,
            # "boot_standardize": 0.1,
            # "bopt_bilog": 0.1,
            # "bopt_log": 0.1,
            # "bopt_standardize": 0.1,
            # "boni_plain": 0.1,
            #"boni_standardize": 0.1,
            #"boni_ilsstandardize": 0.1,
        }
    )
