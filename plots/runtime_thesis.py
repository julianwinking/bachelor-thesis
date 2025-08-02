"""
Comprehensive runtime analysis module for thesis visualizations with multiple chart configurations.
Chart types: runtime curves, boxplots, dual comparisons, dual objectives, and overview grid layouts.
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
    convert_to_numpy,
    sort_methods_within_group,
    group_methods_by_type,
    load_and_filter_results,
    apply_method_alpha_to_boxplot,
    get_thesis_figure_size,
    create_thesis_legend,
    sort_objectives_by_name_and_dimension,
    create_thesis_output_dir,
    save_thesis_plot,
    setup_thesis_style,
    get_thesis_legend_handles_labels,
)

"""
Chart types:
1. "runtime" - Shows cumulative runtime curves over iterations with IQR bands
2. "boxplot" - Shows final runtime values as boxplots for comparison
3. "combined" - Shows both runtime curves and boxplots side by side
4. "dual_runtime" - Shows two runtime charts stacked vertically for two acquisition functions with shared legend
5. "dual_objective" - Shows two runtime charts side by side for two objectives with the same acquisition function
6. "overview" - Shows overview charts with 2 columns (acquisition functions) and rows for each objective
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
    chart_type="runtime",  # Options: "runtime", "boxplot", "combined", "dual_runtime", "dual_objective", "overview"
    method_alpha=None,  # Dict to control alpha/transparency of specific methods
    dual_acquisition_functions=None,  # List of two acquisition functions for dual_runtime type
    dual_objectives=None,  # List of two objectives for dual_objective type
    overview_acquisition_functions=None,  # List of two acquisition functions for overview charts
    method_column_order=None,  # Dict mapping column names to lists of methods for custom legend ordering
):
    # Base path for results
    base_path = os.path.join(project_root, "results", sweep)
    
    if not os.path.exists(base_path):
        print(f"Results directory not found at {base_path}")
        return

    # Create output directory if saving plots (thesis directory)
    if save_plots:
        output_dir = create_thesis_output_dir(output_dir, "thesis/runtime")

    # Load and filter results using common function
    results = load_and_filter_results(
        base_path, objective, dim, seed, acquisition_function, methods
    )
    
    if results[0] is None:
        return
        
    all_results, all_objectives, all_dimensions, all_acquisition_functions = results
    
    print(f"Found acquisition functions: {all_acquisition_functions}")
    print(f"Found objectives: {all_objectives}")
    print(f"Found dimensions: {all_dimensions}")
    
    # Generate thesis-style plots
    if chart_type == "overview":
        generate_overview_chart(
            all_results, all_objectives, overview_acquisition_functions or all_acquisition_functions[:2],
            save_plots, output_dir, methods, method_alpha
        )
    elif chart_type == "dual_runtime":
        generate_dual_runtime_plot(
            all_results, objective, dual_acquisition_functions or all_acquisition_functions[:2],
            save_plots, output_dir, methods, method_alpha
        )
    elif chart_type == "dual_objective":
        if len(all_objectives) != 2:
            print(f"Dual objective chart requires exactly 2 objectives, but found {len(all_objectives)}: {all_objectives}")
            return
        # Convert set to sorted list for consistent ordering by dimension (e.g., ackley2D before ackley10D)
        objectives_list = sort_objectives_by_name_and_dimension(list(all_objectives))
        generate_dual_objective_plot(
            all_results, acquisition_function, objectives_list,
            save_plots, output_dir, methods, method_alpha, method_column_order
        )
    else:
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
        
        # Get thesis figure size
        fig_width, fig_height = get_thesis_figure_size()
        
        # Generate plots based on chart type
        if chart_type == "combined":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height), gridspec_kw={'width_ratios': [4, 1]})
        elif chart_type == "runtime":
            fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))
            ax2 = None
        elif chart_type == "boxplot":
            fig, ax2 = plt.subplots(figsize=(fig_width * 0.6, fig_height))
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
                    runtime_data = get_runtime_data(run_data['results'], method_name, run_data['seed'], run_data['fun_name'])
                    if runtime_data is not None and len(runtime_data) > 0:
                        all_runtimes.append(runtime_data)
                        max_iterations = max(max_iterations, len(runtime_data))
                        method_final_runtimes.append(runtime_data[-1])
                
                if not all_runtimes:
                    continue
                
                # Store final runtimes for boxplot
                if method_final_runtimes:
                    final_runtimes.append(method_final_runtimes)
                    method_labels.append(method_name)
                    n_seeds = len(method_final_runtimes)
                
                # Pad shorter runs and compute statistics
                padded_runtimes = []
                for runtime in all_runtimes:
                    if len(runtime) < max_iterations:
                        # Pad with the last value (cumulative time doesn't decrease)
                        padded_runtime = np.pad(runtime, (0, max_iterations - len(runtime)), mode='edge')
                    else:
                        padded_runtime = runtime
                    padded_runtimes.append(padded_runtime)
                
                padded_runtimes = np.array(padded_runtimes)
                
                # Compute median and quantiles for consistency with regret plots
                median_runtime = np.median(padded_runtimes, axis=0)
                q25_runtime = np.percentile(padded_runtimes, 25, axis=0)
                q75_runtime = np.percentile(padded_runtimes, 75, axis=0)
                
                # Get method color and alpha
                color = color_map[method_name]
                alpha = method_alpha.get(method_name, 1.0) if method_alpha else 1.0
                
                # Plot median line
                iterations = np.arange(len(median_runtime))
                display_name = get_method_display_name(method_name)
                line = ax1.plot(iterations, median_runtime, color=color, alpha=alpha, 
                               label=f"{display_name} (median runtime)", linewidth=2)[0]
                
                # Plot IQR band
                ax1.fill_between(iterations, q25_runtime, q75_runtime, 
                               color=color, alpha=alpha*0.3)
            
            # Configure runtime curves subplot
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('s')
            
            # Set title based on chart type
            objective_display_name = get_objective_display_name(objective)
            if chart_type == "runtime":
                ax1.set_title(f'{objective_display_name}')
            elif chart_type == "combined":
                ax1.set_title(f'{objective_display_name}')
            
            ax1.grid(True, alpha=0.3)
            if len(median_runtime) > 0:
                ax1.set_xlim(0, len(median_runtime)-1)
        
        # Create boxplot
        if ax2 is not None and final_runtimes:
            if methods is not None:
                # Filter methods that have data, then apply proper sorting
                available_methods = [m for m in methods if m in method_labels]
                # Group and sort the available methods to match the runtime plot ordering
                method_groups = group_methods_by_type(available_methods)
                sorted_boxplot_methods = []
                for group_name, group_methods in method_groups.items():
                    sorted_boxplot_methods.extend(sort_methods_within_group(group_methods))
            else:
                sorted_boxplot_methods = sorted_methods
                
            # Filter and reorder data
            boxplot_data = []
            boxplot_method_names = []
            for method in sorted_boxplot_methods:
                if method in method_labels:
                    method_idx = method_labels.index(method)
                    boxplot_data.append(final_runtimes[method_idx])
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
                ax2.set_title(f'{objective_display_name}')
                ax2.set_xticklabels(display_names, rotation=45)
                ax2.set_ylabel('Final Runtime (s)')
                ax2.tick_params(left=True, labelleft=True)
            elif chart_type == "combined":
                ax2.set_title('Final Runtime')
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
            ax2.text(0.5, 0.5, 'No runtime data available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes)
            ax2.set_title('Final Runtime Distribution')
        
        # Check if we have any data to plot
        if not final_runtimes:
            print("Warning: No runtime data available for any methods. Ensure results contain 'cumulative_times' data.")
            if ax1 is not None:
                ax1.text(0.5, 0.5, 'No runtime data available\nEnsure optimization results contain "cumulative_times" data', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax1.transAxes)
                ax1.set_title('Cumulative Runtime vs Iteration')
        
        # Add thesis-style legend below both charts for combined view
        if chart_type == "combined" and ax1 is not None:
            method_labels_dict = get_thesis_legend_handles_labels(ax1, sorted_methods, " (median runtime)")
            
            # Create thesis legend using common function
            create_thesis_legend(fig, ax1, method_labels_dict, sorted_methods)
        elif chart_type == "runtime" and ax1 is not None:
            method_labels_dict = get_thesis_legend_handles_labels(ax1, sorted_methods, " (median runtime)")
            if method_labels_dict:
                create_thesis_legend(fig, ax1, method_labels_dict, sorted_methods)
        
        # Adjust layout for thesis style
        if chart_type == "combined":
            plt.tight_layout(rect=[0, 0.25, 1, 1.0])
        else:
            plt.tight_layout(rect=[0, 0.2, 1, 1.0])
        
        # Save plot if requested (PDF format for thesis)
        if save_plots and output_dir:
            acq_display_name = acq_display_names.get(acquisition_function, acquisition_function or "all_acq_funcs")
            if chart_type == "combined":
                filename = f"runtime_combined_{objective}_{acq_display_name.replace(' ', '_')}.pdf"
            elif chart_type == "boxplot":
                filename = f"runtime_boxplot_{objective}_{acq_display_name.replace(' ', '_')}.pdf"
            else:
                filename = f"runtime_{objective}_{acq_display_name.replace(' ', '_')}.pdf"
            save_thesis_plot(fig, output_dir, filename)
        
        plt.show()


def generate_dual_runtime_plot(all_results, objective, acquisition_functions, save_plots=False, output_dir=None, methods=None, method_alpha=None):
    
    # Prepare data for both acquisition functions
    acq_data = {}
    all_sorted_method_names = None
    
    for acq_func in acquisition_functions:
        runtime_data = get_overview_runtime_data(all_results, objective, acq_func, methods)
        if runtime_data:
            acq_data[acq_func] = runtime_data
            if all_sorted_method_names is None:
                all_sorted_method_names = get_sorted_method_names(list(runtime_data.keys()))
    
    if not acq_data:
        print(f"No runtime data found for objective: {objective} with acquisition functions: {acquisition_functions}")
        return
    
    # Calculate global y-limits across both acquisition functions
    ymin_global, ymax_global = float("inf"), float("-inf")
    for acq_func, runtime_data in acq_data.items():
        for method_name, data in runtime_data.items():
            ymin = np.min(data['lower_bound'])
            ymax = np.max(data['upper_bound'])
            ymin_global = min(ymin_global, ymin)
            ymax_global = max(ymax_global, ymax)
    
    # Get thesis figure size
    fig_width, fig_height = get_thesis_figure_size()
    
    # Create two subplots stacked vertically, each with runtime plot + boxplot
    fig = plt.figure(figsize=(fig_width, fig_height * 1.5))
    
    # Create grid: 2 rows, each row has runtime plot + boxplot
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], hspace=0.1, wspace=0.1)
    
    # First acquisition function (top row)
    ax1_regret = fig.add_subplot(gs[0, 0])
    ax1_box = fig.add_subplot(gs[0, 1])
    
    # Second acquisition function (bottom row) - share x-axis with top row
    ax2_regret = fig.add_subplot(gs[1, 0], sharex=ax1_regret)
    ax2_box = fig.add_subplot(gs[1, 1], sharey=ax1_box)
    
    axes_pairs = [(ax1_regret, ax1_box), (ax2_regret, ax2_box)]
    
    # Plot each acquisition function
    for i, (acq_func, (ax_regret, ax_box)) in enumerate(zip(acquisition_functions, axes_pairs)):
        runtime_data = acq_data[acq_func]
        
        plot_overview_runtime_subplot(ax_regret, runtime_data, method_alpha)
        
        # Configure subplot
        ax_regret.set_ylabel('Runtime (s)')
        if i == len(acquisition_functions) - 1:  # Bottom subplot
            ax_regret.set_xlabel('Iteration')
        else:  # Top subplot
            ax_regret.set_xticklabels([])
        
        # Set title
        objective_display_name = get_objective_display_name(objective)
        ax_regret.set_title(f'{objective_display_name}')
        
        # Set consistent y-axis limits
        ax_regret.set_ylim(ymin_global, ymax_global)
        
        ax_box.set_title('Final Runtime')
        ax_box.set_ylim(ymin_global, ymax_global)
    
    # Create shared legend using the first subplot's handles and labels
    method_labels_dict = get_thesis_legend_handles_labels(ax1_regret, all_sorted_method_names, " (median runtime)")
    
    # Create shared legend below both plots
    create_thesis_legend(fig, ax2_regret, method_labels_dict, all_sorted_method_names)
    
    plt.subplots_adjust(bottom=0.3)
    
    if save_plots:
        filename = f"runtime_dual_{objective}_{acquisition_functions[0]}_{acquisition_functions[1]}.pdf"
        save_thesis_plot(fig, output_dir, filename)
    
    plt.show()


def generate_dual_objective_plot(all_results, acquisition_function, objectives, save_plots=False, output_dir=None, methods=None, method_alpha=None, method_column_order=None):
    
    # Prepare data for both objectives
    obj_data = {}
    all_sorted_method_names = None
    
    for objective in objectives:
        runtime_data = get_overview_runtime_data(all_results, objective, acquisition_function, methods)
        if runtime_data:
            obj_data[objective] = runtime_data
            if all_sorted_method_names is None:
                all_sorted_method_names = get_sorted_method_names(list(runtime_data.keys()))
    
    if not obj_data:
        print(f"No runtime data found for objectives: {objectives} with acquisition function: {acquisition_function}")
        return
    
    # Get thesis figure size
    fig_width, fig_height = get_thesis_figure_size()
    
    # Create two subplots side by side with independent y-axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width*1.1, fig_height*0.7))
    
    axes = [ax1, ax2]
    
    # Plot each objective
    for i, (objective, ax) in enumerate(zip(objectives, axes)):
        runtime_data = obj_data[objective]
        
        plot_overview_runtime_subplot(ax, runtime_data, method_alpha)
        
        # Configure subplot
        ax.set_xlabel('Iteration')
        if i == 0:  # Left subplot
            ax.set_ylabel('s')

        # Set title
        objective_display_name = get_objective_display_name(objective)
        ax.set_title(f'{objective_display_name}')
        
        # Set logarithmic y-axis for better visualization of runtime data
        ax.set_yscale('log')
        
        # Set minimum y-limit to avoid showing very small values that cause step-like behavior
        # This prevents the sharp rise from very small initial runtimes
        current_ylim = ax.get_ylim()
        min_runtime = max(1, current_ylim[0])
        ax.set_ylim(bottom=min_runtime)
        
        if runtime_data:
            # Find the maximum number of iterations from the data
            max_iterations = 0
            for method_data in runtime_data.values():
                max_iterations = max(max_iterations, len(method_data['median']))
            if max_iterations > 0:
                ax.set_xlim(0, max_iterations - 1)
    
    # Create shared legend using the first subplot's handles and labels
    method_labels_dict = get_thesis_legend_handles_labels(ax1, all_sorted_method_names, " (median runtime)")
    
    # Create shared legend below both plots
    if all_sorted_method_names:
        if method_column_order is not None:
            from matplotlib.lines import Line2D
            column_names = list(method_column_order.keys())
            num_columns = len(column_names)
            if num_columns > 0:
                all_handles = []
                all_labels = []
                max_methods_per_column = max(len(methods) for methods in method_column_order.values())
                for col, column_name in enumerate(column_names):
                    methods_in_column = method_column_order[column_name]
                    for method in methods_in_column:
                        if method in method_labels_dict:
                            handle, display_name = method_labels_dict[method]
                            all_handles.append(handle)
                            all_labels.append(display_name)
                    methods_in_this_column = len(methods_in_column)
                    padding_needed = max_methods_per_column - methods_in_this_column
                    for _ in range(padding_needed):
                        all_handles.append(Line2D([0], [0], alpha=0))
                        all_labels.append("")
                legend = fig.legend(
                    all_handles,
                    all_labels,
                    loc='lower center',
                    bbox_to_anchor=(0.5, 0.02),
                    ncol=num_columns,
                    frameon=False,
                    columnspacing=2.0,
                    handletextpad=0.5
                )
        else:
            # Create the unified legend with 2 columns for dual objective view
            handles = []
            labels = []
            for method in all_sorted_method_names:
                if method in method_labels_dict:
                    h, l = method_labels_dict[method]
                    handles.append(h)
                    labels.append(l)
            
            # Create legend with exactly 2 columns
            legend = fig.legend(
                handles,
                labels,
                loc='lower center',
                bbox_to_anchor=(0.5, 0.02),
                ncol=2,  # Force exactly 2 columns
                frameon=False,
                columnspacing=2.0,
                handletextpad=0.5
            )
    
    plt.tight_layout(rect=[0, 0.25, 1, 1.0])
    
    if save_plots:
        filename = f"runtime_dual_objective_{objectives[0]}_{objectives[1]}_{acquisition_function}.pdf"
        save_thesis_plot(fig, output_dir, filename)
    
    plt.show()


def generate_overview_chart(all_results, objectives, acquisition_functions, 
                          save_plots=False, output_dir=None, methods=None, method_alpha=None):
    """Generate big overview chart with 2 columns (acquisition functions) and rows for each objective."""
    
    if len(acquisition_functions) != 2:
        print("Overview chart requires exactly 2 acquisition functions")
        return
        
    if len(objectives) == 0:
        print("No objectives to plot")
        return
    
    # Get thesis figure size and scale for overview
    base_fig_width, base_fig_height = get_thesis_figure_size()
    
    # Calculate figure size for overview (2 columns, n rows)
    n_rows = len(objectives)
    n_cols = 2
    
    # Scale figure size appropriately
    fig_width = base_fig_width
    fig_height = base_fig_height * n_rows * 0.5
    
    # Create subplot grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    
    # Handle case where we have only one row
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Store all method labels for unified legend
    all_method_labels = {}
    
    # Store y-axis limits for each row to ensure consistent scaling
    row_ylims = {}
    
    # Process each objective (row) and acquisition function (column)
    for row_idx, objective in enumerate(objectives):
        row_ylims[row_idx] = [float("inf"), float("-inf")]
        
        for col_idx, acq_func in enumerate(acquisition_functions):
            ax = axes[row_idx, col_idx]
            
            # Get runtime data for this objective and acquisition function
            runtime_data = get_overview_runtime_data(all_results, objective, acq_func, methods)
            
            if runtime_data:
                # Plot runtime data and collect method labels
                method_labels = plot_overview_runtime_subplot(ax, runtime_data, method_alpha)
                all_method_labels.update(method_labels)
                
                # Update y-axis limits for this row
                for method_data in runtime_data.values():
                    ymin = np.min(method_data['lower_bound'])
                    ymax = np.max(method_data['upper_bound'])
                    row_ylims[row_idx][0] = min(row_ylims[row_idx][0], ymin)
                    row_ylims[row_idx][1] = max(row_ylims[row_idx][1], ymax)
            
            # Configure subplot
            if row_idx == n_rows - 1:  # Bottom row
                ax.set_xlabel('Iteration')
            if col_idx == 0:  # Left column
                ax.set_ylabel('Runtime (s)')
            
            # Set titles
            if row_idx == 0:  # Top row
                acq_display_name = get_acquisition_function_display_names().get(acq_func, acq_func)
                ax.set_title(acq_display_name)
            
            if col_idx == 0:  # Left column
                objective_display_name = get_objective_display_name(objective)
                ax.text(-0.1, 0.5, objective_display_name, transform=ax.transAxes, 
                       rotation=90, verticalalignment='center', fontweight='bold')
    
    # Apply consistent y-axis limits within each row
    for row_idx in range(n_rows):
        if row_ylims[row_idx][0] != float("inf"):
            for col_idx in range(n_cols):
                axes[row_idx, col_idx].set_ylim(row_ylims[row_idx])
    
    # Create unified legend below all charts
    if all_method_labels:
        sorted_methods = get_sorted_method_names(list(all_method_labels.keys()))
        create_thesis_legend(fig, axes[-1, -1], all_method_labels, sorted_methods)
    
    # Adjust layout with reduced margins and closer legend
    plt.tight_layout(rect=[0.1, 0.07, 1, 0.95])  # Reduced bottom margin to bring legend closer
    
    # Save plot if requested
    if save_plots:
        filename = f"runtime_overview_{acquisition_functions[0]}_{acquisition_functions[1]}.pdf"
        save_thesis_plot(fig, output_dir, filename)
        
    plt.show()


def get_overview_runtime_data(all_results, objective, acquisition_function, methods):
    # Filter results for the specified objective and acquisition function
    filtered_results = filter_results_by_criteria(
        all_results, 
        objective=objective,
        acquisition_function=acquisition_function
    )
    
    if not filtered_results:
        return None
    
    # Group results by dimension
    dimension_groups = group_results_by_dimension(filtered_results)
    
    # Get data from the first (and likely only) dimension group
    if not dimension_groups:
        return None
        
    dim, dim_results = next(iter(dimension_groups.items()))
    
    # Extract runtime data using the same logic as the main plotting function
    method_histories = defaultdict(list)
    
    for result_dir, results in dim_results.items():
        dir_info = extract_info_from_result_dir(result_dir)
        method_name = dir_info['method']
        runtime_data = get_runtime_data(results, method_name, dir_info['seed'], dir_info['objective'])
        if runtime_data is not None:
            method_histories[method_name].append(runtime_data)
    
    if not method_histories:
        return None
    
    # Get sorted method names for consistent ordering
    sorted_method_names = get_sorted_method_names(list(method_histories.keys()))
    color_map = get_method_color_map(sorted_method_names)
    
    # Prepare data for plotting (same as main function)
    max_len = 0
    for histories in method_histories.values():
        for history in histories:
            max_len = max(max_len, len(history))
    
    # Process data for each method
    method_data = {}
    for method_name in sorted_method_names:
        histories = method_histories[method_name]
        
        # Pad histories to the same length
        padded_histories = []
        for h in histories:
            if len(h) < max_len:
                padded = np.pad(h, (0, max_len - len(h)), mode='edge')
            else:
                padded = h
            padded_histories.append(padded)
        
        padded_histories = np.array(padded_histories)
        
        # Store processed data
        method_data[method_name] = {
            'histories': padded_histories,
            'median': np.median(padded_histories, axis=0),
            'lower_bound': np.percentile(padded_histories, 25, axis=0),
            'upper_bound': np.percentile(padded_histories, 75, axis=0),
            'final_values': [h[-1] for h in histories]
        }
    
    return method_data


def plot_overview_runtime_subplot(ax, runtime_data, method_alpha):
    
    method_labels = {}
    
    # Get method names and sort them
    sorted_method_names = get_sorted_method_names(list(runtime_data.keys()))
    color_map = get_method_color_map(sorted_method_names)
    
    for method_name in sorted_method_names:
        data = runtime_data[method_name]
        color = color_map[method_name]
        
        # Get alpha value for this method
        alpha_value = method_alpha.get(method_name, 1.0) if method_alpha else 1.0
        
        # Apply minimum runtime threshold to prevent step-like behavior in log scale
        # This ensures smoother curves by avoiding very small initial values
        min_runtime_threshold = 0.001  # 1 millisecond minimum
        median_runtime = np.maximum(data['median'], min_runtime_threshold)
        lower_bound = np.maximum(data['lower_bound'], min_runtime_threshold)
        upper_bound = np.maximum(data['upper_bound'], min_runtime_threshold)
        
        # Plot median line
        iterations = np.arange(len(median_runtime))
        display_name = get_method_display_name(method_name)
        line = ax.plot(iterations, median_runtime, color=color, alpha=alpha_value,
                      label=f"{display_name} (median runtime)", linewidth=1.5)[0]
        
        # Plot IQR band
        ax.fill_between(iterations, lower_bound, upper_bound, 
                       color=color, alpha=alpha_value*0.3)
        
        # Store for legend
        method_labels[method_name] = (line, display_name)
    
    return method_labels

##################################################################
# Configuration
##################################################################

if __name__ == "__main__":
    project_root = setup_project_path()
    
    # Apply thesis style
    setup_thesis_style()

    main(
        objective=["ackley2D", "ackley10D"],  # For dual_objective: specify only the objectives we need to optimize loading
        dim=None,                  # Set to None to auto-detect all dimensions
        seed=None,                 # Set to None to auto-detect all seeds
        methods=[
            #"bo_plain", 
            "bopt_standardize", #"boot_standardize",
            #"boot_log", "bopt_log",
            #"bopt_bilog",
            #"boni_plainnonoise",# "boni_plain",
            "boni_standardizenonoise",#"boni_standardize", 
            "boni_standardizenonoiserefit",
            #"boni_standardizeones", "boni_standardizezeros",
            "boni_ilsstandardizenonoise",#"boni_ilsstandardize"
            #"boni_ilsstandardiznonoiserefit",
            #"boni_bsnoise",
            "boni_bsnonoise",  
            "boni_standardizegradient", #"boni_standardizegradientbinary",
            # "turbo_plain", 
            # "turbo_standardize",
            # "turboni_standardize", 
            # "turboni_tr", "turboni_trbinary",
            # "turboni_tradditivenorm", #"turboni_tradditive",
            # "turboni_trbsnorm", #"turboni_trbs",
        ],
        acquisition_function="UpperConfidenceBound", # For dual_objective: set specific acquisition function to optimize loading
        save_plots=True,
        output_dir="figures/thesis/runtime/",
        sweep="final",
        chart_type="dual_objective",  # Options: "runtime", "boxplot", "combined", "dual_runtime", "overview", "dual_objective"
        # To generate overview charts, set chart_type="overview" and specify two acquisition functions in overview_acquisition_functions
        # To generate dual objective charts, set chart_type="dual_objective" and specify objectives in the objective parameter
        method_alpha={  # Control transparency/blending of specific methods
            # Add method-specific alpha values if needed
        },
        method_column_order={  # Control method ordering in columns for dual objective charts
            "Column1": ["bopt_standardize", "boni_standardizenonoise", "boni_standardizenonoiserefit"],
            "Column2": ["boni_ilsstandardizenonoise", "boni_bsnonoise", "boni_standardizegradient"],
            # Example: "Column1": ["boni_standardizenonoise"],
            #          "Column2": ["boni_ilsstandardizenonoise", "boni_bsnonoise"],
            #          "Column3": ["boni_standardizegradient", "boni_standardizegradientbinary"],
        }
    )