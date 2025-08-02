"""
Comprehensive regret visualization module for thesis plots with multiple chart configurations.
Chart types: regret curves, boxplots, combined plots, dual acquisition function comparisons, and overview grids.
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
    get_method_display_name,
    get_objective_display_name,
    get_sorted_method_names,
    filter_results_by_criteria,
    group_results_by_dimension,
    convert_to_numpy,
    group_methods_by_type,
    load_and_filter_results,
    get_thesis_figure_size,
    create_thesis_legend,
    setup_thesis_style,
    sort_objectives_by_name_and_dimension,
    save_thesis_plot,
)
from collections import defaultdict

"""
Chart types:
1. "regret" - Shows regret/convergence curves over iterations with IQR bands
2. "boxplot" - Shows final best-so-far values as boxplots for comparison
3. "combined" - Shows both regret curves and boxplots side by side
4. "dual_regret" - Shows two regret charts stacked vertically for two acquisition functions with shared legend
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
    chart_type="regret",  # Options: "regret", "boxplot", "combined", "dual_regret", "overview"
    method_alpha=None,  # Dict to control alpha/transparency of specific methods
    dual_acquisition_functions=None,  # List of two acquisition functions for dual_regret type
    overview_acquisition_functions=None,  # List of two acquisition functions for overview charts
    independent_boxplot_axes=False,  # For dual_regret: whether boxplots should have independent y-axes
    method_column_order=None,  # Dict mapping column names to lists of methods for custom legend ordering
    log_y_axis=False,  # Enable logarithmic scaling for y-axis in regret charts (boxplots always use linear scaling)
):
    # Base path for results
    base_path = os.path.join(project_root, "results", sweep)
    
    if not os.path.exists(base_path):
        print(f"Results directory not found at {base_path}")
        return

    # Create output directory if saving plots (thesis directory)
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)

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
    
    # Generate thesis-style plots
    if chart_type == "overview":
        # Handle overview charts
        if overview_acquisition_functions is None or len(overview_acquisition_functions) != 2:
            print("Error: overview chart type requires exactly two acquisition functions in overview_acquisition_functions parameter")
            return
        
        # Sort objectives by function and dimension for consistent ordering
        objectives_to_plot = [objective] if objective else all_objectives
        objectives_to_plot = sort_objectives_by_name_and_dimension(objectives_to_plot)
        
        print(f"\nGenerating regret overview chart")
        print(f"Objectives (rows): {objectives_to_plot}")
        print(f"Acquisition functions (columns): {overview_acquisition_functions}")
        
        generate_overview_chart(
            all_results, 
            objectives_to_plot, 
            acquisition_functions=overview_acquisition_functions,
            save_plots=save_plots, 
            output_dir=output_dir,
            methods=methods,
            method_alpha=method_alpha,
            method_column_order=method_column_order,
            log_y_axis=log_y_axis
        )
        
    elif chart_type == "dual_regret":
        # Handle dual regret plot for two acquisition functions
        if dual_acquisition_functions is None or len(dual_acquisition_functions) != 2:
            print("Error: dual_regret chart type requires exactly two acquisition functions in dual_acquisition_functions parameter")
            return
        
        objectives_to_plot = [objective] if objective else all_objectives
        
        for obj in objectives_to_plot:
            print(f"\nGenerating dual regret plot for objective: {obj}, acquisition functions: {dual_acquisition_functions}")
            generate_dual_regret_plot(
                all_results=all_results, 
                objective=obj, 
                acquisition_functions=dual_acquisition_functions,
                save_plots=save_plots, 
                output_dir=output_dir,
                methods=methods,
                method_alpha=method_alpha,
                independent_boxplot_axes=independent_boxplot_axes,
                method_column_order=method_column_order,
                log_y_axis=log_y_axis
            )
    else:
        # Original logic for other chart types
        objectives_to_plot = [objective] if objective else all_objectives
        acq_funcs_to_plot = [acquisition_function] if acquisition_function else all_acquisition_functions
        
        for obj in objectives_to_plot:
            for acq_func in acq_funcs_to_plot:
                print(f"\nGenerating thesis-style plot for objective: {obj}, acquisition function: {acq_func}")
                generate_plot(
                    all_results=all_results, 
                    objective=obj, 
                    acquisition_function=acq_func, 
                    save_plots=save_plots, 
                    output_dir=output_dir,
                    chart_type=chart_type,
                    methods=methods,
                    method_alpha=method_alpha,
                    independent_boxplot_axes=independent_boxplot_axes,
                    method_column_order=method_column_order,
                    log_y_axis=log_y_axis
                )


def generate_plot(all_results, objective, acquisition_function, save_plots=False, output_dir=None, chart_type="regret", methods=None, method_alpha=None, independent_boxplot_axes=False, method_column_order=None, log_y_axis=False):
    
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
        
        # Get RWTH color mapping
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
        
        # Get thesis figure size
        fig_width, fig_height = get_thesis_figure_size()
        
        # Create subplot layout based on chart type with thesis dimensions
        if chart_type == "combined":
            # Use full figure height for combined charts to accommodate legend
            if independent_boxplot_axes:
                # Increase horizontal space when boxplot has independent axes to accommodate tick labels
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(1.1*fig_width, fig_height*0.65), 
                                              gridspec_kw={'width_ratios': [4, 1], 'wspace': 0.15})
            else:
                # Use tighter spacing when boxplot shares axes
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(1.1*fig_width, fig_height*0.65), 
                                              gridspec_kw={'width_ratios': [4, 1], 'wspace': 0.05})
        elif chart_type == "regret":
            # Create figure with space for legend on the right
            fig, ax1 = plt.subplots(figsize=(fig_width, fig_height * 0.8))
            ax2 = None
        elif chart_type == "boxplot":
            # Extra height for legend below the chart
            fig, ax2 = plt.subplots(figsize=(fig_width, fig_height * 1.0))
            ax1 = None
        else:
            raise ValueError(f"Invalid chart_type: {chart_type}. Must be 'regret', 'boxplot', 'combined', or 'dual_regret'")

        # Global y-axis limits for regret plots
        ymin_global, ymax_global = float("inf"), float("-inf")
        
        # Calculate global y-limits from all data for regret plots
        for method_name in sorted_method_names:
            data = method_data[method_name]
            ymin = np.min(data['lower_bound'])
            ymax = np.max(data['upper_bound'])
            ymin_global = min(ymin_global, ymin)
            ymax_global = max(ymax_global, ymax)
        
        # Calculate y-axis limits for boxplot (either shared or independent)
        if chart_type == "combined" and independent_boxplot_axes:
            # Boxplot gets its own y-axis range based on final values
            boxplot_ymin, boxplot_ymax = float("inf"), float("-inf")
            for method_name in sorted_method_names:
                data = method_data[method_name]
                if 'final_values' in data:
                    boxplot_ymin = min(boxplot_ymin, np.min(data['final_values']))
                    boxplot_ymax = max(boxplot_ymax, np.max(data['final_values']))
            if boxplot_ymin != float("inf") and boxplot_ymax != float("-inf"):
                padding = (boxplot_ymax - boxplot_ymin) * 0.05
                boxplot_ylim = (boxplot_ymin - padding, boxplot_ymax + padding)
            else:
                boxplot_ylim = None
        else:
            # Boxplot shares the same y-axis range as regret plot
            boxplot_ylim = None

        # Get display name for acquisition function
        acq_display_name = acq_display_names.get(acquisition_function, acquisition_function) if acquisition_function else "All"

        # Plot regret chart
        if ax1 is not None:
            for i, method_name in enumerate(sorted_method_names):
                data = method_data[method_name]
                color = color_map[method_name]
                display_name = get_method_display_name(method_name)

                # Get alpha values for this method
                line_alpha = method_alpha.get(method_name, 1.0) if method_alpha else 1.0
                band_alpha = method_alpha.get(method_name, 0.3) if method_alpha else 0.3

                # Plot median best-so-far line
                ax1.plot(
                    np.arange(len(data['median'])),
                    data['median'],
                    color=color,
                    alpha=line_alpha,
                    label=display_name
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
                if log_y_axis and ymin_global > 0:
                    # Use logarithmic scaling for y-axis
                    ax1.set_yscale('log')
                    # Set log scale limits with padding
                    log_min = np.log10(ymin_global)
                    log_max = np.log10(ymax_global)
                    padding = (log_max - log_min) * 0.05
                    ax1.set_ylim(10**(log_min - padding), 10**(log_max + padding))
                else:
                    # Use linear scaling for y-axis
                    padding = (ymax_global - ymin_global) * 0.05
                    ax1.set_ylim(ymin_global - padding, ymax_global + padding)

            # Set x-axis limits with no margins (start at 0, end at max iteration) for combined chart
            if chart_type == "combined":
                max_iterations = 0
                for method_name in sorted_method_names:
                    data = method_data[method_name]
                    max_iterations = max(max_iterations, len(data['median']))
                if max_iterations > 0:
                    ax1.set_xlim(0, max_iterations)

            # Style regret plot
            ax1.set_xlabel(r'Iteration')
            ax1.set_ylabel(r'$f(\mathbf{x})$')

            # Add acquisition function label for combined charts
            if chart_type == "combined":
                if acquisition_function == "UpperConfidenceBound":
                    ax1.text(0.02, 0.12, r"\texttt{UCB}", horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes)
                elif acquisition_function == "LogExpectedImprovement":
                    ax1.text(0.02, 0.12, r"\texttt{logEI}", horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes)

        # Plot boxplot
        if ax2 is not None:
            # Use the exact same ordering as the legend will use
            if chart_type == "combined" and method_column_order is not None:
                # Use custom column ordering for combined charts to match legend
                legend_ordered_methods = []
                for col in method_column_order.values():
                    for method in col:
                        if method in labels_final:
                            legend_ordered_methods.append(method)
                sorted_methods = legend_ordered_methods
            else:
                # Use standard grouping for other chart types
                method_groups = group_methods_by_type(sorted_method_names)
                
                # Create the exact order that will appear in the legend (top to bottom within each group)
                legend_ordered_methods = []
                for group_name, group_methods in method_groups.items():
                    # Add methods in the same order they appear in the group
                    for method in group_methods:
                        if method in labels_final:
                            legend_ordered_methods.append(method)
                
                # Use this exact order for the boxplot
                sorted_methods = legend_ordered_methods
            
            data_final = [labels_final[m] for m in sorted_methods]
            display_names = [get_method_display_name(m) for m in sorted_methods]

            # Create boxplot
            bp = ax2.boxplot(
                data_final, 
                tick_labels=display_names if chart_type == "boxplot" else [""] * len(data_final),
                patch_artist=True, 
                showmeans=False,
                meanline=False, 
                boxprops=dict(linewidth=0.5), 
                medianprops=dict(visible=True, linewidth=1.0, linestyle='-', color='black'),
                showfliers=False
            )
            
            # Apply RWTH colors to boxplot
            for patch, method in zip(bp['boxes'], sorted_methods):
                patch.set_facecolor(color_map[method])
                patch.set_alpha(0.7)

            # Draw a colored square above small boxes for color visibility (for combined charts)
            if chart_type == "combined":
                y_min, y_max = ax2.get_ylim()
                y_span = y_max - y_min
                square_height = 0.06 * y_span
                threshold = 0.06 * y_span
                for patch, method in zip(bp['boxes'], sorted_methods):
                    color = color_map[method]
                    verts = patch.get_path().vertices
                    box_x = verts[:, 0]
                    box_y = verts[:, 1]
                    box_bottom = np.min(box_y)
                    box_top = np.max(box_y)
                    box_height = box_top - box_bottom
                    box_center = np.mean(box_x)
                    if box_height < threshold:
                        square_y = box_top + 0.1 * y_span
                        ax2.add_patch(
                            plt.Rectangle(
                                (box_center-0.07, square_y), 0.3, square_height,
                                facecolor=color, zorder=10
                            )
                        )

            # Set y-axis limits for boxplot
            if boxplot_ylim is not None:
                # Use independent boxplot y-axis limits
                ax2.set_ylim(boxplot_ylim)
            elif ymin_global != float("inf") and ymax_global != float("-inf"):
                # Use shared y-axis limits with regret plot
                padding = (ymax_global - ymin_global) * 0.05
                ax2.set_ylim(ymin_global - padding, ymax_global + padding)

            # Style boxplot
            if chart_type == "boxplot":
                ax2.set_xticklabels(display_names, rotation=45)
                ax2.set_ylabel(r'Objective Value')
                ax2.tick_params(left=True, labelleft=True)
                
                # Create legend for standalone boxplot (below the chart, no frame)
                legend_patches = []
                method_labels = {}
                for i, method in enumerate(sorted_methods):
                    patch = plt.Rectangle((0,0),1,1, facecolor=color_map[method], alpha=0.7)
                    display_name = get_method_display_name(method)
                    legend_patches.append(patch)
                    method_labels[method] = (patch, display_name)
                
                # Group methods by type and get max methods per group for consistent layout
                method_groups = group_methods_by_type(sorted_methods)
                
                # Create simple thesis legend using custom function
                create_thesis_legend(
                    fig, ax2, method_labels, sorted_methods
                )
                
            else:
                # Combined plot: minimal labels
                ax2.set_xticklabels([])
                ax2.set_ylabel("")
                if chart_type == "combined" and independent_boxplot_axes:
                    # Show y-axis ticks and labels for independent boxplot axes
                    ax2.tick_params(left=True, labelleft=True, bottom=False, labelbottom=False)
                else:
                    # Hide y-axis ticks and labels for shared axes
                    ax2.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

        # Add grouped legend below the chart(s)
        if ax1 is not None:
            handles, labels = ax1.get_legend_handles_labels()
            
            # Extract method names from labels and create proper method_labels dict
            method_labels = {}
            for h, l in zip(handles, labels):
                # Find the original method key by matching display name
                for method_key in sorted_method_names:
                    if get_method_display_name(method_key) == l:
                        method_labels[method_key] = (h, l)
                        break
            
            # Group methods by type and get max methods per group
            method_groups = group_methods_by_type(list(method_labels.keys()))
            
            # Create legend for combined charts with custom column ordering if specified
            if chart_type == "combined" and method_column_order is not None:
                # Use custom column ordering for combined charts
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
                    fig, ax1, method_labels, sorted_method_names
                )

        plt.tight_layout()
        
        # Handle layout properly for different chart types
        if chart_type == "combined":
            plt.subplots_adjust(bottom=0.48)
        elif chart_type == "regret":
            plt.subplots_adjust(bottom=0.25)
        elif chart_type == "boxplot":
            plt.subplots_adjust(bottom=0.25)
        # Save plot if requested
        if save_plots:
            if chart_type == "combined":
                filename = f"combined_{objective}_{acq_display_name.replace(' ', '_')}_thesis.pdf"
            elif chart_type == "boxplot":
                filename = f"boxplot_{objective}_{acq_display_name.replace(' ', '_')}_thesis.pdf"
            else:
                filename = f"regret_{objective}_{acq_display_name.replace(' ', '_')}_thesis.pdf"
            
            save_thesis_plot(fig, output_dir, filename)
            
        plt.show()


def generate_dual_regret_plot(all_results, objective, acquisition_functions, save_plots=False, output_dir=None, methods=None, method_alpha=None, independent_boxplot_axes=False, method_column_order=None, log_y_axis=False):
    
    # Get utility elements
    acq_display_names = get_acquisition_function_display_names()
    
    # Prepare data for both acquisition functions
    acq_data = {}
    all_sorted_method_names = None
    
    for acq_func in acquisition_functions:
        # Filter results for this acquisition function
        filtered_results = filter_results_by_criteria(
            all_results, 
            objective=objective,
            acquisition_function=acq_func
        )
        
        if not filtered_results:
            print(f"No results found for objective: {objective}, acquisition function: {acq_func}")
            continue
        
        # Group results by dimension (assuming single dimension for dual plot)
        dimension_groups = group_results_by_dimension(filtered_results)
        
        for dim, dim_results in dimension_groups.items():
            # Prepare data
            method_histories = defaultdict(list)
            labels_final = defaultdict(list)
            
            for result_dir, results in dim_results.items():
                dir_info = extract_info_from_result_dir(result_dir)
                method_name = dir_info['method']
                if "history_y" in results:
                    history_y = convert_to_numpy(results["history_y"])
                    method_histories[method_name].append(history_y)
            
            # Get method names and sort them (use first acq function to establish order)
            if all_sorted_method_names is None:
                method_names = list(method_histories.keys())
                all_sorted_method_names = get_sorted_method_names(method_names)
            
            # Process data for this acquisition function
            max_len = 0
            for histories in method_histories.values():
                for history in histories:
                    max_len = max(max_len, len(history))
            
            method_data = {}
            for method_name in all_sorted_method_names:
                if method_name not in method_histories:
                    continue
                    
                histories = method_histories[method_name]
                padded_histories = np.array([np.pad(h, (0, max_len - len(h)), mode='edge') for h in histories])
                
                best_so_far_histories = []
                for h in padded_histories:
                    best_so_far = np.minimum.accumulate(h)
                    best_so_far_histories.append(best_so_far)
                    # Store final value for boxplot
                    labels_final[method_name].append(best_so_far[-1])
                
                best_so_far_histories = np.array(best_so_far_histories)
                
                method_data[method_name] = {
                    'median': np.median(best_so_far_histories, axis=0),
                    'lower_bound': np.percentile(best_so_far_histories, 25, axis=0),
                    'upper_bound': np.percentile(best_so_far_histories, 75, axis=0),
                    'final_values': labels_final[method_name]
                }
            
            acq_data[acq_func] = method_data
            break  # Only handle first dimension for dual plot
    
    if not acq_data:
        print("No data found for dual regret plot")
        return
    
    # Create consistent method ordering for both legend and boxplots
    if method_column_order is not None:
        consistent_method_order = []
        for col in method_column_order.values():
            for method in col:
                if method in all_sorted_method_names:
                    consistent_method_order.append(method)
    else:
        consistent_method_order = all_sorted_method_names

    # Get color mapping
    color_map = get_method_color_map(consistent_method_order)
    
    # Calculate y-limits for regret plots across both acquisition functions
    ymin_global, ymax_global = float("inf"), float("-inf")
    for acq_func, method_data in acq_data.items():
        for method_name, data in method_data.items():
            ymin = np.min(data['lower_bound'])
            ymax = np.max(data['upper_bound'])
            ymin_global = min(ymin_global, ymin)
            ymax_global = max(ymax_global, ymax)
    
    # Calculate y-limits for boxplots (either global or per acquisition function)
    if independent_boxplot_axes:
        # Each boxplot gets its own y-axis range
        boxplot_ylims = {}
        for acq_func, method_data in acq_data.items():
            acq_ymin, acq_ymax = float("inf"), float("-inf")
            for method_name, data in method_data.items():
                if 'final_values' in data:
                    # Calculate boxplot statistics to get better y-axis limits
                    final_values = np.array(data['final_values'])
                    q1 = np.percentile(final_values, 25)
                    q3 = np.percentile(final_values, 75)
                    iqr = q3 - q1
                    lower_whisker = q1 - 1.5 * iqr
                    upper_whisker = q3 + 1.5 * iqr
                    # Use actual data min/max within whisker bounds
                    data_min = np.maximum(np.min(final_values), lower_whisker)
                    data_max = np.minimum(np.max(final_values), upper_whisker)
                    
                    acq_ymin = min(acq_ymin, data_min)
                    acq_ymax = max(acq_ymax, data_max)
            if acq_ymin != float("inf") and acq_ymax != float("-inf"):
                # Always use linear scaling for boxplot y-axis limits
                data_range = acq_ymax - acq_ymin
                if data_range > 0:
                    padding = data_range * 0.15
                else:
                    padding = max(acq_ymax * 0.1, 0.1)
                boxplot_ylims[acq_func] = (acq_ymin - padding, acq_ymax + padding)
    else:
        # All boxplots share the same y-axis range
        boxplot_ymin_global, boxplot_ymax_global = float("inf"), float("-inf")
        for acq_func, method_data in acq_data.items():
            for method_name, data in method_data.items():
                if 'final_values' in data:
                    boxplot_ymin_global = min(boxplot_ymin_global, np.min(data['final_values']))
                    boxplot_ymax_global = max(boxplot_ymax_global, np.max(data['final_values']))
        if boxplot_ymin_global != float("inf") and boxplot_ymax_global != float("-inf"):
            # Always use linear scaling for shared boxplot y-axis limits
            padding = (boxplot_ymax_global - boxplot_ymin_global) * 0.05
            shared_limits = (boxplot_ymin_global - padding, boxplot_ymax_global + padding)
            
            boxplot_ylims = {
                acq_func: shared_limits
                for acq_func in acquisition_functions
            }
    
    # Get thesis figure size
    fig_width, fig_height = get_thesis_figure_size()
    
    # Create two subplots stacked vertically, each with regret + boxplot
    fig = plt.figure(figsize=(1.1*fig_width, fig_height * 1.0))
    
    # Create grid: 2 rows, each row has regret plot + boxplot
    if independent_boxplot_axes:
        # Increase horizontal space when boxplots have independent axes to accommodate tick labels
        gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], hspace=0.1, wspace=0.2)
    else:
        # Use tighter spacing when boxplots share axes
        gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], hspace=0.1, wspace=0.05)
    
    # First acquisition function (top row)
    ax1_regret = fig.add_subplot(gs[0, 0])
    ax1_box = fig.add_subplot(gs[0, 1])
    
    # Second acquisition function (bottom row) - share x-axis with top row
    if independent_boxplot_axes:
        # Don't share y-axis for boxplots when using independent axes
        ax2_regret = fig.add_subplot(gs[1, 0], sharex=ax1_regret)
        ax2_box = fig.add_subplot(gs[1, 1])
    else:
        # Share y-axis for boxplots when using shared axes
        ax2_regret = fig.add_subplot(gs[1, 0], sharex=ax1_regret)
        ax2_box = fig.add_subplot(gs[1, 1], sharey=ax1_box)
    
    axes_pairs = [(ax1_regret, ax1_box), (ax2_regret, ax2_box)]
    
    # Plot each acquisition function
    for i, (acq_func, (ax_regret, ax_box)) in enumerate(zip(acquisition_functions, axes_pairs)):
        if acq_func not in acq_data:
            continue
        
        method_data = acq_data[acq_func]
        
        # Plot regret curves in consistent order
        for method_name in consistent_method_order:
            if method_name not in method_data:
                continue
                
            data = method_data[method_name]
            color = color_map[method_name]
            display_name = get_method_display_name(method_name)
            
            # Get alpha values
            line_alpha = method_alpha.get(method_name, 1.0) if method_alpha else 1.0
            band_alpha = method_alpha.get(method_name, 0.3) if method_alpha else 0.3
            
            # Plot median line (only add label for first subplot to avoid duplicate legend entries)
            label = display_name if i == 0 else None
            ax_regret.plot(
                np.arange(len(data['median'])),
                data['median'],
                color=color,
                alpha=line_alpha,
                label=label
            )
            
            # Plot IQR band
            ax_regret.fill_between(
                np.arange(len(data['median'])),
                data['lower_bound'],
                data['upper_bound'],
                color=color,
                alpha=band_alpha
            )
        
        # Set y-axis limits with padding for regret plot
        if ymin_global != float("inf") and ymax_global != float("-inf"):
            if log_y_axis and ymin_global > 0:
                # Use logarithmic scaling for y-axis
                ax_regret.set_yscale('log')
                # Set log scale limits with padding
                log_min = np.log10(ymin_global)
                log_max = np.log10(ymax_global)
                padding = (log_max - log_min) * 0.05
                ax_regret.set_ylim(10**(log_min - padding), 10**(log_max + padding))
            else:
                # Use linear scaling for y-axis
                padding = (ymax_global - ymin_global) * 0.05
                ax_regret.set_ylim(ymin_global - padding, ymax_global + padding)
        
        # Set y-axis limits for boxplot (either independent or shared)
        if acq_func in boxplot_ylims:
            # Always use linear scaling for boxplots (no log scale)
            ax_box.set_ylim(boxplot_ylims[acq_func])
        
        # Set x-axis limits with no margins (start at 0, end at max iteration)
        max_iterations = max(len(data['median']) for data in method_data.values() if 'median' in data)
        ax_regret.set_xlim(0, max_iterations - 1)
        
        # Set x-axis ticks to include first, last, and every X iteration
        if i == 1:  # Only set x-ticks for bottom subplot (since x-axis is shared)
            x_ticks = [0]
            if max_iterations > 1:
                # Add every 100th iteration (excluding 0)
                x_ticks.extend([k for k in range(100, max_iterations, 100)])
            x_ticks.append(max_iterations)  # Always include the last iteration
            ax_regret.set_xticks(x_ticks)
        
        # Set y-axis ticks - use automatic ticks for log scale, fixed ticks for linear scale
        if log_y_axis and ymin_global > 0:
            # Let matplotlib automatically set appropriate ticks for log scale
            pass
        else:
            # Use fixed ticks for linear scale
            ax_regret.set_yticks([0, 5, 10, 15, 20])
        
        # Only show x-axis labels on bottom subplot
        if i == 0:  # Top subplot
            ax_regret.tick_params(bottom=False, labelbottom=False)  # Hide x-axis ticks and labels
        else:  # Bottom subplot
            ax_regret.set_xlabel(r'Iteration')
            ax_regret.tick_params(bottom=True, labelbottom=True)  # Ensure x-axis ticks and labels are shown

        ax_regret.set_ylabel(r'$f(\mathbf{x})$')
        
        # Add acquisition function label
        if acq_func == "UpperConfidenceBound":
            ax_regret.text(0.02, 0.12, r"\texttt{UCB}", horizontalalignment='left', verticalalignment='center', transform=ax_regret.transAxes)
        elif acq_func == "LogExpectedImprovement":
            ax_regret.text(0.02, 0.12, r"\texttt{logEI}", horizontalalignment='left', verticalalignment='center', transform=ax_regret.transAxes)
        
        # Plot boxplot for this acquisition function
        # Use the exact same ordering as the legend will use
        legend_ordered_methods = [m for m in consistent_method_order if m in method_data and 'final_values' in method_data[m]]
        data_final = [method_data[m]['final_values'] for m in legend_ordered_methods if m in method_data]
        
        if data_final:  # Only create boxplot if we have data
            # Create boxplot
            bp = ax_box.boxplot(
                data_final, 
                tick_labels=[""] * len(data_final),  # No labels for dual plot
                patch_artist=True, 
                showmeans=False,
                meanline=False, 
                boxprops=dict(linewidth=0.5), 
                medianprops=dict(visible=True, linewidth=1.0, linestyle='-', color='black'),
                showfliers=False
            )
            
            # Apply colors to boxplot
            for patch, method in zip(bp['boxes'], legend_ordered_methods):
                if method in color_map:
                    patch.set_facecolor(color_map[method])
                    patch.set_alpha(0.7)
            
            # Draw a colored square above small boxes for color visibility
            y_min, y_max = ax_box.get_ylim()
            y_span = y_max - y_min
            square_height = 0.045 * y_span
            threshold = 0.06 * y_span
            for patch, method in zip(bp['boxes'], legend_ordered_methods):
                color = color_map[method]
                verts = patch.get_path().vertices
                box_x = verts[:, 0]
                box_y = verts[:, 1]
                box_bottom = np.min(box_y)
                box_top = np.max(box_y)
                box_height = box_top - box_bottom
                box_center = np.mean(box_x)
                if box_height < threshold:
                    square_y = box_top + 0.15 * y_span
                    ax_box.add_patch(
                        plt.Rectangle(
                            (box_center-0.06, square_y), 0.3, square_height,
                            facecolor=color, zorder=10
                        )
                    )
        
        # Style boxplot
        ax_box.set_xticklabels([])
        ax_box.set_ylabel("")
        if independent_boxplot_axes:
            # Show y-axis ticks and labels for independent axes
            ax_box.tick_params(left=True, labelleft=True, bottom=False, labelbottom=False)
        else:
            # Hide y-axis ticks and labels for shared axes
            ax_box.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    
    # Create shared legend using the first subplot's handles and labels
    handles, labels = ax1_regret.get_legend_handles_labels()
    
    # Extract method names from labels and create proper method_labels dict
    method_labels = {}
    for h, l in zip(handles, labels):
        # Find the original method key by matching display name
        for method_key in consistent_method_order:
            if get_method_display_name(method_key) == l:
                method_labels[method_key] = (h, l)
                break
    
    # Create shared legend below both plots
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
                    if method in method_labels:
                        handle, display_name = method_labels[method]
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
        create_thesis_legend(fig, ax2_regret, method_labels, consistent_method_order)
    
    plt.subplots_adjust(bottom=0.35)
    
    if save_plots:
        acq1_name = acq_display_names.get(acquisition_functions[0], acquisition_functions[0])
        acq2_name = acq_display_names.get(acquisition_functions[1], acquisition_functions[1])
        filename = f"dual_regret_{objective}_{acq1_name.replace(' ', '_')}_vs_{acq2_name.replace(' ', '_')}_thesis.pdf"
        
        save_thesis_plot(fig, output_dir, filename)
    
    plt.show()


def generate_overview_chart(all_results, objectives, acquisition_functions, 
                          save_plots=False, output_dir=None, methods=None, method_alpha=None, method_column_order=None, log_y_axis=False):
    
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
    fig_width = base_fig_width * 1.8  # Slightly wider for 2 columns (experimented)
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
        row_ymin = float("inf")
        row_ymax = float("-inf")
        
        # First pass: determine y-axis limits for this row
        for col_idx, acq_func in enumerate(acquisition_functions):
            regret_data = get_overview_regret_data(all_results, objective, acq_func, methods)
            
            if regret_data:
                for method_name, data in regret_data.items():
                    if 'upper_bound' in data and 'lower_bound' in data:
                        row_ymin = min(row_ymin, np.min(data['lower_bound']))
                        row_ymax = max(row_ymax, np.max(data['upper_bound']))
        
        # Store row limits
        if row_ymin != float("inf") and row_ymax != float("-inf"):
            if log_y_axis and row_ymin > 0:
                # Use logarithmic scaling for y-axis
                log_min = np.log10(row_ymin)
                log_max = np.log10(row_ymax)
                padding = (log_max - log_min) * 0.05
                row_ylims[row_idx] = (10**(log_min - padding), 10**(log_max + padding))
            else:
                # Use linear scaling for y-axis
                padding = (row_ymax - row_ymin) * 0.05
                row_ylims[row_idx] = (row_ymin - padding, row_ymax + padding)
        
        # Second pass: plot data
        for col_idx, acq_func in enumerate(acquisition_functions):
            ax = axes[row_idx, col_idx]
            
            print(f"Processing regret overview: {objective} with {acq_func}")
            
            # Get data for this objective and acquisition function
            regret_data = get_overview_regret_data(all_results, objective, acq_func, methods)
            
            if not regret_data:
                # If no data, create empty plot with label
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_xlabel(r'Iteration')
                ax.set_ylabel(r'$f(\mathbf{x})$')
                continue
            
            # Plot regret data for this subplot
            method_labels = plot_overview_regret_subplot(ax, regret_data, method_alpha)
            
            # Collect method labels for unified legend
            all_method_labels.update(method_labels)
            
            # Apply row-wise y-axis limits for consistency
            if row_idx in row_ylims:
                if log_y_axis and row_ylims[row_idx][0] > 0:
                    # Use logarithmic scaling for y-axis
                    ax.set_yscale('log')
                ax.set_ylim(row_ylims[row_idx])
            
            # Set labels and title
            # Only show x-axis label on bottom row
            if row_idx == n_rows - 1:
                ax.set_xlabel(r'Iteration')
            else:
                ax.set_xlabel('')
            
            # Only show y-axis label on left column  
            if col_idx == 0:
                ax.set_ylabel(r'$f(\mathbf{x})$')
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
                ax.text(-0.20, 0.5, objective_display_name, rotation=90, 
                       ha='center', va='center', transform=ax.transAxes)
            
            # Add horizontal line at y=0 if relevant
            if row_idx in row_ylims:
                ymin, ymax = row_ylims[row_idx]
                if ymin < 0 < ymax:
                    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3, linewidth=0.5)
    
    # Create unified legend below all charts with normal text size (different from thesis style)
    if all_method_labels:
        # Get sorted method names for consistent ordering
        sorted_method_names = get_sorted_method_names(list(all_method_labels.keys()))
        
        # Filter method labels to only include sorted methods
        filtered_method_labels = {
            method: all_method_labels[method] 
            for method in sorted_method_names 
            if method in all_method_labels
        }
        
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
                
                # Temporarily override legend fontsize for overview charts (different from thesis style)
                original_fontsize = plt.rcParams.get('legend.fontsize', 8)
                plt.rcParams['legend.fontsize'] = 10
                
                # Create legend with custom handles and labels
                legend = fig.legend(all_handles, all_labels, 
                                  loc='upper center', bbox_to_anchor=(0.5, -0.02),
                                  ncol=num_columns, frameon=False, fontsize=10)
                fig.add_artist(legend)
                
                # Restore original fontsize
                plt.rcParams['legend.fontsize'] = original_fontsize
        else:
            # Fallback to default legend creation if no custom order
            # Temporarily override legend fontsize for overview charts (different from thesis style)
            original_fontsize = plt.rcParams.get('legend.fontsize', 8)
            plt.rcParams['legend.fontsize'] = 10
            
            create_thesis_legend(fig, axes[0, 0], filtered_method_labels, sorted_method_names)
            
            # Restore original fontsize
            plt.rcParams['legend.fontsize'] = original_fontsize
    
    # Adjust layout with reduced margins, closer legend, and reduced vertical spacing between rows
    plt.subplots_adjust(
        left=0.1,      # left margin
        right=0.95,    # right margin
        bottom=0.01,  # bottom margin (for legend)
        top=0.95,      # top margin
        hspace=0.22    # height space between subplots (vertical spacing)
    )
    
    # Save plot if requested
    if save_plots:
        acq1_name = acquisition_functions[0].replace(' ', '_')
        acq2_name = acquisition_functions[1].replace(' ', '_')
        filename = f"overview_regret_{acq1_name}_vs_{acq2_name}_thesis.pdf"
        save_thesis_plot(fig, output_dir, filename)
        
    plt.show()


def get_overview_regret_data(all_results, objective, acquisition_function, methods):
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
    
    # Extract regret data using the same logic as the main plotting function
    method_histories = defaultdict(list)
    
    for result_dir, results in dim_results.items():
        dir_info = extract_info_from_result_dir(result_dir)
        method_name = dir_info['method']
        if methods and method_name not in methods:
            continue
            
        # Get objective values
        if 'history_y' in results and results['history_y'] is not None:
            history_y = convert_to_numpy(results['history_y'])
            method_histories[method_name].append(history_y)
    
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
        padded_histories = np.array([np.pad(h, (0, max_len - len(h)), mode='edge') for h in histories])
        
        # Compute best-so-far for each seed
        best_so_far_histories = []
        for h in padded_histories:
            best_so_far = np.minimum.accumulate(h)
            best_so_far_histories.append(best_so_far)
        
        best_so_far_histories = np.array(best_so_far_histories)
        
        # Store processed data
        method_data[method_name] = {
            'best_so_far_histories': best_so_far_histories,
            'median': np.median(best_so_far_histories, axis=0),
            'lower_bound': np.percentile(best_so_far_histories, 25, axis=0),
            'upper_bound': np.percentile(best_so_far_histories, 75, axis=0),
        }
    
    return method_data


def plot_overview_regret_subplot(ax, regret_data, method_alpha):
    method_labels = {}
    
    # Get method names and sort them
    sorted_method_names = get_sorted_method_names(list(regret_data.keys()))
    color_map = get_method_color_map(sorted_method_names)
    
    for method_name in sorted_method_names:
        data = regret_data[method_name]
        color = color_map[method_name]
        display_name = get_method_display_name(method_name)
        
        # Get alpha values for this method
        line_alpha = method_alpha.get(method_name, 1.0) if method_alpha else 1.0
        band_alpha = method_alpha.get(method_name, 0.3) if method_alpha else 0.3
        
        # Plot median best-so-far line
        line_handle = ax.plot(
            np.arange(len(data['median'])),
            data['median'],
            color=color,
            alpha=line_alpha,
            label=display_name
        )[0]
        
        # Store handle for legend
        method_labels[method_name] = (line_handle, display_name)
        
        # Plot IQR band around the median
        ax.fill_between(
            np.arange(len(data['median'])),
            data['lower_bound'],
            data['upper_bound'],
            color=color,
            alpha=band_alpha
        )
    
    return method_labels

##################################################################
# Configuration
##################################################################

if __name__ == "__main__":
    project_root = setup_project_path()
    
    # Setup thesis style
    setup_thesis_style()

    main(
        objective=None,     # Set to None to auto-detect all objectives
        dim=None,                  # Set to None to auto-detect all dimensions
        seed=None,                 # Set to None to auto-detect all seeds
        methods=[
            #"bo_plain", 
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
            "turbo_standardize", "turbo_plain",
            "turboni_standardize", 
            "turboni_tr", "turboni_trbinary",
            "turboni_tradditivenorm",# "turboni_tradditive",
            "turboni_trbsnorm",# "turboni_trbs",
        ],
        acquisition_function=None, # Set to None to auto-detect all acquisition functions
        save_plots=True,
        output_dir="figures/thesis/results/",
        sweep="final",
        chart_type="overview",  # Options: "regret", "boxplot", "combined", "dual_regret", "overview"
        dual_acquisition_functions=["UpperConfidenceBound", "LogExpectedImprovement"],  # For dual_regret: ["ExpectedImprovement", "LogExpectedImprovement"]
        overview_acquisition_functions=["UpperConfidenceBound", "LogExpectedImprovement"],  # For overview: ["UpperConfidenceBound", "LogExpectedImprovement"]
        # To generate overview charts, set chart_type="overview" and specify two acquisition functions in overview_acquisition_functions
        independent_boxplot_axes=True,  # Set to True for independent y-axes on boxplots in dual_regret charts
        method_column_order={  # Control method ordering in columns for combined charts
            "Column1": ["turbo_plain", "turbo_standardize"],
            "Column2": ["turboni_standardize", "turboni_tr", "turboni_trbinary"],
            "Column3": ["turboni_tradditivenorm", "turboni_trbsnorm"],
        }, 
        log_y_axis=False,  # Set to True to enable logarithmic scaling for y-axis in regret charts (boxplots always use linear scaling)
    )

"""
Configs for  thesis plots

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


HNI 1

methods=[
            #"bo_plain", 
            "bopt_standardize", #"boot_standardize",
            #"boot_log", "bopt_log",
            #"bopt_bilog",
            "boni_plainnonoise",#"boni_plain", 
            "boni_standardizenonoise",# "boni_standardize", 
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

wspace=0.15  
bottom=0.30,
fig_height=0.9


HNI 2

methods=[
            #"bo_plain", 
            #"bopt_standardize", #"boot_standardize",
            #"boot_log", "bopt_log",
            #"bopt_bilog",
            #"boni_plainnonoise",#"boni_plain", 
            "boni_standardizenonoise",# "boni_standardize", 
            #"boni_standardizenonoiserefit",
            # "boni_standardizeones", "boni_standardizezeros",
            "boni_ilsstandardizenonoise",# "boni_ilsstandardize", 
            # "boni_ilsstandardiznonoiserefit",
            #"boni_bsnoise",
            "boni_bsnonoise",  
            "boni_standardizegradient", "boni_standardizegradientbinary",
            #"turbo_standardize", "turbo_plain",
            #"turboni_standardize", 
            #"turboni_tr", "turboni_trbinary",
            #"turboni_tradditivenorm",# "turboni_tradditive",
            #"turboni_trbsnorm",# "turboni_trbs",
        ],

method_column_order={  # Control method ordering in columns for combined charts
            "Column1": ["boni_standardizenonoise"],
            "Column2": ["boni_ilsstandardizenonoise", "boni_bsnonoise"],
            "Column3": ["boni_standardizegradient", "boni_standardizegradientbinary"],
        }, 

bottom=0.25
Remember to change box and method names
fig_height=0.8


TRNI

methods=[
            #"bo_plain", 
            #"bopt_standardize", #"boot_standardize",
            #"boot_log", "bopt_log",
            #"bopt_bilog",
            #"boni_plainnonoise",#"boni_plain", 
            #"boni_standardizenonoise",# "boni_standardize", 
            #"boni_standardizenonoiserefit",
            # "boni_standardizeones", "boni_standardizezeros",
            #"boni_ilsstandardizenonoise",# "boni_ilsstandardize", 
            # "boni_ilsstandardiznonoiserefit",
            #"boni_bsnoise",
            # "boni_bsnonoise",  
            #"boni_standardizegradient", "boni_standardizegradientbinary",
            "turbo_standardize", "turbo_plain",
            "turboni_standardize", 
            "turboni_tr", "turboni_trbinary",
            "turboni_tradditivenorm",# "turboni_tradditive",
            "turboni_trbsnorm",# "turboni_trbs",
        ],

method_column_order={  # Control method ordering in columns for combined charts
    "Column1": ["turbo_plain", "turbo_standardize"],
    "Column2": ["turboni_standardize", "turboni_tr", "turboni_trbinary"],
    "Column3": ["turboni_tradditivenorm", "turboni_trbsnorm"],
}, 

bottom=0.35,
wspace=0.2
        

OVERVIEW

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

"""