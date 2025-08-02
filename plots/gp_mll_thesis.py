"""
Comprehensive marginal log-likelihood visualization module for thesis plots with multiple configurations.
Chart types: individual MLL trajectories and overview grid layouts comparing model performance across settings.
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
    get_default_markers,
    get_method_display_name,
    get_objective_display_name,
    get_sorted_method_names,
    filter_results_by_criteria,
    group_results_by_dimension,
    convert_to_numpy,
    load_and_filter_results,
    get_thesis_figure_size,
    sort_objectives_by_name_and_dimension,
    create_thesis_output_dir,
    create_thesis_legend,
)


def main(objective=None, dim=None, seed=0, methods=None, acquisition_function=None, save_plots=False, output_dir=None, sweep="debug", method_alpha=None, chart_type="individual", overview_acquisition_functions=None, method_column_order=None):
    
    # Base path for results
    base_path = os.path.join(project_root, "results", sweep)
    
    if not os.path.exists(base_path):
        print(f"Results directory not found at {base_path}")
        return

    # Create output directory if saving plots (thesis directory)
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
    
    # Simplified plot generation logic
    if chart_type == "overview":
        # Handle overview charts
        if overview_acquisition_functions is None or len(overview_acquisition_functions) != 2:
            print("Error: overview_acquisition_functions must contain exactly 2 acquisition functions")
            return
            
        objectives_to_plot = [objective] if objective else all_objectives
        
        # Sort objectives for consistent ordering (by function name, then dimension)
        objectives_to_plot = sort_objectives_by_name_and_dimension(objectives_to_plot)
        
        # Generate overview chart for MLL
        print(f"\nGenerating MLL overview chart")
        generate_overview_chart(
            all_results=all_results,
            objectives=objectives_to_plot,
            acquisition_functions=overview_acquisition_functions,
            save_plots=save_plots,
            output_dir=output_dir,
            methods=methods,
            method_alpha=method_alpha,
            method_column_order=method_column_order
        )
            
    else:
        # Original individual chart logic
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
        
        # Get thesis figure size
        fig_width, fig_height = get_thesis_figure_size()
        
        # Create plot with thesis sizing
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

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
                label=f"{display_name} (median)",
                markevery=max(1, len(x_values) // 20)  # Show markers every 20th point
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
        ax.set_xlabel(r'Iteration')
        ax.set_ylabel(r'Marginal Log Likelihood')
        
        # Apply scientific notation only to y-axis for large MLL values (> 1000 or < -1000)
        from matplotlib.ticker import ScalarFormatter
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 3))  # Use scientific notation for |values| > 1000
        ax.yaxis.set_major_formatter(formatter)
        
        # Move the scientific notation label to the left to avoid interference with x-axis
        ax.yaxis.get_offset_text().set_x(-0.15)
        ax.yaxis.get_offset_text().set_horizontalalignment('left')
        
        # Show all spines for full outline
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)

        # Create thesis-style legend
        handles, labels = ax.get_legend_handles_labels()
        
        # Extract method names from labels (remove " (median)" suffix)
        method_labels = {}
        for h, l in zip(handles, labels):
            if "(median)" in l:
                method_name = l.replace(" (median)", "")
                # Find the original method key by matching display name
                for method_key in sorted_method_names:
                    if get_method_display_name(method_key) == method_name:
                        method_labels[method_key] = (h, method_name)
                        break
        
        # Create thesis legend using custom function
        create_thesis_legend(fig, ax, method_labels, sorted_method_names)

        plt.tight_layout(rect=[0, 0.1, 1, 1.0])

        if save_plots:
            acq_name = acq_display_name.replace(' ', '_')
            filename = f"mll_{objective}_{acq_name}_thesis.pdf"
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, dpi=300)
            print(f"Saved thesis MLL plot: {filepath}")
            
        plt.show()


def generate_overview_chart(all_results, objectives, acquisition_functions, 
                          save_plots=False, output_dir=None, methods=None, method_alpha=None, method_column_order=None):
    
    if len(acquisition_functions) != 2:
        print("Error: Exactly 2 acquisition functions required for overview chart")
        return
        
    if len(objectives) == 0:
        print("Error: No objectives provided for overview chart")
        return
    
    # Get thesis figure size and scale for overview
    base_fig_width, base_fig_height = get_thesis_figure_size()
    
    # Calculate figure size for overview (2 columns, n rows)
    n_rows = len(objectives)
    n_cols = 2
    
    # Scale figure size appropriately
    fig_width = base_fig_width * 1.8
    fig_height = base_fig_height * n_rows * 0.5
    
    # Create subplot grid with shared y-axes for each row
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), sharey='row')
    plt.subplots_adjust(wspace=0.05)  # Further reduce horizontal spacing since y-axes are shared
    
    # Handle case where we have only one row
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Store all method labels for unified legend
    all_method_labels = {}
    
    # Process each objective (row) and acquisition function (column)
    for row_idx, objective in enumerate(objectives):
        for col_idx, acquisition_function in enumerate(acquisition_functions):
            ax = axes[row_idx, col_idx]
            
            # Get MLL data for this combination
            mll_data = get_overview_mll_data(all_results, objective, acquisition_function, methods)
            
            if mll_data:
                # Plot MLL data in this subplot
                method_labels = plot_overview_mll_subplot(ax, mll_data, method_alpha)
                all_method_labels.update(method_labels)
            else:
                # No data - hide the subplot
                ax.set_visible(False)
                continue
            
            # Set axis labels only for left column and bottom row
            if col_idx == 0:
                ax.set_ylabel(r'MLL')
            else:
                # Remove y-axis labels from right column since y-axis is shared
                ax.set_ylabel('')
                ax.tick_params(labelleft=False)  # Hide y-axis tick labels on right column
            
            if row_idx == n_rows - 1:
                ax.set_xlabel(r'Iteration')
            
            # Add objective label on the left side with proper spacing
            if col_idx == 0:
                objective_display_name = get_objective_display_name(objective)
                ax.text(-0.25, 0.5, objective_display_name, 
                       transform=ax.transAxes, fontsize=12,
                       verticalalignment='center', rotation=90)
            
            # Add acquisition function label at the top
            if row_idx == 0:
                acq_display_names = get_acquisition_function_display_names()
                acq_display_name = acq_display_names.get(acquisition_function, acquisition_function)
                ax.set_title(acq_display_name, fontsize=12, pad=10)
            
            # Show all spines for full outline
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
    
    # Create unified legend below all charts with normal text size (different from thesis style)
    if all_method_labels:
        # Get sorted method names for consistent ordering
        sorted_method_names = get_sorted_method_names(list(all_method_labels.keys()))
        
        # Temporarily override legend fontsize for overview charts (different from thesis style)
        original_fontsize = plt.rcParams.get('legend.fontsize', 8)
        plt.rcParams['legend.fontsize'] = 10
        
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
                        if method in all_method_labels:
                            handle, display_name = all_method_labels[method]
                            all_handles.append(handle)
                            all_labels.append(display_name)
                    
                    # Add padding to make all columns the same length
                    methods_in_this_column = len(methods_in_column)
                    padding_needed = max_methods_per_column - methods_in_this_column
                    for _ in range(padding_needed):
                        all_handles.append(Line2D([0], [0], alpha=0))  # Invisible handle
                        all_labels.append("")  # Empty label
                
                # Create legend with custom handles and labels
                legend = fig.legend(all_handles, all_labels, 
                                  loc='upper center', bbox_to_anchor=(0.5, -0.02),
                                  ncol=num_columns, frameon=False, fontsize=10)
                fig.add_artist(legend)
        else:
            # Fallback to default legend creation if no custom order
            create_thesis_legend(fig, axes[0, 0], all_method_labels, sorted_method_names)
        
        # Restore original fontsize
        plt.rcParams['legend.fontsize'] = original_fontsize
    
    # Adjust layout with reduced margins, closer legend, and reduced vertical spacing between rows
    plt.subplots_adjust(
        left=0.1,      # left margin
        right=0.95,    # right margin
        bottom=0.01,  # bottom margin (for legend)
        top=0.95,      # top margin
        hspace=0.22    # height space between subplots (vertical spacing) - reduced from default
    )
    
    # Save plot if requested
    if save_plots:
        filename = f"mll_overview_thesis.pdf"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=300)
        print(f"Saved thesis MLL overview plot: {filepath}")
        
    plt.show()


def get_overview_mll_data(all_results, objective, acquisition_function, methods):
    # Filter results for the specified objective and acquisition function
    filtered_results = filter_results_by_criteria(
        all_results, 
        objective=objective,
        acquisition_function=acquisition_function
    )
    
    if not filtered_results:
        return None
    
    # Group results by dimension (assuming single dimension for overview)
    dimension_groups = group_results_by_dimension(filtered_results)
    
    # Get data from the first (and likely only) dimension group
    if not dimension_groups:
        return None
        
    dim, dim_results = next(iter(dimension_groups.items()))
    
    # Prepare data for MLL plotting (similar to individual plot logic)
    method_mll_histories = defaultdict(list)

    for result_dir, results in dim_results.items():
        dir_info = extract_info_from_result_dir(result_dir)
        method_name = dir_info['method']
        
        # Check if MLL data is available
        if "best_mlls" in results:
            best_mlls = convert_to_numpy(results["best_mlls"])
            
            # Filter out None values and convert to float
            valid_mlls = []
            for mll in best_mlls:
                if mll is not None:
                    valid_mlls.append(float(mll))
                else:
                    valid_mlls.append(np.nan)
            
            if len(valid_mlls) > 1:  # Only include if we have actual MLL data
                method_mll_histories[method_name].append(np.array(valid_mlls))

    if not method_mll_histories:
        return None
    
    # Filter methods if specified
    if methods:
        method_mll_histories = {k: v for k, v in method_mll_histories.items() if k in methods}
    
    return method_mll_histories


def plot_overview_mll_subplot(ax, method_mll_histories, method_alpha):
    method_labels = {}
    
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
        
        # Compute statistics
        median_mll = np.nanmedian(padded_histories, axis=0)
        lower_bound = np.nanpercentile(padded_histories, 25, axis=0)
        upper_bound = np.nanpercentile(padded_histories, 75, axis=0)
        
        color = color_map[method_name]
        display_name = get_method_display_name(method_name)

        # Get alpha values for this method (default values if not specified)
        line_alpha = method_alpha.get(method_name, 0.8) if method_alpha else 0.8
        band_alpha = method_alpha.get(method_name, 0.3) if method_alpha else 0.3

        # Create x-axis (iteration numbers, starting from 0)
        x_values = np.arange(len(median_mll))
        
        # Plot median MLL line
        line = ax.plot(
            x_values,
            median_mll,
            color=color,
            linewidth=2.5,
            alpha=line_alpha,
            label=f"{display_name}",
            markevery=max(1, len(x_values) // 20)  # Show markers every 20th point
        )[0]

        # Plot IQR band around the median
        ax.fill_between(
            x_values,
            lower_bound,
            upper_bound,
            color=color,
            alpha=band_alpha
        )
        
        # Store method label for legend
        method_labels[method_name] = (line, display_name)
    
    # Apply scientific notation only to y-axis for large MLL values (> 1000 or < -1000)
    from matplotlib.ticker import ScalarFormatter
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))  # Use scientific notation for |values| > 1000
    ax.yaxis.set_major_formatter(formatter)
    
    # Move the scientific notation label to the left to avoid interference with x-axis
    ax.yaxis.get_offset_text().set_x(-0.15)
    ax.yaxis.get_offset_text().set_horizontalalignment('left')
    
    return method_labels

##################################################################
# Configuration
##################################################################

if __name__ == "__main__":
    project_root = setup_project_path()
    
    # Apply thesis style - use full path to paper.mplstyle
    style_path = os.path.join(os.path.dirname(__file__), 'paper.mplstyle')
    plt.style.use(style_path)

    main(
        objective=None,  # Set to None to auto-detect all objectives
        dim=None,        # Set to None to auto-detect all dimensions
        seed=None,       # Set to None to auto-detect all seeds
        methods=[
            "bo_plain", 
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
            # "turbo_standardize", "turbo_plain",
            # "turboni_standardize", 
            # "turboni_tr", "turboni_trbinary",
            # "turboni_tradditivenorm",# "turboni_tradditive",
            # "turboni_trbsnorm",# "turboni_trbs",
        ],
        acquisition_function=None,  # Set to None to auto-detect all acquisition functions
        save_plots=True,
        output_dir="figures/thesis/mll/",
        sweep="final",
        chart_type="individual",  # "individual" or "overview"
        overview_acquisition_functions=["UpperConfidenceBound", "LogExpectedImprovement"],  # For overview charts
        method_column_order={  # Control method ordering in columns for combined charts
            "Column1": ["turbo_plain", "turbo_standardize"],
            "Column2": ["turboni_standardize", "turboni_tr", "turboni_trbinary"],
            "Column3": ["turboni_tradditivenorm", "turboni_trbsnorm"],
        }, 
        method_alpha={  # Control transparency/blending of specific methods
            # "bo_plain": 0.3,
            # "boot_log": 0.3,
            # "boot_standardize": 0.3,
        }
    )

'''
Configuration for thesis plots

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

Take other configs from other overview thesis charts (same for all)
'''