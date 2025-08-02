"""
Generates comprehensive boxplot comparisons of final regret values across multiple objectives and dimensions.
Chart types: grouped boxplots with transformation type annotations for method performance comparison.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    setup_project_path,  
    extract_info_from_result_dir, 
    get_acquisition_function_display_names,
    get_objective_function_display_names,
    get_method_color_map,
    get_method_display_name,
    get_objective_display_name,
    convert_to_numpy,
    load_and_filter_results,
    get_thesis_figure_size,
    setup_thesis_style,
    sort_objectives_by_name_and_dimension,
    create_thesis_output_dir,
    save_thesis_plot,
    apply_method_alpha_to_boxplot,
)
from collections import defaultdict
from matplotlib.ticker import MaxNLocator

def extract_transformation_info(method_name):
    if "bopt" in method_name:
        transform_type = "PT"  # Parameter transformation
    elif "boot" in method_name:
        transform_type = "OT"  # Output transformation
    else:
        transform_type = "None"  # No transformation
    
    # Extract specific transformation method
    if "standardize" in method_name:
        transform_method = "Standardize"
    elif "log" in method_name:
        transform_method = "Log"
    elif "bilog" in method_name:
        transform_method = "Bilog"
    else:
        transform_method = "Plain"
    
    return f"{transform_type} {transform_method}"

def main(
    objective=None,  # Set to None to analyze all objectives
    seed=None,       # Set to None to use all seeds
    methods=None,    # List of methods to include
    acquisition_function=None,  # Set to None to generate combined plot for all acquisition functions
    save_plots=False,
    output_dir=None,
    sweep="debug",
    exclude_methods_by_objective=None,  # Dict mapping objective names to lists of methods to exclude
):
    # Base path for results
    base_path = os.path.join(project_root, "results", sweep)
    
    if not os.path.exists(base_path):
        print(f"Results directory not found at {base_path}")
        return

    # Create output directory if saving plots using thesis utility
    if save_plots:
        output_dir = create_thesis_output_dir(output_dir, "results")

    # Load and filter results using common function
    results = load_and_filter_results(
        base_path, objective, None, seed, acquisition_function, methods
    )
    
    if results[0] is None:  # Check if loading failed
        return
        
    all_results, all_objectives, all_dimensions, all_acquisition_functions = results
    
    print(f"Found acquisition functions: {all_acquisition_functions}")
    print(f"Found objectives: {all_objectives}")
    print(f"Found dimensions: {all_dimensions}")
    print(f"Total results found: {len(all_results)}")
    
    # Debug: show a few sample result directories
    for i, result_dir in enumerate(list(all_results.keys())[:5]):
        print(f"Sample result directory {i+1}: {result_dir}")
        dir_info = extract_info_from_result_dir(result_dir)
        print(f"  Parsed info: {dir_info}")
    
    # Generate combined overview boxplot for all acquisition functions
    print(f"\nGenerating combined overview boxplot for all acquisition functions")
    generate_overview_boxplot(
        all_results=all_results, 
        save_plots=save_plots, 
        output_dir=output_dir,
        methods=methods,
        original_objective_order=objective,
        exclude_methods_by_objective=exclude_methods_by_objective
    )

def generate_overview_boxplot(all_results, save_plots=False, output_dir=None, methods=None, original_objective_order=None, exclude_methods_by_objective=None):
    
    # Get utility elements
    acq_display_names = get_acquisition_function_display_names()
    obj_display_names = get_objective_function_display_names()
    
    # Create shorter acquisition function names
    acq_short_names = {
        'ExpectedImprovement': 'EI',
        'UpperConfidenceBound': 'UCB',
        'LogExpectedImprovement': 'LogEI',
        'ProbabilityOfImprovement': 'PI'
    }
    
    # Define objective groups based on their global minimum value ranges
    objective_groups = {
        'negative': ['hartmann3D', 'hartmann6D'],  # Global minima: f* = -3.86, -3.32
        'zero': ['branin', 'ackley2D', 'ackley10D', 'ackley20D', 'ackley100D', 'rastrigin2D', 'rastrigin10D', 'rastrigin20D'],  # Global minima: f* = 0.0
        'positive': ['friedman10D']  # Global minima: f* = 0.398, f* ≈ 3.18
    }
    
    # Organize data by combination and group
    combination_data_by_group = {group: defaultdict(list) for group in objective_groups.keys()}
    
    # Process all results to extract objective, acquisition function, transformation, and final value
    for result_dir, results in all_results.items():
        dir_info = extract_info_from_result_dir(result_dir)
        method_name = dir_info['method']
        objective = dir_info.get('objective', 'unknown')
        acquisition_function = dir_info.get('acquisition_function', 'unknown')
        
        # Filter by methods if specified
        if methods and method_name not in methods:
            continue
        
        # Skip excluded methods for this objective
        if exclude_methods_by_objective and objective in exclude_methods_by_objective:
            if method_name in exclude_methods_by_objective[objective]:
                continue
        
        # Extract transformation information
        transform_info = extract_transformation_info(method_name)
        
        # Get final objective value (best-so-far)
        if "history_y" in results:
            history_y = convert_to_numpy(results["history_y"])
            best_so_far = np.minimum.accumulate(history_y)
            final_value = best_so_far[-1]
            
            # Find which group this objective belongs to
            objective_group = None
            for group, objectives in objective_groups.items():
                if objective in objectives:
                    objective_group = group
                    break
            
            if objective_group:
                # Create combination key with objective only
                combination_key = f"{objective}"
                combination_data_by_group[objective_group][combination_key].append((final_value, acquisition_function, transform_info))
    
    # Filter out empty groups
    non_empty_groups = {group: data for group, data in combination_data_by_group.items() if data}
    
    # Debug: Print what data we have for each group
    for group_name, group_data in non_empty_groups.items():
        print(f"Group '{group_name}' has {len(group_data)} combinations:")
        for combo, values in group_data.items():
            print(f"  {combo}: {len(values)} values")
    
    if not non_empty_groups:
        print(f"No data found")
        return
    
    # Get thesis figure size and make it wider and taller for better visibility
    fig_width, fig_height = get_thesis_figure_size()
    fig_height *= 1.1  # Make chart taller to accommodate legend
    
    # Create a flat list of all objectives from all groups
    all_objectives = []
    for group_data in non_empty_groups.values():
        all_objectives.extend(group_data.keys())
    
    # Order objectives according to the original order specified in main() function
    if original_objective_order:
        # Filter original order to only include objectives that have data
        available_objectives = set(all_objectives)
        ordered_objectives = [obj for obj in original_objective_order if obj in available_objectives]
        
        # Add any remaining objectives that weren't in the original order (fallback)
        remaining_objectives = [obj for obj in all_objectives if obj not in ordered_objectives]
        all_objectives = ordered_objectives + remaining_objectives
    else:
        # Use utility function for consistent objective sorting
        all_objectives = sort_objectives_by_name_and_dimension(set(all_objectives))
    
    n_objectives = len(all_objectives)
    
    print(f"Creating {n_objectives} subplots for objectives: {all_objectives}")
    
    # Create figure with subplots for each objective function
    # Determine how many columns to use (max 3 per row for readability)
    n_cols = 4
    n_rows = (n_objectives + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with increased height for multiple rows if needed
    fig_height_adjusted = fig_height * max(0.3, n_rows * 0.4)
    fig_width_adjusted = fig_width
    
    # Create figure with increased spacing between subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width_adjusted, fig_height_adjusted), 
                           sharey=False, squeeze=False, gridspec_kw={'hspace': 0.6, 'wspace': 0.4})

    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Hide unused subplots
    for i in range(n_objectives, len(axes)):
        axes[i].set_visible(False)
    
    # Create mapping from transformation info to method names for color lookup
    transform_to_method_mapping = {
        'PT Standardize': 'bopt_standardize',
        'PT Log': 'bopt_log', 
        'PT Bilog': 'bopt_bilog',
        'OT Standardize': 'boot_standardize',
        'OT Log': 'boot_log',
        'OT Bilog': 'boot_bilog',
        'None Plain': 'bo_plain'
    }
    
    # Get color mapping using existing utility function
    method_color_map = get_method_color_map(methods)
    
    # Create transform colors mapping using the method color map
    transform_colors = {}
    for transform_info, method_name in transform_to_method_mapping.items():
        if method_name in method_color_map:
            transform_colors[transform_info] = method_color_map[method_name]
        else:
            # Fallback color for any unmapped transformations
            transform_colors[transform_info] = '#646567'  # RWTH Gray
    
    # Define the desired order of transformation types/methods
    transform_order = [
        'OT Standardize',
        'PT Standardize', 
        'OT Log',
        'PT Log',
        'PT Bilog'
    ]
    
    def get_transform_sort_key(transform_info):
        try:
            return transform_order.index(transform_info)
        except ValueError:
            # If not in the predefined order, put it at the end
            return len(transform_order)
    
    # Plot each objective in its own subplot
    for idx, objective in enumerate(all_objectives):
        # Skip if we have more objectives than axes (should not happen with our setup)
        if idx >= len(axes):
            print(f"Warning: More objectives than available axes. Skipping {objective}.")
            continue
            
        ax = axes[idx]
        
        # Find which group this objective belongs to
        objective_group = None
        for group, objectives in objective_groups.items():
            if objective in objectives:
                objective_group = group
                break
                
        if not objective_group:
            print(f"Warning: Could not determine group for objective {objective}. Skipping.")
            continue
            
        # Get data for this objective from the appropriate group
        data = non_empty_groups.get(objective_group, {}).get(objective, [])
        if not data:
            print(f"Warning: No data found for objective {objective}. Skipping.")
            continue
        
        # Group by transformation info and acquisition function
        transform_acq_data = defaultdict(list)
        for value, acq_func, transform_info in data:
            key = f"{transform_info}-{acq_func}"
            transform_acq_data[key].append(value)
        
        # Prepare data for boxplot - sorted by acquisition function, then transformation
        boxplot_data = []
        combination_labels = []
        box_colors = []
        method_names = []  # Track method names for utility function
        acq_func_positions = {}  # To track positions of each acquisition function
        acq_func_counts = {}     # To track how many boxes per acquisition function
        
        # Group by acquisition function first
        acq_groups = defaultdict(list)
        for key in transform_acq_data.keys():
            transform_info, acq_func = key.rsplit('-', 1)
            acq_groups[acq_func].append((key, transform_info))
        
        # First pass: Count how many boxes per acquisition function
        position = 1  # Start position for box plots
        acq_func_box_counts = {}
        
        for acq_func in sorted(acq_groups.keys()):
            sorted_transforms = sorted(acq_groups[acq_func], key=lambda x: get_transform_sort_key(x[1]))
            valid_count = sum(1 for key, _ in sorted_transforms if len(transform_acq_data[key]) > 0)
            acq_func_box_counts[acq_func] = valid_count
            
        # Second pass: Create boxes and calculate correct positions
        position = 1
        spacing = 0.75  # Spacing between different acquisition function groups
        
        # For each acquisition function, sort by transformation info using the specified order
        for acq_func in sorted(acq_groups.keys()):
            # Sort transformation info by the specified order
            sorted_transforms = sorted(acq_groups[acq_func], key=lambda x: get_transform_sort_key(x[1]))
            
            # Record the starting position for this acquisition function
            acq_func_positions[acq_func] = position
            acq_func_counts[acq_func] = 0
            
            # Add boxes for this acquisition function
            for key, transform_info in sorted_transforms:
                if len(transform_acq_data[key]) > 0:
                    boxplot_data.append(transform_acq_data[key])
                    combination_labels.append("")  # Empty label, we'll add them as a shared label later
                    box_colors.append(transform_colors.get(transform_info, '#646567'))
                    # Get method name for utility function
                    method_name = transform_to_method_mapping.get(transform_info, '')
                    method_names.append(method_name)
                    acq_func_counts[acq_func] += 1
                    position += 1
            
            # Add spacing between acquisition function groups
            position += spacing
        
        if not boxplot_data:
            print(f"Warning: No valid box plot data for {objective}. Skipping.")
            ax.set_visible(False)
            continue
        
        # Calculate positions for the boxplots based on acquisition function grouping
        positions = []
        current_pos = 1
        
        for acq_func in sorted(acq_groups.keys()):
            sorted_transforms = sorted(acq_groups[acq_func], key=lambda x: get_transform_sort_key(x[1]))
            for key, _ in sorted_transforms:
                if len(transform_acq_data[key]) > 0:
                    positions.append(current_pos)
                    current_pos += 1
            
            current_pos += spacing  # Add spacing between acquisition function groups
        
        # Create boxplot with custom positions
        bp = ax.boxplot(
            boxplot_data,
            positions=positions,
            patch_artist=True,
            showmeans=False,
            meanline=False,
            boxprops=dict(linewidth=0.5),
            whiskerprops=dict(linewidth=0.5),
            capprops=dict(linewidth=0.5),
            medianprops=dict(visible=True, linewidth=1.0, linestyle='-', color='black'),
            showfliers=False
        )
        
        # Apply colors using existing utility function
        apply_method_alpha_to_boxplot(bp, method_names, method_color_map)
        
        # Calculate proper y-axis limits based on actual data before drawing colored squares
        all_values = []
        for values_list in boxplot_data:
            all_values.extend(values_list)
        
        if all_values:
            data_min = min(all_values)
            data_max = max(all_values)
            data_range = data_max - data_min
            
            padding = data_range * 0.1
            y_min = data_min - padding
            y_max = data_max + padding
            
            # If this is ackley10D, use the stored limits from ackley2D if available
            if objective == 'ackley10D':
                y_min, y_max = (1,3.6)
            # If this is ackley2D, set specific y-axis range
            elif objective == 'ackley2D':
                y_min, y_max = (-0.2, 3)
            
            # Set the y-axis limits
            ax.set_ylim(y_min, y_max)
        
        # Draw a colored square above small boxes for color visibility - matching thesis style
        y_min, y_max = ax.get_ylim()
        y_span = y_max - y_min
        x_min, x_max = ax.get_xlim()
        x_span = x_max - x_min
        square_height = 0.06 * y_span
        square_width = 0.06 * x_span  # Make width proportional to x-axis span
        threshold = 0.06 * y_span
        for patch, color in zip(bp['boxes'], box_colors):
            # Get the box's coordinates from the PathPatch vertices
            verts = patch.get_path().vertices
            box_x = verts[:, 0]
            box_y = verts[:, 1]
            box_bottom = np.min(box_y)
            box_top = np.max(box_y)
            box_height = box_top - box_bottom
            box_center = np.mean(box_x)
            # If the box is too small, draw a square above it
            if box_height < threshold:
                # Place the square just above the top of the box
                square_y = box_top + 0.1 * y_span
                ax.add_patch(
                    plt.Rectangle(
                        (box_center - square_width/4, square_y), square_width, square_height,
                        facecolor=color, zorder=10
                    )
                )
        
        # Style the subplot
        if idx % n_cols == 0:  # Only label y-axis for leftmost subplots in each row
            ax.set_ylabel(r'$f(\mathbf{x})$')
        
        # Set y-tick labels for all subplots (not just leftmost ones)
        ax.tick_params(axis='y', which='major')
        
        # Create shared x-tick labels for each acquisition function
        x_ticks = []
        x_labels = []
        
        # Add centered labels for each acquisition function group
        for acq_func, start_pos in acq_func_positions.items():
            count = acq_func_counts[acq_func]
            if count > 0:
                # Calculate the center position for the label
                # For exact centering, we need the midpoint between first and last position
                first_pos = start_pos
                last_pos = start_pos + count - 1  # Since positions are 1-indexed
                center_pos = (first_pos + last_pos) / 2
                x_ticks.append(center_pos)
                acq_short = acq_short_names.get(acq_func, acq_func)
                x_labels.append(acq_short)
        
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        
        # Add subtle vertical lines between acquisition function groups
        acq_funcs = sorted(acq_groups.keys())
        for i in range(len(acq_funcs) - 1):  # For all except the last acquisition function
            current_acq = acq_funcs[i]
            count = acq_func_counts[current_acq]
            if count > 0:
                # Calculate separator position: midway in the spacing between groups
                start_pos = acq_func_positions[current_acq]
                last_pos = start_pos + count - 1
                next_start = last_pos + spacing
                separator_pos = last_pos + spacing/2
        
        # Set y-axis tick formatting and spacing
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Set appropriate number of ticks based on the range
        if all_values:
            data_range = max(all_values) - min(all_values)
            if data_range < 1:
                # For small ranges, use more precise ticks
                num_ticks = 6
            elif data_range < 10:
                # For medium ranges, use moderate number of ticks
                num_ticks = 8
            else:
                # For large ranges, use fewer ticks to avoid crowding
                num_ticks = 4
            
            # Let matplotlib choose the tick locations automatically
            ax.locator_params(axis='y', nbins=num_ticks)
        
        # Add horizontal reference lines for global optima (only if they're within the visible range)
        ymin, ymax = ax.get_ylim()
        
        if objective == 'hartmann3D' and ymin <= -3.86 <= ymax:
            ax.axhline(y=-3.86, color="red", linestyle="--", alpha=0.5, linewidth=1, label="Hartmann3D optimum")
        elif objective == 'hartmann6D' and ymin <= -3.32 <= ymax:
            ax.axhline(y=-3.32, color="red", linestyle="--", alpha=0.5, linewidth=1, label="Hartmann6D optimum")
        elif objective == 'branin' and ymin <= 0.398 <= ymax:
            ax.axhline(y=0.398, color="red", linestyle="--", alpha=0.5, linewidth=1, label="Branin optimum")
        elif objective == 'friedman10D' and ymin <= 3.18 <= ymax:
            ax.axhline(y=3.18, color="red", linestyle="--", alpha=0.5, linewidth=1, label="Friedman optimum")
        elif objective_group == 'zero' and ymin <= 0 <= ymax:
            # All have global minimum at 0.0 - only show if 0 is within the visible range
            ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, linewidth=1, label="Global optimum")
        
        # Add horizontal line at y=0 if relevant and within range
        if ymin < 0 < ymax:
            ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3, linewidth=0.5)

        
        # Set subplot title using utility function for consistent naming
        title = get_objective_display_name(objective)
        ax.set_title(title, pad=5)
    
    # Create legend for transformation types below all subplots
    legend_handles = []
    legend_labels = []
    
    # Collect all transformation types that appear in the data
    all_transform_types = set()
    for group_data in non_empty_groups.values():
        for combo_data in group_data.values():
            for _, _, transform_info in combo_data:
                all_transform_types.add(transform_info)
    
    # Create legend entries in the specified order using method display names
    for transform_info in transform_order:
        if transform_info in all_transform_types:
            # Get the corresponding method name and its display name using utility function
            method_name = transform_to_method_mapping.get(transform_info, '')
            display_name = get_method_display_name(method_name) if method_name else transform_info
            
            patch = plt.Rectangle((0,0), 1, 1, 
                                facecolor=transform_colors.get(transform_info, '#646567'), 
                                alpha=0.7)
            legend_handles.append(patch)
            legend_labels.append(display_name)
    
    # Add any remaining transformation types that weren't in the predefined order
    for transform_info in sorted(all_transform_types):
        if transform_info not in transform_order:
            # Get the corresponding method name and its display name using utility function
            method_name = transform_to_method_mapping.get(transform_info, '')
            display_name = get_method_display_name(method_name) if method_name else transform_info
            
            patch = plt.Rectangle((0,0), 1, 1, 
                                facecolor=transform_colors.get(transform_info, '#646567'), 
                                alpha=0.7)
            legend_handles.append(patch)
            legend_labels.append(display_name)
    
    # Add legend below the subplots using thesis-style approach
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc='lower center',
            bbox_to_anchor=(0.46, -0.02),
            ncol=len(legend_handles),
            frameon=False,
            columnspacing=2.0,
            handletextpad=0.5
        )
    
    plt.tight_layout()
    
    plt.subplots_adjust(
        left=0.05,      # Left margin
        right=0.95,     # Right margin  
        top=0.92,       # Top margin (space for titles)
        bottom=0.18,    # Bottom margin (for x-labels and legend) - adjusted to match thesis
        wspace=0.4,    # Width spacing between subplots (increased for better separation)
        hspace=1.15      # Height spacing between subplot rows (if multiple rows)
    )
    
    if save_plots:
        filename = "all_boxplot_comparision_point_transformation.pdf"
        save_thesis_plot(fig, output_dir, filename)
    
    plt.show()

##################################################################
# Configuration
##################################################################

if __name__ == "__main__":
    project_root = setup_project_path()
    
    setup_thesis_style()

    main(
        objective=[
            "ackley2D", "ackley10D", "ackley20D", "ackley100D",
            #"hartmann6D",
            "rastrigin2D", "rastrigin10D", "rastrigin20D",
            "branin", 
        ],  # Set to None to analyze all objectives
        seed=0,              
        methods=[
            #"bo_plain", 
            "bopt_standardize", "boot_standardize",
            "boot_log", "bopt_log",
            #"bopt_bilog",
            # "boni_plain", "boni_plainnonoise",
            # "boni_standardize", "boni_standardizenonoise",
            # "boni_standardizenonoiserefit",
            # "boni_standardizeones", "boni_standardizezeros",
            # "boni_ilsstandardize", "boni_ilsstandardizenonoise",
            # "boni_ilsstandardiznonoiserefit",
            # "boni_bsnoise",
            # "boni_bsnonoise",  
            # "boni_standardizegradient", "boni_standardizegradientbinary",
            # "turbo_plain", "turbo_standardize",
            # "turboni_standardize", 
            # "turboni_tr", "turboni_trbinary",
            # "turboni_tradditive", "turboni_tradditivenorm",
            # "turboni_trbs", "turboni_trbsnorm",
        ],
        acquisition_function=None,  # Combined plot for all acquisition functions
        save_plots=True,
        output_dir="figures/thesis/results/",
        sweep="final",
        exclude_methods_by_objective={  # Dict mapping objective names to lists of methods to exclude
            "ackley2D": ["bopt_log", "boot_log"],
            "ackley10D": ["bopt_log", "boot_log"],
            "ackley20D": ["bopt_log", "boot_log"],
            "ackley100D": ["bopt_log", "boot_log"],
            "branin": ["boot_log"],
        }
    )
