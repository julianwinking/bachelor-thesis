import os
import sys
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# RWTH Colors (100, 75, 50, 25, 10% opacity)
RWTH_COLORS = {
    'blue': ["#00549F", "#407FB7", "#8EBAE5", "#C7DDF2", "#E8F1FA"],
    'black': ["#000000", "#646567", "#9C9E9F", "#CFD1D2", "#ECEDED"],
    'magenta': ["#E30066", "#E96088", "#F19EB1", "#F9D2DA", "#FDEEF0"],
    'yellow': ["#FFED00", "#FFF055", "#FFF59B", "#FFFAD1", "#FFFDEE"],
    'petrol': ["#006165", "#2D7F83", "#7DA4A7", "#BFD0D1", "#E6ECEC"],
    'tuerkis': ["#0098A1", "#00B1B7", "#89CCCF", "#CAE7E7", "#EBF6F6"],
    'green': ["#57AB27", "#8DC060", "#B8D698", "#DDEBCE", "#F2F7EC"],
    'maigreen': ["#BDCD00", "#D0D95C", "#E0E69A", "#F0F3D0", "#F9FAED"],
    'orange': ["#F6A800", "#FABE50", "#FDD48F", "#FEEAC9", "#FFF7EA"],
    'red': ["#CC071E", "#D85C41", "#E69679", "#F3CDBB", "#FAEBE3"],
    'bordeaux': ["#A11035", "#B65256", "#CD8B87", "#E5C5C0", "#F5E8E5"],
    'violet': ["#612158", "#834E75", "#A8859E", "#D2C0CD", "#EDE5EA"],
    'lila': ["#7A6FAC", "#9B91C1", "#BCB5D7", "#DEDAEB", "#F2F0F7"],
}

def get_display_name_mappings():
    return {
        "bo_plain": "BO",

        "boot_standardize": "BO S-OT",
        "boot_log": "BO Log-OT",
        "boot_bilog": "BO Bilog-OT",

        "bopt_standardize": "BO S-PT",
        "bopt_log": "BO Log-PT",
        "bopt_bilog": "BO Bilog-PT",

        "boni_plain": "BO Naive-NI Noise-First",
        "boni_plainnonoise": "BO Naive-NI No-Noise-First",

        "boni_standardize": "BO Naive-NI S-PT Noise-First",
        "boni_standardizenonoise": "BO Naive-NI S-PT", #No-Noise-First",
        "boni_standardizenonoiserefit": "BO Naive-NI S-PT (Refit)",
        "boni_standardizezeros": "BO Naive-NI Only-Zeros",
        "boni_standardizeones": "BO NI Only-Ones",
        "boni_standardizegradient": "BO GB-NI S-PT",
        "boni_standardizegradientbinary": "BO B-GB-NI S-PT",

        "boni_ilsstandardize": "BO ILS-NI Noise-First",
        "boni_ilsstandardizenonoise": "BO ILS-NI S-PT",
        "boni_ilsstandardizenonoiserefit": "BO ILS-NI No-Noise-First (Refit)",

        "boni_bsnoise": "BO BS-NI Noise-First",
        "boni_bsnonoise": "BO BS-NI S-PT",

        "turbo_plain": "TuRBO",
        "turbo_standardize": "TuRBO S-PT",
        "turboni_standardize": "TuRBO Naive-NI S-PT",
        "turboni_tr": "TuRBO TRNI S-PT",
        "turboni_trbinary": "TuRBO B-TRNI S-PT",
        "turboni_tradditive": "TuRBO A-TRNI S-PT No-Norm",
        "turboni_tradditivenorm": "TuRBO A-TRNI S-PT",
        "turboni_trbs": "TuRBO BS-TRNI S-PT No-Norm",
        "turboni_trbsnorm": "TuRBO BS-TRNI S-PT",
    }

def get_acquisition_function_display_names():
    return {
        "ExpectedImprovement": "Expected Improvement",
        "LogExpectedImprovement": "Log Expected Improvement",
        "qLogExpectedImprovement": "qLog Expected Improvement",
    }

def get_objective_function_display_names():
    return {
        "ackley2D": "Ackley 2D",
        "ackley10D": "Ackley 10D",
        "ackley20D": "Ackley 20D",
    }

def get_method_color_map(methods=None):
    # Comprehensive color mapping for all methods in display_name_mappings()
    predefined_map = {
        # Baseline BO methods
        "bo_plain": RWTH_COLORS['blue'][0],
        
        # Outcome Transformation methods
        "boot_standardize": RWTH_COLORS['orange'][0],  
        "boot_log": "#ff9999",             
        
        # Preprocessing Transformation methods
        "bopt_standardize":RWTH_COLORS['green'][0],  
        "bopt_log":"#47EFC8",          
        "bopt_bilog": "#ACEE39",         
        
        # Noise Injection methods - Purple/Pink/Magenta tones
        # Normal NI
        "boni_plain": "#9467bd",                      # purple
        "boni_plainnonoise": "#c2c2f0",               # light purple
        
        # Transformed NI
        "boni_standardize": "#e377c2",                # pink
        "boni_standardizenonoise": "#ff99cc",         # light pink
        "boni_standardizenonoiserefit": "#f2b5d4",    # very light pink
        "boni_standardizezeros": "#ffb3e6",           # light magenta
        "boni_standardizeones": "#c4e17f",            # light lime (contrast)
        "boni_standardizegradient": "#bb8fce",        # medium purple
        "boni_standardizegradientbinary": "#d2b4de",  # light purple

        # ILS NI
        "boni_ilsstandardize": "#8c564b",             # brown
        "boni_ilsstandardizenonoise": "#f7dc6f",      # light yellow
        "boni_ilsstandardizenonoiserefit": "#f2b5d4", # very light pink

        # BS NI
        "boni_bsnoise": "#85c1e9",                    # medium blue
        "boni_bsnonoise": "#00c3ff",                  # light blue

        # Trust Region NI
        "boni_tr": "#7f7f7f",                         # gray
        "boni_trbs": "#76d7c4",                       # light teal

        # TuRBO methods - Teal tones
        "turbo_plain": "#1f77b4",                     # blue (same as BO)
        "turbo_standardize": "#2ca02c",               # green (same as BOPT)
        "turboni_standardize": RWTH_COLORS['orange'][0],              # red (same as BONI)
        "turboni_tr": "#e377c2",                       # purple (same as BONI)
        "turboni_trbinary": "#f2b5d4",                 # brown (same as BONI ILS)
        "turboni_trbs": "#ff7f0e",                       # orange (same as BOOT)
        "turboni_trbsnorm": "#00c3ff",                   # light red (same as BOOT LOG)
        "turboni_tradditivenorm": RWTH_COLORS['lila'][0],               # red (same as BOOT LOG)
    }
    
    if methods is None:
        return predefined_map
    
    # If methods are provided, ensure they have consistent colors based on predefined mapping
    method_color_map = {}
    # Additional colors for new methods not in predefined mapping
    additional_colors = [
        "#d62728",  # red
        "#bcbd22",  # olive
        "#17becf",  # cyan
        "#66b3ff",  # light blue
        "#f8c471",  # medium orange
    ]
    unused_color_idx = 0
    
    for method in methods:
        if method in predefined_map:
            method_color_map[method] = predefined_map[method]
        else:
            # Assign next available color for new methods not in predefined mapping
            if unused_color_idx < len(additional_colors):
                method_color_map[method] = additional_colors[unused_color_idx]
                unused_color_idx += 1
            else:
                # Fallback to basic colors if we run out
                basic_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
                method_color_map[method] = basic_colors[len(method_color_map) % len(basic_colors)]
            
    return method_color_map

def get_sorted_method_names(method_names):
    """Sort method names by grouping (BO, BOOT, BOPT, BONI) and then alphabetically."""
    def get_sort_key(method_name):
        parts = method_name.split('_', 1)
        prefix = parts[0]
        suffix = parts[1] if len(parts) > 1 else ""

        if prefix == "bo":  # Handles "bo_plain" etc.
            order_index = 0
        elif prefix == "boot": # BO OT methods
            order_index = 1
        elif prefix == "bopt": # BO PT methods
            order_index = 2
        elif prefix == "boni": # BO NI methods
            order_index = 3
        elif prefix == "turbo" or prefix == "turboni":  # TuRBO methods
            order_index = 4
        else:
            order_index = 5  # Other methods last

        return (order_index, suffix) # Sort by group index, then by suffix alphabetically
    
    return sorted(method_names, key=get_sort_key)

def get_default_markers():
    return ["o", "s", "^", "D", "v", "<", ">"]

def setup_project_path():
    """Add the project root directory to Python path."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level from plots to the project root
    sys.path.insert(0, project_root)
    return project_root

def load_results(path):
    """Load results from a given path using pickle directly."""
    name = path + "/results.pkl"

    if not os.path.exists(name):
        print(f"Results file not found at: {name}")
        return None

    with open(name, 'rb') as f:
        res = pickle.load(f)
    
    print(f"Loaded results from {name} with keys: {list(res.keys())}")
    return res

def extract_info_from_result_dir(result_dir):
    """Extract method, objective, dimension, acquisition function, and seed from result directory name.
    
    Expected format: method__objective_dim_acqfunc_seed
    Example: bogp_log_ackley2D_2_expectedimprovement_0
    """
    parts = result_dir.split('_')

    info = {
        'method': None,
        'objective': None,
        'dim': None,
        'acquisition_function': None,
        'seed': None
    }

    # Extract method (everything before the second underscore)
    if len(parts) >= 2:
        info['method'] = '_'.join(parts[:2])

    # Extract objective (comes after the method)
    if len(parts) >= 3:
        info['objective'] = parts[2]

    # Extract dimension (comes after the objective)
    if len(parts) >= 4:
        try:
            info['dim'] = int(parts[3])
        except ValueError:
            pass

    # Extract acquisition function (comes after the dimension)
    if len(parts) >= 5:
        info['acquisition_function'] = parts[4]

    # Extract seed (last part)
    if len(parts) >= 6:
        try:
            info['seed'] = int(parts[5])
        except ValueError:
            pass

    return info

def get_method_display_name(method_name):
    """Get a clean display name for a method."""
    # Use the display name mappings, but with a fallback
    display_names = get_display_name_mappings()
    if method_name in display_names:
        return display_names[method_name]
    else:
        return method_name  # Fallback to the raw method name

def get_objective_display_name(objective_name):
    """Get a clean display name for an objective function."""
    # Use the objective display name mappings, but with a fallback
    display_names = get_objective_function_display_names()
    if objective_name in display_names:
        return display_names[objective_name]
    else:
        # Fallback: capitalize first letter and add spaces before numbers
        import re
        formatted = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', objective_name)
        return formatted.title()

def filter_results_by_criteria(all_results, objective=None, dim=None, acquisition_function=None, seed=None):
    """Filter results dictionary based on criteria."""
    filtered_results = {}
    for result_dir, results in all_results.items():
        # Extract information from the result directory name
        dir_info = extract_info_from_result_dir(result_dir)
        
        # Check if all specified criteria match
        objective_match = objective is None or dir_info['objective'] == objective
        dim_match = dim is None or dir_info['dim'] == dim
        acq_match = acquisition_function is None or dir_info['acquisition_function'] == acquisition_function
        seed_match = seed is None or dir_info['seed'] == seed
        
        if objective_match and dim_match and acq_match and seed_match:
            filtered_results[result_dir] = results
    
    return filtered_results

def group_results_by_dimension(results):
    """Group results by dimension."""
    dimension_groups = {}
    for result_dir, result_data in results.items():
        dir_info = extract_info_from_result_dir(result_dir)
        dim = dir_info['dim']
        if dim not in dimension_groups:
            dimension_groups[dim] = {}
        dimension_groups[dim][result_dir] = result_data
    
    return dimension_groups

def ensure_output_dir(output_dir=None, prefix=None, acquisition_function=None, seed=None):
    """Ensure the output directory exists, create it if it doesn't."""
    if output_dir is None and prefix is not None:
        acq_suffix = f"_{acquisition_function}" if acquisition_function else ""
        seed_suffix = f"_{seed}" if seed is not None else ""
        output_dir = f"./{prefix}{acq_suffix}{seed_suffix}"
    
    if output_dir is None:
        output_dir = "./comparison_plots"
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    return output_dir

def save_plot_figure(fig, output_dir, filename, dpi=400):
    """Save a figure to the specified output directory."""
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    print(f"Saved plot to {filepath}")

def convert_to_numpy(data):
    """Convert data to numpy array regardless of input type."""
    if isinstance(data, list):
        return np.array(data)
    elif isinstance(data, torch.Tensor):
        return data.numpy()
    elif isinstance(data, np.ndarray):
        return data
    else:
        return np.array(data)

def sort_methods_within_group(methods):
    """Sort methods within a group according to their order in the predefined_map dictionary.
    
    This ensures that methods appear in the same order as defined in get_method_color_map's predefined_map.
    Methods not in the predefined_map will appear at the end, sorted alphabetically.
    """
    # Get the predefined order from the color map
    predefined_map = get_method_color_map()
    predefined_order = list(predefined_map.keys())
    
    def get_sort_key(method_name):
        if method_name in predefined_order:
            # Return the index in predefined_order for methods that are defined
            return (0, predefined_order.index(method_name))
        else:
            # For methods not in predefined_map, sort alphabetically after all predefined methods
            return (1, method_name)
    
    return sorted(methods, key=get_sort_key)

def group_methods_by_type(method_names):
    """Group methods by their type for organized legend display."""
    groups = {
        'Baseline': [],
        'Outcome Transformations (OT)': [],
        'Preprocessing Transformations (PT)': [],
        'Noise Injection (NI)': [],
        'Trust Region (TuRBO)': []
    }
    
    for method in method_names:
        if method.startswith('bo_'):
            groups['Baseline'].append(method)
        elif method.startswith('boot_'):
            groups['Outcome Transformations (OT)'].append(method)
        elif method.startswith('bopt_'):
            groups['Preprocessing Transformations (PT)'].append(method)
        elif method.startswith('boni_') or method.startswith('botrni_'):
            groups['Noise Injection (NI)'].append(method)
        elif method.startswith('turbo_') or method.startswith('turboni_'):
            groups['Trust Region (TuRBO)'].append(method)
        else:
            # Fallback to baseline for unknown methods
            groups['Baseline'].append(method)
    
    # Sort methods within each group
    for group_name in groups:
        groups[group_name] = sort_methods_within_group(groups[group_name])
    
    # Remove empty groups
    return {k: v for k, v in groups.items() if v}

def load_and_filter_results(base_path, objective=None, dim=None, seed=None, acquisition_function=None, methods=None):
    """
    Common function to load and filter results from the base path.
    
    Returns:
        tuple: (all_results, all_objectives, all_dimensions, all_acquisition_functions)
    """
    # Store results for each method
    all_results = {}
    
    # Keep track of all found acquisition functions and objectives
    all_acquisition_functions = set()
    all_objectives = set()
    all_dimensions = set()

    # Find all directories in the results folder
    result_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    print(f"Found result directories: {result_dirs}")
    
    # Filter directories based on all criteria at once
    for result_dir in result_dirs:
        dir_info = extract_info_from_result_dir(result_dir)
        
        # Combine all filters into one condition
        # Handle objective as either None, single value, or list
        objective_match = (objective is None or 
                          (isinstance(objective, list) and dir_info['objective'] in objective) or
                          (not isinstance(objective, list) and dir_info['objective'] == objective))
        
        if ((seed is None or dir_info['seed'] == seed) and
            objective_match and
            (dim is None or dir_info['dim'] == dim) and
            (acquisition_function is None or dir_info['acquisition_function'] == acquisition_function) and
            (methods is None or dir_info['method'] in methods)):
            
            # Track unique values we've found
            if dir_info['objective']: all_objectives.add(dir_info['objective'])
            if dir_info['dim']: all_dimensions.add(dir_info['dim'])
            if dir_info['acquisition_function']: all_acquisition_functions.add(dir_info['acquisition_function'])
            
            # Load results
            path = os.path.join(base_path, result_dir)
            results = load_results(path)
            
            if results is not None:
                all_results[result_dir] = results
                print(f"Loaded results for {result_dir}")

    if not all_results:
        print("No valid results found for any method")
        print("\nContents of results directory:")
        for root, dirs, files in os.walk(base_path):
            print(f"Directory: {root}")
            print(f"Subdirectories: {dirs}")
            print(f"Files: {files}")
            print()
        return None, None, None, None
    
    return all_results, all_objectives, all_dimensions, all_acquisition_functions

def create_grouped_legend(fig, ax, method_labels_dict, sorted_methods, max_methods_per_group, info_text="Lines show median values with IQR shading"):
    """
    Create a grouped legend below the plot(s) for combined view.
    
    Args:
        fig: matplotlib figure
        ax: matplotlib axis (or main axis for combined plots)
        method_labels_dict: dict mapping method names to (handle, display_name) tuples
        sorted_methods: list of method names in sorted order
        max_methods_per_group: maximum number of methods in any group
        info_text: descriptive text to show at bottom of legend
    """
    # Group methods by type
    method_groups = group_methods_by_type(list(method_labels_dict.keys()))
    
    # Get list of non-empty groups in a consistent order
    non_empty_groups = [(name, methods) for name, methods in method_groups.items() if methods]
    num_groups = len(non_empty_groups)
    
    if num_groups == 0:
        return
    
    # Create grid layout: organize by columns since matplotlib ncol fills column-wise
    all_handles = []
    all_labels = []
    
    # Fill column by column to match matplotlib's ncol behavior
    for col, (group_name, methods) in enumerate(non_empty_groups):
        # Add group header for this column
        header_handle = Line2D([0], [0], color='none', marker='none', linestyle='None')
        all_handles.append(header_handle)
        all_labels.append(f"{group_name}")
        
        # Add methods for this group
        for method in methods:
            if method in method_labels_dict:
                handle, display_name = method_labels_dict[method]
                all_handles.append(handle)
                all_labels.append(display_name)
        
        # Add padding to make all columns the same length
        methods_in_this_group = len(methods)
        padding_needed = max_methods_per_group - methods_in_this_group
        for _ in range(padding_needed):
            empty_handle = Line2D([0], [0], color='none', marker='none', linestyle='None')
            all_handles.append(empty_handle)
            all_labels.append("")
        
        # Add info text only for the first column, empty for others
        if col == 0:
            info_handle = Line2D([0], [0], color='none', marker='none', linestyle='None')
            all_handles.append(info_handle)
            all_labels.append(info_text)
        else:
            empty_handle = Line2D([0], [0], color='none', marker='none', linestyle='None')
            all_handles.append(empty_handle)
            all_labels.append("")
    
    # Create the unified legend with proper grid layout
    main_legend = fig.legend(
        all_handles,
        all_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=num_groups,
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        shadow=True,
        columnspacing=2.0,
        handletextpad=0.5
    )
    
    # Style the group headers and info text
    legend_texts = main_legend.get_texts()
    for i, text in enumerate(legend_texts):
        label_text = text.get_text()
        # Check if this is a group header
        if label_text in method_groups.keys():
            text.set_fontweight('bold')
        # Style the info text
        elif info_text and label_text.startswith(info_text.split()[0]):
            text.set_fontstyle('italic')

def setup_plot_style(plot_style):
    """Set up consistent plot styling."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(plot_style)

def apply_method_alpha_to_boxplot(bp, method_names, color_map, method_alpha=None):
    """Apply consistent colors and alpha values to boxplot elements."""
    # Apply consistent colors and alpha values to boxes
    for i, (patch, method) in enumerate(zip(bp['boxes'], method_names)):
        color = color_map[method]
        box_alpha = method_alpha.get(method, 0.5) if method_alpha else 0.5
        patch.set_facecolor(color)
        patch.set_alpha(box_alpha)
    
    # Color the median lines to match the method colors
    for i, (median_line, method) in enumerate(zip(bp['medians'], method_names)):
        color = color_map[method]
        median_alpha = method_alpha.get(method, 1.0) if method_alpha else 1.0
        median_line.set_color(color)
        median_line.set_linewidth(2)
        median_line.set_alpha(median_alpha)

################################################################################
# Thesis utilities
################################################################################

def get_thesis_figure_size():
    """Calculate thesis-appropriate figure size based on text width."""
    pt = 1./72.27  # 72.27 points to an inch
    text_width = 398.33862
    fig_width = text_width * pt
    golden = (1 + 5 ** 0.5) / 2
    fig_height = fig_width / golden
    return fig_width, fig_height

def get_thesis_color_map(method_names):
    """Get RWTH color mapping for methods."""
    color_map = {}
    # Use RWTH colors in order
    colors = [
        RWTH_COLORS['blue'][0],
        RWTH_COLORS['petrol'][0], 
        RWTH_COLORS['green'][0],
        RWTH_COLORS['orange'][0],
        RWTH_COLORS['red'][0],
        RWTH_COLORS['magenta'][0],
        RWTH_COLORS['tuerkis'][0],
        RWTH_COLORS['maigreen'][0],
        RWTH_COLORS['bordeaux'][0],
        RWTH_COLORS['violet'][0],
        RWTH_COLORS['lila'][0],
        RWTH_COLORS['yellow'][0],
    ]
    
    for i, method in enumerate(method_names):
        color_map[method] = colors[i % len(colors)]
    
    return color_map

def create_thesis_legend(fig, ax, method_labels_dict, sorted_methods):
    """
    Create a grouped legend below the plot with one column per method group (no background, no headers).
    Standardized version used across all thesis files.
    
    Args:
        fig: matplotlib figure
        ax: matplotlib axis 
        method_labels_dict: dict mapping method names to (handle, display_name) tuples
        sorted_methods: list of method names in sorted order
    """
    # Group methods by type to maintain column structure
    method_groups = group_methods_by_type(list(method_labels_dict.keys()))
    
    # Get list of non-empty groups in a consistent order
    non_empty_groups = [(name, methods) for name, methods in method_groups.items() if methods]
    num_groups = len(non_empty_groups)
    
    if num_groups == 0:
        return
    
    # Create grid layout: organize by columns since matplotlib ncol fills column-wise
    all_handles = []
    all_labels = []
    
    # Find the maximum number of methods in any group
    max_methods_per_group = max(len(methods) for _, methods in non_empty_groups)
    
    # Fill column by column to match matplotlib's ncol behavior
    for col, (group_name, methods) in enumerate(non_empty_groups):
        # Add methods for this group (no group header)
        for method in methods:
            if method in method_labels_dict:
                handle, display_name = method_labels_dict[method]
                all_handles.append(handle)
                all_labels.append(display_name)
        
        # Add padding to make all columns the same length
        methods_in_this_group = len(methods)
        padding_needed = max_methods_per_group - methods_in_this_group
        for _ in range(padding_needed):
            all_handles.append(Line2D([0], [0], alpha=0))  # Invisible handle
            all_labels.append("")  # Empty label
    
    # Create the unified legend with proper grid layout
    legend = fig.legend(
        all_handles,
        all_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=num_groups,  # One column per group
        frameon=False,  # No background frame
        columnspacing=2.0,
        handletextpad=0.5
    )
    
    return legend

def setup_thesis_style():
    """Set up thesis-specific plot style using paper.mplstyle."""
    # Apply thesis style - use full path to paper.mplstyle
    script_dir = os.path.dirname(os.path.abspath(__file__))
    style_path = os.path.join(script_dir, 'paper.mplstyle')
    if os.path.exists(style_path):
        plt.style.use(style_path)

def sort_objectives_by_name_and_dimension(objectives):
    """Sort objectives by function name and dimension for consistent ordering."""
    def parse_objective(obj_name):
        # Extract function name and dimension from objective name
        # E.g., "ackley10D" -> ("ackley", 10)
        import re
        match = re.match(r'([a-zA-Z]+)(\d+)D?', obj_name)
        if match:
            func_name = match.group(1).lower()
            dim = int(match.group(2))
            return (func_name, dim)
        else:
            return (obj_name.lower(), 0)
    
    return sorted(objectives, key=parse_objective)

def get_thesis_main_params():
    """Get standard parameter structure for thesis plotting main functions."""
    return {
        'objective': None,  # Set to None to detect all objectives
        'dim': None,        # Set to None to detect all dimensions
        'seed': 0,
        'methods': None,
        'acquisition_function': None,  # Set to None to detect all acquisition functions
        'save_plots': False,
        'output_dir': None,
        'sweep': "debug",
        'method_alpha': None,  # Dict to control alpha/transparency of specific methods
    }

def create_thesis_output_dir(output_dir, plot_type="thesis"):
    """Create standardized output directory for thesis plots."""
    if output_dir is None:
        output_dir = f"figures/{plot_type}/"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_thesis_plot(fig, output_dir, filename, dpi=300):
    """Save a thesis plot with standardized settings."""
    # Ensure PDF extension for thesis plots
    if not filename.endswith('.pdf'):
        filename = filename.replace('.png', '.pdf')
    
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved thesis plot: {filepath}")

def get_thesis_legend_handles_labels(ax, method_names, suffix_to_remove=""):
    """Extract method names from legend labels for thesis plots."""
    handles, labels = ax.get_legend_handles_labels()
    
    # Extract method names from labels
    method_labels = {}
    for h, l in zip(handles, labels):
        if suffix_to_remove and suffix_to_remove in l:
            method_name = l.replace(suffix_to_remove, "")
        else:
            method_name = l
        
        # Find the original method key by matching display name
        for method_key in method_names:
            if get_method_display_name(method_key) == method_name:
                method_labels[method_key] = (h, method_name)
                break
    
    return method_labels