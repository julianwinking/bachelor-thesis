"""
Sensitivity analysis for beam search beam width parameter effects on optimization performance.
Chart types: error bars showing parameter sensitivity across different beam width values.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from utils import (
    setup_project_path, 
    get_acquisition_function_display_names,
    get_objective_display_name,
    convert_to_numpy,
    RWTH_COLORS,
    get_thesis_figure_size,
    setup_thesis_style,
    save_thesis_plot,
)
import matplotlib.ticker as mticker

def main(
    save_plots=True,
    output_dir="figures/thesis/sensitivity",
    sweep="sensitivity_bs"
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
    all_results = load_all_bs_results(base_path)
    
    if not all_results:
        print("No results found")
        return
    
    # Generate sensitivity plot
    print("Generating Beam Search sensitivity analysis plot")
    generate_bs_sensitivity_plot(
        all_results=all_results,
        save_plots=save_plots,
        output_dir=output_dir
    )


def load_all_bs_results(base_path):
    all_results = {}
    
    if not os.path.exists(base_path):
        print(f"Base path does not exist: {base_path}")
        return all_results
    
    # Get all result directories
    result_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    for result_dir in result_dirs:
        result_path = os.path.join(base_path, result_dir)
        
        # Extract information from directory name
        # Format: boni_bsnonoise_ackley2D_2_UpperConfidenceBound_beam1_0
        parts = result_dir.split('_')
        
        if len(parts) < 6:
            continue
            
        # Extract objective and dimension
        objective_name = parts[2]  # ackley2D
        dim = int(parts[3])        # 2
        
        # Extract acquisition function
        acq_func = parts[4]        # UpperConfidenceBound
        
        # Extract beam width
        beam_part = parts[5]       # beam1
        if beam_part.startswith('beam'):
            beam_width = int(beam_part[4:])
        else:
            continue
            
        # Extract seed
        seed = int(parts[6])       # 0
        
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
                    key = (objective_name, dim, acq_func, beam_width, seed)
                    all_results[key] = final_result
                    
            except Exception as e:
                print(f"Error loading {results_file}: {e}")
                continue
    
    return all_results


def generate_bs_sensitivity_plot(all_results, save_plots=False, output_dir=None):
    
    setup_thesis_style()
    
    # Define objectives to plot
    objectives = [
        ("ackley2D", 2),
        ("ackley10D", 10), 
        ("rastrigin2D", 2)
    ]
    
    # Define beam widths
    beam_widths = [1, 2, 3, 5, 10]
    
    # Define acquisition functions
    acquisition_functions = ["UpperConfidenceBound", "LogExpectedImprovement"]
    
    # Create figure with 2x3 subplots (2 rows for acquisition functions, 3 columns for objectives)
    height, width = get_thesis_figure_size()
    height = height * 0.4
    width = width * 1.6

    fig, axes = plt.subplots(2, 3, figsize=(width, height))
    
    # Colors for acquisition functions
    acq_colors = {
        "UpperConfidenceBound": RWTH_COLORS['green'][0],
        "LogExpectedImprovement": RWTH_COLORS['tuerkis'][0]
    }
    
    # Markers for acquisition functions
    acq_markers = {
        "UpperConfidenceBound": "o",
        "LogExpectedImprovement": "s"
    }
    
    # Process each acquisition function (row) and objective (column)
    for acq_idx, acq_func in enumerate(acquisition_functions):
        for obj_idx, (objective_name, dim) in enumerate(objectives):
            ax = axes[acq_idx, obj_idx]
            
            # Get objective display name
            obj_display_name = get_objective_display_name(objective_name)
            
            # Collect data for this objective and acquisition function
            results_for_acq = []
            
            for beam_width in beam_widths:
                beam_results = []
                
                # Collect results for all seeds
                for seed in range(10):  # 0-9
                    key = (objective_name, dim, acq_func, beam_width, seed)
                    if key in all_results:
                        beam_results.append(all_results[key])
                
                if beam_results:
                    results_for_acq.append(beam_results)
            
            # Plot vertical dot plot for this acquisition function
            if results_for_acq:
                plot_vertical_dot_plot(
                    ax, results_for_acq, beam_widths, 
                    color=acq_colors[acq_func], 
                    marker=acq_markers[acq_func],
                    label=get_acquisition_function_display_names().get(acq_func, acq_func)
                )
            
            # Customize subplot
            if acq_idx == 1:  # Only show x-label on bottom row
                ax.set_xlabel("Beam width")
            if obj_idx == 0:  # Only show y-label on leftmost column
                ax.set_ylabel("Best value")
            
            # Set title for top row only
            if acq_idx == 0:
                ax.set_title(f"{obj_display_name}")
            
            # Set x-axis ticks
            ax.set_xticks(beam_widths)
            ax.set_xticklabels([str(bw) for bw in beam_widths])
            format_axis_intelligently(ax)
    
    # Adjust layout for 2x4 subplot grid
    plt.subplots_adjust(
        left=0.10,      # Left margin
        right=0.95,     # Right margin  
        top=0.95,       # Top margin (space for titles)
        bottom=0.10,    # Bottom margin (for x-labels)
        wspace=0.35,     # Width spacing between subplots
        hspace=0.35      # Height spacing between subplot rows
    )
    
    if save_plots and output_dir:
        filename = "bs_sensitivity_analysis.pdf"
        save_thesis_plot(fig, output_dir, filename)
        print(f"Saved sensitivity plot to {os.path.join(output_dir, filename)}")
    
    plt.show()


def plot_vertical_dot_plot(ax, results_list, x_positions, color, marker, label):
    
    for i, (results, x_pos) in enumerate(zip(results_list, x_positions)):
        # Calculate median and error
        median_val = np.median(results)
        
        # Calculate error as interquartile range (IQR)
        q25 = np.percentile(results, 25)
        q75 = np.percentile(results, 75)
        error_lower = median_val - q25 
        error_upper = q75 - median_val
        
        # Plot median value as a dot with asymmetric error bars showing full IQR
        ax.errorbar(x_pos, median_val, yerr=[[error_lower], [error_upper]],
                   c=color, marker=marker, markersize=3, capsize=3, capthick=1,
                   alpha=1, linewidth=0.5)


def format_axis_intelligently(ax):
    def custom_formatter(x, pos):
        # Check if the value is close to an integer
        if abs(x - round(x)) < 1e-8:
            return f'{int(round(x))}'
        else:
            return f'{x:.1f}'
    
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_formatter))


if __name__ == "__main__":
    main() 