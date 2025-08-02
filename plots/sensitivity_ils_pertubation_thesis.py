"""
Sensitivity analysis for iterated local search perturbation strength on optimization performance.
Chart types: error bars showing performance sensitivity to perturbation parameter variations.
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
    sweep="sensitivity_ils"
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
    all_results = load_all_ils_results(base_path)
    
    if not all_results:
        print("No results found")
        return
    
    # Generate sensitivity plot
    print("Generating ILS sensitivity analysis plot")
    generate_ils_sensitivity_plot(
        all_results=all_results,
        save_plots=save_plots,
        output_dir=output_dir
    )


def load_all_ils_results(base_path):

    all_results = {}
    
    if not os.path.exists(base_path):
        print(f"Base path does not exist: {base_path}")
        return all_results
    
    # Get all result directories
    result_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    for result_dir in result_dirs:
        result_path = os.path.join(base_path, result_dir)
        
        # Extract information from directory name
        # Format: boni_ilsstandardize_ackley2D_2_UpperConfidenceBound_pert0.05_0
        parts = result_dir.split('_')
        
        if len(parts) < 6:
            continue
            
        # Extract objective and dimension
        objective_name = parts[2]  # ackley2D
        dim = int(parts[3])        # 2
        
        # Extract acquisition function
        acq_func = parts[4]        # UpperConfidenceBound
        
        # Extract perturbation size
        pert_part = parts[5]       # pert0.05
        if pert_part.startswith('pert'):
            perturbation_size = float(pert_part[4:])
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
                    key = (objective_name, dim, acq_func, perturbation_size, seed)
                    all_results[key] = final_result
                    
            except Exception as e:
                print(f"Error loading {results_file}: {e}")
                continue
    
    return all_results


def generate_ils_sensitivity_plot(all_results, save_plots=False, output_dir=None):
    
    # Define objectives to plot
    objectives = [
        ("ackley2D", 2),
        ("ackley10D", 10), 
        ("rastrigin2D", 2),
        ("rastrigin10D", 10)
    ]
    
    # Define perturbation sizes
    perturbation_sizes = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Define acquisition functions
    acquisition_functions = ["UpperConfidenceBound", "LogExpectedImprovement"]
    
    # Create figure with 2x4 subplots (2 rows for acquisition functions, 4 columns for objectives)
    height, width = get_thesis_figure_size()
    height = height * 0.4 # Increase height for 2 rows
    width = width * 1.6

    fig, axes = plt.subplots(2, 4, figsize=(width, height))
    
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
            
            for pert_size in perturbation_sizes:
                pert_results = []
                
                # Collect results for all seeds
                for seed in range(10):  # 0-9
                    key = (objective_name, dim, acq_func, pert_size, seed)
                    if key in all_results:
                        pert_results.append(all_results[key])
                
                if pert_results:
                    results_for_acq.append(pert_results)
            
            # Plot vertical dot plot for this acquisition function
            if results_for_acq:
                plot_vertical_dot_plot(
                    ax, results_for_acq, perturbation_sizes, 
                    color=acq_colors[acq_func], 
                    marker=acq_markers[acq_func],
                    label=get_acquisition_function_display_names().get(acq_func, acq_func)
                )
            
            # Customize subplot
            if acq_idx == 1:  # Only show x-label on bottom row
                ax.set_xlabel("Perturbation size")
            if obj_idx == 0:  # Only show y-label on leftmost column
                ax.set_ylabel("Best value")
            
            # Set title for top row only
            if acq_idx == 0:
                ax.set_title(f"{obj_display_name}")
            
            # Set x-axis ticks to 0, 0.5, and 1
            ax.set_xticks([0, 0.5, 1])
            ax.set_xticklabels(['0', '0.5', '1'])
            
            # Special handling for Ackley 10D in bottom row (second chart, bottom row)
            if acq_idx == 1 and obj_idx == 1:  # Bottom row, second column
                # Get current ticks and format to max 2 digits
                yticks = ax.get_yticks()
                yticks = yticks[(yticks >= ax.get_ylim()[0]) & (yticks <= ax.get_ylim()[1])]
                
                # Remove duplicates and format
                yticks = np.unique(yticks)
                formatted_labels = []
                for tick in yticks:
                    if abs(tick) >= 100:
                        formatted_labels.append(f'{tick:.0e}')  # Scientific notation
                    elif abs(tick) >= 10:
                        formatted_labels.append(f'{int(round(tick))}')  # Integer
                    elif abs(tick) >= 1:
                        formatted_labels.append(f'{int(round(tick))}')  # Integer
                    elif tick == 0:
                        formatted_labels.append('0')
                    else:
                        formatted_labels.append(f'{tick:.1f}')  # One decimal
                
                ax.set_yticks(yticks)
                ax.set_yticklabels(formatted_labels)
            else:
                format_axis_intelligently(ax)
    
    # Adjust layout for 2x4 subplot grid
    plt.subplots_adjust(
        left=0.10,      # Left margin
        right=0.95,     # Right margin  
        top=0.95,       # Top margin (space for titles)
        bottom=0.10,    # Bottom margin (for x-labels)
        wspace=0.38,     # Width spacing between subplots
        hspace=0.35      # Height spacing between subplot rows
    )
    
    if save_plots and output_dir:
        filename = "ils_sensitivity_analysis.pdf"
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
    yticks = ax.get_yticks()
    yticks = yticks[(yticks >= ax.get_ylim()[0]) & (yticks <= ax.get_ylim()[1])]
    if np.all(np.abs(yticks - np.round(yticks)) < 1e-8):
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
    else:
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))


if __name__ == "__main__":
    setup_thesis_style()
    
    main() 