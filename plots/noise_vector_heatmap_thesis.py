"""
Generates heatmap visualizations of noise injection patterns on 2D objective functions.
Chart types: 2D heatmap plots showing noise vector evolution and function landscapes.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    setup_project_path, 
    extract_info_from_result_dir, 
    get_acquisition_function_display_names,
    get_objective_function_display_names,
    filter_results_by_criteria,
    convert_to_numpy,
    load_and_filter_results,
    RWTH_COLORS,
    get_thesis_figure_size,
    create_thesis_output_dir,
    save_thesis_plot,
    setup_thesis_style,
    get_thesis_color_map,
)

def main(
    objective="ackley2D",
    acquisition_function="LogExpectedImprovement",
    methods=None,
    save_plots=True,
    output_dir="figures/thesis/results/",
    sweep="final",
    noise_threshold=0.5,
    bounds=3.0,
    seed=0,
):
    
    # Default methods for thesis comparison
    if methods is None:
        methods = [
            "turboni_standardize",
            "turboni_tr", 
            "turboni_trbinary",
            "turboni_trbs",
        ]
    
    # Support individual bounds per subplot
    if isinstance(bounds, (float, int)):
        bounds = [float(bounds)] * len(methods)
    elif isinstance(bounds, list):
        if len(bounds) != len(methods):
            raise ValueError("Length of bounds list must match number of methods.")
    else:
        raise ValueError("bounds must be a float or a list of floats.")
    
    # Setup project and style
    project_root = setup_project_path()
    setup_thesis_style()
    
    # Base path for results
    base_path = os.path.join(project_root, "results", sweep)
    
    if not os.path.exists(base_path):
        print(f"Results directory not found at {base_path}")
        return

    # Create output directory if saving plots
    if save_plots:
        output_dir = create_thesis_output_dir(output_dir, "thesis/results")

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
    
    # Filter for only methods that have noise vectors (noise injection methods)
    noise_results = {}
    for result_dir, result_data in all_results.items():
        if "noise_vectors" in result_data and result_data["noise_vectors"]:
            # Check if any noise vectors are not None
            if any(nv is not None for nv in result_data["noise_vectors"]):
                noise_results[result_dir] = result_data
    
    if not noise_results:
        print("No results with noise vectors found. This visualization is only applicable to noise injection methods.")
        return
    
    print(f"Found {len(noise_results)} results with noise vectors")
    
    # Generate thesis-style 4-panel comparison
    print(f"\nGenerating thesis-style 4-panel Ackley 2D function with noise injection comparison")
    generate_thesis_ackley_function_plot(
        all_results=noise_results,
        objective=objective,
        acquisition_function=acquisition_function,
        methods=methods,
        save_plots=save_plots,
        output_dir=output_dir,
        noise_threshold=noise_threshold,
        bounds=bounds
    )


def generate_thesis_ackley_function_plot(all_results, objective, acquisition_function, methods, 
                                       save_plots=False, output_dir=None, noise_threshold=0.5, bounds=None):
    
    # Get utility elements
    acq_display_names = get_acquisition_function_display_names()
    obj_display_names = get_objective_function_display_names()
    
    # Filter results for the specified objective and acquisition function
    filtered_results = filter_results_by_criteria(
        all_results, 
        objective=objective,
        acquisition_function=acquisition_function
    )
    
    if not filtered_results:
        print(f"No results with noise vectors found for objective: {objective}, acquisition function: {acquisition_function}")
        return
    
    # Get thesis figure size and make it wider for 4 panels
    fig_width, fig_height = get_thesis_figure_size()
    fig_width *= 1.1  # Make it wider for 4 panels (experimented)
    fig_height *= 0.75
    
    # Create figure with 4 subplots in one row
    fig, axes = plt.subplots(1, 4, figsize=(fig_width, fig_height), sharey=True)
    
    # Use thesis color map for methods
    method_colors = get_thesis_color_map(methods)
    
    # Method display names
    method_display_names = {
        'boni_standardizenonoise': 'Naive-NI',
        'boni_ilsstandardizenonoise': 'ILS-NI',
        'turboni_tr': 'TR-NI',
        'turboni_tradditivenorm': 'A-TR-NI',
    }
    
    # Process each method
    all_scatters = []  # Store all scatter objects for shared colorbar
    for idx, method_name in enumerate(methods[:4]):  # Limit to 4 methods
        if idx >= 4:
            break
            
        ax = axes[idx]
        
        # Find results for this method
        method_results = {}
        for result_dir, results in filtered_results.items():
            dir_info = extract_info_from_result_dir(result_dir)
            if dir_info['method'] == method_name:
                method_results[result_dir] = results
        
        if not method_results:
            print(f"No results found for method: {method_name}")
            ax.text(0.5, 0.5, f'No data\nfor {method_name}', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title(method_display_names.get(method_name, method_name))
            all_scatters.append(None)
            continue
        
        bound = bounds[idx] if bounds is not None else 3.0
        # Create Ackley 2D function visualization with noise injection
        scatter = plot_ackley_with_noise_injection(ax, method_results, method_name, method_colors.get(method_name, f"C{idx}"), bounds=bound)
        all_scatters.append(scatter)

        # Set title for this subplot
        ax.set_title(method_display_names.get(method_name, method_name))

        # Set y-label only for the first subplot
        if idx == 0:
            ax.set_ylabel(r'$x_2$', labelpad=3)

    # Create a proper column setup for the two colorbars
    valid_scatters = [scatter for scatter in all_scatters if scatter is not None]
    if valid_scatters:
        # Get the first valid scatter and contour for colorbars
        first_scatter = valid_scatters[0]
        first_contour = None
        for ax in axes:
            if hasattr(ax, 'contour_data'):
                first_contour = ax.contour_data
                break
        
        if first_contour is not None:
            # Create figure with subplots for colorbars
            # Calculate positions for the two colorbars side by side
            fig_width = fig.get_size_inches()[0]
            fig_height = fig.get_size_inches()[1]
            
            # Position colorbars below the main plots
            cbar_height = 0.04  # Height of each colorbar
            cbar_width = 0.35   # Width of each colorbar
            cbar_y = 0.15       # Y position of colorbars
            cbar_spacing = 0.1 # Spacing between colorbars
            
            # Left colorbar (noise values)
            cbar_noise_x = 0.15
            cbar_noise = fig.add_axes([cbar_noise_x, cbar_y, cbar_width, cbar_height])
            cbar_noise = fig.colorbar(first_scatter, cax=cbar_noise, orientation='horizontal')
            cbar_noise.set_label(r'$\lambda_i$')
            
            # Right colorbar (function values)
            cbar_func_x = cbar_noise_x + cbar_width + cbar_spacing
            cbar_func = fig.add_axes([cbar_func_x, cbar_y, cbar_width, cbar_height])
            cbar_func = fig.colorbar(first_contour, cax=cbar_func, orientation='horizontal')
            cbar_func.set_label(r'$f(x_1, x_2)$')
            
            # Reduce the number of ticks on the function colorbar and set range to start from 0
            cbar_func.ax.xaxis.set_major_locator(plt.MaxNLocator(5))  # Show only 5 ticks
            
            print("Both colorbars created successfully in column setup")
        else:
            print("No contour data found for function colorbar")

    # Add single x-label centered under all subplots
    fig.text(0.5, 0.25, r'$x_1$', ha='center', va='bottom')
    
    # Adjust layout to accommodate colorbar
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.42, wspace=0.1)
    
    # Save plot if requested
    if save_plots:
        filename = f"ackley_function_thesis_{objective}_{acquisition_function}.pdf"
        save_thesis_plot(plt.gcf(), output_dir, filename)
    
    plt.show()


def plot_ackley_with_noise_injection(ax, method_results, method_name, color, bounds=3.0):
    
    # Define Ackley function
    def ackley_2d(x, y):
        a = 20
        b = 0.2
        c = 2 * np.pi
        d = 2  # dimension
        
        sum1 = x**2 + y**2
        sum2 = np.cos(c * x) + np.cos(c * y)
        
        term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
        term2 = -np.exp(sum2 / d)
        
        return term1 + term2 + a + np.exp(1)
    
    # Create grid for function visualization
    x_range = np.linspace(-bounds, bounds, 100)
    y_range = np.linspace(-bounds, bounds, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = ackley_2d(X, Y)
    
    # Plot function as contour with higher resolution for better gradient
    contour = ax.contourf(X, Y, Z, levels=100, cmap='viridis', alpha=0.7)
    ax.contour(X, Y, Z, levels=30, colors='black', alpha=0.3, linewidths=0.5)
    
    # Store the contour for colorbar creation
    ax.contour_data = contour
    print(f"Stored contour data for {method_name}")
    
    # Extract noise vector data and training points for this method
    all_noise_sequences = []
    training_points = []
    
    for result_dir, results in method_results.items():
        if "noise_vectors" in results and results["noise_vectors"]:
            # Convert noise vectors to numpy arrays and filter out None values
            noise_vectors = []
            for nv in results["noise_vectors"]:
                if nv is not None:
                    noise_vectors.append(convert_to_numpy(nv))
            
            if noise_vectors:
                all_noise_sequences.append(noise_vectors)
        
        # Extract training points
        if 'history_x' in results and 'history_y' in results:
            # Convert optimization history to numpy arrays
            history_x = convert_to_numpy(results['history_x'])
            history_y = convert_to_numpy(results['history_y'])
            
            # Store the actual training points
            if len(history_x.shape) > 1 and history_x.shape[1] == 2:  # 2D case
                training_points.append(history_x)
            elif len(history_x.shape) == 1:  # 1D case, reshape for consistency
                training_points.append(history_x.reshape(-1, 1))
    
    # Overlay noise information if available
    if all_noise_sequences and training_points:
        # Get the latest iteration data (last noise vectors)
        final_noise_vectors = []
        
        # Collect final noise vectors across seeds
        for seed_sequence in all_noise_sequences:
            if seed_sequence:  # Make sure sequence is not empty
                final_noise_vec = seed_sequence[-1]  # Last iteration
                final_noise_vectors.append(final_noise_vec)
        
        # Average the noise vectors across seeds
        avg_noise_vector = np.mean(final_noise_vectors, axis=0)
        avg_training_points = np.mean(training_points, axis=0)
        
        # Ensure noise vector and training points have matching lengths
        min_length = min(len(avg_noise_vector), len(avg_training_points))
        avg_noise_vector = avg_noise_vector[:min_length]
        avg_training_points = avg_training_points[:min_length]
        
        # Plot training points with noise information
        if avg_training_points.shape[1] == 2:  # 2D case
            # Determine vmax based on actual noise data
            noise_max = np.max(avg_noise_vector) if len(avg_noise_vector) > 0 else 1.0
            vmax = noise_max  # Use actual noise data maximum
            
            # Define custom red-yellow-light blue colormap for lambda values
            import matplotlib.colors as mcolors
            from matplotlib.colors import LinearSegmentedColormap
            custom_colors = [RWTH_COLORS["red"][0], "#fee08b", RWTH_COLORS["blue"][0]]
            cmap = LinearSegmentedColormap.from_list("RedYellowBlue", custom_colors)
            
            scatter = ax.scatter(avg_training_points[:, 0], avg_training_points[:, 1], 
                               c=avg_noise_vector, cmap=cmap, 
                               s=30, edgecolors='black', linewidths=1.0, 
                               vmin=0, vmax=vmax, alpha=0.9, zorder=5)
        elif avg_training_points.shape[1] == 1:  # 1D case
            # For 1D, we need to evaluate the function at training points to get y-coordinates
            train_y = []
            for x_val in avg_training_points[:, 0]:
                train_y.append(ackley_2d(x_val, 0))  # Use y=0 for 1D case
            train_y = np.array(train_y)
            
            # Determine vmax based on actual noise data
            noise_max = np.max(avg_noise_vector) if len(avg_noise_vector) > 0 else 1.0
            vmax = noise_max  # Use actual noise data maximum
            
            scatter = ax.scatter(avg_training_points[:, 0], train_y, 
                               c=avg_noise_vector, cmap=cmap, 
                               s=30, edgecolors='black', linewidths=1.0, 
                               vmin=0, vmax=vmax, alpha=0.9, zorder=5)
        
        ax.set_xlim(-bounds, bounds)
        ax.set_ylim(-bounds, bounds)
        
        return scatter
    
    ax.set_xlim(-bounds, bounds)
    ax.set_ylim(-bounds, bounds)
    
    return None


##################################################################
# Configuration
##################################################################

if __name__ == "__main__":
    
    project_root = setup_project_path()
    setup_thesis_style()

    main(
        objective="ackley2D",
        acquisition_function="LogExpectedImprovement",
        methods=[
            "boni_standardizenonoise",
            "boni_ilsstandardizenonoise",
            "turboni_tr", 
            "turboni_tradditivenorm",
        ],
        save_plots=True,
        output_dir="figures/thesis/results/",
        sweep="final",
        noise_threshold=0.5,
        bounds=7.0,
        seed=4,
    ) 