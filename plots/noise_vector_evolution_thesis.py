"""
Thesis visualization of noise vector evolution patterns in noise injection optimization methods.
Chart types: heatmaps showing adaptive noise patterns across optimization iterations.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle

from utils import (
    setup_project_path, 
    extract_info_from_result_dir, 
    get_method_color_map,
    filter_results_by_criteria,
    convert_to_numpy,
    RWTH_COLORS,
    get_thesis_figure_size,
    create_thesis_output_dir,
    save_thesis_plot,
    load_and_filter_results,
    setup_thesis_style,
)

def main(objective=None, seed=0, methods=None, acquisition_function=None, save_plots=False, output_dir=None, sweep="final", noise_threshold=0.5):
    # Base path for results
    project_root = setup_project_path()
    base_path = os.path.join(project_root, "results", sweep)
    
    if not os.path.exists(base_path):
        print(f"Results directory not found at {base_path}")
        return

    if save_plots:
        output_dir = create_thesis_output_dir(output_dir, "thesis/results")

    # Use utils.py function to load and filter results
    all_results, all_objectives, all_dimensions, all_acquisition_functions = load_and_filter_results(
        base_path, objective=objective, dim=None, seed=seed, acquisition_function=acquisition_function, methods=methods
    )
    
    if all_results is None:
        print("No valid results found")
        return

    print(f"Loaded {len(all_results)} result directories for seed {seed}")
    print(f"Found objectives: {all_objectives}")
    print(f"Found dimensions: {all_dimensions}")
    print(f"Found acquisition functions: {all_acquisition_functions}")
    
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
    
    # Generate thesis-style comparison plot
    if methods:
        print(f"\nGenerating thesis-style 4-panel noise vector comparison")
        generate_thesis_noise_vector_plot(
            all_results=noise_results,
            objective=objective,
            acquisition_function=acquisition_function,
            methods=methods,
            save_plots=save_plots,
            output_dir=output_dir,
            noise_threshold=noise_threshold
        )


def generate_thesis_noise_vector_plot(all_results, objective, acquisition_function, methods, 
                                    save_plots=False, output_dir=None, noise_threshold=0.5):
    
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
    fig_width *= 1.1  # Make it wider for 4 panels to fit letter height of thesis (experimented)
    fig_height *= 0.75
    
    # Create figure with 4 subplots in one row
    fig, axes = plt.subplots(1, 4, figsize=(fig_width, fig_height), sharey=True)
    
    # Use utils.py color mapping and display names
    color_map = get_method_color_map(methods)
    display_names = {
        'boni_standardizenonoise': 'Naive-NI',
        'boni_ilsstandardizenonoise': 'ILS-NI',
        'turboni_tr': 'TR-NI',
        'turboni_tradditivenorm': 'A-TR-NI',
    }
    
    # Process each method
    all_ims = []  # Store all image objects for shared colorbar
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
            ax.set_title(display_names.get(method_name, method_name))
            all_ims.append(None)
            continue
        
        # Extract noise vector data for this method (across seeds)
        all_noise_sequences = []
        for result_dir, results in method_results.items():
            if "noise_vectors" in results and results["noise_vectors"]:
                noise_vectors = []
                for nv in results["noise_vectors"]:
                    if nv is not None:
                        noise_vectors.append(convert_to_numpy(nv))
                if noise_vectors:
                    all_noise_sequences.append(noise_vectors)
        
        if not all_noise_sequences:
            print(f"No valid noise vector data for method: {method_name}")
            ax.text(0.5, 0.5, f'No noise data\nfor {method_name}', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title(display_names.get(method_name, method_name))
            all_ims.append(None)
            continue
        
        # Process noise vector data
        processed_data = process_noise_vector_data(all_noise_sequences, noise_threshold)
        
        # Create noise heatmap for this method
        color = color_map.get(method_name, f"C{idx}")
        im = plot_thesis_noise_heatmap(ax, processed_data, method_name, color, noise_threshold)
        all_ims.append(im)

        # Set title for this subplot
        ax.set_title(display_names.get(method_name, method_name))

        # Set y-label only for the first subplot
        if idx == 0:
            ax.set_ylabel('Training point index')

    # Add shared colorbar
    valid_ims = [im for im in all_ims if im is not None]
    if valid_ims:
        # Create colorbar using the first valid image
        cbar = fig.colorbar(valid_ims[0], ax=axes, orientation='horizontal', 
                           fraction=0.05, pad=0.08, aspect=30)
        
        # Add lambda label to the left of the colorbar at the same height
        cbar_pos = cbar.ax.get_position()
        fig.text(cbar_pos.x0, cbar_pos.y0 + cbar_pos.height/2, r"$\lambda_i$", 
                ha='right', va='center')

    # Add single x-label centered under all subplots
    fig.text(0.5, 0.2, 'Iteration', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.38, wspace=0.1)
    
    if save_plots:
        filename = f"noise_vector_thesis_{objective}_{acquisition_function}.pdf"
        save_thesis_plot(plt.gcf(), output_dir, filename)
    
    plt.show()


def plot_thesis_noise_heatmap(ax, processed_data, method_name, color, noise_threshold):
    if 'heatmap' not in processed_data:
        return
    
    heatmap_data = processed_data['heatmap']
    first_visible = processed_data.get('first_visible_iteration', np.full(heatmap_data.shape[1], 0))
    
    # Create masked array where NaN values (unobserved points) are masked
    masked_data = np.ma.masked_invalid(heatmap_data.T)
    
    # Create heatmap with custom colormap: red-yellow-light blue
    from matplotlib.colors import LinearSegmentedColormap

    # Define custom red-yellow-light blue colormap (red-green colorblind-friendly)
    custom_colors = [RWTH_COLORS["red"][0], "#fee08b", RWTH_COLORS["blue"][0]]
    cmap = LinearSegmentedColormap.from_list("RedYellowBlue", custom_colors)
    cmap.set_bad(color='lightgray', alpha=0.3)  # Color for unobserved points
    
    # Determine vmax based on actual data range
    data_max = np.nanmax(heatmap_data) if not np.all(np.isnan(heatmap_data)) else 1.0
    vmax = max(1.0, data_max)  # At least 1.0 to accommodate noise factor 1

    im = ax.imshow(masked_data, aspect='auto', cmap=cmap.reversed(), vmin=0, vmax=vmax,
                   interpolation='nearest', origin='lower')
    
    ax.set_yticks([0, 125, 250, 375, 500])
    
    return im


def process_noise_vector_data(all_noise_sequences, noise_threshold=0.5):
    # Find the maximum length across all sequences
    max_len = 0
    for sequence in all_noise_sequences:
        for noise_vec in sequence:
            max_len = max(max_len, len(noise_vec))
    # Find the maximum number of iterations
    max_iterations = max(len(sequence) for sequence in all_noise_sequences)
    processed = {
        'raw_sequences': all_noise_sequences,
        'max_training_points': max_len,
        'max_iterations': max_iterations,
        'noise_threshold': noise_threshold,
    }
    # Only support a single seed
    if all_noise_sequences:
        if len(all_noise_sequences) != 1:
            raise ValueError("This script only supports a single seed. Please filter your results accordingly.")
        sequence = all_noise_sequences[0]
        padded_sequence = []
        visibility_sequence = []
        for i in range(max_iterations):
            if i < len(sequence):
                noise_vec = sequence[i]
                current_points = len(noise_vec)
                full_vec = np.full(max_len, np.nan)
                full_vec[:current_points] = noise_vec
                visibility_vec = np.zeros(max_len)
                visibility_vec[:current_points] = 1
                padded_sequence.append(full_vec)
                visibility_sequence.append(visibility_vec)
            else:
                if padded_sequence:
                    padded_sequence.append(padded_sequence[-1].copy())
                    visibility_sequence.append(visibility_sequence[-1].copy())
                else:
                    padded_sequence.append(np.full(max_len, np.nan))
                    visibility_sequence.append(np.zeros(max_len))
        padded_data = np.array(padded_sequence)  # Shape: (max_iterations, max_training_points)
        point_visibility_data = np.array(visibility_sequence)
        print("DEBUG: padded_data shape:", padded_data.shape)
        print("DEBUG: unique values in padded_data:", np.unique(padded_data[~np.isnan(padded_data)]))
        processed['heatmap'] = padded_data  # No averaging
        processed['std_heatmap'] = np.full_like(padded_data, np.nan)  # Not used
        processed['median_heatmap'] = np.full_like(padded_data, np.nan)  # Not used
        processed['point_visibility'] = point_visibility_data
        # First visible iteration
        first_visible_iteration = np.full(max_len, np.nan)
        for point_idx in range(max_len):
            for iter_idx in range(max_iterations):
                if point_visibility_data[iter_idx, point_idx] > 0:
                    first_visible_iteration[point_idx] = iter_idx
                    break
        processed['first_visible_iteration'] = first_visible_iteration
        # Proportion of noisy points (not averaged)
        noisy_points = (padded_data > noise_threshold).astype(float)
        noisy_points[np.isnan(padded_data)] = np.nan
        processed['proportion_noisy'] = np.nanmean(noisy_points, axis=1)  # Per iteration
        processed['total_noise_sum'] = np.nansum(padded_data, axis=1)  # Per iteration
    return processed


##################################################################
# Configuration
##################################################################

if __name__ == "__main__":
    
    project_root = setup_project_path()
    
    # Use utils.py thesis style setup
    setup_thesis_style()

    main(
        objective="ackley10D",     # Set to specific objective for thesis
        seed=0,      # Set to None to auto-detect all seeds
        methods=[
            "boni_standardizenonoise",
            "boni_ilsstandardizenonoise",
            "turboni_tr", 
            "turboni_tradditivenorm",
        ],
        acquisition_function="UpperConfidenceBound",  # Set to None to auto-detect all acquisition functions
        save_plots=True,
        output_dir="figures/thesis/results/",
        sweep="final",
        noise_threshold=0.5,
    )
