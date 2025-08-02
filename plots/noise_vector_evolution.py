"""
Analyzes and visualizes noise vector evolution patterns in optimization with noise injection methods.
Chart types: heatmaps, line plots, summary statistics, and combined visualizations showing noise adaptation.
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
    ensure_output_dir,
    save_plot_figure,
    convert_to_numpy,
    load_and_filter_results,
    setup_plot_style
)


def main(
    objective=None,  # Set to None to detect all objectives
    dim=None,        # Set to None to detect all dimensions
    seed=0,
    methods=None,    # Only noise injection methods will have noise vectors
    acquisition_function=None,  # Set to None to detect all acquisition functions
    save_plots=False,
    output_dir=None,
    sweep="benchmark_point_of_transformation",
    chart_type="heatmap",  # Options: "heatmap", "line", "summary", "combined"
    noise_threshold=0.5,  # Threshold to distinguish between noise/no-noise (factors > threshold = noise)
):
    # Base path for results
    project_root = setup_project_path()
    base_path = os.path.join(project_root, "results", sweep)
    
    if not os.path.exists(base_path):
        print(f"Results directory not found at {base_path}")
        return

    # Create output directory if saving plots
    if save_plots:
        prefix = f"noise_vector_{objective}_{dim}" if objective and dim else "noise_vector_comparison"
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
    
    # Generate plots for each combination
    objectives_to_plot = [objective] if objective else all_objectives
    acq_funcs_to_plot = [acquisition_function] if acquisition_function else all_acquisition_functions
    
    for obj in objectives_to_plot:
        for acq_func in acq_funcs_to_plot:
            print(f"\nGenerating noise vector plot for objective: {obj}, acquisition function: {acq_func}")
            generate_noise_vector_plot(
                all_results=noise_results, 
                objective=obj, 
                acquisition_function=acq_func, 
                save_plots=save_plots, 
                output_dir=output_dir,
                chart_type=chart_type,
                methods=methods,
                noise_threshold=noise_threshold
            )


def generate_noise_vector_plot(all_results, objective, acquisition_function, 
                             save_plots=False, output_dir=None, chart_type="heatmap", 
                             methods=None, noise_threshold=0.5):
    
    # Get utility elements
    acq_display_names = get_acquisition_function_display_names()
    
    # Filter results for the specified objective and acquisition function
    filtered_results = filter_results_by_criteria(
        all_results, 
        objective=objective,
        acquisition_function=acquisition_function
    )
    
    if not filtered_results:
        print(f"No results with noise vectors found for objective: {objective}, acquisition function: {acquisition_function}")
        return
    
    # Group results by dimension
    dimension_groups = group_results_by_dimension(filtered_results)
    
    # Set plot style
    setup_plot_style(PLOT_STYLE)
    
    # Create a plot for each dimension
    for dim, dim_results in dimension_groups.items():
        # Prepare noise vector data and full results data
        method_noise_data = defaultdict(list)
        method_results_data = defaultdict(list)
        
        for result_dir, results in dim_results.items():
            dir_info = extract_info_from_result_dir(result_dir)
            method_name = dir_info['method']
            
            if "noise_vectors" in results and results["noise_vectors"]:
                # Convert noise vectors to numpy arrays and filter out None values
                noise_vectors = []
                for nv in results["noise_vectors"]:
                    if nv is not None:
                        noise_vectors.append(convert_to_numpy(nv))
                
                if noise_vectors:
                    method_noise_data[method_name].append(noise_vectors)
                    method_results_data[method_name].append(results)
        
        if not method_noise_data:
            print(f"No valid noise vector data found for dimension {dim}")
            continue
        
        # Get method names and sort them
        method_names = list(method_noise_data.keys())
        sorted_method_names = get_sorted_method_names(method_names)
        
        # Get consistent color mapping
        color_map = get_method_color_map(sorted_method_names)
        
        # Get display names
        objective_display_name = get_objective_display_name(objective)
        acq_display_name = acq_display_names.get(acquisition_function, acquisition_function) if acquisition_function else "All"
        
        # Create separate plots for each method
        for method_idx, method_name in enumerate(sorted_method_names):
            if method_name not in method_noise_data:
                continue
                
            method_display_name = get_method_display_name(method_name)
            color = color_map.get(method_name, f"C{method_idx}")
            
            # Get all noise vector sequences for this method (across seeds)
            all_noise_sequences = method_noise_data[method_name]
            all_results = method_results_data[method_name]
            
            if not all_noise_sequences:
                continue
            
            # Process noise vector data for visualization
            processed_data = process_noise_vector_data(all_noise_sequences, noise_threshold)
            # Add results data for training point access
            processed_data['results'] = all_results
            
            # Create subplot layout: heatmap on left, function plot on right
            fig, (ax_heatmap, ax_function) = plt.subplots(1, 2, figsize=(16, 7))
            
            # Store legend elements for placement below subplots
            heatmap_legend_elements = []
            function_legend_elements = []
            
            # Create visualizations for this method
            if ax_heatmap is not None:
                heatmap_legend_elements = plot_noise_heatmap(ax_heatmap, processed_data, method_display_name, color, method_idx, len(sorted_method_names))
            
            if ax_function is not None:
                function_legend_elements = plot_test_function_with_noise(ax_function, processed_data, method_display_name, objective, dim)
            
            # Style and label plots
            if ax_heatmap is not None:
                style_heatmap_plot(ax_heatmap, objective_display_name, acq_display_name, dim, chart_type)
            
            if ax_function is not None:
                style_function_plot(ax_function, objective_display_name, acq_display_name, dim, chart_type)
            
            # Add legends below subplots
            if heatmap_legend_elements:
                fig.legend(handles=heatmap_legend_elements, 
                          bbox_to_anchor=(0.25, 0.02), loc='lower center', 
                          ncol=len(heatmap_legend_elements), fontsize=10)
            
            if function_legend_elements:
                fig.legend(handles=function_legend_elements, 
                          bbox_to_anchor=(0.75, 0.02), loc='lower center', 
                          ncol=len(function_legend_elements), fontsize=10)
            
            # Add overall title and layout
            fig.suptitle(f"Noise Vector Analysis: {method_display_name} - {objective_display_name} ({acq_display_name})", 
                        fontsize=16, fontweight="bold")
            fig.tight_layout(rect=[0, 0.05, 1, 0.95])
            
            # Save plot if requested
            if save_plots:
                filename = f"noise_vector_{chart_type}_{method_name}_{objective}_{dim}D_{acquisition_function}.png"
                save_plot_figure(fig, output_dir, filename)
            
            plt.show()


def process_noise_vector_data(all_noise_sequences, noise_threshold=0.5):
    # Find the maximum length across all sequences and seeds
    max_len = 0
    for sequence in all_noise_sequences:
        for noise_vec in sequence:
            max_len = max(max_len, len(noise_vec))
    
    # Find the maximum number of iterations across all seeds
    max_iterations = max(len(sequence) for sequence in all_noise_sequences)
    
    processed = {
        'raw_sequences': all_noise_sequences,
        'max_training_points': max_len,
        'max_iterations': max_iterations,
        'noise_threshold': noise_threshold,
        'n_seeds': len(all_noise_sequences)
    }
    
    # Create aggregated heatmap data (average across seeds)
    if all_noise_sequences:
        # Create 3D array with NaN for unobserved points (seeds × iterations × training_points)
        padded_data = []
        point_visibility_data = []  # Track when each point becomes visible
        
        for sequence in all_noise_sequences:
            padded_sequence = []
            visibility_sequence = []
            
            for i in range(max_iterations):
                if i < len(sequence):
                    noise_vec = sequence[i]
                    current_points = len(noise_vec)
                    
                    # Create full vector with NaN for unobserved points
                    full_vec = np.full(max_len, np.nan)
                    full_vec[:current_points] = noise_vec
                    
                    # Track visibility: 1 if point is observed, 0 if not yet observed
                    visibility_vec = np.zeros(max_len)
                    visibility_vec[:current_points] = 1
                    
                    padded_sequence.append(full_vec)
                    visibility_sequence.append(visibility_vec)
                else:
                    # Use the last available state for iterations beyond this seed's data
                    if padded_sequence:
                        padded_sequence.append(padded_sequence[-1].copy())
                        visibility_sequence.append(visibility_sequence[-1].copy())
                    else:
                        # Fallback: no points observed
                        padded_sequence.append(np.full(max_len, np.nan))
                        visibility_sequence.append(np.zeros(max_len))
            
            padded_data.append(padded_sequence)
            point_visibility_data.append(visibility_sequence)
        
        padded_data = np.array(padded_data)  # Shape: (n_seeds, max_iterations, max_training_points)
        point_visibility_data = np.array(point_visibility_data)  # Shape: (n_seeds, max_iterations, max_training_points)
        
        # Calculate statistics (ignoring NaN values)
        processed['mean_heatmap'] = np.nanmean(padded_data, axis=0)  # Average across seeds
        processed['std_heatmap'] = np.nanstd(padded_data, axis=0)
        processed['median_heatmap'] = np.nanmedian(padded_data, axis=0)
        
        # Track when points first become visible (averaged across seeds)
        processed['point_visibility'] = np.mean(point_visibility_data, axis=0)  # Average visibility across seeds
        
        # Calculate the first iteration when each point becomes visible (across all seeds)
        first_visible_iteration = np.full(max_len, np.nan)
        for point_idx in range(max_len):
            for iter_idx in range(max_iterations):
                if np.any(point_visibility_data[:, iter_idx, point_idx] > 0):
                    first_visible_iteration[point_idx] = iter_idx
                    break
        processed['first_visible_iteration'] = first_visible_iteration
        
        # Calculate proportion of noisy points over time (only for visible points)
        noisy_points = (padded_data > noise_threshold).astype(float)
        noisy_points[np.isnan(padded_data)] = np.nan  # Keep NaN where points are not visible
        processed['proportion_noisy'] = np.nanmean(noisy_points, axis=(0, 2))  # Average across seeds and visible training points
        processed['proportion_noisy_per_seed'] = np.nanmean(noisy_points, axis=2)  # Per seed, averaged across visible training points
        
        # Calculate total noise factor sum over time (only for visible points)
        visible_data = padded_data.copy()
        visible_data[np.isnan(visible_data)] = 0  # Treat unobserved points as 0 for sum calculation
        processed['total_noise_sum'] = np.mean(np.nansum(padded_data, axis=2), axis=0)  # Sum across visible training points, average across seeds
        processed['total_noise_sum_per_seed'] = np.nansum(padded_data, axis=2)  # Per seed
        
    return processed


def plot_test_function_with_noise(ax, processed_data, method_name, objective_name, dim):
    if objective_name == "ackley2D" or (objective_name == "ackley" and dim == 2):
        # Create Ackley 2D function visualization
        import numpy as np
        
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
        x_range = np.linspace(-5, 5, 100)
        y_range = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = ackley_2d(X, Y)
        
        # Plot function as contour
        contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
        ax.contour(X, Y, Z, levels=20, colors='black', alpha=0.3, linewidths=0.5)
        
        # Add colorbar for function values
        cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
        cbar.set_label('Function Value', rotation=270, labelpad=15)
        
        # Now overlay noise information if available
        if 'raw_sequences' in processed_data and processed_data['raw_sequences']:
            # Get the latest iteration data (last noise vectors)
            all_sequences = processed_data['raw_sequences']
            
            # Collect all final noise vectors across seeds and actual training points
            final_noise_vectors = []
            training_points = []
            
            # Extract actual training data from optimization results
            if 'results' in processed_data:
                for result in processed_data['results']:
                    if 'history_x' in result and 'history_y' in result:
                        # Convert optimization history to numpy arrays
                        history_x = convert_to_numpy(result['history_x'])
                        history_y = convert_to_numpy(result['history_y'])
                        
                        # Store the actual training points
                        if len(history_x.shape) > 1 and history_x.shape[1] == 2:  # 2D case
                            training_points.append(history_x)
                        elif len(history_x.shape) == 1:  # 1D case, reshape for consistency
                            training_points.append(history_x.reshape(-1, 1))
            
            # Also collect noise vectors for the corresponding training points
            for seed_sequence in all_sequences:
                if seed_sequence:  # Make sure sequence is not empty
                    final_noise_vec = seed_sequence[-1]  # Last iteration
                    final_noise_vectors.append(final_noise_vec)
            
            if final_noise_vectors and training_points:
                # Average the noise vectors across seeds
                avg_noise_vector = np.mean(final_noise_vectors, axis=0)
                avg_training_points = np.mean(training_points, axis=0)
                
                # Ensure noise vector and training points have matching lengths
                min_length = min(len(avg_noise_vector), len(avg_training_points))
                avg_noise_vector = avg_noise_vector[:min_length]
                avg_training_points = avg_training_points[:min_length]
                
                # Plot training points with noise information
                # Handle both 1D and 2D cases
                if avg_training_points.shape[1] == 2:  # 2D case
                    # Determine vmax based on actual noise data
                    noise_max = np.max(avg_noise_vector) if len(avg_noise_vector) > 0 else 2.0
                    vmax = max(2.0, noise_max)  # At least 2.0 to accommodate noise factor 2
                    
                    scatter = ax.scatter(avg_training_points[:, 0], avg_training_points[:, 1], 
                                       c=avg_noise_vector, cmap='RdYlBu_r', 
                                       s=100, edgecolors='black', linewidths=1.5, 
                                       vmin=0, vmax=vmax, alpha=0.9, zorder=5)
                elif avg_training_points.shape[1] == 1:  # 1D case
                    # For 1D, we need to evaluate the function at training points to get y-coordinates
                    train_y = []
                    for x_val in avg_training_points[:, 0]:
                        train_y.append(ackley_2d(x_val, 0))  # Use y=0 for 1D case
                    train_y = np.array(train_y)
                    
                    # Determine vmax based on actual noise data
                    noise_max = np.max(avg_noise_vector) if len(avg_noise_vector) > 0 else 2.0
                    vmax = max(2.0, noise_max)  # At least 2.0 to accommodate noise factor 2
                    
                    scatter = ax.scatter(avg_training_points[:, 0], train_y, 
                                       c=avg_noise_vector, cmap='RdYlBu_r', 
                                       s=100, edgecolors='black', linewidths=1.5, 
                                       vmin=0, vmax=vmax, alpha=0.9, zorder=5)
                
                # Add colorbar for noise factors
                cbar_noise = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
                cbar_noise.set_label('Noise Factor', rotation=270, labelpad=15)
                
                # Create legend elements but don't add them to the plot
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                           markersize=8, label='High Noise (≈2.0)'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                           markersize=8, label='Low Noise (≈0.0)'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
                           markersize=8, label='Training Points')
                ]
                
                ax.set_xlabel('x_1')
                ax.set_ylabel('x_2')
                ax.set_title(f'Ackley 2D Function\nwith Noise Vector Overlay')
                ax.set_xlim(-5, 5)
                ax.set_ylim(-5, 5)
                
                return legend_elements
        
        # If no noise data is available, still set the basic formatting
        ax.set_xlabel('x_1')
        ax.set_ylabel('x_2')
        ax.set_title(f'Ackley 2D Function')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        
    else:
        # For other functions, show a placeholder or simple visualization
        ax.text(0.5, 0.5, f'Function visualization\nfor {objective_name} {dim}D\nnot yet implemented', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(f'{method_name} - {objective_name} {dim}D Function')
    
    # Return empty list if no legend elements were created
    return []


def plot_noise_heatmap(ax, processed_data, method_name, color, method_idx, total_methods):
    if 'mean_heatmap' not in processed_data:
        return []
    
    heatmap_data = processed_data['mean_heatmap']
    first_visible = processed_data.get('first_visible_iteration', np.full(heatmap_data.shape[1], 0))
    
    # Create custom colormap: gray for unobserved, RdYlBu_r for observed
    import matplotlib.colors as mcolors
    
    # Create masked array where NaN values (unobserved points) are masked
    masked_data = np.ma.masked_invalid(heatmap_data.T)
    
    # Create heatmap with custom colormap
    cmap = plt.cm.RdYlBu_r
    cmap.set_bad(color='lightgray', alpha=0.3)  # Color for unobserved points
    
    # Determine vmax based on actual data range to handle noise factors up to 2
    data_max = np.nanmax(heatmap_data) if not np.all(np.isnan(heatmap_data)) else 2.0
    vmax = max(2.0, data_max)  # At least 2.0 to accommodate noise factor 2
    
    im = ax.imshow(masked_data, aspect='auto', cmap=cmap, vmin=0, vmax=vmax, 
                   interpolation='nearest', origin='lower')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Noise Factor', rotation=270, labelpad=15)
    
    # Mark when each point first becomes visible
    max_points = heatmap_data.shape[1]
    for point_idx in range(max_points):
        first_iter = first_visible[point_idx]
        if not np.isnan(first_iter):
            # Add a vertical line at the first iteration this point was observed
            ax.axvline(x=first_iter, ymin=point_idx/max_points, ymax=(point_idx+1)/max_points, 
                      color='black', linewidth=2, alpha=0.8)
            
            # Add a small marker
            ax.plot(first_iter, point_idx, marker='o', color='white', markersize=4, 
                   markeredgecolor='black', markeredgewidth=1)
    
    # Create legend elements but don't add them to the plot
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='white', markerfacecolor='white',
               markeredgecolor='black', markersize=6, label='Point first observed', linestyle='None'),
        Line2D([0], [0], color='lightgray', linewidth=8, alpha=0.5, label='Unobserved points')
    ]
    
    # Style
    ax.set_xlabel('Optimization Iteration')
    ax.set_ylabel('Training Point Index')
    ax.set_title(f'Noise History for All Observed Points\n(Averaged across {processed_data["n_seeds"]} seeds)')
    
    # Set y-axis to show all points
    ax.set_ylim(-0.5, max_points - 0.5)
    
    # Add grid to better see point indices
    ax.set_yticks(range(0, max_points, max(1, max_points // 10)))
    ax.grid(True, alpha=0.3, axis='y')
    
    return legend_elements


def plot_noise_trajectories(ax, processed_data, method_name, color):
    if 'raw_sequences' not in processed_data:
        return
    
    # Plot a few representative trajectories
    all_sequences = processed_data['raw_sequences']
    max_iterations = min(processed_data['max_iterations'], 50)  # Limit for readability
    
    # Select first few training points to show as lines
    max_points_to_show = min(10, processed_data['max_training_points'])
    
    for point_idx in range(max_points_to_show):
        trajectories = []
        for seed_idx, sequence in enumerate(all_sequences):
            trajectory = []
            for iter_idx in range(min(len(sequence), max_iterations)):
                if point_idx < len(sequence[iter_idx]):
                    trajectory.append(sequence[iter_idx][point_idx])
                else:
                    # This training point doesn't exist yet
                    trajectory.append(np.nan)
            trajectories.append(trajectory)
        
        # Plot median trajectory with IQR
        if trajectories:
            trajectories = np.array(trajectories)
            iterations = np.arange(trajectories.shape[1])
            
            # Calculate median and IQR
            median_traj = np.nanmedian(trajectories, axis=0)
            q25_traj = np.nanpercentile(trajectories, 25, axis=0)
            q75_traj = np.nanpercentile(trajectories, 75, axis=0)
            
            alpha = 0.7 if point_idx < 5 else 0.3  # Make first few points more prominent
            
            ax.plot(iterations, median_traj, color=color, alpha=alpha, linewidth=1.5, 
                   label=f'Point {point_idx}' if point_idx < 5 else None)
            ax.fill_between(iterations, q25_traj, q75_traj, color=color, alpha=alpha*0.3)
    
    ax.set_xlabel('Optimization Iteration')
    ax.set_ylabel('Noise Factor')
    ax.set_title(f'{method_name} - Noise Factor Trajectories')
    ax.set_ylim(-0.1, 2.1)
    ax.axhline(y=processed_data['noise_threshold'], color='red', linestyle='--', alpha=0.5, 
              label=f'Noise threshold ({processed_data["noise_threshold"]})')
    # Add reference line for factor 2
    ax.axhline(y=2.0, color='darkred', linestyle=':', alpha=0.5, label='Max noise (2.0)')
    if any(label is not None for label in [line.get_label() for line in ax.lines[-6:]]):  # If we have labels
        ax.legend(loc='upper right', fontsize=8)


def plot_noise_summary(ax, processed_data, method_name, color):
    if 'proportion_noisy' not in processed_data:
        return
    
    iterations = np.arange(len(processed_data['proportion_noisy']))
    
    # Plot proportion of noisy points
    ax2 = ax.twinx()
    
    # Main plot: Total noise sum
    line1 = ax.plot(iterations, processed_data['total_noise_sum'], color=color, linewidth=2, 
                   label='Total Noise Sum')
    ax.set_ylabel('Total Noise Sum', color=color)
    ax.tick_params(axis='y', labelcolor=color)
    
    # Secondary plot: Proportion of noisy points
    line2 = ax2.plot(iterations, processed_data['proportion_noisy'], color='orange', linewidth=2, 
                    linestyle='--', label='Proportion Noisy')
    ax2.set_ylabel('Proportion of Noisy Points', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.set_ylim(0, 1)
    
    ax.set_xlabel('Optimization Iteration')
    ax.set_title(f'Noise Summary Statistics')
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper right')


def style_function_plot(ax, objective_name, acq_name, dim, chart_type):
    """Apply styling to function plot."""
    if chart_type != "combined":
        ax.set_title(f'Test Function: {objective_name} {dim}D ({acq_name})', 
                    fontweight='bold', fontsize=14)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Style borders
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('black')


def style_heatmap_plot(ax, objective_name, acq_name, dim, chart_type):
    if chart_type != "combined":
        ax.set_title(f'Noise Vector Evolution: {objective_name} {dim}D ({acq_name})', 
                    fontweight='bold', fontsize=14)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Style borders
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('black')


def style_line_plot(ax, objective_name, acq_name, dim, chart_type):
    if chart_type != "combined":
        ax.set_title(f'Noise Factor Trajectories: {objective_name} {dim}D ({acq_name})', 
                    fontweight='bold', fontsize=14)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Style borders
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('black')


##################################################################
# Configuration
##################################################################

if __name__ == "__main__":
    
    project_root = setup_project_path()

    PLOT_STYLE = {
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 12,
        "figure.figsize": (12, 7),
        "figure.dpi": 100,
    }

    main(
        objective="ackley10D",     # Set to None to auto-detect all objectives
        dim=None,           # Set to None to auto-detect all dimensions
        seed=0,      # Set to None to auto-detect all seeds
        methods=[
            # "boni_plain", "boni_plainnonoise",
            # "boni_standardize", "boni_standardizenonoise",
            # "boni_standardizenonoiserefit",
            # "boni_standardizeones", "boni_standardizezeros",
            # "boni_ilsstandardize", "boni_ilsstandardizenonoise",
            # "boni_ilsstandardiznonoiserefit",
            # "boni_bsnoise",
            # "boni_bsnonoise",  
            # "boni_standardizegradient", "boni_standardizegradientbinary",
            "turbo_plain", "turbo_standardize",
            "turboni_standardize",
            "turboni_tr", "turboni_trbinary",
            "turboni_trbs", "turboni_trbsnorm",
        ],
        acquisition_function=None,  # Set to None to auto-detect all acquisition functions
        save_plots=True,
        output_dir="figures/xseed_noise_vector",
        sweep="final",
        chart_type="combined",  # Options: "heatmap", "line", "summary", "combined"
        noise_threshold=0.5,  # Threshold to distinguish noise/no-noise
    )
