"""
Utility module for analyzing experimental result completeness and identifying missing data configurations.
"""
import os
from collections import defaultdict
from utils import (
    setup_project_path, 
    extract_info_from_result_dir
)

def analyze_directory_names_only(base_path):
    # Store directory info for each result dir
    all_dirs_dict = {}
    
    # Keep track of all found info
    all_acquisition_functions = set()
    all_objectives = set()
    all_dimensions = set()
    
    # Find all directories in the results folder
    if not os.path.exists(base_path):
        print(f"ERROR: Results directory not found at {base_path}")
        return {}, [], [], []
        
    result_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    # Process each directory
    for result_dir in result_dirs:
        dir_info = extract_info_from_result_dir(result_dir)
        
        # Track unique values we've found
        if dir_info['objective']: all_objectives.add(dir_info['objective'])
        if dir_info['dim']: all_dimensions.add(dir_info['dim'])
        if dir_info['acquisition_function']: all_acquisition_functions.add(dir_info['acquisition_function'])
        
        # Store directory info
        all_dirs_dict[result_dir] = dir_info
    
    return all_dirs_dict, list(all_objectives), list(all_dimensions), list(all_acquisition_functions)

def analyze_results_completeness(
    base_path,
    expected_objectives=None,
    expected_dims=None,
    expected_acquisition_functions=None,
    expected_seeds=None,
    sweep="final"
):  
    print("="*80)
    print(f"RESULTS COMPLETENESS ANALYSIS - {sweep.upper()} SWEEP")
    print("="*80)
    
    if not os.path.exists(base_path):
        print(f"ERROR: Results directory not found at {base_path}")
        return
    
    # Analyze only directory names (fast mode)
    print(f"Analyzing directory names in {base_path}...")
    all_dirs_dict, found_objectives, found_dimensions, found_acquisition_functions = analyze_directory_names_only(base_path)
        
    # Extract all found information from result directories
    found_methods = set()
    found_seeds = set()
    result_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(set))))
    
    print(f"\nScanning {len(all_dirs_dict)} result directories...")
    
    for result_dir in all_dirs_dict.keys():
        # We already have the extracted info in the all_dirs_dict dictionary
        dir_info = all_dirs_dict[result_dir]
        
        method = dir_info['method']
        objective = dir_info['objective']
        dim = dir_info['dim']
        acq_func = dir_info['acquisition_function']
        seed = dir_info['seed']
        
        if method: found_methods.add(method)
        if seed is not None: found_seeds.add(seed)
        
        # Store in matrix for completeness checking
        if all([method, objective, dim, acq_func, seed is not None]):
            result_matrix[method][objective][dim][acq_func].add(seed)
        
    # Convert sets to sorted lists for better display
    found_methods = sorted(list(found_methods))
    found_objectives = sorted(list(found_objectives))
    found_dimensions = sorted(list(found_dimensions))
    found_acquisition_functions = sorted(list(found_acquisition_functions))
    found_seeds = sorted(list(found_seeds))
    
    # Use expected values or fallback to found values
    methods_to_check = found_methods
    objectives_to_check = expected_objectives if expected_objectives else found_objectives
    dims_to_check = expected_dims if expected_dims else found_dimensions
    acq_funcs_to_check = expected_acquisition_functions if expected_acquisition_functions else found_acquisition_functions
    seeds_to_check = expected_seeds if expected_seeds else found_seeds
    
    # Print summary of what was found
    print(f"\nSUMMARY OF FOUND RESULTS:")
    print(f"   Methods: {len(found_methods)} found")
    print(f"   Objectives: {len(found_objectives)} found - {found_objectives}")
    print(f"   Dimensions: {len(found_dimensions)} found - {found_dimensions}")
    print(f"   Acquisition Functions: {len(found_acquisition_functions)} found - {found_acquisition_functions}")
    print(f"   Seeds: {len(found_seeds)} found - {found_seeds}")
    
    # Print what we're checking against
    print(f"\nCHECKING COMPLETENESS AGAINST:")
    print(f"   Expected Methods ({len(methods_to_check)}): {methods_to_check}")
    print(f"   Expected Objectives ({len(objectives_to_check)}): {objectives_to_check}")
    print(f"   Expected Dimensions ({len(dims_to_check)}): {dims_to_check}")
    print(f"   Expected Acquisition Functions ({len(acq_funcs_to_check)}): {acq_funcs_to_check}")
    print(f"   Expected Seeds ({len(seeds_to_check)}): {seeds_to_check}")
    
    # Calculate total expected combinations
    total_expected = len(methods_to_check) * len(objectives_to_check) * len(dims_to_check) * len(acq_funcs_to_check) * len(seeds_to_check)
    print(f"\nTotal expected result combinations: {total_expected}")
    
    # Check completeness
    missing_results = []
    complete_count = 0
    
    print(f"\nDETAILED COMPLETENESS CHECK:")
    print("-" * 80)
    
    for method in methods_to_check:
        method_complete = True
        method_missing = []
        
        for objective in objectives_to_check:
            for dim in dims_to_check:
                for acq_func in acq_funcs_to_check:
                    found_seeds_for_combo = result_matrix[method][objective][dim][acq_func]
                    missing_seeds = set(seeds_to_check) - found_seeds_for_combo
                    
                    if missing_seeds:
                        method_complete = False
                        for seed in missing_seeds:
                            missing_combo = (method, objective, dim, acq_func, seed)
                            missing_results.append(missing_combo)
                            method_missing.append(f"{objective}_{dim}_{acq_func}_seed{seed}")
                    else:
                        complete_count += len(seeds_to_check)
        
        # Print method status
        status = "COMPLETE" if method_complete else "INCOMPLETE"
        print(f"{method:35} {status}")
        
        if not method_complete and method_missing:
            print(f"   Missing: {method_missing[:5]}{'...' if len(method_missing) > 5 else ''}")
    
    print("-" * 80)
    
    # Final statistics
    found_combinations = len(all_dirs_dict)
    completion_rate = (found_combinations / total_expected) * 100 if total_expected > 0 else 0
    
    print(f"\nFINAL STATISTICS:")
    print(f"   Found combinations: {found_combinations}/{total_expected}")
    print(f"   Completion rate: {completion_rate:.1f}%")
    print(f"   Missing combinations: {len(missing_results)}")
    
    if missing_results:
        print(f"\nMISSING RESULTS (showing first 20):")
        for i, (method, objective, dim, acq_func, seed) in enumerate(missing_results[:20]):
            print(f"   {i+1:2d}. {method}__{objective}_{dim}_{acq_func}_{seed}")
        if len(missing_results) > 20:
            print(f"   ... and {len(missing_results) - 20} more")
    
    # Check for unexpected results (if expected lists were provided)
    if expected_objectives:
        unexpected_objectives = set(found_objectives) - set(expected_objectives)
        if unexpected_objectives:
            print(f"\nUNEXPECTED OBJECTIVES FOUND: {sorted(list(unexpected_objectives))}")
    
    # Method-wise breakdown per objective for each acquisition function
    for acq_func in sorted(found_acquisition_functions):
        print(f"\nMETHOD-WISE BREAKDOWN PER OBJECTIVE - {acq_func.upper()}:")
        
        # Calculate column widths
        method_col_width = 35
        objective_col_width = 19 
        total_col_width = 19
        
        # Calculate total table width
        table_width = method_col_width + len(found_objectives) * objective_col_width + total_col_width
        print("-" * table_width)
        
        # Header
        header = f"{'Method / Objective Function':<{method_col_width}}"
        for objective in found_objectives:
            header += f"{objective:<{objective_col_width}}"
        header += f"{'Total':<{total_col_width}}"
        print(header)
        print("-" * table_width)
        
        # For per-objective breakdown, expected is just the number of seeds
        expected_per_objective = len(seeds_to_check)
        
        # Track totals for final summary for this acquisition function
        objective_totals = {obj: 0 for obj in found_objectives}
        grand_total_found = 0
        
        for method in sorted(found_methods):
            row = f"{method:<{method_col_width}}"
            method_total = 0
            
            # Calculate results for each found objective for this specific acquisition function
            for objective in found_objectives:
                # Count unique seeds for this method-objective-acqfunc combination across all dims
                objective_seeds = set()
                for dim in dims_to_check:
                    objective_seeds.update(result_matrix[method][objective][dim][acq_func])
                
                objective_results = len(objective_seeds)
                method_total += objective_results
                objective_totals[objective] += objective_results
                
                # Format: found/expected (%) with consistent width
                completion = (objective_results / expected_per_objective) * 100 if expected_per_objective > 0 else 0
                cell_content = f"{objective_results:2d}/{expected_per_objective:2d} ({completion:5.1f}%)"
                row += f"{cell_content:<{objective_col_width}}"
            
            grand_total_found += method_total
            # Total column: method total across all objectives for this acquisition function
            total_expected_method = len(found_objectives) * expected_per_objective
            total_completion = (method_total / total_expected_method) * 100 if total_expected_method > 0 else 0
            total_cell = f"{method_total:3d}/{total_expected_method:3d} ({total_completion:5.1f}%)"
            row += f"{total_cell:<{total_col_width}}"
            
            print(row)
        
        # Print totals row for this acquisition function
        print("-" * table_width)
        totals_row = f"{'TOTALS':<{method_col_width}}"
        
        # Calculate totals for each objective column for this acquisition function
        for objective in found_objectives:
            expected_obj_total = len(found_methods) * expected_per_objective
            completion = (objective_totals[objective] / expected_obj_total) * 100 if expected_obj_total > 0 else 0
            cell_content = f"{objective_totals[objective]:2d}/{expected_obj_total:2d} ({completion:5.1f}%)"
            totals_row += f"{cell_content:<{objective_col_width}}"
        
        # Grand total column for this acquisition function
        total_expected_all = len(found_methods) * len(found_objectives) * expected_per_objective
        overall_completion = (grand_total_found / total_expected_all) * 100 if total_expected_all > 0 else 0
        grand_total_cell = f"{grand_total_found:3d}/{total_expected_all:4d} ({overall_completion:5.1f}%)"
        totals_row += f"{grand_total_cell:<{total_col_width}}"
        print(totals_row)
    
    print("="*80)
    
    return {
        'total_expected': total_expected,
        'total_found': found_combinations,
        'completion_rate': completion_rate,
        'missing_results': missing_results,
        'found_methods': found_methods,
        'found_objectives': found_objectives,
        'found_dimensions': found_dimensions,
        'found_acquisition_functions': found_acquisition_functions,
        'found_seeds': found_seeds
    }


def main():
    project_root = setup_project_path()
    
    # Configuration matching the plotting script
    sweep = "final"
    
    # You can specify expected values or set to None to auto-detect
    expected_objectives = None  # Set to None to auto-detect
    expected_dims = None  # Set to None to auto-detect  
    expected_acquisition_functions = None  # Set to None to auto-detect
    expected_seeds = list(range(20))  # Set to None to auto-detect (or specify like [0, 1, 2, 3, 4])
    
    # Base path for results
    base_path = os.path.join(project_root, "results", sweep)
    
    # Run the analysis
    analyze_results_completeness(
        base_path=base_path,
        expected_objectives=expected_objectives,
        expected_dims=expected_dims,
        expected_acquisition_functions=expected_acquisition_functions,
        expected_seeds=expected_seeds,
        sweep=sweep
    )


if __name__ == "__main__":
    main()
