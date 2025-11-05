import os
import sys
import time
import glob
import psutil
import numpy as np
from math import comb
from datetime import datetime
from Hubbard_SingleBand import *
from Clusters_Square import *
from spin_operators import *
from utils_state import *
from utils_math import *
from utils_file import *

# Default parameters
DEFAULT_N = 3
DEFAULT_U = 1.0
DEFAULT_T = 0.1
def parse_arguments(rank=0):
    params = {}
    params['N'] = DEFAULT_N
    params['U'] = DEFAULT_U
    params['t'] = DEFAULT_T
    params['t2'] = None
    params['t3'] = None
    params['sz'] = None
    params['s2'] = None
    params['s2_fix'] = False
    params['type'] = None
    params['delta'] = None
    params['delta2'] = None
    params['restart'] = False
    params['type_delta'] = None

    params['Ncell'] = None
    params['Ncut'] = None
    params['ratio'] = None

    for arg in sys.argv[1:]:
        if '=' not in arg:
            continue
        key, value = arg.split('=')
        key = key.upper()
        if key == 'N'.upper():
            params['N'] = int(value)
        elif key == 'U'.upper():
            params['U'] = float(value)
        elif key == 't'.upper():
            params['t'] = float(value)
        elif key == 't2'.upper():
            params['t2'] = float(value)
        elif key == 't3'.upper():
            params['t3'] = float(value)
        elif key == 'sz'.upper():
            params['sz'] = float(value)
        elif key == 's2'.upper():
            params['s2'] = float(value)
        elif key == 'S2_FIX'.upper():
            params['s2_fix'] = bool(value)
        elif key == 'type'.upper():
            params['type'] = value
        elif key == 'delta'.upper():
            params['delta'] = float(value)
        elif key == 'delta2'.upper():
            params['delta2'] = float(value)
        elif key == 'RESTART'.upper():
            params['restart'] = bool(value)
        elif key == 'TYPE_DELTA'.upper():
            params['type_delta'] = value
        elif key == 'NCELL'.upper():
            params['Ncell'] = int(value)
        elif key == 'NCUT'.upper():
            params['Ncut'] = int(value)
        elif key == 'RATIO'.upper():
            params['ratio'] = float(value)
    
    params['sz'] = tune_sz(params['sz'], params['N'], rank=rank)
    return params

def setup_work_environment(params, rank=0):
    """Setup the working environment and parameters"""
    result_dir1 = f"Block_U{params['U']:.4f}_t{params['t']:.4f}"
    if params['t2'] is not None:
        result_dir1 = f"{result_dir1}_tp{params['t2']:.4f}"

    result_dir2 = f"N{params['N']}"
    result_dir3 = f"N{params['N']}"
    if params['sz'] is not None:
        result_dir2 = f"{result_dir2}_sz{params['sz']:.4f}"
        result_dir3 = f"{result_dir3}_sz{params['sz']:.4f}"
    if params['s2'] is not None:
        result_dir2 = f"{result_dir2}_s{params['s2']:.0f}"
        result_dir3 = f"{result_dir3}_s{params['s2']:.0f}"
    if params['type'] is not None:
        result_dir2 = f"{result_dir2}_{params['type']}"
    
    if rank == 0:
        os.makedirs("tmp", exist_ok=True)
        if not params['restart']:
            os.makedirs(f"{result_dir1}/{result_dir2}", exist_ok=True)
        else:
            os.makedirs(f"{result_dir1}/{result_dir2}_restart", exist_ok=True)
        print(f"Working directory: {result_dir1}/{result_dir2}")

    return result_dir1, result_dir2, result_dir3


def distribute_work(work_items, rank, size):
    """Distribute work items among MPI ranks using round-robin with extra tasks to later ranks"""
    total_items = len(work_items)
    items_per_rank = total_items // size
    remainder = total_items % size
    
    # Use round-robin distribution for the base items
    work_items_for_rank = []
    base_items = items_per_rank * size  # Number of items distributed in round-robin
    
    # Add base round-robin items
    for i in range(rank, base_items, size):
        work_items_for_rank.append((i, work_items[i]))
    
    # Add extra items to later ranks if there are remainder items
    if remainder > 0 and rank >= size - remainder:
        extra_item_idx = base_items + (rank - (size - remainder))
        work_items_for_rank.append((extra_item_idx, work_items[extra_item_idx]))
    
    return work_items_for_rank

def cluster_save_results(result_dir, hole, class_idx, cluster_idx, rank, cluster_time, 
                        cluster, bonds, 
                        coeffs, error, T11m1_norm, overlap):
    try:
        base_filename = f"{result_dir}/hole{hole}_class{class_idx}_cluster{cluster_idx}"
        with open(f"{base_filename}_results.txt", 'w') as f:
            write_results_to_file(f, hole, class_idx, cluster_idx, rank, cluster_time, 
                                cluster, bonds, coeffs, error, T11m1_norm, overlap)
        write_file(f"{base_filename}_results.txt", f"Task is finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        raise IOError(f"Failed to save cluster results: {str(e)}")


def cluster_process(cluster, params, params_cluster, base_filename=None):
    try:
        model = Hubbard_SingleBand(params['N'], params['U'], params['t'])

        bonds = bond_analysis_spin(cluster)
        # Setting the type of bonds, if None do nothing
        if len(bonds) >= 1 and canonical_bond_type(cluster, bonds[0][0]) == 1:
            model.set_bonds_by_class(bonds[0], params['t'])
        
        if len(bonds) >= 1 and canonical_bond_type(cluster, bonds[0][0]) == 2:
            model.set_bonds_by_class(bonds[0], params['t2'])
        elif len(bonds) >= 2 and canonical_bond_type(cluster, bonds[1][0]) == 2:
            model.set_bonds_by_class(bonds[1], params['t2']) 

        model.set_block(False)
        model.set_states(nsites=params['N'], nelec=params['N'], sz=params['sz'])
        model.set_states_sort()
        model.block_s2(params['N'], params['sz'], params['s2'])

        if params['restart']:
            model.restart(base_filename)
        else:
            model.calc_hamiltonian()
            model.solve(model.s2_Us)

        model.calc_s2()
        model.calc_heff_halffilled(params, params_cluster)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to process cluster: {str(e)}")

def cluster_process_work_item(hole, class_idx, clusters, params, rank):
    params_cluster = {}
    params_cluster['hole'] = hole
    params_cluster['class_idx'] = class_idx
    params_cluster['rank'] = rank
    try:
        # Process first cluster to get model
        if not params['restart']:
            model = cluster_process(clusters.clusters_classified[hole][class_idx][0], params, params_cluster)
            result_dir = f"{params['result_dir']}"
            base_filename = f"{params['result_dir']}/hole{hole}_class{class_idx}"
            np.save(f"{base_filename}_eigvals.npy", model.eigvals)
            np.save(f"{base_filename}_eigvecs.npy", model.eigvecs)
        else:
            if os.path.exists(f"{params['result_dir1']}/{params['result_dir2']}"):
                restart_filename = f"{params['result_dir1']}/{params['result_dir2']}/hole{hole}_class{class_idx}"
            elif os.path.exists(f"{params['result_dir1']}/{params['result_dir3']}"):
                restart_filename = f"{params['result_dir1']}/{params['result_dir3']}/hole{hole}_class{class_idx}"
            else:
                print(f"Error: The directory does not exist: {params['result_dir1']}/{params['result_dir3']}!\nPlease run it in advance...")
                exit(1)
            model = cluster_process(clusters.clusters_classified[hole][class_idx][0], params, params_cluster, restart_filename)
            result_dir = f"{params['result_dir']}_restart"
            base_filename = f"{params['result_dir']}_restart/hole{hole}_class{class_idx}"
        
        filename_tmp=f"tmp/{params['result_dir1']}/{params['result_dir2']}"
        filename_out=f"{result_dir}/data_hole{hole}_class{class_idx}"
        
        if params['sz'] is None:
            filename_tmp += f"_rank{rank}"
            filename_out += f"_rank{rank}"
        else:
            filename_tmp += f"_sz{params['sz']:.4f}_rank{rank}"
            filename_out += f"_sz{params['sz']:.4f}_rank{rank}"
        
        if os.path.exists(f"{filename_tmp}_None.txt"):
            os.rename(f"{filename_tmp}_None.txt"           , f"{filename_out}_None.txt")
        if os.path.exists(f"{filename_tmp}_Energy.txt"):
            os.rename(f"{filename_tmp}_Energy.txt"         , f"{filename_out}_Energy.txt")
        if os.path.exists(f"{filename_tmp}_Energy_s2.txt"):
            os.rename(f"{filename_tmp}_Energy_s2.txt"      , f"{filename_out}_Energy_s2.txt")
        if os.path.exists(f"{filename_tmp}_occ.txt"):
            os.rename(f"{filename_tmp}_occ.txt"            , f"{filename_out}_occ.txt")
        if os.path.exists(f"{filename_tmp}_occ_s2.txt"):
            os.rename(f"{filename_tmp}_occ_s2.txt"         , f"{filename_out}_occ_s2.txt")
        if os.path.exists(f"{filename_tmp}_single.txt"):
            os.rename(f"{filename_tmp}_single.txt"         , f"{filename_out}_single.txt")
        if os.path.exists(f"{filename_tmp}_single_s2.txt"):
            os.rename(f"{filename_tmp}_single_s2.txt"      , f"{filename_out}_single_s2.txt")
        if os.path.exists(f"{filename_tmp}_multi.txt"):
            os.rename(f"{filename_tmp}_multi.txt"          , f"{filename_out}_multi.txt")
        if os.path.exists(f"{filename_tmp}_multi_s2.txt"):
            os.rename(f"{filename_tmp}_multi_s2.txt"       , f"{filename_out}_multi_s2.txt")
        pattern = f"{filename_tmp}_multi_iter*.txt"
        matched_files = glob.glob(pattern)
        if len(matched_files) > 0:
            os.makedirs(f"{filename_out}_multi", exist_ok=True)
            for file in matched_files:
                iter=file.split("_iter")[1].split(".txt")[0]
                os.rename(file, f"{filename_out}_multi/rank{rank}_iter{iter}.txt")
        pattern = f"{filename_tmp}_multi_s2_iter*.txt"
        matched_files = glob.glob(pattern)
        if len(matched_files) > 0:
            os.makedirs(f"{filename_out}_multi_s2", exist_ok=True)
            for file in matched_files:
                iter=file.split("_iter")[1].split(".txt")[0]
                os.rename(file, f"{filename_out}_multi_s2/rank{rank}_iter{iter}.txt")
        
        # Check the problem of the model
        if model.error:
            print(f"Error in rank {rank}")
            return

        # Save Heff for the first cluster, if the model is not error
        np.savetxt(f"{base_filename}_Heff.npy", model.Heff)
        np.savetxt(f"{base_filename}_T11m1.npy", model.T11m1)
        np.savetxt(f"{base_filename}_t11_selected_indices.npy", model.t11_selected_indices, fmt="%d")
        np.savetxt(f"{base_filename}_t11_selected_occupation.npy", model.t11_selected_occupation)
        np.savetxt(f"{base_filename}_double_occupation_expectation.npy", model.double_occupation_expectation)
        np.savetxt(f"{base_filename}_states.npy", model.states, fmt="%d")
        np.savetxt(f"{base_filename}_s2_digonal.npy", np.column_stack([model.S2_diag, model.S2_error]))
        np.savetxt(f"{base_filename}_s2_selected.npy", np.column_stack([model.S2_diag[model.t11_selected_indices], model.S2_error[model.t11_selected_indices]]))
        
        # Process all clusters in this class
        for cluster_idx in range(len(clusters.clusters_classified[hole][class_idx])):
            t0_cluster = time.time()
            
            bonds = bond_analysis_spin(clusters.clusters_classified[hole][class_idx][cluster_idx])
            coeffs, error = model.calc_spin_coeff(bonds, params['s2'])
            # Save individual cluster results
            cluster_save_results(result_dir, hole, class_idx, cluster_idx, rank, 
                               time.time() - t0_cluster, 
                               clusters.clusters_classified[hole][class_idx][cluster_idx], 
                               bonds, 
                               coeffs, error, model.T11m1_norm, model.overlap)
        # Clean up
        model.clear()
        del model
    except Exception as e:
        raise RuntimeError(f"Failed to process work item: {str(e)}")


def get_memory_usage():
    """Get current memory usage of the process in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024  # Convert to GB

def analyze_matrix_memory(N, sz=None):
    complex_size = 16
    if sz is None:
        dim = comb(2*N, N)
        dim_eff = 2**N
    else:
        #Here we assuming Nelec=Nsite
        Nup=int((N+2*sz)/2)
        Ndo=int((N-2*sz)/2)
        dim = comb(N, Nup) * comb(N, Ndo)
        dim_eff = comb(N, Nup)
    bond_size = N * (N-1) / 2 + 2
    
    # Calculate matrix sizes
    hamiltonian_size = dim * dim * complex_size
    spin_matrix = dim_eff * dim_eff * complex_size
    
    # Calculate total memory for all matrices
    total_matrix_memory = hamiltonian_size * 2 + spin_matrix * bond_size
    
    # Convert to MB for readability
    def to_gb(bytes):
        return bytes / (1024 * 1024 * 1024)
    
    analysis = {
        "dimension": dim,
        "dimension_eff": dim_eff,
        "hamiltonian": {
            "size_bytes": hamiltonian_size,
            "size_gb": to_gb(hamiltonian_size)
        },
        "spin_matrices": {
            "size_bytes": spin_matrix,
            "size_gb": to_gb(spin_matrix)
        },
        "total": {
            "size_bytes": total_matrix_memory,
            "size_gb": to_gb(total_matrix_memory)
        }
    }
    
    return analysis 

def write_memory_status(result_dir, all_memory):
    """Write memory usage statistics to a file"""
    with open(os.path.join(result_dir, "memory_status.txt"), "w") as f:
        f.write("Memory Usage Statistics (MB)\n")
        f.write("=" * 50 + "\n\n")
        
        for rank_idx, rank_memory in enumerate(all_memory):
            f.write(f"Rank {rank_idx} Memory Usage:\n")
            f.write("-" * 30 + "\n")
            for stage, memory in rank_memory:
                f.write(f"{stage}: {memory:.2f} MB\n")
            f.write("\n")
                
            # Calculate memory differences
            if len(rank_memory) > 1:
                f.write("Memory Changes:\n")
                for i in range(1, len(rank_memory)):
                    prev_stage, prev_memory = rank_memory[i-1]
                    curr_stage, curr_memory = rank_memory[i]
                    diff = curr_memory - prev_memory
                    f.write(f"{prev_stage} -> {curr_stage}: {diff:+.2f} MB\n")
            f.write("\n")

def write_timing_status(result_dir, all_times, total_time):
    """Write timing statistics to a file"""
    with open(os.path.join(result_dir, "timing_status.txt"), "w") as f:
        f.write("Timing Statistics\n")
        f.write("=" * 50 + "\n\n")
        
        # Write total execution time
        f.write(f"Total execution time: {total_time:.6f} seconds\n\n")
        
        # Write per-rank timing information
        for rank_idx, rank_times in enumerate(all_times):
            f.write(f"Rank {rank_idx} Timing:\n")
            f.write("-" * 30 + "\n")
            for hole, class_idx, time in rank_times:
                f.write(f"Work item (hole={hole}, class={class_idx}): {time:.6f} seconds\n")
            f.write("\n")

