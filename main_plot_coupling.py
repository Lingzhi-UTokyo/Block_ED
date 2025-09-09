'''
This program is used to calculate the figure of double occupation
'''

import os
import sys
from mpi4py import MPI
import time
from Clusters_Square import *
import matplotlib.pyplot as plt
from utils_main import *
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Monitoring configuration
MONITOR_MEMORY = True  # Set to False to disable detailed memory monitoring
MONITOR_TIME = True    # Set to False to disable detailed timing monitoring

def setup_work_environment(N, sz=None):
    """Setup the working environment and parameters"""
    #result_dir = f"results_N{N}_U{U}_t{t}"
    if sz is None:
        result_dir = f"Block_Plot_N{N}"
    else:
        result_dir = f"Block_Plot_N{N}_sz{sz:.4f}"
    if rank == 0:
        os.makedirs(result_dir, exist_ok=True)
        print(f"Working directory: {result_dir}")
    return result_dir

def parse_arguments():
    N = 3
    restart = False
    type = None
    sz = None

    for arg in sys.argv[1:]:
        try:
            if '=' not in arg:
                continue
            key, value = arg.split('=')
            key = key.lower()
            
            if key == 'n':
                N = int(value)
            elif key == 'restart':
                restart = bool(value)
            elif key == 'type':
                type = value
            elif key == 'sz':
                sz = float(value)
        except ValueError:
            if is_root():
                print(f"Error: Invalid value for {key}")
                print("N must be an integer, U and t must be floating point numbers")
            sys.exit(1)

    if sz is not None:
        sz_new = round(sz * 2) / 2
        if abs(sz - sz_new) > 1e-6:
            if is_root():
                print(f"Warning: sz={sz} is not a half integer, set sz to {sz_new}")
        sz = sz_new
        if N%2 == 0:
            if int(sz * 2) % 2 != 0:
                sz = 0.0
                if is_root():
                    print(f"Warning: sz does not match the number of sites {N}, set sz to {sz}")
        else:
            if int(sz * 2) % 2 == 0:
                sz = 0.5
                if is_root():
                    print(f"Warning: sz does not match the number of sites {N}, set sz to {sz}")
        
    return N, type, sz, restart

def main(N, type, sz, restart):
    # Setup environment
    result_dir = setup_work_environment(N, sz)
    
    # Initialize variables for all ranks
    clusters = None
    work_items = None
    U=1.0
    #t_list = list(np.linspace(0.4000, 0.6000, 11))
    t_list = list(np.linspace(0.0200, 0.6000, 30))
    
    # Compute clusters on rank 0
    if rank == 0:
        print("Computing clusters...")
        clusters = Clusters_Square(N)
        clusters.compute_clsuters(if_print_time=True)
        clusters.classify_clusters(if_print_time=True)
        clusters.print_info()
        
        # Prepare work items
        work_items = [(hole, class_idx) 
                     for hole in range(len(clusters.clusters_classified))
                     for class_idx in range(len(clusters.clusters_classified[hole]))]
        print(f"\nPrepared {len(work_items)} work items for distribution")
        print("Broadcasting clusters...")
    
    # Ensure all ranks are ready to receive data
    comm.Barrier()
    
    # Broadcast data to all ranks
    # Rank 0 sends the data, other ranks receive it
    clusters = comm.bcast(clusters, root=0)
    work_items = comm.bcast(work_items, root=0)
    
    # Process work items
    rank_work_items = distribute_work(work_items, rank, size)
    
    if rank == 0:
        print("Starting computation...")
        print(f"Number of work items: {len(work_items)}")
        print(f"Number of ranks: {size}")
        print(f"Number of work items per rank: {len(rank_work_items)}\n")
    
    
    suffix=f""
    if type is not None:
        suffix = f"{suffix}_{type}"
    if sz is not None:
        suffix = f"{suffix}_sz{sz:.4f}"
    if restart:
        suffix = f"{suffix}_restart"
    errors_list=[]
    t11s_list=[]
    for work_idx, (hole, class_idx) in rank_work_items:
        # Process work item
        t0 = time.time()
        x_list=[]
        errors=[]
        t11s=[]
        for idx in range(len(t_list)):
            t=float(t_list[idx])
            x_list.append(t/U)
            base_filename = f"Block_U{U:.4f}_t{t:.4f}/N{N}{suffix}/hole{hole}_class{class_idx}_cluster0_results.txt"
            errors.append(read_key(base_filename, "Relative"))
            t11s.append(read_key(base_filename, "T11"))
        errors_list.append(errors)
        t11s_list.append(t11s)

        # Print progress for rank 0
        if rank == 0:
            print(f"Rank 0 completed task {work_idx+1} / {len(rank_work_items)}, time: {time.time()-t0:.1f}s")

    # Ensure all ranks have finished their work before gathering results
    comm.Barrier()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        if rank == 0:
            print("Usage: python main.py [N=n] [U=u] [t=t]")
        sys.exit(0)
    
    N, type, sz, restart = parse_arguments()
    if rank == 0:
        print(f"Using parameters: N={N}, type={type}, sz={sz}, restart={restart}")
    main(N, type, sz, restart)
    


