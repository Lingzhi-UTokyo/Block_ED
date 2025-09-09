from datetime import datetime
from mpi_config import comm, rank, size, is_root, barrier, broadcast, gather
from Hubbard_SingleBand import *
from Clusters_Square import *
from spin_operators import *
from utils_state import *
from utils_math import *
from utils_main import *
from utils_file import setup_work_environment_previous


# Monitoring configuration
FORCE_DISTRIBUTE = True

def main(params):
    # Setup environment
    result_dir1, result_dir2, result_dir3 = setup_work_environment(params, rank=rank)
    params['result_dir1'] = result_dir1
    params['result_dir2'] = result_dir2
    params['result_dir3'] = result_dir3
    params['result_dir'] = f"{result_dir1}/{result_dir2}"
    
    # Initialize timing and memory tracking
    t0_total = time.time()
    computation_times = []
    rank_memory_usage = [("initial", get_memory_usage())]
    
    # Initialize variables for all ranks
    clusters = None
    work_items = None
    
    # Compute clusters on rank 0
    if is_root():
        # Analyze memory usage for the largest cluster
        memory_analysis = analyze_matrix_memory(params['N'], params['sz'])
        print("\nMemory Analysis:")
        print(f"Maximum Hilbert space dimension: {memory_analysis['dimension']}")
        print(f"Maximum effective space dimension: {memory_analysis['dimension_eff']}")
        print(f"Hamiltonian matrix size: {memory_analysis['hamiltonian']['size_gb']:.10f} GB")
        print(f"Spin matrices size:: {memory_analysis['spin_matrices']['size_gb']:.10f} GB")
        print(f"Total matrix memory: {memory_analysis['total']['size_gb']:.10f} GB\n")

        print("Computing clusters...")
        clusters = Clusters_Square(params['N'])
        clusters.compute_clsuters(t2=params['t2'], if_print_time=True)
        clusters.classify_clusters(t2=params['t2'], if_print_time=True)
        clusters.print_info()
        
        # Prepare work items
        work_items_all = [(hole, class_idx) 
                     for hole in range(len(clusters.clusters_classified))
                     for class_idx in range(len(clusters.clusters_classified[hole]))]
        print(f"\nPrepared {len(work_items_all)} work items for distribution")

        work_items=[]
        if params['restart'] and not FORCE_DISTRIBUTE:
            #os.system(f"rm -rf {result_dir}_restart/restart_flag.txt")
            for idx, (hole, class_idx) in enumerate(work_items_all):
                pattern = f"Block_U{params['U']:.4f}_t{params['t']:.4f}/N{params['N']}*/hole{hole}_class{class_idx}_cluster0_results.txt"
                files = glob.glob(pattern)
                errors=[]
                t11s=[]
                for file in files:
                    errors.append(read_key(file, "Relative") or 100000)
                    t11s.append(read_key(file, "T11") or 100000)
                
                idx_selected=t11s.index(min(t11s))
                if not errors or errors[idx_selected]>0.05:
                    work_items.append(((hole, class_idx)))
                else:
                    base_path = files[idx_selected].replace("_cluster0_results.txt", "")
                    if base_path == f"Block_U{params['U']:.4f}_t{params['t']:.4f}/N{params['N']}_{params['type']}_restart/hole{hole}_class{class_idx}":
                        continue
                    os.system(f"cp {base_path}* {result_dir}_restart/")
                    os.system(f"rm {result_dir}_restart/*_eigvals.npy")
                    os.system(f"rm {result_dir}_restart/*_eigvecs.npy")
                    write_file(f"{result_dir}_restart/restart_flag.txt", f"cp {base_path}* ")
        else:
            work_items = work_items_all
        
        rank_memory_usage.append(("after_cluster_computation", get_memory_usage()))
        
        print("Broadcasting clusters...")
    
    # Ensure all ranks are ready to receive data
    barrier()
    
    # Broadcast data to all ranks
    # Rank 0 sends the data, other ranks receive it
    clusters = broadcast(clusters, root=0)
    work_items = broadcast(work_items, root=0)
    
    rank_memory_usage.append(("after_broadcast", get_memory_usage()))
    
    # Process work items
    rank_work_items = distribute_work(work_items, rank, size)
    if is_root(size-1):
        print(f"Restart from previous computation: {params['restart']}") if params['restart'] else print("Starting computation...")
        print(f"We adopt the \"{params['type']}\" scheme to calculate T11")
        print(f"Number of work items: {len(work_items)}")
        print(f"Number of ranks: {size}")
        print(f"Number of work items per rank: {len(rank_work_items)}")
        print(f"Up to now, time cost: {time.time()-t0_total:.1f}s\n")
        if params['type'] == 'adiabatic':
            dir1, dir2, dir3 = setup_work_environment_previous(params)
            if params['restart']:
                dir2 = f"{dir2}_restart"
            print(f"In the adiabatic process, read data from {dir1}/{dir2}", flush=True)
    
    for work_idx, (hole, class_idx) in rank_work_items:
        # Process work item
        t0 = time.time()
        cluster_process_work_item(hole, class_idx, clusters, params, rank)
        computation_times.append((hole, class_idx, time.time()-t0))
        rank_memory_usage.append((f"class_{hole}_{class_idx}", get_memory_usage()))
        
        # Print progress for rank 0
        if is_root(size-1):
            print(f"Rank {size-1} completed task {work_idx+1} / {len(rank_work_items)}, time: {time.time()-t0:.1f}s", flush=True)


    # Ensure all ranks have finished their work before gathering results
    barrier()
                    
    # Record final memory usage
    rank_memory_usage.append(("final", get_memory_usage()))
    all_times = gather(computation_times, root=0)
    all_memory = gather(rank_memory_usage, root=0)
    
    # Print results on rank 0
    if is_root():
        total_time = time.time() - t0_total
        print(f"\nTotal execution time: {total_time:.6f} seconds")

        result_dir = f"{params['result_dir']}"
        if params['restart']:
            result_dir = f"{result_dir}_restart"
        write_timing_status(result_dir, all_times, total_time)
        print("Timing statistics written to timing_status.txt")
        
        write_memory_status(result_dir, all_memory)
        print("Memory statistics written to memory_status.txt")

    
    # Final barrier to ensure all ranks exit together
    barrier()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        if is_root():
            print("Usage: python main.py [N=n] [U=u] [t=t]")
            print("  N: number of sites (default: 3)")
            print("  U: Hubbard U parameter (default: 6.0)")
            print("  t: hopping parameter (default: 1.0)")
            print("\nExamples:")
            print("  python main.py              # Use default values")
            print("  python main.py N=4 U=3 t=2  # Custom values")
        sys.exit(0)
    
    params = parse_arguments(rank=rank)
    
    if is_root():
        print(f"Task is started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Using parameters: N={params['N']}, U={params['U']}, t={params['t']}, t2={params['t2']}, type={params['type']}, sz={params['sz']}, restart={params['restart']}")
    
    main(params)
    
    if is_root():
        print(f"Task is finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


