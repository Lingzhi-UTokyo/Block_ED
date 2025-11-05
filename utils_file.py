import numpy as np

def write_file(filename, strings):
    with open(filename, "a") as f:
        f.write(f"{strings}\n")

def find_bond_vector(bond, cluster):
    """Find the bond vector of a bond"""
    site1, site2 = bond
    x1, y1 = cluster[site1]
    x2, y2 = cluster[site2]
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    return sorted([dx, dy], reverse=True)

def write_basic_info(f, hole, class_idx, cluster_idx, rank, cluster_time):
    """Write basic information about the calculation."""
    f.write(f"Hole: {hole}\n")
    f.write(f"Class: {class_idx}\n")
    f.write(f"Cluster: {cluster_idx}\n")
    f.write(f"Rank: {rank}\n")
    f.write(f"Computation time: {cluster_time:.6f} seconds\n")

def write_cluster_points(f, cluster):
    """Write cluster points information."""
    f.write("\n=== Cluster Points ===\n")
    for i, point in enumerate(cluster):
        f.write(f"Point {i}: {point}\n")

def get_all_possible_vectors(cluster, max_distance=None):
    """Get all possible bond vectors in the first quadrant below y=x."""
    if max_distance is None:
        max_distance = len(cluster)
    vectors = []
    for dx in range(max_distance):
        for dy in range(dx + 1):
            if dx == 0 and dy == 0:
                continue
            vectors.append((dx, dy))
    return sorted(vectors, key=lambda v: v[0]**2 + v[1]**2)


def write_coefficients(f, cluster, bonds, coeffs_list, error, T11m1_norm, overlap):
    """Write individual bond coefficients."""
    f.write("\n=== Individual Bond Coefficients ===\n")
    f.write(f"Constant term: {coeffs_list[0].real:.10f}\n")
    
    # Create a dictionary to store coefficients by vector
    coeffs_by_vector = {}
    for class_idx, bond_group in enumerate(bonds):
        if len(bond_group) > 0 and len(bond_group[0]) == 2:
            dx, dy = find_bond_vector(bond_group[0], cluster)
            if (dx, dy) not in coeffs_by_vector:
                coeffs_by_vector[(dx, dy)] = []
            coeffs_by_vector[(dx, dy)].append((class_idx, bond_group, coeffs_list[class_idx+1]))
    
    # Output all possible vectors
    all_vectors = get_all_possible_vectors(cluster)
    for bond_idx, (dx, dy) in enumerate(all_vectors):
        if (dx, dy) in coeffs_by_vector:
            for class_idx, bond_group, coeffs in coeffs_by_vector[(dx, dy)]:
                f.write(f"\nBond vector ({dx}, {dy}), J{bond_idx+1}:\n")
                for idx, (bond, coef) in enumerate(zip(bond_group, coeffs)):
                    site1, site2 = bond
                    f.write(f"    {idx}: Sites {site1}-{site2}: {coef.real:.10f}  +  {coef.imag:.10f}i\n")
        else:
            f.write(f"\nBond vector ({dx}, {dy}), J{bond_idx+1}: (not present in cluster)\n")
    
    f.write(f"\nFour-site bond\n")
    for class_idx, bond_group in enumerate(bonds):
        if len(bond_group) > 0 and len(bond_group[0]) == 4:
            for idx, bond in enumerate(bond_group):
                f.write(f"    class {idx//3+1} type {idx%3+1} : Four-site ({bond[0]}-{bond[1]}) * ({bond[2]}-{bond[3]}): {coeffs_list[class_idx+1][idx].real:.10f}  +  {coeffs_list[class_idx+1][idx].imag:.10f}i\n")
                if (idx+1)%3==0:
                    f.write("\n")

    f.write(f"\nSix-site bond\n")
    for class_idx, bond_group in enumerate(bonds):
        if len(bond_group) > 0 and len(bond_group[0]) == 6:
            for idx, bond in enumerate(bond_group):
                f.write(f"    class {idx//15+1} type {idx%15+1} : Six-site ({bond[0]}-{bond[1]}) * ({bond[2]}-{bond[3]}) * ({bond[4]}-{bond[5]}): {coeffs_list[class_idx+1][idx].real:.10f}  +  {coeffs_list[class_idx+1][idx].imag:.10f}i\n")
                if (idx+1)%15==0:
                    f.write("\n")
    
    """Write error information."""
    f.write("\n=== Individual Fit Error ===\n")
    f.write(f"Relative Error: {error[0]:.10f}\n")
    f.write(f"Residual: {error[1]:.10f}\n")
    f.write(f"R^2: {error[2]:.10f}\n")
    f.write(f"T11-1 norm: {T11m1_norm:.10f}\n")
    if overlap is not None:
        f.write(f"Overlap: {overlap:.10f}\n")
    


def write_errors(f, individual_error, class_error):
    """Write error information."""
    f.write("\n=== Individual Fit Error ===\n")
    np.savetxt(f, [individual_error])
    
    f.write("\n=== Class-Averaged Fit Error ===\n")
    np.savetxt(f, [class_error])

def write_bond_structure(f, cluster, bonds):
    """Write bond structure information."""
    f.write("\n=== Bond Structure ===")
    
    # Create a dictionary to store bonds by their vector
    bonds_by_vector = {}
    for class_idx, bond_group in enumerate(bonds):
        if len(bond_group) > 0 and len(bond_group[0]) == 2:
            dx, dy = find_bond_vector(bond_group[0], cluster)
            if (dx, dy) not in bonds_by_vector:
                bonds_by_vector[(dx, dy)] = []
            bonds_by_vector[(dx, dy)].append((class_idx, bond_group))
    
    # Output all possible vectors
    all_vectors = get_all_possible_vectors(cluster)
    for bond_idx, (dx, dy) in enumerate(all_vectors):
        f.write(f"\nBond vector ({dx}, {dy}), J{bond_idx+1}:")
        if (dx, dy) in bonds_by_vector:
            f.write("\n")
            for class_idx, bond_group in bonds_by_vector[(dx, dy)]:
                for idx, bond in enumerate(bond_group):
                    site1, site2 = bond
                    x1, y1 = cluster[site1]
                    x2, y2 = cluster[site2]
                    f.write(f"    {idx}: Sites {site1}-{site2} ({x1},{y1})-({x2},{y2})\n")
        else:
            f.write(" (not present in cluster)\n")
    
    # Output square bonds if any
    square_bonds = [bond for bond_group in bonds for bond in bond_group if len(bond) == 4]
    if square_bonds:
        f.write("\nFour-site bonds:\n")
        for class_idx, bond_group in enumerate(bonds):
            square_bonds_in_class = [bond for bond in bond_group if len(bond) == 4]
            if square_bonds_in_class:
                for idx, bond in enumerate(square_bonds_in_class):
                    f.write(f"    {idx//3+1} type {idx%3+1}: Four-site bond: {bond}\n")
                    if (idx+1)%3==0:
                        f.write("\n")

    # Output square bonds if any
    six_bonds = [bond for bond_group in bonds for bond in bond_group if len(bond) == 6]
    if six_bonds:
        f.write("\nSix-site bonds:\n")
        for class_idx, bond_group in enumerate(bonds):
            six_bonds_in_class = [bond for bond in bond_group if len(bond) == 6]
            if six_bonds_in_class:
                for idx, bond in enumerate(square_bonds_in_class):
                    f.write(f"    {idx//15+1} type {idx%15+1}: Six-site bond: {bond}\n")
                    if (idx+1)%15==0:
                        f.write("\n")

def write_results_to_file(f, hole, class_idx, cluster_idx, rank, cluster_time, 
                         cluster, bonds, coeffs, error, T11m1_norm, overlap):
    """Write all results to a file in a structured format."""
    write_basic_info(f, hole, class_idx, cluster_idx, rank, cluster_time)
    write_cluster_points(f, cluster)
    write_bond_structure(f, cluster, bonds)
    write_coefficients(f, cluster, bonds, coeffs, error, T11m1_norm, overlap)


def read_previous(filename):
    eigvals = np.load(f"{filename}_eigvals.npy", allow_pickle=False)
    eigvecs = np.load(f"{filename}_eigvecs.npy", allow_pickle=False)
    selected_indices = np.load(f"{filename}_selected_indices.npy", dtype=int)
    return eigvals, eigvecs, selected_indices

def setup_work_environment_previous(params):
    if params['delta'] is None:
        raise ValueError(f"delta is not set")
    if params['delta2'] is None:
        params['delta2'] = 0.0
    t1_previous = params['t'] - params['delta']
    t2_previous = params['t2'] - params['delta2'] if params['t2'] is not None else None
            
    result_dir1 = f"Block_U{params['U']:.4f}_t{t1_previous:.4f}"
    if t2_previous is not None:
        result_dir1 = f"{result_dir1}_tp{t2_previous:.4f}"

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

    return result_dir1, result_dir2, result_dir3
