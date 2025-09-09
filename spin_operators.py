import numpy as np
from utils_math import Matrix_Out


def compute_S2_basis(state1, state2):
    N = len(state1)
    if state1==state2:
        # Define local spin magnitude s_i (for S_i^2 eigenvalue s(s+1))
        # and m_i for S_i^z eigenvalue
        s_vals = np.where(np.abs(state1) == 1, 0.5, 0.0)  # spin-1/2 if singly occupied
        m_vals = [ 0.0 for _ in range(N)]
        for i in range(N):
            if state1[i] == 1 or state1[i] == -1:
                m_vals[i] = 0.5 * state1[i]
            else:
                m_vals[i] = 0.0
        
        # Compute sum of S_i^2 = s_i(s_i+1)
        S2_site = np.sum(s_vals * (s_vals + 1))
        # Compute pairwise dot products S_i · S_j
        S2_pair = 0.0
        for i in range(N):
            for j in range(i+1, N):
                S2_pair += m_vals[i] * m_vals[j]
    
        # Total S^2
        S2_total = S2_site + 2 * S2_pair
    else:
        S2_site=0
        S2_pair = 0.0
        for i in range(N):
            for j in range(i+1, N):
                # 非零条件一：S_i^+ S_j^-
                if (state1[i],state2[i]) == ( 1,-1) \
                and (state1[j],state2[j]) == (-1, 1) \
                and all(state1[k]==state2[k] for k in range(N) if k not in (i,j)):
                    S2_pair += 0.5  # 因为两项相加：S_i^+S_j^- 和 S_i^-S_j^+ 之一，前面乘2
                # 非零条件二：S_i^- S_j^+
                if (state1[i],state2[i]) == (-1, 1) \
                and (state1[j],state2[j]) == ( 1,-1) \
                and all(state1[k]==state2[k] for k in range(N) if k not in (i,j)):
                    S2_pair += 0.5
                
        S2_total = S2_site + 2 * S2_pair

    return S2_total

def compute_S2_vec(basis, eigvec1, eigvec2):
    N=len(basis)
    if len(basis)!=len(eigvec1):
        print("Error in compute_S2, please match the size of basis states and vectors!")

    S2=0.0j
    for i in range(N):
        for j in range(N):
            S2 += np.conjugate(eigvec1[i])*eigvec2[j]*compute_S2_basis(basis[i], basis[j])
    return S2

def compute_S2_matrix(basis):
    N=len(basis)
    basis_matrix=np.zeros([N,N], dtype=complex)
    for i in range(N):
        basis_matrix[i][i] = compute_S2_basis(basis[i], basis[i])
        for j in range(i+1, N):
            basis_matrix[i][j] = compute_S2_basis(basis[i], basis[j])
            basis_matrix[j][i] = np.conjugate(basis_matrix[i][j])
    return basis_matrix
    #return eigvecs.conj().T @ basis_matrix @ eigvecs, basis_matrix

def spin_moments(states, bond):
    dimspin = len(states)
    Identity = np.eye(dimspin, dtype=complex)
    SzSz = np.zeros((dimspin, dimspin), dtype=complex)
    for i, state1 in enumerate(states):
        SzSz[i, i] = Spin_z(state1[bond[0]]) * Spin_z(state1[bond[1]])
    return SzSz

def Spin_z(occupation):
    if occupation == 2 or occupation == 0:
        return 0
    elif occupation == 1:
        return 1/2
    elif occupation == -1:
        return -1/2
    else:
        raise ValueError(f"Invalid occupation: {occupation}")

def Spin_Flip_Plus(occupation):
    if occupation == -1:
        return 1
    elif occupation in (1, 2, 0):
        return -100
    else:
        raise ValueError(f"Invalid occupation: {occupation}")

def Spin_Flip_Minus(occupation):
    if occupation == 1:
        return -1
    elif occupation in (-1, 2, 0):
        return -100
    else:
        raise ValueError(f"Invalid occupation: {occupation}")

def spin_matrix_J_ij(states, bond):
    dimspin = len(states)
    Identity = np.eye(dimspin, dtype=complex)

    SzSz = np.zeros((dimspin, dimspin), dtype=complex)
    SpSm = np.zeros((dimspin, dimspin), dtype=complex)
    SmSp = np.zeros((dimspin, dimspin), dtype=complex)

    site1, site2 = bond[0], bond[1]
    for i, state1 in enumerate(states):
        # Calculate SzSz term for diagonal elements
        SzSz[i, i] = Spin_z(state1[site1]) * Spin_z(state1[site2])
        for j, state2 in enumerate(states):
            # Convert tuple to list for modification
            state2_list = list(state2)
            
            # Save original occupations
            occupation_1, occupation_2 = state2_list[site1], state2_list[site2]
            
            # Calculate S+S- term
            state2_list[site1] = Spin_Flip_Plus(occupation_1)
            state2_list[site2] = Spin_Flip_Minus(occupation_2)
            SpSm[i, j] = 1.0 + 0.0j if np.linalg.norm(np.array(state1) - np.array(state2_list)) < 1e-6 else 0.0 + 0.0j
            
            # Calculate S-S+ term
            state2_list[site1] = Spin_Flip_Minus(occupation_1)
            state2_list[site2] = Spin_Flip_Plus(occupation_2)
            SmSp[i, j] = 1.0 + 0.0j if np.linalg.norm(np.array(state1) - np.array(state2_list)) < 1e-6 else 0.0 + 0.0j

    return SzSz + 0.5 * (SpSm + SmSp)

def spin_matrix_JJ_ij_old(states, bond):
    dimspin = len(states)
    site1, site2, site3, site4 = bond[0], bond[1], bond[2], bond[3]
    
    # Initialize matrices for different terms
    SzSzSzSz = np.zeros((dimspin, dimspin), dtype=complex)  # S_i^z S_j^z S_k^z S_l^z
    SzSzSpSm = np.zeros((dimspin, dimspin), dtype=complex)  # S_i^z S_j^z (S_k^+ S_l^- + S_k^- S_l^+) * 0.5
    SpSmSzSz = np.zeros((dimspin, dimspin), dtype=complex)  # (S_i^+ S_j^- + S_i^- S_j^+) S_k^z S_l^z * 0.5
    SpSmSpSm = np.zeros((dimspin, dimspin), dtype=complex)  # (S_i^+ S_j^- + S_i^- S_j^+)(S_k^+ S_l^- + S_k^- S_l^+) * 0.25

    for i, state1 in enumerate(states):
        # Calculate SzSzSzSz term (diagonal elements)
        SzSzSzSz[i, i] = Spin_z(state1[site1]) * Spin_z(state1[site2]) * Spin_z(state1[site3]) * Spin_z(state1[site4])
        
        for j, state2 in enumerate(states):
            # Calculate SpSmSzSz term
            state2_list = list(state2)
            occ1, occ2 = state2_list[site1], state2_list[site2]
            
            # S_i^+ S_j^-
            state2_list[site1] = Spin_Flip_Plus(occ1)
            state2_list[site2] = Spin_Flip_Minus(occ2)
            if np.linalg.norm(np.array(state1) - np.array(state2_list)) < 1e-6:
                SpSmSzSz[i, j] += 0.5 * Spin_z(state1[site3]) * Spin_z(state1[site4])
            
            # S_i^- S_j^+
            state2_list[site1] = Spin_Flip_Minus(occ1)
            state2_list[site2] = Spin_Flip_Plus(occ2)
            if np.linalg.norm(np.array(state1) - np.array(state2_list)) < 1e-6:
                SpSmSzSz[i, j] += 0.5 * Spin_z(state1[site3]) * Spin_z(state1[site4])
            
            # Calculate SzSzSpSm term
            state2_list = list(state2)
            occ3, occ4 = state2_list[site3], state2_list[site4]
                
            # S_k^+ S_l^-
            state2_list[site3] = Spin_Flip_Plus(occ3)
            state2_list[site4] = Spin_Flip_Minus(occ4)
            if np.linalg.norm(np.array(state1) - np.array(state2_list)) < 1e-6:
                SzSzSpSm[i, j] += 0.5 * Spin_z(state1[site1]) * Spin_z(state1[site2])
                
            # S_k^- S_l^+
            state2_list[site3] = Spin_Flip_Minus(occ3)
            state2_list[site4] = Spin_Flip_Plus(occ4)
            if np.linalg.norm(np.array(state1) - np.array(state2_list)) < 1e-6:
                SzSzSpSm[i, j] += 0.5 * Spin_z(state1[site1]) * Spin_z(state1[site2])

            # Calculate SpSmSpSm term
            state2_list = list(state2)
            occ1, occ2, occ3, occ4 = state2_list[site1], state2_list[site2], state2_list[site3], state2_list[site4]
            
            # S_i^+ S_j^- S_k^+ S_l^-
            state2_list[site1] = Spin_Flip_Plus(occ1)
            state2_list[site2] = Spin_Flip_Minus(occ2)
            state2_list[site3] = Spin_Flip_Plus(occ3)
            state2_list[site4] = Spin_Flip_Minus(occ4)
            if np.linalg.norm(np.array(state1) - np.array(state2_list)) < 1e-6:
                SpSmSpSm[i, j] += 0.25
            
            # S_i^+ S_j^- S_k^- S_l^+
            state2_list[site1] = Spin_Flip_Plus(occ1)
            state2_list[site2] = Spin_Flip_Minus(occ2)
            state2_list[site3] = Spin_Flip_Minus(occ3)
            state2_list[site4] = Spin_Flip_Plus(occ4)
            if np.linalg.norm(np.array(state1) - np.array(state2_list)) < 1e-6:
                SpSmSpSm[i, j] += 0.25
            
            # S_i^- S_j^+ S_k^+ S_l^-
            state2_list[site1] = Spin_Flip_Minus(occ1)
            state2_list[site2] = Spin_Flip_Plus(occ2)
            state2_list[site3] = Spin_Flip_Plus(occ3)
            state2_list[site4] = Spin_Flip_Minus(occ4)
            if np.linalg.norm(np.array(state1) - np.array(state2_list)) < 1e-6:
                SpSmSpSm[i, j] += 0.25
            
            # S_i^- S_j^+ S_k^- S_l^+
            state2_list[site1] = Spin_Flip_Minus(occ1)
            state2_list[site2] = Spin_Flip_Plus(occ2)
            state2_list[site3] = Spin_Flip_Minus(occ3)
            state2_list[site4] = Spin_Flip_Plus(occ4)
            if np.linalg.norm(np.array(state1) - np.array(state2_list)) < 1e-6:
                SpSmSpSm[i, j] += 0.25

    # Combine all terms
    return SzSzSzSz + SzSzSpSm + SpSmSzSz + SpSmSpSm 

def spin_matrix_JJ_ij(states, bond):
    site1, site2, site3, site4 = bond[0], bond[1], bond[2], bond[3]

    spin_matrix1=spin_matrix_J_ij(states, [site1, site2])
    spin_matrix2=spin_matrix_J_ij(states, [site3, site4])

    return spin_matrix1 @ spin_matrix2

def spin_matrix_JJJ_ij(states, bond):
    site1, site2, site3, site4, site5, site6 = bond[0], bond[1], bond[2], bond[3], bond[4], bond[5]

    spin_matrix1=spin_matrix_J_ij(states, [site1, site2])
    spin_matrix2=spin_matrix_J_ij(states, [site3, site4])
    spin_matrix3=spin_matrix_J_ij(states, [site5, site6])

    return spin_matrix1 @ spin_matrix2 @ spin_matrix3
    
def spin_matrix_square(states, plaquette):
    bonds_list = [
        [plaquette[0], plaquette[1], plaquette[2], plaquette[3]],  # (S1·S2)(S3·S4)
        [plaquette[0], plaquette[3], plaquette[1], plaquette[2]],  # (S1·S4)(S2·S3)
        [plaquette[0], plaquette[2], plaquette[1], plaquette[3]]   # (S1·S3)(S2·S4)
    ]
    factors = [1, 1, -1]

    dimspin = len(states)
    spin_matrix = np.zeros((dimspin, dimspin), dtype=complex) 
    for idx, bonds in enumerate(bonds_list):
        spin_matrix += factors[idx] * spin_matrix_JJ_ij(states, bonds)

    return spin_matrix


def spin_matrix_square_bonds(states, plaquette):
    bonds_list = [
        [plaquette[0], plaquette[1], plaquette[2], plaquette[3]],  # (S1·S2)(S3·S4)
        [plaquette[0], plaquette[3], plaquette[1], plaquette[2]],  # (S1·S4)(S2·S3)
        [plaquette[0], plaquette[2], plaquette[1], plaquette[3]]   # (S1·S3)(S2·S4)
    ]

    spin_matrix=[]
    for bonds in bonds_list:
        spin_matrix.append(spin_matrix_JJ_ij(states, bonds))

    return spin_matrix, bonds_list