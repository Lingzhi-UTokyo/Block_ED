import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from spin_operators import *
from utils_state import *
from utils_math import *


class Hubbard_SingleBand:
    def __init__(self, N,  U, *args):
        self.N = N
        self.U = U
        if len(args) == 1 and isinstance(args[0], list):
            self.t_values = args[0]
        else:
            self.t_values = args

        # Initialize empty containers
        self._init_containers()
        self.dimspin = 2**N

    def _init_containers(self):
        """Initialize empty containers for data storage."""
        self.bonds = []    # i, j: index1, index2
        self.states = []
        self.hoppings = [] # t=<000...i_{\sigma}...000|H|000...j_{\sigma}...000>

        self.Ham = None
        self.eigvals = None
        self.eigvecs = None
        self.Heff = None

    def clear(self):
        """Clear all data in the model to free memory."""
        # Clear numpy arrays and sparse matrices
        for attr in ['Ham', 'eigvals', 'eigvecs', 'Heff']:
            if hasattr(self, attr):
                if isinstance(getattr(self, attr), (np.ndarray, sparse.spmatrix)):
                    delattr(self, attr)
                    setattr(self, attr, None)
            
        # Clear lists
        self.bonds = []
        self.states = []
        self.hoppings = []
        
        # Reset dimensions
        self.dimspin = 0

    def __del__(self):
        """Destructor to ensure memory is freed when object is deleted."""
        self.clear()

    def set_bonds_func(self, func, *args):
        self.bonds, self.hoppings=func(*args)

    def set_bonds(self, bonds, hoppings):
        """Directly set bonds and hoppings without using a function.
        
        Args:
            bonds (list): List of bonds, where each bond is a list of two site indices
            hoppings (list): List of hopping values corresponding to each bond
        """
        if len(bonds) != len(hoppings):
            raise ValueError("Number of bonds must match number of hoppings")
        self.bonds = bonds
        self.hoppings = hoppings

    def set_bonds_by_class(self, bond_classes, class_hoppings):
        """Set bonds and hoppings by bond classes.
        
        Args:
            bond_classes (list): 
                - For single class: list of bonds
                - For multiple classes: list of bond classes, where each class is a list of bonds
            class_hoppings (float or list): 
                - For single class: single hopping value
                - For multiple classes: list of hopping values, one for each bond class
        """
        # Handle single class case
        if isinstance(class_hoppings, (int, float)):
            self.bonds = bond_classes
            self.hoppings = [class_hoppings] * len(bond_classes)
            return
            
        # Handle multiple classes case
        if len(bond_classes) != len(class_hoppings):
            raise ValueError("Number of bond classes must match number of hopping values")
            
        # Flatten bond classes into a single list of bonds
        self.bonds = [bond for bond_class in bond_classes for bond in bond_class]
        
        # Create hoppings list by repeating the class hopping value for each bond in the class
        self.hoppings = []
        for bond_class, hopping in zip(bond_classes, class_hoppings):
            self.hoppings.extend([hopping] * len(bond_class))


    def set_states(self, nsites, nelec):
        self.states=Model_States_Nele(nsites, nelec, [0, 1,-1, 2])

    def set_states_sort(self):
        self.states=Model_State_Sort(self.states)

    def calc_ham_t_ij(self, state1_ref, state2_ref):
        state1=np.array(state1_ref)
        state2=np.array(state2_ref)
        if(sum_elec(state1, self.N) != sum_elec(state2, self.N)):
            return 0.0+0.0j
        
        #print(self.bonds)
        #print(self.hoppings)
        res=0.0+0.0j
        for m in range(self.N):
            for n in range(self.N):
                # Here we only consider bond list considering each bond once in the form of [m<n]
                index=find_index(self.bonds, (m,n) if m<=n else (n,m))
                if(index<0): continue
                if(m<n):
                    t=self.hoppings[index]
                else:
                    t=(self.hoppings[index]).conjugate()


                sign=1.0+0.0j
                spin_m=state1[m]
                spin_n=state2[n]
                
                # c^{\dagger}_{m\up}c_{n\up}
                # Remove Up spin and create Up spin
                if(spin_m==2):                     # Up, Down
                    sign_m=sign_state(state1,m)    # Before this site
                    state1[m]=-1                   # 0 , Down
                elif(spin_m==1):                   # Up, 0
                    sign_m=sign_state(state1,m)    # Before this site
                    state1[m]=0                    # 0 , 0
                else:                              # No Up spin
                    sign_m=0.0                     # 0 , 0 or 0, Down

                if(spin_n==2):                     # Up, Down
                    sign_n=sign_state(state2,n)    # Before this site
                    state2[n]=-1                   # 0 , Down
                elif(spin_n==1):                   # Up, 0
                    sign_n=sign_state(state2,n)    # Before this site
                    state2[n]=0                    # 0 , 0
                else:                              # No Up spin
                    sign_n=0.0                     # 0 , 0 or 0, Down

                if(judge_state_same(state1, state2)):
                    sign=sign_m*sign_n
                    res+=sign*t
                #if(abs(sign)>1e-6 and judge_state_same(state1, state2)):
                #    print(state1, state2, "site_m", m, "spin_m", spin_m, "site_n", n, "spin_n", spin_n)
                state1[m]=spin_m
                state2[n]=spin_n
                #if(abs(sign)>1e-6 and judge_state_same(state1, state2)):
                #    print(state1, state2, "site_m", m, "spin_m", spin_m, "site_n", n, "spin_n", spin_n)

                ## c^{\dagger}_{m\down}c_{n\down}
                # Remove Down spin and create Down spin
                if(spin_m==2):                        # Up, Down
                    sign_m=sign_state(state1,m)*(-1)  # Before this site * Up
                    state1[m]=1                       # Up, 0
                elif(spin_m==-1):                     # 0 , Down
                    sign_m=sign_state(state1,m)       # Before this site
                    state1[m]=0                       # 0 , 0
                else:                                 # No Down spin
                    sign_m=0.0                        # 0 , 0 or Up, 0

                if(spin_n==2):                        # Up, Down
                    sign_n=sign_state(state2,n)*(-1)  # Before this site * Up
                    state2[n]=1                       # Up, 0
                elif(spin_n==-1):                     # 0 , Down
                    sign_n=sign_state(state2,n)       # Before this site
                    state2[n]=0                       # 0 , 0
                else:                                 # No Down spin
                    sign_n=0.0                        # 0 , 0 or Up, 0

                if(judge_state_same(state1, state2)):
                    sign=sign_m*sign_n
                    res+=sign*t
                #if(abs(sign)>1e-6 and judge_state_same(state1, state2)):
                #    print(state1, state2, "site_m", m, "spin_m", spin_m, "site_n", n, "spin_n", spin_n)
                state1[m]=spin_m
                state2[n]=spin_n
                #if(abs(sign)>1e-6 and judge_state_same(state1, state2)):
                #    print(state1, state2, "site_m", m, "spin_m", spin_m, "site_n", n, "spin_n", spin_n)
        #print()
        return res
                
    def calc_ham_U_ij(self, state1_ref, state2_ref):
        state1=np.array(state1_ref)
        state2=np.array(state2_ref)

        if(judge_state_diff(state1, state2)):
            return 0.0+0.0j
        
        res=0.0+0.0j
        for m in range(self.N):
            if(state2[m]==2):
                res+=self.U

        return res
    
    def calc_ham_ij(self, state1, state2):
        res1=self.calc_ham_t_ij(state1, state2)
        res2=self.calc_ham_U_ij(state1, state2)
        #print("i ", state1, "j ", state2, "res1 ", res1, "res2 ", res2)
        return res1+res2
    
    def calc_hamiltonian(self):
        """Calculate the Hamiltonian matrix using sparse format."""
        NStates = len(self.states)
        # Initialize sparse matrix in COO format
        rows = []
        cols = []
        data = []
        
        for i in range(NStates):
            for j in range(NStates):
                val = self.calc_ham_ij(self.states[i], self.states[j])
                if abs(val) > 1e-10:  # Only store non-zero elements
                    rows.append(i)
                    cols.append(j)
                    data.append(val)
        
        # Convert to CSR format for efficient operations
        self.Ham = sparse.csr_matrix((data, (rows, cols)), shape=(NStates, NStates), dtype=complex)
        
        # Verify hermiticity
        if not np.allclose(self.Ham.toarray(), self.Ham.toarray().conj().T):
            print("The Hamiltonian is not hermitian! Please check your model and code!")
            exit(0)

    def solve(self):
        """Solve the eigenvalue problem using appropriate solver based on matrix size."""
        if self.Ham is None:
            raise ValueError("Hamiltonian matrix must be calculated before solving")
        
        N = self.Ham.shape[0]
        k = min(100, N-2)  # Ensure k < N-1
        
        if k < 10:  # For small matrices, use dense solver
            self.eigvals, self.eigvecs = np.linalg.eigh(self.Ham.toarray())
        else:  # For larger matrices, use sparse solver
            self.eigvals, self.eigvecs = eigsh(self.Ham, k=k, which='SA', return_eigenvectors=True)
            # Convert eigenvectors to dense format only for sparse case
            if isinstance(self.eigvecs, sparse.spmatrix):
                self.eigvecs = self.eigvecs.toarray()

    def calc_double_occupation_expectation(self):
        double_occupation_matrix=calc_double_occupation_matrix(self.states)
        return np.diag(self.eigvecs.conj().T @ double_occupation_matrix @ self.eigvecs)

    def calc_heff_halffilled(self):
        """Calculate the effective Hamiltonian for half-filling using sparse operations."""
        if self.eigvals is None or self.eigvecs is None:
            raise ValueError("Eigenvalues and eigenvectors must be calculated first")
            
        double_occupation_expectation = self.calc_double_occupation_expectation()
        sorted_indices = np.argsort(double_occupation_expectation)
        
        values_resort = self.eigvals[sorted_indices]
        vectors_resort = self.eigvecs[:,sorted_indices]

        Lambda = np.diag(values_resort[0:self.dimspin])
        S_BD = vectors_resort[0:self.dimspin, 0:self.dimspin]

        U, Sigma, VH = np.linalg.svd(S_BD)
        self.Heff = U @ VH @ Lambda @ VH.conj().T @ U.conj().T

    def set_heff(self, Heff):
        self.Heff=Heff

    def calc_spin_coeff(self, bonds):
        Ms_all = []
        Ms_class = []
        
        Ms_all.append(np.eye(self.dimspin, dtype=complex))
        Ms_class.append(np.eye(self.dimspin, dtype=complex))
        for i in range(len(bonds)):
            Ms_class.append(np.zeros([self.dimspin, self.dimspin],dtype=complex))
            for j in range(len(bonds[i])):
                bond=bonds[i][j]
                if(len(bond)==2):
                    Ms_all.append(spin_matrix_J_ij(self.states[0:self.dimspin], bond))
                elif(len(bond)==4):
                    Ms_all.append(spin_matrix_square(self.states[0:self.dimspin], bond))
                Ms_class[-1] += Ms_all[-1]
    
        b = self.Heff.flatten()

        A_all = np.array([Ms_all[i].flatten() for i in range(len(Ms_all))]).T
        individual_coeffs_temp = np.linalg.solve(A_all.T @ A_all, A_all.T @ b)

        # First element is constant offset
        individual_coeffs = [individual_coeffs_temp[0]]
        index = 1
        for bond_group in bonds:
            if len(bond_group) != 0:
                individual_coeffs.append(list(individual_coeffs_temp[index:index + len(bond_group)]))
                index += len(bond_group)

        A_class = np.array([Ms_class[i].flatten() for i in range(len(Ms_class))]).T
        class_coeffs = np.linalg.solve(A_class.T @ A_class, A_class.T @ b)

        individual_error = objective_function(individual_coeffs_temp, A_all, b)
        class_error = objective_function(class_coeffs, A_class, b)

        return (individual_coeffs, class_coeffs, individual_error, class_error)
        
