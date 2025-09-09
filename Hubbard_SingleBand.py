import time
import math
import random
import numpy as np
from scipy.linalg import block_diag
from mpi_config import rank, size, is_root
from spin_operators import *
from utils_state import *
from utils_math import *
from utils_file import setup_work_environment_previous

def compute_T11m1_norm(eigvecs, indices, dimspin):
    try:
        S_BD=eigvecs[0:dimspin, indices]
        U, Sigma, VH=np.linalg.svd(S_BD)
        T11m1=U @ np.diag(Sigma) @ U.conj().T-np.eye(dimspin)
        return norm_matrix(T11m1)
    except:
        return np.inf

def block2m(eigvals_block, eigvecs_block, states_block, states):
    eigvals = np.concatenate(eigvals_block)
    eigvecs = block_diag(*eigvecs_block)

    states_block_flat = [s for block in states_block for s in block]
    perm = [states_block_flat.index(s) for s in states]
    eigvals_indices=np.argsort(eigvals)

    return eigvals[eigvals_indices], eigvecs[:, eigvals_indices][perm, :]

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
        self.dimspin_s2 = -1

    def _init_containers(self):
        """Initialize empty containers for data storage."""
        self.bonds = []    # i, j: index1, index2
        self.states = []
        self.hoppings = [] # t=<000...i_{\sigma}...000|H|000...j_{\sigma}...000>

        self.Ham = None
        self.eigvals = None
        self.eigvecs = None
        self.sz = None

        self.Heff = None
        self.error = False
        self.T11m1 = None
        self.T11m1_norm = np.inf
        self.t11_selected_indices = None
        self.t11_selected_occupation = None
        self.double_occupation_expectation = None

        self.block = False
        self.Ham_block = None
        self.eigvals_block = None
        self.eigvecs_block = None
        self.states_block = None
        self.S2 = None
        self.S2_eigvals = None
        self.S2_eigvecs = None

        self.eigvals_s2 = None

        self.s2_U = None
        self.s2_Us = None
        self.s2_U_spin = None
        self.s2_Us_spin = None

        self.s2_U_selected = None
        self.s2_Us_selected = None
        self.s2_U_spin_selected = None
        self.s2_Us_spin_selected = None

    def clear(self):
        # Clear numpy arrays
        for attr in ['Ham', 'eigvals', 'eigvecs', 'Heff']:
            if hasattr(self, attr) and isinstance(getattr(self, attr), np.ndarray):
                delattr(self, attr)
                setattr(self, attr, None)
            
        # Clear lists
        self.bonds = []
        self.states = []
        self.hoppings = []
        
        # Reset dimensions
        self.dimspin = 0
        self.dimspin_s2 = -1
        

    def __del__(self):
        self.clear()

    def set_block(self, block):
        self.block = block

    def set_bonds_func(self, func, *args):
        self.bonds, self.hoppings=func(*args)

    def set_bonds(self, bonds, hoppings):
        if len(bonds) != len(hoppings):
            raise ValueError("Number of bonds must match number of hoppings")
        self.bonds = bonds
        self.hoppings = hoppings

    def set_bonds_by_class(self, bond_classes, class_hoppings):
        # Handle single class case
        if class_hoppings is None:
            return
        elif isinstance(class_hoppings, (int, float)):
            self.bonds.extend(bond_classes)
            self.hoppings.extend([class_hoppings for _ in bond_classes])
            return
            
        # Handle multiple classes case
        if len(bond_classes) != len(class_hoppings):
            raise ValueError("Number of bond classes must match number of hopping values")
        self.bonds.extend([bond for bond_class in bond_classes for bond in bond_class])
        for bond_class, hopping in zip(bond_classes, class_hoppings):
            self.hoppings.extend([hopping for _ in bond_class])


    def set_states(self, nsites, nelec, sz=None, s2=None):
        if not self.block:
            if sz is None:
                self.states = Model_States_Nele(nsites, nelec, [0, 1,-1, 2])
            else:
                self.states = Model_States_Nele_Sz(nsites, nelec, sz, [0, 1,-1, 2])
                self.sz = sz
                self.dimspin = len([s for s in self.states if is_half_filled(s)])
        else:
            self.states = Model_States_Nele(nsites, nelec, [0, 1,-1, 2])
            sz_lim = nelec if nelec <= nsites else 2*nsites-nelec
            if sz_lim < 0:
                print("Error in setting the nelec: {nelec} and nsites: {nsites}")
                exit(1)
            self.states_block = []
            for i in range(-sz_lim, sz_lim+1,2):
                self.states_block.append(Model_States_Nele_Sz(nsites, nelec, i*0.5, [0, 1,-1, 2]))
            

    def set_states_sort(self):
        self.states=Model_State_Sort(self.states)
        if self.block:
            for i in range(len(self.states_block)):
                self.states_block[i] = Model_State_Sort(self.states_block[i])


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
                    res-=sign*t
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
                    res-=sign*t
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
        if not self.block:
            NStates=len(self.states)
            self.Ham=np.zeros([NStates, NStates], dtype=complex)
            for i in range(NStates):
                for j in range(NStates):
                    self.Ham[i][j]=self.calc_ham_ij(self.states[i], self.states[j])
            #Matrix_Out(Chop(self.Ham))
            is_hermitian = np.allclose(self.Ham, self.Ham.conj().T)
            if not is_hermitian:
                print("The Hamitonian is not hermitian! Please check your model and code!")
                exit(0)
        else:
            self.Ham_block=[np.zeros([len(states), len(states)], dtype=complex) for states in self.states_block]
            for idx in range(len(self.states_block)):
                states=self.states_block[idx]
                for i in range(len(states)):
                    for j in range(len(states)):
                        self.Ham_block[idx][i][j]=self.calc_ham_ij(states[i], states[j])

    
    def solve(self, Us=None):
        if not self.block:
            if Us is None:
                self.eigvals, self.eigvecs=np.linalg.eigh(self.Ham)
            else:
                # We construct the Hamtonian in the S^2 space and then transform it back into original form
                eigvals_list = []
                eigvecs_list = []
                #hams_list = []
                for U_trans in Us:
                    eigvals, eigvecs=np.linalg.eigh(U_trans.conj().T @ self.Ham @ U_trans)
                    #hams_list.append(U_trans.conj().T @ self.Ham @ U_trans)
                    eigvals_list.append(eigvals)
                    eigvecs_list.append(U_trans @ eigvecs)
                self.eigvals = np.concatenate(eigvals_list)
                self.eigvecs = np.concatenate(eigvecs_list, axis=1)
                #delta = np.diag(self.eigvals) - self.eigvecs.conj().T  @ self.Ham @ self.eigvecs
                #print(f"Delta: {np.max(np.abs(delta))}")

        else:
            self.eigvals_block = []
            self.eigvecs_block = []
            for idx in range(len(self.states_block)):
                eigvals_tmp, eigvecs_tmp = np.linalg.eigh(self.Ham_block[idx])
                self.eigvals_block.append(eigvals_tmp)
                self.eigvecs_block.append(eigvecs_tmp)
            self.eigvals, self.eigvecs=block2m(self.eigvals_block, self.eigvecs_block, self.states_block, self.states)
            
    

    def set_heff(self, Heff):
        self.Heff=Heff

    def restart(self, base_filename):
        try:
            self.read_eigvals(base_filename+"_eigvals.npy")
            self.read_eigvecs(base_filename+"_eigvecs.npy")
        except:
            raise RuntimeError("Failed to read eigvals or eigvecs")

    def read_eigvals(self, filename):
        try:
            # Try to read as .npy file first
            self.eigvals = np.load(filename, allow_pickle=False)
        except (FileNotFoundError, OSError, ValueError):
            try:
                # If .npy fails, try to read as .txt file
                self.eigvals = np.loadtxt(filename, dtype=complex)
            except (FileNotFoundError, OSError, ValueError) as e:
                raise RuntimeError(f"Failed to read eigvals from {filename} (both .npy and .txt formats): {str(e)}")

    def read_eigvecs(self, filename):
        try:
            # Try to read as .npy file first
            self.eigvecs = np.load(filename, allow_pickle=False)
        except (FileNotFoundError, OSError, ValueError):
            try:
                # If .npy fails, try to read as .txt file
                self.eigvecs = np.loadtxt(filename, dtype=complex)
            except (FileNotFoundError, OSError, ValueError) as e:
                raise RuntimeError(f"Failed to read eigvecs from {filename} (both .npy and .txt formats): {str(e)}")

    def calc_double_occupation_expectation(self):
        double_occupation_matrix=calc_double_occupation_matrix(self.states)
        return np.diag(self.eigvecs.conj().T @ double_occupation_matrix @ self.eigvecs)

    def calc_heff_halffilled(self, params, params_cluster):
        self.t11_selected_indices = self.select_eigvecs(params, params_cluster)

        self.double_occupation_expectation=self.calc_double_occupation_expectation()
        self.t11_selected_occupation=self.double_occupation_expectation[self.t11_selected_indices]
        try:
            S_BD=self.eigvecs[0:self.dimspin, self.t11_selected_indices]
            U, Sigma, VH=np.linalg.svd(S_BD)
            self.T11m1=U @ np.diag(Sigma) @ U.conj().T-np.eye(self.dimspin)
            self.T11m1_norm=norm_matrix(self.T11m1)

            Lambda=np.diag(self.eigvals[self.t11_selected_indices])
            self.Heff=U @ VH @ Lambda @ VH.conj().T @ U.conj().T
        except:
            self.error=True
            self.T11m1=None
            self.T11m1_norm=np.inf
            self.Heff=None
            self.t11_selected_indices=None
            self.t11_selected_occupation=None
            self.double_occupation_expectation=None
    
    def select_eigvecs(self, params, params_cluster):
        t11_selected_indices = None
        method = params['type']

        dimspin = self.dimspin
        eigvals = self.eigvals
        eigvecs = self.eigvecs
        double_occupation = self.calc_double_occupation_expectation()

        best_norm=0
        if method is None:
            t11_selected_indices, best_norm = self.min_t11_occ(double_occupation, eigvecs, dimspin)
            if best_norm is np.inf:
                t11_selected_indices, best_norm = self.min_t11_single(double_occupation, eigvecs, dimspin, ratio=8)
        elif method.lower() == "occ":
            t11_selected_indices, best_norm = self.min_t11_occ(double_occupation, eigvecs, dimspin)
        elif method.lower() == "energy":
            t11_selected_indices, best_norm = self.min_t11_energy(double_occupation, eigvecs, dimspin, eigvals)
        elif method.lower() == 'single':
            t11_selected_indices, best_norm = self.min_t11_single(double_occupation, eigvecs, dimspin, ratio=8)
        elif method.lower() == 'multi':
            t11_selected_indices, best_norm = self.min_t11_multi(double_occupation, eigvecs, dimspin, ratio=8, n_restarts=40, max_iters_rand=4)
        elif method.lower() == 'adiabatic':
            dir1, dir2, dir3 = setup_work_environment_previous(params)
            if params['restart']:
                dir2 = f"{dir2}_restart"
            filename_indices = f"{dir1}/{dir2}/hole{params_cluster['hole']}_class{params_cluster['class_idx']}"
            filename_eigvecs = f"{dir1}/{dir3}/hole{params_cluster['hole']}_class{params_cluster['class_idx']}"
            t11_selected_indices = self.max_overlap_adiabatic(eigvecs, filename_eigvecs, filename_indices)
        else:
            raise ValueError(f"Invalid method: {method}")
        
        return t11_selected_indices
    
    def select_eigvecs_s2(self, params, params_cluster):
        t11_selected_indices = None
        s2 = params['s2']
        method = params['type']

        if s2 is None:
            raise ValueError(f"s2 is not set")

        dimspin_s2 = [ len(indices) for indices in self.s2_indices_spin ]
        eigvals_s2 = [ self.eigvals[indices] for indices in self.s2_indices ]
        eigvecs_s2 = [ self.eigvecs[:, indices] for indices in self.s2_indices ]
        double_occupation_s2 = [ self.double_occupation_expectation[indices] for indices in self.s2_indices ]

        best_norm=0
        if method is None:
            t11_selected_indices, best_norm = self.min_t11_occ(double_occupation_s2, eigvecs_s2, dimspin_s2)
            if best_norm is np.inf:
                t11_selected_indices, best_norm = self.min_t11_single(double_occupation_s2, eigvecs_s2, dimspin_s2, ratio=8)
        elif method.lower() == "occ":
            t11_selected_indices, best_norm = self.min_t11_occ(double_occupation_s2, eigvecs_s2, dimspin_s2)
        elif method.lower() == "energy":
            t11_selected_indices, best_norm = self.min_t11_energy(double_occupation_s2, eigvecs_s2, dimspin_s2, eigvals_s2)
        elif method.lower() == 'single':
            t11_selected_indices, best_norm = self.min_t11_single(double_occupation_s2, eigvecs_s2, dimspin_s2, ratio=8)
        elif method.lower() == 'multi':
            t11_selected_indices, best_norm = self.min_t11_multi(double_occupation_s2, eigvecs_s2, dimspin_s2, ratio=8, n_restarts=40, max_iters_rand=4)
        elif method.lower() == 'adiabatic':
            dir1, dir2, dir3 = setup_work_environment_previous(params)
            if params['restart']:
                dir2 = f"{dir2}_restart"
            filename_indices = f"{dir1}/{dir2}/hole{params_cluster['hole']}_class{params_cluster['class_idx']}"
            filename_eigvecs = f"{dir1}/{dir3}/hole{params_cluster['hole']}_class{params_cluster['class_idx']}"
            t11_selected_indices = self.max_overlap_adiabatic(eigvecs_s2, filename_eigvecs, filename_indices)
        else:
            raise ValueError(f"Invalid method: {method}")
        
        return t11_selected_indices

        
    def min_t11_occ(self, double_occ, eigvecs, dimspin):
        sorted_indices=np.argsort(double_occ)
        return sorted_indices[0:dimspin], compute_T11m1_norm(eigvecs, sorted_indices[0:dimspin], dimspin)

    def min_t11_occ_s2(self, double_occ_s2, eigvecs_s2, dimspin_s2):
        sorted_indices_s2= [np.argsort(double_occ) for double_occ in double_occ_s2]
        sorted_indices_selected= [sorted_indices[0:dimspin] for sorted_indices, dimspin in zip(sorted_indices_s2, dimspin_s2)]
        return np.concatenate(sorted_indices_selected), compute_T11m1_norm(np.concatenate(eigvecs_s2, axis=1),np.concatenate(sorted_indices_selected), np.sum(dimspin_s2))

    def min_t11_energy(self, double_occ, eigvecs, dimspin, eigvals):
        sorted_indices=np.argsort(eigvals)
        sorted_indices_selected = sorted_indices[0:dimspin]
        sorted_by_occ = np.argsort(double_occ[sorted_indices_selected])
        return sorted_indices_selected[sorted_by_occ], compute_T11m1_norm(eigvecs, sorted_indices_selected[sorted_by_occ], dimspin)

    def min_t11_energy_s2(self, double_occ_s2, eigvecs_s2, dimspin_s2, eigvals_s2):
        sorted_indices_s2= [np.argsort(eigvals) for eigvals in eigvals_s2]
        sorted_indices_selected_s2= [sorted_indices[0:dimspin] for sorted_indices, dimspin in zip(sorted_indices_s2, dimspin_s2)]
        sorted_by_occ_s2= [np.argsort(double_occ[sorted_indices_selected]) for double_occ, sorted_indices_selected in zip(double_occ_s2, sorted_indices_selected_s2)]
        sorted_indices_selected_by_occ_s2= [ sorted_indices_selected[sorted_by_occ] for sorted_indices_selected, sorted_by_occ in zip(sorted_indices_selected_s2, sorted_by_occ_s2)]
        return np.concatenate(sorted_indices_selected_by_occ_s2), compute_T11m1_norm(np.concatenate(eigvecs_s2, axis=1), np.concatenate(sorted_indices_selected_by_occ_s2), np.sum(dimspin_s2))

    def greedy_swap(self, init_idx, others, eigvecs, dimspin, f=None):
        t11_selected_indices=init_idx.copy()
        t11_swap_indices=others.copy()

        best_norm=compute_T11m1_norm(eigvecs, t11_selected_indices, dimspin)
        if best_norm is np.inf and f is not None:
            f.write(f"Failure in SVD, norm is set to inf\n")

        for i in range(len(t11_selected_indices)):
            norm_list=[]
            for j in range(len(t11_swap_indices)):
                t11_selected_indices[i], t11_swap_indices[j] = t11_swap_indices[j], t11_selected_indices[i]

                t0=time.time()
                norm = compute_T11m1_norm(eigvecs, t11_selected_indices, dimspin)
                if f is not None and norm is not np.inf:
                    f.write(f"Time cost in SVD: {(time.time()-t0)*1000:.4f}ms, norm: {norm:.12f}\n")
                elif f is not None and norm is np.inf:
                    f.write(f"Time cost in SVD: {(time.time()-t0)*1000:.4f}ms, failure in SVD\n")
                norm_list.append(norm)

                t11_selected_indices[i], t11_swap_indices[j] = t11_swap_indices[j], t11_selected_indices[i]

            index_min=np.argmin(norm_list)
            if norm_list[index_min]<best_norm:
                best_norm=norm_list[index_min]
                t11_selected_indices[i], t11_swap_indices[index_min] = t11_swap_indices[index_min], t11_selected_indices[i]
        
        if best_norm is np.inf:
            if f is not None:
                f.write(f"All svd failed, norm is set to inf, return the initial indices\n")
            return init_idx.copy(), others.copy(), best_norm

        return t11_selected_indices, t11_swap_indices, best_norm

    def min_t11_single(self, double_occ, eigvecs, dimspin, 
                       ratio=5):
        t0=time.time()
        sorted_indices=np.argsort(double_occ)
        size_space_mint11 = min(ratio*dimspin, len(double_occ))
        init_idx=sorted_indices[0:dimspin].copy()
        others=sorted_indices[dimspin:size_space_mint11].copy()

        if self.sz is None:
            f_svd=open(f"tmp/N{self.N}_U{self.U:.4f}_t{self.t_values[0]:.4f}_rank{rank}_single.txt", "w")
        else:
            f_svd=open(f"tmp/N{self.N}_U{self.U:.4f}_t{self.t_values[0]:.4f}_sz{self.sz:.4f}_rank{rank}_single.txt", "w")
        f_svd.write(f"Start the greedy algorithm to find the best t11 indices\n")

        t11_selected_indices, t11_swap_indices, best_norm=self.greedy_swap(init_idx, others, eigvecs, dimspin, f_svd)
        f_svd.write(f"Best |T11-1| = {best_norm:.12f}\n")
        f_svd.write(f"Total time cost: {time.time()-t0:.1f}s\n")
        f_svd.close()

        # Resort the final selected indices by double occupation values
        sorted_by_occ = np.argsort(double_occ[t11_selected_indices])
        return t11_selected_indices[sorted_by_occ], best_norm

    def min_t11_multi(self, double_occ, eigvecs, dimspin, 
                      ratio=4, rand_frac=0.10, ratio_rand_swap=2,
                      n_restarts=10, max_iters_rand=None):
        t0=time.time()
        iter_rand = 0
        flag_rand = False
        sorted_indices=np.argsort(double_occ)
        size_space_mint11 = min(ratio*dimspin, len(double_occ))

        if self.sz is None:
            f=open(f"tmp/N{self.N}_U{self.U:.4f}_t{self.t_values[0]:.4f}_rank{rank}_multi.txt", "w")
        else:
            f=open(f"tmp/N{self.N}_U{self.U:.4f}_t{self.t_values[0]:.4f}_sz{self.sz:.4f}_rank{rank}_multi.txt", "w")

        if max_iters_rand is None:
            max_iters_rand=n_restarts
            
        f.write(f"N={self.N}, U={self.U:.4f}, t={self.t_values[0]:.4f}, rank={rank}\n")
        f.write(f"n_restarts={n_restarts}, ratio={ratio}, rand_frac={rand_frac}, max_iters_rand={max_iters_rand}\n\n")
        f.write(f"Start to find the best t11 indices\n")
        f.flush()

        best_selected_indices=sorted_indices[0:dimspin].copy()
        best_swap_indices=sorted_indices[dimspin:size_space_mint11].copy()
        best_norm = compute_T11m1_norm(eigvecs, best_selected_indices, dimspin)
        if best_norm is np.inf:
            f.write(f"Failure in SVD, norm is set to inf\n")

        for iter_restarts in range(n_restarts):
            # 构造初始 idx：rand_frac 部分随机，其余按 double_occ 最小
            t1=time.time()
            if self.sz is None:
                f_svd=open(f"tmp/N{self.N}_U{self.U:.4f}_t{self.t_values[0]:.4f}_rank{rank}_multi_iter{iter_restarts}.txt", "w")
            else:
                f_svd=open(f"tmp/N{self.N}_U{self.U:.4f}_t{self.t_values[0]:.4f}_sz{self.sz:.4f}_rank{rank}_multi_iter{iter_restarts}.txt", "w")
            f_svd.write(f"Start the greedy algorithm to find the best t11 indices\n")
            
            if not flag_rand:
                # 使用当前最佳解进行贪婪搜索
                cur_selected_indices, cur_swap_indices, cur_norm=self.greedy_swap(best_selected_indices, best_swap_indices, eigvecs, dimspin, f_svd)
            else:
                # 创建随机化的初始解
                # 按double_occ排序
                sorted_selected = best_selected_indices[np.argsort(double_occ[best_selected_indices])].copy()
                sorted_swap = best_swap_indices[np.argsort(double_occ[best_swap_indices])].copy()
                
                # 随机交换部分元素
                n_rand = max(8, int(dimspin * rand_frac))
                if n_rand > 0 and len(sorted_swap) >= n_rand and dimspin >= n_rand:
                    # 选择要交换的空间
                    select_space = np.concatenate([sorted_selected[-n_rand:], sorted_swap[:n_rand*ratio_rand_swap]])
                    
                    # 随机选择n_rand个元素作为新的selected_indices
                    selected_indices = np.random.choice(len(select_space), size=n_rand, replace=False)
                    remaining_indices = np.setdiff1d(np.arange(len(select_space)), selected_indices)
                    
                    # 创建新的初始解
                    cur_selected_indices = sorted_selected.copy()
                    cur_swap_indices = sorted_swap.copy()
                    
                    # 替换selected_indices的最后n_rand个元素
                    cur_selected_indices[-n_rand:] = select_space[selected_indices]
                    cur_swap_indices[:n_rand*ratio_rand_swap] = select_space[remaining_indices]
                else:
                    # 如果无法进行随机交换，使用原始解
                    cur_selected_indices = sorted_selected.copy()
                    cur_swap_indices = sorted_swap.copy()
                
                # 对随机化的解进行贪婪搜索
                cur_selected_indices, cur_swap_indices, cur_norm=self.greedy_swap(cur_selected_indices, cur_swap_indices, eigvecs, dimspin, f_svd)
                
            f_svd.write(f"Best |T11-1| = {cur_norm:.12f}\n")
            f_svd.write(f"Total time cost: {time.time()-t1:.1f}s\n")
            f_svd.close()

            if cur_norm < best_norm and abs(cur_norm-best_norm)>1e-8:
                f.write(f"Succeed to find better solution at iteration {iter_restarts+1} / {n_restarts}, time cost: {time.time()-t1:.1f}s, current norm: {cur_norm:.12f}, best norm: {best_norm:.12f}")
                if not flag_rand:
                    f.write(f"\n")
                else:
                    f.write(f", random iteration {iter_rand + 1} / {max_iters_rand}\n")
                flag_rand=False
                iter_rand=0
                best_norm, best_selected_indices, best_swap_indices = cur_norm, cur_selected_indices.copy(), cur_swap_indices.copy()
            else:
                f.write(f" Failed to find better solution at iteration {iter_restarts+1} / {n_restarts}, time cost: {time.time()-t1:.1f}s, current norm: {cur_norm:.12f}, best norm: {best_norm:.12f}")
                if not flag_rand:
                    f.write(f", random swap starts!\n")
                    flag_rand=True
                    iter_rand=0
                else:
                    f.write(f", random iteration {iter_rand + 1} / {max_iters_rand}\n")
                    iter_rand+=1
                    if iter_rand>=max_iters_rand:
                        f.write(f"Break the loop!\n")
                        break
            f.flush()
                    
        f.write(f"Best |T11-1| = {best_norm:.12f}\n")
        f.write(f"Total time cost: {time.time()-t0:.1f}s\n")
        f.close()

        selected_double_occ = double_occ[best_selected_indices]
        sorted_by_occ = np.argsort(selected_double_occ)
        return best_selected_indices[sorted_by_occ], best_norm
    
    def max_overlap_adiabatic(self, eigvecs, filename_eigvecs, filename_indices):
        t0=time.time()
        try:
            eigvecs_previous = np.load(f"{filename_eigvecs}_eigvecs.npy", allow_pickle=False)
        except:
            eigvecs_previous = np.loadtxt(f"{filename_eigvecs}_eigvecs.npy", dtype=complex)
        selected_indices_previous = np.loadtxt(f"{filename_indices}_t11_selected_indices.npy", dtype=int)
        eigvecs_previous = eigvecs_previous[:, selected_indices_previous]

        Norms_Matrix = np.abs(eigvecs_previous.conj().T @ eigvecs) ** 2
        Norms_vector = np.sum(Norms_Matrix, axis=0)
        
        max_overlap_indices = np.argsort(Norms_vector)[-len(selected_indices_previous):]
        return max_overlap_indices
        
    
    def calc_spin_coeff(self, bonds, s2=None):
        dimspin = self.dimspin
        Ms_all = []
        Ms_all.append(np.eye(dimspin, dtype=complex))
        for bond_group in bonds:
            for bond in bond_group:
                if(len(bond)==2):
                    Ms_all.append(spin_matrix_J_ij(self.states[0:self.dimspin], bond))
                elif(len(bond)==4):
                    Ms_all.append(spin_matrix_JJ_ij(self.states[0:self.dimspin], bond))
                elif(len(bond)==6):
                    Ms_all.append(spin_matrix_JJJ_ij(self.states[0:self.dimspin], bond))
        
        
        b = self.Heff.flatten()
        A_all = np.array([Ms_all[i].flatten() for i in range(len(Ms_all))]).T
        coeffs_temp = np.linalg.solve(A_all.T @ A_all, A_all.T @ b)

        # First element is constant offset
        coeffs = [coeffs_temp[0]]
        index = 1
        for bond_group in bonds:
            if len(bond_group) != 0:
                coeffs.append(list(coeffs_temp[index:index + len(bond_group)]))
                index += len(bond_group)
        error = derivative_objective_function(coeffs_temp, A_all, b)
        return (coeffs, error)
    
    def calc_spin_coeff_class(self, bonds):
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
    
    def calc_s2(self):
        self.S2_basis = compute_S2_matrix(self.states)

        self.S2 = self.eigvecs.conj().T @ self.S2_basis @ self.eigvecs
        self.S4 = self.eigvecs.conj().T @ self.S2_basis @ self.S2_basis @ self.eigvecs

        self.S2_diag = np.array([self.S2[i][i].real for i in range(len(self.S2))])
        self.S2_error = np.array([self.S4[i][i] - self.S2[i][i]**2 for i in range(len(self.S2))])

    def block_s2(self, N, sz, s2):
        if s2 is None or sz is None:
            self.s2_U  = None
            self.s2_Us = None
            self.s2_U_spin = None
            self.s2_Us_spin = None

            self.s2_indices=None
            self.s2_indices_spin=None
            return
        
        s2_list = []
        s_start = 0.5 if N%2==1 else 0
        s_end   = N*0.5
        s_num = int((s_end - s_start)+1)
        s_list = np.linspace(s_start, s_end, s_num)
        for target_s in s_list:
            s2_list.append(target_s*(target_s+1))

        try:
            eigvals = np.load(f"S2/S2_basis_eigvals_N{N}_sz{sz}.npy", allow_pickle=False)
            eigvecs = np.load(f"S2/S2_basis_eigvecs_N{N}_sz{sz}.npy", allow_pickle=False)
            eigvecs_spin = np.load(f"S2/S2_basis_eigvecs_spin_N{N}_sz{sz}.npy", allow_pickle=False)
            eigvals_spin = np.load(f"S2/S2_basis_eigvals_spin_N{N}_sz{sz}.npy", allow_pickle=False)
        except:
            S2_basis = compute_S2_matrix(self.states)
            S2_basis_spin = compute_S2_matrix(self.states[0:self.dimspin])
            eigvals, eigvecs = np.linalg.eigh(S2_basis)
            eigvals_spin, eigvecs_spin = np.linalg.eigh(S2_basis_spin)
            np.save(f"S2/S2_basis_eigvals_N{N}_sz{sz}.npy", eigvals)
            np.save(f"S2/S2_basis_eigvecs_N{N}_sz{sz}.npy", eigvecs)
            np.save(f"S2/S2_basis_eigvals_spin_N{N}_sz{sz}.npy", eigvals_spin)
            np.save(f"S2/S2_basis_eigvecs_spin_N{N}_sz{sz}.npy", eigvecs_spin)

        
        self.s2_U = eigvecs
        self.s2_Us=[]
        self.s2_indices=[]
        self.s2_U_spin = eigvecs_spin
        self.s2_Us_spin=[]
        self.s2_indices_spin=[]
        for target_s2 in s2_list:
            idx_selected=[]
            idx_selected_spin=[]
            for idx, eigval in enumerate(eigvals):
                if abs(target_s2-eigval)<1e-6:
                    idx_selected.append(idx)
            for idx, eigval in enumerate(eigvals_spin):
                if abs(target_s2-eigval)<1e-6:
                    idx_selected_spin.append(idx)
            
            self.s2_indices.append(idx_selected)
            self.s2_Us.append(eigvecs[:, idx_selected])

            self.s2_indices_spin.append(idx_selected_spin)
            self.s2_Us_spin.append(eigvecs_spin[:, idx_selected_spin])


    def read_s2(self, N, sz, s2):
        if s2 is None or sz is None:
            self.s2_U = None
            self.s2_U_selected = None
            self.s2_U_spin = None
            self.s2_U_spin_selected = None
            return


        target_s = s2 + (0.5 if N%2==1 else 0)
        target_s2 = target_s*(target_s+1)
        try:
            eigvals = np.load(f"S2/S2_basis_eigvals_N{N}_sz{sz}.npy", allow_pickle=False)
            eigvecs = np.load(f"S2/S2_basis_eigvecs_N{N}_sz{sz}.npy", allow_pickle=False)
            eigvals_spin = np.load(f"S2/S2_basis_eigvals_spin_N{N}_sz{sz}.npy", allow_pickle=False)
            eigvecs_spin = np.load(f"S2/S2_basis_eigvecs_spin_N{N}_sz{sz}.npy", allow_pickle=False)
        except:
            S2_basis = compute_S2_matrix(self.states)
            S2_basis_spin = compute_S2_matrix(self.states[0:self.dimspin])
            eigvals, eigvecs = np.linalg.eigh(S2_basis)
            eigvals_spin, eigvecs_spin = np.linalg.eigh(S2_basis_spin)
            np.save(f"S2/S2_basis_eigvals_N{N}_sz{sz}.npy", eigvals)
            np.save(f"S2/S2_basis_eigvecs_N{N}_sz{sz}.npy", eigvecs)
            np.save(f"S2/S2_basis_eigvals_spin_N{N}_sz{sz}.npy", eigvals_spin)
            np.save(f"S2/S2_basis_eigvecs_spin_N{N}_sz{sz}.npy", eigvecs_spin)

        idx_selected=[]
        idx_spin_selected=[]
        for idx, eigval in enumerate(eigvals):
            if abs(target_s2-eigval)<1e-6:
                idx_selected.append(idx)
        for idx, eigval in enumerate(eigvals_spin):
            if abs(target_s2-eigval)<1e-6:
                idx_spin_selected.append(idx)

        self.dimspin_s2 = len(idx_spin_selected)
        self.s2_U = eigvecs
        self.s2_U_selected = eigvecs[:, idx_selected]
        self.s2_U_spin = eigvecs_spin
        self.s2_U_spin_selected = eigvecs_spin[:, idx_spin_selected]
    

    def read_test(self, N, sz, s2):
        try:
            eigvals = np.load(f"S2/S2_basis_eigvals_N{N}_sz{sz}.npy", allow_pickle=False)
            eigvecs = np.load(f"S2/S2_basis_eigvecs_N{N}_sz{sz}.npy", allow_pickle=False)
        except:
            S2_basis = compute_S2_matrix(self.states)
            eigvals, eigvecs = np.linalg.eigh(S2_basis)
            np.save(f"S2/S2_basis_eigvals_N{N}_sz{sz}.npy", eigvals)
            np.save(f"S2/S2_basis_eigvecs_N{N}_sz{sz}.npy", eigvecs)

        target_s = s2+ (0.5 if N%2==1 else 0)
        target_s2 = target_s*(target_s+1)

        idx_selected=[]
        for idx, eigval in enumerate(eigvals):
            if abs(target_s2-eigval)<1e-6:
                idx_selected.append(idx)
        return eigvecs


