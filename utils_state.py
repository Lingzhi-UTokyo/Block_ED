import math
import numpy as np
import itertools

def sum_elec(state, index):
    return sum([abs(state[i]) for i in range(index)])

def total_mag(state):
    return sum([spin for spin in state if abs(spin)==1])

def find_index(lst, value):
    try:
        return lst.index(value)
    except ValueError:
        return -1

def is_half_filled(state):
    return all(s in (-1, 1) for s in state)

def sign_fermi(n):
    return 1 if n % 2 == 0 else -1

def sign_state(state, index):
    return sign_fermi(sum_elec(state,index))

def judge_state_same(state1, state2):
    return np.all((state1-state2) == 0)

def judge_state_diff(state1, state2):
    return np.any((state1-state2))

def Model_States_All(N, *args):
    if len(args) == 1 and isinstance(args[0], list):
        s_values = args[0]
    else:
        s_values = args

    return list(itertools.product(s_values, repeat=N))

def Model_States_Nele(Nsite, Nele, *args):
    states = Model_States_All(Nsite, *args)
    filtered_states = [state for state in states if sum_elec(state, Nsite)==Nele]

    return filtered_states

def Model_States_Nele_Sz(Nsite, Nele, Sz, *args):
    states = Model_States_All(Nsite, *args)
    filtered_states = [state for state in states if sum_elec(state, Nsite)==Nele and math.fabs(total_mag(state)-2*Sz) < 1e-6]
    return filtered_states

def sort_states(state_list):
    sorted_list = sorted(state_list, key=lambda s: (abs(total_mag(s)), -total_mag(s)))
    sorted_mags = [total_mag(s) for s in sorted_list]
    counts = {mag: len(list(group)) for mag, group in itertools.groupby(sorted_mags)}
    return sorted_list, counts

def Model_State_Sort(states):
    half_filled_states = [s for s in states if is_half_filled(s)]
    other_states = [s for s in states if not is_half_filled(s)]

    sorted_half_filled, counts_half_filled = sort_states(half_filled_states)
    sorted_other, counts_other = sort_states(other_states)

    return sorted_half_filled + sorted_other

def Model_States_Spin(N):
    return list(itertools.product([1,-1], repeat=2*N))

def calc_double_occupation(state):
    return state.count(2)

def calc_double_occupation_matrix(states):
    DoubleOccupation = [0]*len(states)
    for i, state in enumerate(states):
        DoubleOccupation[i] = calc_double_occupation(state)
    return np.diag(DoubleOccupation) 

def tune_sz(sz, N, rank=0):
    if sz is None:
        return None
    else:
        sz_new = round(sz * 2) / 2
        if abs(sz - sz_new) > 1e-6:
            if rank == 0:
                print(f"Warning: sz={sz} is not a half integer, set sz to {sz_new}")
        if N%2 == 0:
            if int(sz_new * 2) % 2 != 0:
                sz_new = 0.0
                if rank == 0:
                    print(f"Warning: sz does not match the number of sites {N}, set sz to {sz_new}")
        elif N%2 == 1:
            if int(sz_new * 2) % 2 == 0:
                sz_new = 0.5
                if rank == 0:
                    print(f"Warning: sz does not match the number of sites {N}, set sz to {sz_new}")
        else:
            if rank == 0:
                raise ValueError(f"N is not an integer, N={N}!")
    
        return sz_new