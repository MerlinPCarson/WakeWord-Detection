'''
Module which generates FST to be used for
decoding keyword spotting output during
inference.

Written with assistance from Prof. Kain's
Automatic Speech Recognition WFST notebook.
'''
import numpy as np
import pynini as ni
from graphviz import render

def save_pdf(fst,path):
    fst.draw(path + '.dot', portrait=True)
    render('dot', 'pdf', path + '.dot', renderer='cairo')

def smooth(posterior_probs):
    # What are our thoughts on transitions?
    # This can be used to adjust costs to
    # increase likelihood to stay in current
    # state, or add cost to transition to
    # other state.
    test_transition_matrix = [[1, 1],
                              [1, 1]]

    # Set up state table.
    state_table = ni.SymbolTable()
    state_table.add_symbol('<ε>')
    state_table.add_symbol('other')
    state_table.add_symbol('wakeword')

    # Convert to log-probability costs.
    obs = -np.log(posterior_probs)

    P = 2           # Two states.
    T = len(obs)    # How many timepoints. This will be fixed
                    # when we integrate it into the streaming
                    # inference process.

    fst = ni.Fst()  # Creating FST.
    fst.set_input_symbols(state_table)    # Setting up input symbols.
    fst.set_output_symbols(state_table)   # Setting up output symbols.
    state = fst.add_state()                 # Adds a new state, and returns state ID.
    fst.set_start(state)                    # Setting start state using state ID.
    for t in range(T-1):
        for p in range(P):
            fst.add_state()
    for p in range(P):
        state = fst.add_state()
        fst.set_final(state)

    # fst.add_arc(state, arc)
    # "np.Arc: An FST arc is essentially a tuple of input label,
    #  output label, weight, and the next state ID."
    for p in range(P):      # Add initial arcs from start position.
        init = -np.log(1./P) + obs[0, p]
        fst.add_arc(0, ni.Arc(p+1, p+1, init, p+1))
    for t in range(1,T):    # Add arcs spanning each timepoint, each state.
        for p_from in range(P):
            for p_to in range(P):
                cost = obs[t, p_to]
                if p_to == p_from:  # If we are already in the same state,
                                    # should be more likely to stay in state.
                    cost -= test_transition_matrix[p_to][p_from]
                fst.add_arc((t-1) * P + p_from + 1, ni.Arc(p_to + 1, p_to + 1, cost, t * P + p_to + 1))
    #save_pdf(fst, "fst")

    # Find best path to ensure smoothed posterior for superframe.
    smoothed = ni.shortestpath(fst, nshortest=1, unique=True)
    print('smoothed superframe posterior: ', smoothed.stringify(token_type=state_table))
    return smoothed


if __name__ == "__main__":
    # Tests:
    test_posterior1 = [[0.8, 0.2],
                       [0.9, 0.1],
                       [0.5, 0.5],
                       [0.4, 0.6],
                       [0.2, 0.8],
                       [0.6, 0.4],  # Test whether we leave wakeword state.
                       [0.3, 0.7],
                       [0.4, 0.6],
                       [0.5, 0.5],
                       [0.9, 0.1]]

    test_posterior2 = [[0.8, 0.2],
                       [0.9, 0.1],
                       [0.5, 0.5],
                       [0.55, 0.45],
                       [0.2, 0.8],  # Test whether we enter errant wakeword state.
                       [0.6, 0.4],
                       [0.7, 0.3],
                       [0.8, 0.2],
                       [0.3, 0.7],
                       [0.9, 0.1]]

    smooth(test_posterior1)
    smooth(test_posterior2)