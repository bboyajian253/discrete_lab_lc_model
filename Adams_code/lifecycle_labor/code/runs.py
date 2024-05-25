# lifecycle_labor: runs
# contains pre-set run specifications (eg, solve or no? calibrate or no?)

import toolbox
import analyze_plotlc
import solve
import simulate
import calibrate
import numpy as np
import time


def output(state_solns, simlc, r, w, mD, par, par2):
    """
    Output solution/simluation results
    """

    # Print parameters
    calibrate.print_params(r, w, par, par2)

    # analyze parametrized model
    # analyze.output_aggs(simlc, wgt, par, path=par2.path)
    analyze_plotlc.plot_lcprofiles(simlc, par, par2)

def run_model(r, w, mD, par, par2, shock, solve_=True, calibrate_=True, no_calibrate_but_sim_=False, output_=True):
    """
    Given all model parameters (except population weights),
    (i)   Solve model and simulate L-C outcomes,
    (ii)  Calibrate population weights given L-C outcomes, [REMOVED]
    (iii) Compute and output implied model aggregates.
    """
    path = par2.path

    # solve life-cycle outcomes
    if solve_:

        t0 = time.time()
        state_solns = solve.solve_lc(r, w, par, par2)
        for lbl, v in state_solns.items():
            np.save(path + lbl + '_lc', v)
        t1 = time.time()
        print(t1 - t0)

    # load state-specific solutions
    lbls = ['c', 'n', 'kk']
    state_solns = {}
    for lbl in lbls:
        state_solns[lbl] = np.load(path + lbl + '_lc.npy')

    # simulate without re-calibrating model
    if no_calibrate_but_sim_:

        simlc = simulate.sim_lc(state_solns, r, w, par, par2, shock)
        for lbl in simlc.keys():
            np.save(path + f'sim{lbl}.npy', simlc[lbl])

    # load simulated life-cycles
    lbls = ['e_tru', 'eph_tru', 'n_tru', 'k', 'h', 'n', 'c', 'heff',
            'y', 'tax', 'taxe', 'taxc', 'e', 'eph']
    simlc = {}
    for lbl in lbls:
        simlc[lbl] = np.load(path + f'sim{lbl}.npy')

    # output parameters and aggregate statistics
    if output_:
        output(state_solns, simlc, r, w, mD, par, par2)


    return [state_solns, simlc]
