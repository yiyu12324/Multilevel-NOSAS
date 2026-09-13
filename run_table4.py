#!/usr/bin/env python3
"""Three-level NOSAS with filtering coarse level (Section 4, N2 = 1) -- paper Table 4.

Reproduces the whole table for a given grid size 1/h. The code variable
``mu`` is the level-0 threshold underlined eta_1, and ``eta`` the filtering
threshold underlined mu, both as in the paper (Section 4).

Usage:
    python3 run_table4.py [1/h]        (default: 1024)

Rows are independent and run in parallel subprocesses. Set ``PARALLEL`` to the
number of concurrent rows (default 4) or to ``0`` for a serial sweep.

Printed ``RESULT:`` lines:
``1/h  N1  eta_1  mu  #E  Iters  kappa  lmax  lmin``.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

NX_DEFAULT = 1024
N1_VALS = [64, 128]
ETA1_VALS = [0.3, 0.5]
MU_VALS = [0.1, 0.3, 0.5, 0.7, 0.9]


def run_row(args):
    nx, n1, eta1, mu = args
    from nosas2d_engine import make_engine
    cfg = {'nx': nx, 'n_subdomains_list': [n1, 1],
           'mu_list': [eta1], 'eta_list': [mu],
           'mode': 'filter', 'deg': 1,
           'cache_file': 'preproc2d_cache.pkl'}
    eng = make_engine(cfg)
    res = eng.run()
    n_eigen = res['flag_sums'][-1]
    return (f"RESULT: {nx}x{nx} {n1} {eta1} {mu} {n_eigen} "
            f"{res['iters']} {res['cond']:.6f} {res['lmax']:.6f} "
            f"{res['lmin']:.6f}")


def main():
    nx = int(sys.argv[1]) if len(sys.argv) > 1 else NX_DEFAULT
    rows = [(nx, n1, eta1, mu)
            for n1 in N1_VALS for eta1 in ETA1_VALS for mu in MU_VALS]
    npar = int(os.environ.get('PARALLEL', '4'))
    if 'NOSAS2D_L0_PARALLEL' not in os.environ:
        os.environ['NOSAS2D_L0_PARALLEL'] = str(max(1, 8 // npar))
    if npar <= 1 or len(rows) <= 1:
        for r in rows:
            print(run_row(r), flush=True)
        return
    if os.name == 'posix':
        import multiprocessing as mp
        try:
            mp.set_start_method('fork', force=True)
        except RuntimeError:
            pass
    with ProcessPoolExecutor(max_workers=min(npar, len(rows))) as ex:
        for line in ex.map(run_row, rows):
            print(line, flush=True)


if __name__ == '__main__':
    main()