#!/usr/bin/env python3
"""Three-level multilevel NOSAS, traditional approach (Section 3) -- paper Table 3.

Reproduces the whole table for a given grid size 1/h. The code variables
``mu_list`` are the level-dependent thresholds underlined eta_l in the paper.

Usage:
    python3 run_table3.py [1/h]        (default: 1024)

Rows are independent and run in parallel subprocesses. Set ``PARALLEL`` to the
number of concurrent rows (default 4) or to ``0`` for a serial sweep.

Printed ``RESULT:`` lines:
``1/h  N1  N2  eta_1  eta_2  #E  Iters  kappa  lmax  lmin``.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

NX_DEFAULT = 1024
N1_VALS = [64, 128]
N2_VALS = [4, 8]
ETA_VALS = [0.3, 0.5]


def run_row(args):
    nx, n1, n2, eta1, eta2 = args
    from nosas2d_engine import make_engine
    cfg = {'nx': nx, 'n_subdomains_list': [n1, n2],
           'mu_list': [eta1, eta2], 'eta_list': [],
           'mode': 'trad', 'deg': 1,
           'cache_file': 'preproc2d_cache.pkl'}
    eng = make_engine(cfg)
    res = eng.run()
    n_eigen = res['flag_sums'][-1]
    return (f"RESULT: {nx}x{nx} {n1} {n2} {eta1} {eta2} {n_eigen} "
            f"{res['iters']} {res['cond']:.6f} {res['lmax']:.6f} "
            f"{res['lmin']:.6f}")


def main():
    nx = int(sys.argv[1]) if len(sys.argv) > 1 else NX_DEFAULT
    rows = [(nx, n1, n2, eta1, eta2)
            for n2 in N2_VALS for n1 in N1_VALS
            for eta1 in ETA_VALS for eta2 in ETA_VALS]
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