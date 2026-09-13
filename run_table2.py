#!/usr/bin/env python3
"""Two-level NOSAS on the unit square (Section 3) -- paper Table 2.

Reproduces the whole table for a given grid size 1/h. The code variable
``mu`` (level-0 threshold) is the parameter underlined eta_1 in the paper.

Usage:
    python3 run_table2.py [1/h]        (default: 1024)

Rows are independent and run in parallel subprocesses. Set ``PARALLEL`` to the
number of concurrent rows (default 4) or to ``0`` for a serial sweep.

Printed ``RESULT:`` lines: ``1/h  N1  eta_1  #E  Iters  kappa  lmax  lmin``.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

ETA1_VALS = [0.3, 0.5]
N1_VALS = [64, 128]


def run_row(args):
    nx, n1, eta = args
    from nosas2d_engine import make_engine
    cfg = {'nx': nx, 'n_subdomains_list': [n1], 'mu_list': [eta],
           'eta_list': [], 'mode': 'trad', 'deg': 1,
           'cache_file': 'preproc2d_cache.pkl'}
    eng = make_engine(cfg)
    res = eng.run()
    n_eigen = res['flag_sums'][-1]
    return (f"RESULT: {nx}x{nx} {n1} {eta} {n_eigen} {res['iters']} "
            f"{res['cond']:.6f} {res['lmax']:.6f} {res['lmin']:.6f}")


def main():
    nx = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    rows = [(nx, n1, eta) for n1 in N1_VALS for eta in ETA1_VALS]
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