#!/usr/bin/env python3
"""Four-level NOSAS on a 2048 x 2048 grid, N = [256, 32, 4], eta_1 = 0.5
(N1=256, N2=32, N3=4) -- paper Table 5.

  (a) traditional approach (Section 3 at every level)
  (b) filtering approach   (Section 3 level-0 + Section 4 filtering)
  (c) hybrid approach      (Cref{alg:hybrid})

Code variables: ``mu`` is the level-0 threshold underlined eta_1,
``mu_list`` the per-level thresholds and ``eta_list`` the filtering
thresholds underlined mu, as in the paper.

Usage:
    python3 run_table5.py [1/h]        (default: 2048)

Rows are independent and run in parallel subprocesses. Set ``PARALLEL`` to the
number of concurrent rows (default 1) or to ``0`` for a serial sweep.  NB the
2048 rows are memory-hungry (~20-40 GB peak each); keep ``PARALLEL`` at 1-2.

Printed ``RESULT:`` lines end with ``tag #E Iters kappa lmax lmin``,
where tag is ``T eta_2 eta_3``, ``F mu_1 mu_2`` or ``H mu eta_2 eta_3``.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

NX_DEFAULT = 2048
N_LIST = [256, 32, 4]
ETA1 = 0.5
THRESH_VALS = [0.3, 0.5, 0.7]


def row_specs():
    specs = []
    # (a) traditional: eta_2, eta_3 in {0.3, 0.5, 0.7}
    for eta2 in THRESH_VALS:
        for eta3 in THRESH_VALS:
            specs.append(([ETA1, eta2, eta3], [], 'trad',
                          f'T {eta2} {eta3}'))
    # (b) filtering: mu_1, mu_2 in {0.3, 0.5, 0.7}
    for mu1 in THRESH_VALS:
        for mu2 in THRESH_VALS:
            specs.append(([ETA1], [mu1, mu2], 'filter',
                          f'F {mu1} {mu2}'))
    # (c) hybrid: (mu; eta_2, eta_3)
    for mu, eta2, eta3 in [(0.3, 0.3, 0.3), (0.3, 0.5, 0.5),
                           (0.5, 0.3, 0.3), (0.5, 0.5, 0.5),
                           (0.7, 0.3, 0.3), (0.7, 0.5, 0.5)]:
        specs.append(([ETA1, eta2, eta3], [mu], 'hybrid',
                      f'H {mu} {eta2} {eta3}'))
    return specs


def run_row(args):
    nx, mu_list, eta_list, mode, tag = args
    from nosas2d_engine import make_engine
    cfg = {'nx': nx, 'n_subdomains_list': list(N_LIST),
           'mu_list': list(mu_list), 'eta_list': list(eta_list),
           'mode': mode, 'deg': 1, 'cache_file': 'preproc2d_cache.pkl'}
    eng = make_engine(cfg)
    res = eng.run()
    n_eigen = res['flag_sums'][-1]
    return (f"RESULT: {nx}x{nx} {N_LIST} {tag} {n_eigen} {res['iters']} "
            f"{res['cond']:.6f} {res['lmax']:.6f} {res['lmin']:.6f}")


def main():
    nx = int(sys.argv[1]) if len(sys.argv) > 1 else NX_DEFAULT
    rows = [(nx, mu_list, eta_list, mode, tag)
            for mu_list, eta_list, mode, tag in row_specs()]
    npar = int(os.environ.get('PARALLEL', '1'))
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