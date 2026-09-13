# Multilevel NOSAS — Numerical Test Code

Python implementation to reproduce the numerical results of

> **Multilevel Nonoverlapping Spectral Additive Schwarz Methods (NOSAS)**
> Yi Yu

This repository contains the scripts that generated every table in the paper
(J. Sci. Comput.). There is exactly one script per paper table
(`run_table2.py` ... `run_table10.py`, matching Tables 2 ... 10 of the paper).

The optimized 2D implementations are thin drivers on top of the shared engines
`nosas2d_engine.py` (2D) and `nosas3d_engine.py` (3D); the BDDC half of
Table 6 uses the optimized `bddc_deluxe.py` (sparse Schur complements, sparse
coarse solve, parallel subdomain/entity stages), mathematically identical to
the paper's BDDC code.

## Table ↔ script mapping

| Paper table | Script |
|---|---|
| Table 2 (2D two-level)                | `run_table2.py`  |
| Table 3 (2D three-level, traditional) | `run_table3.py`  |
| Table 4 (2D three-level, filtering)   | `run_table4.py`  |
| Table 5 (2D four-level)               | `run_table5.py`  |
| Table 6 (3D two-level, NOSAS + BDDC deluxe) | `run_table6.py`  |
| Table 7 (3D three-level, filtering)   | `run_table7.py`  |
| Table 8 (3D four-level, filtering)    | `run_table8.py`  |
| Table 9 (3D four-level, traditional)  | `run_table9.py`  |
| Table 10 (3D three-level, hybrid)     | `run_table10.py` |

## Data

The 3D tests use the **SPE10** permeability model (`spe_perm.dat`, from
[Christie & Blunt 2001](https://www.spe.org/web/csp/datasets/set02.htm)).
The file must sit in the same directory as the scripts.

## Dependencies

- Python 3.10+
- `numpy`, `scipy`
- `dolfinx`, `ufl`, `petsc4py`, `slepc4py`
- `pymetis`
- `mpi4py`

## Usage

```bash
# 2D tables (optional arg: grid size 1/h, paper default if omitted)
python3 run_table2.py [1024]
python3 run_table3.py [1024]
python3 run_table4.py [1024]
python3 run_table5.py [2048]

# 3D tables (need spe_perm.dat in CWD; h is the coarsening level)
python3 run_table6.py h=1     # full Table 6: NOSAS half + BDDC deluxe half
run_table6.py runs its rows on multiple cores; the BDDC half uses the same
N1=128 subdomains as the NOSAS half (the paper's choice).
python3 run_table7.py         # full Table 7 sweep (3-level filtering)
# Tables 8-10 are single-scenario scripts (as used to generate the paper data);
# adjust their parameters at the top for each row of the table.
```

Each script prints `RESULT:` lines (tab-aligned columns matching the paper
tables) for easy copy-paste. Grid preprocessing is cached in
`preproc2d_cache.pkl` / `preproc_cache.pkl` (regenerated automatically).

The 2D tables (`run_table2.py` ... `run_table5.py`) run their independent rows
in parallel subprocesses for throughput. The number of concurrent rows is the
`PARALLEL` environment variable (default 4, `0` forces a serial sweep); keep
the 2048-grid `run_table5.py` rows at 1-2 since each needs ~20-40 GB of RAM.

The per-row level-0 subdomain loop is itself parallelised inside the engine
(`NOSAS2D_L0_PARALLEL`, number of worker processes). This only kicks in for
`nx >= 512` rows that do not need the level-0 solvers again (three-level and
filter/hybrid modes; classic two-level `run_table2.py` rows stay serial).
The row scripts automatically budget the two levels of parallelism
(`8 // PARALLEL` level-0 workers per row) so `python3 run_table4.py 1024` just
works; override with `NOSAS2D_L0_PARALLEL=N` if needed. The first run of a
fresh grid size always uses the serial path so the cache can be built once.

## License

MIT — see [LICENSE](LICENSE).