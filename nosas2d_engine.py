"""Shared optimized 2D NOSAS engine (unit square, P1, Poisson, rho = 1).

Reproduces the reference numerical behavior of, and is structurally modelled on
the 3D engine (``nosas3d_engine.py``):

  run_table2.py   (two-level Section 3, recursion)        -> mode='trad', 2 lvl
  run_table3.py   (three-level traditional Section 3)     -> mode='trad', 3 lvl
  run_table4a/b   (three-level Section 4 filtering, flat) -> mode='filter'
  run_table5a     (traditional 4-level)                   -> mode='trad', 4 lvl
  run_table5b     (filtering 4-level)                     -> mode='filter'
  run_table5c     (hybrid multilevel)                     -> mode='hybrid'

The engine keeps the PROVEN optimizations of the 3D engine:

  - numba kernels ``_add_block`` / ``_csr_rows_extract`` / ``_csr_block_gather``
    (identical implementations) used for the block-assembly of the global
    matrix ``Global_Solver = D_inv - A_QG^T InvS_Blk A_QG`` and for
    ``W = InvS_Blk @ A_QG`` over the entity blocks that tile ``GG``.
  - ``inv_ready``: per-entity block inverses are computed once and reused
    (bit-identical: the entity blocks tile ``GG`` with no overlap) across
    levels / filter iterations, avoiding repeated ``np.linalg.inv`` calls.
  - P4 partial eigenvalue solve: dense ``eigh`` GEP ``S vs A0`` where
    ``A0 = S * TS_Blk`` with subset-by-index selection gated by the env var
    ``NOSAS_P4_SYGVX`` (default enabled).  Only eigenvalues below ``mu`` are
    selected (retry grows the window until ``D[-1] > mu`` so the selection is
    complete); falls back to full ``eigh`` on failure.
  - Preprocessing cache (pickle) keyed by grid size & partition so the mesh /
    partition / entity extraction is computed once and reused.

Memory / solver notes (2D engine):

  - The level-0 R_0T = -A4_inv_A3 is dead data for filter / hybrid /
    multi-level traditional runs and is only consumed by the 2-level
    traditional path (``_two_level_assembly``).  ``compute_level0`` skips it
    everywhere else.
  - For 2-level runs on very large grids (> 2M total DOFs, e.g. nx=2048) the
    concatenated int-64 R0T triple (~1.55 GB/subdomain) is replaced by dense
    per-subdomain blocks ``{rows, cols, D=+A4_inv_A3}`` (the R_0T sign is
    applied inside the matvec kernels) and ``A4_inv_A3`` is destroyed right
    after the Schur complement is formed (the reference also destroys it).
    ``NOSAS2D_FORCE_R0T_BLOCKS=1`` forces the block path at any size.
  - The final coarse Global_Solver factorisation defaults to the reference
    PREONLY + CHOLESKY; ``NOSAS2D_FINAL_SPLU=1`` (or an automatic switch for
    coarse solves with > 2M rows) uses a one-shot scipy SuperLU factorisation
    instead.

Naming (same swap as the 3D code / paper):
  ``mu`` / ``mu_list`` -> paper eta (Section 3 eigenvalue threshold)
  ``eta`` / ``eta_list`` -> paper mu (Section 4 filtering threshold)
"""
import os
import gc
import sys
import time
import pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from mpi4py import MPI
from dolfinx import mesh, fem
import pymetis
from petsc4py import PETSc
import ufl
from scipy.sparse import csr_matrix
from scipy.linalg import eigh, cholesky, solve_triangular
from scipy.sparse.csgraph import connected_components
from numba import njit

# --------------------------------------------------------------------------
# sparse B_Gamma heuristic for Section-4 filter (avoids 4.5 GB+ dense
# Cholesky on the whole-interface operator when physical RAM is tight)
# --------------------------------------------------------------------------

def _use_sparse_filter_bgamma(n):
    """Return True when the whole-interface dense B_Gamma would be too
    large.  ``NOSAS2D_FILTER_SPARSE=1`` forces sparse; otherwise the
    threshold is n² × 8 bytes > 3 GB  (≈ 19 365 DOFs)."""
    if os.environ.get('NOSAS2D_FILTER_SPARSE', '0') == '1':
        return True
    return n * n * 8.0 > 3.0e9          # ~19 365 DOFs


# --------------------------------------------------------------------------
# parallel level-0 subdomain loop (multiprocessing over subdomains)
#
# The per-subdomain work (submesh, local assembly, A4 solves, eigenproblem)
# is independent, so forked worker processes compute disjoint subdomains and
# return plain numpy/scipy results.  Only used when the level-0 A4 KSPs are
# dead downstream (filter / hybrid / multi-level traditional) -- the 2-level
# traditional PCG still needs them and keeps the original serial path.
# The mesh is shared out-of-band through the module global (fork inherits it,
# so the pickled task argument is only the subdomain index).
# --------------------------------------------------------------------------

_LEVEL0_WORKER_D = None   # light payload (membership / gamma / interior / TS / mu)
_LEVEL0_WORKER_MESH = None   # parent mesh, inherited copy-on-write via fork
_LEVEL0_WORKER_DOFMAP = None


def _level0_sygvx(Schur_np, A1_mat, mu_val, len_m):
    use_sygvx = (os.environ.get('NOSAS_P4_SYGVX', '1') == '1')
    if use_sygvx:
        k_p4 = min(len_m, 150)
        _retry = 0
        while True:
            try:
                D, Q = eigh(Schur_np, A1_mat, subset_by_index=(0, k_p4 - 1))
            except np.linalg.LinAlgError:
                D, Q = eigh(Schur_np, A1_mat)
                break
            if D[-1] > mu_val or k_p4 >= len_m:
                break
            k_p4 = min(len_m, 2 * k_p4)
            _retry += 1
        if _retry:
            print(f"  [P4] sygvx retry -> k={k_p4} ({_retry}x)", flush=True)
        return D, Q
    return eigh(Schur_np, A1_mat)


def _level0_petsc_mat_from_csr(M):
    """Worker mirror of NOSAS2DEngine._petsc_mat_from_csr."""
    if M.shape[0] == 0 and M.shape[1] == 0:
        return PETSc.Mat().createAIJ(size=(0, 0),
                                     csr=(np.array([0], dtype=PETSc.IntType),
                                          np.array([], dtype=PETSc.IntType),
                                          np.array([], dtype=PETSc.ScalarType)))
    M = M.tocsr()
    return PETSc.Mat().createAIJ(
        size=(M.shape[0], M.shape[1]),
        csr=(M.indptr.astype(PETSc.IntType),
             M.indices.astype(PETSc.IntType),
             M.data.astype(PETSc.ScalarType)))


def _level0_eigenproblem_standalone(Schur_np, ts_csr, m, mu_val):
    """Worker mirror of NOSAS2DEngine._compute_eigenproblem_from_schur."""
    len_m = len(m)
    is_m = PETSc.IS().createGeneral(m.astype(PETSc.IntType), comm=PETSc.COMM_WORLD)
    _ts_mat_use = _level0_petsc_mat_from_csr(ts_csr)
    TS_Blk_mm = _ts_mat_use.createSubMatrix(is_m, is_m)
    TS_Blk_mm.assemble()
    _ts_mat_use.destroy()
    TS_Blk_mm_np = NOSAS2DEngine._petsc_to_scipy_dense(TS_Blk_mm)
    A1_mat = Schur_np * TS_Blk_mm_np
    D, Q = _level0_sygvx(Schur_np, A1_mat, mu_val, len_m)
    A1_mat_csr = csr_matrix(A1_mat)
    indptr, indices, values = A1_mat_csr.indptr, A1_mat_csr.indices, A1_mat_csr.data
    row_indices_local = np.repeat(np.arange(len(np.diff(indptr))), np.diff(indptr))
    S_Blk_rows = m[row_indices_local]
    S_Blk_cols = m[indices]
    S_Blk_vals = values
    eigens_idx = np.where(D < mu_val)[0]
    Q = Q[:, eigens_idx]
    D = 1 - D[eigens_idx]
    flag = len(eigens_idx)
    if len(eigens_idx) == 0:
        A_QG_data = {'m': m, 'A1Q': np.array([]).reshape(len_m, 0), 'n_eigen': 0}
        D_inv = np.array([])
    else:
        A1Q = A1_mat @ Q
        A_QG_data = {'m': m, 'A1Q': A1Q, 'n_eigen': len(eigens_idx)}
        D_inv = 1.0 / D
    is_m.destroy()
    TS_Blk_mm.destroy()
    return S_Blk_rows, S_Blk_cols, S_Blk_vals, A_QG_data, D_inv, flag


def _level0_sub_inner(msh, dofmap, membership, m, n, ts_csr, mu_val, deg, sub_id):
    f = PETSc.ScalarType(1.0)
    _t0 = time.time()
    sub_tri_ids = np.where(membership == sub_id)[0]

    submesh, entity_map, vertex_map, geometry_map = mesh.create_submesh(
        msh, msh.topology.dim, sub_tri_ids)
    V_sub = fem.functionspace(submesh, ("Lagrange", deg))
    dofmap_sub = V_sub.dofmap
    num_sub_cells = submesh.topology.index_map(2).size_local
    sub_cell_indices = np.arange(num_sub_cells, dtype=np.int32)
    cell_map = entity_map.sub_topology_to_topology(sub_cell_indices, inverse=False)
    sub_dofs_flat = dofmap_sub.list.flatten()
    orig_dofs_flat = dofmap.list[cell_map].flatten()
    sort_idx = np.argsort(sub_dofs_flat)
    sorted_sub = sub_dofs_flat[sort_idx]
    sorted_orig = orig_dofs_flat[sort_idx]
    _, unique_idx = np.unique(sorted_sub, return_index=True)
    all_sub_dofs = sorted_orig[unique_idx]
    sorted_idx = np.argsort(all_sub_dofs)
    sorted_global = all_sub_dofs[sorted_idx]
    all_target = np.concatenate([m, n])
    indices = np.searchsorted(sorted_global, all_target)
    all_local = sorted_idx[indices]
    m_local, n_local = all_local[:len(m)], all_local[len(m):]

    V_local = fem.functionspace(submesh, ("Lagrange", deg))
    u, v = ufl.TrialFunction(V_local), ufl.TestFunction(V_local)
    a_local = fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    L_local = fem.form(ufl.inner(f, v) * ufl.dx)
    A_local = fem.assemble_matrix(a_local)
    b_local = fem.assemble_vector(L_local)
    A_local = PETSc.Mat().createAIJ(
        size=(A_local.index_map(0).size_local, A_local.index_map(1).size_local),
        csr=(A_local.indptr.astype(PETSc.IntType),
             A_local.indices.astype(PETSc.IntType),
             A_local.data.astype(PETSc.ScalarType)))
    is_m_local = PETSc.IS().createGeneral(m_local.astype(PETSc.IntType), comm=PETSc.COMM_WORLD)
    is_n_local = PETSc.IS().createGeneral(n_local.astype(PETSc.IntType), comm=PETSc.COMM_WORLD)
    A1 = A_local.createSubMatrix(is_m_local, is_m_local); A1.assemble()
    A2 = A_local.createSubMatrix(is_m_local, is_n_local); A2.assemble()
    A3 = A_local.createSubMatrix(is_n_local, is_m_local); A3.assemble()
    A4 = A_local.createSubMatrix(is_n_local, is_n_local); A4.assemble()
    ksp_A4 = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
    ksp_A4.setOperators(A4)
    ksp_A4.setType(PETSc.KSP.Type.PREONLY)
    ksp_A4.getPC().setType(PETSc.PC.Type.CHOLESKY)
    ksp_A4.setUp()

    b_vec = PETSc.Vec().createWithArray(b_local.array)
    b_n = b_vec.getSubVector(is_n_local)
    ksp_A4.solve(b_n, b_n)
    g_vec = b_vec.getSubVector(is_m_local) - A2 * b_n
    g_rows = all_sub_dofs[is_m_local]
    g_vals = g_vec.getArray()
    len_m, len_n = len(m_local), len(n_local)

    solutions = np.zeros((len_n, len_m), dtype=PETSc.ScalarType)
    x_temp = PETSc.Vec().createSeq(len_n)
    for j in range(len_m):
        values = np.zeros(len_n, dtype=PETSc.ScalarType)
        A3.getValues(np.arange(len_n, dtype=PETSc.IntType), np.array([j], dtype=PETSc.IntType), values)
        b = PETSc.Vec().createWithArray(values, comm=PETSc.COMM_WORLD)
        ksp_A4.solve(b, x_temp)
        solutions[:, j] = x_temp.getArray()
        b.destroy()
    x_temp.destroy()
    A4_inv_A3_data = solutions.ravel()
    _A4_csr = csr_matrix((A4_inv_A3_data,
                          (np.repeat(np.arange(len_n), len_m),
                           np.tile(np.arange(len_m), len_n))))
    A4_inv_A3 = PETSc.Mat().createAIJ(
        size=(len_n, len_m),
        csr=(_A4_csr.indptr.astype(PETSc.IntType),
             _A4_csr.indices.astype(PETSc.IntType),
             A4_inv_A3_data.astype(PETSc.ScalarType)))
    A4_inv_A3.assemble()

    Schur = A1 - A2 * A4_inv_A3
    is_m = PETSc.IS().createGeneral(m.astype(PETSc.IntType), comm=PETSc.COMM_WORLD)
    Schur_np = NOSAS2DEngine._petsc_to_scipy_dense(Schur)
    Schur_np = (Schur_np + Schur_np.T) / 2

    S_Blk_rows, S_Blk_cols, S_Blk_vals, A_QG_data, D_inv, flag = \
        _level0_eigenproblem_standalone(Schur_np, ts_csr, m, mu_val)

    Schur_csr = csr_matrix(Schur_np)
    indptr, ids, values = Schur_csr.indptr, Schur_csr.indices, Schur_csr.data
    row_indices_local = np.repeat(np.arange(len(np.diff(indptr))), np.diff(indptr))
    S_rows = m[row_indices_local]
    S_cols = m[ids]
    S_vals = values

    is_m_local.destroy(); is_n_local.destroy(); is_m.destroy()
    b_vec.destroy(); b_n.destroy(); g_vec.destroy()
    A_local.destroy(); A1.destroy(); A2.destroy(); A3.destroy(); A4.destroy()
    A4_inv_A3.destroy(); ksp_A4.destroy()
    gc.collect()
    print(f"[T] L0 sub{sub_id}: (worker, wall {time.time() - _t0:.2f}s) Gamma {len_m}, "
          f"Int {len_n}, flag {flag}", flush=True)
    return {
        'sub_id': int(sub_id),
        'S_Blk_rows': S_Blk_rows.copy(), 'S_Blk_cols': S_Blk_cols.copy(),
        'S_Blk_vals': S_Blk_vals.copy(), 'A_QG_data': A_QG_data,
        'D_inv': D_inv.copy(), 'flag1': int(flag),
        'S_rows': S_rows.copy(), 'S_cols': S_cols.copy(), 'S_vals': S_vals.copy(),
        'g_rows': g_rows.copy(), 'g_vals': g_vals.copy(),
    }


def _level0_worker(sub_ids):
    """Process a chunk of subdomains in a forked worker.

    The mesh and its dofmap are inherited from the parent (copy-on-write via
    fork), so no mesh rebuild is needed; only per-subdomain objects are
    created here."""
    d = _LEVEL0_WORKER_D
    msh = _LEVEL0_WORKER_MESH
    dofmap = _LEVEL0_WORKER_DOFMAP
    membership = d['membership']
    out = []
    for sub_id in sub_ids:
        m = d['gamma'][sub_id]
        n = d['interior'][sub_id]
        out.append(_level0_sub_inner(msh, dofmap, membership, m, n,
                                     d['TS_Blk_csr'], d['mu_level0'], d['deg'], sub_id))
    return out


# --------------------------------------------------------------------------
# numba kernels (byte-identical to nosas3d_engine.py)
# --------------------------------------------------------------------------

@njit(cache=True, nogil=True)
def _add_block(Goff, nz, contrib):
    m = nz.shape[0]
    for i in range(m):
        ni = nz[i]
        c_row = contrib[i]
        for j in range(m):
            Goff[ni, nz[j]] += c_row[j]


@njit(cache=True, nogil=True)
def _csr_rows_extract(indptr, indices, data, rows):
    n = rows.shape[0]
    cnt = 0
    for r in range(n):
        cnt += indptr[rows[r] + 1] - indptr[rows[r]]
    cols_all = np.empty(cnt, dtype=np.int64)
    vals_all = np.empty(cnt, dtype=np.float64)
    row_of = np.empty(cnt, dtype=np.int64)
    c = 0
    for r in range(n):
        i0, i1 = indptr[rows[r]], indptr[rows[r] + 1]
        for p in range(i0, i1):
            cols_all[c] = indices[p]
            vals_all[c] = data[p]
            row_of[c] = r
            c += 1
    ucols = np.unique(cols_all)
    u = ucols.shape[0]
    Adense = np.zeros((n, u))
    for p in range(cnt):
        col = cols_all[p]
        lo, hi = 0, u - 1
        while lo <= hi:
            mid = (lo + hi) >> 1
            if ucols[mid] < col:
                lo = mid + 1
            elif ucols[mid] > col:
                hi = mid - 1
            else:
                Adense[row_of[p], mid] += vals_all[p]
                break
    return ucols, Adense


@njit(cache=True, nogil=True)
def _csr_block_gather(indptr, indices, data, loc, pos):
    n = loc.shape[0]
    for j in range(n):
        pos[loc[j]] = j
    S = np.zeros((n, n))
    for i in range(n):
        r = loc[i]
        for p in range(indptr[r], indptr[r + 1]):
            j = pos[indices[p]]
            if j >= 0:
                S[i, j] += data[p]
    for j in range(n):
        pos[loc[j]] = -1
    return S


# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------


class _SpluSolver:
    """scipy.superlu-backed solver with a PETSc KSP-like ``solve(b, x)``
    interface, selected for the final coarse Global_Solver by the env var
    ``NOSAS2D_FINAL_SPLU=1`` (or automatically when the coarse solve has more
    than 2M rows).  Factored once, reused for every PCG iteration."""

    def __init__(self, mat):
        from scipy.sparse.linalg import splu
        self.n = mat.getSize()[0]
        self.lu = None
        if self.n == 0:
            print("[SPLU] coarse solver is 0x0 -- skipping factorisation")
            return
        I, J, V = mat.getValuesCSR()
        Acsc = csr_matrix((V, J, I), shape=(self.n, self.n)).tocsc()
        _t0 = time.time()
        self.lu = splu(Acsc, permc_spec='COLAMD')
        print(f"[SPLU] final coarse Global_Solver({self.n} x {self.n}) "
              f"factorised in {time.time() - _t0:.2f}s")

    def solve(self, b, x):
        if self.lu is None or self.n == 0:
            x.set(0.0)
            return
        x.setArray(self.lu.solve(b.getArray()))

    def destroy(self):
        pass


class NOSAS2DEngine:

    def __init__(self, cfg):
        self.cfg = cfg
        self.start_time = time.time()
        self.n_dofs_total = None
        self.dofs_bc = None
        self.msh = None
        self.dofmap = None
        self.n_subdomains = 0
        self._cache_used = False   # level-0 parallel uses the cache-restored mesh
        # per-subdomain level-0 data (mu independent + per-mu eigen data)
        self.results = []
        self.local_solvers = None        # level-0 A4 KSPs
        self.gg = None
        self.S_csr_GG = None             # global system matrix on GG (CSR)
        self.g_arr = None                # global RHS on GG
        self.R0T_level0 = None           # (rows, cols, data) of -A4_inv_A3 (triple path)
        self.R0T_blocks = None           # list of {'rows','cols','D'} per-sub dense -A4_inv_A3 (block path)
        self.inv_cache = {}              # frozenset(dofs) -> (dofs, inverse)
        self._destroyed = []

    # ------------------------------------------------------------------
    # small helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _petsc_to_scipy_dense(petsc_mat):
        I, J, V = petsc_mat.getValuesCSR()
        return csr_matrix((V, J, I), shape=petsc_mat.getSize()).toarray()

    @staticmethod
    def _petsc_to_scipy_csr(petsc_mat):
        I, J, V = petsc_mat.getValuesCSR()
        return csr_matrix((V, J, I), shape=petsc_mat.getSize())

    def _petsc_mat_from_csr(self, M, dtype=None):
        if M.shape[0] == 0 and M.shape[1] == 0:
            return PETSc.Mat().createAIJ(size=(0, 0),
                                         csr=(np.array([0], dtype=PETSc.IntType),
                                              np.array([], dtype=PETSc.IntType),
                                              np.array([], dtype=PETSc.ScalarType)))
        M = M.tocsr()
        return PETSc.Mat().createAIJ(
            size=(M.shape[0], M.shape[1]),
            csr=(M.indptr.astype(PETSc.IntType),
                 M.indices.astype(PETSc.IntType),
                 M.data.astype(PETSc.ScalarType)))

    # ------------------------------------------------------------------
    # preprocessing (mesh / bc / partition / entities / TS_Blk) -- cached
    # ------------------------------------------------------------------

    def preprocess(self):
        c = self.cfg
        nx = c['nx']
        n_subdomains = c['n_subdomains_list'][0]
        deg = c['deg']
        _cache_file = c.get('cache_file', 'preproc2d_cache.pkl')
        _key = (nx, tuple(c['n_subdomains_list']), deg)
        _pc = None
        if os.path.exists(_cache_file):
            try:
                with open(_cache_file, 'rb') as _f:
                    _pc = pickle.load(_f)
                if _pc.get('key') != _key:
                    _pc = None
            except Exception:
                _pc = None

        if _pc is not None:
            print(f"[cache] preprocess cache hit for {_key}")
            import basix, basix.ufl
            _x = np.asarray(_pc['x'])
            if _x.shape[1] != 2:
                if not np.allclose(_x[:, 2], 0.0):
                    raise RuntimeError('cached geometry not planar z=0')
                _x = _x[:, :2]
            msh = mesh.create_mesh(MPI.COMM_SELF, _pc['cells'],
                                   basix.ufl.element("Lagrange", "triangle", deg, shape=(2,)),
                                   _x)
            self._restore_from_cache(msh, _pc)
            print(f"[cache] n_dofs_total {self.n_dofs_total}, GG {len(self.gg)}, "
                  f"Subd_E {len(self.Subd_E)}, Subd_V {len(self.Subd_V)}")
            return

        print(f"\n=== Creating unit-square mesh (nx={nx}) ===")
        msh = mesh.create_unit_square(MPI.COMM_SELF, nx, nx, mesh.CellType.triangle)
        print(f"Total cells: {msh.topology.index_map(2).size_global}, "
              f"DOFs: {msh.geometry.dofmap.shape[0] if hasattr(msh.geometry.dofmap,'shape') else len(msh.geometry.x)}")

        boundary_tags = {10: lambda x: np.isclose(x[1], 0.0), 11: lambda x: np.isclose(x[0], 1.0),
                         12: lambda x: np.isclose(x[1], 1.0), 13: lambda x: np.isclose(x[0], 0.0)}
        facet_indices, facet_values = [], []
        for tag, func in boundary_tags.items():
            facets = mesh.locate_entities_boundary(msh, 1, func)
            facet_indices.append(facets)
            facet_values.append(np.full(len(facets), tag, dtype=np.int32))
        facet_indices = np.hstack(facet_indices)
        facet_values = np.hstack(facet_values)
        facet_tags = mesh.meshtags(msh, 1, facet_indices, facet_values)

        V = fem.functionspace(msh, ("Lagrange", deg))
        dofmap = V.dofmap
        n_dofs_total = dofmap.index_map.size_global
        facets = facet_tags.indices[facet_tags.values != -1]
        dofs_bc = np.unique(fem.locate_dofs_topological(V, 1, facets).astype(np.int32))
        print(f"Total DOFs: {n_dofs_total}, Boundary DOFs: {len(dofs_bc)}")

        msh.topology.create_connectivity(2, 0)
        triangles = msh.topology.connectivity(2, 0).array.reshape(-1, 3)
        num_tris = len(triangles)
        msh.topology.create_connectivity(1, 2)
        edge_to_cells = msh.topology.connectivity(1, 2)
        num_edges = edge_to_cells.num_nodes

        adjacency = [[] for _ in range(num_tris)]
        for e in range(num_edges):
            cells = edge_to_cells.links(e)
            if len(cells) == 2:
                c1, c2 = cells
                adjacency[c1].append(c2)
                adjacency[c2].append(c1)
        num_cuts, group_assignment = pymetis.part_graph(n_subdomains, adjacency=adjacency)
        group_assignment = np.array(group_assignment)
        print(f"Metis partition: {num_cuts} cuts into {n_subdomains} domains")

        (GG, Subd_V, II, Subd_E, subdomain_all_dofs, Gamma, Interior,
         TS_Blk_csr) = self._extract_first_level_variables(
            group_assignment, n_subdomains, dofs_bc, dofmap, n_dofs_total)
        print(f"GG: {len(GG)}, Subd_V: {len(Subd_V)}, II: {len(II)}, Subd_E: {len(Subd_E)}")

        self.msh = msh
        self.dofmap = dofmap
        self.n_dofs_total = n_dofs_total
        self.dofs_bc = dofs_bc
        self.membership = group_assignment
        self.subdomain_all_dofs = subdomain_all_dofs
        self.gg = GG
        self.Subd_V = Subd_V
        self.II = II
        self.Subd_E = Subd_E
        self.Gamma = Gamma
        self.Interior = Interior
        self.TS_Blk_csr = TS_Blk_csr
        self.TS_Blk_mat = self._petsc_mat_from_csr(TS_Blk_csr)
        self.TS_Blk_mat.assemble()

        self._save_cache(_cache_file, _key, msh, group_assignment,
                         subdomain_all_dofs, GG, Subd_V, II, Subd_E,
                         Gamma, Interior, TS_Blk_csr)
        print(f"[cache] preprocess saved to {_cache_file}")
        print(f"[T] preprocess: {time.time() - self.start_time:.2f}s")

    @staticmethod
    def _save_cache(_cache_file, _key, msh, group_assignment,
                    subdomain_all_dofs, GG, Subd_V, II, Subd_E,
                    Gamma, Interior, TS_Blk_csr):
        """Write the preprocessing cache atomically.

        Concurrent row processes may build cache entries at the same time, so
        the dump is serialized with an advisory lock and written to a temp file
        that is renamed over the real one (readers never see a partial dump).
        """
        import fcntl
        data = {
            'key': _key,
            'x': msh.geometry.x.copy(),
            'cells': msh.topology.connectivity(2, 0).array.copy().reshape(-1, 3),
            'membership': group_assignment,
            'subdomain_all_dofs': subdomain_all_dofs,
            'GG': GG, 'Subd_V': Subd_V, 'II': II, 'Subd_E': Subd_E,
            'Gamma': Gamma, 'Interior': Interior,
            'TS_Blk_csr': TS_Blk_csr,
        }
        _tmp = _cache_file + '.tmp'
        _lock = _tmp + '.lock'
        try:
            with open(_lock, 'w') as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)
                with open(_tmp, 'wb') as _f:
                    pickle.dump(data, _f)
                os.replace(_tmp, _cache_file)
        finally:
            try:
                os.unlink(_lock)
            except OSError:
                pass

    def _restore_from_cache(self, msh, pc):
        V = fem.functionspace(msh, ("Lagrange", self.cfg['deg']))
        dofmap = V.dofmap
        self._cache_used = True
        self.msh = msh
        self.dofmap = dofmap
        self.n_dofs_total = dofmap.index_map.size_global
        self.membership = pc['membership']
        self.subdomain_all_dofs = pc['subdomain_all_dofs']
        self.gg = pc['GG']
        self.Subd_V = pc['Subd_V']
        self.II = pc['II']
        self.Subd_E = pc['Subd_E']
        self.Gamma = pc['Gamma']
        self.Interior = pc['Interior']
        self.TS_Blk_csr = pc['TS_Blk_csr']
        self.TS_Blk_mat = self._petsc_mat_from_csr(self.TS_Blk_csr)
        self.TS_Blk_mat.assemble()
        # dofs_bc recomputed from restored mesh
        boundary_tags = {10: lambda x: np.isclose(x[1], 0.0), 11: lambda x: np.isclose(x[0], 1.0),
                         12: lambda x: np.isclose(x[1], 1.0), 13: lambda x: np.isclose(x[0], 0.0)}
        facet_indices, facet_values = [], []
        for tag, func in boundary_tags.items():
            facets = mesh.locate_entities_boundary(msh, 1, func)
            facet_indices.append(facets)
            facet_values.append(np.full(len(facets), tag, dtype=np.int32))
        facet_indices = np.hstack(facet_indices)
        facet_values = np.hstack(facet_values)
        facet_tags = mesh.meshtags(msh, 1, facet_indices, facet_values)
        facets = facet_tags.indices[facet_tags.values != -1]
        self.dofs_bc = np.unique(fem.locate_dofs_topological(V, 1, facets).astype(np.int32))

    def _extract_first_level_variables(self, group_assignment, n_subdomains,
                                       dofs_bc, dofmap, n_dofs_total):
        subdomain_all_dofs = [None] * n_subdomains
        v = np.zeros(n_dofs_total, dtype=np.int32)
        v2 = np.zeros(n_dofs_total, dtype=np.int32)
        for sub_id in range(n_subdomains):
            sub_tri_ids = np.where(group_assignment == sub_id)[0]
            sub_dofs = np.unique(np.concatenate([dofmap.cell_dofs(cell_idx) for cell_idx in sub_tri_ids]))
            subdomain_all_dofs[sub_id] = sub_dofs
            v[sub_dofs] += 1
            v2[sub_dofs] += (sub_id + 1)
        GG = np.setdiff1d(np.where(v >= 2)[0], dofs_bc)
        Subd_V = np.setdiff1d(np.where(v >= 3)[0], dofs_bc)
        II = np.setdiff1d(np.where(v == 1)[0], dofs_bc)
        v2_temp = v2.copy()
        v2_temp[Subd_V] = 0
        v2_temp[dofs_bc] = 0
        v2_temp[II] = 0
        Subd_E_candidates = []
        for sub_id in range(n_subdomains):
            sub_dofs = subdomain_all_dofs[sub_id]
            non_zero_mask = v2_temp[sub_dofs] != 0
            if not non_zero_mask.any():
                continue
            edge_dofs = sub_dofs[non_zero_mask]
            edge_values = v2_temp[edge_dofs]
            unique_values, inverse_indices = np.unique(edge_values, return_inverse=True)
            for i, val in enumerate(unique_values):
                if val == 0:
                    continue
                edge = edge_dofs[inverse_indices == i]
                if len(edge) > 0:
                    Subd_E_candidates.append(np.sort(edge))
        Subd_E = []
        edge_counter = 0
        for new_edge in Subd_E_candidates:
            new_set = set(new_edge)
            if not new_set:
                continue
            v2_vals = v2_temp[np.array(list(new_set), dtype=np.int32)]
            has_neg = np.any(v2_vals < 0)
            if not has_neg:
                edge_counter += 1
                Subd_E.append(np.array(sorted(new_set), dtype=np.int32))
                v2_temp[np.array(list(new_set), dtype=np.int32)] = -edge_counter
                continue
            neg_vals = v2_vals[v2_vals < 0]
            if len(neg_vals) == len(v2_vals) and len(set(neg_vals)) == 1:
                same_eid = -neg_vals[0]
                existing = Subd_E[same_eid - 1]
                if existing is not None and len(existing) == len(new_edge) and \
                        np.array_equal(np.sort(existing), np.sort(new_edge)):
                    continue
            involved_eids = set()
            for d in new_set:
                if v2_temp[d] < 0:
                    involved_eids.add(-v2_temp[d])
            all_sets = [new_set]
            eid_to_idx = {}
            for eid in involved_eids:
                idx = eid - 1
                eid_to_idx[eid] = idx
                all_sets.append(set(Subd_E[idx]))
            for eid in involved_eids:
                Subd_E[eid_to_idx[eid]] = None
            dof_count = {}
            for s in all_sets:
                for d in s:
                    dof_count[d] = dof_count.get(d, 0) + 1
            added = set()
            for s in all_sets:
                non_conflict = tuple(sorted(d for d in s if dof_count[d] == 1))
                conflict = [d for d in s if dof_count[d] > 1]
                if non_conflict and non_conflict not in added:
                    edge_counter += 1
                    Subd_E.append(np.array(non_conflict, dtype=np.int32))
                    v2_temp[np.array(non_conflict, dtype=np.int32)] = -edge_counter
                    added.add(non_conflict)
                for d in conflict:
                    if (d,) not in added:
                        edge_counter += 1
                        Subd_E.append(np.array([d], dtype=np.int32))
                        v2_temp[d] = -edge_counter
                        added.add((d,))
        Subd_E = [e for e in Subd_E if e is not None]
        all_rows = np.concatenate([Subd_V] + [np.repeat(edge, len(edge)) for edge in Subd_E])
        all_cols = np.concatenate([Subd_V] + [np.tile(edge, len(edge)) for edge in Subd_E])
        all_vals = np.ones(len(all_rows), dtype=PETSc.ScalarType)
        TS_Blk_csr = csr_matrix((all_vals, (all_rows, all_cols)), shape=(n_dofs_total, n_dofs_total))
        Gamma = [None] * n_subdomains
        Interior = [None] * n_subdomains
        for sub_id in range(n_subdomains):
            sub_dofs = subdomain_all_dofs[sub_id]
            Interior[sub_id] = np.setdiff1d(np.setdiff1d(sub_dofs, dofs_bc), GG)
            Gamma[sub_id] = np.intersect1d(sub_dofs, GG)
        return (GG, Subd_V, II, Subd_E, subdomain_all_dofs, Gamma, Interior, TS_Blk_csr)

    # ------------------------------------------------------------------
    # Section-3 P4: generalized eigenproblem S vs A0 (= S * TS_Blk)
    # ------------------------------------------------------------------

    def _compute_eigenproblem_from_schur(self, Schur_np, TS_Blk_coarse_csr, m, mu_val):
        """Identical numerics to the reference ``compute_eigenproblem_from_schur``.

        Returns S_Blk rows/cols/vals (CSR of A1_mat = Schur*TS_Blk_mm),
        A_QG_data, D_inv, flag.
        """
        len_m = len(m)
        is_m = PETSc.IS().createGeneral(m.astype(PETSc.IntType), comm=PETSc.COMM_WORLD)
        _ts_use = TS_Blk_coarse_csr if TS_Blk_coarse_csr is not None else self.TS_Blk_csr
        _ts_mat_use = self._petsc_mat_from_csr(_ts_use)
        TS_Blk_mm = _ts_mat_use.createSubMatrix(is_m, is_m)
        TS_Blk_mm.assemble()
        _ts_mat_use.destroy()
        TS_Blk_mm_np = self._petsc_to_scipy_dense(TS_Blk_mm)
        A1_mat = Schur_np * TS_Blk_mm_np

        use_sygvx = (os.environ.get('NOSAS_P4_SYGVX', '1') == '1')
        if use_sygvx:
            k_p4 = min(len_m, 150)
            _retry = 0
            while True:
                try:
                    D, Q = eigh(Schur_np, A1_mat, subset_by_index=(0, k_p4 - 1))
                except np.linalg.LinAlgError:
                    D, Q = eigh(Schur_np, A1_mat)
                    break
                if D[-1] > mu_val or k_p4 >= len_m:
                    break
                k_p4 = min(len_m, 2 * k_p4)
                _retry += 1
            if _retry:
                print(f"  [P4] sygvx retry -> k={k_p4} ({_retry}x)", flush=True)
        else:
            D, Q = eigh(Schur_np, A1_mat)

        A1_mat_csr = csr_matrix(A1_mat)
        indptr, indices, values = A1_mat_csr.indptr, A1_mat_csr.indices, A1_mat_csr.data
        row_indices_local = np.repeat(np.arange(len(np.diff(indptr))), np.diff(indptr))
        S_Blk_rows = m[row_indices_local]
        S_Blk_cols = m[indices]
        S_Blk_vals = values
        eigens_idx = np.where(D < mu_val)[0]
        Q = Q[:, eigens_idx]
        D = 1 - D[eigens_idx]
        flag = len(eigens_idx)
        if len(eigens_idx) == 0:
            A_QG_data = {'m': m, 'A1Q': np.array([]).reshape(len_m, 0), 'n_eigen': 0}
            D_inv = np.array([])
        else:
            A1Q = A1_mat @ Q
            A_QG_data = {'m': m, 'A1Q': A1Q, 'n_eigen': len(eigens_idx)}
            D_inv = 1.0 / D
        is_m.destroy()
        TS_Blk_mm.destroy()
        return S_Blk_rows, S_Blk_cols, S_Blk_vals, A_QG_data, D_inv, flag

    # ------------------------------------------------------------------
    # Level-0 subdomain computation (raw data computed once; eigen per mu)
    # ------------------------------------------------------------------

    def compute_level0(self, mu_level0):
        c = self.cfg
        self.n_subdomains = c['n_subdomains_list'][0]
        n_levels = len(c['n_subdomains_list'])
        need_r0t = (c.get('mode', 'trad') == 'trad' and n_levels == 1)
        npar = int(os.environ.get('NOSAS2D_L0_PARALLEL', '8'))
        if (not need_r0t and self.n_subdomains >= 2 and npar >= 2
                and c.get('nx', 0) >= 512
                and getattr(self, '_cache_used', False)):
            self._compute_level0_parallel(mu_level0)
            return
        self._compute_level0_serial(mu_level0)

    def _compute_level0_parallel(self, mu_level0):
        """Level-0 subdomain loop split over forked worker processes.

        Only used where the level-0 A4 KSPs are never needed again (see
        ``compute_level0``); each subdomain is computed independently and the
        results are merged in subdomain order, so numerics are identical to
        the serial path.  The parent mesh is inherited by the workers via fork
        (no copy); only the light per-subdomain payload is pickled."""
        global _LEVEL0_WORKER_D, _LEVEL0_WORKER_MESH, _LEVEL0_WORKER_DOFMAP
        c = self.cfg
        n_subdomains = self.n_subdomains
        npar = int(os.environ.get('NOSAS2D_L0_PARALLEL', '8'))
        _t0 = time.time()
        _LEVEL0_WORKER_MESH = self.msh
        _LEVEL0_WORKER_DOFMAP = self.dofmap
        _LEVEL0_WORKER_D = {
            'deg': c['deg'], 'membership': self.membership,
            'gamma': self.Gamma, 'interior': self.Interior,
            'TS_Blk_csr': self.TS_Blk_csr, 'mu_level0': mu_level0,
        }
        if os.name == 'posix':
            import multiprocessing as mp
            try:
                mp.set_start_method('fork', force=True)
            except RuntimeError:
                pass
        nworkers = min(npar, n_subdomains)
        base = n_subdomains // nworkers
        extra = n_subdomains % nworkers
        chunks, start = [], 0
        for w in range(nworkers):
            cnt = base + (1 if w < extra else 0)
            chunks.append(list(range(start, start + cnt)))
            start += cnt
        out_by_id = {}
        with ProcessPoolExecutor(max_workers=nworkers) as ex:
            for out_list in ex.map(_level0_worker, chunks):
                for out in out_list:
                    out_by_id[out['sub_id']] = out
        _LEVEL0_WORKER_D = None
        _LEVEL0_WORKER_MESH = None
        _LEVEL0_WORKER_DOFMAP = None

        self.results = [None] * n_subdomains
        S_rows, S_cols, S_vals = [], [], []
        g_rows, g_vals = [], []
        for sub_id in range(n_subdomains):
            out = out_by_id[sub_id]
            self.results[sub_id] = {
                'sub_id': sub_id,
                'S_Blk_rows': out['S_Blk_rows'], 'S_Blk_cols': out['S_Blk_cols'],
                'S_Blk_vals': out['S_Blk_vals'], 'A_QG_data': out['A_QG_data'],
                'D_inv': out['D_inv'], 'flag1': out['flag1'],
            }
            S_rows.append(out['S_rows'])
            S_cols.append(out['S_cols'])
            S_vals.append(out['S_vals'])
            g_rows.append(out['g_rows'])
            g_vals.append(out['g_vals'])
        print(f"[T] Level0 raw subdomain loop (parallel): {time.time() - _t0:.2f}s")
        self.local_solvers = None
        self.R0T_level0 = None
        self.R0T_blocks = None

        # global Schur S on GG and global RHS g on GG (same as serial path)
        GG = np.sort(self.gg.copy())
        self.gg = GG
        S_rows_full = np.concatenate(S_rows)
        S_cols_full = np.concatenate(S_cols)
        S_vals_full = np.concatenate(S_vals)
        S_csr = csr_matrix((S_vals_full, (S_rows_full, S_cols_full)),
                           shape=(self.n_dofs_total, self.n_dofs_total))
        gg_pos = np.full(self.n_dofs_total, -1, dtype=np.int64)
        gg_pos[GG] = np.arange(len(GG))
        keep = (gg_pos[S_rows_full] >= 0) & (gg_pos[S_cols_full] >= 0)
        self.S_csr_GG = csr_matrix((S_vals_full[keep],
                                    (gg_pos[S_rows_full[keep]], gg_pos[S_cols_full[keep]])),
                                   shape=(len(GG), len(GG)))
        g_rows_full = np.concatenate(g_rows)
        g_vals_full = np.concatenate(g_vals)
        g_full = np.zeros(self.n_dofs_total, dtype=PETSc.ScalarType)
        np.add.at(g_full, g_rows_full, g_vals_full)
        self.g_arr = g_full[GG].astype(PETSc.ScalarType)
        print(f"[T] Level0 global S/g on GG: {time.time() - _t0:.2f}s")

    def _compute_level0_serial(self, mu_level0):
        c = self.cfg
        n_subdomains = c['n_subdomains_list'][0]
        deg = c['deg']
        msh, dofmap = self.msh, self.dofmap
        f = PETSc.ScalarType(1.0)
        _t0 = time.time()

        self.results = [None] * n_subdomains
        self.local_solvers = [None] * n_subdomains
        S_rows, S_cols, S_vals = [], [], []
        g_rows, g_vals = [], []
        R0T_rows, R0T_cols, R0T_data = [], [], []
        R0T_blk_list = []
        # Level-0 R_0T = -A4_inv_A3 is consumed ONLY by the 2-level traditional
        # path (_two_level_assembly).  Filter / hybrid / multi-level runs keep
        # the fine-level R_0T as dead data, so skip storing it entirely.
        need_r0t = (self.cfg.get('mode', 'trad') == 'trad' and
                    len(self.cfg['n_subdomains_list']) == 1)
        # Dense per-subdomain R0T blocks (instead of the concatenated int-64
        # triple) are used for very large grids where the triple alone blows up
        # (nx=2048: ~1.55 GB/sub).  Below the threshold behaviour is identical.
        use_blocks = need_r0t and (os.environ.get('NOSAS2D_FORCE_R0T_BLOCKS', '0') == '1'
                                   or self.n_dofs_total > 2_000_000)
        TS_Blk_csr = self.TS_Blk_csr

        for sub_id in range(n_subdomains):
            sub_tri_ids = np.where(self.membership == sub_id)[0]
            m, n = self.Gamma[sub_id], self.Interior[sub_id]
            _p0 = time.time()
            submesh, entity_map, vertex_map, geometry_map = mesh.create_submesh(
                msh, msh.topology.dim, sub_tri_ids)
            V_sub = fem.functionspace(submesh, ("Lagrange", deg))
            dofmap_sub = V_sub.dofmap
            num_sub_cells = submesh.topology.index_map(2).size_local
            sub_cell_indices = np.arange(num_sub_cells, dtype=np.int32)
            cell_map = entity_map.sub_topology_to_topology(sub_cell_indices, inverse=False)
            sub_dofs_flat = dofmap_sub.list.flatten()
            orig_dofs_flat = dofmap.list[cell_map].flatten()
            sort_idx = np.argsort(sub_dofs_flat)
            sorted_sub = sub_dofs_flat[sort_idx]
            sorted_orig = orig_dofs_flat[sort_idx]
            _, unique_idx = np.unique(sorted_sub, return_index=True)
            all_sub_dofs = sorted_orig[unique_idx]
            sorted_idx = np.argsort(all_sub_dofs)
            sorted_global = all_sub_dofs[sorted_idx]
            all_target = np.concatenate([m, n])
            indices = np.searchsorted(sorted_global, all_target)
            all_local = sorted_idx[indices]
            m_local, n_local = all_local[:len(m)], all_local[len(m):]

            V_local = fem.functionspace(submesh, ("Lagrange", deg))
            u, v = ufl.TrialFunction(V_local), ufl.TestFunction(V_local)
            a_local = fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
            L_local = fem.form(ufl.inner(f, v) * ufl.dx)
            A_local = fem.assemble_matrix(a_local)
            b_local = fem.assemble_vector(L_local)
            A_local = PETSc.Mat().createAIJ(
                size=(A_local.index_map(0).size_local, A_local.index_map(1).size_local),
                csr=(A_local.indptr.astype(PETSc.IntType),
                     A_local.indices.astype(PETSc.IntType),
                     A_local.data.astype(PETSc.ScalarType)))
            is_m_local = PETSc.IS().createGeneral(m_local.astype(PETSc.IntType), comm=PETSc.COMM_WORLD)
            is_n_local = PETSc.IS().createGeneral(n_local.astype(PETSc.IntType), comm=PETSc.COMM_WORLD)
            A1 = A_local.createSubMatrix(is_m_local, is_m_local); A1.assemble()
            A2 = A_local.createSubMatrix(is_m_local, is_n_local); A2.assemble()
            A3 = A_local.createSubMatrix(is_n_local, is_m_local); A3.assemble()
            A4 = A_local.createSubMatrix(is_n_local, is_n_local); A4.assemble()
            ksp_A4 = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
            ksp_A4.setOperators(A4)
            ksp_A4.setType(PETSc.KSP.Type.PREONLY)
            ksp_A4.getPC().setType(PETSc.PC.Type.CHOLESKY)
            ksp_A4.setUp()
            self.local_solvers[sub_id] = ksp_A4
            _p1 = time.time()

            b_vec = PETSc.Vec().createWithArray(b_local.array)
            b_n = b_vec.getSubVector(is_n_local)
            ksp_A4.solve(b_n, b_n)
            g_vec = b_vec.getSubVector(is_m_local) - A2 * b_n
            g_rows.append(all_sub_dofs[is_m_local])
            g_vals.append(g_vec.getArray())
            len_m, len_n = len(m_local), len(n_local)

            solutions = np.zeros((len_n, len_m), dtype=PETSc.ScalarType)
            x_temp = PETSc.Vec().createSeq(len_n)
            for j in range(len_m):
                values = np.zeros(len_n, dtype=PETSc.ScalarType)
                A3.getValues(np.arange(len_n, dtype=PETSc.IntType), np.array([j], dtype=PETSc.IntType), values)
                b = PETSc.Vec().createWithArray(values, comm=PETSc.COMM_WORLD)
                ksp_A4.solve(b, x_temp)
                solutions[:, j] = x_temp.getArray()
                b.destroy()
            x_temp.destroy()
            if use_blocks:
                # dense block: rows=Interior (sorted), cols=Gamma (sorted),
                # D = +A4_inv_A3 (sign flipped in the matvec kernels)
                R0T_blk_list.append({'rows': n, 'cols': m, 'D': solutions})
            elif need_r0t:
                R0T_rows.append(np.repeat(n, len_m))
                R0T_cols.append(np.tile(m, len_n))
                R0T_data.append(-solutions.ravel())
            A4_inv_A3_data = solutions.ravel()
            _A4_csr = csr_matrix((A4_inv_A3_data,
                                  (np.repeat(np.arange(len_n), len_m),
                                   np.tile(np.arange(len_m), len_n))))
            A4_inv_A3 = PETSc.Mat().createAIJ(
                size=(len_n, len_m),
                csr=(_A4_csr.indptr.astype(PETSc.IntType),
                     _A4_csr.indices.astype(PETSc.IntType),
                     A4_inv_A3_data.astype(PETSc.ScalarType)))
            A4_inv_A3.assemble()

            Schur = A1 - A2 * A4_inv_A3
            is_m = PETSc.IS().createGeneral(m.astype(PETSc.IntType), comm=PETSc.COMM_WORLD)
            Schur_np = self._petsc_to_scipy_dense(Schur)
            Schur_np = (Schur_np + Schur_np.T) / 2
            _p2 = time.time()

            S_Blk_rows, S_Blk_cols, S_Blk_vals, A_QG_data, D_inv, flag = \
                self._compute_eigenproblem_from_schur(Schur_np, TS_Blk_csr, m, mu_level0)
            _p3 = time.time()
            print(f"[T] L0 sub{sub_id}: P1(asm)={_p1-_p0:.2f}s P2(A4invA3,g)={_p2-_p1:.2f}s "
                  f"P4(eigs)={_p3-_p2:.2f}s (Gamma {len_m}, Int {len_n}, flag {flag})", flush=True)

            Schur_csr = csr_matrix(Schur_np)
            indptr, ids, values = Schur_csr.indptr, Schur_csr.indices, Schur_csr.data
            row_indices_local = np.repeat(np.arange(len(np.diff(indptr))), np.diff(indptr))
            S_rows.append(m[row_indices_local])
            S_cols.append(m[ids])
            S_vals.append(values)

            self.results[sub_id] = {
                'sub_id': sub_id,
                'S_Blk_rows': S_Blk_rows.copy(),
                'S_Blk_cols': S_Blk_cols.copy(),
                'S_Blk_vals': S_Blk_vals.copy(),
                'A_QG_data': A_QG_data,
                'D_inv': D_inv.copy(),
                'flag1': int(flag),
            }
            is_m_local.destroy(); is_n_local.destroy(); is_m.destroy()
            b_vec.destroy(); b_n.destroy(); g_vec.destroy()
            A_local.destroy(); A1.destroy(); A2.destroy(); A3.destroy(); A4.destroy()
            A4_inv_A3.destroy()
            gc.collect()
        print(f"[T] Level0 raw subdomain loop: {time.time() - _t0:.2f}s")

        # global Schur S on GG and global RHS g on GG
        GG = np.sort(self.gg.copy())
        self.gg = GG
        S_rows_full = np.concatenate(S_rows)
        S_cols_full = np.concatenate(S_cols)
        S_vals_full = np.concatenate(S_vals)
        S_csr = csr_matrix((S_vals_full, (S_rows_full, S_cols_full)),
                           shape=(self.n_dofs_total, self.n_dofs_total))
        gg_pos = np.full(self.n_dofs_total, -1, dtype=np.int64)
        gg_pos[GG] = np.arange(len(GG))
        keep = (gg_pos[S_rows_full] >= 0) & (gg_pos[S_cols_full] >= 0)
        self.S_csr_GG = csr_matrix((S_vals_full[keep],
                                    (gg_pos[S_rows_full[keep]], gg_pos[S_cols_full[keep]])),
                                   shape=(len(GG), len(GG)))
        g_rows_full = np.concatenate(g_rows)
        g_vals_full = np.concatenate(g_vals)
        g_full = np.zeros(self.n_dofs_total, dtype=PETSc.ScalarType)
        np.add.at(g_full, g_rows_full, g_vals_full)
        self.g_arr = g_full[GG].astype(PETSc.ScalarType)
        if use_blocks:
            self.R0T_blocks = R0T_blk_list
        elif need_r0t:
            self.R0T_level0 = (np.concatenate(R0T_rows), np.concatenate(R0T_cols),
                               np.concatenate(R0T_data))
        print(f"[T] Level0 global S/g on GG: {time.time() - _t0:.2f}s")

    def select_level0(self, mu0):
        """Re-run the per-mu selection only (eigen GEP already done in
        compute_level0). Stored per-subdomain flag/A_QG/D_inv are reused."""
        n_subdomains = self.cfg['n_subdomains_list'][0]
        self.S_Blk_rows_list = [None] * n_subdomains
        self.S_Blk_cols_list = [None] * n_subdomains
        self.S_Blk_vals_list = [None] * n_subdomains
        self.A_QG_data_list = [None] * n_subdomains
        self.D_inv_G_list = [None] * n_subdomains
        flag1 = np.zeros(n_subdomains, dtype=np.int32)
        for r in self.results:
            sid = r['sub_id']
            self.S_Blk_rows_list[sid] = r['S_Blk_rows']
            self.S_Blk_cols_list[sid] = r['S_Blk_cols']
            self.S_Blk_vals_list[sid] = r['S_Blk_vals']
            self.A_QG_data_list[sid] = r['A_QG_data']
            self.D_inv_G_list[sid] = r['D_inv']
            flag1[sid] = r['flag1']
        self.flag1 = flag1

    def _sym_slice_dense(self, S_Blk_rows, S_Blk_cols, S_Blk_vals, rownames, s):
        sel_pos = np.full(self.n_dofs_total, -1, dtype=np.int64)
        sel_pos[rownames] = np.arange(len(rownames))
        keep = (sel_pos[s] >= 0) & (S_Blk_rows >= 0)
        # build S_Blk restricted to subset s (rows & cols global dofs)
        # (rows already global; columns global too)
        keep = (sel_pos[S_Blk_rows] >= 0) & (sel_pos[S_Blk_cols] >= 0) if S_Blk_cols is not None else keep
        return S_Blk_rows, S_Blk_cols, S_Blk_vals, sel_pos

    # ------------------------------------------------------------------
    # coarsening chain (identical to reference perform_coarsening)
    # ------------------------------------------------------------------

    def _extract_coarse_variables(self, gamma_list, n_coarse_subdomains, group_assignment_coarse):
        dofs_bc = self.dofs_bc
        subdomain_all_dofs = [None] * n_coarse_subdomains
        coarse_v = np.zeros(self.n_dofs_total, dtype=np.int32)
        coarse_v2 = np.zeros(self.n_dofs_total, dtype=np.int32)
        for coarse_id in range(n_coarse_subdomains):
            orig_sub_ids = np.where(group_assignment_coarse == coarse_id)[0]
            dofs_set = set()
            for sub_id in orig_sub_ids:
                dofs_set.update(gamma_list[sub_id])
            group_dofs = np.array(list(dofs_set), dtype=np.int32)
            subdomain_all_dofs[coarse_id] = group_dofs
            coarse_v[group_dofs] += 1
            coarse_v2[group_dofs] += (coarse_id + 1)
        coarse_GG = np.setdiff1d(np.where(coarse_v >= 2)[0], dofs_bc)
        coarse_Subd_V = np.setdiff1d(np.where(coarse_v >= 3)[0], dofs_bc)
        coarse_II = np.setdiff1d(np.where(coarse_v == 1)[0], dofs_bc)
        return coarse_GG, coarse_Subd_V, coarse_II, subdomain_all_dofs, coarse_v, coarse_v2

    def _extract_coarse_edges(self, subdomain_all_dofs, coarse_Subd_V, coarse_II,
                              coarse_v2, n_coarse_subdomains):
        dofs_bc = self.dofs_bc
        coarse_v2[coarse_Subd_V] = 0
        coarse_v2[dofs_bc] = 0
        coarse_v2[coarse_II] = 0
        coarse_Subd_E_candidates = []
        for coarse_id in range(n_coarse_subdomains):
            group_dofs = subdomain_all_dofs[coarse_id]
            non_zero_mask = coarse_v2[group_dofs] != 0
            if not non_zero_mask.any():
                continue
            edge_dofs = group_dofs[non_zero_mask]
            edge_values = coarse_v2[edge_dofs]
            unique_values, inverse_indices = np.unique(edge_values, return_inverse=True)
            for i, val in enumerate(unique_values):
                if val == 0:
                    continue
                entity = edge_dofs[inverse_indices == i]
                if len(entity) > 0:
                    coarse_Subd_E_candidates.append(np.sort(entity))
        coarse_Subd_E = []
        edge_counter = 0
        for new_edge in coarse_Subd_E_candidates:
            new_set = set(new_edge)
            if not new_set:
                continue
            v2_vals = coarse_v2[np.array(list(new_set), dtype=np.int32)]
            has_neg = np.any(v2_vals < 0)
            if not has_neg:
                edge_counter += 1
                coarse_Subd_E.append(np.array(sorted(new_set), dtype=np.int32))
                coarse_v2[np.array(list(new_set), dtype=np.int32)] = -edge_counter
                continue
            neg_vals = v2_vals[v2_vals < 0]
            if len(neg_vals) == len(v2_vals) and len(set(neg_vals)) == 1:
                same_eid = -neg_vals[0]
                existing = coarse_Subd_E[same_eid - 1]
                if existing is not None and len(existing) == len(new_edge) and \
                        np.array_equal(np.sort(existing), np.sort(new_edge)):
                    continue
            involved_eids = set()
            for d in new_set:
                if coarse_v2[d] < 0:
                    involved_eids.add(-coarse_v2[d])
            all_sets = [new_set]
            eid_to_idx = {}
            for eid in involved_eids:
                idx = eid - 1
                eid_to_idx[eid] = idx
                all_sets.append(set(coarse_Subd_E[idx]))
            for eid in involved_eids:
                coarse_Subd_E[eid_to_idx[eid]] = None
            dof_count = {}
            for s in all_sets:
                for d in s:
                    dof_count[d] = dof_count.get(d, 0) + 1
            added = set()
            for s in all_sets:
                non_conflict = tuple(sorted(d for d in s if dof_count[d] == 1))
                conflict = [d for d in s if dof_count[d] > 1]
                if non_conflict and non_conflict not in added:
                    edge_counter += 1
                    coarse_SubD_E = None
                    coarse_Subd_E.append(np.array(non_conflict, dtype=np.int32))
                    coarse_v2[np.array(non_conflict, dtype=np.int32)] = -edge_counter
                    added.add(non_conflict)
                for d in conflict:
                    if (d,) not in added:
                        edge_counter += 1
                        coarse_Subd_E.append(np.array([d], dtype=np.int32))
                        coarse_v2[d] = -edge_counter
                        added.add((d,))
        # (unused line kept out): _ = None
        coarse_Subd_E = [e for e in coarse_Subd_E if e is not None]
        return coarse_Subd_E

    def _construct_TS_Blk_csr(self, Subd_V, Subd_E):
        all_rows = np.concatenate([Subd_V] + [np.repeat(edge, len(edge)) for edge in Subd_E])
        all_cols = np.concatenate([Subd_V] + [np.tile(edge, len(edge)) for edge in Subd_E])
        all_vals = np.ones(len(all_rows), dtype=PETSc.ScalarType)
        return csr_matrix((all_vals, (all_rows, all_cols)), shape=(self.n_dofs_total, self.n_dofs_total))

    @staticmethod
    def _extract_gamma_interior(subdomain_all_dofs, coarse_GG, dofs_bc, n_coarse_subdomains):
        coarse_Gamma = [None] * n_coarse_subdomains
        coarse_Interior = [None] * n_coarse_subdomains
        for coarse_id in range(n_coarse_subdomains):
            group_dofs = subdomain_all_dofs[coarse_id]
            coarse_Interior[coarse_id] = np.setdiff1d(np.setdiff1d(group_dofs, dofs_bc), coarse_GG)
            coarse_Gamma[coarse_id] = np.intersect1d(group_dofs, coarse_GG)
        return coarse_Gamma, coarse_Interior

    def perform_coarsening(self, gamma_list, n_coarse_subdomains, level_name):
        dofs_bc = self.dofs_bc
        n_groups = len(gamma_list)
        adj_sub = [[] for _ in range(n_groups)]
        for i in range(n_groups):
            gamma_i = gamma_list[i]
            for j in range(i + 1, n_groups):
                if np.intersect1d(gamma_i, gamma_list[j], assume_unique=True).size > 0:
                    adj_sub[i].append(j)
                    adj_sub[j].append(i)
        _, group_assignment_coarse = pymetis.part_graph(n_coarse_subdomains, adjacency=adj_sub)
        group_assignment_coarse = np.array(group_assignment_coarse)
        unique_coarse_ids = np.unique(group_assignment_coarse)
        valid_coarse_ids = [cid for cid in range(n_coarse_subdomains) if cid in unique_coarse_ids]
        actual_n_subdomains = len(valid_coarse_ids)
        print(f"{level_name} actual effective subdomains: {actual_n_subdomains} "
              f"(configured: {n_coarse_subdomains})")
        id_map = {old_id: new_id for new_id, old_id in enumerate(valid_coarse_ids)}
        group_assignment_coarse_new = np.array([id_map[old_id] for old_id in group_assignment_coarse])
        coarse_GG, coarse_Subd_V, coarse_II, subdomain_all_dofs, coarse_v, coarse_v2 = \
            self._extract_coarse_variables(gamma_list, actual_n_subdomains,
                                           group_assignment_coarse_new)
        coarse_Subd_E = self._extract_coarse_edges(subdomain_all_dofs, coarse_Subd_V,
                                                   coarse_II, coarse_v2, actual_n_subdomains)
        coarse_TS_Blk_csr = self._construct_TS_Blk_csr(coarse_Subd_V, coarse_Subd_E)
        coarse_TS_Blk_mat = self._petsc_mat_from_csr(coarse_TS_Blk_csr)
        coarse_TS_Blk_mat.assemble()
        coarse_Gamma, coarse_Interior = self._extract_gamma_interior(
            subdomain_all_dofs, coarse_GG, dofs_bc, actual_n_subdomains)
        return (coarse_GG, coarse_Subd_V, coarse_II, coarse_Subd_E,
                coarse_TS_Blk_csr, coarse_TS_Blk_mat, coarse_Gamma, coarse_Interior,
                subdomain_all_dofs, group_assignment_coarse_new, actual_n_subdomains)

    def build_coarsening_chain(self):
        c = self.cfg
        n_levels = len(c['n_subdomains_list'])
        coarse_levels_data = []
        current_gamma_list = self.Gamma
        current_Subd_V = self.Subd_V
        current_Subd_E = self.Subd_E
        for level_idx in range(1, n_levels):
            target_n = c['n_subdomains_list'][level_idx]
            mu_val = c['mu_list'][level_idx] if level_idx < len(c['mu_list']) else c['mu_list'][-1]
            eta_val = c['eta_list'][level_idx] if c.get('eta_list') and level_idx < len(c['eta_list']) \
                else (c['eta_list'][-1] if c.get('eta_list') else None)
            level_name = f"Level {level_idx + 1}"
            (coarse_GG, coarse_Subd_V, coarse_II, coarse_Subd_E,
             coarse_TS_Blk_csr, coarse_TS_Blk_mat, coarse_Gamma, coarse_Interior,
             subdomain_all_dofs_coarse, group_assignment_coarse, actual_n) = self.perform_coarsening(
                current_gamma_list, target_n, level_name)
            coarse_levels_data.append({
                'GG': coarse_GG, 'Subd_V': coarse_Subd_V, 'Subd_E': coarse_Subd_E,
                'TS_Blk_csr': coarse_TS_Blk_csr, 'TS_Blk_mat': coarse_TS_Blk_mat,
                'Gamma': coarse_Gamma, 'Interior': coarse_Interior,
                'subdomain_all_dofs': subdomain_all_dofs_coarse,
                'group_assignment': group_assignment_coarse,
                'actual_n': actual_n, 'mu': mu_val, 'eta': eta_val, 'II': coarse_II,
            })
            # Reference scripts feed the next level either the full group dofs
            # (run_table4a/5b: subdomain_all_dofs_coarse) or only the group
            # Gamma intersections (run_table3/5a/5c: coarse_Gamma).  replicate.
            feed_rule = ('trad', 'hybrid')
            current_gamma_list = coarse_Gamma if c['mode'] in feed_rule \
                else subdomain_all_dofs_coarse
            current_Subd_V = coarse_Subd_V
            current_Subd_E = coarse_Subd_E
        self.coarse_levels_data = coarse_levels_data
        return coarse_levels_data

    # ------------------------------------------------------------------
    # final global assembly via numba entity-block kernels + inv cache
    # ------------------------------------------------------------------

    def _build_global_from_level(self, A_QG_data, S_Blk_rows, S_Blk_cols, S_Blk_vals,
                                 Subd_V, Subd_E, coarse_GG, D_inv_G, inv_ready=None,
                                 cache_key=None):
        """Build (S_Blk_GG, A_QG_GG, InvS_Blk, W, Global_Solver) exactly as the
        reference assembles them, using numba entity-block kernels.  When
        ``inv_ready`` list of ``(dofs, inverse)`` is given, entity inverses are
        reused (bit-identical)."""
        _t0 = time.time()
        n_gg = len(coarse_GG)
        gg_pos = np.full(self.n_dofs_total, -1, dtype=np.int64)
        gg_pos[coarse_GG] = np.arange(n_gg)

        # --- S_Blk restricted to coarse_GG ---
        all_r = np.concatenate([r for r in S_Blk_rows if r is not None and len(r) > 0]) if any(
            r is not None and len(r) > 0 for r in S_Blk_rows) else np.array([], dtype=np.int64)
        all_c = np.concatenate([c for c in S_Blk_cols if c is not None and len(c) > 0]) if any(
            c is not None and len(c) > 0 for c in S_Blk_cols) else np.array([], dtype=np.int64)
        all_v = np.concatenate([v for v in S_Blk_vals if v is not None and len(v) > 0]) if any(
            v is not None and len(v) > 0 for v in S_Blk_vals) else np.array([], dtype=np.float64)
        if all_r.size:
            keep = (gg_pos[all_r] >= 0) & (gg_pos[all_c] >= 0)
            S_Blk_GG = csr_matrix((all_v[keep],
                                   (gg_pos[all_r[keep]], gg_pos[all_c[keep]])),
                                  shape=(n_gg, n_gg))
        else:
            S_Blk_GG = csr_matrix((n_gg, n_gg))

        # --- A_QG restricted to coarse_GG ---
        all_rows, all_cols, all_vals = [], [], []
        col_offset = 0
        for data in A_QG_data:
            if data is None:
                continue
            m_indices = data['m']
            A1Q = data['A1Q']
            n_eigen = data['n_eigen']
            if n_eigen == 0 or len(m_indices) == 0:
                continue
            keepA = gg_pos[m_indices] >= 0
            loc = gg_pos[m_indices[keepA]]
            all_rows.append(np.repeat(loc, n_eigen))
            all_cols.append(np.tile(np.arange(n_eigen) + col_offset, len(loc)))
            all_vals.append(A1Q[keepA].ravel())
            col_offset += n_eigen
        if len(all_rows) > 0:
            all_rows = np.concatenate(all_rows); all_cols = np.concatenate(all_cols)
            all_vals = np.concatenate(all_vals)
        else:
            all_rows = np.array([], dtype=np.int64); all_cols = np.array([], dtype=np.int64)
            all_vals = np.array([], dtype=np.float64)
        A_QG_GG = csr_matrix((all_vals, (all_rows, all_cols)), shape=(n_gg, col_offset))
        print(f"[T]  assemble b: A_QG_GG({n_gg} x {col_offset}): {time.time()-_t0:.2f}s")

        # --- InvS_Blk over entity blocks (vertex V + edges E), reusing inv cache ---
        _tc = time.time()
        ready = {}
        if inv_ready is not None:
            for _d, _inv in inv_ready:
                ready[frozenset(_d)] = (_d, _inv)
        rows, cols, vals = [], [], []
        ent_loc, ent_inv = [], []
        n_reused = 0
        pos_scratch = np.full(n_gg, -1, dtype=np.int64)
        for entity in list(Subd_V) + list(Subd_E):
            entity = np.atleast_1d(entity)
            loc = gg_pos[entity]
            loc = loc[loc >= 0]
            if len(loc) == 0:
                continue
            key = frozenset(np.sort(entity))
            inv = None
            if key in ready:
                _d, _inv = ready[key]
                pmap = {int(dd): k for k, dd in enumerate(_d)}
                perm = np.array([pmap[int(dof)] for dof in entity], dtype=np.int64)
                inv = _inv[np.ix_(perm, perm)]
                n_reused += 1
            else:
                S_dense = _csr_block_gather(S_Blk_GG.indptr, S_Blk_GG.indices, S_Blk_GG.data,
                                            loc.astype(np.int64), pos_scratch)
                try:
                    inv = np.linalg.inv(S_dense)
                except np.linalg.LinAlgError:
                    inv = np.linalg.pinv(S_dense)
            nent = len(loc)
            rows.append(np.repeat(loc, nent)); cols.append(np.tile(loc, nent))
            vals.append(inv.ravel())
            ent_loc.append(loc); ent_inv.append(inv)
        if rows:
            rows = np.concatenate(rows); cols = np.concatenate(cols)
            vals = np.concatenate(vals)
        else:
            rows = np.array([], dtype=np.int64); cols = np.array([], dtype=np.int64)
            vals = np.array([], dtype=np.float64)
        InvS_Blk_csr = csr_matrix((vals, (rows, cols)), shape=(n_gg, n_gg))
        if n_reused:
            print(f"  [inv_ready] reused {n_reused}/{len(ent_loc)} entity inverses")
        print(f"[T]  assemble c: InvS_Blk_GG({n_gg} x {n_gg}): {time.time()-_tc:.2f}s")

        # --- W = InvS_Blk @ A_QG over entity blocks (block diagonal -> exact) ---
        _td = time.time()
        W_rows, W_cols, W_vals = [], [], []
        for loc, inv in zip(ent_loc, ent_inv):
            Abl = A_QG_GG[loc, :]
            Wn = inv @ Abl.toarray()
            nent = len(loc)
            W_rows.append(np.repeat(loc, col_offset))
            W_cols.append(np.tile(np.arange(col_offset), nent))
            W_vals.append(Wn.ravel())
        if W_rows:
            W_csr = csr_matrix((np.concatenate(W_vals),
                                (np.concatenate(W_rows), np.concatenate(W_cols))),
                               shape=(n_gg, col_offset))
        else:
            W_csr = csr_matrix((n_gg, col_offset))
        print(f"[T]  assemble d: W shell: {time.time()-_td:.2f}s")

        # --- Global_Solver = D_inv - A_QG^T W  (numba kernels, entity blocks) ---
        _tgs = time.time()
        if D_inv_G is None:
            D_inv_sel = np.ones(col_offset)
        else:
            _d_parts = [d for d in D_inv_G if d is not None and len(d) > 0]
            D_inv_sel = np.concatenate(_d_parts) if _d_parts else np.array([], dtype=np.float64)
            if len(D_inv_sel) < col_offset:
                D_inv_sel = np.ones(col_offset)
        AQ_indptr = A_QG_GG.indptr
        AQ_indices = A_QG_GG.indices
        AQ_data = A_QG_GG.data
        Goff_dense = np.zeros((col_offset, col_offset))
        for loc, inv in zip(ent_loc, ent_inv):
            nz_cols, Adense = _csr_rows_extract(AQ_indptr, AQ_indices, AQ_data,
                                                loc.astype(np.int64))
            if nz_cols.shape[0] == 0:
                continue
            Wn = inv @ Adense
            _add_block(Goff_dense, nz_cols, Adense.T @ Wn)
        rr, cc = np.nonzero(Goff_dense)
        _gs_csr = csr_matrix((-Goff_dense[rr, cc], (rr, cc)), shape=(col_offset, col_offset))
        _gs_csr = _gs_csr + csr_matrix((D_inv_sel, (np.arange(col_offset), np.arange(col_offset))),
                                       shape=(col_offset, col_offset))
        del Goff_dense
        print(f"[T]  assemble f: Global_Solver({col_offset} x {col_offset}): {time.time()-_tgs:.2f}s")
        return S_Blk_GG, A_QG_GG, InvS_Blk_csr, W_csr, _gs_csr

    def _assemble_R0T(self, R0T_rows, R0T_cols, R0T_data, II_sel, GG_sel):
        R0T_all_rows, R0T_all_cols, R0T_all_data = map(np.concatenate,
                                                       [R0T_rows, R0T_cols, R0T_data])
        R_0T_csr = csr_matrix((R0T_all_data, (R0T_all_rows, R0T_all_cols)),
                              shape=(self.n_dofs_total, self.n_dofs_total))
        ii_pos = np.full(self.n_dofs_total, -1, dtype=np.int64)
        ii_pos[II_sel] = np.arange(len(II_sel))
        gg_pos = np.full(self.n_dofs_total, -1, dtype=np.int64)
        gg_pos[GG_sel] = np.arange(len(GG_sel))
        keep = (ii_pos[R0T_all_rows] >= 0) & (gg_pos[R0T_all_cols] >= 0)
        R_0T_sel = csr_matrix((R0T_all_data[keep],
                               (ii_pos[R0T_all_rows[keep]], gg_pos[R0T_all_cols[keep]])),
                              shape=(len(II_sel), len(GG_sel)))
        return self._petsc_mat_from_csr(R_0T_sel)

    # ------------------------------------------------------------------
    # dense per-subdomain R0T blocks (2-level path on very large grids)
    # ------------------------------------------------------------------

    def _build_r0t_blocks(self):
        """Position-mapped per-subdomain dense blocks of R_0T = -A4_inv_A3.

        Level-0 rows are interior dofs (position in ``self.II``) and columns
        are interface dofs (position in ``self.gg``); both index sets are
        sorted ascending, and each subdomain's Interior/Gamma subsets are
        sorted, so ``searchsorted`` gives the exact CSR position mapping used
        by ``_assemble_R0T``.  ``D = +A4_inv_A3`` is stored and the R_0T sign
        is applied in the matvec kernels.
        """
        blocks = self.R0T_blocks
        return {
            'kind': 'blocks',
            'n_II': len(self.II),
            'n_GG': len(self.gg),
            'D': [blk['D'] for blk in blocks],
            'rpos': [np.searchsorted(self.II, blk['rows']).astype(np.int32) for blk in blocks],
            'cpos': [np.searchsorted(self.gg, blk['cols']).astype(np.int32) for blk in blocks],
        }

    @staticmethod
    def _r0t_block_multT(R0B, x_np, y_np):
        """y = R_0T^T * x (x in II order, y in GG order), row blocks of the
        R0T triple accumulated per subdomain (deterministic BLAS order)."""
        y_np[:] = 0.0
        for D, rp, cp in zip(R0B['D'], R0B['rpos'], R0B['cpos']):
            if len(rp) == 0 or len(cp) == 0:
                continue
            np.add.at(y_np, cp, -(x_np[rp] @ D))

    @staticmethod
    def _r0t_block_mult(R0B, x_np, y_np):
        """y = R_0T * x (x in GG order, y in II order).  Rows of different
        subdomains are disjoint, so per-subdomain row scatter is exact."""
        y_np[:] = 0.0
        for D, rp, cp in zip(R0B['D'], R0B['rpos'], R0B['cpos']):
            if len(rp) == 0 or len(cp) == 0:
                continue
            y_np[rp] -= D @ x_np[cp]

    def _make_global_solver(self, mat):
        """KSP for the final coarse Global_Solver.

        Defaults to the reference PREONLY + CHOLESKY (bit-identical), with a
        scipy-superlu fallback selected by ``NOSAS2D_FINAL_SPLU=1`` or when
        the coarse solve has more than 2M rows (well past the eigen-coarse
        matrices used in this repo, but required for whole-domain coarse
        solves in Section-4 mode at h=1/OOM scales)."""
        nrows = mat.getSize()[0]
        use_splu = (os.environ.get('NOSAS2D_FINAL_SPLU', '0') == '1') or (nrows > 2_000_000)
        if not use_splu:
            ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
            ksp.setOperators(mat)
            ksp.setType(PETSc.KSP.Type.PREONLY)
            ksp.getPC().setType(PETSc.PC.Type.CHOLESKY)
            ksp.setUp()
            return ksp
        return _SpluSolver(mat)

    # ------------------------------------------------------------------
    # traditional Section-3 subdomain level (identical to reference)
    # ------------------------------------------------------------------

    def compute_subdomain_level(self, S_Blk_prev_rows, S_Blk_prev_cols, S_Blk_prev_vals,
                                A_QG_prev_data, D_inv_prev,
                                subdomain_all_dofs, Gamma_list, Interior_list,
                                Subd_V_list, Subd_E_list, TS_Blk_coarse_csr,
                                n_coarse_subdomains, group_assignment_coarse, mu_val):
        Local_Solver = [None] * n_coarse_subdomains
        A_QG_data = [None] * n_coarse_subdomains
        flag = np.zeros(n_coarse_subdomains, dtype=np.int32)
        R0T_all_rows, R0T_all_cols, R0T_all_data = [], [], []
        D_inv_L = [None] * n_coarse_subdomains
        S_Blk_rows = [None] * n_coarse_subdomains
        S_Blk_cols = [None] * n_coarse_subdomains
        S_Blk_vals = [None] * n_coarse_subdomains
        n_dofs = self.n_dofs_total
        for coarse_id in range(n_coarse_subdomains):
            sub_ids_in_group = np.where(group_assignment_coarse == coarse_id)[0]
            s, m, n = subdomain_all_dofs[coarse_id], Gamma_list[coarse_id], Interior_list[coarse_id]
            grp_S_rows = np.concatenate([S_Blk_prev_rows[sub_id] for sub_id in sub_ids_in_group])
            grp_S_cols = np.concatenate([S_Blk_prev_cols[sub_id] for sub_id in sub_ids_in_group])
            grp_S_vals = np.concatenate([S_Blk_prev_vals[sub_id] for sub_id in sub_ids_in_group])
            S_Blk_local_csr = csr_matrix((grp_S_vals, (grp_S_rows, grp_S_cols)),
                                         shape=(n_dofs, n_dofs))
            S_Blk_local = self._petsc_mat_from_csr(S_Blk_local_csr)
            S_Blk_local.assemble()
            local_A_QG_rows, local_A_QG_cols, local_A_QG_vals = [], [], []
            col_offset = 0
            for sub_id in sub_ids_in_group:
                data = A_QG_prev_data[sub_id]
                m_indices, A1Q, n_eigen = data['m'], data['A1Q'], data['n_eigen']
                local_A_QG_rows.append(np.repeat(m_indices, n_eigen))
                local_A_QG_cols.append(np.tile(np.arange(n_eigen) + col_offset, len(m_indices)))
                local_A_QG_vals.append(A1Q.ravel())
                col_offset += n_eigen
            all_rows, all_cols, all_vals = map(np.concatenate,
                                               [local_A_QG_rows, local_A_QG_cols, local_A_QG_vals])
            A_QG_local_csr = csr_matrix((all_vals, (all_rows, all_cols)),
                                        shape=(n_dofs, col_offset))
            A_QG_local = self._petsc_mat_from_csr(A_QG_local_csr)
            A_QG_local.assemble()
            D_local_vec = np.concatenate([D_inv_prev[sub_id] for sub_id in sub_ids_in_group])
            indices = np.arange(len(D_local_vec), dtype=np.int32)
            diag_csr = csr_matrix((D_local_vec, (indices, indices)),
                                  shape=(len(D_local_vec), len(D_local_vec)))
            InvD_G_local = self._petsc_mat_from_csr(diag_csr)
            InvD_G_local.assemble()
            invD_local_vec = 1.0 / D_local_vec
            invD_G_local_csr = csr_matrix((invD_local_vec, (indices, indices)),
                                          shape=(len(D_local_vec), len(D_local_vec)))
            rows, cols, vals = [], [], []
            s_set = set(s)
            s_to_local = np.full(n_dofs, -1, dtype=np.int32)
            s_to_local[s] = np.arange(len(s))
            for entity_list in [Subd_V_list, Subd_E_list]:
                for entity in entity_list:
                    entity = np.atleast_1d(entity).astype(PETSc.IntType)
                    if not (set(entity) & s_set):
                        continue
                    is_entity = PETSc.IS().createGeneral(entity, comm=PETSc.COMM_WORLD)
                    S_sub = S_Blk_local.createSubMatrix(is_entity, is_entity)
                    S_sub.assemble()
                    I_s, J_s, V_s = S_sub.getValuesCSR()
                    S_dense = csr_matrix((V_s, J_s, I_s), shape=S_sub.getSize()).toarray()
                    inv = np.linalg.inv(S_dense)
                    n_entity = len(entity)
                    entity_local = s_to_local[entity]
                    rows.extend(np.repeat(entity_local, n_entity))
                    cols.extend(np.tile(entity_local, n_entity))
                    vals.extend(inv.ravel())
                    is_entity.destroy()
                    S_sub.destroy()
            InvS_Blk_s = csr_matrix((vals, (rows, cols)), shape=(len(s), len(s)))
            InvS_Blk_PETSc = self._petsc_mat_from_csr(InvS_Blk_s)
            InvS_Blk_PETSc.assemble()
            is_n = PETSc.IS().createGeneral(n.astype(PETSc.IntType), comm=PETSc.COMM_WORLD)
            is_n_in_s = s_to_local[n]
            is_n_petsc = PETSc.IS().createGeneral(is_n_in_s.astype(PETSc.IntType),
                                                  comm=PETSc.COMM_WORLD)
            InvS_Blk_local = InvS_Blk_PETSc.createSubMatrix(is_n_petsc, is_n_petsc)
            InvS_Blk_local.assemble()
            is_n_petsc.destroy()
            U = A_QG_local.createSubMatrix(is_n, None)
            U.assemble()
            W = InvS_Blk_local * U
            Global_Solver = InvD_G_local - U.transposeMatMult(W)
            WT = W.copy().transpose()
            WT.assemble()
            ksp = PETSc.KSP().create()
            ksp.setType('preonly')
            ksp.getPC().setType('lu')
            ksp.getPC().setFactorSolverType('mumps')
            ksp.setOperators(Global_Solver)
            ksp.setUp()
            n_rows, n_cols = WT.getSize()
            solutions = np.zeros((n_rows, n_cols), dtype=PETSc.ScalarType)
            x_temp = PETSc.Vec().createSeq(n_rows)
            for j in range(n_cols):
                b = WT.getColumnVector(j)
                ksp.solve(b, x_temp)
                solutions[:, j] = x_temp.getArray()
                b.destroy()
            x_temp.destroy()
            X_data = solutions.ravel()
            X_csr = csr_matrix((X_data, (np.repeat(np.arange(n_rows), n_cols),
                                         np.tile(np.arange(n_cols), n_rows))),
                               shape=(n_rows, n_cols))
            X = self._petsc_mat_from_csr(X_csr)
            X.assemble()
            yy = InvS_Blk_local + W * X
            Local_Solver[coarse_id] = yy
            S_mm = S_Blk_local_csr[m][:, m]
            S_mn = S_Blk_local_csr[m][:, n]
            A_QG_m = A_QG_local_csr[m]
            A_QG_n = A_QG_local_csr[n]
            A1 = S_mm - A_QG_m @ invD_G_local_csr @ A_QG_m.T
            A2 = S_mn - A_QG_m @ invD_G_local_csr @ A_QG_n.T
            A3 = A2.T
            indptr_yy, indices_yy, values_yy = yy.getValuesCSR()
            yy_csr = csr_matrix((values_yy, indices_yy, indptr_yy), shape=yy.getSize())
            yy_A3 = yy_csr @ A3
            yy_A3_mat = self._petsc_mat_from_csr(yy_A3)
            indptr, indices, values = yy_A3_mat.getValuesCSR()
            row_indices_local = np.repeat(np.arange(len(np.diff(indptr))), np.diff(indptr))
            R0T_all_rows.append(n[row_indices_local])
            R0T_all_cols.append(m[indices])
            R0T_all_data.append(-values)
            Schur_np = ((Schur_np0 := (A1 - A2 @ yy_A3)) + Schur_np0.T) / 2
            Schur_np = Schur_np.toarray()
            (S_Blk_rows[coarse_id], S_Blk_cols[coarse_id], S_Blk_vals[coarse_id],
             A_QG_data[coarse_id], D_inv_L[coarse_id], flag[coarse_id]) = \
                self._compute_eigenproblem_from_schur(Schur_np, TS_Blk_coarse_csr, m, mu_val)
            yy_A3_mat.destroy()
            S_Blk_local.destroy()
            A_QG_local.destroy()
            InvD_G_local.destroy()
            InvS_Blk_PETSc.destroy()
            InvS_Blk_local.destroy()
            U.destroy()
            W.destroy()
            WT.destroy()
            Global_Solver.destroy()
            X.destroy()
            is_n.destroy()
            ksp.destroy()
        return (Local_Solver, A_QG_data, flag, R0T_all_rows, R0T_all_cols, R0T_all_data,
                D_inv_L, S_Blk_rows, S_Blk_cols, S_Blk_vals)

    # ------------------------------------------------------------------
    # Section-4 filter (references run_table4/5b)
    # ------------------------------------------------------------------

    def section4_filter(self, A_QG_np_s, B_Gamma, D_inv, s, eta, use_invD,
                        level_name='', group_id=-1, inv_ready=None, sparse=False):
        """Section-4 filtering eigenproblem.

        Dense path (``sparse=False``, default) is bit-identical to the
        reference: Cholesky ``B_Gamma = L L^T`` then ``W = L^-1 A_QG`` and
        ``WtW = W^T W = A_QG^T B_Gamma^-1 A_QG`` solved as a (generalised)
        eigenproblem.  The sparse path (``sparse=True``) computes the same
        invariant quantities with a sparse SuperLU factorisation instead of a
        dense Cholesky, so the *algorithm* is unchanged and the dense
        n x n B_Gamma / Cholesky factors are never materialised.
        """
        n_eigen_old = A_QG_np_s.shape[1]
        if sparse:
            from scipy.sparse.linalg import splu
            _B = B_Gamma.tocsr()
            lu = splu(_B, permc_spec='COLAMD')
            # X = B_Gamma^-1 A_QG  =>  A_QG^T B_Gamma^-1 A_QG (= dense WtW)
            X = lu.solve(A_QG_np_s)
            WtW = A_QG_np_s.T @ X
        else:
            L = cholesky(B_Gamma, lower=True)
            W_mat = solve_triangular(L, A_QG_np_s, lower=True)
            WtW = W_mat.T @ W_mat
        if use_invD:
            eigvals, eigvecs = eigh(WtW, np.diag(D_inv))
        else:
            eigvals, eigvecs = eigh(WtW)
        threshold = 1 - eta
        if len(eigvals) > 0:
            label = 'genGEP' if use_invD else 'stdGEP'
            n_gt = np.sum(eigvals > threshold)
            print(f'  {label}[{level_name},grp={group_id}]: eig=[{eigvals.min():.4f}, '
                  f'{eigvals.max():.4f}] n_gt={n_gt}/{len(eigvals)}')
            if eigvals.max() > 1.0 + 1e-12:
                print(f'  WARNING: max eigenvalue {eigvals.max():.6f} > 1!')
        idx = np.where(eigvals > threshold)[0]
        n_selected = len(idx)
        if n_selected == 0:
            return np.zeros((len(s), 0)), 0, np.array([])
        x = eigvecs[:, idx]
        D_x = eigvals[idx]
        if use_invD:
            z = x * D_inv[:, np.newaxis]
            D_inv_selected = D_inv[idx]
        else:
            z = x.copy()
            D_inv_selected = np.ones(n_selected)
        WtW_inv_z = np.linalg.solve(WtW, z)
        if sparse:
            # xi = B_Gamma^-1 A_QG WtW^-1 z (= dense L^-T W WtW^-1 z)
            xi = lu.solve(A_QG_np_s @ WtW_inv_z)
        else:
            H_np = W_mat @ WtW_inv_z
            xi = solve_triangular(L.T, H_np, lower=False)
        xi_D = xi @ np.diag(D_x)
        if sparse:
            hat_U_s = np.asarray(_B.dot(xi_D))
        else:
            hat_U_s = B_Gamma @ xi_D
        return hat_U_s, n_selected, D_inv_selected

    def compute_level_section4(self, S_Blk_prev_rows, S_Blk_prev_cols, S_Blk_prev_vals,
                               A_QG_prev_data, D_inv_G_prev,
                               subdomain_all_dofs, Gamma_list, Interior_list,
                               Subd_V_list, Subd_E_list,
                               n_coarse_subdomains, group_assignment_coarse,
                               eta, use_invD, level_name='', inv_ready=None):
        A_QG_data = [None] * n_coarse_subdomains
        flag = np.zeros(n_coarse_subdomains, dtype=np.int32)
        D_inv_G = [None] * n_coarse_subdomains
        S_Blk_rows = [None] * n_coarse_subdomains
        S_Blk_cols = [None] * n_coarse_subdomains
        S_Blk_vals = [None] * n_coarse_subdomains
        n_dofs = self.n_dofs_total
        for coarse_id in range(n_coarse_subdomains):
            sub_ids_in_group = np.where(group_assignment_coarse == coarse_id)[0]
            s, m, n = subdomain_all_dofs[coarse_id], Gamma_list[coarse_id], Interior_list[coarse_id]
            local_A_QG_rows, local_A_QG_cols, local_A_QG_vals = [], [], []
            col_offset = 0
            for sub_id in sub_ids_in_group:
                data = A_QG_prev_data[sub_id]
                m_indices, A1Q, n_eigen = data['m'], data['A1Q'], data['n_eigen']
                local_A_QG_rows.append(np.repeat(m_indices, n_eigen))
                local_A_QG_cols.append(np.tile(np.arange(n_eigen) + col_offset, len(m_indices)))
                local_A_QG_vals.append(A1Q.ravel())
                col_offset += n_eigen
            all_rows = np.concatenate(local_A_QG_rows) if local_A_QG_rows else np.array([], dtype=np.int32)
            all_cols = np.concatenate(local_A_QG_cols) if local_A_QG_cols else np.array([], dtype=np.int32)
            all_vals = np.concatenate(local_A_QG_vals) if local_A_QG_vals else np.array([], dtype=PETSc.ScalarType)
            A_QG_local_csr = csr_matrix((all_vals, (all_rows, all_cols)), shape=(n_dofs, col_offset))
            grp_S_rows = np.concatenate([S_Blk_prev_rows[sub_id] for sub_id in sub_ids_in_group])
            grp_S_cols = np.concatenate([S_Blk_prev_cols[sub_id] for sub_id in sub_ids_in_group])
            grp_S_vals = np.concatenate([S_Blk_prev_vals[sub_id] for sub_id in sub_ids_in_group])
            S_Blk_local_csr = csr_matrix((grp_S_vals, (grp_S_rows, grp_S_cols)),
                                         shape=(n_dofs, n_dofs))
            # For a huge whole-interface group (e.g. the single N2=1 group)
            # forming B_Gamma as a dense array and its Cholesky factor costs
            # several GB that can push macOS into SIGBUS territory.  Keep the
            # dense path bit-identical for everything small and switch to a
            # sparse SuperLU path when the dense B_Gamma would exceed ~3 GB.
            _use_sparse = _use_sparse_filter_bgamma(len(s))
            S_Blk_s_csr = None
            if _use_sparse:
                S_Blk_s_csr = csr_matrix(S_Blk_local_csr[s][:, s])
                B_Gamma = S_Blk_s_csr
            else:
                B_Gamma = S_Blk_local_csr[np.ix_(s, s)].toarray()
                np.add(B_Gamma, B_Gamma.T, out=B_Gamma)
                B_Gamma *= 0.5
            A_QG_np_s = A_QG_local_csr[s].toarray()
            # release the scratch used to build the level's CSR matrices
            del A_QG_local_csr, all_rows, all_cols, all_vals
            del grp_S_rows, grp_S_cols, grp_S_vals
            local_A_QG_rows.clear(); local_A_QG_cols.clear(); local_A_QG_vals.clear()
            gc.collect()
            n_eigen_old = A_QG_np_s.shape[1] if A_QG_np_s.size > 0 else 0
            D_inv_local = None
            if n_eigen_old == 0 or len(s) == 0:
                hat_U_s = np.array([]).reshape(len(s), 0)
                n_selected = 0
                D_inv_s = np.array([])
                flag[coarse_id] = 0
            else:
                D_inv_local = np.concatenate([D_inv_G_prev[sub_id] for sub_id in sub_ids_in_group])
                hat_U_s, n_selected, D_inv_s = self.section4_filter(
                    A_QG_np_s, B_Gamma, D_inv_local, s, eta, use_invD,
                    level_name, coarse_id, sparse=_use_sparse)
                flag[coarse_id] = n_selected
            del B_Gamma, A_QG_np_s
            del D_inv_local
            gc.collect()
            A_QG_data[coarse_id] = {'m': s, 'A1Q': hat_U_s, 'n_eigen': n_selected}
            D_inv_G[coarse_id] = D_inv_s
            if _use_sparse:
                S_Blk_s_csr = S_Blk_s_csr.tocsr()
            else:
                S_Blk_s = S_Blk_local_csr[s][:, s]
                S_Blk_s_csr = csr_matrix(S_Blk_s)
                del S_Blk_s
            indptr, indices, values = S_Blk_s_csr.indptr, S_Blk_s_csr.indices, S_Blk_s_csr.data
            row_counts = np.diff(indptr)
            row_indices_local = np.repeat(np.arange(len(row_counts)), row_counts)
            S_Blk_rows[coarse_id] = s[row_indices_local]
            S_Blk_cols[coarse_id] = s[indices]
            S_Blk_vals[coarse_id] = values
            del S_Blk_local_csr, S_Blk_s_csr
            gc.collect()
        return A_QG_data, flag, D_inv_G, S_Blk_rows, S_Blk_cols, S_Blk_vals

    # ------------------------------------------------------------------
    # hybrid Section-3 + Section-4 subdomain level (run_table5c)
    # ------------------------------------------------------------------

    def compute_subdomain_level_hybrid(self, S_Blk_prev_rows, S_Blk_prev_cols, S_Blk_prev_vals,
                                       A_QG_prev_data, D_inv_prev,
                                       subdomain_all_dofs, Gamma_list, Interior_list,
                                       Subd_V_list, Subd_E_list, TS_Blk_coarse_csr,
                                       n_coarse_subdomains, group_assignment_coarse,
                                       mu_val, eta_val, level_name=''):
        Local_Solver = [None] * n_coarse_subdomains
        A_QG_data = [None] * n_coarse_subdomains
        flag = np.zeros(n_coarse_subdomains, dtype=np.int32)
        R0T_all_rows, R0T_all_cols, R0T_all_data = [], [], []
        D_inv_L = [None] * n_coarse_subdomains
        S_Blk_rows = [None] * n_coarse_subdomains
        S_Blk_cols = [None] * n_coarse_subdomains
        S_Blk_vals = [None] * n_coarse_subdomains
        n_dofs = self.n_dofs_total
        for coarse_id in range(n_coarse_subdomains):
            sub_ids_in_group = np.where(group_assignment_coarse == coarse_id)[0]
            s, m, n = subdomain_all_dofs[coarse_id], Gamma_list[coarse_id], Interior_list[coarse_id]
            grp_S_rows = np.concatenate([S_Blk_prev_rows[sub_id] for sub_id in sub_ids_in_group])
            grp_S_cols = np.concatenate([S_Blk_prev_cols[sub_id] for sub_id in sub_ids_in_group])
            grp_S_vals = np.concatenate([S_Blk_prev_vals[sub_id] for sub_id in sub_ids_in_group])
            S_Blk_local_csr = csr_matrix((grp_S_vals, (grp_S_rows, grp_S_cols)),
                                         shape=(n_dofs, n_dofs))
            S_Blk_local = self._petsc_mat_from_csr(S_Blk_local_csr)
            S_Blk_local.assemble()
            local_A_QG_rows, local_A_QG_cols, local_A_QG_vals = [], [], []
            col_offset = 0
            for sub_id in sub_ids_in_group:
                data = A_QG_prev_data[sub_id]
                m_indices, A1Q, n_eigen = data['m'], data['A1Q'], data['n_eigen']
                local_A_QG_rows.append(np.repeat(m_indices, n_eigen))
                local_A_QG_cols.append(np.tile(np.arange(n_eigen) + col_offset, len(m_indices)))
                local_A_QG_vals.append(A1Q.ravel())
                col_offset += n_eigen
            all_rows, all_cols, all_vals = map(np.concatenate,
                                               [local_A_QG_rows, local_A_QG_cols, local_A_QG_vals])
            A_QG_local_csr = csr_matrix((all_vals, (all_rows, all_cols)),
                                        shape=(n_dofs, col_offset))
            A_QG_local = self._petsc_mat_from_csr(A_QG_local_csr)
            A_QG_local.assemble()
            D_local_vec = np.concatenate([D_inv_prev[sub_id] for sub_id in sub_ids_in_group])

            if eta_val is not None and col_offset > 0 and len(s) > 0:
                B_Gamma = S_Blk_local_csr[np.ix_(s, s)].toarray()
                np.add(B_Gamma, B_Gamma.T, out=B_Gamma)
                B_Gamma *= 0.5
                A_QG_np_s = A_QG_local_csr[s].toarray()
                if A_QG_np_s.size > 0 and A_QG_np_s.shape[1] > 0:
                    use_invD = True
                    hat_U_s, n_selected, _ = self.section4_filter(
                        A_QG_np_s, B_Gamma, D_local_vec, s, eta_val, use_invD,
                        level_name, coarse_id)
                    if n_selected > 0:
                        new_rows = np.repeat(s, n_selected)
                        new_cols = np.tile(np.arange(n_selected), len(s))
                        new_vals = hat_U_s.ravel().astype(PETSc.ScalarType)
                        A_QG_local_csr = csr_matrix((new_vals, (new_rows, new_cols)),
                                                    shape=(n_dofs, n_selected))
                        A_QG_local = self._petsc_mat_from_csr(A_QG_local_csr)
                        A_QG_local.assemble()
                        col_offset = n_selected
                        D_local_vec = np.ones(n_selected, dtype=PETSc.ScalarType)
                    else:
                        A_QG_local_csr = csr_matrix(([], ([], [])), shape=(n_dofs, 0))
                        A_QG_local = self._petsc_mat_from_csr(A_QG_local_csr)
                        A_QG_local.assemble()
                        col_offset = 0
                        D_local_vec = np.array([], dtype=PETSc.ScalarType)

            indices = np.arange(max(len(D_local_vec), 1), dtype=np.int32)
            if len(D_local_vec) > 0:
                diag_csr = csr_matrix((D_local_vec, (indices[:len(D_local_vec)], indices[:len(D_local_vec)])),
                                      shape=(len(D_local_vec), len(D_local_vec)))
                InvD_G_local = self._petsc_mat_from_csr(diag_csr)
                InvD_G_local.assemble()
                invD_local_vec = 1.0 / D_local_vec
                invD_G_local_csr = csr_matrix(
                    (invD_local_vec, (indices[:len(D_local_vec)], indices[:len(D_local_vec)])),
                    shape=(len(D_local_vec), len(D_local_vec)))
            else:
                InvD_G_local = None
                invD_G_local_csr = csr_matrix(([], ([], [])), shape=(0, 0))

            if len(s) == 0 or len(n) == 0 or len(m) == 0:
                A_QG_data[coarse_id] = {'m': m, 'A1Q': np.array([]).reshape(len(m), 0), 'n_eigen': 0}
                D_inv_L[coarse_id] = np.array([])
                flag[coarse_id] = 0
                S_Blk_rows[coarse_id] = np.array([], dtype=np.int32)
                S_Blk_cols[coarse_id] = np.array([], dtype=np.int32)
                S_Blk_vals[coarse_id] = np.array([], dtype=PETSc.ScalarType)
                if InvD_G_local is not None:
                    InvD_G_local.destroy()
                S_Blk_local.destroy()
                A_QG_local.destroy()
                continue

            rows, cols, vals = [], [], []
            s_set = set(s)
            s_to_local = np.full(n_dofs, -1, dtype=np.int32)
            s_to_local[s] = np.arange(len(s))
            for entity_list in [Subd_V_list, Subd_E_list]:
                for entity in entity_list:
                    entity = np.atleast_1d(entity).astype(PETSc.IntType)
                    if not (set(entity) & s_set):
                        continue
                    is_entity = PETSc.IS().createGeneral(entity, comm=PETSc.COMM_WORLD)
                    S_sub = S_Blk_local.createSubMatrix(is_entity, is_entity)
                    S_sub.assemble()
                    I_s, J_s, V_s = S_sub.getValuesCSR()
                    S_dense = csr_matrix((V_s, J_s, I_s), shape=S_sub.getSize()).toarray()
                    inv = np.linalg.inv(S_dense)
                    n_entity = len(entity)
                    entity_local = s_to_local[entity]
                    rows.extend(np.repeat(entity_local, n_entity))
                    cols.extend(np.tile(entity_local, n_entity))
                    vals.extend(inv.ravel())
                    is_entity.destroy()
                    S_sub.destroy()
            InvS_Blk_s = csr_matrix((vals, (rows, cols)), shape=(len(s), len(s)))
            InvS_Blk_PETSc = self._petsc_mat_from_csr(InvS_Blk_s)
            InvS_Blk_PETSc.assemble()
            is_n = PETSc.IS().createGeneral(n.astype(PETSc.IntType), comm=PETSc.COMM_WORLD)
            is_n_in_s = s_to_local[n]
            is_n_petsc = PETSc.IS().createGeneral(is_n_in_s.astype(PETSc.IntType),
                                                  comm=PETSc.COMM_WORLD)
            InvS_Blk_local = InvS_Blk_PETSc.createSubMatrix(is_n_petsc, is_n_petsc)
            InvS_Blk_local.assemble()
            is_n_petsc.destroy()

            if col_offset == 0:
                Local_Solver[coarse_id] = InvS_Blk_local
                S_mm = S_Blk_local_csr[m][:, m].toarray()
                S_mm = (S_mm + S_mm.T) / 2
                (S_Blk_rows[coarse_id], S_Blk_cols[coarse_id], S_Blk_vals[coarse_id],
                 A_QG_data[coarse_id], D_inv_L[coarse_id], flag[coarse_id]) = \
                    self._compute_eigenproblem_from_schur(S_mm, TS_Blk_coarse_csr, m, mu_val)
                InvS_Blk_PETSc.destroy()
                InvS_Blk_local.destroy()
                is_n.destroy()
                S_Blk_local.destroy()
                A_QG_local.destroy()
                if InvD_G_local is not None:
                    InvD_G_local.destroy()
                continue

            U = A_QG_local.createSubMatrix(is_n, None)
            U.assemble()
            W_mat = InvS_Blk_local * U
            Global_Solver = InvD_G_local - U.transposeMatMult(W_mat)
            WT = W_mat.copy().transpose()
            WT.assemble()
            ksp = PETSc.KSP().create()
            ksp.setType('preonly')
            ksp.getPC().setType('lu')
            ksp.getPC().setFactorSolverType('mumps')
            ksp.setOperators(Global_Solver)
            ksp.setUp()
            n_rows, n_cols = WT.getSize()
            solutions = np.zeros((n_rows, n_cols), dtype=PETSc.ScalarType)
            x_temp = PETSc.Vec().createSeq(n_rows)
            for j in range(n_cols):
                b = WT.getColumnVector(j)
                ksp.solve(b, x_temp)
                solutions[:, j] = x_temp.getArray()
                b.destroy()
            x_temp.destroy()
            X_data = solutions.ravel()
            X_csr = csr_matrix((X_data, (np.repeat(np.arange(n_rows), n_cols),
                                         np.tile(np.arange(n_cols), n_rows))),
                               shape=(n_rows, n_cols))
            X = self._petsc_mat_from_csr(X_csr)
            X.assemble()
            yy = InvS_Blk_local + W_mat * X
            Local_Solver[coarse_id] = yy
            S_mm = S_Blk_local_csr[m][:, m]
            S_mn = S_Blk_local_csr[m][:, n]
            A_QG_m = A_QG_local_csr[m]
            A_QG_n = A_QG_local_csr[n]
            A1 = S_mm - A_QG_m @ invD_G_local_csr @ A_QG_m.T
            A2 = S_mn - A_QG_m @ invD_G_local_csr @ A_QG_n.T
            A3 = A2.T
            indptr_yy, indices_yy, values_yy = yy.getValuesCSR()
            yy_csr = csr_matrix((values_yy, indices_yy, indptr_yy), shape=yy.getSize())
            yy_A3 = yy_csr @ A3
            yy_A3_mat = self._petsc_mat_from_csr(yy_A3)
            indptr, indices, values = yy_A3_mat.getValuesCSR()
            row_indices_local = np.repeat(np.arange(len(np.diff(indptr))), np.diff(indptr))
            R0T_all_rows.append(n[row_indices_local])
            R0T_all_cols.append(m[indices])
            R0T_all_data.append(-values)
            Schur_np = A1 - A2 @ yy_A3
            Schur_np = ((Schur_np + Schur_np.T) / 2).toarray()
            (S_Blk_rows[coarse_id], S_Blk_cols[coarse_id], S_Blk_vals[coarse_id],
             A_QG_data[coarse_id], D_inv_L[coarse_id], flag[coarse_id]) = \
                self._compute_eigenproblem_from_schur(Schur_np, TS_Blk_coarse_csr, m, mu_val)
            yy_A3_mat.destroy()
            S_Blk_local.destroy()
            A_QG_local.destroy()
            InvD_G_local.destroy()
            InvS_Blk_PETSc.destroy()
            InvS_Blk_local.destroy()
            U.destroy()
            W_mat.destroy()
            WT.destroy()
            Global_Solver.destroy()
            X.destroy()
            is_n.destroy()
            ksp.destroy()
        return (Local_Solver, A_QG_data, flag, R0T_all_rows, R0T_all_cols, R0T_all_data,
                D_inv_L, S_Blk_rows, S_Blk_cols, S_Blk_vals)

    # ------------------------------------------------------------------
    # PCG (flat, filtering) and recursive PCG (traditional / hybrid)
    # ------------------------------------------------------------------

    def pcg_nosas_flat(self, A, b, max_it, tol, InvS_Blk, W, ksp_global):
        x = b.copy(); x.set(0.0)
        r = b.copy()
        bnrm2 = b.norm(PETSc.NormType.NORM_2)
        error = r.norm(PETSc.NormType.NORM_2) / bnrm2
        print(f' PCG residual(0) = {error}')
        alpha = np.zeros(max_it); beta = np.zeros(max_it)
        iter_count = -1
        for iter_count in range(max_it):
            z = r.copy()
            InvS_Blk.mult(r, z)
            _, W_n = W.getSize()
            WTr = PETSc.Vec().createWithArray(np.zeros(W_n))
            W.multTranspose(r, WTr)
            solution = WTr.copy()
            ksp_global.solve(WTr, solution)
            temp = z.copy(); temp.set(0.0)
            W.mult(solution, temp)
            z.axpy(1.0, temp)
            rho = r.dot(z)
            if iter_count > 0:
                if rho == 0 or rho_1 == 0:
                    print(f"  rho or rho_1 is zero at iter {iter_count}, aborting")
                    break
                beta[iter_count] = rho / rho_1
                p.scale(beta[iter_count]); p.axpy(1.0, z)
            else:
                p = z.copy()
            q = p.copy()
            A.mult(p, q)
            p_dot_q = p.dot(q)
            if p_dot_q == 0:
                print(f"  p.dot(q) = 0 at iter {iter_count}, aborting")
                break
            alpha[iter_count] = rho / p_dot_q
            x.axpy(alpha[iter_count], p)
            r.axpy(-alpha[iter_count], q)
            error = r.norm(PETSc.NormType.NORM_2) / bnrm2
            print(f' PCG residual({iter_count+1}) = {error:e}')
            if error <= tol:
                break
            rho_1 = rho
        flag = 0 if error <= tol else 1
        lambdamax = lambdamin = condnumber = float('nan')
        if iter_count >= 0 and np.all(beta[:iter_count+1] >= 0):
            d = np.zeros(iter_count+1); s = np.zeros(iter_count)
            d[0] = 1.0 / alpha[0] if alpha[0] != 0 else 0.0
            for i in range(iter_count):
                d[i+1] = beta[i+1]/alpha[i] + 1.0/alpha[i+1]
                s[i] = -np.sqrt(beta[i+1])/alpha[i]
            T = np.zeros((iter_count+1, iter_count+1))
            T[iter_count, iter_count] = d[iter_count]
            for i in range(iter_count):
                T[i, i] = d[i]; T[i, i+1] = s[i]; T[i+1, i] = s[i]
            if np.all(np.isfinite(T)):
                lambda_vals = np.linalg.eigvals(T)
                lambdamax = np.max(lambda_vals)
                lambdamin = np.min(lambda_vals)
                condnumber = lambdamax / lambdamin
        if flag == 1:
            print(f'PCG did NOT converge, residual {error:e}')
        else:
            print(f'PCG {iter_count + 1} iters, residual {error:e}, '
                  f'lambda_max={lambdamax:g}, lambda_min={lambdamin:g}, cond={condnumber:g}')
        return x, error, iter_count + 1, flag, lambdamax, lambdamin, condnumber

    def _nosas_preconditioner_recursive(self, r, z, ksp_list, Local_Solver_list, InvS_Blk_list,
                                        W_list, R_0T_list, is_GG_list, is_II_list,
                                        Interior_list, level_idx, n_dofs_total):
        current_ksp = ksp_list[level_idx]
        current_Local_Solver = Local_Solver_list[level_idx]
        current_InvS_Blk = InvS_Blk_list[level_idx]
        current_W = W_list[level_idx]
        current_R_0T = R_0T_list[level_idx]
        current_is_GG = is_GG_list[level_idx + 1]
        current_is_II = is_II_list[level_idx]
        current_Interior = Interior_list[level_idx]

        next_level_idx = level_idx + 1
        has_next_level = (next_level_idx < len(ksp_list))

        r_new = PETSc.Vec().createMPI(n_dofs_total, comm=PETSc.COMM_WORLD)
        r_new.set(0.0)
        r_new.setValues(is_GG_list[level_idx], r.getArray())
        r_new.assemble()

        b = PETSc.Vec().createWithArray(r_new.getValues(current_is_GG))
        r_II = PETSc.Vec().createWithArray(r_new.getValues(current_is_II))

        b_temp = PETSc.Vec().createMPI(b.getSize(), comm=PETSc.COMM_WORLD)
        if isinstance(current_R_0T, dict) and current_R_0T.get('kind') == 'blocks':
            self._r0t_block_multT(current_R_0T, r_II.getArray(), b_temp.getArray())
        else:
            current_R_0T.multTranspose(r_II, b_temp)
        b.axpy(1.0, b_temp)

        r_II.destroy()
        b_temp.destroy()

        if has_next_level:
            w_g = PETSc.Vec().createMPI(b.getSize(), comm=PETSc.COMM_WORLD)
            self._nosas_preconditioner_recursive(
                b, w_g, ksp_list, Local_Solver_list, InvS_Blk_list, W_list,
                R_0T_list, is_GG_list, is_II_list, Interior_list,
                next_level_idx, n_dofs_total)
        else:
            W_T_b = PETSc.Vec().createMPI(current_W.getSize()[1], comm=PETSc.COMM_WORLD)
            current_W.multTranspose(b, W_T_b)
            current_ksp.solve(W_T_b, W_T_b)
            W_w_temp = current_W * W_T_b
            InvS_Blk_b = current_InvS_Blk * b
            w_g = InvS_Blk_b.copy()
            w_g.axpy(1.0, W_w_temp)
            w_g.assemble()
            W_T_b.destroy()
            W_w_temp.destroy()
            InvS_Blk_b.destroy()

        U_gamma = PETSc.Vec().createMPI(n_dofs_total, comm=PETSc.COMM_WORLD)
        U_gamma.set(0.0)
        U_gamma.setValues(current_is_GG, w_g)
        if isinstance(current_R_0T, dict) and current_R_0T.get('kind') == 'blocks':
            _ui = np.empty(current_R_0T['n_II'], dtype=PETSc.ScalarType)
            self._r0t_block_mult(current_R_0T, w_g.getArray(), _ui)
            U_gamma_II = PETSc.Vec().createWithArray(_ui)
        else:
            U_gamma_II = current_R_0T * w_g
        U_gamma.setValues(current_is_II, U_gamma_II)
        U_gamma.assemble()
        w_g.destroy()

        U_I = PETSc.Vec().createMPI(n_dofs_total, comm=PETSc.COMM_WORLD)
        U_I.set(0.0)
        for i in range(len(current_Interior)):
            nodes = current_Interior[i]
            yy = current_Local_Solver[i]
            is_n = PETSc.IS().createGeneral(nodes.astype(PETSc.IntType), comm=PETSc.COMM_WORLD)
            r_n = PETSc.Vec().createWithArray(r_new.getValues(is_n))
            if isinstance(yy, PETSc.KSP):
                U_I_n = r_n.copy()
                yy.solve(r_n, U_I_n)
            else:
                U_I_n = yy * r_n
            U_I.setValues(is_n, U_I_n)
            is_n.destroy()
            r_n.destroy()
            U_I_n.destroy()
        U_I.assemble()

        U_gamma.axpy(1.0, U_I)

        z.setArray(U_gamma.getValues(is_GG_list[level_idx]))
        z.assemble()

        r_new.destroy()
        b.destroy()
        U_gamma_II.destroy()
        U_gamma.destroy()
        U_I.destroy()

    def pcg_nosas_multilevel(self, A, b, max_it, tol, ksp_list, Local_Solver_list,
                             InvS_Blk_list, W_list, R_0T_list, is_GG_list, is_II_list,
                             Interior_list, n_dofs_total):
        x = b.copy(); x.set(0.0)
        r = b.copy()
        bnrm2 = b.norm(PETSc.NormType.NORM_2)
        error = r.norm(PETSc.NormType.NORM_2) / bnrm2
        print(f' PCG residual(0) = {error}')
        alpha, beta = np.zeros(max_it), np.zeros(max_it)
        iter_count = -1
        for iter_count in range(max_it):
            z = r.copy()
            self._nosas_preconditioner_recursive(
                r, z, ksp_list, Local_Solver_list, InvS_Blk_list, W_list,
                R_0T_list, is_GG_list, is_II_list, Interior_list, 0, n_dofs_total)
            rho = r.dot(z)
            if iter_count > 0:
                beta[iter_count] = rho / rho_1
                p.scale(beta[iter_count])
                p.axpy(1.0, z)
            else:
                p = z.copy()
            q = p.copy()
            A.mult(p, q)
            alpha[iter_count] = rho / p.dot(q)
            x.axpy(alpha[iter_count], p)
            r.axpy(-alpha[iter_count], q)
            error = r.norm(PETSc.NormType.NORM_2) / bnrm2
            print(f' PCG residual({iter_count + 1}) = {error:e}')
            if error <= tol:
                break
            rho_1 = rho
        flag = 0 if error <= tol else 1
        lambdamax = lambdamin = condnumber = 0.0
        if flag == 0 and iter_count >= 0:
            d = np.zeros(iter_count + 1)
            s = np.zeros(iter_count)
            d[0] = 1.0 / alpha[0]
            for i in range(iter_count):
                d[i + 1] = beta[i + 1] / alpha[i] + 1.0 / alpha[i + 1]
                s[i] = -np.sqrt(beta[i + 1]) / alpha[i]
            T = np.zeros((iter_count + 1, iter_count + 1))
            T[iter_count, iter_count] = d[iter_count]
            for i in range(iter_count):
                T[i, i] = d[i]
                T[i, i + 1] = s[i]
                T[i + 1, i] = s[i]
            lambda_vals = np.linalg.eigvals(T)
            lambdamax, lambdamin = np.max(lambda_vals), np.min(lambda_vals)
            condnumber = lambdamax / lambdamin
            print(f'PCG converged in {iter_count + 1} iters, residual {error:e}, '
                  f'lambda_max={lambdamax:g}, lambda_min={lambdamin:g}, cond={condnumber:g}')
        elif flag == 0:
            print(f'PCG converged in {iter_count + 1} iters, residual {error:e}')
        else:
            print(f'PCG did NOT converge in {iter_count + 1} iters, residual {error:e}')
        return x, error, iter_count + 1, flag, lambdamax, lambdamin, condnumber

    # ------------------------------------------------------------------
    # top-level path runners
    # ------------------------------------------------------------------

    def _two_level_assembly(self, final_dinv='true'):
        """run_table2 2-level path (recursive PCG with one coarse level)."""
        n = self.n_dofs_total
        is_GG0_petsc = PETSc.IS().createGeneral(self.gg.astype(PETSc.IntType),
                                                comm=PETSc.COMM_WORLD)
        if self.R0T_blocks is not None:
            R_0T = self._build_r0t_blocks()
        else:
            R_0T = self._assemble_R0T([self.R0T_level0[0]],
                                      [self.R0T_level0[1]],
                                      [self.R0T_level0[2]],
                                      self.II, self.gg)
        D_inv_G = self.D_inv_G_list if final_dinv == 'true' else None
        _, A_QG_GG, InvS_Blk_csr, W_csr, gs_csr = self._build_global_from_level(
            self.A_QG_data_list, self.S_Blk_rows_list, self.S_Blk_cols_list,
            self.S_Blk_vals_list, self.Subd_V, self.Subd_E, self.gg, D_inv_G)
        InvS_Blk = self._petsc_mat_from_csr(InvS_Blk_csr); InvS_Blk.assemble()
        W = self._petsc_mat_from_csr(W_csr); W.assemble()
        Global_Solver = self._petsc_mat_from_csr(gs_csr); Global_Solver.assemble()
        all_R_0T = [R_0T]
        all_InvS_Blk = [InvS_Blk]
        all_W = [W]
        all_Global_Solver = [Global_Solver]
        all_is_II = [PETSc.IS().createGeneral(self.II.astype(PETSc.IntType),
                                              comm=PETSc.COMM_WORLD)]
        all_is_GG = [is_GG0_petsc]
        all_Interior = [self.Interior]
        all_Local_Solver = [self.local_solvers]
        return (all_Global_Solver, all_Local_Solver, all_InvS_Blk, all_W, all_R_0T,
                all_is_II, all_is_GG, all_Interior)

    def run_traditional(self, max_it=200, tol=1e-6, final_dinv='true'):
        c = self.cfg
        n_levels = len(c['n_subdomains_list'])
        n_coarse_levels = n_levels - 1
        GG = self.gg
        S = self._petsc_mat_from_csr(self.S_csr_GG); S.assemble()
        g = PETSc.Vec().createWithArray(self.g_arr.copy())

        if n_coarse_levels > 0:
            (all_Global_Solver, all_Local_Solver, all_InvS_Blk, all_W, all_R_0T,
             all_is_II, all_is_GG, all_Interior, all_flag_l) = self._run_traditional_multilevel(final_dinv)
            print(f"=== {n_coarse_levels + 2} levels ===")
        else:
            (all_Global_Solver, all_Local_Solver, all_InvS_Blk, all_W, all_R_0T,
             all_is_II, all_is_GG, all_Interior) = self._two_level_assembly(final_dinv)
            all_flag_l = [self.flag1]

        ksp_list = []
        for i in range(len(all_Global_Solver)):
            ksp_list.append(self._make_global_solver(all_Global_Solver[i]))

        if n_coarse_levels == 0:
            Local_Solver_list = all_Local_Solver
        else:
            Local_Solver_list = all_Local_Solver[1:]
        InvS_Blk_list = all_InvS_Blk
        W_list = all_W
        R_0T_list = all_R_0T
        is_GG_list = [PETSc.IS().createGeneral(GG.astype(PETSc.IntType), comm=PETSc.COMM_WORLD)] + all_is_GG
        is_II_list = all_is_II
        Interior_list = all_Interior

        x, error, iters, flag, lmax, lmin, cond = self.pcg_nosas_multilevel(
            A=S, b=g, max_it=max_it, tol=tol, ksp_list=ksp_list,
            Local_Solver_list=Local_Solver_list, InvS_Blk_list=InvS_Blk_list,
            W_list=W_list, R_0T_list=R_0T_list, is_GG_list=is_GG_list,
            is_II_list=is_II_list, Interior_list=Interior_list, n_dofs_total=self.n_dofs_total)
        for k in ksp_list:
            k.destroy()
        return {'GG': len(GG), 'iters': iters, 'error': error, 'flag': flag,
                'lmax': lmax, 'lmin': lmin, 'cond': cond, 'N_s0': int(np.sum(self.flag1)),
                'flag_sums': [int(np.sum(f)) for f in all_flag_l]}

    def _run_traditional_multilevel(self, final_dinv='true'):
        c = self.cfg
        n_levels = len(c['n_subdomains_list'])
        n_coarse_levels = n_levels - 1
        self.build_coarsening_chain()
        prev_S_Blk_rows, prev_S_Blk_cols, prev_S_Blk_vals = \
            self.S_Blk_rows_list, self.S_Blk_cols_list, self.S_Blk_vals_list
        prev_A_QG_data, prev_InvD_G = self.A_QG_data_list, self.D_inv_G_list
        prev_Subd_V, prev_Subd_E = self.Subd_V, self.Subd_E
        all_Local_Solver, all_flag = [self.local_solvers], [self.flag1]
        all_R_0T, all_InvS_Blk, all_W, all_Global_Solver = [], [], [], []
        all_is_II, all_is_GG, all_Interior = [], [], []
        for level_idx in range(n_coarse_levels):
            data = self.coarse_levels_data[level_idx]
            actual_n = data['actual_n']
            (Local_Solver_coarse, A_QG_data_coarse, flag_coarse,
             R0T_all_rows, R0T_all_cols, R0T_all_data,
             D_inv_coarse, S_Blk_rows_coarse, S_Blk_cols_coarse, S_Blk_vals_coarse) = \
                self.compute_subdomain_level(
                    prev_S_Blk_rows, prev_S_Blk_cols, prev_S_Blk_vals,
                    prev_A_QG_data, prev_InvD_G,
                    data['subdomain_all_dofs'], data['Gamma'], data['Interior'],
                    prev_Subd_V, prev_Subd_E, data['TS_Blk_csr'],
                    actual_n, data['group_assignment'], data['mu'])
            all_Local_Solver.append(Local_Solver_coarse)
            all_flag.append(flag_coarse)
            prev_S_Blk_rows, prev_S_Blk_cols, prev_S_Blk_vals = \
                S_Blk_rows_coarse, S_Blk_cols_coarse, S_Blk_vals_coarse
            prev_A_QG_data, prev_InvD_G = A_QG_data_coarse, D_inv_coarse
            prev_Subd_V, prev_Subd_E = data['Subd_V'], data['Subd_E']

            # assemble the level's coarse solver tower for the recursion
            R_0T = self._assemble_R0T(R0T_all_rows, R0T_all_cols, R0T_all_data,
                                      data['II'], data['GG'])
            D_inv_lvl = prev_InvD_G if final_dinv == 'true' else None
            _, _, InvS_Blk_csr, W_csr, gs_csr = self._build_global_from_level(
                prev_A_QG_data, prev_S_Blk_rows, prev_S_Blk_cols, prev_S_Blk_vals,
                data['Subd_V'], data['Subd_E'], data['GG'], D_inv_lvl)
            InvS_Blk = self._petsc_mat_from_csr(InvS_Blk_csr); InvS_Blk.assemble()
            W = self._petsc_mat_from_csr(W_csr); W.assemble()
            Global_Solver = self._petsc_mat_from_csr(gs_csr); Global_Solver.assemble()
            all_R_0T.append(R_0T)
            all_InvS_Blk.append(InvS_Blk)
            all_W.append(W)
            all_Global_Solver.append(Global_Solver)
            all_is_II.append(PETSc.IS().createGeneral(data['II'].astype(PETSc.IntType),
                                                      comm=PETSc.COMM_WORLD))
            all_is_GG.append(PETSc.IS().createGeneral(data['GG'].astype(PETSc.IntType),
                                                      comm=PETSc.COMM_WORLD))
            all_Interior.append(data['Interior'])
        return (all_Global_Solver, all_Local_Solver, all_InvS_Blk, all_W, all_R_0T,
                all_is_II, all_is_GG, all_Interior, all_flag)

    def _release_filter_level0(self):
        """Free level-0 data that the flat filtering path never touches again.

        The level-0 A4 KSPs, the raw per-subdomain results dict, the PETSc
        TS_Blk copy and the R0T data are dead for ``run_filtering``; dropping
        them before the (potentially huge) Section-4 filter keeps the peak
        RSS small enough that the whole-interface dense B_Gamma / Cholesky
        factors stay well inside physical RAM."""
        _ls = getattr(self, 'local_solvers', None)
        if _ls is not None:
            for _k in _ls:
                if _k is not None:
                    try:
                        _k.destroy()
                    except Exception:
                        pass
            self.local_solvers = None
        if getattr(self, 'results', None) is not None:
            self.results = None
        _ts = getattr(self, 'TS_Blk_mat', None)
        if _ts is not None:
            try:
                _ts.destroy()
            except Exception:
                pass
            self.TS_Blk_mat = None
        self.R0T_level0 = None
        self.R0T_blocks = None
        self.inv_cache = None
        gc.collect()

    def run_filtering(self, max_it=200, tol=1e-6):
        c = self.cfg
        n_levels = len(c['n_subdomains_list'])
        n_coarse_levels = n_levels - 1
        GG = self.gg
        S = self._petsc_mat_from_csr(self.S_csr_GG); S.assemble()
        g = PETSc.Vec().createWithArray(self.g_arr.copy())

        all_flag = [self.flag1]
        if n_coarse_levels > 0:
            self.build_coarsening_chain()
            self._release_filter_level0()
            prev_S_Blk_rows, prev_S_Blk_cols, prev_S_Blk_vals = \
                self.S_Blk_rows_list, self.S_Blk_cols_list, self.S_Blk_vals_list
            prev_A_QG_data, prev_D_inv_G = self.A_QG_data_list, self.D_inv_G_list
            prev_Subd_V, prev_Subd_E = self.Subd_V, self.Subd_E
            n_section4_levels = n_coarse_levels
            for level_idx in range(n_section4_levels):
                data = self.coarse_levels_data[level_idx]
                actual_n = data['actual_n']
                use_invD = (level_idx == 0)
                eta_val = c['eta_list'][level_idx] if level_idx < len(c['eta_list']) else c['eta_list'][-1]
                level_name = f"Level {level_idx + 2}"
                A_QG_data_coarse, flag_coarse, D_inv_G_coarse, \
                    S_Blk_rows_coarse, S_Blk_cols_coarse, S_Blk_vals_coarse = \
                    self.compute_level_section4(
                        prev_S_Blk_rows, prev_S_Blk_cols, prev_S_Blk_vals,
                        prev_A_QG_data, prev_D_inv_G,
                        data['subdomain_all_dofs'], data['Gamma'], data['Interior'],
                        prev_Subd_V, prev_Subd_E,
                        actual_n, data['group_assignment'],
                        eta_val, use_invD, level_name)
                all_flag.append(flag_coarse)
                prev_S_Blk_rows, prev_S_Blk_cols, prev_S_Blk_vals = \
                    S_Blk_rows_coarse, S_Blk_cols_coarse, S_Blk_vals_coarse
                prev_A_QG_data = A_QG_data_coarse
                prev_D_inv_G = D_inv_G_coarse
                prev_Subd_V, prev_Subd_E = data['Subd_V'], data['Subd_E']
            _, _, InvS_Blk_csr, W_csr, gs_csr = self._build_global_from_level(
                prev_A_QG_data, prev_S_Blk_rows, prev_S_Blk_cols, prev_S_Blk_vals,
                Subd_V=self.Subd_V, Subd_E=self.Subd_E, coarse_GG=GG, D_inv_G=None)
        else:
            _, _, InvS_Blk_csr, W_csr, gs_csr = self._build_global_from_level(
                self.A_QG_data_list, self.S_Blk_rows_list, self.S_Blk_cols_list,
                self.S_Blk_vals_list, Subd_V=self.Subd_V, Subd_E=self.Subd_E,
                coarse_GG=GG, D_inv_G=self.D_inv_G_list)

        InvS_Blk = self._petsc_mat_from_csr(InvS_Blk_csr); InvS_Blk.assemble()
        W = self._petsc_mat_from_csr(W_csr); W.assemble()
        Global_Solver = self._petsc_mat_from_csr(gs_csr); Global_Solver.assemble()
        ksp_global = self._make_global_solver(Global_Solver)

        x, error, iters, flag, lmax, lmin, cond = self.pcg_nosas_flat(
            A=S, b=g, max_it=max_it, tol=tol,
            InvS_Blk=InvS_Blk, W=W, ksp_global=ksp_global)
        ksp_global.destroy()
        return {'GG': len(GG), 'iters': iters, 'error': error, 'flag': flag,
                'lmax': lmax, 'lmin': lmin, 'cond': cond, 'N_s0': int(np.sum(self.flag1)),
                'flag_sums': [int(np.sum(f)) for f in all_flag]}

    def run_hybrid(self, max_it=200, tol=1e-6, final_dinv='true'):
        c = self.cfg
        n_levels = len(c['n_subdomains_list'])
        n_coarse_levels = n_levels - 1
        GG = self.gg
        S = self._petsc_mat_from_csr(self.S_csr_GG); S.assemble()
        g = PETSc.Vec().createWithArray(self.g_arr.copy())

        self.build_coarsening_chain()
        prev_S_Blk_rows, prev_S_Blk_cols, prev_S_Blk_vals = \
            self.S_Blk_rows_list, self.S_Blk_cols_list, self.S_Blk_vals_list
        prev_A_QG_data, prev_InvD_G = self.A_QG_data_list, self.D_inv_G_list
        prev_Subd_V, prev_Subd_E = self.Subd_V, self.Subd_E
        all_Local_Solver, all_flag = [self.local_solvers], [self.flag1]
        all_R_0T, all_InvS_Blk, all_W, all_Global_Solver = [], [], [], []
        all_is_II, all_is_GG, all_Interior = [], [], []
        print(f"=== {n_coarse_levels + 2} levels (Hybrid Multilevel) ===")
        for level_idx in range(n_coarse_levels):
            data = self.coarse_levels_data[level_idx]
            actual_n = data['actual_n']
            eta_val = c['eta_list'][level_idx] if level_idx < len(c['eta_list']) else c['eta_list'][-1]
            (Local_Solver_coarse, A_QG_data_coarse, flag_coarse,
             R0T_all_rows, R0T_all_cols, R0T_all_data,
             D_inv_coarse, S_Blk_rows_coarse, S_Blk_cols_coarse, S_Blk_vals_coarse) = \
                self.compute_subdomain_level_hybrid(
                    prev_S_Blk_rows, prev_S_Blk_cols, prev_S_Blk_vals,
                    prev_A_QG_data, prev_InvD_G,
                    data['subdomain_all_dofs'], data['Gamma'], data['Interior'],
                    prev_Subd_V, prev_Subd_E, data['TS_Blk_csr'],
                    actual_n, data['group_assignment'], data['mu'], eta_val,
                    f"Level {level_idx + 2}")
            all_Local_Solver.append(Local_Solver_coarse)
            all_flag.append(flag_coarse)
            prev_S_Blk_rows, prev_S_Blk_cols, prev_S_Blk_vals = \
                S_Blk_rows_coarse, S_Blk_cols_coarse, S_Blk_vals_coarse
            prev_A_QG_data, prev_InvD_G = A_QG_data_coarse, D_inv_coarse
            prev_Subd_V, prev_Subd_E = data['Subd_V'], data['Subd_E']
            R_0T = self._assemble_R0T(R0T_all_rows, R0T_all_cols, R0T_all_data,
                                      data['II'], data['GG'])
            D_inv_lvl = prev_InvD_G if final_dinv == 'true' else None
            _, _, InvS_Blk_csr, W_csr, gs_csr = self._build_global_from_level(
                prev_A_QG_data, prev_S_Blk_rows, prev_S_Blk_cols, prev_S_Blk_vals,
                data['Subd_V'], data['Subd_E'], data['GG'], D_inv_lvl)
            InvS_Blk = self._petsc_mat_from_csr(InvS_Blk_csr); InvS_Blk.assemble()
            W = self._petsc_mat_from_csr(W_csr); W.assemble()
            Global_Solver = self._petsc_mat_from_csr(gs_csr); Global_Solver.assemble()
            all_R_0T.append(R_0T)
            all_InvS_Blk.append(InvS_Blk)
            all_W.append(W)
            all_Global_Solver.append(Global_Solver)
            all_is_II.append(PETSc.IS().createGeneral(data['II'].astype(PETSc.IntType),
                                                      comm=PETSc.COMM_WORLD))
            all_is_GG.append(PETSc.IS().createGeneral(data['GG'].astype(PETSc.IntType),
                                                      comm=PETSc.COMM_WORLD))
            all_Interior.append(data['Interior'])

        ksp_list = []
        for i in range(len(all_Global_Solver)):
            ksp_list.append(self._make_global_solver(all_Global_Solver[i]))

        if n_coarse_levels == 0:
            Local_Solver_list = all_Local_Solver
        else:
            Local_Solver_list = all_Local_Solver[1:]
        InvS_Blk_list = all_InvS_Blk
        W_list = all_W
        R_0T_list = all_R_0T
        is_GG_list = [PETSc.IS().createGeneral(GG.astype(PETSc.IntType), comm=PETSc.COMM_WORLD)] + all_is_GG
        is_II_list = all_is_II
        Interior_list = all_Interior

        x, error, iters, flag, lmax, lmin, cond = self.pcg_nosas_multilevel(
            A=S, b=g, max_it=max_it, tol=tol, ksp_list=ksp_list,
            Local_Solver_list=Local_Solver_list, InvS_Blk_list=InvS_Blk_list,
            W_list=W_list, R_0T_list=R_0T_list, is_GG_list=is_GG_list,
            is_II_list=is_II_list, Interior_list=Interior_list,
            n_dofs_total=self.n_dofs_total)
        for k in ksp_list:
            k.destroy()
        return {'GG': len(GG), 'iters': iters, 'error': error, 'flag': flag,
                'lmax': lmax, 'lmin': lmin, 'cond': cond, 'N_s0': int(np.sum(self.flag1)),
                'flag_sums': [int(np.sum(f)) for f in all_flag]}

    # ------------------------------------------------------------------
    # dispatcher
    # ------------------------------------------------------------------

    def run(self, max_it=200, tol=1e-6):
        mode = self.cfg.get('mode', 'trad')
        if mode == 'filter':
            return self.run_filtering(max_it=max_it, tol=tol)
        if mode == 'hybrid':
            return self.run_hybrid(max_it=max_it, tol=tol)
        return self.run_traditional(max_it=max_it, tol=tol)


def make_engine(cfg):
    eng = NOSAS2DEngine(cfg)
    eng.preprocess()
    eng.compute_level0(cfg['mu_list'][0])
    eng.select_level0(cfg['mu_list'][0])
    return eng