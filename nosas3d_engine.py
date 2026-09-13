"""Shared optimized NOSAS 3D SPE10 engine (derived from 3d_alg1_clean.py).

Keeps the proven architecture of 3d_alg1_clean.py:
  - numba kernels (_add_block, _csr_rows_extract, _csr_block_gather)
  - per-subdomain implicit Schur (only A1_csr/A2_csr + CHOLMOD A4 factor stored)
  - action-based global Schur (no dense S_GG)
  - per-entity InvS_Blk, lazy W shell, sparse Global_Solver via numba gather
  - CHOLMOD factor for the coarse Global_Solver
  - PCG + Lanczos eigenvalue reporting

Additions to reproduce the ORIGINAL table scripts (run_table6..10.py) exactly:
  - Level-0 full eigh (matching the old scripts, no sygvx subsetting)
  - Level-0 reuse across (mu, eta) combos: submesh/A-blocks/eigh computed once
  - per-level use_invD control (Section-4 levels: genGEP vs stdGEP)
  - final_dinv: 'true' -> coarse Global_Solver uses D_inv (Table 6)
                'ones' -> Global_Solver = I - A_QG^T W (Tables 7, 8)
  - enable_neg_eigen toggles the negative-eigenvalue Section-4 wallet
    (default off for old-tables fidelity)

The engine exposes a small state machine so a driver can:
    eng = NOSAS3DEngine(cfg)
    eng.preprocess()
    eng.compute_subdomains()          # raw Level-0 (once)
    for combo in combos:
        eng.select_level0(mu0)
        metrics = eng.run_multilevel(eta_list, eta2_list, use_invD_levels, final_dinv)
"""
import os
import gc
import sys
import time
import pickle
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import pymetis
from petsc4py import PETSc
import ufl
from scipy.sparse import csr_matrix
from scipy.linalg import eigh, eig
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu
from sksparse.cholmod import cho_factor
from numba import njit

# --------------------------------------------------------------------------
# numba kernels (identical to 3d_alg1_clean.py)
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

class NOSAS3DEngine:

    def __init__(self, cfg):
        self.cfg = cfg
        self.start_time = time.time()
        self.n_dofs_total = None
        self.results = []          # raw Level-0 per-subdomain data
        self.subs = []             # implicit-Schur actions (Level-0)
        self.g_arr = None          # global g on GG
        self.GG = None
        self.flag1 = None

    # ---- helpers ----

    @staticmethod
    def _petsc_to_scipy_dense(petsc_mat):
        I, J, V = petsc_mat.getValuesCSR()
        return csr_matrix((V, J, I), shape=petsc_mat.getSize()).toarray()

    @staticmethod
    def _petsc_to_scipy_csr(petsc_mat):
        I, J, V = petsc_mat.getValuesCSR()
        return csr_matrix((V, J, I), shape=petsc_mat.getSize())

    # ---- preprocessing ----

    def preprocess(self):
        c = self.cfg
        h, n_subdomains = c['h'], c['n_subdomains_list'][0]
        deg, name = c['deg'], c.get('name', 'nosas')
        perm_file = c.get('perm_file', 'spe_perm.dat')
        nx, ny, nz = 60, 220, 85

        _preproc_cache_file = c.get('cache_file', 'preproc_cache.pkl')
        _preproc_key = (h, n_subdomains, tuple(c['n_subdomains_list']), deg,
                        (60, 220, 85),
                        os.path.getmtime(perm_file), os.path.getsize(perm_file))
        _preproc_cache = None
        if os.path.exists(_preproc_cache_file):
            try:
                with open(_preproc_cache_file, 'rb') as _f:
                    _pc = pickle.load(_f)
                if _pc.get('key') == _preproc_key:
                    _preproc_cache = _pc
            except Exception:
                _preproc_cache = None

        if _preproc_cache is not None:
            print("[cache] preprocess cache hit: rebuild mesh + load derived arrays")
            import basix, basix.ufl
            msh = mesh.create_mesh(MPI.COMM_SELF, _preproc_cache['cells'],
                                   basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,)),
                                   _preproc_cache['x'])
            V = fem.functionspace(msh, ("Lagrange", deg))
            dofmap = V.dofmap
            self.dofmap = dofmap
            self.n_dofs_total = dofmap.index_map.size_global
            self.cell_rho_values = _preproc_cache['cell_rho_values']
            self.membership = _preproc_cache['membership']
            self.subdomain_all_dofs = _preproc_cache['subdomain_all_dofs']
            self.dofs_bc = _preproc_cache['dofs_bc']
            self.GG = _preproc_cache['GG']
            self.II = _preproc_cache['II']
            self.Subd_E = _preproc_cache['Subd_E']
            self.Subd_F = _preproc_cache['Subd_F']
            self.entity_list = _preproc_cache['entity_list']
            self.Gamma = _preproc_cache['Gamma']
            self.Interior = _preproc_cache['Interior']
            print(f"[cache] loaded: GG {len(self.GG)}, Subd_F_all {len(_preproc_cache['Subd_F_all'])}, "
                  f"Int {len(self.II)}, entity {len(self.entity_list)} "
                  f"({len(self.Subd_E)} edges + {len(self.Subd_F)} faces), dofs_bc {len(self.dofs_bc)}")
        else:
            print(f"\n=== Creating mesh (h={h}) ===")
            n_cubes_x, n_cubes_y, n_cubes_z = int(60 / h), int(220 / h), int(85 / h)
            msh = mesh.create_box(MPI.COMM_SELF, [np.array([0, 0, 0]), np.array([60, 220, 85])],
                                  [n_cubes_x, n_cubes_y, n_cubes_z], mesh.CellType.tetrahedron)
            print(f"Total cells: {msh.topology.index_map(3).size_global}")

            # read permeability
            with open(perm_file, 'r') as f:
                data = []
                lines_read = 0
                for line in f:
                    if lines_read >= 187000:
                        break
                    data.extend([float(x) for x in line.strip().split()])
                    lines_read += 1
            rho_data = np.array(data).reshape(nz, ny, nx)
            print(f"p range: {rho_data.min():.2f} ~ {rho_data.max():.2f}")

            def get_cell_rho_values(cells):
                geom = msh.geometry
                centroids = np.mean(geom.x[msh.geometry.dofmap][cells], axis=1)
                indices_x = np.floor(centroids[:, 0] / h).astype(int)
                indices_y = np.floor(centroids[:, 1] / h).astype(int)
                indices_z = np.floor(centroids[:, 2] / h).astype(int)
                step_x, step_y, step_z = int(60 / n_cubes_x), int(220 / n_cubes_y), int(85 / n_cubes_z)
                return rho_data[indices_z * step_z, indices_y * step_y, indices_x * step_x]

            # boundary markers
            boundary_funcs = [(lambda x: np.isclose(x[2], 0.0), 10), (lambda x: np.isclose(x[2], 85.0), 11),
                              (lambda x: np.isclose(x[1], 0.0), 12), (lambda x: np.isclose(x[1], 220.0), 13),
                              (lambda x: np.isclose(x[0], 0.0), 14), (lambda x: np.isclose(x[0], 60.0), 15)]
            facet_indices, facet_values = [], []
            for func, tag in boundary_funcs:
                facets = mesh.locate_entities_boundary(msh, 2, func)
                facet_indices.append(facets)
                facet_values.append(np.full(len(facets), tag, dtype=np.int32))
            facet_indices = np.hstack(facet_indices)
            facet_values = np.hstack(facet_values)
            facet_tags = mesh.meshtags(msh, 2, facet_indices, facet_values)

            V = fem.functionspace(msh, ("Lagrange", deg))
            dofmap = V.dofmap
            self.dofmap = dofmap
            self.n_dofs_total = dofmap.index_map.size_global
            facets = np.concatenate([facet_tags.find(tag) for tag in [10, 11, 12, 13, 14, 15]])
            self.dofs_bc = np.unique(fem.locate_dofs_topological(V, 2, facets).astype(np.int32))
            print(f"Total DOFs: {self.n_dofs_total}, Boundary DOFs: {len(self.dofs_bc)}")

            # mesh data
            msh.topology.create_connectivity(3, 0)
            tetrahedra = msh.topology.connectivity(3, 0).array.reshape(-1, 4)
            num_tets = len(tetrahedra)
            msh.topology.create_connectivity(2, 3)
            face_to_cells = msh.topology.connectivity(2, 3)
            num_faces = face_to_cells.num_nodes

            self.cell_rho_values = get_cell_rho_values(np.arange(num_tets, dtype=np.int32))

            # domain decomposition
            adjacency = [[] for _ in range(num_tets)]
            for f in range(num_faces):
                cells = face_to_cells.links(f)
                if len(cells) == 2:
                    c1, c2 = cells[0], cells[1]
                    adjacency[c1].append(c2)
                    adjacency[c2].append(c1)
            num_cuts, membership = pymetis.part_graph(n_subdomains, adjacency=adjacency)
            membership = np.array(membership)
            del adjacency
            self.membership = membership

            # Level-0 core variables
            subdomain_all_dofs = [np.array([], dtype=np.int32) for _ in range(n_subdomains)]
            v = np.zeros(self.n_dofs_total, dtype=np.int32)
            v2 = np.zeros(self.n_dofs_total, dtype=np.int32)
            for sub_id in range(n_subdomains):
                sub_tri_ids = np.where(membership == sub_id)[0]
                sub_dofs_set = set()
                for cell_idx in sub_tri_ids:
                    sub_dofs_set.update(dofmap.cell_dofs(cell_idx))
                sub_dofs = np.array(list(sub_dofs_set), dtype=np.int32)
                subdomain_all_dofs[sub_id] = sub_dofs
                v[sub_dofs] += 1
                v2[sub_dofs] += (sub_id + 1)
            self.subdomain_all_dofs = subdomain_all_dofs

            v3 = v2.copy()
            GG = np.setdiff1d(np.where(v >= 2)[0], self.dofs_bc)
            Subd_F_all = np.setdiff1d(np.where(v == 2)[0], self.dofs_bc)
            II = np.setdiff1d(np.where(v == 1)[0], self.dofs_bc)
            print(f"GG: {len(GG)}, Subd_F_all: {len(Subd_F_all)}, Int: {len(II)}")

            # edges
            v2[Subd_F_all] = 0; v2[self.dofs_bc] = 0; v2[II] = 0
            Subd_E_candidates = []
            for sub_id in range(n_subdomains):
                sub_dofs = subdomain_all_dofs[sub_id]
                non_zero_mask = v2[sub_dofs] != 0
                if not non_zero_mask.any():
                    continue
                edge_dofs = sub_dofs[non_zero_mask]
                edge_values = v2[edge_dofs]
                for val in np.unique(edge_values):
                    if val == 0:
                        continue
                    edge = edge_dofs[edge_values == val]
                    if len(edge) > 0:
                        Subd_E_candidates.append(np.sort(edge))
            Subd_E = []
            edge_counter = 0
            for new_edge in Subd_E_candidates:
                new_set = set(new_edge)
                if not new_set:
                    continue
                v2_vals = v2[np.array(list(new_set), dtype=np.int32)]
                has_neg = np.any(v2_vals < 0)
                if not has_neg:
                    edge_counter += 1
                    Subd_E.append(np.array(sorted(new_set), dtype=np.int32))
                    v2[np.array(list(new_set), dtype=np.int32)] = -edge_counter
                    continue
                neg_vals = v2_vals[v2_vals < 0]
                if len(neg_vals) == len(v2_vals) and len(set(neg_vals)) == 1:
                    same_eid = -neg_vals[0]
                    existing = Subd_E[same_eid - 1]
                    if existing is not None and len(existing) == len(new_edge) and np.array_equal(np.sort(existing), np.sort(new_edge)):
                        continue
                involved_eids = set()
                for d in new_set:
                    if v2[d] < 0:
                        involved_eids.add(-v2[d])
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
                        v2[np.array(non_conflict, dtype=np.int32)] = -edge_counter
                        added.add(non_conflict)
                    for d in conflict:
                        if (d,) not in added:
                            edge_counter += 1
                            Subd_E.append(np.array([d], dtype=np.int32))
                            v2[d] = -edge_counter
                            added.add((d,))
            Subd_E = [e for e in Subd_E if e is not None]
            self.Subd_E = Subd_E
            print(f"Subd_E: {len(Subd_E)} edges")

            # faces
            Subd_E_all = np.array([d for edge in Subd_E for d in edge], dtype=np.int32)
            v3[Subd_E_all] = 0; v3[self.dofs_bc] = 0; v3[II] = 0
            Subd_F = []
            for sub_id in range(n_subdomains):
                sub_dofs = subdomain_all_dofs[sub_id]
                non_zero_mask = v3[sub_dofs] != 0
                if not non_zero_mask.any():
                    continue
                face_dofs = sub_dofs[non_zero_mask]
                face_values = v3[face_dofs]
                for val in np.unique(face_values):
                    if val == 0:
                        continue
                    face = face_dofs[face_values == val]
                    if len(face) > 0:
                        Subd_F.append(face)
                        v3[face] = 0
            self.Subd_F = Subd_F
            print(f"Subd_F: {len(Subd_F)} faces")

            self.entity_list = [e for e in Subd_E] + [f for f in Subd_F]
            print(f"entity_list: {len(self.entity_list)} entities "
                  f"({len(Subd_E)} edges + {len(Subd_F)} faces)")

            # Gamma/Interior
            Gamma, Interior = [None] * n_subdomains, [None] * n_subdomains
            for sub_id in range(n_subdomains):
                sub_dofs = subdomain_all_dofs[sub_id]
                Interior[sub_id] = np.setdiff1d(np.setdiff1d(sub_dofs, self.dofs_bc), GG)
                Gamma[sub_id] = np.intersect1d(sub_dofs, GG)
            self.Gamma, self.Interior = Gamma, Interior
            self.GG = GG
            self.II = II

            with open(_preproc_cache_file, 'wb') as _f:
                pickle.dump({'key': _preproc_key, 'x': msh.geometry.x.copy(),
                             'cells': msh.topology.connectivity(3, 0).array.copy().reshape(-1, 4),
                             'cell_rho_values': self.cell_rho_values, 'membership': membership,
                             'subdomain_all_dofs': subdomain_all_dofs,
                             'v': v, 'v2': v2, 'v3': v3, 'GG': GG, 'Subd_F_all': Subd_F_all,
                             'II': II, 'dofs_bc': self.dofs_bc, 'Subd_E': Subd_E, 'Subd_F': Subd_F,
                             'entity_list': self.entity_list, 'Gamma': Gamma, 'Interior': Interior}, _f)
            print(f"[cache] preprocess saved to {_preproc_cache_file}")
        print(f"[T] preprocess: {time.time() - self.start_time:.2f}s")
        self.msh = msh

    # ---- Level-0 raw subdomain computation (fully reusable across mu) ----

    def compute_subdomains(self, mu_level0=None):
        # mu_level0: Level-0 eigenpair completeness threshold for the sygvx
        # subset solver (= max mu0 ever passed to select_level0). If None,
        # falls back to the full eigh (byte-identical to the old tables).
        c = self.cfg
        n_subdomains = c['n_subdomains_list'][0]
        msh, deg = self.msh, c['deg']
        f = PETSc.ScalarType(1.0)
        _t0 = time.time()

        for sub_id in range(n_subdomains):
            sub_tet_ids = np.where(self.membership == sub_id)[0]
            m, n = self.Gamma[sub_id], self.Interior[sub_id]
            len_m, len_n = len(m), len(n)
            _p0 = time.time()
            print(f"  subdomain {sub_id}: {len(sub_tet_ids)} tets, Gamma:{len_m}, Interior:{len_n}", flush=True)

            submesh, entity_map, vertex_map, geometry_map = mesh.create_submesh(msh, msh.topology.dim, sub_tet_ids)
            V_sub = fem.functionspace(submesh, ("Lagrange", deg))
            dofmap_sub = V_sub.dofmap
            num_sub_cells = submesh.topology.index_map(3).size_local
            sub_cell_indices = np.arange(num_sub_cells, dtype=np.int32)
            cell_map = entity_map.sub_topology_to_topology(sub_cell_indices, inverse=False)
            sub_rho_values = self.cell_rho_values[cell_map]

            sub_dofs_flat = dofmap_sub.list.flatten()
            orig_dofs_flat = self.dofmap.list[cell_map].flatten() if hasattr(self.dofmap.list, 'flatten') else np.asarray(self.dofmap.list).flatten()[cell_map]
            sort_idx = np.argsort(sub_dofs_flat)
            sorted_sub = sub_dofs_flat[sort_idx]
            sorted_orig = orig_dofs_flat[sort_idx]
            _, unique_idx = np.unique(sorted_sub, return_index=True)
            all_sub_dofs = sorted_orig[unique_idx]
            sorted_idx = np.argsort(all_sub_dofs)
            sorted_global = all_sub_dofs[sorted_idx]
            m_local = sorted_idx[np.searchsorted(sorted_global, m)]
            n_local = sorted_idx[np.searchsorted(sorted_global, n)]

            V_local = fem.functionspace(submesh, ("Lagrange", deg))
            u, v = ufl.TrialFunction(V_local), ufl.TestFunction(V_local)
            DG0 = fem.functionspace(submesh, ("DG", 0))
            rho_func = fem.Function(DG0)
            rho_func.x.array[:] = sub_rho_values
            a_local = fem.form(ufl.inner(rho_func * ufl.grad(u), ufl.grad(v)) * ufl.dx)
            L_local = fem.form(ufl.inner(f, v) * ufl.dx)
            A_local_sp = fem.assemble_matrix(a_local)
            b_local = fem.assemble_vector(L_local)

            A_local = PETSc.Mat().createAIJ(size=(A_local_sp.index_map(0).size_local, A_local_sp.index_map(1).size_local),
                                            csr=(A_local_sp.indptr.astype(PETSc.IntType),
                                                 A_local_sp.indices.astype(PETSc.IntType),
                                                 A_local_sp.data.astype(PETSc.ScalarType)))
            is_m_local = PETSc.IS().createGeneral(m_local.astype(PETSc.IntType), comm=PETSc.COMM_WORLD)
            is_n_local = PETSc.IS().createGeneral(n_local.astype(PETSc.IntType), comm=PETSc.COMM_WORLD)
            A1 = A_local.createSubMatrix(is_m_local, is_m_local); A1.assemble()
            A2 = A_local.createSubMatrix(is_m_local, is_n_local); A2.assemble()
            A3 = A_local.createSubMatrix(is_n_local, is_m_local); A3.assemble()
            A4 = A_local.createSubMatrix(is_n_local, is_n_local); A4.assemble()
            _p1 = time.time()

            # A4^{-1} A3 via CHOLMOD sparse factor (keeps only L)
            A4_csr = self._petsc_to_scipy_csr(A4)
            A3_dense = self._petsc_to_scipy_dense(A3)
            try:
                A4_fac = cho_factor(A4_csr.tocsc())
            except Exception:
                A4_fac = splu(A4_csr)
            all_solutions = A4_fac.solve(A3_dense)
            _p2 = time.time()

            # dense Schur (only for the local GEP)
            A1_dense = self._petsc_to_scipy_dense(A1)
            A2_dense = self._petsc_to_scipy_dense(A2)
            Schur_np_raw = A1_dense - A2_dense @ all_solutions
            del all_solutions, A2_dense

            b_local_array = b_local.array.copy()
            b_m_values = b_local_array[m_local]
            b_n_values = b_local_array[n_local]
            _p3 = time.time()

            # local entity-decouple mask (no global TS_Blk)
            Schur_np = (Schur_np_raw + Schur_np_raw.T) / 2
            in_gamma = np.zeros(self.n_dofs_total, dtype=bool)
            in_gamma[m] = True
            mask = np.zeros((len_m, len_m))
            for ent in self.entity_list:
                s = ent[in_gamma[ent]]
                if s.size:
                    loc = np.searchsorted(m, s)
                    mask[np.ix_(loc, loc)] += 1.0
            A1_masked = np.multiply(Schur_np, mask)
            _p4 = time.time()

            # P4 generalized eigen: default sygvx subset (k smallest) with a
            # completeness retry (identical to 3d_alg1_clean.py). Selection
            # 判据 λ<mu0 stays bit-identical: if D[-1] > mu_level0, no true
            # eigenpair below mu_level0 can lie beyond the computed k smallest.
            use_sygvx = (os.environ.get('NOSAS_P4_SYGVX', '1') == '1') and mu_level0 is not None
            if use_sygvx:
                k_p4 = min(len_m, 150)
                _p4_retry = 0
                while True:
                    try:
                        D, Q = eigh(Schur_np, A1_masked, subset_by_index=(0, k_p4 - 1))
                    except np.linalg.LinAlgError:
                        D, Q = eigh(Schur_np, A1_masked)
                        break
                    if D[-1] > mu_level0 or k_p4 >= len_m:
                        break
                    k_p4 = min(len_m, 2 * k_p4)
                    _p4_retry += 1
                if _p4_retry:
                    print(f"[T]  L0 sub{sub_id}: P4 sygvx retry -> k={k_p4} ({_p4_retry}x)", flush=True)
            else:
                # full eigh (matches original table scripts)
                D, Q = eigh(Schur_np, A1_masked)
            del Schur_np, A1_dense
            _p5 = time.time()
            print(f"[T]  L0 sub{sub_id}: P1(submesh/asm)={_p1-_p0:.2f}s "
                  f"P2(A4_inv·A3,chol)={_p2-_p1:.2f}s P3(Schur+g)={_p3-_p2:.2f}s "
                  f"P4(mask+eigh)={_p5-_p3:.2f}s", flush=True)

            # S_Blk rows/cols/values come from A1_masked (csr sparsity of the mask)
            A1_csr = csr_matrix(A1_masked)
            s_indptr, s_indices, s_vals = A1_csr.indptr, A1_csr.indices, A1_csr.data
            s_row_indices_local = np.repeat(np.arange(len(np.diff(s_indptr))), np.diff(s_indptr))

            self.results.append({
                'sub_id': sub_id,
                'm': m.copy(),
                'Interior': self.Interior[sub_id].copy(),
                'b_m': b_m_values.copy(),
                'b_n': b_n_values.copy(),
                'A1_csr': self._petsc_to_scipy_csr(A1),
                'A2_csr': self._petsc_to_scipy_csr(A2),   # A3 = A2^T
                'A4_inv': A4_fac,
                'A1_mask_full': A1_masked,      # len_m × len_m dense (reused per mu)
                'D_full': D,                    # ascending eigenvalues (full eigh)
                'Q_full': Q,                    # len_m × len_m eigenvectors
                'S_Blk_rows': m[s_row_indices_local].copy(),
                'S_Blk_cols': m[s_indices].copy(),
                'S_Blk_vals': s_vals.copy(),
            })

            is_m_local.destroy(); is_n_local.destroy()
            A_local.destroy(); A1.destroy(); A2.destroy(); A3.destroy(); A4.destroy()
            gc.collect()
        print(f"[T] Level0 raw subdomain loop: {time.time() - _t0:.2f}s")
        _t0 = time.time()

        # global g + implicit Schur action subs (selection independent)
        GG = np.sort(self.GG.copy())
        self.GG = GG
        g_arr = np.zeros(len(GG))
        subs = []
        for r in self.results:
            _idx = np.searchsorted(GG, np.asarray(r['m']))
            r['gg_idx'] = _idx
            r['A3_csr'] = r['A2_csr'].T.tocsr()
            xI = r['A4_inv'].solve(r['b_n'])
            g_arr[_idx] += r['b_m'] - r['A2_csr'] @ xI
            subs.append(r)
        self.g_arr = g_arr
        self.subs = subs
        print(f"[T] Level0 global g assembly: {time.time() - _t0:.2f}s")

    # ---- Level-0 selection for a given mu0 (pure slice, cheap) ----

    def select_level0(self, mu0):
        n_subdomains = self.cfg['n_subdomains_list'][0]
        self.S_Blk_rows_list = [None] * n_subdomains
        self.S_Blk_cols_list = [None] * n_subdomains
        self.S_Blk_vals_list = [None] * n_subdomains
        self.A_QG_data_list = [None] * n_subdomains
        self.D_inv_G_list = [None] * n_subdomains
        flag1 = np.zeros(n_subdomains, dtype=np.int32)
        for r in self.results:
            D = r['D_full']; Q = r['Q_full']; A1M = r['A1_mask_full']
            eigens_idx = np.where(D < mu0)[0]
            Q_s = Q[:, eigens_idx].copy()
            D_s = 1 - D[eigens_idx].copy()
            if len(eigens_idx) > 0:
                A1Q = A1M @ Q_s
                D_inv = 1.0 / D_s
            else:
                A1Q = np.array([]).reshape(len(r['m']), 0)
                D_inv = np.array([])
            self.S_Blk_rows_list[r['sub_id']] = r['S_Blk_rows']
            self.S_Blk_cols_list[r['sub_id']] = r['S_Blk_cols']
            self.S_Blk_vals_list[r['sub_id']] = r['S_Blk_vals']
            self.A_QG_data_list[r['sub_id']] = {'m': r['m'], 'A1Q': A1Q, 'n_eigen': int(len(eigens_idx))}
            self.D_inv_G_list[r['sub_id']] = D_inv
            flag1[r['sub_id']] = int(len(eigens_idx))
        self.flag1 = flag1

    # ---- coarsening (identical to reference) ----

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
        coarse_Subd_E_from_v3 = np.setdiff1d(np.where(coarse_v >= 3)[0], dofs_bc)
        coarse_II = np.setdiff1d(np.where(coarse_v == 1)[0], dofs_bc)
        return coarse_GG, coarse_Subd_E_from_v3, coarse_II, subdomain_all_dofs, coarse_v, coarse_v2

    def _extract_coarse_edges_faces(self, subdomain_all_dofs, coarse_Subd_E_from_v3, coarse_II,
                                    coarse_v2, coarse_v, n_coarse_subdomains):
        dofs_bc = self.dofs_bc
        coarse_v2[coarse_Subd_E_from_v3] = 0
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
                if existing is not None and len(existing) == len(new_edge) and np.array_equal(np.sort(existing), np.sort(new_edge)):
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
                    coarse_Subd_E.append(np.array(non_conflict, dtype=np.int32))
                    coarse_v2[np.array(non_conflict, dtype=np.int32)] = -edge_counter
                    added.add(non_conflict)
                for d in conflict:
                    if (d,) not in added:
                        edge_counter += 1
                        coarse_Subd_E.append(np.array([d], dtype=np.int32))
                        coarse_v2[d] = -edge_counter
                        added.add((d,))
        coarse_Subd_E = [e for e in coarse_Subd_E if e is not None]
        coarse_v3 = coarse_v.copy()
        Subd_E_all = np.array([d for e in coarse_Subd_E for d in e], dtype=np.int32) if coarse_Subd_E else np.array([], dtype=np.int32)
        coarse_v3[Subd_E_all] = 0
        coarse_v3[dofs_bc] = 0
        coarse_v3[coarse_II] = 0
        coarse_v3[coarse_Subd_E_from_v3] = 0
        remaining_mask = (coarse_v3 >= 2) & ~np.isin(np.arange(self.n_dofs_total, dtype=np.int32), dofs_bc)
        coarse_doppel = np.where(remaining_mask)[0]
        coarse_Subd_F = []
        for coarse_id in range(n_coarse_subdomains):
            group_dofs = subdomain_all_dofs[coarse_id]
            non_zero_mask = coarse_v3[group_dofs] != 0
            if not non_zero_mask.any():
                continue
            face_dofs = group_dofs[non_zero_mask]
            face_values = coarse_v3[face_dofs]
            unique_values, inverse_indices = np.unique(face_values, return_inverse=True)
            for i, val in enumerate(unique_values):
                if val == 0:
                    continue
                entity = face_dofs[inverse_indices == i]
                if len(entity) > 0:
                    coarse_Subd_F.append(entity)
                    coarse_v3[entity] = 0
        return coarse_Subd_E, coarse_Subd_F

    def _extract_gamma_interior(self, subdomain_all_dofs, coarse_GG, n_coarse):
        coarse_Gamma = [None] * n_coarse
        coarse_Interior = [None] * n_coarse
        for cid in range(n_coarse):
            group_dofs = subdomain_all_dofs[cid]
            coarse_Interior[cid] = np.setdiff1d(np.setdiff1d(group_dofs, self.dofs_bc), coarse_GG)
            coarse_Gamma[cid] = np.intersect1d(group_dofs, coarse_GG)
        return coarse_Gamma, coarse_Interior

    def perform_coarsening(self, gamma_list, n_coarse_subdomains, level_name):
        n_groups = len(gamma_list)
        dofs_bc = self.dofs_bc
        target = min(n_coarse_subdomains, n_groups)
        if target < 2:
            subdomain_all_dofs = [np.unique(np.concatenate(gamma_list))]
            v_coarse = np.zeros(self.n_dofs_total, dtype=np.int32)
            for gamma in gamma_list:
                v_coarse[gamma] += 1
            coarse_GG = np.setdiff1d(np.where(v_coarse >= 2)[0], dofs_bc)
            coarse_Subd_E, coarse_Subd_F = [], []
            coarse_Gamma = [coarse_GG.copy()]
            coarse_Interior = [np.setdiff1d(subdomain_all_dofs[0], np.union1d(coarse_GG, dofs_bc))]
            group_assignment = np.zeros(n_groups, dtype=np.int32)
            return (coarse_GG, coarse_Subd_E, coarse_Subd_F,
                    coarse_Gamma, coarse_Interior, subdomain_all_dofs,
                    group_assignment, 1)
        adj_sub = [[] for _ in range(n_groups)]
        gamma_sets = [set(g) for g in gamma_list]
        for i in range(n_groups):
            si = gamma_sets[i]
            for j in range(i + 1, n_groups):
                if not si.isdisjoint(gamma_sets[j]):
                    adj_sub[i].append(j)
                    adj_sub[j].append(i)
        _, group_assignment_coarse = pymetis.part_graph(target, adjacency=adj_sub)
        group_assignment_coarse = np.array(group_assignment_coarse)
        unique_coarse_ids = np.unique(group_assignment_coarse)
        valid_coarse_ids = [cid for cid in range(n_coarse_subdomains) if cid in unique_coarse_ids]
        actual_n = len(valid_coarse_ids)
        id_map = {old_id: new_id for new_id, old_id in enumerate(valid_coarse_ids)}
        group_assignment_coarse_new = np.array([id_map[old_id] for old_id in group_assignment_coarse])
        coarse_GG, coarse_Subd_E_from_v3, coarse_II, subdomain_all_dofs, coarse_v, coarse_v2 = \
            self._extract_coarse_variables(gamma_list, actual_n, group_assignment_coarse_new)
        coarse_Subd_E, coarse_Subd_F = self._extract_coarse_edges_faces(
            subdomain_all_dofs, coarse_Subd_E_from_v3, coarse_II, coarse_v2, coarse_v, actual_n)
        coarse_Gamma, coarse_Interior = self._extract_gamma_interior(subdomain_all_dofs, coarse_GG, actual_n)
        return (coarse_GG, coarse_Subd_E, coarse_Subd_F, coarse_Gamma, coarse_Interior,
                subdomain_all_dofs, group_assignment_coarse_new, actual_n)

    # ---- Section-4 filter (reference algorithm, with optional negative-eigenwidow) ----

    def _section4_filter(self, A_QG_sparse, S_Blk_s_sparse, D_inv, eta, eta2, use_invD,
                         level_name='', group_id=-1, s_dofs=None, inv_ready=None, occ=None):
        n_eigen_old = A_QG_sparse.shape[1]
        if n_eigen_old == 0:
            return np.zeros((S_Blk_s_sparse.shape[0], 0)), 0, np.array([]), np.zeros((0, 0))
        S_sym = (S_Blk_s_sparse + S_Blk_s_sparse.T) / 2
        n_comp, labels = connected_components(S_sym)
        WtW = np.zeros((n_eigen_old, n_eigen_old))
        _sizes = [int(np.sum(labels == cc)) for cc in range(n_comp)]
        _W_collect = None
        for comp in range(n_comp):
            idx = np.where(labels == comp)[0]
            B_block = S_sym[np.ix_(idx, idx)].toarray()
            B_block = (B_block + B_block.T) / 2
            A_QG_block = A_QG_sparse[idx, :].toarray()
            B_block_inv = np.linalg.inv(B_block)
            if inv_ready is not None and s_dofs is not None and \
                    (occ is None or bool(np.all(occ[s_dofs[idx]] == 1))):
                inv_ready.append((s_dofs[idx].copy(), B_block_inv.copy()))
            nz_cols = np.where(np.any(np.abs(A_QG_block) > 1e-14, axis=0))[0]
            if len(nz_cols) > 0.9 * n_eigen_old:
                if _W_collect is None:
                    _W_collect = np.zeros((A_QG_sparse.shape[0], n_eigen_old))
                _W_collect[idx, :] += B_block_inv @ A_QG_block
            elif len(nz_cols) > 0:
                An = A_QG_block[:, nz_cols]
                _add_block(WtW, nz_cols, An.T @ (B_block_inv @ An))
        if _W_collect is not None:
            A_QG_dense_all = A_QG_sparse.toarray()
            WtW += A_QG_dense_all.T @ _W_collect
            del A_QG_dense_all, _W_collect
        if use_invD:
            if np.any(D_inv <= 0):
                eigvals_c, eigvecs_c = eig(WtW, np.diag(D_inv))
                eigvals = np.real(eigvals_c)
                eigvecs = np.real(eigvecs_c)
            else:
                eigvals, eigvecs = eigh(WtW, np.diag(D_inv))
            sort_idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[sort_idx]
            eigvecs = eigvecs[:, sort_idx]
        else:
            eigvals, eigvecs = eigh(WtW)
        threshold = 1 - eta
        legacy = self.cfg.get('legacy_section4', False)
        if legacy:
            idx1 = np.where(eigvals > threshold)[0]
            idx = idx1
        else:
            idx1 = np.where(eigvals > threshold)[0]
            if self.cfg.get('enable_neg_eigen', False):
                idx2 = np.where(-eigvals > eta2)[0]
            else:
                idx2 = np.array([], dtype=int)
            idx = np.concatenate([idx1, idx2]) if len(idx2) else idx1
        n_selected = len(idx)
        if n_selected == 0:
            return np.zeros((S_Blk_s_sparse.shape[0], 0)), 0, np.array([]), np.zeros((0, 0))
        x = eigvecs[:, idx]
        D_x = eigvals[idx]
        if use_invD:
            z = x * D_inv[:, np.newaxis]
            if legacy:
                D_inv_selected = D_inv[idx]
            else:
                d = np.sum(x * (D_inv[:, np.newaxis] * x), axis=0)
                D_inv_selected = 1.0 / d
        else:
            z = x.copy()
            D_inv_selected = np.ones(n_selected)
        WtW_inv_z = np.linalg.solve(WtW, z)
        if legacy:
            _B = WtW_inv_z * D_x[np.newaxis, :]
        else:
            _B = WtW_inv_z * (D_inv_selected * D_x)[np.newaxis, :]
        if A_QG_sparse.nnz > 0.5 * A_QG_sparse.shape[0] * A_QG_sparse.shape[1]:
            hat_U_s = A_QG_sparse.toarray() @ _B
        else:
            hat_U_s = A_QG_sparse.dot(_B)
        G_off = _B.T @ WtW @ _B
        return hat_U_s, n_selected, D_inv_selected, G_off

    def _compute_level_section4(self, S_Blk_prev_rows, S_Blk_prev_cols, S_Blk_prev_vals,
                                A_QG_prev_data, D_inv_G_prev,
                                subdomain_all_dofs, Gamma_list, Interior_list,
                                Subd_E_prev, Subd_F_prev,
                                n_coarse_subdomains, group_assignment_coarse,
                                eta, eta2, use_invD, level_name='', inv_ready=None):
        A_QG_data = [None] * n_coarse_subdomains
        flag = np.zeros(n_coarse_subdomains, dtype=np.int32)
        D_inv_G = [None] * n_coarse_subdomains
        G_off_G = [None] * n_coarse_subdomains
        S_Blk_rows = [None] * n_coarse_subdomains
        S_Blk_cols = [None] * n_coarse_subdomains
        S_Blk_vals = [None] * n_coarse_subdomains
        occ = np.zeros(self.n_dofs_total, dtype=np.int32)
        for cid in range(n_coarse_subdomains):
            s_c = subdomain_all_dofs[cid]
            if s_c is not None and len(s_c) > 0:
                occ[s_c] += 1
        for coarse_id in range(n_coarse_subdomains):
            sub_ids_in_group = np.where(group_assignment_coarse == coarse_id)[0]
            if len(sub_ids_in_group) == 0:
                continue
            s, m, n = subdomain_all_dofs[coarse_id], Gamma_list[coarse_id], Interior_list[coarse_id]
            local_A_QG_rows, local_A_QG_cols, local_A_QG_vals = [], [], []
            col_offset = 0
            for sub_id in sub_ids_in_group:
                data = A_QG_prev_data[sub_id]
                if data is None or data['n_eigen'] == 0:
                    continue
                m_indices, A1Q, n_eigen = data['m'], data['A1Q'], data['n_eigen']
                local_A_QG_rows.append(np.repeat(m_indices, n_eigen))
                local_A_QG_cols.append(np.tile(np.arange(n_eigen) + col_offset, len(m_indices)))
                local_A_QG_vals.append(A1Q.ravel())
                col_offset += n_eigen
            all_rows = np.concatenate(local_A_QG_rows) if local_A_QG_rows else np.array([], dtype=np.int32)
            all_cols = np.concatenate(local_A_QG_cols) if local_A_QG_cols else np.array([], dtype=np.int32)
            all_vals = np.concatenate(local_A_QG_vals) if local_A_QG_vals else np.array([], dtype=PETSc.ScalarType)
            good_s4 = [(S_Blk_prev_rows[sid], S_Blk_prev_cols[sid], S_Blk_prev_vals[sid])
                       for sid in sub_ids_in_group
                       if S_Blk_prev_rows[sid] is not None and len(S_Blk_prev_rows[sid]) > 0]
            if len(good_s4) == 0:
                continue
            grp_S_rows = np.concatenate([e[0] for e in good_s4])
            grp_S_cols = np.concatenate([e[1] for e in good_s4])
            grp_S_vals = np.concatenate([e[2] for e in good_s4])

            s_arr = np.asarray(s)
            remap = np.full(self.n_dofs_total, -1, dtype=np.int32)
            remap[s_arr] = np.arange(len(s_arr))
            keep_S = (remap[grp_S_rows] >= 0) & (remap[grp_S_cols] >= 0)
            loc_S_rows = remap[grp_S_rows[keep_S]]
            loc_S_cols = remap[grp_S_cols[keep_S]]
            S_Blk_s_sparse = csr_matrix((grp_S_vals[keep_S], (loc_S_rows, loc_S_cols)), shape=(len(s), len(s)))

            keep_A = remap[all_rows] >= 0
            loc_A_rows = remap[all_rows[keep_A]]
            n_eigen_old = col_offset
            if n_eigen_old == 0 or len(s) == 0:
                A_QG_np_s = np.array([]).reshape(len(s), 0)
            else:
                A_QG_np_s = csr_matrix((all_vals[keep_A], (loc_A_rows, all_cols[keep_A])), shape=(len(s), col_offset))
            if n_eigen_old == 0 or len(s) == 0:
                hat_U_s = np.array([]).reshape(len(s), 0)
                n_selected = 0
                D_inv_s = np.array([])
                flag[coarse_id] = 0
            else:
                D_inv_vals = [D_inv_G_prev[sid] for sid in sub_ids_in_group
                              if D_inv_G_prev[sid] is not None and len(D_inv_G_prev[sid]) > 0]
                if len(D_inv_vals) == 0:
                    continue
                D_inv_local = np.concatenate(D_inv_vals)
                hat_U_s, n_selected, D_inv_s, G_off_s = self._section4_filter(
                    A_QG_np_s, S_Blk_s_sparse, D_inv_local, eta, eta2, use_invD, level_name, coarse_id,
                    s_dofs=s, inv_ready=inv_ready, occ=occ)
                flag[coarse_id] = n_selected
                G_off_G[coarse_id] = G_off_s
            A_QG_data[coarse_id] = {'m': s, 'A1Q': hat_U_s, 'n_eigen': n_selected}
            D_inv_G[coarse_id] = D_inv_s
            S_Blk_s = S_Blk_s_sparse
            S_Blk_s_csr = csr_matrix(S_Blk_s)
            indptr, indices, values = S_Blk_s_csr.indptr, S_Blk_s_csr.indices, S_Blk_s_csr.data
            row_counts = np.diff(indptr)
            row_indices_local = np.repeat(np.arange(len(row_counts)), row_counts)
            S_Blk_rows[coarse_id] = s[row_indices_local]
            S_Blk_cols[coarse_id] = s[indices]
            S_Blk_vals[coarse_id] = values
        return A_QG_data, flag, D_inv_G, S_Blk_rows, S_Blk_cols, S_Blk_vals, G_off_G

    # ---- final assembly (reference algorithms) ----

    def _build_InvS_Blk_3d(self, Subd_E_in, Subd_F_in, S_Blk_rows_in, S_Blk_cols_in,
                           S_Blk_vals_in, coarse_GG_in, inv_ready=None):
        gg_pos = np.full(self.n_dofs_total, -1, dtype=np.int64)
        gg_pos[coarse_GG_in] = np.arange(len(coarse_GG_in))
        rows, cols, vals = [], [], []
        ent_loc, ent_inv = [], []
        ready = {}
        if inv_ready is not None:
            for _d, _inv in inv_ready:
                ready[frozenset(_d)] = (_d, _inv)
        n_reused = 0
        shared = []
        for entity in Subd_F_in + Subd_E_in:
            loc = gg_pos[entity]
            loc = loc[loc >= 0]
            if len(loc) == 0:
                continue
            key = frozenset(np.sort(entity))
            if key in ready:
                _d, _inv = ready[key]
                pos = {int(dd): k for k, dd in enumerate(_d)}
                perm = np.array([pos[int(dof)] for dof in entity], dtype=np.int64)
                inv = _inv[np.ix_(perm, perm)]
                n_reused += 1
                n = len(loc)
                rows.append(np.repeat(loc, n)); cols.append(np.tile(loc, n)); vals.append(inv.ravel())
                ent_loc.append(loc); ent_inv.append(inv)
            else:
                shared.append((entity, loc))
        if shared:
            sel_global = np.unique(np.concatenate([ent for ent, _ in shared]))
            sel_pos = np.full(self.n_dofs_total, -1, dtype=np.int64)
            sel_pos[sel_global] = np.arange(len(sel_global))
            all_r = np.concatenate([r for r in S_Blk_rows_in if r is not None])
            all_c = np.concatenate([c for c in S_Blk_cols_in if c is not None])
            all_v = np.concatenate([v for v in S_Blk_vals_in if v is not None])
            keep = (sel_pos[all_r] >= 0) & (sel_pos[all_c] >= 0)
            S_sel = csr_matrix((all_v[keep],
                                (sel_pos[all_r[keep]], sel_pos[all_c[keep]])),
                               shape=(len(sel_global), len(sel_global)))
            _pos_scratch = np.full(len(sel_global), -1, dtype=np.int64)
            for entity, loc in shared:
                loc_idx = sel_pos[entity]
                S_dense = _csr_block_gather(S_sel.indptr, S_sel.indices, S_sel.data,
                                            loc_idx.astype(np.int64), _pos_scratch)
                try:
                    inv = np.linalg.inv(S_dense)
                except np.linalg.LinAlgError:
                    inv = np.linalg.pinv(S_dense)
                n = len(loc)
                rows.append(np.repeat(loc, n)); cols.append(np.tile(loc, n)); vals.append(inv.ravel())
                ent_loc.append(loc); ent_inv.append(inv)
        rows = np.concatenate(rows) if rows else np.array([], dtype=np.int64)
        cols = np.concatenate(cols) if cols else np.array([], dtype=np.int64)
        vals = np.concatenate(vals) if vals else np.array([], dtype=PETSc.ScalarType)
        InvS = csr_matrix((vals.astype(PETSc.ScalarType), (rows, cols)),
                          shape=(len(coarse_GG_in), len(coarse_GG_in)))
        return InvS, ent_loc, ent_inv

    def _build_W_shell(self, A_QG_csr_in, ent_loc_in, ent_inv_in):
        ctx = {'A': A_QG_csr_in, 'loc': ent_loc_in, 'inv': ent_inv_in,
               'n': A_QG_csr_in.shape[0], 'X': A_QG_csr_in.shape[1]}

        class _WContext:
            def mult(self, mat, x, y):
                va = x.array_r
                t = ctx['A'] @ va
                res = np.zeros(ctx['n'])
                for loc, inv in zip(ctx['loc'], ctx['inv']):
                    res[loc] += inv @ t[loc]
                y.array_w[:] = res

            def multTranspose(self, mat, x, y):
                ra = x.array_r
                w = np.zeros(ctx['n'])
                for loc, inv in zip(ctx['loc'], ctx['inv']):
                    w[loc] += inv @ ra[loc]
                y.array_w[:] = ctx['A'].T @ w

        W = PETSc.Mat().createPython((ctx['n'], ctx['X']), comm=PETSc.COMM_WORLD)
        W.setPythonContext(_WContext())
        W.setUp()
        return W

    def _assemble_final_global_3d(self, A_QG_data, S_Blk_rows, S_Blk_cols, S_Blk_vals,
                                  Subd_E_list, Subd_F_list, coarse_GG, D_inv_G, G_off_G=None,
                                  inv_ready=None):
        _t0 = time.time()
        n_gg = len(coarse_GG)
        gg_pos = np.full(self.n_dofs_total, -1, dtype=np.int64)
        gg_pos[coarse_GG] = np.arange(n_gg)
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
            all_rows = np.concatenate(all_rows)
            all_cols = np.concatenate(all_cols)
            all_vals = np.concatenate(all_vals)
        else:
            all_rows = np.array([], dtype=np.int64); all_cols = np.array([], dtype=np.int64); all_vals = np.array([])
        A_QG_csr = csr_matrix((all_vals, (all_rows, all_cols)), shape=(n_gg, col_offset))
        print(f"[T]   assemble b: A_QG(GG x {col_offset}): {time.time()-_t0:.2f}s")
        _tc = time.time()
        InvS_csr, ent_loc, ent_inv = self._build_InvS_Blk_3d(
            Subd_E_list, Subd_F_list, S_Blk_rows, S_Blk_cols, S_Blk_vals, coarse_GG, inv_ready)
        InvS_Blk = PETSc.Mat().createAIJ(size=(n_gg, n_gg),
                                         csr=(InvS_csr.indptr.astype(PETSc.IntType),
                                              InvS_csr.indices.astype(PETSc.IntType),
                                              InvS_csr.data.astype(PETSc.ScalarType)))
        InvS_Blk.assemble()
        print(f"[T]   assemble c: InvS_Blk(GG x GG): {time.time()-_tc:.2f}s")
        if D_inv_G is None:
            D_inv_filtered = np.ones(col_offset)
        else:
            D_inv_filtered = np.concatenate([d for d in D_inv_G if d is not None and len(d) > 0])
        _td = time.time()
        W = self._build_W_shell(A_QG_csr, ent_loc, ent_inv)
        print(f"[T]   assemble d: W shell: {time.time()-_td:.2f}s")
        _tgs = time.time()
        if G_off_G is not None:
            G_off_full = np.zeros((col_offset, col_offset))
            _col_off = 0
            for _g in G_off_G:
                if _g is None or _g.size == 0:
                    continue
                _k = _g.shape[0]
                G_off_full[_col_off:_col_off + _k, _col_off:_col_off + _k] = _g
                _col_off += _k
            _gs_csr = csr_matrix(np.diag(D_inv_filtered) - G_off_full)
        else:
            AQ_indptr = A_QG_csr.indptr
            AQ_indices = A_QG_csr.indices
            AQ_data = A_QG_csr.data
            Goff_dense = np.zeros((col_offset, col_offset))
            for loc, inv in zip(ent_loc, ent_inv):
                nz_cols, Adense = _csr_rows_extract(AQ_indptr, AQ_indices, AQ_data,
                                                    loc.astype(np.int64))
                if nz_cols.shape[0] == 0:
                    continue
                Wn = inv @ Adense
                _add_block(Goff_dense, nz_cols, Adense.T @ Wn)
            rr, cc = np.nonzero(Goff_dense)
            _gs_csr = csr_matrix((-Goff_dense[rr, cc], (rr, cc)),
                                 shape=(col_offset, col_offset))
            _gs_csr = _gs_csr + csr_matrix(
                (D_inv_filtered, (np.arange(col_offset), np.arange(col_offset))),
                shape=(col_offset, col_offset))
            del Goff_dense
        Global_Solver = PETSc.Mat().createAIJ(
            size=(col_offset, col_offset),
            csr=(_gs_csr.indptr.astype(PETSc.IntType),
                 _gs_csr.indices.astype(PETSc.IntType),
                 _gs_csr.data.astype(PETSc.ScalarType)))
        Global_Solver.assemble()
        print(f"[T]   assemble f: Global_Solver({col_offset}x{col_offset}): {time.time()-_tgs:.2f}s")
        return A_QG_csr, InvS_Blk, W, Global_Solver, _gs_csr

    # ---- implicit Schur action + PCG ----

    def _implicit_schur_action(self, n, subs):
        class _ImplicitSchurAction:
            def __init__(self, n, subs):
                self.n = n
                self.subs = subs

            def mult(self, x, y):
                xa = x.getArray()
                res = np.zeros(self.n)
                for r in self.subs:
                    xg = xa[r['gg_idx']]
                    w = r['A3_csr'] @ xg
                    w2 = r['A4_inv'].solve(w)
                    res[r['gg_idx']] += r['A1_csr'] @ xg - r['A2_csr'] @ w2
                y.setArray(res)

        return _ImplicitSchurAction(n, subs)

    def pcg_nosas(self, A, b, max_it, tol, InvS_Blk, W, gs_fac, ksp_global):
        x = b.copy(); x.set(0.0)
        r = b.copy()
        bnrm2 = b.norm(PETSc.NormType.NORM_2)
        error = r.norm(PETSc.NormType.NORM_2) / bnrm2
        print(f' PCG residual(0) = {error:e}')
        alpha, beta = np.zeros(max_it), np.zeros(max_it)
        _t_inv, _t_glob, _t_ax = 0.0, 0.0, 0.0
        iters = 0
        lmax = lmin = cond = 0.0
        for iter_count in range(max_it):
            z = r.copy()
            _t0 = time.time()
            InvS_Blk.mult(r, z)
            _t1 = time.time()
            _, W_n = W.getSize()
            if W_n > 0:
                WTr = PETSc.Vec().createMPI(W_n, comm=PETSc.COMM_WORLD)
                W.multTranspose(r, WTr)
                if gs_fac is not None:
                    sol = WTr.copy()
                    sol.array = gs_fac.solve(np.array(WTr.array, copy=True))
                else:
                    sol = WTr.copy()
                    ksp_global.solve(WTr, sol)
                z.axpy(1.0, W * sol)
            _t2 = time.time()
            _t_inv += _t1 - _t0; _t_glob += _t2 - _t1
            rho = r.dot(z)
            if iter_count > 0:
                beta[iter_count] = rho / rho_1
                p.scale(beta[iter_count]); p.axpy(1.0, z)
            else:
                p = z.copy()
            q = p.copy()
            _t3 = time.time()
            A.mult(p, q)
            _t4 = time.time()
            _t_ax += _t4 - _t3
            alpha[iter_count] = rho / p.dot(q)
            x.axpy(alpha[iter_count], p)
            r.axpy(-alpha[iter_count], q)
            error = r.norm(PETSc.NormType.NORM_2) / bnrm2
            print(f' PCG residual({iter_count + 1}) = {error:e}')
            iters = iter_count + 1
            if error <= tol:
                break
            rho_1 = rho
        flag = 0 if error <= tol else 1
        if flag == 0 and iters > 1:
            d = np.zeros(iters); s = np.zeros(iters - 1)
            d[0] = 1.0 / alpha[0]
            for i in range(iters - 1):
                d[i + 1] = beta[i + 1] / alpha[i] + 1.0 / alpha[i + 1]
                s[i] = -np.sqrt(beta[i + 1]) / alpha[i]
            T = np.zeros((iters, iters))
            T[iters - 1, iters - 1] = d[iters - 1]
            for i in range(iters - 1):
                T[i, i] = d[i]; T[i, i + 1] = s[i]; T[i + 1, i] = s[i]
            lambda_vals = np.linalg.eigvals(T)
            lmax, lmin = np.max(lambda_vals), np.min(lambda_vals)
            cond = lmax / lmin
            print(f'PCG iter {iters}, res {error:e}, lmax={lmax:g}, lmin={lmin:g}, cond={cond:g}')
        elif flag == 0:
            print(f'PCG iter 0, residual {error:e}')
        else:
            print(f'PCG did NOT converge in {iters} iters, residual {error:e}')
        print(f'[T] PCG loop {iters} iters (InvS={_t_inv:.2f}s Global={_t_glob:.2f}s A.mult={_t_ax:.2f}s)')
        return x, error, iters, flag, lmax, lmin, cond

    # ---- top-level per-combo run ----

    def run_multilevel(self, eta_list=None, eta2_list=None, use_invD_levels=None,
                       final_dinv='ones', max_it=200, tol=1e-6, mu_override=None):
        c = self.cfg
        n_levels = len(c['n_subdomains_list'])
        n_coarse_levels = n_levels - 1
        GG = self.GG
        flag1 = self.flag1

        _t0 = time.time()
        if n_coarse_levels > 0:
            coarse_levels_data = []
            current_gamma_list = self.Gamma
            current_Subd_E = self.Subd_E
            current_Subd_F = self.Subd_F
            for level_idx in range(1, n_levels):
                target_n = c['n_subdomains_list'][level_idx]
                level_name = f"Level {level_idx + 1}"
                (coarse_GG, coarse_Subd_E, coarse_Subd_F,
                 coarse_Gamma, coarse_Interior, subdomain_all_dofs_coarse,
                 group_assignment_coarse, actual_n) = self.perform_coarsening(
                    current_gamma_list, target_n, level_name)
                coarse_levels_data.append({
                    'GG': coarse_GG, 'Subd_E': coarse_Subd_E, 'Subd_F': coarse_Subd_F,
                    'Gamma': coarse_Gamma, 'Interior': coarse_Interior,
                    'subdomain_all_dofs': subdomain_all_dofs_coarse,
                    'group_assignment': group_assignment_coarse, 'actual_n': actual_n})
                current_gamma_list = subdomain_all_dofs_coarse
                current_Subd_E = coarse_Subd_E
                current_Subd_F = coarse_Subd_F
            print(f"[T] coarsening chain: {time.time()-_t0:.2f}s")

            prev_S_Blk_rows = self.S_Blk_rows_list[:]
            prev_S_Blk_cols = self.S_Blk_cols_list[:]
            prev_S_Blk_vals = self.S_Blk_vals_list[:]
            prev_A_QG_data = self.A_QG_data_list[:]
            prev_D_inv_G = self.D_inv_G_list[:]
            prev_Subd_E, prev_Subd_F = self.Subd_E, self.Subd_F

            inv_ready_out = None
            flag_sum_list = [int(np.sum(flag1))]
            for level_idx in range(n_coarse_levels):
                data = coarse_levels_data[level_idx]
                actual_n = data['actual_n']
                use_invD = use_invD_levels[level_idx] if use_invD_levels is not None else True
                eta_val = eta_list[level_idx] if eta_list is not None and level_idx < len(eta_list) else None
                eta2_val = eta2_list[level_idx] if eta2_list is not None and level_idx < len(eta2_list) else None
                level_name = f"Level {level_idx + 2}"
                if eta_val is None:
                    raise ValueError("Section-4 eta required for this path")
                _tcl = time.time()
                inv_ready_arg = []
                A_QG_data_coarse, flag_coarse, D_inv_G_coarse, \
                    S_Blk_rows_coarse, S_Blk_cols_coarse, S_Blk_vals_coarse, G_off_G_coarse = \
                    self._compute_level_section4(
                        prev_S_Blk_rows, prev_S_Blk_cols, prev_S_Blk_vals,
                        prev_A_QG_data, prev_D_inv_G,
                        data['subdomain_all_dofs'], data['Gamma'], data['Interior'],
                        prev_Subd_E, prev_Subd_F,
                        actual_n, data['group_assignment'],
                        eta_val, eta2_val, use_invD, level_name,
                        inv_ready=inv_ready_arg)
                flag_sum_list.append(int(np.sum(flag_coarse)))
                if level_idx == n_coarse_levels - 1:
                    inv_ready_out = inv_ready_arg
                print(f"[T] compute_level_section4 {level_name}: {time.time()-_tcl:.2f}s")
                prev_S_Blk_rows, prev_S_Blk_cols, prev_S_Blk_vals = \
                    S_Blk_rows_coarse, S_Blk_cols_coarse, S_Blk_vals_coarse
                prev_A_QG_data = A_QG_data_coarse
                prev_D_inv_G = D_inv_G_coarse
                prev_Subd_E, prev_Subd_F = data['Subd_E'], data['Subd_F']

            final_D_inv_g = None if final_dinv == 'ones' else prev_D_inv_G
            _ta = time.time()
            A_QG_csr, InvS_Blk, W, Global_Solver, gs_csr = self._assemble_final_global_3d(
                prev_A_QG_data, prev_S_Blk_rows, prev_S_Blk_cols, prev_S_Blk_vals,
                self.Subd_E, self.Subd_F, GG, final_D_inv_g, None, inv_ready_out)
            print(f"[T] assemble_final_global_3d: {time.time()-_ta:.2f}s")
        else:
            final_D_inv_g = None if final_dinv == 'ones' else self.D_inv_G_list
            flag_sum_list = [int(np.sum(flag1))]
            _ta = time.time()
            A_QG_csr, InvS_Blk, W, Global_Solver, gs_csr = self._assemble_final_global_3d(
                self.A_QG_data_list, self.S_Blk_rows_list, self.S_Blk_cols_list, self.S_Blk_vals_list,
                self.Subd_E, self.Subd_F, GG, final_D_inv_g, None)
            print(f"[T] assemble_final_global_3d: {time.time()-_ta:.2f}s")

        # global solver factorisation (CHOLMOD if SPD else PETSc LU)
        gs_size = Global_Solver.getSize()
        gs_fac = None
        ksp_global = None
        if gs_size[0] > 0 and gs_size[1] > 0:
            _tksp = time.time()
            try:
                gs_fac = cho_factor(gs_csr.tocsc())
                print(f"[T] gs CHOLMOD factor ({gs_csr.shape[0]}²): {time.time()-_tksp:.2f}s")
            except Exception as _e:
                print(f"  WARNING: CHOLMOD failed ({_e}), falling back to PETSc LU")
                gs_fac = None
                ksp_global = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
                ksp_global.setOperators(Global_Solver)
                ksp_global.setType(PETSc.KSP.Type.PREONLY)
                ksp_global.getPC().setType(PETSc.PC.Type.LU)
                ksp_global.setUp()
        else:
            print("  WARNING: Global_Solver degenerate, using InvS_Blk only")

        Schur_A = self._implicit_schur_action(len(GG), self.subs)
        g_gamma = PETSc.Vec().createWithArray(self.g_arr.copy())
        x, error, iters, flag, lmax, lmin, cond = self.pcg_nosas(
            A=Schur_A, b=g_gamma, max_it=max_it, tol=tol,
            InvS_Blk=InvS_Blk, W=W, gs_fac=gs_fac, ksp_global=ksp_global)
        if ksp_global is not None:
            ksp_global.destroy()

        _N_s0 = int(np.sum(flag1))
        return {
            'GG': len(GG), 'iters': iters, 'error': error, 'flag': flag,
            'lmax': lmax, 'lmin': lmin, 'cond': cond,
            'N_s0': _N_s0, 'flag1': flag1.copy(), 'flag_sums': flag_sum_list,
        }


if __name__ == '__main__':
    # quick self-test for Table 6 style single-level run
    cfg = {'n_subdomains_list': [128], 'mu': 0.5, 'h': 2, 'deg': 1,
           'name': 'selftest', 'enable_neg_eigen': False}
    eng = NOSAS3DEngine(cfg)
    eng.preprocess()
    eng.compute_subdomains()
    eng.select_level0(0.5)
    m = eng.run_multilevel(eta_list=None, eta2_list=None, use_invD_levels=None,
                           final_dinv='true')
    print("SELF-TEST:", m)