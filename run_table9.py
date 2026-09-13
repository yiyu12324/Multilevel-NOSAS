import gc
import sys
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import pymetis
from petsc4py import PETSc
import ufl
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import spsolve, splu
from scipy.sparse.csgraph import connected_components
from scipy.linalg import eigh, solve_triangular
from slepc4py import SLEPc

import time
sys.stdout.reconfigure(line_buffering=True)

start_time = time.time()

# ====================== Configuration ======================
n_subdomains_list = [128,16,2]  # change for different level counts
mu_list = [0.5, 0.5, 0.5]          # one mu per coarse level

n_levels = len(n_subdomains_list)
n_coarse_levels = n_levels - 1
n_fine_subdomains = n_subdomains_list[0]

deg = 1
h = 1

# ==================== 0. Read permeability data ====================
print("=== Reading permeability data ===")
perm_file = 'spe_perm.dat'
nx, ny, nz = 60, 220, 85

with open(perm_file, 'r') as f:
    data = []
    lines_read = 0
    for line in f:
        if lines_read >= 187000: break
        data.extend([float(x) for x in line.strip().split()])
        lines_read += 1

rho_data = np.array(data).reshape(nz, ny, nx)
print(f"Data shape: {rho_data.shape}, p range: {rho_data.min():.2f} ~ {rho_data.max():.2f}")

# ==================== 1. Create 3D mesh ====================
print("\n=== Creating mesh ===")
n_cubes_x, n_cubes_y, n_cubes_z = int(60 / h), int(220 / h), int(85 / h)
print(f"h={h}, Cube grid: {n_cubes_x}*{n_cubes_y}*{n_cubes_z}")

msh = mesh.create_box(MPI.COMM_SELF, [np.array([0, 0, 0]), np.array([60, 220, 85])],
                      [n_cubes_x, n_cubes_y, n_cubes_z], mesh.CellType.tetrahedron)

# ==================== 2. Get cell permeability values ====================
def get_cell_rho_values(cells):
    geom = msh.geometry
    centroids = np.mean(geom.x[msh.geometry.dofmap][cells], axis=1)
    indices_x = np.floor(centroids[:, 0] / h).astype(int)
    indices_y = np.floor(centroids[:, 1] / h).astype(int)
    indices_z = np.floor(centroids[:, 2] / h).astype(int)
    step_x, step_y, step_z = int(60 / n_cubes_x), int(220 / n_cubes_y), int(85 / n_cubes_z)
    return rho_data[indices_z * step_z, indices_y * step_y, indices_x * step_x]

# ==================== 3. Boundary markers ====================
print("\n=== Creating boundary markers ===")
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

# ==================== 4. Boundary DOFs ====================
V = fem.functionspace(msh, ("Lagrange", deg))
dofmap = V.dofmap
n_dofs_total = dofmap.index_map.size_global
facets = np.concatenate([facet_tags.find(tag) for tag in [10, 11, 12, 13, 14, 15]])
dofs_bc = np.unique(fem.locate_dofs_topological(V, 2, facets).astype(np.int32))
print(f"Total DOFs: {n_dofs_total}, Boundary DOFs: {len(dofs_bc)}")

# ==================== 5. Mesh data (3D) ====================
msh.topology.create_connectivity(3, 0)
tetrahedra = msh.topology.connectivity(3, 0).array.reshape(-1, 4)
num_tets = len(tetrahedra)
msh.topology.create_connectivity(2, 3)
face_to_cells = msh.topology.connectivity(2, 3)
num_faces = face_to_cells.num_nodes

# ==================== 6. Assign material properties ====================
print("\n=== Assigning material properties ===")
cell_rho_values = get_cell_rho_values(np.arange(num_tets, dtype=np.int32))
print(f"Cell p range: {cell_rho_values.min():.3f}~{cell_rho_values.max():.3f}")

# ==================== 7. Level 0 partition (3D) ====================
print("\n=== Domain decomposition ===")
adjacency = [[] for _ in range(num_tets)]
for f in range(num_faces):
    cells = face_to_cells.links(f)
    if len(cells) == 2:
        c1, c2 = cells[0], cells[1]
        adjacency[c1].append(c2)
        adjacency[c2].append(c1)

num_cuts, group_assignment = pymetis.part_graph(n_fine_subdomains, adjacency=adjacency)
group_assignment = np.array(group_assignment)

# ====================== Helper ======================
def petsc_to_scipy(petsc_mat):
    I, J, V = petsc_mat.getValuesCSR()
    return csr_matrix((V, J, I), shape=petsc_mat.getSize()).toarray()


def extract_entities_from_level(subdomain_all_dofs, GG, II, dofs_bc, n_sub):
    v_arr = np.zeros(n_dofs_total, dtype=np.int32)
    v2_arr = np.zeros(n_dofs_total, dtype=np.int32)
    for cid in range(n_sub):
        grp_dofs = subdomain_all_dofs[cid]
        v_arr[grp_dofs] += 1
        v2_arr[grp_dofs] += (cid + 1)

    v3_arr = v2_arr.copy()
    Subd_F_all = np.setdiff1d(np.where(v_arr == 2)[0], dofs_bc)

    v2_work = v2_arr.copy()
    v2_work[Subd_F_all] = 0
    v2_work[dofs_bc] = 0
    v2_work[II] = 0

    Subd_E_candidates = []
    for cid in range(n_sub):
        grp_dofs = subdomain_all_dofs[cid]
        nz_mask = v2_work[grp_dofs] != 0
        if not nz_mask.any():
            continue
        edofs = grp_dofs[nz_mask]
        evals = v2_work[edofs]
        uvals, inv_indices = np.unique(evals, return_inverse=True)
        for i, val in enumerate(uvals):
            if val == 0:
                continue
            entity = edofs[inv_indices == i]
            if len(entity) > 0:
                Subd_E_candidates.append(np.sort(entity))

    Subd_E = []
    edge_counter = 0
    for new_edge in Subd_E_candidates:
        new_set = set(new_edge)
        if not new_set:
            continue
        vvals = v2_arr[np.array(list(new_set), dtype=np.int32)]
        has_neg = np.any(vvals < 0)
        if not has_neg:
            edge_counter += 1
            Subd_E.append(np.array(sorted(new_set), dtype=np.int32))
            v2_arr[np.array(list(new_set), dtype=np.int32)] = -edge_counter
            continue
        neg_vals = vvals[vvals < 0]
        if len(neg_vals) == len(vvals) and len(set(neg_vals)) == 1:
            same_eid = -neg_vals[0]
            existing = Subd_E[same_eid - 1]
            if existing is not None and len(existing) == len(new_edge) and np.array_equal(np.sort(existing), np.sort(new_edge)):
                continue
        involved_eids = set()
        for d in new_set:
            if v2_arr[d] < 0:
                involved_eids.add(-v2_arr[d])
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
                v2_arr[np.array(non_conflict, dtype=np.int32)] = -edge_counter
                added.add(non_conflict)
            for d in conflict:
                if (d,) not in added:
                    edge_counter += 1
                    Subd_E.append(np.array([d], dtype=np.int32))
                    v2_arr[d] = -edge_counter
                    added.add((d,))
    Subd_E = [e for e in Subd_E if e is not None]

    Subd_E_all = np.array([d for edge in Subd_E for d in edge], dtype=np.int32) if Subd_E else np.array([], dtype=np.int32)
    v3_arr[Subd_E_all] = 0
    v3_arr[dofs_bc] = 0
    v3_arr[II] = 0

    Subd_F = []
    for cid in range(n_sub):
        grp_dofs = subdomain_all_dofs[cid]
        nz_mask = v3_arr[grp_dofs] != 0
        if not nz_mask.any():
            continue
        fdofs = grp_dofs[nz_mask]
        fvals = v3_arr[fdofs]
        uvals, inv_indices = np.unique(fvals, return_inverse=True)
        for i, val in enumerate(uvals):
            if val == 0:
                continue
            face = fdofs[inv_indices == i]
            if len(face) > 0:
                Subd_F.append(face)
                v3_arr[face] = 0

    return Subd_E, Subd_F


def build_A1_masked(Schur_sp, Subd_E, Subd_F, m):
    """A1_mat = Schur ∘ TS_Blk 的局部构造（不构造全局 TS_Blk）。

    TS_Blk 的点乘效果：保留 Schur 在"同一 entity（点/边/面）内部"的块，
    去掉跨 entity 的连接。等价于对每个 entity 提取局部标号子块再取并集。
    Schur_sp: m×m scipy 稀疏（局部标号）
    Subd_E/Subd_F: entity DOF 列表（全局标号）
    m: 本组 Gamma DOF（全局标号）
    返回 m×m scipy CSR 稀疏 A1_mat。
    """
    entities = list(Subd_E) + list(Subd_F)
    if len(m) == 0:
        return csr_matrix((0, 0))
    if len(entities) == 0:
        return csr_matrix((len(m), len(m)))
    m_sorted = np.sort(m)
    # 特判：单个 entity 覆盖全部 m → TS_Blk[m,m] 全 1 → A1_mat = Schur
    if len(entities) == 1 and np.all(np.isin(m, entities[0])):
        return Schur_sp
    rows_list, cols_list = [], []
    for entity in entities:
        e = np.unique(entity[np.isin(entity, m)])
        if len(e) == 0:
            continue
        loc = np.searchsorted(m_sorted, e)
        rows_list.append(np.repeat(loc, len(loc)))
        cols_list.append(np.tile(loc, len(loc)))
    if len(rows_list) == 0:
        return csr_matrix((len(m), len(m)))
    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    # 与原版 build_TS_Blk 一致：重复 (i,j) 条目在 CSR 构造时求和，
    # TS 即"entity 覆盖计数"矩阵（重叠处值 = k，与全局 TS_Blk[m,m] 逐位一致）
    TS = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(m), len(m)))
    return Schur_sp.multiply(TS).tocsr()


def build_nosas_global_operator(A_QG_data, D_inv_G, InvS_GG, GG, n_total):
    valid_qg = [d for d in A_QG_data if d is not None and d['n_eigen'] > 0]
    n_eigen = sum(d['n_eigen'] for d in valid_qg)
    n_gg = len(GG)

    if n_eigen > 0 and n_gg > 0:
        A_rows = [np.repeat(d['m'], d['n_eigen']) for d in A_QG_data if d is not None and d['n_eigen'] > 0]
        col_off = 0
        A_cols_list = []
        for d in A_QG_data:
            if d is None or d['n_eigen'] == 0:
                continue
            A_cols_list.append(np.tile(np.arange(d['n_eigen']) + col_off, len(d['m'])))
            col_off += d['n_eigen']
        A_vals = [d['A1Q'].ravel() for d in A_QG_data if d is not None and d['n_eigen'] > 0]
        A_rows = np.concatenate(A_rows)
        A_cols = np.concatenate(A_cols_list)
        A_vals = np.concatenate(A_vals)
        A_csr = csr_matrix((A_vals, (A_rows, A_cols)), shape=(n_total, n_eigen))
        A_petsc = PETSc.Mat().createAIJ(
            size=(n_total, n_eigen),
            csr=(A_csr.indptr.astype(PETSc.IntType),
                 A_csr.indices.astype(PETSc.IntType),
                 A_csr.data.astype(PETSc.ScalarType)),
            comm=PETSc.COMM_WORLD
        )
        A_petsc.assemble()
        is_GG_p = PETSc.IS().createGeneral(GG.astype(PETSc.IntType), comm=PETSc.COMM_WORLD)
        A_GG = A_petsc.createSubMatrix(is_GG_p, None)
        A_GG.assemble()

        D_vals = np.concatenate([d for d in D_inv_G if d is not None and len(d) > 0])
        n_tot = len(D_vals)
        D_inv_csr = csr_matrix((D_vals, (np.arange(n_tot), np.arange(n_tot))), shape=(n_tot, n_tot))
        D_inv_petsc = PETSc.Mat().createAIJ(
            size=(n_tot, n_tot),
            csr=(D_inv_csr.indptr.astype(PETSc.IntType),
                 D_inv_csr.indices.astype(PETSc.IntType),
                 D_inv_csr.data.astype(PETSc.ScalarType)),
            comm=PETSc.COMM_WORLD
        )
        D_inv_petsc.assemble()

        W = InvS_GG * A_GG
        Global_Solver = D_inv_petsc - A_GG.transposeMatMult(W)
        is_GG_p.destroy()
    elif n_gg > 0:
        empty_csr_w = csr_matrix((InvS_GG.getSize()[0], 0))
        W = PETSc.Mat().createAIJ(size=(InvS_GG.getSize()[0], 0),
                                   csr=(empty_csr_w.indptr.astype(PETSc.IntType),
                                        empty_csr_w.indices.astype(PETSc.IntType),
                                        empty_csr_w.data.astype(PETSc.ScalarType)),
                                   comm=PETSc.COMM_WORLD)
        W.assemble()
        # When no eigenvectors, Global_Solver is (0, 0) — matching 2D's assemble_global_matrices
        empty_csr_gg = csr_matrix((0, 0))
        Global_Solver = PETSc.Mat().createAIJ(
            size=(0, 0),
            csr=(empty_csr_gg.indptr.astype(PETSc.IntType),
                 empty_csr_gg.indices.astype(PETSc.IntType),
                 empty_csr_gg.data.astype(PETSc.ScalarType)),
            comm=PETSc.COMM_WORLD
        )
    else:
        empty_csr = csr_matrix((0, 0))
        W = PETSc.Mat().createAIJ(size=(0, 0),
                                   csr=(empty_csr.indptr.astype(PETSc.IntType),
                                        empty_csr.indices.astype(PETSc.IntType),
                                        empty_csr.data.astype(PETSc.ScalarType)),
                                   comm=PETSc.COMM_WORLD)
        W.assemble()
        Global_Solver = InvS_GG
    return W, Global_Solver


# ====================== Krylov-Schur helpers (Level 2, large DOFs) ======================
# Level 2 groups have n_interior ~ 50k: the dense yy = InvS_Blk_local + W*X (n×n) and
# dense eigh(Schur, A1_mat) blow up memory.  Instead:
#   * never form yy — apply it implicitly via Woodbury:
#       yy·r = A·r + A·U·C^{-1}·(U^T·(A·r)),   A = InvS_Blk_local, C = D_inv - U^T·A·U
#   * solve the GEP  S_hat·v = λ·B·v  (B = A1_masked = Schur ∘ TS_Blk, block-diagonal-ish)
#     by Cholesky B = L·L^T per connected component and SLEPc Krylov-Schur on the
#     standard symmetric problem T = L^{-1}·S_hat·L^{-T}  (smallest eigenvalues).
#   * Local_Solver and R_0T become implicit (Python shell) mats, never materialized.

def _yy_mult(A_csr, U_csr, C_lu, r):
    """Implicit yy@r = A@r + A@U@C^{-1}@(U^T@(A@r)).  All sparse / small ops."""
    w1 = A_csr @ r
    if U_csr.shape[1] > 0:
        w2 = U_csr.T @ w1
        w3 = C_lu.solve(w2)
        w4 = U_csr @ w3
        w5 = A_csr @ w4
        return w1 + w5
    return w1


def _S_hat_mult(A1_csr, A2_csr, A3_csr, A_csr, U_csr, C_lu, v):
    """Implicit S_hat@v = A1@v - A2@(yy@(A3@v)).  S_hat = A1 - A2·yy·A3."""
    t = _yy_mult(A_csr, U_csr, C_lu, A3_csr @ v)
    return A1_csr @ v - A2_csr @ t


def _build_B_blocks(A1_sp, A2_sp, A3_sp, A_csr, U_csr, C_lu, Subd_E, Subd_F, m):
    """B = (A1 - A2·yy·A3) ∘ TS_Blk as explicit block-diagonal dense blocks.

    B is block diagonal: one dense block per cluster of entities (edges/faces)
    that share DOFs.  Each block has exactly the same values as build_A1_masked
    (intra-entity Schur blocks, entity-overlap weights summed, cross-entity
    coupling zeroed).  No global m×m matrix is ever assembled — memory stays
    O(max block size²).  Returns list of (loc, B_dense) with loc = indices into
    sorted(m); empty list when there are no entities.
    """
    entities = [e for e in list(Subd_E) + list(Subd_F) if len(e) > 0]
    if len(m) == 0 or len(entities) == 0:
        return []
    m_sorted = np.sort(m)
    locs = []
    for entity in entities:
        e = np.unique(entity[np.isin(entity, m)])
        if len(e) == 0:
            continue
        locs.append(np.searchsorted(m_sorted, e))
    if len(locs) == 0:
        return []
    # ---- cluster entities that share DOFs (union-find over the overlap graph) ----
    n_ent = len(locs)
    dof_ent = {}
    for ei, loc in enumerate(locs):
        for d in loc:
            dof_ent.setdefault(int(d), []).append(ei)
    parent = list(range(n_ent))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    for ents in dof_ent.values():
        for k in range(1, len(ents)):
            _union(ents[0], ents[k])
    clusters = {}
    for ei in range(n_ent):
        clusters.setdefault(_find(ei), []).append(ei)
    # ---- one dense block per cluster ----
    col = U_csr.shape[1]
    blocks = []
    for ents in clusters.values():
        cloc = np.sort(np.unique(np.concatenate([locs[ei] for ei in ents])))
        cloc_pos = {int(d): i for i, d in enumerate(cloc)}
        Bc = np.zeros((len(cloc), len(cloc)))
        for ei in ents:
            loc = locs[ei]
            A1_e = A1_sp[loc][:, loc].toarray()      # |e|×|e| intra-entity block of A1
            A2e = A2_sp[loc, :]                      # |e|×n
            A3e = A3_sp[:, loc]                      # n×|e|
            A2e_A = A2e @ A_csr                      # |e|×n  (A2_e·A)
            part1 = (A2e_A @ A3e).toarray()          # A2_e·A·A3_e
            if col > 0:
                # yy = A + A·U·C^{-1}·U^T·A  ⇒  A2_e·yy·A3_e = part1 + part2
                A3e_A = A_csr @ A3e                  # n×|e|  (A·A3_e)
                WTeA3 = (U_csr.T @ A3e_A).toarray()  # col×|e|  (U^T·A·A3_e)
                Cinv_WTeA3 = C_lu.solve(WTeA3)       # col×|e|  (C^{-1}·U^T·A·A3_e)
                A2e_AU = (A2e_A @ U_csr).toarray()   # |e|×col  (A2_e·A·U)
                part2 = A2e_AU @ Cinv_WTeA3          # A2_e·A·U·C^{-1}·U^T·A·A3_e
            else:
                part2 = np.zeros_like(part1)
            B_e = (A1_e - part1 - part2)
            B_e = (B_e + B_e.T) / 2
            ci = np.array([cloc_pos[int(d)] for d in loc])
            Bc[np.ix_(ci, ci)] += B_e                # overlap weight = summed blocks
        Bc = (Bc + Bc.T) / 2
        blocks.append((cloc, Bc))
    return blocks


def _cholesky_blocks(B_blocks):
    """Cholesky of the explicit block-diagonal B per block, with a tiny
    regularization for numerically non-PD blocks.  Returns (L_blocks, comp_dofs)."""
    L_blocks, comp_dofs = [], []
    for cloc, Bd in B_blocks:
        k = len(cloc)
        if k == 0:
            continue
        if k == 1:
            v = Bd[0, 0]
            if v <= 0:
                v = 1e-10
            L_blocks.append(np.array([[np.sqrt(v)]]))
            comp_dofs.append(cloc)
            continue
        Bd = (Bd + Bd.T) / 2
        try:
            L = np.linalg.cholesky(Bd)
        except np.linalg.LinAlgError:
            w = np.linalg.eigvalsh(Bd)
            reg = max(1e-10, -w.min() + 1e-10)
            L = np.linalg.cholesky(Bd + reg * np.eye(k))
        L_blocks.append(L)
        comp_dofs.append(cloc)
    return L_blocks, comp_dofs


def _apply_L_invT(L_blocks, comp_dofs, y):
    out = y.copy()
    for L, dofs in zip(L_blocks, comp_dofs):
        if L.shape[0] == 1:
            out[dofs] = y[dofs] / L[0, 0]
        else:
            out[dofs] = solve_triangular(L.T, y[dofs], lower=False)
    return out


def _apply_L_inv(L_blocks, comp_dofs, y):
    out = y.copy()
    for L, dofs in zip(L_blocks, comp_dofs):
        if L.shape[0] == 1:
            out[dofs] = y[dofs] / L[0, 0]
        else:
            out[dofs] = solve_triangular(L, y[dofs], lower=True)
    return out


def _eps_krylov_schur(T_shell, L_blocks, comp_dofs, mu_val, m_size, col_offset, nev_cap=None):
    """SLEPc Krylov-Schur on T = L^{-1}·S_hat·L^{-T} (HEP), smallest eigenvalues.
    Adaptive nev: double until the largest converged eigenvalue is >= mu or the
    cap is reached.  nev_cap: hard limit on nev (None → child eigenvector count;
    e.g. L2 wants ~20, L1 can afford ~m/5).  Returns (Q, lam_list) with
    B-orthonormal Q columns and lam < mu."""
    if nev_cap is not None:
        max_nev = min(int(nev_cap), m_size)
    else:
        max_nev = col_offset if col_offset > 0 else min(200, m_size)
    nev = min(max(100, max_nev // 4), 600) if max_nev > 0 else 0
    nev = min(max(nev, 1), m_size)
    while True:
        ncv = min(max(2 * nev + 50, nev + 2), max(m_size, 1))
        eps = SLEPc.EPS().create()
        eps.setOperators(T_shell)
        eps.setProblemType(SLEPc.EPS.ProblemType.HEP)
        eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        eps.setDimensions(nev, ncv)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_MAGNITUDE)
        eps.setTolerances(tol=1e-8, max_it=10000)
        eps.solve()
        nconv = eps.getConverged()
        evals = sorted(eps.getEigenvalue(i).real for i in range(nconv))
        need_more = (nconv > 0 and nconv == nev and nev < max_nev
                     and evals[-1] < mu_val - 1e-8)
        if need_more:
            eps.destroy()
            nev = min(2 * nev, max_nev)
            continue
        Q_list, lam_list = [], []
        for i in range(nconv):
            lam = eps.getEigenvalue(i).real
            if lam < mu_val - 1e-8:
                # petsc4py 3.24: getEigenvector(i, Vr, Vi) fills Vr, returns None
                vr = PETSc.Vec().createSeq(m_size)
                eps.getEigenvector(i, vr, None)
                y = vr.getArray().copy()
                try:
                    v = _apply_L_invT(L_blocks, comp_dofs, y)
                    Q_list.append(v)
                    lam_list.append(lam)
                finally:
                    del y
                    vr.destroy()
        eps.destroy()
        break
    if len(Q_list) > 0:
        Q = np.column_stack(Q_list)
        order = np.argsort(lam_list)
        Q = Q[:, order]
        lam_list = np.array(lam_list)[order]
    else:
        Q = np.zeros((m_size, 0))
        lam_list = np.array([])
    return Q, lam_list


class _EpsMatCtx:
    """Python shell mat: T@v = L^{-1}·(S_hat·(L^{-T}·v))."""

    def __init__(self, A1, A2, A3, A, U, C_lu, L_blocks, comp_dofs):
        self.A1, self.A2, self.A3 = A1, A2, A3
        self.A, self.U, self.C_lu = A, U, C_lu
        self.L_blocks, self.comp_dofs = L_blocks, comp_dofs

    def mult(self, mat, x, y):
        v = x.getArray(readonly=True)
        try:
            t = _apply_L_invT(self.L_blocks, self.comp_dofs, v)
            s = _S_hat_mult(self.A1, self.A2, self.A3, self.A, self.U, self.C_lu, t)
            y.setArray(_apply_L_inv(self.L_blocks, self.comp_dofs, s))
        finally:
            del v  # petsc4py releases the read lock when the numpy buffer is freed


class _LocalSolverShellCtx:
    """Python shell mat for the level-2 interior solver: u = yy@r (implicit)."""

    def __init__(self, A_csr, U_csr, C_lu):
        self.A, self.U, self.C_lu = A_csr, U_csr, C_lu

    def mult(self, mat, x, y):
        r = x.getArray(readonly=True)
        try:
            y.setArray(_yy_mult(self.A, self.U, self.C_lu, r))
        finally:
            del r


class _R0TShellCtx:
    """Python shell mat for R_0T = -yy·A3 (II×GG): mult/multTranspose stay implicit.
    groups: per-subdomain dicts {m, n, A2_csr, A3_csr, A_csr, U_csr, C_lu}."""

    def __init__(self, groups, ii_map, gg_map):
        self.groups = groups
        self.ii_map = ii_map
        self.gg_map = gg_map
        self.n_II = int(np.sum(ii_map >= 0))
        self.n_GG = int(np.sum(gg_map >= 0))

    def mult(self, mat, x, y):
        xa = x.getArray(readonly=True)
        try:
            ya = np.zeros(self.n_II)
            for g in self.groups:
                m_loc = self.gg_map[g['m']]
                w = np.zeros(len(m_loc))
                ok = m_loc >= 0
                w[ok] = xa[m_loc[ok]]
                u = _yy_mult(g['A_csr'], g['U_csr'], g['C_lu'], g['A3_csr'] @ w)
                n_loc = self.ii_map[g['n']]
                ok_n = n_loc >= 0
                ya[n_loc[ok_n]] -= u[ok_n]
            y.setArray(ya)
        finally:
            del xa

    def multTranspose(self, mat, x, y):
        xa = x.getArray(readonly=True)
        try:
            ya = np.zeros(self.n_GG)
            for g in self.groups:
                n_loc = self.ii_map[g['n']]
                r = np.zeros(len(n_loc))
                ok = n_loc >= 0
                r[ok] = xa[n_loc[ok]]
                t = _yy_mult(g['A_csr'], g['U_csr'], g['C_lu'], r)
                v = g['A2_csr'] @ t
                m_loc = self.gg_map[g['m']]
                ok_m = m_loc >= 0
                ya[m_loc[ok_m]] -= v[ok_m]
            y.setArray(ya)
        finally:
            del xa


# ====================== Level 0 variable extraction ======================
print("\n=== Extracting Level 0 variables ===")
subdomain0_all_dofs = [None] * n_fine_subdomains
v0 = np.zeros(n_dofs_total, dtype=np.int32)
v2_0 = np.zeros(n_dofs_total, dtype=np.int32)

for sub_id in range(n_fine_subdomains):
    sub_tet_ids = np.where(group_assignment == sub_id)[0]
    sub_dofs = np.unique(np.concatenate([dofmap.cell_dofs(cell_idx) for cell_idx in sub_tet_ids]))
    subdomain0_all_dofs[sub_id] = sub_dofs
    v0[sub_dofs] += 1
    v2_0[sub_dofs] += (sub_id + 1)

GG_lev0 = np.setdiff1d(np.where(v0 >= 2)[0], dofs_bc)
II_lev0 = np.setdiff1d(np.where(v0 == 1)[0], dofs_bc)

# Extract Subd_E / Subd_F for Level 0 (TS_Blk 掩码局部构造，无需全局 TS_Blk)
Subd_E_lev0, Subd_F_lev0 = extract_entities_from_level(
    subdomain0_all_dofs, GG_lev0, II_lev0, dofs_bc, n_fine_subdomains)
print(f"Subd_E (Edge DOFs, {len(Subd_E_lev0)} edges)")
print(f"Subd_F (Face DOFs, {len(Subd_F_lev0)} faces)")

# Gamma / Interior for Level 0
Gamma_lev0 = [None] * n_fine_subdomains
Interior_lev0 = [None] * n_fine_subdomains
for sub_id in range(n_fine_subdomains):
    sub_dofs = subdomain0_all_dofs[sub_id]
    Interior_lev0[sub_id] = np.setdiff1d(np.setdiff1d(sub_dofs, dofs_bc), GG_lev0)
    Gamma_lev0[sub_id] = np.intersect1d(sub_dofs, GG_lev0)
print(f"GG (Interface DOFs): {len(GG_lev0)}")
print(f"II (Interior DOFs): {len(II_lev0)}")


# ====================== Level 0 subdomain computation ======================
print(f"\n=== Level 0 subdomain computation ({n_fine_subdomains} subdomains) ===")
Local_Solver0 = [None] * n_fine_subdomains
S_rows, S_cols, S_vals = [], [], []
g_rows, g_vals = [], []
D_inv_G_lev0 = [None] * n_fine_subdomains
S_Blk_rows_lev0 = [None] * n_fine_subdomains
S_Blk_cols_lev0 = [None] * n_fine_subdomains
S_Blk_vals_lev0 = [None] * n_fine_subdomains
A_QG_data_lev0 = [None] * n_fine_subdomains
flag_lev0 = np.zeros(n_fine_subdomains, dtype=np.int32)
f_val = PETSc.ScalarType(1.0)

mu_level0 = mu_list[0] if n_coarse_levels > 0 else 0.5

for sub_id in range(n_fine_subdomains):
    sub_tet_ids = np.where(group_assignment == sub_id)[0]
    m = Gamma_lev0[sub_id]
    n = Interior_lev0[sub_id]
    len_m, len_n = len(m), len(n)

    if len_m == 0:
        continue

    submesh, entity_map, vertex_map, geometry_map = mesh.create_submesh(msh, msh.topology.dim, sub_tet_ids)
    V_sub = fem.functionspace(submesh, ("Lagrange", deg))
    dofmap_sub = V_sub.dofmap

    # Build global→local DOF mapping (matching 5level_3d.py / 4level_3d.py)
    num_sub_cells = submesh.topology.index_map(3).size_local
    sub_cell_indices = np.arange(num_sub_cells, dtype=np.int32)
    cell_map = entity_map.sub_topology_to_topology(sub_cell_indices, inverse=False)
    sub_rho_values = cell_rho_values[cell_map]

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
    m_local = all_local[:len_m]
    n_local = all_local[len_m:]

    u = ufl.TrialFunction(V_sub)
    v = ufl.TestFunction(V_sub)
    DG0 = fem.functionspace(submesh, ("DG", 0))
    rho_func = fem.Function(DG0)
    rho_func.x.array[:] = sub_rho_values

    a_local = fem.form(ufl.inner(rho_func * ufl.grad(u), ufl.grad(v)) * ufl.dx)
    L_local = fem.form(ufl.inner(f_val, v) * ufl.dx)
    A_local_sp = fem.assemble_matrix(a_local)
    b_local = fem.assemble_vector(L_local)

    A_local = PETSc.Mat().createAIJ(size=(A_local_sp.index_map(0).size_local, A_local_sp.index_map(1).size_local),
                                     csr=(A_local_sp.indptr.astype(PETSc.IntType),
                                          A_local_sp.indices.astype(PETSc.IntType),
                                          A_local_sp.data.astype(PETSc.ScalarType)))

    is_m_local = PETSc.IS().createGeneral(m_local.astype(PETSc.IntType), comm=PETSc.COMM_WORLD)
    is_n_local = PETSc.IS().createGeneral(n_local.astype(PETSc.IntType), comm=PETSc.COMM_WORLD)

    A1 = A_local.createSubMatrix(is_m_local, is_m_local)
    A1.assemble()
    A2 = A_local.createSubMatrix(is_m_local, is_n_local)
    A2.assemble()
    A3 = A_local.createSubMatrix(is_n_local, is_m_local)
    A3.assemble()
    A4 = A_local.createSubMatrix(is_n_local, is_n_local)
    A4.assemble()

    ksp_A4 = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
    ksp_A4.setOperators(A4)
    ksp_A4.setType(PETSc.KSP.Type.PREONLY)
    ksp_A4.getPC().setType(PETSc.PC.Type.CHOLESKY)
    ksp_A4.setUp()
    Local_Solver0[sub_id] = ksp_A4

    indptr_n, indices_n, values_n = A3.getValuesCSR()
    A3_csr = csr_matrix((values_n, indices_n, indptr_n), shape=(len_n, len_m))
    A3_dense = A3_csr.toarray()
    del A3_csr
    indptr_a4, indices_a4, values_a4 = A4.getValuesCSR()
    A4_csc = csr_matrix((values_a4, indices_a4, indptr_a4), shape=(len_n, len_n)).tocsc()
    solutions = spsolve(A4_csc, A3_dense)
    del A4_csc, A3_dense
    A4_inv_A3_data = solutions.ravel()
    del solutions
    A4_inv_A3_indptr = np.arange(0, len_n * len_m + 1, len_m, dtype=PETSc.IntType)
    A4_inv_A3 = PETSc.Mat().createAIJ(
        size=(len_n, len_m),
        csr=(A4_inv_A3_indptr,
             np.tile(np.arange(len_m), len_n).astype(PETSc.IntType),
             A4_inv_A3_data.astype(PETSc.ScalarType)),
        comm=PETSc.COMM_WORLD
    )

    Schur = A1 - A2 * A4_inv_A3

    b_vec = PETSc.Vec().createWithArray(b_local.array)
    b_n = b_vec.getSubVector(is_n_local)
    ksp_A4.solve(b_n, b_n)
    g_vec = b_vec.getSubVector(is_m_local) - A2 * b_n
    g_rows.append(all_sub_dofs[is_m_local])
    g_vals.append(g_vec.getArray())

    Schur_np = petsc_to_scipy(Schur)
    Schur_np = (Schur_np + Schur_np.T) / 2

    # 局部构造 A1_mat = Schur ∘ TS_Blk（无需全局 TS_Blk）
    Schur_sp = csr_matrix(Schur_np)
    A1_mat = build_A1_masked(Schur_sp, Subd_E_lev0, Subd_F_lev0, m).toarray()

    D, Q = eigh(Schur_np, A1_mat)

    A1_csr = csr_matrix(A1_mat)
    indptr, indices, values = A1_csr.indptr, A1_csr.indices, A1_csr.data
    row_counts = np.diff(indptr)
    row_indices_local = np.repeat(np.arange(len(row_counts)), row_counts)
    S_Blk_rows_lev0[sub_id] = m[row_indices_local]
    S_Blk_cols_lev0[sub_id] = m[indices]
    S_Blk_vals_lev0[sub_id] = values

    # Schur CSR for global system
    Schur_csr = csr_matrix(Schur_np)
    indptr_s, indices_s, values_s = Schur_csr.indptr, Schur_csr.indices, Schur_csr.data
    row_counts_s = np.diff(indptr_s)
    row_indices_local_s = np.repeat(np.arange(len(row_counts_s)), row_counts_s)
    S_rows.append(m[row_indices_local_s])
    S_cols.append(m[indices_s])
    S_vals.append(values_s)

    eigens_idx = np.where(D < mu_level0)[0]
    Q = Q[:, eigens_idx]
    D = 1 - D[eigens_idx]
    flag_lev0[sub_id] = len(eigens_idx)

    if len(eigens_idx) == 0:
        A_QG_data_lev0[sub_id] = {'m': m, 'A1Q': np.array([]).reshape(len(m), 0), 'n_eigen': 0}
        D_inv_G_lev0[sub_id] = np.array([])
    else:
        A1Q = A1_csr @ Q
        A_QG_data_lev0[sub_id] = {'m': m, 'A1Q': A1Q, 'n_eigen': len(eigens_idx)}
        D = 1.0 / D
        D_inv_G_lev0[sub_id] = D

    is_m_local.destroy()
    is_n_local.destroy()
    A_local.destroy()
    A2.destroy()
    A3.destroy()
    A4.destroy()
    A4_inv_A3.destroy()
    g_vec.destroy()
    gc.collect()

print(f"Level 0 sum(flag) = {sum(flag_lev0)}")

# ====================== Level 0 global assembly ======================
print("\n=== Level 0 global assembly ===")
S_rows_full, S_cols_full, S_vals_full = np.concatenate(S_rows), np.concatenate(S_cols), np.concatenate(S_vals)
S_csr = csr_matrix((S_vals_full, (S_rows_full, S_cols_full)), shape=(n_dofs_total, n_dofs_total))
S = PETSc.Mat().createAIJ(size=(n_dofs_total, n_dofs_total),
                          csr=(S_csr.indptr.astype(PETSc.IntType), S_csr.indices.astype(PETSc.IntType),
                               S_csr.data.astype(PETSc.ScalarType)))
is_GG_level0 = PETSc.IS().createGeneral(GG_lev0.astype(PETSc.IntType), comm=PETSc.COMM_WORLD)
S = S.createSubMatrix(is_GG_level0, is_GG_level0)
S.assemble()

g_rows_full, g_vals_full = np.concatenate(g_rows), np.concatenate(g_vals)
g_full = PETSc.Vec().createSeq(n_dofs_total)
g_full.setValues(g_rows_full.astype(PETSc.IntType), g_vals_full.astype(PETSc.ScalarType), addv=PETSc.InsertMode.ADD_VALUES)
g_full.assemble()
g = PETSc.Vec().createWithArray(g_full.getArray()[GG_lev0].astype(PETSc.ScalarType))


# ====================== Multi-level coarsening ======================
current_Gamma = Gamma_lev0
current_GG = GG_lev0
current_II = II_lev0

level_data_list = []

for level_idx in range(1, n_levels):
    target_n = n_subdomains_list[level_idx]
    mu_val = mu_list[level_idx] if level_idx < len(mu_list) else mu_list[-1]
    n_prev_groups = len(current_Gamma)

    print(f"\n=== Level {level_idx} coarsening: {n_prev_groups} -> {target_n} groups ===")

    adj = [[] for _ in range(n_prev_groups)]
    for i in range(n_prev_groups):
        gamma_i = current_Gamma[i]
        for j in range(i + 1, n_prev_groups):
            if np.intersect1d(gamma_i, current_Gamma[j], assume_unique=True).size > 0:
                adj[i].append(j)
                adj[j].append(i)

    safe_n = min(target_n, n_prev_groups)
    _, coarse_assignment = pymetis.part_graph(safe_n, adjacency=adj)
    coarse_assignment = np.array(coarse_assignment)

    print(f"  Coarse groups: {safe_n}")
    n_c = safe_n
    coarse_subdomain_dofs = [None] * n_c
    coarse_v = np.zeros(n_dofs_total, dtype=np.int32)
    coarse_v2 = np.zeros(n_dofs_total, dtype=np.int32)

    for cid in range(n_c):
        child_ids = np.where(coarse_assignment == cid)[0]
        dofs_set = set()
        for child_id in child_ids:
            dofs_set.update(current_Gamma[child_id])
        grp_dofs = np.array(list(dofs_set), dtype=np.int32)
        coarse_subdomain_dofs[cid] = grp_dofs
        coarse_v[grp_dofs] += 1
        coarse_v2[grp_dofs] += (cid + 1)

    coarse_GG = np.setdiff1d(np.where(coarse_v >= 2)[0], dofs_bc)
    coarse_II = np.setdiff1d(np.where(coarse_v == 1)[0], dofs_bc)

    coarse_Subd_E, coarse_Subd_F = extract_entities_from_level(
        coarse_subdomain_dofs, coarse_GG, coarse_II, dofs_bc, n_c)

    coarse_Gamma_list = [None] * n_c
    coarse_Interior_list = [None] * n_c
    for cid in range(n_c):
        grp_dofs = coarse_subdomain_dofs[cid]
        coarse_Interior_list[cid] = np.setdiff1d(np.setdiff1d(grp_dofs, dofs_bc), coarse_GG)
        coarse_Gamma_list[cid] = np.intersect1d(grp_dofs, coarse_GG)

    print(f"  coarse_GG: {len(coarse_GG)}, coarse_II: {len(coarse_II)}")
    print(f"  Subd_E: {len(coarse_Subd_E)} edges, Subd_F: {len(coarse_Subd_F)} faces")

    level_data_list.append({
        'n_c': n_c,
        'subdomain_all_dofs': coarse_subdomain_dofs,
        'Gamma': coarse_Gamma_list,
        'Interior': coarse_Interior_list,
        'GG': coarse_GG,
        'II': coarse_II,
        'Subd_E': coarse_Subd_E,
        'Subd_F': coarse_Subd_F,
        'mu': mu_val,
        'group_assignment': coarse_assignment
    })

    current_Gamma = coarse_Gamma_list
    current_GG = coarse_GG
    current_II = coarse_II

print(f"\n=== Coarsening done: {n_levels} total levels ===")


# ====================== Compute each coarse level ======================
def compute_coarse_level(prev_S_Blk_rows, prev_S_Blk_cols, prev_S_Blk_vals,
                          prev_A_QG_data, prev_D_inv_G,
                          cur_data, prev_Subd_E, prev_Subd_F,
                          level_name='', use_krylov=False, krylov_nev_cap=None):
    n_c = cur_data['n_c']
    subdomain_all_dofs = cur_data['subdomain_all_dofs']
    Gamma_list = cur_data['Gamma']
    Interior_list = cur_data['Interior']
    Subd_E = cur_data['Subd_E']
    Subd_F = cur_data['Subd_F']
    mu_val = cur_data['mu']
    group_assignment = cur_data['group_assignment']

    Local_Solver = [None] * n_c
    A_QG_data = [None] * n_c
    flag = np.zeros(n_c, dtype=np.int32)
    R0T_rows, R0T_cols, R0T_vals = [], [], []
    krylov_groups = []
    D_inv_G = [None] * n_c
    S_Blk_rows = [None] * n_c
    S_Blk_cols = [None] * n_c
    S_Blk_vals = [None] * n_c

    for cid in range(n_c):
        t_cid_start = time.time()
        child_ids = np.where(group_assignment == cid)[0]
        if len(child_ids) == 0:
            continue

        s = subdomain_all_dofs[cid]
        print(f"  [{level_name}] Group {cid + 1}/{n_c} ({len(s)} DOFs, {len(child_ids)} children)", flush=True)
        m = Gamma_list[cid]
        n = Interior_list[cid]
        len_m, len_n = len(m), len(n)

        if len_n == 0 and len_m == 0:
            continue

        # Build local S_Blk from children (always, matching 5level_3d.py even when m empty)
        good_S = [(prev_S_Blk_rows[cid], prev_S_Blk_cols[cid], prev_S_Blk_vals[cid])
                  for cid in child_ids
                  if prev_S_Blk_rows[cid] is not None and len(prev_S_Blk_rows[cid]) > 0]
        if len(good_S) == 0:
            continue

        grp_S_rows = np.concatenate([e[0] for e in good_S])
        grp_S_cols = np.concatenate([e[1] for e in good_S])
        grp_S_vals = np.concatenate([e[2] for e in good_S])
        S_Blk_local_csr = csr_matrix((grp_S_vals, (grp_S_rows, grp_S_cols)), shape=(n_dofs_total, n_dofs_total))
        S_Blk_local = PETSc.Mat().createAIJ(
            size=S_Blk_local_csr.shape,
            csr=(S_Blk_local_csr.indptr.astype(PETSc.IntType),
                 S_Blk_local_csr.indices.astype(PETSc.IntType),
                 S_Blk_local_csr.data.astype(PETSc.ScalarType)),
            comm=PETSc.COMM_WORLD
        )
        S_Blk_local.assemble()

        # Build local A_QG
        local_A_QG_rows, local_A_QG_cols, local_A_QG_vals = [], [], []
        col_offset = 0
        for child_id in child_ids:
            data = prev_A_QG_data[child_id]
            if data is None or data['n_eigen'] == 0:
                continue
            m_indices, A1Q, n_eigen = data['m'], data['A1Q'], data['n_eigen']
            local_A_QG_rows.append(np.repeat(m_indices, n_eigen))
            local_A_QG_cols.append(np.tile(np.arange(n_eigen) + col_offset, len(m_indices)))
            local_A_QG_vals.append(A1Q.ravel())
            col_offset += n_eigen

        if col_offset == 0:
            # No eigenvalue data from children → no coarse space.
            # Don't continue: InvS_Blk_local + R0T are still needed (matching 2D pattern).
            A_QG_data[cid] = {'m': np.array([], dtype=np.int32), 'A1Q': np.array([]).reshape(0, 0), 'n_eigen': 0}
            D_inv_G[cid] = np.array([], dtype=PETSc.ScalarType)
            flag[cid] = 0
            S_Blk_rows[cid] = np.array([], dtype=np.int32)
            S_Blk_cols[cid] = np.array([], dtype=np.int32)
            S_Blk_vals[cid] = np.array([], dtype=PETSc.ScalarType)

        if len(local_A_QG_rows) > 0:
            all_rows, all_cols, all_vals = map(np.concatenate, [local_A_QG_rows, local_A_QG_cols, local_A_QG_vals])
        else:
            all_rows, all_cols, all_vals = np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=PETSc.ScalarType)
        A_QG_local_csr = csr_matrix((all_vals, (all_rows, all_cols)), shape=(n_dofs_total, col_offset))

        A_QG_local = PETSc.Mat().createAIJ(
            size=A_QG_local_csr.shape,
            csr=(A_QG_local_csr.indptr.astype(PETSc.IntType),
                 A_QG_local_csr.indices.astype(PETSc.IntType),
                 A_QG_local_csr.data.astype(PETSc.ScalarType)),
            comm=PETSc.COMM_WORLD
        )
        A_QG_local.assemble()

        # Build local D_inv and D (D_inv = 1/(1-λ), D = 1-λ)
        D_inv_child_list = [prev_D_inv_G[cid] for cid in child_ids
                            if prev_D_inv_G[cid] is not None and len(prev_D_inv_G[cid]) > 0]
        if len(D_inv_child_list) == 0:
            D_local_csr = csr_matrix((0, 0))
            D_inv_petsc = PETSc.Mat().createAIJ(size=(0, 0))
            D_inv_petsc.assemble()
        else:
            D_inv_child_vec = np.concatenate(D_inv_child_list)
            D_child_vec = 1.0 / D_inv_child_vec
            indices = np.arange(len(D_inv_child_vec), dtype=np.int32)
            D_local_csr = csr_matrix((D_child_vec, (indices, indices)),
                                     shape=(len(D_inv_child_vec), len(D_inv_child_vec)))
            D_inv_csr = csr_matrix((D_inv_child_vec, (indices, indices)),
                                   shape=(len(D_inv_child_vec), len(D_inv_child_vec)))
            D_inv_petsc = PETSc.Mat().createAIJ(
                size=(len(D_inv_child_vec), len(D_inv_child_vec)),
                csr=(D_inv_csr.indptr.astype(PETSc.IntType),
                     D_inv_csr.indices.astype(PETSc.IntType),
                     D_inv_csr.data.astype(PETSc.ScalarType)),
                comm=PETSc.COMM_WORLD
            )
            D_inv_petsc.assemble()

        # ---- InvS_Blk_local (interior blocks) ----
        rows_iblk, cols_iblk, vals_iblk = [], [], []
        s_set = set(s)
        for entity_list in [prev_Subd_E, prev_Subd_F]:
            for entity in entity_list:
                entity = np.atleast_1d(entity).astype(PETSc.IntType)
                if not (set(entity) & s_set):
                    continue
                entity_filt = np.array([d for d in entity if d in s_set], dtype=PETSc.IntType)
                if len(entity_filt) == 0:
                    continue
                is_entity = PETSc.IS().createGeneral(entity_filt, comm=PETSc.COMM_WORLD)
                S_sub = S_Blk_local.createSubMatrix(is_entity, is_entity)
                S_sub.assemble()
                I_s, J_s, V_s = S_sub.getValuesCSR()
                S_dense = csr_matrix((V_s, J_s, I_s), shape=S_sub.getSize()).toarray()
                inv = np.linalg.inv(S_dense)
                n_entity = len(entity_filt)
                rows_iblk.extend(np.repeat(entity_filt, n_entity))
                cols_iblk.extend(np.tile(entity_filt, n_entity))
                vals_iblk.extend(inv.ravel())
                is_entity.destroy()
                S_sub.destroy()

        InvS_Blk_scipy = csr_matrix((vals_iblk, (rows_iblk, cols_iblk)), shape=(n_dofs_total, n_dofs_total))
        if use_krylov:
            # ===== Krylov-Schur branch (large level): never form dense yy / Schur / eigh =====
            A_csr = InvS_Blk_scipy[n][:, n].tocsr()     # n×n block-diagonal inverse
            U_csr = A_QG_local_csr[n]                   # n×col (child eigenvectors)
            col = U_csr.shape[1]
            if col > 0:
                W_csr = A_csr @ U_csr
                C_csr = D_inv_csr - U_csr.T @ W_csr     # coarse Gramian, col×col
                C_csr = ((C_csr + C_csr.T) * 0.5).tocsc()
                try:
                    C_lu = splu(C_csr)
                except RuntimeError:
                    C_lu = splu((C_csr + 1e-10 * identity(col, format='csc')).tocsc())
            else:
                C_lu = None

            # Local_Solver: implicit yy (needed by the V-cycle even when len_m == 0)
            if len_n > 0:
                ls_ctx = _LocalSolverShellCtx(A_csr, U_csr, C_lu)
                yy_shell = PETSc.Mat().createPython(size=(len_n, len_n), comm=PETSc.COMM_WORLD)
                yy_shell.setPythonContext(ls_ctx)
                yy_shell.assemble()
                Local_Solver[cid] = yy_shell

            if len_m == 0:
                S_Blk_rows[cid] = np.array([], dtype=np.int32)
                S_Blk_cols[cid] = np.array([], dtype=np.int32)
                S_Blk_vals[cid] = np.array([], dtype=PETSc.ScalarType)
                A_QG_data[cid] = {'m': np.array([], dtype=np.int32), 'A1Q': np.array([]).reshape(0, 0), 'n_eigen': 0}
                D_inv_G[cid] = np.array([], dtype=PETSc.ScalarType)
                flag[cid] = 0
            else:
                # ---- sparse Schur blocks (same definitions as the dense path) ----
                A_QG_m = A_QG_local_csr[m]
                A_QG_n = A_QG_local_csr[n]
                S_mm = S_Blk_local_csr[m][:, m]
                S_mn = S_Blk_local_csr[m][:, n]
                A1_sp = (S_mm - A_QG_m @ D_local_csr @ A_QG_m.T).tocsr()
                A2_sp = (S_mn - A_QG_m @ D_local_csr @ A_QG_n.T).tocsr()
                A3_sp = A2_sp.T
                A1_sp = ((A1_sp + A1_sp.T) * 0.5).tocsr()
                A2_sp = A2_sp.tocsr()

                # B = (A1 - A2·yy·A3) ∘ TS_Blk as explicit block-diagonal blocks
                B_blocks = _build_B_blocks(A1_sp, A2_sp, A3_sp, A_csr, U_csr, C_lu,
                                           Subd_E, Subd_F, m)

                # S_Blk output = B entries (same values as the original A1_mat)
                sblk_rows, sblk_cols, sblk_vals = [], [], []
                for cloc, Bc in B_blocks:
                    n_b = len(cloc)
                    sblk_rows.append(np.repeat(m[cloc], n_b))
                    sblk_cols.append(np.tile(m[cloc], n_b))
                    sblk_vals.append(Bc.ravel())
                if len(sblk_rows) > 0:
                    S_Blk_rows[cid] = np.concatenate(sblk_rows)
                    S_Blk_cols[cid] = np.concatenate(sblk_cols)
                    S_Blk_vals[cid] = np.concatenate(sblk_vals)
                else:
                    S_Blk_rows[cid] = np.array([], dtype=np.int32)
                    S_Blk_cols[cid] = np.array([], dtype=np.int32)
                    S_Blk_vals[cid] = np.array([], dtype=PETSc.ScalarType)

                # Cholesky per block of B: B = L·L^T
                L_blocks, comp_dofs = _cholesky_blocks(B_blocks)

                # SLEPc Krylov-Schur on T = L^{-1}·S_hat·L^{-T}, smallest eigenvalues
                T_ctx = _EpsMatCtx(A1_sp, A2_sp, A3_sp, A_csr, U_csr, C_lu, L_blocks, comp_dofs)
                T_shell = PETSc.Mat().createPython(size=(len_m, len_m), comm=PETSc.COMM_WORLD)
                T_shell.setPythonContext(T_ctx)
                T_shell.assemble()
                # nev budget: L1 wants ~m/3 eigenvalues, coarser levels ~200
                if krylov_nev_cap is None:
                    local_nev_cap = max(len_m // 3, 20)
                else:
                    local_nev_cap = krylov_nev_cap
                Q, lam_list = _eps_krylov_schur(T_shell, L_blocks, comp_dofs, mu_val, len_m, col,
                                                nev_cap=local_nev_cap)
                T_shell.destroy()

                k = Q.shape[1]
                flag[cid] = k
                if k == 0:
                    A_QG_data[cid] = {'m': m, 'A1Q': np.array([]).reshape(len(m), 0), 'n_eigen': 0}
                    D_inv_G[cid] = np.array([])
                else:
                    # A1Q = B @ Q, block by block (never assemble the global B)
                    A1Q = np.zeros((len_m, k))
                    for cloc, Bc in B_blocks:
                        A1Q[cloc] = Bc @ Q[cloc]
                    A_QG_data[cid] = {'m': m, 'A1Q': A1Q, 'n_eigen': k}
                    D_inv_G[cid] = 1.0 / (1.0 - np.asarray(lam_list))

                # R_0T context for the implicit R_0T = -yy·A3 (assembled by the main loop)
                if len_n > 0:
                    krylov_groups.append({'m': m, 'n': n, 'A2_csr': A2_sp, 'A3_csr': A3_sp,
                                          'A_csr': A_csr, 'U_csr': U_csr, 'C_lu': C_lu})
        else:
            InvS_Blk = PETSc.Mat().createAIJ(
                size=InvS_Blk_scipy.shape,
                csr=(InvS_Blk_scipy.indptr.astype(PETSc.IntType),
                     InvS_Blk_scipy.indices.astype(PETSc.IntType),
                     InvS_Blk_scipy.data.astype(PETSc.ScalarType)),
                comm=PETSc.COMM_WORLD
            )
            InvS_Blk.assemble()

            is_n = PETSc.IS().createGeneral(n.astype(PETSc.IntType), comm=PETSc.COMM_WORLD)
            InvS_Blk_local = InvS_Blk.createSubMatrix(is_n, is_n)
            InvS_Blk_local.assemble()

            # ---- Core computation ----
            U = A_QG_local.createSubMatrix(is_n, None)
            U.assemble()

            W = InvS_Blk_local * U
            Global_Solver = D_inv_petsc - U.transposeMatMult(W)

            WT = W.copy().transpose()
            WT.assemble()

            n_rows, n_cols = WT.getSize()
            I_gs, J_gs, V_gs = Global_Solver.getValuesCSR()
            GS_dense = csr_matrix((V_gs, J_gs, I_gs), shape=(n_rows, n_rows)).toarray()
            I_wt, J_wt, V_wt = WT.getValuesCSR()
            WT_dense = csr_matrix((V_wt, J_wt, I_wt), shape=(n_rows, n_cols)).toarray()
            solutions = np.linalg.solve(GS_dense, WT_dense)
            del GS_dense, WT_dense

            X_data = solutions.ravel()
            X_csr = csr_matrix((X_data,
                                (np.repeat(np.arange(n_rows), n_cols),
                                 np.tile(np.arange(n_cols), n_rows))),
                               shape=(n_rows, n_cols))
            X = PETSc.Mat().createAIJ(
                size=(n_rows, n_cols),
                csr=(X_csr.indptr.astype(PETSc.IntType),
                     X_csr.indices.astype(PETSc.IntType),
                     X_csr.data.astype(PETSc.ScalarType)),
                comm=PETSc.COMM_WORLD
            )
            X.assemble()

            yy = InvS_Blk_local + W * X
            Local_Solver[cid] = yy

            if len_m > 0:
                # ---- Schur complement for eigenvalue ----
                S_mm = S_Blk_local_csr[m][:, m]
                S_mn = S_Blk_local_csr[m][:, n]

                A_QG_m = A_QG_local_csr[m]
                A_QG_n = A_QG_local_csr[n]

                A1 = S_mm - A_QG_m @ D_local_csr @ A_QG_m.T
                A2 = S_mn - A_QG_m @ D_local_csr @ A_QG_n.T
                A3 = A2.T

                indptr_yy, indices_yy, values_yy = yy.getValuesCSR()
                yy_csr = csr_matrix((values_yy, indices_yy, indptr_yy), shape=yy.getSize())

                Schur = A1 - A2 @ yy_csr @ A3

                Schur_sp = (Schur + Schur.T) / 2
                Schur_dense = Schur_sp.toarray()
                A1_mat = build_A1_masked(Schur_sp, Subd_E, Subd_F, m).toarray()
                D, Q = eigh(Schur_dense, A1_mat)
                A1_csr = csr_matrix(A1_mat)
                del Schur_dense, A1_mat
                indptr, indices, values = A1_csr.indptr, A1_csr.indices, A1_csr.data
                row_counts = np.diff(indptr)
                row_indices_local = np.repeat(np.arange(len(row_counts)), row_counts)
                S_Blk_rows[cid] = m[row_indices_local]
                S_Blk_cols[cid] = m[indices]
                S_Blk_vals[cid] = values

                # R0T = -A2^T * yy (negative)
                yy_A3 = yy_csr @ A3
                yy_A3_mat = PETSc.Mat().createAIJ(
                    size=yy_A3.shape,
                    csr=(yy_A3.indptr.astype(PETSc.IntType),
                         yy_A3.indices.astype(PETSc.IntType),
                         yy_A3.data.astype(PETSc.ScalarType)),
                    comm=PETSc.COMM_WORLD
                )
                indptr_yy, indices_yy, values_yy = yy_A3_mat.getValuesCSR()
                row_counts_yy = np.diff(indptr_yy)
                row_indices_local_yy = np.repeat(np.arange(len(row_counts_yy)), row_counts_yy)
                R0T_rows.append(n[row_indices_local_yy])
                R0T_cols.append(m[indices_yy])
                R0T_vals.append(-values_yy)
                yy_A3_mat.destroy()

                eigens_idx = np.where(D < mu_val)[0]
                Q = Q[:, eigens_idx]
                D = 1 - D[eigens_idx]
                flag[cid] = len(eigens_idx)

                if len(eigens_idx) == 0:
                    A_QG_data[cid] = {'m': m, 'A1Q': np.array([]).reshape(len(m), 0), 'n_eigen': 0}
                    D_inv_G[cid] = np.array([])
                else:
                    A1Q = A1_csr @ Q
                    A_QG_data[cid] = {'m': m, 'A1Q': A1Q, 'n_eigen': len(eigens_idx)}
                    D = 1.0 / D
                    D_inv_G[cid] = D

            else:
                # len_m == 0: no eigenvalue, save empty data (matching 5level_3d.py level 4)
                S_Blk_rows[cid] = np.array([], dtype=np.int32)
                S_Blk_cols[cid] = np.array([], dtype=np.int32)
                S_Blk_vals[cid] = np.array([], dtype=PETSc.ScalarType)
                A_QG_data[cid] = {'m': np.array([], dtype=np.int32), 'A1Q': np.array([]).reshape(0, 0), 'n_eigen': 0}
                D_inv_G[cid] = np.array([], dtype=PETSc.ScalarType)
                flag[cid] = 0

        t_cid_elapsed = time.time() - t_cid_start
        print(f"  [{level_name}] Group {cid + 1}/{n_c} done ({t_cid_elapsed:.1f}s, {sum(flag)} EVs)", flush=True)
        gc.collect()
        S_Blk_local.destroy()
        A_QG_local.destroy()
        if not use_krylov:
            InvS_Blk.destroy()
            InvS_Blk_local.destroy()
            U.destroy()
            W.destroy()
            WT.destroy()
            Global_Solver.destroy()
            X.destroy()
            is_n.destroy()

    krylov_r0t_groups = krylov_groups if use_krylov else None
    return (Local_Solver, A_QG_data, flag, R0T_rows, R0T_cols, R0T_vals,
            D_inv_G, S_Blk_rows, S_Blk_cols, S_Blk_vals, krylov_r0t_groups)


# ====================== Build InvS_Blk on GG ======================
def build_InvS_Blk_on_GG(S_Blk_rows, S_Blk_cols, S_Blk_vals, Subd_E, Subd_F, GG):
    """Build the block inverse of S_Blk restricted to GG.
    Returns (Inv_GG, S_rows, S_cols, S_vals) or (empty_0x0, [], [], []) when GG is empty."""
    valid = [i for i, r in enumerate(S_Blk_rows) if r is not None and hasattr(r, '__len__') and len(r) > 0]
    if len(valid) == 0 or GG is None or len(GG) == 0:
        empty_csr = csr_matrix((0, 0))
        empty_mat = PETSc.Mat().createAIJ(
            size=(0, 0),
            csr=(empty_csr.indptr.astype(PETSc.IntType),
                 empty_csr.indices.astype(PETSc.IntType),
                 empty_csr.data.astype(PETSc.ScalarType)),
            comm=PETSc.COMM_WORLD
        )
        return empty_mat, np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=PETSc.ScalarType)

    S_rows = np.concatenate([S_Blk_rows[i] for i in valid])
    S_cols = np.concatenate([S_Blk_cols[i] for i in valid])
    S_vals = np.concatenate([S_Blk_vals[i] for i in valid])
    # scipy 稀疏切片提取 entity 块，避免全局 PETSc S_petsc + 上万次 createSubMatrix
    S_csr = csr_matrix((S_vals, (S_rows, S_cols)), shape=(n_dofs_total, n_dofs_total))

    rows, cols, vals = [], [], []
    for entity_list in [Subd_E, Subd_F]:
        for entity in entity_list:
            entity = np.atleast_1d(entity).astype(PETSc.IntType)
            n_entity = len(entity)
            S_dense = S_csr[entity][:, entity].toarray()
            inv = np.linalg.inv(S_dense)
            rows.extend(np.repeat(entity, n_entity))
            cols.extend(np.tile(entity, n_entity))
            vals.extend(inv.ravel())

    # 直接构造 GG×GG（entity DOF ⊆ GG，三元组过滤映射即可，无需全局 Inv_petsc）
    if len(GG) > 0:
        gg_map = np.full(n_dofs_total, -1, dtype=np.int32)
        gg_map[GG] = np.arange(len(GG))
        rows_a = np.asarray(rows, dtype=np.int32)
        cols_a = np.asarray(cols, dtype=np.int32)
        vals_a = np.asarray(vals, dtype=PETSc.ScalarType)
        keep = (gg_map[rows_a] >= 0) & (gg_map[cols_a] >= 0)
        Inv_GG_csr = csr_matrix((vals_a[keep],
                                 (gg_map[rows_a[keep]], gg_map[cols_a[keep]])),
                                shape=(len(GG), len(GG)))
        Inv_GG = PETSc.Mat().createAIJ(
            size=(len(GG), len(GG)),
            csr=(Inv_GG_csr.indptr.astype(PETSc.IntType),
                 Inv_GG_csr.indices.astype(PETSc.IntType),
                 Inv_GG_csr.data.astype(PETSc.ScalarType)),
            comm=PETSc.COMM_WORLD
        )
        Inv_GG.assemble()
    else:
        empty_csr = csr_matrix((0, 0))
        Inv_GG = PETSc.Mat().createAIJ(
            size=(0, 0),
            csr=(empty_csr.indptr.astype(PETSc.IntType),
                 empty_csr.indices.astype(PETSc.IntType),
                 empty_csr.data.astype(PETSc.ScalarType)),
            comm=PETSc.COMM_WORLD
        )
        Inv_GG.assemble()

    return Inv_GG, S_rows, S_cols, S_vals


# ====================== Compute all coarse levels ======================
all_Local_Solver = [Local_Solver0]
all_flag = [flag_lev0]
S_Blk_rows_list = [S_Blk_rows_lev0]
S_Blk_cols_list = [S_Blk_cols_lev0]
S_Blk_vals_list = [S_Blk_vals_lev0]
A_QG_data_list = [A_QG_data_lev0]
D_inv_G_list = [D_inv_G_lev0]

prev_Subd_E = Subd_E_lev0
prev_Subd_F = Subd_F_lev0

# V-cycle data
all_R_0T = []
all_InvS_Blk = []
all_W = []
all_Global_Solver = []
all_is_GG = []
all_is_II = []
all_Interior_list = []

for level_idx in range(n_coarse_levels):
    data = level_data_list[level_idx]
    print(f"\n=== Computing Level {level_idx + 1} ({data['n_c']} groups) ===")
    print(f"  >>> MU_USED = {data['mu']}  (mu_list={mu_list})")

    Local_Solver_lvl, A_QG_data_lvl, flag_lvl, \
        R0T_rows, R0T_cols, R0T_vals, \
        D_inv_G_lvl, S_Blk_rows_lvl, S_Blk_cols_lvl, S_Blk_vals_lvl, \
        krylov_r0t_groups = \
        compute_coarse_level(
            S_Blk_rows_list[-1], S_Blk_cols_list[-1], S_Blk_vals_list[-1],
            A_QG_data_list[-1], D_inv_G_list[-1],
            data, prev_Subd_E, prev_Subd_F,
            level_name=f'L{level_idx + 1}',
            use_krylov=True,
            krylov_nev_cap=(None if level_idx < n_coarse_levels - 1 else 600))

    all_Local_Solver.append(Local_Solver_lvl)
    all_flag.append(flag_lvl)
    S_Blk_rows_list.append(S_Blk_rows_lvl)
    S_Blk_cols_list.append(S_Blk_cols_lvl)
    S_Blk_vals_list.append(S_Blk_vals_lvl)
    A_QG_data_list.append(A_QG_data_lvl)
    D_inv_G_list.append(D_inv_G_lvl)

    # Build R_0T — always build a valid PETSc Mat (even empty), matching 2D pattern
    # (2D's assemble_global_matrices always builds R_0T, never None)
    is_GG_lvl = PETSc.IS().createGeneral(data['GG'].astype(PETSc.IntType), comm=PETSc.COMM_WORLD)
    is_II_lvl = PETSc.IS().createGeneral(data['II'].astype(PETSc.IntType), comm=PETSc.COMM_WORLD)

    II_lvl = data['II']
    GG_lvl = data['GG']
    ii_map = np.full(n_dofs_total, -1, dtype=np.int32)
    ii_map[II_lvl] = np.arange(len(II_lvl))
    gg_map = np.full(n_dofs_total, -1, dtype=np.int32)
    gg_map[GG_lvl] = np.arange(len(GG_lvl))

    if krylov_r0t_groups is not None:
        # Level 2: implicit R_0T = -yy·A3 (shell mat, never materialized as n×m dense)
        r0t_ctx = _R0TShellCtx(krylov_r0t_groups, ii_map, gg_map)
        R0T_sub = PETSc.Mat().createPython(size=(len(II_lvl), len(GG_lvl)), comm=PETSc.COMM_WORLD)
        R0T_sub.setPythonContext(r0t_ctx)
        R0T_sub.assemble()
    elif len(R0T_rows) > 0:
        r0t_rows = np.concatenate(R0T_rows)
        r0t_cols = np.concatenate(R0T_cols)
        r0t_vals = np.concatenate(R0T_vals)
        # 直接构造 II×GG 局部矩阵（三元组本来就是 II×GG 条目，无需全局 R_0T_mat）
        keep = (ii_map[r0t_rows] >= 0) & (gg_map[r0t_cols] >= 0)
        R0T_sub_csr = csr_matrix((r0t_vals[keep],
                                  (ii_map[r0t_rows[keep]], gg_map[r0t_cols[keep]])),
                                 shape=(len(II_lvl), len(GG_lvl)))
        R0T_sub = PETSc.Mat().createAIJ(
            size=(len(II_lvl), len(GG_lvl)),
            csr=(R0T_sub_csr.indptr.astype(PETSc.IntType),
                 R0T_sub_csr.indices.astype(PETSc.IntType),
                 R0T_sub_csr.data.astype(PETSc.ScalarType)),
            comm=PETSc.COMM_WORLD
        )
        R0T_sub.assemble()
    else:
        empty_csr = csr_matrix((0, 0))
        R0T_sub = PETSc.Mat().createAIJ(
            size=(0, 0),
            csr=(empty_csr.indptr.astype(PETSc.IntType),
                 empty_csr.indices.astype(PETSc.IntType),
                 empty_csr.data.astype(PETSc.ScalarType)),
            comm=PETSc.COMM_WORLD
        )
        R0T_sub.assemble()
    all_R_0T.append(R0T_sub)

    all_is_GG.append(is_GG_lvl)
    all_is_II.append(is_II_lvl)

    all_Interior_list.append(data['Interior'])

    # Build InvS_Blk, W, Global_Solver for this level
    # InvS_GG is always a valid PETSc Mat (even 0x0 for empty GG, matching 5level_3d.py)
    InvS_GG, _, _, _ = build_InvS_Blk_on_GG(
        S_Blk_rows_lvl, S_Blk_cols_lvl, S_Blk_vals_lvl,
        data['Subd_E'], data['Subd_F'], data['GG'])

    W_lvl, Global_Solver_lvl = build_nosas_global_operator(
        A_QG_data_lvl, D_inv_G_lvl, InvS_GG, data['GG'], n_dofs_total)

    all_W.append(W_lvl)
    all_InvS_Blk.append(InvS_GG)
    all_Global_Solver.append(Global_Solver_lvl)

    prev_Subd_E = data['Subd_E']
    prev_Subd_F = data['Subd_F']

    print(f"  Level {level_idx + 1} sum(flag) = {sum(flag_lvl)}")


# ====================== Recursive V-cycle preconditioner ======================
def nosas_preconditioner_vcycle(r, z, ksp_list, Local_Solver_list, InvS_Blk_list, W_list,
                                R_0T_list, is_GG_list, is_II_list, Interior_list,
                                level_idx, n_dofs_total):
    """V-cycle (exact 2D pattern: has_next_level, unconditional R_0T, n_dofs_total param)."""
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

    # R_0T correction (unconditional, matching 2D)
    r_II = PETSc.Vec().createWithArray(r_new.getValues(current_is_II))
    b_temp_coarse = PETSc.Vec().createMPI(b.getSize(), comm=PETSc.COMM_WORLD)
    current_R_0T.multTranspose(r_II, b_temp_coarse)
    b.axpy(1.0, b_temp_coarse)
    r_II.destroy()
    b_temp_coarse.destroy()

    coarsest_level = (level_idx == len(ksp_list) - 1)

    if coarsest_level:
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
    else:
        w_g = PETSc.Vec().createMPI(b.getSize(), comm=PETSc.COMM_WORLD)
        nosas_preconditioner_vcycle(b, w_g, ksp_list, Local_Solver_list, InvS_Blk_list, W_list,
                                     R_0T_list, is_GG_list, is_II_list, Interior_list,
                                     next_level_idx, n_dofs_total)

    U_gamma = PETSc.Vec().createMPI(n_dofs_total, comm=PETSc.COMM_WORLD)
    U_gamma.set(0.0)
    U_gamma.setValues(current_is_GG, w_g)
    U_gamma_II = current_R_0T * w_g
    U_gamma.setValues(current_is_II, U_gamma_II)
    U_gamma_II.destroy()
    U_gamma.assemble()
    w_g.destroy()

    U_I = PETSc.Vec().createMPI(n_dofs_total, comm=PETSc.COMM_WORLD)
    U_I.set(0.0)
    for i in range(len(current_Interior)):
        nodes = current_Interior[i]
        yy = current_Local_Solver[i]
        if yy is None:
            continue
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
    U_gamma.destroy()
    U_I.destroy()


# ====================== PCG solver ======================
def pcg_nosas_multilevel(A, b, max_it, tol, n_dofs_total,
                         ksp_list, Local_Solver_list, InvS_Blk_list, W_list,
                         R_0T_list, is_GG_list, is_II_list, Interior_list):
    x = b.copy()
    x.set(0.0)
    r = b.copy()
    bnrm2 = b.norm(PETSc.NormType.NORM_2)
    error = r.norm(PETSc.NormType.NORM_2) / bnrm2
    print(f' PCG residual(0) = {error}')
    alpha, beta = np.zeros(max_it), np.zeros(max_it)

    for iter_count in range(max_it):
        z = r.copy()
        nosas_preconditioner_vcycle(r, z, ksp_list, Local_Solver_list, InvS_Blk_list, W_list,
                                    R_0T_list, is_GG_list, is_II_list, Interior_list,
                                    0, n_dofs_total)

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

    if flag == 0 and iter_count > 0:
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
        print(f'PCG converged in 0 iters, residual {error:e}')
    else:
        print(f'PCG did NOT converge in {iter_count + 1} iters, residual {error:e}')

    return x, error, iter_count + 1, flag


# ====================== Set up solver ======================
if n_coarse_levels > 0:
    ksp_list = []
    for gs in all_Global_Solver:
        ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
        ksp.setOperators(gs)
        ksp.setType(PETSc.KSP.Type.PREONLY)
        ksp.getPC().setType(PETSc.PC.Type.CHOLESKY)
        ksp.setUp()
        ksp_list.append(ksp)

    Local_Solver_list = all_Local_Solver[1:]  # skip level 0
    InvS_Blk_list = all_InvS_Blk
    W_list = all_W
    R_0T_list = all_R_0T
    is_GG_list = [is_GG_level0] + all_is_GG
    is_II_list = all_is_II
    Interior_list = all_Interior_list
else:
    ksp_list = []
    Local_Solver_list = []
    InvS_Blk_list = []
    W_list = []
    R_0T_list = []
    is_GG_list = []
    is_II_list = []
    Interior_list = []

max_iter, tol = 200, 1e-6

print("\n=== PCG solve ===")
if n_coarse_levels > 0:
    x, error, iter_count, flag = pcg_nosas_multilevel(
        A=S, b=g, max_it=max_iter, tol=tol, n_dofs_total=n_dofs_total,
        ksp_list=ksp_list, Local_Solver_list=Local_Solver_list,
        InvS_Blk_list=InvS_Blk_list, W_list=W_list,
        R_0T_list=R_0T_list, is_GG_list=is_GG_list,
        is_II_list=is_II_list, Interior_list=Interior_list)
else:
    # 2-level mode: build R_0T + InvS_Blk + W on GG_0 (matching 2D pattern)
    print("\n=== 2-level mode: building solvers ===")
    is_II0 = PETSc.IS().createGeneral(II_lev0.astype(PETSc.IntType), comm=PETSc.COMM_WORLD)
    is_GG0 = is_GG_level0

    # R0T_level0 storage removed (OOM at h=1, dead data for multilevel).
    # 2-level mode needs R_0T = -A4_inv_A3. Use empty as placeholder.
    empty_csr = csr_matrix((0, 0))
    R_0T = PETSc.Mat().createAIJ(size=(0, 0),
                                 csr=(empty_csr.indptr.astype(PETSc.IntType),
                                      empty_csr.indices.astype(PETSc.IntType),
                                      empty_csr.data.astype(PETSc.ScalarType)))
    R_0T.assemble()

    InvS_Blk0, _, _, _ = build_InvS_Blk_on_GG(
        S_Blk_rows_lev0, S_Blk_cols_lev0, S_Blk_vals_lev0,
        Subd_E_lev0, Subd_F_lev0, GG_lev0)

    W0, Global_Solver0 = build_nosas_global_operator(
        A_QG_data_lev0, D_inv_G_lev0, InvS_Blk0, GG_lev0, n_dofs_total)

    ksp_global0 = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
    ksp_global0.setOperators(Global_Solver0)
    ksp_global0.setType(PETSc.KSP.Type.PREONLY)
    ksp_global0.getPC().setType(PETSc.PC.Type.CHOLESKY)
    ksp_global0.setUp()

    x, error, iter_count, flag = pcg_nosas_multilevel(
        A=S, b=g, max_it=max_iter, tol=tol, n_dofs_total=n_dofs_total,
        ksp_list=[ksp_global0],
        Local_Solver_list=[Local_Solver0],
        InvS_Blk_list=[InvS_Blk0],
        W_list=[W0],
        R_0T_list=[R_0T],
        is_GG_list=[is_GG_level0, is_GG_level0],
        is_II_list=[is_II0],
        Interior_list=[Interior_lev0])

end_time = time.time()
print(f'\nTotal time: {end_time - start_time:.4f} s')
print(f'n_dofs_total = {n_dofs_total}')
print(f'interior dofs = {n_dofs_total - len(dofs_bc)}')
print(f'sum(flag1) = {sum(flag_lev0)}')
if n_coarse_levels > 0:
    for i, flg in enumerate(all_flag[1:]):
        print(f'sum(flag{i + 2}) = {sum(flg)}')
print("=== Done ===")






