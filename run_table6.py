"""Table 6: 3D two-level NOSAS (a) + BDDC deluxe (b) on SPE10.
Based on user's proven flat 2-level code (h=2, n=102, mu=0.5 — verified on 32 cores / 128GB).
No R0T dead storage. Uses BLAS-3 A4⁻¹A3 via KSP.matSolve.

Both halves use N1=128 subdomains (the paper's choice).

Usage: python run_table6.py [h=4|2|1]
  h=4: small test (fast, local)
  h=2: quick local check
  h=1: production run (HPC), reproduces the paper table

Output: RESULT: N1  eta  N_s  iters  lambdamax  lambdamin  cond
        (NOSAS half, a), then the BDDC deluxe half (b) via bddc_deluxe.run_bddc
        printing RESULT: N1  th_f  th_e  #E  iters  lmax  lmin  cond.
"""
import gc, sys
import numpy as np

import bddc_deluxe

N1 = 128
H = 4
for arg in sys.argv[1:]:
    if arg.startswith('h='):
        H = int(arg.split('=')[1])
ETA_VALS = [0.3, 0.5, 0.7]

TEMPLATE = r'''
import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx import mesh, fem
import pymetis
from petsc4py import PETSc
import ufl
from scipy.sparse import csr_matrix
from scipy.linalg import eigh
import time, gc, sys

n_subdomains = __N1__
deg = 1
mu_level0 = __MU__
h = __H__
start_time_total = time.time()

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
rho_data = np.array(data, dtype=np.float64).reshape(nz, ny, nx)
print(f"Data shape: {rho_data.shape}, p range: {rho_data.min():.2f} ~ {rho_data.max():.2f}")
sys.stdout.flush()
del data

# ==================== 1. Create mesh ====================
print(f"\n=== Creating mesh ===")
n_cubes_x, n_cubes_y, n_cubes_z = int(60 / h), int(220 / h), int(85 / h)
print(f"h={h}, Cube grid: {n_cubes_x}*{n_cubes_y}*{n_cubes_z}")
msh = mesh.create_box(MPI.COMM_SELF, [np.array([0, 0, 0]), np.array([60, 220, 85])],
                      [n_cubes_x, n_cubes_y, n_cubes_z], mesh.CellType.tetrahedron)
print(f"Total cells: {msh.topology.index_map(3).size_global}")
sys.stdout.flush()

def get_cell_rho_values(cells):
    geom = msh.geometry
    centroids = np.mean(geom.x[msh.geometry.dofmap][cells], axis=1)
    indices_x = np.floor(centroids[:, 0] / h).astype(np.int32)
    indices_y = np.floor(centroids[:, 1] / h).astype(np.int32)
    indices_z = np.floor(centroids[:, 2] / h).astype(np.int32)
    step_x, step_y, step_z = int(60 / n_cubes_x), int(220 / n_cubes_y), int(85 / n_cubes_z)
    return rho_data[indices_z * step_z, indices_y * step_y, indices_x * step_x]

# ==================== 3. Create boundary markers ====================
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
print("Boundary markers created")

# ==================== 4. Extract boundary DOFs ====================
V = fem.functionspace(msh, ("Lagrange", deg))
dofmap = V.dofmap
n_dofs_total = dofmap.index_map.size_global
facets = np.concatenate([facet_tags.find(tag) for tag in [10, 11, 12, 13, 14, 15]])
dofs_bc = np.unique(fem.locate_dofs_topological(V, 2, facets).astype(np.int32))
print(f"Total DOFs: {n_dofs_total}, Boundary DOFs: {len(dofs_bc)}")

# ==================== 5. Mesh data ====================
msh.topology.create_connectivity(3, 0)
tetrahedra = msh.topology.connectivity(3, 0).array.reshape(-1, 4)
num_tets = len(tetrahedra)
msh.topology.create_connectivity(2, 3)
face_to_cells = msh.topology.connectivity(2, 3)
num_faces = face_to_cells.num_nodes
print(f"Tetrahedra: {num_tets}")

# ==================== 6. Assign material properties ====================
print("\n=== Assigning material properties ===")
cell_rho_values = get_cell_rho_values(np.arange(num_tets, dtype=np.int32))
print(f"Cell p range: {cell_rho_values.min():.3f}~{cell_rho_values.max():.3f}")

# ==================== 7. Domain decomposition ====================
print("\n=== Domain decomposition ===")
adjacency = [[] for _ in range(num_tets)]
for f in range(num_faces):
    cells = face_to_cells.links(f)
    if len(cells) == 2:
        c1, c2 = cells[0], cells[1]
        adjacency[c1].append(c2); adjacency[c2].append(c1)
num_cuts, membership = pymetis.part_graph(n_subdomains, adjacency=adjacency)
membership = np.array(membership, dtype=np.int32)
print(f"Domain decomposition done, cuts: {num_cuts}")
del adjacency

# ==================== 8. Extract core variables ====================
subdomain_all_dofs = [np.array([], dtype=np.int32) for _ in range(n_subdomains)]
v = np.zeros(n_dofs_total, dtype=np.int32)
v2 = np.zeros(n_dofs_total, dtype=np.int32)
for sub_id in range(n_subdomains):
    sub_tri_ids = np.where(membership == sub_id)[0]
    sub_dofs_set = set()
    for cell_idx in sub_tri_ids:
        sub_dofs_set.update(dofmap.cell_dofs(cell_idx))
    sub_dofs = np.array(list(sub_dofs_set), dtype=np.int32)
    subdomain_all_dofs[sub_id] = sub_dofs
    v[sub_dofs] += 1
    v2[sub_dofs] += (sub_id + 1)

v3 = v2.copy()
GG = np.setdiff1d(np.where(v >= 2)[0], dofs_bc)
Subd_F_all = np.setdiff1d(np.where(v == 2)[0], dofs_bc)
II = np.setdiff1d(np.where(v == 1)[0], dofs_bc)
print(f"GG: {len(GG)}, Subd_V: {len(Subd_F_all)}, Int: {len(II)}")

# ==================== 9. Extract Subd_E ====================
v2[Subd_F_all] = 0; v2[dofs_bc] = 0; v2[II] = 0
Subd_E_candidates = []
for sub_id in range(n_subdomains):
    sub_dofs = subdomain_all_dofs[sub_id]
    non_zero_mask = v2[sub_dofs] != 0
    if not non_zero_mask.any(): continue
    edge_dofs = sub_dofs[non_zero_mask]
    edge_values = v2[edge_dofs]
    for val in np.unique(edge_values):
        if val == 0: continue
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

# ==================== 10. Extract Subd_F ====================
Subd_E_all = np.array([d for edge in Subd_E for d in edge], dtype=np.int32)
v3[Subd_E_all] = 0; v3[dofs_bc] = 0; v3[II] = 0
Subd_F = []
for sub_id in range(n_subdomains):
    sub_dofs = subdomain_all_dofs[sub_id]
    non_zero_mask = v3[sub_dofs] != 0
    if not non_zero_mask.any(): continue
    face_dofs = sub_dofs[non_zero_mask]
    face_values = v3[face_dofs]
    for val in np.unique(face_values):
        if val == 0: continue
        face = face_dofs[face_values == val]
        if len(face) > 0:
            Subd_F.append(face)
            v3[face] = 0

# ==================== 11. Construct TS_Blk ====================
print("\n=== Constructing TS_Blk ===")
all_rows, all_cols = [], []
for edge in Subd_E:
    n_e = len(edge)
    all_rows.extend(np.repeat(edge, n_e))
    all_cols.extend(np.tile(edge, n_e))
for face in Subd_F:
    n_f = len(face)
    all_rows.extend(np.repeat(face, n_f))
    all_cols.extend(np.tile(face, n_f))
all_rows = np.array(all_rows, dtype=np.int32)
all_cols = np.array(all_cols, dtype=np.int32)
all_vals = np.ones(len(all_rows), dtype=PETSc.ScalarType)
TS_Blk_csr = csr_matrix((all_vals, (all_rows, all_cols)), shape=(n_dofs_total, n_dofs_total))
TS_Blk = PETSc.Mat().createAIJ(size=(n_dofs_total, n_dofs_total),
                               csr=(TS_Blk_csr.indptr.astype(PETSc.IntType),
                                    TS_Blk_csr.indices.astype(PETSc.IntType),
                                    TS_Blk_csr.data.astype(PETSc.ScalarType)))
print(f"TS_Blk Shape: {TS_Blk.getSize()}, Non-zeros: {TS_Blk.getInfo()['nz_used']}")
del TS_Blk_csr, all_rows, all_cols, all_vals
gc.collect()

# ==================== 12. Subdomain Gamma/Interior ====================
print("\n=== Subdomain computation (3D with coefficient) ===")
Gamma, Interior = [None] * n_subdomains, [None] * n_subdomains
for sub_id in range(n_subdomains):
    sub_dofs = subdomain_all_dofs[sub_id]
    Interior[sub_id] = np.setdiff1d(np.setdiff1d(sub_dofs, dofs_bc), GG)
    Gamma[sub_id] = np.intersect1d(sub_dofs, GG)

f = PETSc.ScalarType(1.0)

S_rows_list, S_cols_list, S_vals_list = [], [], []
S_Blk_rows_list, S_Blk_cols_list, S_Blk_vals_list = [], [], []
g_rows_list, g_vals_list = [], []
A_QG_data, InvD_G = [], []

def petsc_to_scipy_dense(petsc_mat):
    I, J, V = petsc_mat.getValuesCSR()
    return csr_matrix((V, J, I), shape=petsc_mat.getSize()).toarray()

# ==================== 13. Subdomain computation loop ====================
print("=== Subdomain computation ===")
for sub_id in range(n_subdomains):
    sub_tet_ids = np.where(membership == sub_id)[0]
    m_gamma, n_int = Gamma[sub_id], Interior[sub_id]
    len_m, len_n = len(m_gamma), len(n_int)
    print(f"Subdomain {sub_id}: {len(sub_tet_ids)} tets, Gamma:{len_m}, Interior:{len_n}")
    sys.stdout.flush()

    submesh, entity_map, vertex_map, geometry_map = mesh.create_submesh(msh, msh.topology.dim, sub_tet_ids)
    V_local = fem.functionspace(submesh, ("Lagrange", deg))
    dofmap_sub = V_local.dofmap
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
    m_local = sorted_idx[np.searchsorted(sorted_global, m_gamma)]
    n_local = sorted_idx[np.searchsorted(sorted_global, n_int)]
    u = ufl.TrialFunction(V_local); v = ufl.TestFunction(V_local)
    DG0 = fem.functionspace(submesh, ("DG", 0))
    rho_func = fem.Function(DG0); rho_func.x.array[:] = sub_rho_values
    a_local = fem.form(ufl.inner(rho_func * ufl.grad(u), ufl.grad(v)) * ufl.dx)
    L_local = fem.form(ufl.inner(f, v) * ufl.dx)
    A_local_sp = fem.assemble_matrix(a_local)
    b_local = fem.assemble_vector(L_local)

    A_local = PETSc.Mat().createAIJ(size=(A_local_sp.index_map(0).size_local, A_local_sp.index_map(1).size_local),
                                    csr=(A_local_sp.indptr.astype(PETSc.IntType),
                                         A_local_sp.indices.astype(PETSc.IntType),
                                         A_local_sp.data.astype(PETSc.ScalarType)))
    del A_local_sp

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

    # A4_inv_A3 via KSP.matSolve (BLAS-3)
    row_idx = np.arange(len_n, dtype=PETSc.IntType)
    col_idx = np.arange(len_m, dtype=PETSc.IntType)
    A3_dense = PETSc.Mat().createDense([len_n, len_m], comm=PETSc.COMM_WORLD)
    A3_dense.setUp()
    A3_dense_np = np.zeros((len_n, len_m), dtype=PETSc.ScalarType)
    A3.getValues(row_idx, col_idx, A3_dense_np)
    A3_dense.setValues(row_idx, col_idx, A3_dense_np)
    A3_dense.assemble()
    X_dense = PETSc.Mat().createDense([len_n, len_m], comm=PETSc.COMM_WORLD)
    X_dense.setUp()
    ksp_A4.matSolve(A3_dense, X_dense)
    all_solutions = np.zeros((len_n, len_m), dtype=PETSc.ScalarType)
    X_dense.getValues(row_idx, col_idx, all_solutions)
    A3_dense.destroy(); X_dense.destroy()
    rows_tmp = np.repeat(np.arange(len_n), len_m)
    cols_tmp = np.tile(np.arange(len_m), len_n)
    vals_flat = all_solutions.ravel()
    A4_inv_A3_sp = csr_matrix((vals_flat, (rows_tmp, cols_tmp)), shape=(len_n, len_m))
    del rows_tmp, cols_tmp, vals_flat
    A4_inv_A3 = PETSc.Mat().createAIJ(size=(len_n, len_m),
                                      csr=(A4_inv_A3_sp.indptr.astype(PETSc.IntType),
                                           A4_inv_A3_sp.indices.astype(PETSc.IntType),
                                           A4_inv_A3_sp.data.astype(PETSc.ScalarType)))
    del A4_inv_A3_sp

    # Compute g = b_m - A2 * A4_inv_A3 * b_n
    b_local_array = b_local.array.copy()
    del b_local
    b_local_vec = PETSc.Vec().createWithArray(b_local_array)
    b_m_values = b_local_vec.getValues(is_m_local)
    b_n_values = b_local_vec.getValues(is_n_local)
    del b_local_array
    b_n_vec = PETSc.Vec().createWithArray(b_n_values.copy())
    x_n_vec = b_n_vec.duplicate()
    ksp_A4.solve(b_n_vec, x_n_vec)
    A2_x_n = PETSc.Vec().createMPI(len_m, comm=PETSc.COMM_WORLD)
    A2.mult(x_n_vec, A2_x_n)
    g_local = b_m_values - A2_x_n.getArray()
    g_vals_list.append(g_local.copy())
    g_rows_list.append(m_gamma)
    del b_local_vec, b_m_values, b_n_values, b_n_vec, x_n_vec, A2_x_n

    # Schur complement
    SS = A2.matMult(A4_inv_A3); SS.assemble()
    Schur = A1.copy(); Schur.axpy(-1.0, SS); Schur.assemble()
    A4_inv_A3.destroy(); SS.destroy()

    # Eigen decomposition
    is_m = PETSc.IS().createGeneral(m_gamma.astype(PETSc.IntType), comm=PETSc.COMM_WORLD)
    Schur_np = petsc_to_scipy_dense(Schur)
    Schur_np = (Schur_np + Schur_np.T) / 2

    # Save full Schur for S_GG
    S_rows_list.append(np.repeat(m_gamma, len_m))
    S_cols_list.append(np.tile(m_gamma, len_m))
    S_vals_list.append(Schur_np.ravel().copy())

    A1_np = petsc_to_scipy_dense(A1)
    TS_Blk_mm = TS_Blk.createSubMatrix(is_m, is_m); TS_Blk_mm.assemble()
    TS_Blk_mm_np = petsc_to_scipy_dense(TS_Blk_mm)
    A1_masked = np.multiply(Schur_np, TS_Blk_mm_np)
    del TS_Blk_mm_np
    D, Q = eigh(Schur_np, A1_masked)
    del Schur_np

    A1_csr = csr_matrix(A1_masked)
    s_indptr, s_indices, s_vals = A1_csr.indptr, A1_csr.indices, A1_csr.data
    s_row_indices_local = np.repeat(np.arange(len(np.diff(s_indptr))), np.diff(s_indptr))
    S_Blk_rows_list.append(m_gamma[s_row_indices_local])
    S_Blk_cols_list.append(m_gamma[s_indices])
    S_Blk_vals_list.append(s_vals.copy())
    del A1_csr, s_indptr, s_indices, s_vals, s_row_indices_local

    eigens_idx = np.where(D < mu_level0)[0]
    Q = Q[:, eigens_idx]; D = 1 - D[eigens_idx]
    if len(eigens_idx) > 0:
        A1Q = A1_masked @ Q
        A_QG_data.append({'m': m_gamma.copy(), 'A1Q': A1Q.copy(), 'n_eigen': len(eigens_idx)})
        InvD_G.append(1.0 / D.copy())
        del A1Q
    del A1_np, A1_masked, Q, D, eigens_idx

    is_m_local.destroy(); is_n_local.destroy(); is_m.destroy()
    A_local.destroy(); A1.destroy(); A2.destroy(); A3.destroy(); A4.destroy()
    ksp_A4.destroy()

    gc.collect()

# ==================== 14. Assemble S_GG and g_gamma ====================
print("\n=== Assembling S_GG ==="); sys.stdout.flush()
GG.sort()
is_GG = PETSc.IS().createGeneral(GG.astype(PETSc.IntType), comm=PETSc.COMM_WORLD)

S_rows = np.concatenate(S_rows_list)
S_cols = np.concatenate(S_cols_list)
S_vals = np.concatenate(S_vals_list)
del S_rows_list, S_cols_list, S_vals_list
S_rows_gg = np.searchsorted(GG, S_rows)
S_cols_gg = np.searchsorted(GG, S_cols)
del S_rows, S_cols
S_GG_csr = csr_matrix((S_vals, (S_rows_gg, S_cols_gg)), shape=(len(GG), len(GG)))
del S_rows_gg, S_cols_gg, S_vals
S_GG = PETSc.Mat().createAIJ(size=(len(GG), len(GG)),
                             csr=(S_GG_csr.indptr.astype(PETSc.IntType),
                                  S_GG_csr.indices.astype(PETSc.IntType),
                                  S_GG_csr.data.astype(PETSc.ScalarType)))
del S_GG_csr
gc.collect()

print("=== Assembling g_gamma ==="); sys.stdout.flush()
g_rows = np.concatenate(g_rows_list)
g_vals = np.concatenate(g_vals_list)
del g_rows_list, g_vals_list
g_full = PETSc.Vec().createSeq(n_dofs_total)
g_full.setValues(g_rows.astype(PETSc.IntType), g_vals.astype(PETSc.ScalarType),
                 addv=PETSc.InsertMode.ADD_VALUES); g_full.assemble()
del g_rows, g_vals
g_gamma = PETSc.Vec().createWithArray(g_full.getValues(is_GG))
del g_full
gc.collect()

# ==================== 15. Assemble S_Blk ====================
print("=== Assembling S_Blk ===")
S_Blk_rows, S_Blk_cols, S_Blk_vals = map(np.concatenate, [S_Blk_rows_list, S_Blk_cols_list, S_Blk_vals_list])
del S_Blk_rows_list, S_Blk_cols_list, S_Blk_vals_list
S_Blk_csr = csr_matrix((S_Blk_vals, (S_Blk_rows, S_Blk_cols)), shape=(n_dofs_total, n_dofs_total))
del S_Blk_rows, S_Blk_cols, S_Blk_vals
S_Blk_petsc = PETSc.Mat().createAIJ(size=(n_dofs_total, n_dofs_total),
                                    csr=(S_Blk_csr.indptr.astype(PETSc.IntType),
                                         S_Blk_csr.indices.astype(PETSc.IntType),
                                         S_Blk_csr.data.astype(PETSc.ScalarType)))
del S_Blk_csr
gc.collect()

# ==================== 16. Assemble A_QG ====================
print("=== Assembling A_QG ===")
n_eigen_total = sum(data['n_eigen'] for data in A_QG_data)
all_A_QG_rows, all_A_QG_cols, all_A_QG_vals = [], [], []
col_offset = 0
for data in A_QG_data:
    m_indices, A1Q, n_eigen = data['m'], data['A1Q'], data['n_eigen']
    all_A_QG_rows.append(np.repeat(m_indices, n_eigen))
    all_A_QG_cols.append(np.tile(np.arange(n_eigen) + col_offset, len(m_indices)))
    all_A_QG_vals.append(A1Q.ravel())
    col_offset += n_eigen
all_A_QG_rows, all_A_QG_cols, all_A_QG_vals = map(np.concatenate, [all_A_QG_rows, all_A_QG_cols, all_A_QG_vals])
A_QG_csr = csr_matrix((all_A_QG_vals, (all_A_QG_rows, all_A_QG_cols)), shape=(n_dofs_total, n_eigen_total))
del all_A_QG_rows, all_A_QG_cols, all_A_QG_vals
A_QG = PETSc.Mat().createAIJ(size=(n_dofs_total, n_eigen_total),
                             csr=(A_QG_csr.indptr.astype(PETSc.IntType),
                                  A_QG_csr.indices.astype(PETSc.IntType),
                                  A_QG_csr.data.astype(PETSc.ScalarType)))
del A_QG_csr

# ==================== 17. Construct InvS_Blk ====================
print("\n=== Constructing InvS_Blk ===")
rows, cols, vals = [], [], []
for face in Subd_F:
    face = face.astype(PETSc.IntType)
    is_face = PETSc.IS().createGeneral(face, comm=PETSc.COMM_WORLD)
    S_sub = S_Blk_petsc.createSubMatrix(is_face, is_face)
    inv = np.linalg.inv(petsc_to_scipy_dense(S_sub))
    n_f = len(face)
    rows.extend(np.repeat(face, n_f))
    cols.extend(np.tile(face, n_f))
    vals.extend(inv.ravel())
    del inv; del S_sub
for edge in Subd_E:
    edge = edge.astype(PETSc.IntType)
    is_edge = PETSc.IS().createGeneral(edge, comm=PETSc.COMM_WORLD)
    S_sub = S_Blk_petsc.createSubMatrix(is_edge, is_edge)
    inv = np.linalg.inv(petsc_to_scipy_dense(S_sub))
    n_e = len(edge)
    rows.extend(np.repeat(edge, n_e))
    cols.extend(np.tile(edge, n_e))
    vals.extend(inv.ravel())
    del inv; del S_sub
rows, cols, vals = map(np.array, [rows, cols, vals])
InvS_Blk = csr_matrix((vals.astype(PETSc.ScalarType), (rows, cols)), shape=(n_dofs_total, n_dofs_total))
InvS_Blk = PETSc.Mat().createAIJ(size=(n_dofs_total, n_dofs_total),
                                 csr=(InvS_Blk.indptr.astype(PETSc.IntType),
                                      InvS_Blk.indices.astype(PETSc.IntType),
                                      InvS_Blk.data.astype(PETSc.ScalarType)))
del rows, cols, vals
gc.collect()

# ==================== 18. Coarse space & Global Solver ====================
print("=== Constructing global solver ===")
InvS_Blk = InvS_Blk.createSubMatrix(is_GG, is_GG); InvS_Blk.assemble()

indices_arr = np.arange(len(InvD_combined := np.concatenate(InvD_G)), dtype=np.int32)
InvD_G_mat = csr_matrix((InvD_combined, (indices_arr, indices_arr)), shape=(len(InvD_combined), len(InvD_combined)))
InvD_G_mat = PETSc.Mat().createAIJ(size=InvD_G_mat.shape,
                                   csr=(InvD_G_mat.indptr.astype(PETSc.IntType),
                                        InvD_G_mat.indices.astype(PETSc.IntType),
                                        InvD_G_mat.data.astype(PETSc.ScalarType)))
del InvD_G, indices_arr, InvD_combined
gc.collect()

A_QG_petsc = A_QG.createSubMatrix(is_GG, None); A_QG_petsc.assemble()
W = InvS_Blk * A_QG_petsc
Global_Solver = InvD_G_mat - A_QG_petsc.transposeMatMult(W)

# ==================== 19. PCG ====================
def NOSAS_preconditioner(r, z, InvS_Blk, W, ksp_global):
    InvS_Blk.mult(r, z)
    _, W_n = W.getSize()
    WTr = PETSc.Vec().createWithArray(np.zeros(W_n))
    W.multTranspose(r, WTr)
    solution = WTr.copy()
    ksp_global.solve(WTr, solution)
    temp = z.copy(); temp.set(0.0)
    W.mult(solution, temp)
    z.axpy(1.0, temp)

def pcg_NoSAS(A, b, max_it, tol, InvS_Blk, W, ksp_global):
    x = b.copy(); x.set(0.0)
    r = b.copy()
    bnrm2 = b.norm(PETSc.NormType.NORM_2)
    error = r.norm(PETSc.NormType.NORM_2) / bnrm2
    print(f' PCG residual(0) = {error:e}')
    alpha, beta = np.zeros(max_it), np.zeros(max_it)
    for iter_count in range(max_it):
        z = r.copy()
        NOSAS_preconditioner(r, z, InvS_Blk, W, ksp_global)
        rho = r.dot(z)
        if iter_count > 0:
            beta[iter_count] = rho / rho_1
            p.scale(beta[iter_count]); p.axpy(1.0, z)
        else:
            p = z.copy()
        q = p.copy()
        A.mult(p, q)
        alpha[iter_count] = rho / p.dot(q)
        x.axpy(alpha[iter_count], p)
        r.axpy(-alpha[iter_count], q)
        error = r.norm(PETSc.NormType.NORM_2) / bnrm2
        print(f' PCG residual({iter_count + 1}) = {error:e}')
        if error <= tol: break
        rho_1 = rho
    flag = 0 if error <= tol else 1
    if error <= tol and iter_count > 0:
        d = np.zeros(iter_count + 1); s = np.zeros(iter_count) if iter_count > 0 else np.zeros(0)
        d[0] = 1.0 / alpha[0]
        for i in range(iter_count):
            d[i + 1] = beta[i + 1] / alpha[i] + 1.0 / alpha[i + 1]
            s[i] = -np.sqrt(beta[i + 1]) / alpha[i]
        T = np.zeros((iter_count + 1, iter_count + 1))
        for i in range(iter_count):
            T[i, i] = d[i]; T[i, i + 1] = s[i]; T[i + 1, i] = s[i]
        T[iter_count, iter_count] = d[iter_count]
        lambda_vals = np.linalg.eigvals(T)
        condnumber = np.max(lambda_vals) / np.min(lambda_vals)
    if flag == 1:
        print(f'PCG did NOT converge in {iter_count + 1} iters, residual {error:e}')
    else:
        print(f'PCG converged in {iter_count + 1} iters, residual {error:e}, '
              f'lambda_max={np.max(lambda_vals):g}, lambda_min={np.min(lambda_vals):g}, cond={condnumber:g}')
    # Capture via globals
    global __pcg_iters, __pcg_lmax, __pcg_lmin, __pcg_cond, __pcg_converged
    if flag == 0 and iter_count > 0:
        __pcg_iters = iter_count + 1
        __pcg_lmax = np.max(lambda_vals)
        __pcg_lmin = np.min(lambda_vals)
        __pcg_cond = condnumber
        __pcg_converged = True
    elif flag == 0:
        __pcg_iters = 0
        __pcg_lmax = 1.0; __pcg_lmin = 1.0; __pcg_cond = 1.0
        __pcg_converged = True
    else:
        __pcg_iters = iter_count + 1
        __pcg_lmax = 0.0; __pcg_lmin = 0.0; __pcg_cond = 0.0
        __pcg_converged = False
    return x, error, iter_count + 1, flag

# ==================== 20. Solve ====================
print("=== PCG on Schur system ===")
ksp_global = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
ksp_global.setOperators(Global_Solver)
ksp_global.setType(PETSc.KSP.Type.PREONLY)
ksp_global.getPC().setType(PETSc.PC.Type.CHOLESKY)
ksp_global.setUp()

x, error, iter_count, flag = pcg_NoSAS(
    A=S_GG, b=g_gamma, max_it=200, tol=1e-6,
    InvS_Blk=InvS_Blk, W=W, ksp_global=ksp_global)

ksp_global.destroy()

end_time = time.time()
print(f"Total time: {end_time - start_time_total:.4f} s")

# N_s = total eigenvectors
N_s = sum(data['n_eigen'] for data in A_QG_data)
print(f"RESULT: __N1_OUT__  __MU_OUT__  {N_s:5d}  {__pcg_iters:3d}  {__pcg_lmax:.4f}  {__pcg_lmin:.4f}  {__pcg_cond:.4f}")
'''

print(f"Table 5a: 3D two-level NOSAS on SPE10 (N1={N1}, h={H})")
for eta in ETA_VALS:
    code = TEMPLATE
    code = code.replace("__N1__", str(N1))
    code = code.replace("__H__", str(H))
    code = code.replace("__MU__", str(eta))
    code = code.replace("__N1_OUT__", str(N1))
    code = code.replace("__MU_OUT__", f"{eta:4.1f}")
    
    print(f"\n===== eta={eta} =====")
    ns = {'np': np, '__builtins__': __builtins__}
    # Initialize globals
    ns['__pcg_iters'] = -1
    ns['__pcg_lmax'] = 0.0
    ns['__pcg_lmin'] = 0.0
    ns['__pcg_cond'] = 0.0
    ns['__pcg_converged'] = False
    
    exec(code, ns)
    del ns
    gc.collect()

print("\n=== Done (NOSAS part) ===")

# ===== Table 6(b): BDDC deluxe comparison (paper Table 6, right half) =====
# Three paper thresholds eta -> threshold = 1/eta, N1=128 subdomains.
print("\n===== Table 6(b): BDDC deluxe =====")
bddc_deluxe.run_bddc(h=H, n_subdomains=N1)
print("\n=== Entire script done ===")
