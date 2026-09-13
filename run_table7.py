import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import pymetis
from petsc4py import PETSc
import ufl
from scipy.sparse import csr_matrix
from scipy.linalg import eigh, cholesky, solve_triangular
import time, sys, gc

# ==================== Parameters ====================
H = 1
MU_VALS = [0.3, 0.5, 0.7]   # code mu = paper eta_0 (eigenvalue threshold)
ETA_VALS = [0.3, 0.5, 0.7]  # code eta = paper mu (filter threshold)
N1, N2 = 128, 1

TEMPLATE = r"""
start_time = time.time()

# ====================== Parameters ======================
n_subdomains_list = [128,1]
mu_list = [0.5]
eta_list = [0.5]
deg = 1
h = 1

n_subdomains = n_subdomains_list[0]
n_levels = len(n_subdomains_list)
n_coarse_levels = n_levels - 1

# ==================== 0. 读取渗透率数据 ====================
print("=== 读取渗透率数据 ===")
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
print(f"数据形状: {rho_data.shape}, p范围: {rho_data.min():.2f} ~ {rho_data.max():.2f}")

# ==================== 1. 创建矩形体网格 ====================
print("\n=== 创建网格 ===")
n_cubes_x, n_cubes_y, n_cubes_z = int(60 / h), int(220 / h), int(85 / h)
print(f"h={h}, 立方体网格: {n_cubes_x}*{n_cubes_y}*{n_cubes_z}")

msh = mesh.create_box(MPI.COMM_SELF, [np.array([0, 0, 0]), np.array([60, 220, 85])],
                      [n_cubes_x, n_cubes_y, n_cubes_z], mesh.CellType.tetrahedron)
print(f"总单元数: {msh.topology.index_map(3).size_global}")

# ==================== 2. 获取单元p值的函数 ====================
def get_cell_rho_values(cells):
    geom = msh.geometry
    centroids = np.mean(geom.x[msh.geometry.dofmap][cells], axis=1)
    indices_x = np.floor(centroids[:, 0] / h).astype(int)
    indices_y = np.floor(centroids[:, 1] / h).astype(int)
    indices_z = np.floor(centroids[:, 2] / h).astype(int)
    step_x, step_y, step_z = int(60 / n_cubes_x), int(220 / n_cubes_y), int(85 / n_cubes_z)
    return rho_data[indices_z * step_z, indices_y * step_y, indices_x * step_x]

# ==================== 3. 创建边界标记 ====================
print("\n=== 创建边界标记 ===")
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
print("边界标记创建完成")

# ==================== 4. 提取边界自由度 ====================
V = fem.functionspace(msh, ("Lagrange", deg))
dofmap = V.dofmap
n_dofs_total = dofmap.index_map.size_global
facets = np.concatenate([facet_tags.find(tag) for tag in [10, 11, 12, 13, 14, 15]])
dofs_bc = np.unique(fem.locate_dofs_topological(V, 2, facets).astype(np.int32))
print(f"总自由度: {n_dofs_total}, 边界自由度: {len(dofs_bc)}")

# ==================== 5. 网格数据 ====================
msh.topology.create_connectivity(3, 0)
tetrahedra = msh.topology.connectivity(3, 0).array.reshape(-1, 4)
num_tets = len(tetrahedra)
msh.topology.create_connectivity(2, 3)
face_to_cells = msh.topology.connectivity(2, 3)
num_faces = face_to_cells.num_nodes
print(f"四面体数量: {num_tets}")

# ==================== 6. 分配材料属性 ====================
print("\n=== 分配材料属性 ===")
cell_rho_values = get_cell_rho_values(np.arange(num_tets, dtype=np.int32))
print(f"单元p范围: {cell_rho_values.min():.3f}~{cell_rho_values.max():.3f}")

# ==================== 7. 区域分解 ====================
print("\n=== 区域分解 ===")
adjacency = [[] for _ in range(num_tets)]
for f in range(num_faces):
    cells = face_to_cells.links(f)
    if len(cells) == 2:
        c1, c2 = cells[0], cells[1]
        adjacency[c1].append(c2)
        adjacency[c2].append(c1)
num_cuts, membership = pymetis.part_graph(n_subdomains, adjacency=adjacency)
membership = np.array(membership)
print(f"区域分解完成，分割数: {num_cuts}")

# ==================== 8. 提取Level 0核心变量 ====================
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
print(f"GG: {len(GG)}, Subd_F_all: {len(Subd_F_all)}, Int: {len(II)}")

# ==================== Helper functions ====================
def petsc_to_scipy_dense(petsc_mat):
    I, J, V = petsc_mat.getValuesCSR()
    return csr_matrix((V, J, I), shape=petsc_mat.getSize()).toarray()


def build_TS_Blk(Subd_E, Subd_F, name):
    e_rows = [np.repeat(entity, len(entity)) for entity in Subd_E]
    f_rows = [np.repeat(entity, len(entity)) for entity in Subd_F]
    e_cols = [np.tile(entity, len(entity)) for entity in Subd_E]
    f_cols = [np.tile(entity, len(entity)) for entity in Subd_F]
    all_rows_list = e_rows + f_rows
    all_cols_list = e_cols + f_cols
    if len(all_rows_list) == 0:
        empty_csr = csr_matrix((n_dofs_total, n_dofs_total))
        ts = PETSc.Mat().createAIJ(size=(n_dofs_total, n_dofs_total),
                                    csr=(empty_csr.indptr.astype(PETSc.IntType),
                                         empty_csr.indices.astype(PETSc.IntType),
                                         empty_csr.data.astype(PETSc.ScalarType)))
        print(f"{name} Shape: ({ts.getSize()[0]}, {ts.getSize()[1]}), Non-zeros: 0 (empty)")
        return ts
    all_rows = np.concatenate(all_rows_list)
    all_cols = np.concatenate(all_cols_list)
    all_vals = np.ones(len(all_rows), dtype=PETSc.ScalarType)
    ts_csr = csr_matrix((all_vals, (all_rows, all_cols)), shape=(n_dofs_total, n_dofs_total))
    ts = PETSc.Mat().createAIJ(size=(n_dofs_total, n_dofs_total),
                                csr=(ts_csr.indptr.astype(PETSc.IntType),
                                     ts_csr.indices.astype(PETSc.IntType),
                                     ts_csr.data.astype(PETSc.ScalarType)))
    nz = ts.getInfo()['nz_used']
    print(f"{name} Shape: ({ts.getSize()[0]}, {ts.getSize()[1]}), Non-zeros: {nz}")
    return ts


def extract_edges_from_v2(subdomain_all_dofs, v2_arr, zero_mask, dofs_bc, II, n_sub):
    v2_arr[zero_mask] = 0
    v2_arr[dofs_bc] = 0
    v2_arr[II] = 0
    Subd_E_candidates = []
    for cid in range(n_sub):
        grp_dofs = subdomain_all_dofs[cid]
        nz_mask = v2_arr[grp_dofs] != 0
        if not nz_mask.any():
            continue
        edofs = grp_dofs[nz_mask]
        evals = v2_arr[edofs]
        for val in np.unique(evals):
            if val == 0:
                continue
            entity = edofs[evals == val]
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
    return Subd_E


# ==================== Edge/face extraction + TS_Blk (Level 0) ====================
Subd_E = extract_edges_from_v2(subdomain_all_dofs, v2, Subd_F_all, dofs_bc, II, n_subdomains)
print(f"Subd_E: {len(Subd_E)} edges")

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
print(f"Subd_F: {len(Subd_F)} faces")

print("\n=== 构造TS_Blk稀疏矩阵 ===")
TS_Blk = build_TS_Blk(Subd_E, Subd_F, "TS_Blk_L0")

# ==================== 12. Level 0 Gamma/Interior ====================
print("\n=== Level 0 子区域计算 ===")
Gamma, Interior = [None] * n_subdomains, [None] * n_subdomains
for sub_id in range(n_subdomains):
    sub_dofs = subdomain_all_dofs[sub_id]
    Interior[sub_id] = np.setdiff1d(np.setdiff1d(sub_dofs, dofs_bc), GG)
    Gamma[sub_id] = np.intersect1d(sub_dofs, GG)

f = PETSc.ScalarType(1.0)

# ==================== Multilevel helper functions ====================
def extract_coarse_variables(gamma_list, n_coarse_subdomains, group_assignment_coarse, dofs_bc):
    subdomain_all_dofs = [None] * n_coarse_subdomains
    coarse_v = np.zeros(n_dofs_total, dtype=np.int32)
    coarse_v2 = np.zeros(n_dofs_total, dtype=np.int32)
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

def extract_coarse_edges_faces(subdomain_all_dofs, coarse_Subd_E_from_v3, coarse_II, dofs_bc, coarse_v2, coarse_v, n_coarse_subdomains):
    coarse_v2[coarse_Subd_E_from_v3] = 0
    coarse_v2[dofs_bc] = 0
    coarse_v2[coarse_II] = 0
    coarse_Subd_E_candidates = []
    for coarse_id in range(n_coarse_subdomains):
        group_dofs = subdomain_all_dofs[coarse_id]
        non_zero_mask = coarse_v2[group_dofs] != 0
        if not non_zero_mask.any(): continue
        edge_dofs = group_dofs[non_zero_mask]
        edge_values = coarse_v2[edge_dofs]
        unique_values, inverse_indices = np.unique(edge_values, return_inverse=True)
        for i, val in enumerate(unique_values):
            if val == 0: continue
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
    # Extract Subd_F from remaining v==2 DOFs (entities not in Subd_E)
    coarse_v3 = coarse_v.copy()
    Subd_E_all = np.array([d for e in coarse_Subd_E for d in e], dtype=np.int32) if coarse_Subd_E else np.array([], dtype=np.int32)
    coarse_v3[Subd_E_all] = 0
    coarse_v3[dofs_bc] = 0
    coarse_v3[coarse_II] = 0
    coarse_v3[coarse_Subd_E_from_v3] = 0
    remaining_mask = (coarse_v3 >= 2) & ~np.isin(np.arange(n_dofs_total, dtype=np.int32), dofs_bc)
    coarse_doppel = np.where(remaining_mask)[0]
    coarse_Subd_F = []
    for coarse_id in range(n_coarse_subdomains):
        group_dofs = subdomain_all_dofs[coarse_id]
        non_zero_mask = coarse_v3[group_dofs] != 0
        if not non_zero_mask.any(): continue
        face_dofs = group_dofs[non_zero_mask]
        face_values = coarse_v3[face_dofs]
        unique_values, inverse_indices = np.unique(face_values, return_inverse=True)
        for i, val in enumerate(unique_values):
            if val == 0: continue
            entity = face_dofs[inverse_indices == i]
            if len(entity) > 0:
                coarse_Subd_F.append(entity)
                coarse_v3[entity] = 0
    return coarse_Subd_E, coarse_Subd_F

def extract_gamma_interior(subdomain_all_dofs, coarse_GG, dofs_bc, n_coarse):
    coarse_Gamma = [None] * n_coarse
    coarse_Interior = [None] * n_coarse
    for cid in range(n_coarse):
        group_dofs = subdomain_all_dofs[cid]
        coarse_Interior[cid] = np.setdiff1d(np.setdiff1d(group_dofs, dofs_bc), coarse_GG)
        coarse_Gamma[cid] = np.intersect1d(group_dofs, coarse_GG)
    return coarse_Gamma, coarse_Interior

def perform_coarsening(gamma_list, n_coarse_subdomains, dofs_bc, level_name):
    n_groups = len(gamma_list)
    target = min(n_coarse_subdomains, n_groups)
    if target < 2:
        print(f"{level_name}: only {n_groups} group(s), using 1 coarse group")
        subdomain_all_dofs = [np.unique(np.concatenate(gamma_list))]
        # Compute coarse GG: DOFs shared between >= 2 fine subdomains
        v_coarse = np.zeros(n_dofs_total, dtype=np.int32)
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
    t0_adj = time.time()
    adj_sub = [[] for _ in range(n_groups)]
    gamma_sets = [set(g) for g in gamma_list]
    for i in range(n_groups):
        si = gamma_sets[i]
        for j in range(i + 1, n_groups):
            if not si.isdisjoint(gamma_sets[j]):
                adj_sub[i].append(j)
                adj_sub[j].append(i)
    t_adj = time.time() - t0_adj
    print(f"  {level_name} adjacency built in {t_adj:.3f}s ({n_groups} groups)")
    _, group_assignment_coarse = pymetis.part_graph(target, adjacency=adj_sub)
    group_assignment_coarse = np.array(group_assignment_coarse)
    unique_coarse_ids = np.unique(group_assignment_coarse)
    valid_coarse_ids = [cid for cid in range(n_coarse_subdomains) if cid in unique_coarse_ids]
    actual_n = len(valid_coarse_ids)
    print(f"{level_name} actual coarse subdomains: {actual_n} (target: {n_coarse_subdomains})")
    id_map = {old_id: new_id for new_id, old_id in enumerate(valid_coarse_ids)}
    group_assignment_coarse_new = np.array([id_map[old_id] for old_id in group_assignment_coarse])
    coarse_GG, coarse_Subd_E_from_v3, coarse_II, subdomain_all_dofs, coarse_v, coarse_v2 = \
        extract_coarse_variables(gamma_list, actual_n, group_assignment_coarse_new, dofs_bc)
    coarse_Subd_E, coarse_Subd_F = extract_coarse_edges_faces(
        subdomain_all_dofs, coarse_Subd_E_from_v3, coarse_II, dofs_bc, coarse_v2, coarse_v, actual_n)
    coarse_Gamma, coarse_Interior = extract_gamma_interior(subdomain_all_dofs, coarse_GG, dofs_bc, actual_n)
    return (coarse_GG, coarse_Subd_E, coarse_Subd_F, coarse_Gamma, coarse_Interior,
            subdomain_all_dofs, group_assignment_coarse_new, actual_n)

def section4_filter(A_QG_np_s, S_Blk_s_sparse, D_inv, eta, use_invD,
                    level_name='', group_id=-1, Uinv_s=None):
    n_eigen_old = A_QG_np_s.shape[1]
    if Uinv_s is not None:
        W_mat = Uinv_s @ A_QG_np_s
    else:
        B_Gamma = S_Blk_s_sparse.toarray()
        B_Gamma = (B_Gamma + B_Gamma.T) / 2
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
        print(f'  {label}[{level_name},grp={group_id}]: eig=[{eigvals.min():.4f},{eigvals.max():.4f}] n_gt={n_gt}/{len(eigvals)}')
        if eigvals.max() > 1.0 + 1e-12:
            print(f'  WARNING: max eig {eigvals.max():.6f} > 1!')
    idx = np.where(eigvals > threshold)[0]
    n_selected = len(idx)
    if n_selected == 0:
        return np.zeros((S_Blk_s_sparse.shape[0], 0)), 0, np.array([])
    x = eigvecs[:, idx]
    D_x = eigvals[idx]
    if use_invD:
        z = x * D_inv[:, np.newaxis]
        D_inv_selected = D_inv[idx]
    else:
        z = x.copy()
        D_inv_selected = np.ones(n_selected)
    WtW_inv_z = np.linalg.solve(WtW, z)
    H_np = W_mat @ WtW_inv_z
    if Uinv_s is not None:
        xi = Uinv_s.T @ H_np
        hat_U_s = S_Blk_s_sparse @ xi @ np.diag(D_x)
    else:
        xi = solve_triangular(L.T, H_np, lower=False)
        hat_U_s = B_Gamma @ xi @ np.diag(D_x)
    return hat_U_s, n_selected, D_inv_selected

def compute_level_section4(S_Blk_prev_rows, S_Blk_prev_cols, S_Blk_prev_vals,
                           A_QG_prev_data, D_inv_G_prev,
                           subdomain_all_dofs, Gamma_list, Interior_list,
                           Subd_E_prev, Subd_F_prev,
                           n_coarse_subdomains, group_assignment_coarse,
                             eta, use_invD, level_name='', Uinv_csr_in=None):
    A_QG_data = [None] * n_coarse_subdomains
    flag = np.zeros(n_coarse_subdomains, dtype=np.int32)
    D_inv_G = [None] * n_coarse_subdomains
    S_Blk_rows = [None] * n_coarse_subdomains
    S_Blk_cols = [None] * n_coarse_subdomains
    S_Blk_vals = [None] * n_coarse_subdomains
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

        # Fast submatrix via triplet filtering (skip CSR fancy indexing)
        s_arr = np.asarray(s)
        remap = np.full(n_dofs_total, -1, dtype=np.int32)
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
            A_QG_np_s = csr_matrix((all_vals[keep_A], (loc_A_rows, all_cols[keep_A])), shape=(len(s), col_offset)).toarray()
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
            LARGE_GROUP = 12000
            if len(s) > LARGE_GROUP and Uinv_csr_in is not None:
                U_coo = Uinv_csr_in.tocoo()
                keep_U = (remap[U_coo.row] >= 0) & (remap[U_coo.col] >= 0)
                loc_u_rows = remap[U_coo.row[keep_U]]
                loc_u_cols = remap[U_coo.col[keep_U]]
                Uinv_s = csr_matrix((U_coo.data[keep_U], (loc_u_rows, loc_u_cols)), shape=(len(s), len(s)))
            else:
                Uinv_s = None
            hat_U_s, n_selected, D_inv_s = section4_filter(
                A_QG_np_s, S_Blk_s_sparse, D_inv_local, eta, use_invD, level_name, coarse_id, Uinv_s)
            flag[coarse_id] = n_selected
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
    return A_QG_data, flag, D_inv_G, S_Blk_rows, S_Blk_cols, S_Blk_vals

def build_InvS_Blk_3d(Subd_E_in, Subd_F_in, S_Blk_petsc_in):
    rows, cols, vals = [], [], []
    rows_U, cols_U, vals_U = [], [], []
    for face in Subd_F_in:
        face = face.astype(PETSc.IntType)
        is_face = PETSc.IS().createGeneral(face, comm=PETSc.COMM_WORLD)
        S_sub = S_Blk_petsc_in.createSubMatrix(is_face, is_face)
        S_sub_np = petsc_to_scipy_dense(S_sub)
        U_inv = np.linalg.inv(np.linalg.cholesky(S_sub_np))
        inv = U_inv.T @ U_inv
        n_f = len(face)
        rows.extend(np.repeat(face, n_f)); cols.extend(np.tile(face, n_f)); vals.extend(inv.ravel())
        rows_U.extend(np.repeat(face, n_f)); cols_U.extend(np.tile(face, n_f)); vals_U.extend(U_inv.ravel())
        del inv, U_inv, S_sub_np, S_sub
        is_face.destroy()
    for edge in Subd_E_in:
        edge = edge.astype(PETSc.IntType)
        is_edge = PETSc.IS().createGeneral(edge, comm=PETSc.COMM_WORLD)
        S_sub = S_Blk_petsc_in.createSubMatrix(is_edge, is_edge)
        S_sub_np = petsc_to_scipy_dense(S_sub)
        U_inv = np.linalg.inv(np.linalg.cholesky(S_sub_np))
        inv = U_inv.T @ U_inv
        n_e = len(edge)
        rows.extend(np.repeat(edge, n_e)); cols.extend(np.tile(edge, n_e)); vals.extend(inv.ravel())
        rows_U.extend(np.repeat(edge, n_e)); cols_U.extend(np.tile(edge, n_e)); vals_U.extend(U_inv.ravel())
        del inv, U_inv, S_sub_np, S_sub
        is_edge.destroy()
    rows, cols, vals = map(np.array, [rows, cols, vals])
    InvS = csr_matrix((vals.astype(PETSc.ScalarType), (rows, cols)), shape=S_Blk_petsc_in.getSize())
    InvS = PETSc.Mat().createAIJ(size=InvS.shape,
                                 csr=(InvS.indptr.astype(PETSc.IntType),
                                      InvS.indices.astype(PETSc.IntType),
                                      InvS.data.astype(PETSc.ScalarType)))
    rows_U, cols_U, vals_U = map(np.array, [rows_U, cols_U, vals_U])
    Uinv_csr = csr_matrix((vals_U.astype(PETSc.ScalarType), (rows_U, cols_U)), shape=S_Blk_petsc_in.getSize())
    del rows, cols, vals, rows_U, cols_U, vals_U
    return InvS, Uinv_csr

def build_A_QG_petsc(A_QG_data_list_in, n_dofs_total_in):
    n_eigen_total = sum(data['n_eigen'] for data in A_QG_data_list_in)
    if n_eigen_total == 0:
        return PETSc.Mat().createAIJ(size=(n_dofs_total_in, 0))
    all_rows, all_cols, all_vals = [], [], []
    col_offset = 0
    for data in A_QG_data_list_in:
        m_indices, A1Q, n_eigen = data['m'], data['A1Q'], data['n_eigen']
        all_rows.append(np.repeat(m_indices, n_eigen))
        all_cols.append(np.tile(np.arange(n_eigen) + col_offset, len(m_indices)))
        all_vals.append(A1Q.ravel())
        col_offset += n_eigen
    all_rows, all_cols, all_vals = map(np.concatenate, [all_rows, all_cols, all_vals])
    A_QG_csr = csr_matrix((all_vals, (all_rows, all_cols)), shape=(n_dofs_total_in, n_eigen_total))
    return PETSc.Mat().createAIJ(size=(n_dofs_total_in, n_eigen_total),
                                 csr=(A_QG_csr.indptr.astype(PETSc.IntType),
                                      A_QG_csr.indices.astype(PETSc.IntType),
                                      A_QG_csr.data.astype(PETSc.ScalarType)))

def assemble_final_global_3d(A_QG_data, S_Blk_rows, S_Blk_cols, S_Blk_vals,
                              Subd_E_list, Subd_F_list, coarse_GG, D_inv_G):
    S_Blk_rows_full = np.concatenate([r for r in S_Blk_rows if r is not None])
    S_Blk_cols_full = np.concatenate([c for c in S_Blk_cols if c is not None])
    S_Blk_vals_full = np.concatenate([v for v in S_Blk_vals if v is not None])
    S_Blk_csr = csr_matrix((S_Blk_vals_full, (S_Blk_rows_full, S_Blk_cols_full)),
                            shape=(n_dofs_total, n_dofs_total))
    S_Blk_full = PETSc.Mat().createAIJ(
        size=(n_dofs_total, n_dofs_total),
        csr=(S_Blk_csr.indptr.astype(PETSc.IntType),
             S_Blk_csr.indices.astype(PETSc.IntType),
             S_Blk_csr.data.astype(PETSc.ScalarType)))
    is_coarse_GG = PETSc.IS().createGeneral(coarse_GG.astype(PETSc.IntType), comm=PETSc.COMM_WORLD)
    S_Blk = S_Blk_full.createSubMatrix(is_coarse_GG, is_coarse_GG); S_Blk.assemble()
    n_eigen_total = sum(data['n_eigen'] for data in A_QG_data if data is not None)
    all_rows, all_cols, all_vals = [], [], []
    col_offset = 0
    for data in A_QG_data:
        if data is None: continue
        m_indices = data['m']
        A1Q = data['A1Q']
        n_eigen = data['n_eigen']
        if n_eigen == 0 or len(m_indices) == 0: continue
        all_rows.append(np.repeat(m_indices, n_eigen))
        all_cols.append(np.tile(np.arange(n_eigen) + col_offset, len(m_indices)))
        all_vals.append(A1Q.ravel())
        col_offset += n_eigen
    if len(all_rows) > 0:
        all_rows = np.concatenate(all_rows)
        all_cols = np.concatenate(all_cols)
        all_vals = np.concatenate(all_vals)
    A_QG_csr = csr_matrix((all_vals, (all_rows, all_cols)), shape=(n_dofs_total, col_offset))
    A_QG_full = PETSc.Mat().createAIJ(
        size=(n_dofs_total, col_offset),
        csr=(A_QG_csr.indptr.astype(PETSc.IntType),
             A_QG_csr.indices.astype(PETSc.IntType),
             A_QG_csr.data.astype(PETSc.ScalarType)))
    A_QG = A_QG_full.createSubMatrix(is_coarse_GG, None); A_QG.assemble()
    InvS_Blk_full, _ = build_InvS_Blk_3d(Subd_E_list, Subd_F_list, S_Blk_full)
    InvS_Blk = InvS_Blk_full.createSubMatrix(is_coarse_GG, is_coarse_GG); InvS_Blk.assemble()
    W = InvS_Blk * A_QG
    _, n_cols = A_QG.getSize()
    if D_inv_G is None:
        D_inv_filtered = np.ones(n_cols)
    else:
        D_inv_filtered = np.concatenate([d for d in D_inv_G if d is not None and len(d) > 0])
    D_inv_csr = csr_matrix((D_inv_filtered, (np.arange(n_cols), np.arange(n_cols))), shape=(n_cols, n_cols))
    D_inv_mat = PETSc.Mat().createAIJ(
        size=(n_cols, n_cols),
        csr=(D_inv_csr.indptr.astype(PETSc.IntType),
             D_inv_csr.indices.astype(PETSc.IntType),
             D_inv_csr.data.astype(PETSc.ScalarType)))
    D_inv_mat.assemble()
    Global_Solver = D_inv_mat - A_QG.transposeMatMult(W)
    D_inv_mat.destroy()
    S_Blk_full.destroy()
    A_QG_full.destroy()
    InvS_Blk_full.destroy()
    return S_Blk, A_QG, InvS_Blk, W, Global_Solver, is_coarse_GG

# ==================== 13. Level 0 子区域计算 ====================
print(f"\n=== Level 0 子区域计算 (n_subdomains={n_subdomains}) ===")
S_rows_list, S_cols_list, S_vals_list = [], [], []
S_Blk_rows_list = [None] * n_subdomains
S_Blk_cols_list = [None] * n_subdomains
S_Blk_vals_list = [None] * n_subdomains
g_rows_list, g_vals_list = [], []
A_QG_data_list = [None] * n_subdomains
D_inv_G_list = [None] * n_subdomains
flag1 = np.zeros(n_subdomains, dtype=np.int32)

for sub_id in range(n_subdomains):
    sub_tet_ids = np.where(membership == sub_id)[0]
    m, n = Gamma[sub_id], Interior[sub_id]
    len_m, len_n = len(m), len(n)
    print(f"子区域{sub_id}: {len(sub_tet_ids)}个四面体, Gamma:{len_m}, Interior:{len_n}")

    submesh, entity_map, vertex_map, geometry_map = mesh.create_submesh(msh, msh.topology.dim, sub_tet_ids)
    V_sub = fem.functionspace(submesh, ("Lagrange", deg))
    dofmap_sub = V_sub.dofmap
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

    ksp_A4 = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
    ksp_A4.setOperators(A4)
    ksp_A4.setType(PETSc.KSP.Type.PREONLY)
    ksp_A4.getPC().setType(PETSc.PC.Type.CHOLESKY)
    ksp_A4.setUp()

    rhs = PETSc.Vec().createMPI(len_n, comm=PETSc.COMM_WORLD)
    sol = PETSc.Vec().createMPI(len_n, comm=PETSc.COMM_WORLD)
    all_solutions = np.zeros((len_n, len_m), dtype=PETSc.ScalarType)
    for j in range(len_m):
        values_j = np.zeros(len_n, dtype=PETSc.ScalarType)
        A3.getValues(np.arange(len_n, dtype=PETSc.IntType), np.array([j], dtype=PETSc.IntType), values_j)
        rhs.setArray(values_j); rhs.assemble()
        ksp_A4.solve(rhs, sol)
        all_solutions[:, j] = sol.getArray()

    rows_tmp = np.repeat(np.arange(len_n), len_m)
    cols_tmp = np.tile(np.arange(len_m), len_n)
    A4_inv_A3_sp = csr_matrix((all_solutions.ravel(), (rows_tmp, cols_tmp)), shape=(len_n, len_m))
    A4_inv_A3 = PETSc.Mat().createAIJ(size=(len_n, len_m),
                                      csr=(A4_inv_A3_sp.indptr.astype(PETSc.IntType),
                                           A4_inv_A3_sp.indices.astype(PETSc.IntType),
                                           A4_inv_A3_sp.data.astype(PETSc.ScalarType)))
    rhs.destroy(); sol.destroy()

    # g = b_m - A2 * A4_inv_A3 * b_n
    b_local_array = b_local.array.copy()
    b_local_vec = PETSc.Vec().createWithArray(b_local_array)
    b_m_values = b_local_vec.getValues(is_m_local)
    b_n_values = b_local_vec.getValues(is_n_local)
    b_n_vec = PETSc.Vec().createWithArray(b_n_values.copy())
    x_n_vec = b_n_vec.duplicate()
    ksp_A4.solve(b_n_vec, x_n_vec)
    A2_x_n = PETSc.Vec().createMPI(len_m, comm=PETSc.COMM_WORLD)
    A2.mult(x_n_vec, A2_x_n)
    g_local = b_m_values - A2_x_n.getArray()
    g_vals_list.append(g_local.copy())
    g_rows_list.append(m)
    b_local_vec.destroy(); b_n_vec.destroy(); x_n_vec.destroy(); A2_x_n.destroy()

    # Schur complement
    SS = A2.matMult(A4_inv_A3); SS.assemble()
    Schur = A1.copy(); Schur.axpy(-1.0, SS); Schur.assemble()
    A4_inv_A3.destroy(); SS.destroy()

    # Save Schur data
    S_rows_list.append(np.repeat(m, len_m))
    S_cols_list.append(np.tile(m, len_m))
    S_vals_list.append(petsc_to_scipy_dense(Schur).ravel().copy())

    # Eigenproblem
    is_m = PETSc.IS().createGeneral(m.astype(PETSc.IntType), comm=PETSc.COMM_WORLD)
    Schur_np = petsc_to_scipy_dense(Schur)
    Schur_np = (Schur_np + Schur_np.T) / 2
    TS_Blk_mm = TS_Blk.createSubMatrix(is_m, is_m); TS_Blk_mm.assemble()
    TS_Blk_mm_np = petsc_to_scipy_dense(TS_Blk_mm)
    A1_masked = np.multiply(Schur_np, TS_Blk_mm_np)
    D, Q = eigh(Schur_np, A1_masked)

    # Save S_Blk data
    A1_csr = csr_matrix(A1_masked)
    s_indptr, s_indices, s_vals = A1_csr.indptr, A1_csr.indices, A1_csr.data
    s_row_indices_local = np.repeat(np.arange(len(np.diff(s_indptr))), np.diff(s_indptr))
    S_Blk_rows_list[sub_id] = m[s_row_indices_local]
    S_Blk_cols_list[sub_id] = m[s_indices]
    S_Blk_vals_list[sub_id] = s_vals.copy()

    # Filter eigenvalues
    eigens_idx = np.where(D < mu_list[0])[0]
    Q = Q[:, eigens_idx]; D_filtered = 1 - D[eigens_idx]
    flag1[sub_id] = len(eigens_idx)

    if len(eigens_idx) > 0:
        A1Q = A1_masked @ Q
        A_QG_data_list[sub_id] = {'m': m.copy(), 'A1Q': A1Q.copy(), 'n_eigen': len(eigens_idx)}
        D_inv_G_list[sub_id] = 1.0 / D_filtered.copy()
    else:
        A_QG_data_list[sub_id] = {'m': m.copy(), 'A1Q': np.array([]).reshape(len(m), 0), 'n_eigen': 0}
        D_inv_G_list[sub_id] = np.array([])

    is_m_local.destroy(); is_n_local.destroy(); is_m.destroy()
    A_local.destroy(); A1.destroy(); A2.destroy(); A3.destroy(); A4.destroy()
    ksp_A4.destroy()

# ==================== 14. Level 0 全局组装 ====================
print("\n=== Level 0 全局组装 ===")
GG.sort()
is_GG = PETSc.IS().createGeneral(GG.astype(PETSc.IntType), comm=PETSc.COMM_WORLD)

S_rows = np.concatenate(S_rows_list); S_cols = np.concatenate(S_cols_list)
S_vals = np.concatenate(S_vals_list)
S_rows_gg = np.searchsorted(GG, S_rows); S_cols_gg = np.searchsorted(GG, S_cols)
S_GG_csr = csr_matrix((S_vals, (S_rows_gg, S_cols_gg)), shape=(len(GG), len(GG)))
S_GG = PETSc.Mat().createAIJ(size=(len(GG), len(GG)),
                             csr=(S_GG_csr.indptr.astype(PETSc.IntType),
                                  S_GG_csr.indices.astype(PETSc.IntType),
                                  S_GG_csr.data.astype(PETSc.ScalarType)))

# Build S_Blk_petsc (block-diagonal S) at Level 0 for Uinv_csr
S_Blk_full_csr = csr_matrix((S_vals, (S_rows, S_cols)), shape=(n_dofs_total, n_dofs_total))
S_Blk_petsc = PETSc.Mat().createAIJ(
    size=(n_dofs_total, n_dofs_total),
    csr=(S_Blk_full_csr.indptr.astype(PETSc.IntType),
         S_Blk_full_csr.indices.astype(PETSc.IntType),
         S_Blk_full_csr.data.astype(PETSc.ScalarType)))
InvS_full, Uinv_csr = build_InvS_Blk_3d(Subd_E, Subd_F, S_Blk_petsc)
InvS_full.destroy(); S_Blk_petsc.destroy()

g_rows = np.concatenate(g_rows_list); g_vals = np.concatenate(g_vals_list)
g_full = PETSc.Vec().createSeq(n_dofs_total)
g_full.setValues(g_rows.astype(PETSc.IntType), g_vals.astype(PETSc.ScalarType),
                 addv=PETSc.InsertMode.ADD_VALUES); g_full.assemble()
g_gamma = PETSc.Vec().createWithArray(g_full.getValues(is_GG))
g_full.destroy()

# ==================== 15. Multi-level coarsening + Section 4 filtering ====================
print("\n=== Multi-level coarsening + filtering ===")
all_flag = [flag1]
if n_coarse_levels > 0:
    coarse_levels_data = []
    current_gamma_list = Gamma
    current_Subd_E = Subd_E
    current_Subd_F = Subd_F
    print(f"Level 1: {n_subdomains} subdomains")
    for level_idx in range(1, n_levels):
        target_n = n_subdomains_list[level_idx]
        level_name = f"Level {level_idx + 1}"
        (coarse_GG, coarse_Subd_E, coarse_Subd_F,
         coarse_Gamma, coarse_Interior, subdomain_all_dofs_coarse,
         group_assignment_coarse, actual_n) = perform_coarsening(
            current_gamma_list, target_n, dofs_bc, level_name)
        coarse_levels_data.append({
            'GG': coarse_GG, 'Subd_E': coarse_Subd_E, 'Subd_F': coarse_Subd_F,
            'Gamma': coarse_Gamma, 'Interior': coarse_Interior,
            'subdomain_all_dofs': subdomain_all_dofs_coarse,
            'group_assignment': group_assignment_coarse, 'actual_n': actual_n
        })
        current_gamma_list = subdomain_all_dofs_coarse
        current_Subd_E = coarse_Subd_E
        current_Subd_F = coarse_Subd_F

    total_levels = 1 + n_coarse_levels
    print(f"=== {total_levels} levels (Section 3 + {n_coarse_levels} Section 4 levels) ===")

    prev_S_Blk_rows = [S_Blk_rows_list[i] for i in range(len(S_Blk_rows_list))]
    prev_S_Blk_cols = [S_Blk_cols_list[i] for i in range(len(S_Blk_cols_list))]
    prev_S_Blk_vals = [S_Blk_vals_list[i] for i in range(len(S_Blk_vals_list))]
    prev_A_QG_data = A_QG_data_list[:]
    prev_D_inv_G = D_inv_G_list[:]
    prev_Subd_E, prev_Subd_F = Subd_E, Subd_F

    n_section4_levels = n_coarse_levels
    for level_idx in range(n_section4_levels):
        data = coarse_levels_data[level_idx]
        actual_n = data['actual_n']
        use_invD = (level_idx == 0)
        eta_val = eta_list[level_idx] if level_idx < len(eta_list) else eta_list[-1]
        level_name = f"Level {level_idx + 2}"
        A_QG_data_coarse, flag_coarse, D_inv_G_coarse, \
            S_Blk_rows_coarse, S_Blk_cols_coarse, S_Blk_vals_coarse = \
            compute_level_section4(
                prev_S_Blk_rows, prev_S_Blk_cols, prev_S_Blk_vals,
                prev_A_QG_data, prev_D_inv_G,
                data['subdomain_all_dofs'], data['Gamma'], data['Interior'],
                prev_Subd_E, prev_Subd_F,
                actual_n, data['group_assignment'],
                eta_val, use_invD, level_name, Uinv_csr)
        all_flag.append(flag_coarse)
        prev_S_Blk_rows, prev_S_Blk_cols, prev_S_Blk_vals = \
            S_Blk_rows_coarse, S_Blk_cols_coarse, S_Blk_vals_coarse
        prev_A_QG_data = A_QG_data_coarse
        prev_D_inv_G = D_inv_G_coarse
        prev_Subd_E, prev_Subd_F = data['Subd_E'], data['Subd_F']

    S_Blk, A_QG, InvS_Blk, W, Global_Solver, is_GG_coarse = assemble_final_global_3d(
        prev_A_QG_data, prev_S_Blk_rows, prev_S_Blk_cols, prev_S_Blk_vals,
        Subd_E, Subd_F, GG, None)
else:
    # Single level (no coarse levels)
    S_Blk, A_QG, InvS_Blk, W, Global_Solver, is_GG_coarse = assemble_final_global_3d(
        A_QG_data_list, S_Blk_rows_list, S_Blk_cols_list, S_Blk_vals_list,
        Subd_E, Subd_F, GG, D_inv_G_list)

# ==================== 18. PCG求解(Schur系统) ====================
def NOSAS_preconditioner(r, z, InvS_Blk_in, W_in, ksp_global_in):
    InvS_Blk_in.mult(r, z)
    if ksp_global_in is not None:
        _, W_n = W_in.getSize()
        if W_n > 0:
            WTr = PETSc.Vec().createMPI(W_n, comm=PETSc.COMM_WORLD)
            W_in.multTranspose(r, WTr)
            sol = WTr.copy()
            ksp_global_in.solve(WTr, sol)
            z.axpy(1.0, W_in * sol)


def pcg_NoSAS(A, b, max_it, tol, InvS_Blk_in, W_in, ksp_global_in):
    global __pcg_res
    x = b.copy(); x.set(0.0)
    r = b.copy()
    bnrm2 = b.norm(PETSc.NormType.NORM_2)
    error = r.norm(PETSc.NormType.NORM_2) / bnrm2
    print(f' PCG residual(0) = {error:e}')
    alpha, beta = np.zeros(max_it), np.zeros(max_it)
    for iter_count in range(max_it):
        z = r.copy()
        NOSAS_preconditioner(r, z, InvS_Blk_in, W_in, ksp_global_in)
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
    if flag == 0 and iter_count > 0:
        d = np.zeros(iter_count + 1); s = np.zeros(iter_count)
        d[0] = 1.0 / alpha[0]
        for i in range(iter_count):
            d[i + 1] = beta[i + 1] / alpha[i] + 1.0 / alpha[i + 1]
            s[i] = -np.sqrt(beta[i + 1]) / alpha[i]
        T = np.zeros((iter_count + 1, iter_count + 1))
        T[iter_count, iter_count] = d[iter_count]
        for i in range(iter_count):
            T[i, i] = d[i]; T[i, i + 1] = s[i]; T[i + 1, i] = s[i]
        lambda_vals = np.linalg.eigvals(T)
        lambdamax, lambdamin = np.max(lambda_vals), np.min(lambda_vals)
        condnumber = lambdamax / lambdamin
        __pcg_res = (iter_count + 1, condnumber, lambdamax, lambdamin)
        print(f'PCG iter {iter_count + 1}, res {error:e}, '
              f'lmax={lambdamax:g}, lmin={lambdamin:g}, cond={condnumber:g}')
    elif flag == 0:
        print(f'PCG iter 0, residual {error:e}')
        __pcg_res = (0, 0.0, 0.0, 0.0)
    else:
        print(f'PCG did NOT converge in {iter_count + 1} iters, residual {error:e}')
        __pcg_res = (iter_count + 1, 0.0, 0.0, 0.0)
    return x, error, iter_count + 1, flag


print("\n=== 开始PCG求解 ===")
gs_size = Global_Solver.getSize()
if gs_size[0] > 0 and gs_size[1] > 0:
    ksp_global = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
    ksp_global.setOperators(Global_Solver)
    ksp_global.setType(PETSc.KSP.Type.PREONLY)
    ksp_global.getPC().setType(PETSc.PC.Type.CHOLESKY)
    ksp_global.setUp()
else:
    print("  WARNING: Global_Solver is degenerate, using InvS_Blk only")
    ksp_global = None

x, error, iter_count, flag = pcg_NoSAS(
    A=S_GG, b=g_gamma, max_it=200, tol=1e-6,
    InvS_Blk_in=InvS_Blk, W_in=W, ksp_global_in=ksp_global)

ksp_global.destroy()
N_s = sum(flag1)
end_time = time.time()
"""

PCG_LINE = ("        print(f'PCG iter {iter_count + 1}, res {error:e}, '\n"
            "              f'lmax={lambdamax:g}, lmin={lambdamin:g}, cond={condnumber:g}')")

END = ('ksp_global.destroy()\nend_time = time.time()\n'
       'print(f"代码运行时间: {end_time - start_time:.4f} 秒")\n'
       'print(f"\\n=== 结果摘要 ===")\n'
       'print(f"Schur系统求解，界面GG自由度: {len(GG)}")\n'
       'print(f"n_subdomains_list = {n_subdomains_list}")\n'
       'print(f"h = {h}")\n'
       'print(f"sum(flag1) = {sum(flag1)}")\n'
       "for i in range(1, len(all_flag)):\n"
       '    print(f"sum(flag{i+1}) = {sum(all_flag[i])}")')

print()
print("Table 6: 3lvl filter  N1={} N2={} h={}".format(N1, N2, H))
print("mu_vals (paper eta0) = {}".format(MU_VALS))
print("eta_vals (paper mu) = {}".format(ETA_VALS))
print("{:>4} {:>4} {:>5} {:>4} {:>7} {:>7} {:>7}".format("eta0", "mu", "#E", "it", "kappa", "lmax", "lmin"))
for eta_val in MU_VALS:
    for mu_val in ETA_VALS:
        code = TEMPLATE
        code = code.replace("mu_list = [0.5]", "mu_list = [{}]".format(eta_val))
        code = code.replace("eta_list = [0.5]", "eta_list = [{}]".format(mu_val))
        code = code.replace("h = 1", "h = {}".format(H))
        print("\n===== eta0={}, mu={} =====".format(eta_val, mu_val))
        ns = {'np': np, 'mesh': mesh, 'fem': fem, 'pymetis': pymetis,
              'PETSc': PETSc, 'ufl': ufl, 'MPI': MPI,
              'csr_matrix': csr_matrix, 'eigh': eigh,
              'cholesky': cholesky, 'solve_triangular': solve_triangular,
              'time': time, 'gc': gc,
              '__builtins__': __builtins__}
        exec(code, ns)
        r = ns.get('__pcg_res', (0, 0.0, 0.0, 0.0))
        N_s = ns.get('N_s', 0)
        print("{:4.1f} {:4.1f} {:5d} {:4d} {:7.2f} {:7.2f} {:7.2f}".format(
            eta_val, mu_val, N_s, r[0], r[1], r[2], r[3]))
        del ns
        gc.collect()
