"""Optimized BDDC deluxe for SPE10 (paper Table 6b).

Mathematically identical to BDDC_3d_deluxe_spe10_fenicsx.py (paper's BDDC
column) but restructured so h=1 with 128 subdomains runs fast on a laptop:
  * subdomain matrices stay SPARSE (no dense toarray / dense sub_full copies);
  * Schur complements use a sparse LU factorization of the interior block;
  * the coarse (primal) preconditioner solve uses a sparse supernodal LU
    rather than a dense full inverse;
  * the always-unused tildeSG assembly and the two O(n_HW^3) dense ops
    (full eigendecomposition of hatSd, dense solve) are removed;
  * stages 03/04/05/06 run in parallel across cores (multiprocessing 'fork',
    read-only shared data via copy-on-write; no code duplication).

Every face/edge generalized eigenvalue problem and the DD scaling are kept
bit-for-bit as in the original, so (#E, iters, cond) match.

Usage:
    from bddc_deluxe import run_bddc
    rows = run_bddc(h=1, n_subdomains=128)     # paper thresholds 1/0.3,1/0.5,1/0.7
    rows = run_bddc(h=2, n_subdomains=128, thresholds=[(1/0.3,1/0.3)])
"""
import multiprocessing as mp
mp.set_start_method('fork', force=True)
from concurrent.futures import ProcessPoolExecutor

import os
import sys
import time

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import pymetis
import ufl
from scipy.sparse import csr_matrix, lil_matrix, block_diag as sp_block_diag
from scipy.sparse.linalg import splu
from scipy.linalg import eigh

DEFAULT_THRESHOLDS = [(1 / 0.3, 1 / 0.3), (1 / 0.5, 1 / 0.5), (1 / 0.7, 1 / 0.7)]
_MAX_WORKERS = min(10, os.cpu_count() or 4)

_timers = {}
_W = {}


def _tic(name):
    _timers[name] = _timers.get(name, 0.0) - time.time()


def _toc(name):
    _timers[name] = _timers.get(name, 0.0) + time.time()


def _print_timers():
    print("\n=== Timing Summary (BDDC) ===")
    for k, v in sorted(_timers.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:.1f}s")
    print(f"  TOTAL: {sum(v for v in _timers.values()):.1f}s")
    sys.stdout.flush()


def _pmap(fn, items, ncores=None):
    n = ncores or _MAX_WORKERS
    if n <= 1 or len(items) <= 1:
        return [fn(i) for i in items]
    with ProcessPoolExecutor(max_workers=n) as ex:
        return list(ex.map(fn, items))


def _schur(A_GG, A_GI, A_II, b_G, b_I):
    """Sparse Schur complement S = A_GG - A_GI A_II^-1 A_GI^T (+ g)."""
    if A_II.shape[0] == 0:
        return A_GG.toarray().copy(), b_G.copy()
    lu = splu(A_II.tocsc())
    X = lu.solve(A_GI.T.toarray())          # (n_interior, n_G)
    S = A_GG.toarray() - A_GI.toarray() @ X
    S = (S + S.T) / 2.0
    g = b_G - A_GI.toarray() @ lu.solve(b_I)
    return S, g


# ----------------------------------------------------------------------
# Stage 03: subdomain assembly + Schur complements (parallel)
# ----------------------------------------------------------------------
def _assemble_sub(sid):
    msh = _W['msh']
    dofmap = _W['dofmap']
    rho_vals = _W['rho_vals']
    membership = _W['membership']
    Gamma = _W['Gamma']
    Interior = _W['Interior']
    verts_per_sub = _W['verts_per_sub']
    deg = _W['deg']
    f_one = _W['f_one']

    tid = np.where(membership == sid)[0]
    mg = Gamma[sid]
    n_int = Interior[sid]

    sm, em, _, _ = mesh.create_submesh(msh, msh.topology.dim, tid)
    Vsm = fem.functionspace(sm, ("Lagrange", deg))
    dms = Vsm.dofmap
    nsc = sm.topology.index_map(3).size_local
    sci = np.arange(nsc, dtype=np.int32)
    cm = em.sub_topology_to_topology(sci, inverse=False)
    srv = rho_vals[cm]

    sdflat = dms.list.flatten()
    odflat = dofmap.list[cm].flatten()
    si = np.argsort(sdflat)
    ssrt = sdflat[si]
    osrt = odflat[si]
    _, ui = np.unique(ssrt, return_index=True)
    all_sd = osrt[ui]
    g2l = {int(g): int(l) for l, g in enumerate(all_sd)}
    ml = np.array([g2l[d] for d in mg], dtype=np.int32)
    nl = np.array([g2l[d] for d in n_int], dtype=np.int32)

    Vloc = fem.functionspace(sm, ("Lagrange", deg))
    u = ufl.TrialFunction(Vloc)
    vf = ufl.TestFunction(Vloc)
    DG0 = fem.functionspace(sm, ("DG", 0))
    rf = fem.Function(DG0)
    rf.x.array[:] = srv
    a_loc = fem.form(ufl.inner(rf * ufl.grad(u), ufl.grad(vf)) * ufl.dx)
    L_loc = fem.form(ufl.inner(f_one, vf) * ufl.dx)
    A_sp = fem.assemble_matrix(a_loc)
    b_loc = fem.assemble_vector(L_loc)
    nloc = A_sp.index_map(0).size_local
    A_csr = csr_matrix((A_sp.data, A_sp.indices, A_sp.indptr), shape=(nloc, nloc))
    b_np = b_loc.array.copy()

    # S^{(i)} on Gamma
    A_GG = A_csr[ml][:, ml]
    A_GI = A_csr[ml][:, nl]
    A_II = A_csr[nl][:, nl]
    S0, g0 = _schur(A_GG, A_GI, A_II, b_np[ml], b_np[nl])

    # S_r^{(i)}: remove vertices from A, recompute Schur on non-vertex interface
    vert_in_sd = [d for d in all_sd if d in verts_per_sub[sid]]
    keep_local = np.array([g2l[d] for d in all_sd if d not in vert_in_sd],
                          dtype=np.int32)
    all_sd_r = [d for d in all_sd if d not in vert_in_sd]
    g2l_r = {int(d): int(i) for i, d in enumerate(all_sd_r)}
    Ar = A_csr[keep_local][:, keep_local]
    br = b_np[keep_local]

    G_iface = np.array([d for d in Gamma[sid] if d not in Interior[sid]],
                       dtype=np.int32)
    gam_r = np.array([d for d in G_iface if d in all_sd_r], dtype=np.int32)
    n_int_r = np.array([d for d in Interior[sid] if d in all_sd_r], dtype=np.int32)
    gl_r = np.array([g2l_r[d] for d in gam_r], dtype=np.int32)
    il_r = np.array([g2l_r[d] for d in n_int_r], dtype=np.int32)
    S_r, g_r = _schur(Ar[gl_r][:, gl_r], Ar[gl_r][:, il_r], Ar[il_r][:, il_r],
                      br[gl_r], br[il_r])

    return (S0, mg.copy(), g0.copy(),
            S_r, gam_r.copy(), g_r.copy(),
            A_csr.copy(), all_sd.copy(), b_np.copy())


# ----------------------------------------------------------------------
# Stage 04/05: entity GEPs (parallel)
# ----------------------------------------------------------------------
def _cond_schur(sid, Nd):
    """Condensed Schur on Nd from S_r^{(i)} (SPD); F' excludes vertices."""
    S_loc, mg, _ = _W['sub_schur'][sid]
    verts = _W['verts_per_sub'][sid]
    Nd_set = set(Nd)
    ndl = np.array([i for i, d in enumerate(mg) if d in Nd_set], dtype=np.int32)
    Se = S_loc[np.ix_(ndl, ndl)]
    grl = np.array([i for i, d in enumerate(mg)
                    if d not in Nd_set and d not in verts], dtype=np.int32)
    if len(grl) == 0:
        return Se
    Seg = S_loc[np.ix_(ndl, grl)]
    Sgg = S_loc[np.ix_(grl, grl)]
    return Se - Seg @ np.linalg.solve(Sgg, Seg.T)


def _face_eig(tup):
    fi, (Nd, s1, s2) = tup
    n = len(Nd)
    S1 = _cond_schur(s1, Nd)
    S2 = _cond_schur(s2, Nd)
    S0_loc1, mg01, _ = _W['sub_schur0'][s1]
    S0_loc2, mg02, _ = _W['sub_schur0'][s2]
    loc1 = np.array([i for i, d in enumerate(mg01) if d in set(Nd)], dtype=np.int32)
    loc2 = np.array([i for i, d in enumerate(mg02) if d in set(Nd)], dtype=np.int32)
    S11 = S0_loc1[np.ix_(loc1, loc1)]
    S22 = S0_loc2[np.ix_(loc2, loc2)]
    X1 = np.linalg.solve(S1 + S2, S2)
    X2 = np.linalg.solve(S11 + S22, S22)
    Tilde_S = (S1 @ X1 + (S1 @ X1).T) / 2
    S_eig = (S11 @ X2 + (S11 @ X2).T) / 2
    evs, evec = eigh(S_eig, Tilde_S)
    return (fi, n, evs, evec)


def _face_te(fi):
    """Build TEf/FagF from the precomputed eigensolution and threshold."""
    n, evs, evec = _W['face_eigs'][fi]
    si_f = np.argsort(evs)
    e1 = si_f[evs[si_f] >= _W['th_f'] - 1e-12]
    Ql = [evec[:, j] for j in e1]
    Ql += [evec[:, j] for j in range(n) if j not in e1]
    return (fi, len(e1), np.column_stack(Ql) if Ql else np.eye(n))


def _edge_eig(tup):
    ei, Nd = tup
    ow = _W['Edge_to_Sub'][ei]
    if len(ow) < 2:
        return (ei, len(Nd), np.full(len(Nd), 1.0), np.eye(len(Nd)))
    ne = len(Nd)
    As = np.zeros((ne, ne))
    Bs = np.zeros((ne, ne))
    for oid in ow:
        Sc = _cond_schur(oid, Nd)
        S_loc, mg_s, _ = _W['sub_schur'][oid]
        loc_s = np.array([i for i, d in enumerate(mg_s) if d in set(Nd)],
                         dtype=np.int32)
        Su = S_loc[np.ix_(loc_s, loc_s)]
        As += np.linalg.inv(Sc)
        Bs += np.linalg.inv(Su)
    As = (As + As.T) / 2
    Bs = (Bs + Bs.T) / 2
    evs, evec = eigh(Bs, As)
    return (ei, ne, evs, evec)


def _edge_te(ei):
    """Build TEe/FlagE from the precomputed eigensolution and threshold."""
    n_e, evs, evec = _W['edge_eigs'][ei]
    si_e = np.argsort(evs)
    edge_th = 1.0 / max(_W['th_e'], 1e-15)
    e1 = si_e[evs[si_e] <= edge_th + 1e-12]
    Ql = [evec[:, j] for j in e1]
    Ql += [evec[:, j] for j in range(n_e) if j not in e1]
    return (ei, len(e1), np.column_stack(Ql) if Ql else np.eye(n_e))


# ----------------------------------------------------------------------
# Stage 06: TE transform (parallel)
# ----------------------------------------------------------------------
def _te_sub(sid):
    W = _W
    pr = W['PRIM'][sid]
    du = W['DUAL'][sid]
    Il = W['Interior'][sid]
    GL = list(pr) + list(du)
    eis = W['EntityIS'][sid]
    Fd = []
    for eidx, Nd in eis:
        npp = min(W['Flag_all'][eidx], len(Nd))
        Fd.extend(Nd[:npp])
        Fd.extend(Nd[npp:])
    Fd.extend(Il)
    Tb = [W['TE_all'][eidx] for eidx, _ in eis] + [np.eye(len(Il))]
    Tmat = sp_block_diag([csr_matrix(b) for b in Tb])
    A_csr, all_sd, b_np = W['sub_full'][sid]
    lm_f = {int(d): int(i) for i, d in enumerate(all_sd)}
    fd = np.array([lm_f[d] for d in Fd], dtype=np.int32)
    ANV = Tmat.T @ A_csr[fd][:, fd] @ Tmat
    bNV = Tmat.T @ b_np[fd]
    GP = np.array([Fd.index(d) for d in GL], dtype=np.int32)
    IP = np.array([Fd.index(d) for d in Il], dtype=np.int32)
    S, g = _schur(ANV[GP][:, GP], ANV[GP][:, IP], ANV[IP][:, IP],
                  bNV[GP], bNV[IP])
    TRe = W['TR'][sid]
    m = len(TRe)
    rows = np.repeat(TRe, m)
    cols = np.tile(TRe, m)
    hsg = csr_matrix((S.ravel(), (rows, cols)), shape=(W['n_HW'], W['n_HW']))
    ga = np.zeros(W['n_HW'])
    for a in range(m):
        ga[TRe[a]] = g[a]
    return (hsg, ga, S)


# ----------------------------------------------------------------------
def run_bddc(h=2, n_subdomains=128, thresholds=None):
    """Run the BDDC deluxe comparison rows of paper Table 6 (b)."""
    if thresholds is None:
        thresholds = list(DEFAULT_THRESHOLDS)
    _timers.clear()
    start_time = time.time()
    deg = 1
    print(f"=== BDDC deluxe: SPE10 h={h}, N1={n_subdomains} ===")
    sys.stdout.flush()

    # ---------- load SPE10 ----------
    perm_file = 'spe_perm.dat'
    nx, ny, nz = 60, 220, 85
    if not os.path.exists(perm_file):
        perm_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 perm_file)
    with open(perm_file, 'r') as f:
        data = []
        for i, line in enumerate(f):
            if i >= 187000:
                break
            data.extend([float(x) for x in line.strip().split()])
    rho_data = np.array(data, dtype=np.float64).reshape(nz, ny, nx)
    del data
    print(f"  rho shape {rho_data.shape}")

    # ---------- mesh ----------
    n_cubes = [int(60 / h), int(220 / h), int(85 / h)]

    def get_cell_rho(cells):
        geom = msh.geometry
        cents = np.mean(geom.x[msh.geometry.dofmap][cells], axis=1)
        ix = np.floor(cents[:, 0] / h).astype(np.int32)
        iy = np.floor(cents[:, 1] / h).astype(np.int32)
        iz = np.floor(cents[:, 2] / h).astype(np.int32)
        sx, sy, sz = int(60 / n_cubes[0]), int(220 / n_cubes[1]), int(85 / n_cubes[2])
        return rho_data[iz * sz, iy * sy, ix * sx]

    msh = mesh.create_box(MPI.COMM_SELF, [np.array([0, 0, 0]), np.array([60, 220, 85])],
                          n_cubes, mesh.CellType.tetrahedron)

    # ---------- BC ----------
    bfuncs = [(lambda x: np.isclose(x[2], 0.0), 10), (lambda x: np.isclose(x[2], 85.0), 11),
              (lambda x: np.isclose(x[1], 0.0), 12), (lambda x: np.isclose(x[1], 220.0), 13),
              (lambda x: np.isclose(x[0], 0.0), 14), (lambda x: np.isclose(x[0], 60.0), 15)]
    fi, fv = [], []
    for fn, tg in bfuncs:
        fcs = mesh.locate_entities_boundary(msh, 2, fn)
        fi.append(fcs)
        fv.append(np.full(len(fcs), tg, dtype=np.int32))
    ftags = mesh.meshtags(msh, 2, np.hstack(fi), np.hstack(fv))

    V = fem.functionspace(msh, ("Lagrange", deg))
    dofmap = V.dofmap
    n_dofs_total = dofmap.index_map.size_global
    facets = np.concatenate([ftags.find(t) for t in [10, 11, 12, 13, 14, 15]])
    dofs_bc = np.unique(fem.locate_dofs_topological(V, 2, facets).astype(np.int32))

    msh.topology.create_connectivity(3, 0)
    tets = msh.topology.connectivity(3, 0).array.reshape(-1, 4)
    msh.topology.create_connectivity(2, 3)
    ftc = msh.topology.connectivity(2, 3)
    rho_vals = get_cell_rho(np.arange(len(tets), dtype=np.int32))

    # ---------- METIS partition ----------
    _tic("01_Metis_partition")
    adj = [[] for _ in range(len(tets))]
    for f in range(ftc.num_nodes):
        c = ftc.links(f)
        if len(c) == 2:
            adj[c[0]].append(c[1])
            adj[c[1]].append(c[0])
    _, membership = pymetis.part_graph(n_subdomains, adjacency=adj)
    membership = np.array(membership, dtype=np.int32)
    del adj
    _toc("01_Metis_partition")

    # ---------- DOF ownership ----------
    sub_all_dofs = [np.array([], dtype=np.int32) for _ in range(n_subdomains)]
    v = np.zeros(n_dofs_total, dtype=np.int32)
    for sid in range(n_subdomains):
        tid = np.where(membership == sid)[0]
        sd = set()
        for c in tid:
            sd.update(dofmap.cell_dofs(c))
        a = np.array(list(sd), dtype=np.int32)
        sub_all_dofs[sid] = a
        v[a] += 1
    GG = np.setdiff1d(np.where(v >= 2)[0], dofs_bc)
    del v
    print(f"  DOFs={n_dofs_total}, GG={len(GG)}")

    d2o = {}
    for sid in range(n_subdomains):
        for d in sub_all_dofs[sid]:
            d2o.setdefault(int(d), []).append(sid)
    d2o = {d: sorted(o) for d, o in d2o.items() if len(o) >= 2}

    Gamma = [None] * n_subdomains
    Interior = [None] * n_subdomains
    for sid in range(n_subdomains):
        sd = sub_all_dofs[sid]
        Interior[sid] = np.setdiff1d(np.setdiff1d(sd, dofs_bc), GG)
        Gamma[sid] = np.intersect1d(sd, GG)

    # ---------- entities ----------
    _tic("02_Entity_detection")
    GG_set = set(GG)
    fg = {}
    for d, o in d2o.items():
        if len(o) == 2 and d in GG_set:
            p = tuple(o)
            fg.setdefault(p, []).append(d)
    FaceInfo = [(np.sort(np.array(d, dtype=np.int32)), p[0], p[1])
                for p, d in fg.items()]
    eg = {}
    for d, o in d2o.items():
        if len(o) >= 3 and d in GG_set:
            p = tuple(o)
            eg.setdefault(p, []).append(d)
    Edges_list = [np.sort(np.array(d, dtype=np.int32)) for _, d in eg.items()]
    Verts_list = [e for e in Edges_list if len(e) == 1]
    Edges_list = [e for e in Edges_list if len(e) > 1]
    print(f"  Faces={len(FaceInfo)}, Edges={len(Edges_list)}, Verts={len(Verts_list)}")

    n_faces = len(FaceInfo)
    n_edges = len(Edges_list)
    n_verts = len(Verts_list)
    Verts_arr = np.array([x[0] for x in Verts_list], dtype=np.int32)

    EntityIS = [[] for _ in range(n_subdomains)]
    Face_to_Sub, Edge_to_Sub = [], []
    for fii, (Nd, s1, s2) in enumerate(FaceInfo):
        Face_to_Sub.append([s1, s2])
        for s in [s1, s2]:
            EntityIS[s].append((fii, Nd))
    for ei, Nd in enumerate(Edges_list):
        ow = d2o[Nd[0]]
        Edge_to_Sub.append(ow)
        for s in ow:
            EntityIS[s].append((n_faces + ei, Nd))
    for vi, vd in enumerate(Verts_list):
        for sid in range(n_subdomains):
            if vd[0] in sub_all_dofs[sid]:
                EntityIS[sid].append((n_faces + n_edges + vi, vd))

    verts_per_sub = [set() for _ in range(n_subdomains)]
    for vd in Verts_list:
        for sid in range(n_subdomains):
            if vd[0] in Gamma[sid]:
                verts_per_sub[sid].add(vd[0])
    _toc("02_Entity_detection")

    # ---------- stage 03 (parallel) ----------
    _tic("03_Subdomain_assembly")
    f_one = 1.0
    sub_schur = [None] * n_subdomains
    sub_schur0 = [None] * n_subdomains
    sub_full = [None] * n_subdomains

    _W.update(dict(msh=msh, dofmap=dofmap, rho_vals=rho_vals,
                   membership=membership, Gamma=Gamma, Interior=Interior,
                   verts_per_sub=verts_per_sub, deg=deg, f_one=f_one))
    t0a = time.time()
    parts = _pmap(_assemble_sub, range(n_subdomains))
    for sid, (S0, mg, g0, S_r, gam_r, g_r, A_csr, all_sd, b_np) in enumerate(parts):
        sub_schur0[sid] = (S0, mg, g0)
        sub_schur[sid] = (S_r, gam_r, g_r)
        sub_full[sid] = (A_csr, all_sd, b_np)
    del parts
    _toc("03_Subdomain_assembly")
    print(f"  Assembly done: {time.time() - t0a:.1f}s (wall {time.time() - start_time:.1f}s)")
    sys.stdout.flush()

    _W.update(sub_schur=sub_schur, sub_schur0=sub_schur0, sub_full=sub_full,
              Edge_to_Sub=Edge_to_Sub)

    # ---------- entity eigensolutions (threshold-independent, once) ----------
    _tic("04_Face_GEP")
    r_faces = _pmap(_face_eig, list(enumerate(FaceInfo)))
    face_eigs = [None] * n_faces
    for fii, n, evs, evec in r_faces:
        face_eigs[fii] = (n, evs, evec)
    del r_faces
    _toc("04_Face_GEP")

    _tic("05_Edge_GEP")
    r_edges = _pmap(_edge_eig, list(enumerate(Edges_list)))
    edge_eigs = [None] * n_edges
    for ei2, ne, evs, evec in r_edges:
        edge_eigs[ei2] = (ne, evs, evec)
    del r_edges
    _toc("05_Edge_GEP")
    print(f"  Entity GEPs done (wall {time.time() - start_time:.1f}s)")
    sys.stdout.flush()

    _W['face_eigs'] = face_eigs
    _W['edge_eigs'] = edge_eigs

    results = []
    for Threshold_face, Threshold_edge in thresholds:
        t0 = time.time()
        _W['th_f'] = Threshold_face
        _W['th_e'] = Threshold_edge

        # ---------- face/edge TE (threshold selection) ----------
        _tic("04b_TE_selection")
        r_faces = _pmap(_face_te, range(n_faces))
        TEf = [None] * n_faces
        FlagF = np.zeros(n_faces, dtype=np.int32)
        for fii, flag, T in r_faces:
            TEf[fii] = T
            FlagF[fii] = flag
        del r_faces
        r_edges = _pmap(_edge_te, range(n_edges))
        TEe = [None] * n_edges
        FlagE = np.zeros(n_edges, dtype=np.int32)
        for ei, flag, T in r_edges:
            TEe[ei] = T
            FlagE[ei] = flag
        del r_edges
        _toc("04b_TE_selection")

        TEv = [np.eye(1) for _ in Verts_list]
        FlagV = np.ones(len(Verts_list), dtype=np.int32)
        TE_all = list(TEf) + list(TEe) + TEv
        Flag_all = list(FlagF) + list(FlagE) + list(FlagV)

        # ---------- prim / dual ----------
        PRIM = [None] * n_subdomains
        DUAL = [None] * n_subdomains
        for sid in range(n_subdomains):
            gs = Gamma[sid]
            ps = set()
            ds = set()
            for eidx, Nd in EntityIS[sid]:
                np_p = min(Flag_all[eidx], len(Nd))
                for k in range(np_p):
                    ps.add(Nd[k])
                for k in range(np_p, len(Nd)):
                    ds.add(Nd[k])
            for d in gs:
                if d not in ps and d not in ds:
                    ps.add(d)
            PRIM[sid] = np.array(sorted(ps), dtype=np.int32)
            DUAL[sid] = np.array(sorted(ds), dtype=np.int32)
        for sid in range(n_subdomains):
            PRIM[sid] = np.array([d for d in PRIM[sid] if d in GG_set],
                                 dtype=np.int32)
            DUAL[sid] = np.array([d for d in DUAL[sid] if d in GG_set],
                                 dtype=np.int32)

        dual_dofs = set()
        for sid in range(n_subdomains):
            for d in DUAL[sid]:
                dual_dofs.add(d)

        HW1 = np.unique(np.concatenate([PRIM[s] for s in range(n_subdomains)]))
        HW2 = np.unique(np.concatenate([DUAL[s] for s in range(n_subdomains)]))
        Hat_W = np.unique(np.concatenate([HW1, HW2]))
        n_HW = len(Hat_W)
        HW2i = {int(d): int(i) for i, d in enumerate(Hat_W)}

        TWl = []
        for sid in range(n_subdomains):
            for d in PRIM[sid]:
                TWl.append(HW2i[d])
        TW = list(set(TWl))
        TR = [None] * n_subdomains
        BR = [None] * n_subdomains
        for sid in range(n_subdomains):
            xa = [HW2i[d] for d in PRIM[sid]]
            xb = [HW2i[d] for d in DUAL[sid]]
            yy = len(TW)
            TW.extend(xb)
            TR[sid] = np.array(xa + xb, dtype=np.int32)
            BR[sid] = np.concatenate([np.array(xa, dtype=np.int32),
                                      np.arange(yy, yy + len(xb), dtype=np.int32)])
        n_TW = len(TW)
        TW_arr = np.array(TW, dtype=np.int32)
        n_prim = n_HW - len(dual_dofs)
        _n_coarse = len(set(TWl))
        print(f"  Hat_W={n_HW}, Tilde_W={n_TW}, prim={n_prim} "
              f"({n_prim / max(n_HW, 1) * 100:.1f}%)")
        sys.stdout.flush()

        h2t = {}
        for tp, hi in enumerate(TW_arr):
            h2t[int(hi)] = tp
        TRGa = csr_matrix((np.ones(n_TW), (np.arange(n_TW), TW_arr)),
                          shape=(n_TW, n_HW))

        # ---------- TE transform (parallel) ----------
        _tic("06_TE_Transform")
        _W.update(PRIM=PRIM, DUAL=DUAL, Interior=Interior, EntityIS=EntityIS,
                  Flag_all=Flag_all, TE_all=TE_all, TR=TR, BR=BR, n_HW=n_HW,
                  n_TW=n_TW)
        te_out = _pmap(_te_sub, range(n_subdomains))
        hatSG = [None] * n_subdomains
        gG = [None] * n_subdomains
        SSnew = [None] * n_subdomains
        for sid, (hsg, ga, S) in enumerate(te_out):
            hatSG[sid] = hsg
            gG[sid] = ga
            SSnew[sid] = S
        del te_out
        _toc("06_TE_Transform")

        # ---------- DD scaling (sequential, small blocks) ----------
        _tic("07_DD_Scaling")
        DD_data = []
        DD_rows = []
        DD_cols = []
        glist_sid = {}
        ndloc_sid = {}
        for sid in range(n_subdomains):
            pr = PRIM[sid]
            du = DUAL[sid]
            GL = list(pr) + list(du)
            glist_sid[sid] = GL
            nm = {}
            for eidx, Nd in EntityIS[sid]:
                nm[eidx] = np.array([GL.index(d) for d in Nd], dtype=np.int32)
            ndloc_sid[sid] = nm
        for sid in range(n_subdomains):
            pr = PRIM[sid]
            du = DUAL[sid]
            GL = glist_sid[sid]
            eis = EntityIS[sid]
            npp = len(pr)
            Dg = np.zeros((len(GL), len(GL)))
            for eidx, Nd in eis:
                Ndl = ndloc_sid[sid].get(eidx)
                if Ndl is None or len(Nd) == 0:
                    continue
                if eidx < n_faces:
                    ow = Face_to_Sub[eidx]
                elif eidx < n_faces + n_edges:
                    ow = Edge_to_Sub[eidx - n_faces]
                else:
                    Dg[np.ix_(Ndl, Ndl)] = np.eye(len(Nd))
                    continue
                if len(ow) <= 1:
                    Dg[np.ix_(Ndl, Ndl)] = np.eye(len(Nd))
                    continue
                Ssum = None
                Sself = None
                for oid in ow:
                    oNdl = ndloc_sid[oid].get(eidx)
                    if oNdl is None:
                        continue
                    S_F = SSnew[oid][np.ix_(oNdl, oNdl)]
                    if oid == sid:
                        Sself = S_F
                    Ssum = S_F if Ssum is None else Ssum + S_F
                if Sself is not None and Ssum is not None:
                    Dd = np.linalg.solve(Ssum, Sself).T
                    Dg[np.ix_(Ndl, Ndl)] = Dd
                else:
                    Dg[np.ix_(Ndl, Ndl)] = np.eye(len(Nd))
            BRE = BR[sid]
            m = len(GL)
            ra = np.array([h2t[int(BRE[a])] if a < npp else BRE[a]
                           for a in range(m)], dtype=np.int32)
            DD_data.append(Dg.ravel())
            DD_rows.append(np.repeat(ra, m))
            DD_cols.append(np.tile(ra, m))
        DDm = csr_matrix((np.concatenate(DD_data),
                          (np.concatenate(DD_rows), np.concatenate(DD_cols))),
                         shape=(n_TW, n_TW))
        del DD_data, DD_rows, DD_cols
        hv = np.array([HW2i[vv] for vv in Verts_arr if vv in HW2i], dtype=np.int32)
        sel = np.isin(TW_arr, hv)
        tp_idx = np.where(sel)[0]
        DDm[tp_idx, tp_idx] = 1.0
        _toc("07_DD_Scaling")

        # ---------- PCG ----------
        _tic("08_PCG")
        hat_data = []
        hat_rows = []
        hat_cols = []
        gGm = np.zeros(n_HW)
        for sid in range(n_subdomains):
            h = hatSG[sid].tocoo()
            hat_data.append(h.data)
            hat_rows.append(h.row)
            hat_cols.append(h.col)
            gGm += gG[sid]
        hatSGm = csr_matrix((np.concatenate(hat_data),
                             (np.concatenate(hat_rows), np.concatenate(hat_cols))),
                            shape=(n_HW, n_HW))
        del hat_data, hat_rows, hat_cols

        # coarse (primal) Schur complement, sparse
        S_coarse = lil_matrix((_n_coarse, _n_coarse))
        dual_inv = [None] * n_subdomains
        pd_block = [None] * n_subdomains
        p_tilde_sid = [None] * n_subdomains
        dual_off = np.zeros(n_subdomains, dtype=np.int32)
        dual_nd = np.zeros(n_subdomains, dtype=np.int32)
        for sid in range(n_subdomains):
            npp = len(PRIM[sid])
            nd = len(DUAL[sid])
            dual_nd[sid] = nd
            p_tilde_sid[sid] = np.array(
                [h2t[int(BR[sid][a])] for a in range(npp)], dtype=np.int32)
            pt = p_tilde_sid[sid]
            Sm = SSnew[sid][:npp, :npp]
            for a in range(npp):
                for b in range(npp):
                    S_coarse[pt[a], pt[b]] += Sm[a, b]
            if nd > 0:
                dd = SSnew[sid][npp:, npp:]
                dd = (dd + dd.T) / 2.0
                dual_inv[sid] = np.linalg.inv(dd)
                pd_block[sid] = SSnew[sid][:npp, npp:]
                dual_off[sid] = BR[sid][npp] - _n_coarse
        for sid in range(n_subdomains):
            if dual_nd[sid] > 0:
                pt = p_tilde_sid[sid]
                corr = pd_block[sid] @ dual_inv[sid] @ pd_block[sid].T
                for a in range(len(pt)):
                    for b in range(len(pt)):
                        S_coarse[pt[a], pt[b]] -= corr[a, b]
        S_coarse_csr = S_coarse.tocsr()
        S_coarse_lu = splu((S_coarse_csr + S_coarse_csr.T) / 2.0)
        del S_coarse, S_coarse_csr

        TDR = DDm @ TRGa

        def prec(r):
            v = TDR @ r
            x_d = np.zeros(n_TW - _n_coarse)
            for sid in range(n_subdomains):
                nd = dual_nd[sid]
                if nd == 0:
                    continue
                do = dual_off[sid]
                x_d[do:do + nd] = dual_inv[sid] @ v[_n_coarse + do:_n_coarse + do + nd]
            b_p = v[:_n_coarse].copy()
            for sid in range(n_subdomains):
                nd = dual_nd[sid]
                if nd == 0:
                    continue
                do = dual_off[sid]
                b_p[p_tilde_sid[sid]] -= pd_block[sid] @ x_d[do:do + nd]
            x_p = S_coarse_lu.solve(b_p)
            for sid in range(n_subdomains):
                nd = dual_nd[sid]
                if nd == 0:
                    continue
                do = dual_off[sid]
                x_d[do:do + nd] -= dual_inv[sid] @ (pd_block[sid].T @ x_p[p_tilde_sid[sid]])
            x_t = np.empty(n_TW)
            x_t[:_n_coarse] = x_p
            x_t[_n_coarse:] = x_d
            return TDR.T @ x_t

        Aop = hatSGm
        bop = gGm
        max_it = 200
        tol = 1e-6
        x_pcg = np.zeros(n_HW)
        r_pcg = bop.copy()
        bnrm2 = max(np.linalg.norm(bop), 1e-15)
        error = np.linalg.norm(r_pcg) / bnrm2
        alpha_arr = np.zeros(max_it)
        beta_arr = np.zeros(max_it)
        rho_old = 0
        p_pcg = None
        for it in range(max_it):
            z_pcg = prec(r_pcg)
            rho = np.dot(r_pcg, z_pcg)
            if it > 0:
                beta_arr[it] = rho / rho_old
                p_pcg = beta_arr[it] * p_pcg + z_pcg
            else:
                p_pcg = z_pcg.copy()
            q_pcg = Aop @ p_pcg
            alpha_arr[it] = rho / max(np.dot(p_pcg, q_pcg), 1e-30)
            x_pcg += alpha_arr[it] * p_pcg
            r_pcg -= alpha_arr[it] * q_pcg
            error = np.linalg.norm(r_pcg) / bnrm2
            if it % 5 == 0:
                print(f"  PCG residual({it + 1}) = {error:.2e}")
            if error <= tol:
                break
            rho_old = rho
        flag = 0 if error <= tol else 1
        lmax_out = 0.0
        lmin_out = 0.0
        pcg_cond = 0.0
        print(f"  PCG iters={it + 1}, residual={error:.2e}")
        if flag == 0 and it > 0:
            d_arr = np.zeros(it + 1)
            s_arr = np.zeros(it)
            d_arr[0] = 1.0 / alpha_arr[0]
            for i in range(it):
                d_arr[i + 1] = beta_arr[i + 1] / alpha_arr[i] + 1.0 / alpha_arr[i + 1]
                s_arr[i] = -np.sqrt(beta_arr[i + 1]) / alpha_arr[i]
            Tmat = np.zeros((it + 1, it + 1))
            Tmat[it, it] = d_arr[it]
            for i in range(it):
                Tmat[i, i] = d_arr[i]
                Tmat[i, i + 1] = s_arr[i]
                Tmat[i + 1, i] = s_arr[i]
            lmb = np.linalg.eigvals(Tmat)
            lmax_out = np.max(lmb)
            lmin_out = np.min(lmb)
            pcg_cond = lmax_out / max(lmin_out, 1e-15)
        elif flag == 0:
            print("  PCG cond: N/A (0 iters)")
        else:
            print(f"  PCG did NOT converge in {it + 1} iters")
        _toc("08_PCG")

        total_face_prim = int(FlagF.sum())
        total_edge_prim = int(FlagE.sum())
        print(f"  RESULT: N1={n_subdomains:d} th_f={Threshold_face:.4f} "
              f"th_e={Threshold_edge:.4f} #E={n_prim:d} iters={it + 1:d} "
              f"lmax={lmax_out:.4f} lmin={lmin_out:.4f} cond={pcg_cond:.4f} "
              f"({time.time() - t0:.1f}s)")
        results.append({
            'N1': n_subdomains, 'th_f': Threshold_face, 'th_e': Threshold_edge,
            'NE': n_prim, 'iters': it + 1, 'lmax': lmax_out, 'lmin': lmin_out,
            'cond': pcg_cond,
            'hat': n_HW, 'tilde': n_TW,
            'face_ratio': total_face_prim / max(sum(len(f[0]) for f in FaceInfo), 1),
            'edge_ratio': total_edge_prim / max(sum(len(e) for e in Edges_list), 1),
            'time': time.time() - t0,
            'total_time': time.time() - start_time,
        })
        sys.stdout.flush()

    _print_timers()
    print("\n=== BDDC Summary ===")
    print(f"{'N1':>4} {'th_f':>8} {'th_e':>8} {'#E':>6} {'iters':>6} {'lmax':>8} "
          f"{'lmin':>8} {'cond':>8} {'time':>6}")
    for r in results:
        print(f"{r['N1']:>4} {r['th_f']:>8.4f} {r['th_e']:>8.4f} {r['NE']:>6} "
              f"{r['iters']:>6} {r['lmax']:>8.4f} {r['lmin']:>8.4f} "
              f"{r['cond']:>8.4f} {r['time']:>5.1f}s")
    return results


if __name__ == '__main__':
    h_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    n_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    run_bddc(h=h_arg, n_subdomains=n_arg)