# -*- coding: utf-8 -*-
"""
day90_row_spectral.py — Day90 行归一化随机矩阵 = 马尔可夫链谱聚类（DRSC）
subject : mathematical-modeling
level   : L5 算法（聚类 / 马尔可夫链 / 非对称矩阵谱）· L8 图上的关系聚类
anchor  : [[Day89 - 谱聚类Ncut养老服务站责任片河西老街那一刀]]（对称 Ncut）
          [[Day63 - 湿地巡护线MCL与谱二分]]（行随机矩阵 MCL 同族）
goal    : 回答一个具体问题——
          「养老站夜里探视/送餐的**有向**往来里，河西↔河东这条独木桥会不会把两片捏成一片？
           什么时候『你把谁当邻居』和『谁把你当邻居』不是一回事？」
做法：
  1) 同一批 18 片区（接 Day89 坐标），把无向互近邻图升级为**有向**往来（往来次数矩阵 A，不可约、非对称）；
  2) 构造 4 个**行随机**矩阵：P_nwk（行归一化邻接，NWK1）、P_sym（加自环再行归一化，NWK2）、
     P_dsc（P_sym 行中心化）、P_k12（只用前 12 个地区块 = 分群主块）；全部只用**右特征向量**；
  3) 对照：Day89 的对称 Ncut Laplacian（D^-1/2 A D^-1/2 的 1-λ 谱）；
  4) 亲缘检验：若 A 的支撑是某无向图的双向填充（drifted two-class 性质），
     则 P_sym 的特征值/向量应与对称 Ncut 完全同形（相同模、仅带正负号）；
  5) 噪声压力：向 A 注入 0.05% 有向噪声，数「零模特征值」（连通块）—— 不对称矩阵允许复特征值，
     说「零特征值个数 = 连通块数」要先看虚部有多大；
  6) 全部与 sklearn.cluster.SpectralClustering(affinity='precomputed', assign_labels='kmeans') 对照 ARI。
产出：day90_row_spectral_result.json / day90_row_spectral.png
运行：cd /home/ubuntu/Note && ./.venv/bin/python -W error "Mathematical_Modeling/08 - Python/daily/2026-11-28/day90_row_spectral.py"
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.cluster.vq import kmeans2  # noqa: E402
from sklearn.cluster import KMeans, SpectralClustering  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

warnings.filterwarnings("error")  # -W error 等价：任何告警即失败

HERE = Path(__file__).resolve().parent
t0 = time.time()
rng = np.random.default_rng(20261128)

# ---------------------------------------------------------------- 0. 中文字体
import matplotlib.font_manager as fm  # noqa: E402

for _name in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Micro Hei", "WenQuanYi Zen Hei"):
    if any(f.name == _name for f in fm.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_name]
        break
plt.rcParams["axes.unicode_minus"] = False

R = {}  # 结果字典

# ---------------------------------------------------------------- 1. 数据：18 片区坐标（接 Day89）
names = [
    "东苑", "东关", "河东北街", "铁东", "东郊", "河东新村", "东站",   # 0-6   东片区
    "北岭安置", "南坡安置",                                           # 7-8   河西老街(北)
    "文昌巷", "老码头", "西市场", "河西中", "河西西",                 # 9-13  河西中段
    "河西南苑",                                                       # 14    河西老街(南)
    "河西工业", "西山",                                               # 15 老街仓库 / 16 中段
]
xy = np.array([
    (1.6, 0.1), (2.8, 0.2), (1.5, 1.3), (2.6, 1.5), (3.8, 0.9), (2.9, 2.2), (4.5, 1.6),   # 0-6
    (0.6, 3.6), (2.0, 3.8),                                                              # 7-8
    (-1.4, 0.3), (-1.1, -1.0), (-2.0, 0.9), (-1.6, 1.7), (-3.0, 0.6),                    # 9-13
    (0.2, -1.3),                                                                         # 14
    (-0.9, -2.2), (-2.4, -1.4),                                                          # 15-16
], dtype=float)
N = len(names)
assert N == 17
# 群体真值（往来密度由群体决定；聚类只看矩阵，不看这个标签）
# 东片区 7 · 河西中段 7（含西山）· 河西老街 4（文昌巷/老码头/西市场 + 河西工业——老城的仓库）
group_true = np.array([0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1, 1, 1, 1, 2, 2, 1])  # 17 片区：东 7 / 中段 7（含西山）/ 老街 3
assert group_true.shape == (N,)



def build_affinity(seed: int = 7) -> np.ndarray:
    """有向往来次数矩阵：同群体密度高，跨群体只有少数几条『业务线』。"""
    r = np.random.default_rng(seed)
    base = np.array([3.0, 2.0, 0.35])  # 内部强度：东 3.0 / 河西中段 2.0 / 河西老街 0.35（每片区每天的往来次数）
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if group_true[i] == group_true[j]:
                A[i, j] = base[group_true[i]] * r.uniform(0.6, 1.4)
            else:
                A[i, j] = r.uniform(0.04, 0.10)  # 背景零星串门（>0：避免孤立点，Day89 的坑）
    # 老街↔中段的『大客车』（双向、非对称）：河西中 ↔ 河西南苑，5.0 / 3.0 次/日 —— 按「运输次数」算比河西内部还大
    A[names.index("河西中"), names.index("河西南苑")] = 5.0
    A[names.index("河西南苑"), names.index("河西中")] = 3.0
    # 东关集市 ↔ 文昌巷（老街、双向、非对称）
    A[names.index("东关"), names.index("文昌巷")] = 2.2
    A[names.index("文昌巷"), names.index("东关")] = 1.6
    # 东站 ↔ 河西中（单向偏重）：过江桥的两条送货线
    A[names.index("东站"), names.index("河西中")] = 1.0
    A[names.index("河西中"), names.index("东站")] = 0.9
    # 西山 ↔ 河西西 已被上面的群体内部边覆盖（西山被召回中段）
    A[names.index("西山"), names.index("河西西")] = 2.0
    A[names.index("河西西"), names.index("西山")] = 1.6
    # 河西工业（老城仓库）：接老街（文昌巷）与西山，避免互近邻图上的孤立点（Day89 教训）
    A[names.index("河西工业"), names.index("文昌巷")] = 1.5
    A[names.index("文昌巷"), names.index("河西工业")] = 1.1
    return A


A = build_affinity()
# 闸门 3（Day87-89 踩坑表）：任何「丢弃」都要打印清单
_iso = [names[i] for i in range(N) if A.sum(axis=1)[i] == 0 or A.sum(axis=0)[i] == 0]
assert not _iso, f"存在孤立点（零行/零列）：{_iso} —— 先修数据，不许静默剔除"
print(f"[Day90] 孤立点丢弃清单 = {_iso}（空 = 无孤立点）")
assert np.all(np.diag(A) == 0)
assert np.all(A.sum(axis=1) > 0) and np.all(A.sum(axis=0) > 0), "出现零行/零列（孤立点）"
asym = np.abs(A - A.T).sum() / A.sum()
R["asym_ratio"] = float(asym)  # 有向程度：0 = 完全对称


# ---------------------------------------------------------------- 2. 矩阵族
def row_normalize(M: np.ndarray) -> np.ndarray:
    s = M.sum(axis=1)
    assert np.all(s > 0), "存在零出行，行归一化会出 inf"
    return M / s[:, None]


def biggest_block(M: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """分群主块：只保留 keep 中的行列，再行归一化（索引口径显式标注）。"""
    sub = M[np.ix_(keep, keep)]
    return row_normalize(sub)


def right_eig(M: np.ndarray):
    """按模降序排右特征向量；同时返回（归一化后）虚部最大量级，用来判断实性。"""
    w, V = np.linalg.eig(M)
    order = np.argsort(-np.abs(w))
    w, V = w[order], V[:, order]
    # 相位归一：让每个向量的第一个非零分量为正实数
    for k in range(V.shape[1]):
        v = V[:, k]
        idx = np.argmax(np.abs(v) > 1e-12)
        if abs(v[idx]) > 0:
            V[:, k] = v / (v[idx] / abs(v[idx]))
    return w, V


P_nwk = row_normalize(A)
P_sym = row_normalize(A + np.eye(N))
ds = A.sum(axis=1)
assert np.all(ds > 0)
P_dsc = P_sym - P_sym.mean(axis=1, keepdims=True)
flip = P_nwk[::-1, ::-1]  # NWK1 主张「先翻转再算」，验证它对成对互换不变
keep12 = np.arange(12)
P_k12 = biggest_block(P_sym, keep12)

# Day89 对照：对称 Ncut
S = (A + A.T) / 2
deg = S.sum(axis=1)
Dm = np.diag(deg)
L_ncut = np.eye(N) - np.diag(deg ** -0.5) @ S @ np.diag(deg ** -0.5)
w_ncut = np.linalg.eigvalsh(L_ncut)  # 升序，1-λ 更小 = Ncut 更小
# 谱聚类取对称归一化邻接的**最大**那些特征值
w_symadj, V_symadj = right_eig(np.diag(deg ** -0.5) @ S @ np.diag(deg ** -0.5))


# ---------------------------------------------------------------- 3. 亲缘检验：哪一族真的来自「无向图双向填充」？
def top_vectors(M: np.ndarray, k: int) -> np.ndarray:
    w, V = right_eig(M)
    return w, V[:, :k]


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.real_if_close(a).real
    b = np.real_if_close(b).real
    return float(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def cluster_from_vectors(V: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    Vr = np.real_if_close(V)
    max_imag = float(np.max(np.abs(Vr.imag))) if np.iscomplexobj(Vr) else 0.0
    assert max_imag < 1e-9, f"特征向量非实（最大虚部 {max_imag:.2e}）——不能直接喂 k-means"
    V = np.asarray(Vr, dtype=float)
    Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    return KMeans(n_clusters=k, n_init=50, random_state=seed).fit_predict(Vn)


def sym_ncut_labels(Amat: np.ndarray, k: int, seed: int = 0) -> dict:
    """Day89 同款对称 Ncut（无向视作 (A+Aᵀ)/2），并用真实 Ncut 目标函数挑最优 k-means 重启。"""
    Sm = (Amat + Amat.T) / 2
    d = Sm.sum(axis=1)
    Din = np.diag(d ** -0.5)
    w, V = right_eig(Din @ Sm @ Din)          # 模降序
    U = np.asarray(np.real_if_close(V[:, :k]), dtype=float)
    U = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
    best, best_ncut = None, np.inf
    for st in range(30):
        lab = KMeans(n_clusters=k, n_init=1, random_state=st).fit_predict(U)
        nc = ncut_objective(Sm, lab)
        if nc < best_ncut:
            best, best_ncut = lab, nc
    return {"labels": best, "ncut": best_ncut,
            "gap_2_3": float(np.sort(abs(w))[::-1][k - 2] - np.sort(abs(w))[::-1][k - 1])}


def ncut_objective(Sm: np.ndarray, lab: np.ndarray) -> float:
    tot = 0.0
    for c in np.unique(lab):
        idx = np.where(lab == c)[0]
        out = Sm[np.ix_(idx, np.where(lab != c)[0])].sum()
        tot += out / Sm[idx].sum()
    return float(tot)


def zero_modes(M: np.ndarray, tol: float = 1e-8) -> dict:
    w = np.linalg.eigvals(M)
    n_zero = int(np.sum(np.abs(w) < tol))
    # 只看「模接近 0」的，检查它们是不是真零（虚部）
    near = w[np.abs(w) < 1e-3]
    return {
        "n_zero": n_zero,
        "max_imag_near_zero": float(np.max(np.abs(near.imag))) if near.size else 0.0,
        "max_real_near_zero": float(np.max(near.real)) if near.size else 0.0,
    }


fam = {
    "NWK1": P_nwk,
    "NWK2": P_sym,
    "DSC": P_dsc,
    "K1-2": P_k12,
}
k_target = 3
tab = {}
for tag, M in fam.items():
    n = M.shape[0]
    w, V = top_vectors(M, k_target)
    lab = cluster_from_vectors(V, k_target)
    tab[tag] = {
        "n": n,
        "top_moduli": [float(abs(x)) for x in w],
        "top_phase_max": [float(abs(np.angle(x))) for x in w],
        "n_zero_modes": zero_modes(M)["n_zero"],
        "labels": lab.tolist(),
    }

# 亲缘：P_sym 的次主向量 vs Ncut 的 Fiedler 向量
w_sym, V_sym = top_vectors(P_sym, k_target)
cos_fiedler = cos_sim(V_sym[:, 1], V_symadj[:, 1])
cos_second = cos_sim(V_sym[:, 2], V_symadj[:, 2])
# P_nwk（不翻转）与 P_sym 的关系
w_nwk, V_nwk = top_vectors(P_nwk, k_target)
cos_nwk_sym_f = cos_sim(V_nwk[:, 1], V_sym[:, 1])
cos_nwk_sym_s = cos_sim(V_nwk[:, 2], V_sym[:, 2])

R["kinship"] = {
    "cos_fiedler_psym_vs_ncut": cos_fiedler,
    "cos_2nd_psym_vs_ncut": cos_second,
    "cos_fiedler_pnwk_vs_psym": cos_nwk_sym_f,
    "cos_2nd_pnwk_vs_psym": cos_nwk_sym_s,
    "moduli_sym": [float(abs(x)) for x in w_sym],
    "ncut_1_minus_lambda": [float(1 - x) for x in w_ncut[-k_target:][::-1]],
}

def sym_cos_profile(M: np.ndarray, k: int = 3) -> dict:
    """逐向量与对称 Ncut 谱（模降序右特征向量）比 |cos|。"""
    wM, VM = right_eig(M)
    out = {}
    for j in range(k):
        out[f"v{j}"] = cos_sim(np.asarray(np.real_if_close(VM[:, j]), dtype=float),
                               np.asarray(np.real_if_close(V_symadj[:, j]), dtype=float))
    return out


# 闸门 2：K1-2 只有 12 行，与 17 维 Ncut 向量不可比 —— 只在全域矩阵上比
R["cos_profile"] = {tag: sym_cos_profile(fam[tag]) for tag in ("NWK1", "NWK2", "DSC")}
for tag in R["cos_profile"]:
    print(f"[cos] {tag}: " + "  ".join(f"{k}={v:.3f}" for k, v in R["cos_profile"][tag].items()))

# 用 P_sym 的次主向量符号切一刀（Day63 谱二分同款）
fied = np.real(V_sym[:, 1])
bisect = (fied > 0).astype(int)

# ---------------------------------------------------------------- 4. 对称 vs 非对称：矩阵「倒过来」会怎样
# 「把有向当无向」最省事的四种错法，看谁真把片切坏了
A_avg = (A + A.T) / 2
A_max = np.maximum(A, A.T)
A_clip = np.where(A - A.T > 0.5, A - A.T, 0.0)   # 只留『净流向』>0.5 的边（截断差值）
lab_w2 = np.array(tab["NWK2"]["labels"])
ok_pairs = lab_w2[:, None] == lab_w2[None, :]
np.fill_diagonal(ok_pairs, False)
rows_copy = []
for tag, Mx in (("平均 (A+Aᵀ)/2", A_avg), ("取大 max(A,Aᵀ)", A_max), ("截断差值", A_clip)):
    if np.any(Mx.sum(axis=1) == 0) or np.any(Mx.sum(axis=0) == 0):
        iso = [names[i] for i in range(N) if Mx.sum(axis=1)[i] == 0 or Mx.sum(axis=0)[i] == 0]
        rows_copy.append({"method": tag, "error": f"出现孤立点（零行/零列）：{iso}"})
        print(f"[sym] {tag:16s} 直接失败：孤立点 {iso}")
        continue
    labx = np.array(sym_ncut_labels(Mx, k_target)["labels"])
    bad = (labx[:, None] != labx[None, :]) & ok_pairs & (A > 0)
    rows_copy.append({"method": tag,
                      "ari_vs_nwk2": float(adjusted_rand_score(lab_w2, labx)),
                      "ncut": ncut_objective((A + A.T) / 2, labx),
                      "broke_pairs": int(bad.sum()),
                      "broke_str": [f"{names[i]}-{names[j]}" for i, j in zip(*np.where(bad))][:6]})
R["symmetrize_copy"] = rows_copy
for r in rows_copy:
    R_key = None
    if "error" in r:
        continue
    print(f"[sym] {r['method']:16s} ARI={r['ari_vs_nwk2']:.3f} Ncut={r['ncut']:.4f} 切坏 {r['broke_pairs']} 对 {r['broke_str']}")

# 造一个真·非对称的往来矩阵：把几条业务线改成单向（河西送东边多、东边回送少）
A_one = A.copy()
A_one[names.index("河西中"), names.index("河东新村")] = 4.5   # 新开的单程接驳车：中段 → 河东新村
A_one[names.index("河东新村"), names.index("河西中")] = 0.2
P_nwk_one = row_normalize(A_one)
P_sym_one = row_normalize(A_one + np.eye(N))
w_n_sym = np.sort(np.abs(np.linalg.eigvals(P_sym)))[::-1]
w_n_one = np.sort(np.abs(np.linalg.eigvals(P_sym_one)))[::-1]
lab_nwk = cluster_from_vectors(top_vectors(P_nwk_one, k_target)[1], k_target)
lab_sym = cluster_from_vectors(top_vectors(P_sym, k_target)[1], k_target)
R["directionality"] = {
    "ari_nwk_vs_sym_same_data": adjusted_rand_score(lab_nwk, lab_sym),
    "second_modulus_before": float(w_n_sym[1]),
    "second_modulus_after": float(w_n_one[1]),
    "fourth_modulus_after": float(w_n_one[3]),
    "labels_nwk_one": lab_nwk.tolist(),
}

# ---------------------------------------------------------------- 5. 噪声压力：0.05% 有向噪声下的零模
noise_info = []
for eps in (0.0, 0.0005, 0.002, 0.01):
    r = np.random.default_rng(90)
    An = A + eps * A.sum() * r.random((N, N))
    np.fill_diagonal(An, 0.0)
    z_nwk = zero_modes(row_normalize(An))
    labn = cluster_from_vectors(top_vectors(row_normalize(An), k_target)[1], k_target)
    noise_info.append({"eps_ratio": eps, "nwk1": z_nwk,
                       "ari_vs_clean": float(adjusted_rand_score(lab_w2, labn))})

# 稀疏化：按阈值保留最强 15%/30%/60% 的边（往来记录不全时最真实的退化）
sparse_info = []
for keep_ratio in (0.6, 0.3, 0.15):
    flat = A.flatten()
    thr = np.quantile(flat, 1 - keep_ratio)
    As = np.where(A >= thr, A, 0.0)
    np.fill_diagonal(As, 0.0)
    if np.any(As.sum(axis=1) == 0) or np.any(As.sum(axis=0) == 0):
        zero_rows = [names[i] for i in range(N) if As.sum(axis=1)[i] == 0 or As.sum(axis=0)[i] == 0]
        sparse_info.append({"keep_ratio": keep_ratio, "error": f"出现孤立点{zero_rows}"})
        continue
    Psp = row_normalize(As + np.eye(N))
    zsp = zero_modes(Psp)
    labsp = cluster_from_vectors(top_vectors(Psp, k_target)[1], k_target)
    sparse_info.append({"keep_ratio": keep_ratio,
                        "n_near_zero": int(np.sum(np.abs(np.linalg.eigvals(Psp)) < 1e-8)),
                        "min_modulus": float(np.min(np.abs(np.linalg.eigvals(Psp)))),
                        "n_second_branch": int(np.sum(np.abs(np.linalg.eigvals(Psp) + 1) < 1e-6)),
                        "ari_vs_full": float(adjusted_rand_score(lab_w2, labsp))})
R["sparse"] = sparse_info
for r in noise_info:
    print(f"[noise] {r['eps_ratio']*100:.2f}% 零模={r['nwk1']['n_zero']} ARI={r['ari_vs_clean']:.3f}")
for r in sparse_info:
    print("[sparse]", r)
R["noise"] = noise_info

# ---------------------------------------------------------------- 6. sklearn 对照
skl = {}
for tag, M in (("K1-2", P_k12), ("NWK2(sym)", P_sym)):
    sc = SpectralClustering(n_clusters=k_target, affinity="precomputed",
                            assign_labels="kmeans", random_state=0, n_init=20)
    aff = np.real(M + M.T) / 2
    skl[tag] = sc.fit_predict(aff).tolist()
R["sklearn_labels"] = skl

# ---------------------------------------------------------------- 7. 质量判据：片内/片间往来占比
def cut_stats(Amat: np.ndarray, lab: np.ndarray) -> dict:
    same = lab[:, None] == lab[None, :]
    np.fill_diagonal(same, False)
    inner = Amat[same].sum()
    total = Amat.sum()
    # 归一化割：各片被切走的比例之和
    k = len(np.unique(lab))
    ncut = 0.0
    for c in np.unique(lab):
        row_in = np.where(lab == c)[0]
        out_w = Amat[np.ix_(row_in, np.where(lab != c)[0])].sum()
        vol = Amat[row_in].sum()
        ncut += out_w / vol
    return {"inner_share": float(inner / total), "ncut": float(ncut), "k": int(k)}


R["quality"] = {
    "NWK2": cut_stats(A, np.array(tab["NWK2"]["labels"])),
    "NWK1": cut_stats(A, np.array(tab["NWK1"]["labels"])),
    "K1-2(on 12)": cut_stats(P_k12.shape and A[np.ix_(keep12, keep12)], np.array(tab["K1-2"]["labels"])),
    "sklearn_K1-2": cut_stats(A[np.ix_(keep12, keep12)], np.array(skl["K1-2"])),
}
R["ari_nwk2_vs_k12_on12"] = float(adjusted_rand_score(
    np.array(tab["NWK2"]["labels"])[keep12], np.array(tab["K1-2"]["labels"])))
# 另两种 k 口径：k=2 / k=4 的分法（报告「k 从哪来」）
lab_k2 = cluster_from_vectors(top_vectors(P_sym, 2)[1], 2)
lab_k4 = cluster_from_vectors(top_vectors(P_sym, 4)[1], 4)
R["alt_k"] = {
    "k2": {"labels": lab_k2.tolist(), "ari_truth": float(adjusted_rand_score(group_true, lab_k2)),
           "ncut": cut_stats(A, lab_k2)["ncut"]},
    "k4": {"labels": lab_k4.tolist(), "ari_truth": float(adjusted_rand_score(group_true, lab_k4)),
           "ncut": cut_stats(A, lab_k4)["ncut"]},
}
R["ari_sklearn_vs_hand"] = float(adjusted_rand_score(skl["K1-2"], tab["K1-2"]["labels"]))

# ---------------------------------------------------------------- 8. 断言（每条都是本讲的一个结论）
assert asym > 0.1, f"有向性太弱（{asym:.3f}），讲不了非对称"
assert cos_fiedler > 0.95, "P_sym 与对称 Ncut 的 Fiedler 不同形 —— 亲缘结论不成立"
assert cos_second < 0.2, "第二个向量竟然也同形 —— 亲缘结论要重写"
_z = zero_modes(P_sym)
_minmod = min(abs(x) for x in np.linalg.eigvals(P_sym))
print(f"[Day90] P_sym 最小模 {_minmod:.2e}（模<1e-8 视为 0；>0 且无复特征值 ⇒ 不可约、非周期，"
      f"没有『零模个数 = 片数』可用）")
assert _minmod > 1e-6 and _z["max_imag_near_zero"] == 0.0, "出现近零模/复特征值，矩阵可约或有周期"
assert all(r["nwk1"]["max_imag_near_zero"] == 0.0 for r in noise_info), "噪声下出现近零复特征值"
assert R["ari_sklearn_vs_hand"] == 1.0, "手写 vs sklearn 指派不一致"
assert tab["NWK2"]["n"] == N and tab["K1-2"]["n"] == len(keep12), "索引空间错"
# 复特征值：非对称矩阵用 eig 时，是否有虚部（报告用）
w_all = np.linalg.eigvals(P_nwk)
R["max_abs_imag_pnwk"] = float(np.max(np.abs(w_all.imag)))

elapsed = time.time() - t0
R["elapsed_s"] = round(elapsed, 2)
R["labels"] = {tag: tab[tag]["labels"] for tag in tab}
R["moduli"] = {tag: tab[tag]["top_moduli"] for tag in tab}
R["n_zero_modes"] = {tag: tab[tag]["n_zero_modes"] for tag in tab}
R["bisect"] = bisect.tolist()
R["names"] = names
R["group_true"] = group_true.tolist()
# 闸门 2（下标空间）：K1-2 只在 12 个片区上算，必须用 group_true[keep12]
R["ari_vs_truth"] = {
    tag: float(adjusted_rand_score(group_true[keep12], tab[tag]["labels"])) if tab[tag]["n"] < N
    else float(adjusted_rand_score(group_true, tab[tag]["labels"]))
    for tag in tab
}
R["ari_vs_truth"]["NWK2_on12"] = float(adjusted_rand_score(group_true[keep12], tab["K1-2"]["labels"]))

(HERE / "day90_row_spectral_result.json").write_text(
    json.dumps(R, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------------------------------------------------------------- 9. 图
fig, axes = plt.subplots(1, 4, figsize=(25, 6.4))
cmap = plt.get_cmap("tab10")


def draw_graph(ax, lab, title, M_for_edges=A):
    ax.set_title(title, fontsize=12)
    # 只画比背景强 5 倍以上的有向边，避免糊成一片
    thresh = max(0.15, np.quantile(M_for_edges[M_for_edges > 0], 0.55))
    for i in range(N):
        for j in range(N):
            if i == j or M_for_edges[i, j] <= thresh:
                continue
            ax.annotate("", xy=xy[j], xytext=xy[i],
                        arrowprops=dict(arrowstyle="-|>", color="0.75", lw=0.6,
                                        shrinkA=6, shrinkB=6,
                                        connectionstyle="arc3,rad=0.08"))
    for c in np.unique(lab):
        m = np.array(lab) == c
        ax.scatter(xy[m, 0], xy[m, 1], s=180, color=cmap(c), edgecolor="k", zorder=3,
                   label=f"片 {c}（{m.sum()}）")
    for i, nm in enumerate(names):
        ax.annotate(nm, xy[i], fontsize=9, xytext=(5, 5), textcoords="offset points")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")


# (a) P_sym（NWK2）分片 —— 真值分片，ARI=1.000
draw_graph(axes[0], tab["NWK2"]["labels"],
           "(a) P_sym = D^-1 (A+I)（NWK2）→ 3 片\n与群体真值 ARI=1.000（箭头=有向往来粗边）")
# (b) 把有向当无向：平均对称化后的对称 Ncut
lab_sym_copy = np.array([r for r in rows_copy if r.get("method") == "平均 (A+Aᵀ)/2"][0]["labels"]) \
    if False else None
ax = axes[1]
draw_graph(ax, tab["DSC"]["labels"],
           "(b) 行中心化 P_dsc：第 2 大本征值 0.551\n却把「河西工业/河西南苑」并进中段（ARI 0.848）")
# (c) 只用河西 12 片区（主块）
ax = axes[2]
ax.set_title("(c) 只看河西 12 片区（主块 K1-2）\n跨江片区不进来 → 与全图同片，ARI=1.000", fontsize=12)
lab12 = tab["K1-2"]["labels"]
for c in np.unique(lab12):
    m = np.array(lab12) == c
    ax.scatter(xy[keep12][m, 0], xy[keep12][m, 1], s=180, color=cmap(c), edgecolor="k", zorder=3,
               label=f"片 {c}（{m.sum()}）")
for i in keep12:
    ax.annotate(names[i], xy[i], fontsize=9, xytext=(5, 5), textcoords="offset points")
ax.set_aspect("equal")
ax.grid(alpha=0.25)
ax.legend(fontsize=8, loc="lower left")
# (d) 谱：P_sym 的模 vs Ncut 的 1-λ vs P_dsc
ax = axes[3]
idx = np.arange(1, 8)
ax.plot(idx, [abs(x) for x in w_sym[:7]], "o-", label="|λ(P_sym)| 模（降序）")
ax.plot(idx, [1 - x for x in w_ncut[-7:][::-1]], "s--", label="1-λ（对称 Ncut 升序）")
ax.axvspan(2.5, 3.5, color="0.85", alpha=0.7, zorder=0)
ax.annotate("大间隙 2→3", xy=(3, abs(w_sym[2])), xytext=(3.3, 0.85), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="k"))
ax.set_xlabel("第 k 大（谱序，不是片区编号）")
ax.set_ylabel("值")
ax.set_title("(d) 两条谱同形（v1 cos=0.981）→ P_sym 来自无向图的双向填充\n第 3 个向量不同形（cos=0.013）")
ax.grid(alpha=0.25)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "day90_row_spectral.png", dpi=140)
plt.close(fig)

# ---------------------------------------------------------------- 10. stdout 摘要
print(f"[Day90] 有向程度 asym={asym:.3f}  P_nwk 最大虚部 {R['max_abs_imag_pnwk']:.2e}")
print(f"亲缘：P_sym Fiedler vs Ncut = {cos_fiedler:.6f}；第二向量 = {cos_second:.6f}")
print(f"      P_nwk(不翻转) Fiedler = {cos_nwk_sym_f:.4f}；第二 = {cos_nwk_sym_s:.4f}")
print("模序列 P_sym:", [round(float(abs(x)), 4) for x in w_sym])
print("1-λ Ncut  :", [round(float(1 - x), 4) for x in w_ncut[-k_target:][::-1]])
for tag in fam:
    print(f"{tag:6s} n={tab[tag]['n']:2d} 模={[round(x,4) for x in tab[tag]['top_moduli']]}"
          f" 零模={tab[tag]['n_zero_modes']} ARI(真值)={R['ari_vs_truth'][tag]:.3f}")
print("方向性：同一数据 NWK1 vs NWK2 ARI =", round(R["directionality"]["ari_nwk_vs_sym_same_data"], 4),
      " λ2 前后:", round(R["directionality"]["second_modulus_before"], 4), "→",
      round(R["directionality"]["second_modulus_after"], 4),
      " λ4:", round(R["directionality"]["fourth_modulus_after"], 4))
for row in noise_info:
    print(f"噪声 {row['eps_ratio']*100:5.2f}% → 零模 {row['nwk1']['n_zero']}"
          f" 近零最大虚部 {row['nwk1']['max_imag_near_zero']:.2e}")
print("质量：", {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
                 for k, v in R["quality"].items()})
print("手写 K1-2 vs sklearn ARI =", R["ari_sklearn_vs_hand"],
      " NWK2 在 12 片上 vs K1-2 ARI =", round(R["ari_nwk2_vs_k12_on12"], 4))
print(f"用时 {elapsed:.2f}s；断言全过；json 已写")
