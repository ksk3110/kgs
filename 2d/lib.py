import numpy as np
from dolfinx import mesh, fem
from dolfinx.fem import petsc
import ufl


def right_side_cp_to_whole(right_side, origin_coord=[0.0, 0.0]):
    """右側半分の制御点から全体の座標を生成する。

    Parameters
    ----------
    right_side : ndarray of shape (n_control, 2)
        原点座標に対する右半分の制御点[x, y]列のパラメータ
    origin_coord : ndarray of shape (2)
        原点座標[x, y]

    Returns
    -------
    ndarray
        全体の座標

    Examples
    --------
    >>> right_side_cp_to_whole(np.array([[15, 2], [10, 5], [0, 7]]), [50, 0])
    array([[65,  2],
           [60,  5],
           [50,  7],
           [40,  5],
           [35,  2]])
    """

    left_side = right_side[::-1] * [-1, 1]  # 右半分の制御点のx座標を反転して左半分にする
    if left_side[0][0] == 0:
        left_side = left_side[1:]  # 原点座標が0の場合は除去
    coords = np.concatenate((right_side, left_side))

    # 原点座標を変更
    coords += origin_coord
    return coords

def cp_to_density(cp, V_rho, path_width=1.0):
    """制御点から壁の密度場を生成する。

    Parameters
    ----------
    cp : ndarray of shape (n, 2)
        制御点[x, y]列
    V_rho : functionspace
        密度場の関数空間

    Returns
    -------
    ndarray
        密度場の値
    """
    rho_array = np.zeros(V_rho.dofmap.index_map.size_local)
    coords = V_rho.tabulate_dof_coordinates()

    segments_start = cp[:-1]
    segments_end = cp[1:]
    segment_length = len(segments_start)

    for i, coord in enumerate(coords):
        x, y = coord[0], coord[1]

        for j in range(segment_length):
            x1, y1 = segments_start[j]
            x2, y2 = segments_end[j]

            if _distance_to_segment(x, y, x1, y1, x2, y2) <= path_width / 2:
                rho_array[i] = 1.0
                break

    return rho_array

def _distance_to_segment(x, y, x1, y1, x2, y2):
    """点(x, y)と線分(x1, y1)(x2, y2)の距離を計算する。"""
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return np.hypot(x - x1, y - y1)

    # ベクトルの投影比率 t を計算
    t = ((x - x1) * dx + (y - y1) * dy) / (dx**2 + dy**2)
    t = np.clip(t, 0.0, 1.0) # 線分の外側に行かないようクリップ

    # 最も近い線分上の点の座標
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    return np.hypot(x - closest_x, y - closest_y)

def solve_helmholtz(domain, V, rho, k_val, f_source, open_boundaries):
    """
    ヘルムホルツ方程式の弱形式を解く。

    Parameters
    ----------
    V : ufl.FunctionSpace
        関数空間。
    rho : ufl.Function
        密度関数。
    k_val : float
        波数。
    f_source : np.ndarray
        ソース関数の座標。
    open_boundaries : list
        開き境界の条件。

    Returns
    -------
    ufl.Function
        ヘルムホルツ方程式の解

    """
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    open_facets = mesh.locate_entities_boundary(domain, fdim, open_boundaries)
    facet_tag = mesh.meshtags(domain, fdim, open_facets, np.full_like(open_facets, 2))
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)

    # 設計壁
    in_wall = ufl.conditional(
        rho > 0.5, 1.0, 0.0
    )

    ki = in_wall * (k_val * 1.5)
    kr = k_val

    # 弱形式
    p = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    p_r, p_i = ufl.split(p)[0], ufl.split(p)[1]
    v_r, v_i = v[0], v[1]

    a = (ufl.dot(ufl.grad(p_r), ufl.grad(v_r)) -
         (kr**2 - ki**2) * p_r * v_r - (2 * kr * ki) * p_i * v_r) * ufl.dx
    a += (ufl.dot(ufl.grad(p_i), ufl.grad(v_i)) -
          (kr**2 - ki**2) * p_i * v_i + (2 * kr * ki) * p_r * v_i) * ufl.dx

    k_constant = fem.Constant(domain, k_val)
    a += -k_constant * p_i * v_r * ds(2)
    a += k_constant * p_r * v_i * ds(2)

    # 音源
    f_expr = fem.Function(V)
    sigma = 0.15

    def source_gaussian(x):
        r_sq = (x[0] - f_source[0])**2 + (x[1] - f_source[1])**2
        val = (1.0 / (2 * np.pi * sigma**2)) * np.exp(-r_sq / (2 * sigma**2))
        return np.vstack((val, np.zeros_like(val)))

    f_expr.interpolate(source_gaussian)
    L = (f_expr[0] * v_r + f_expr[1] * v_i) * ufl.dx

    problem = petsc.LinearProblem(
        a, L, bcs=[],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="acoustic_"
    )
    p_sol = problem.solve()

    return p_sol


# 軽量実装
def cp_to_density_light(cp, V_rho, path_width=1.0):
    """軽量バージョン"""
    coords = V_rho.tabulate_dof_coordinates()
    # 空間全体のX, Y座標をそれぞれ独立した配列として取得
    X = coords[:, 0]
    Y = coords[:, 1]

    # 最終的なマスク（初期値はすべてFalse）
    inside_any_segment = np.zeros(len(coords), dtype=bool)

    segments_start = cp[:-1]
    segments_end = cp[1:]
    half_width = path_width / 2.0

    for j in range(len(segments_start)):
        x1, y1 = segments_start[j]
        x2, y2 = segments_end[j]

        # distance_to_segmentをベクトル演算に対応させたものを一発で適用
        # （引数にスカラーではなく、配列 X, Y を丸ごと渡す）
        d = _distance_to_segment_vector(X, Y, x1, y1, x2, y2)

        # 現在の線分の内側にある場所を True に更新
        inside_any_segment |= (d <= half_width)

    # Trueの場所を 1.0 に変換
    rho_array = inside_any_segment.astype(np.float64)
    return rho_array

def _distance_to_segment_vector(X, Y, x1, y1, x2, y2):
    """全座標配列(X, Y)と、1つの線分との最短距離をNumPyで一括計算"""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return np.hypot(X - x1, Y - y1)

    # 全点一括で投影比率 t を計算（ブロードキャスト）
    t = ((X - x1) * dx + (Y - y1) * dy) / (dx**2 + dy**2)
    t = np.clip(t, 0.0, 1.0)

    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    return np.hypot(X - closest_x, Y - closest_y)
