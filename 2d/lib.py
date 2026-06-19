import numpy as np

def right_side_cp_to_whole(right_side, origin_coord):
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

right_side_cp_to_whole

def cp_to_density(cp, V_rho):
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

    for i, coord in enumerate(coords):
        x, y = coord[0], coord[1]
        # 設計領域内のみ
        if 10 <= x <= 40 and 9 <= y <= 10:
            # 線形補間でy座標を決定
            idx = np.searchsorted(control_x, x) - 1
            idx = np.clip(idx, 0, n_control - 2)
            t = (x - control_x[idx]) / (control_x[idx + 1] - control_x[idx])
            y_wall = cp[idx] * (1 - t) + cp[idx + 1] * t
            # 壁の厚さ0.4m
            if y >= y_wall and y <= 10:
                rho_array[i] = 1.0

    return rho_array
