import numpy as np
import ufl
import basix.ufl
from dolfinx import mesh, fem, io
from mpi4py import MPI
from scipy.optimize import minimize

from lib import right_side_cp_to_whole, cp_to_density_light, solve_helmholtz

# 基本パラメータ
f = 500.0
f_source = np.array([50.0, 2.0])
omega = 2 * np.pi * f
c = 343.0 # 音速
k_val = omega / c

# メッシュ生成（粗くして高速化）
x_min, x_max = 0.0, 100.0
y_min, y_max = 0.0, 30.0
res = 5 # 単位長さあたりのメッシュの解像度
nx = int((x_max - x_min) * res)
ny = int((y_max - y_min) * res)
domain = mesh.create_rectangle(
    MPI.COMM_WORLD, [[x_min, y_min], [x_max, y_max]], [nx, ny],
    cell_type=mesh.CellType.quadrilateral
)

# 関数空間（関数とは基底関数のこと）
sub_element = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
element = basix.ufl.blocked_element(sub_element, shape=(2,))
V = fem.functionspace(domain, element) # 関数空間（ベクトル場）
V_rho = fem.functionspace(domain, sub_element) # 密度場

# 境界条件
def open_boundaries(x):
    return (np.isclose(x[0], x_min) | np.isclose(x[0], x_max) |
            np.isclose(x[1], y_min) | np.isclose(x[1], y_max))

# 設計変数を減らす（パラメトリック表現）
n_control = 20  # 制御点の数
control_x = np.linspace(10, 40, n_control)

# 目的関数
iteration = [0]
def objective(params):
    """
    目的関数
    Args:
        params (ndarray): 右半分の制御点列のパラメータ
    Returns:
        float: 目的関数の値
    """

    iteration[0] += 1
    rho_array = cp_to_density_light(right_side_cp_to_whole(params))
    p_sol = solve_helmholtz(V, rho_array, k_val, f_source, open_boundaries)

    x = ufl.SpatialCoordinate(domain)
    obj_region = ufl.conditional(
        ufl.And(ufl.And(x[0] >= 20.0, x[0] <= 30.0),
                ufl.And(x[1] >= 0.0, x[1] <= 3.0)),
        1.0, 0.0
    )

    p_r, p_i = p_sol[0], p_sol[1]
    J = fem.assemble_scalar(fem.form(
        (p_r**2 + p_i**2) * obj_region * ufl.dx
    ))

    print(f"Iter {iteration[0]}: Objective = {J:.6e}, Wall shape = {params[:3]}")
    return J

# 初期設計（直線）
params_init = np.full(n_control, 9.6)

# COBYLA（勾配不要）
result = minimize(
    objective, params_init,
    method='COBYLA',
    bounds=[(2.0, 10.0)] * n_control,
    options={'maxiter': 100, 'rhobeg': 0.5}
)

# 最終形状を保存
rho_opt_array = cp_to_density_light(right_side_cp_to_whole(result.x))
rho_opt = fem.Function(V_rho)
rho_opt.x.array[:] = rho_opt_array

with io.XDMFFile(domain.comm, "optimal_density.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(rho_opt)

print(f"最適化完了！最終形状: {result.x}")
