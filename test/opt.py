import numpy as np
import ufl
import basix.ufl
from dolfinx import mesh, fem, io
from dolfinx.fem import petsc
from mpi4py import MPI
from scipy.optimize import minimize

# 基本パラメータ
f = 500.0
omega = 2 * np.pi * f
c = 343.0
k_val = omega / c

# メッシュ生成（粗くして高速化）
x_min, x_max = 0.0, 50.0
y_min, y_max = 0.0, 10.0
nx, ny = 200, 40  # 粗くした
domain = mesh.create_rectangle(
    MPI.COMM_WORLD, [[x_min, y_min], [x_max, y_max]], [nx, ny]
)

# 関数空間
sub_element = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
element = basix.ufl.blocked_element(sub_element, shape=(2,))
V = fem.functionspace(domain, element)
V_rho = fem.functionspace(domain, sub_element)

# 境界条件
def open_boundaries(x):
    return (np.isclose(x[0], x_min) | np.isclose(x[0], x_max) |
            np.isclose(x[1], y_min) | np.isclose(x[1], y_max))

tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
open_facets = mesh.locate_entities_boundary(domain, fdim, open_boundaries)
facet_tag = mesh.meshtags(domain, fdim, open_facets, np.full_like(open_facets, 2))
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)

# 設計変数を減らす（パラメトリック表現）
n_control = 20  # 制御点の数
control_x = np.linspace(10, 40, n_control)

def params_to_density(params):
    """制御点から密度場を生成"""
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
            y_wall = params[idx] * (1 - t) + params[idx + 1] * t
            # 壁の厚さ0.4m
            if y >= y_wall and y <= 10:
                rho_array[i] = 1.0

    return rho_array

# Helmholtz求解
def solve_helmholtz(rho_array):
    rho = fem.Function(V_rho)
    rho.x.array[:] = rho_array

    x = ufl.SpatialCoordinate(domain)
    t = 0.4

    # 固定壁
    left_wall = ufl.conditional(
        ufl.And(x[0] >= 10.0 - t, x[0] <= 10.0),
        ufl.conditional(x[1] >= 2.0, 1.0, 0.0), 0.0
    )
    right_wall = ufl.conditional(
        ufl.And(x[0] >= 40.0, x[0] <= 40.0 + t),
        ufl.conditional(x[1] >= 2.0, 1.0, 0.0), 0.0
    )

    # 設計壁
    design_wall = ufl.conditional(
        ufl.And(ufl.And(x[0] >= 10.0, x[0] <= 40.0),
                ufl.And(x[1] >= 9.0, x[1] <= 10.0)),
        rho, 0.0
    )

    in_wall = ufl.conditional(
        left_wall + right_wall + design_wall > 0.5, 1.0, 0.0
    )

    ki = in_wall * (k_val * 1.5)
    kr = k_val

    # 弱形式
    p = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    p_r, p_i = p[0], p[1]
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
    source_coord = np.array([25.0, 2.0, 0.0])
    sigma = 0.15

    def source_gaussian(x):
        r_sq = (x[0] - source_coord[0])**2 + (x[1] - source_coord[1])**2
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

# 目的関数
iteration = [0]
def objective(params):
    iteration[0] += 1
    rho_array = params_to_density(params)
    p_sol = solve_helmholtz(rho_array)

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
rho_opt_array = params_to_density(result.x)
rho_opt = fem.Function(V_rho)
rho_opt.x.array[:] = rho_opt_array

with io.XDMFFile(domain.comm, "optimal_density.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(rho_opt)

print(f"最適化完了！最終形状: {result.x}")
