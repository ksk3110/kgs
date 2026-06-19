import numpy as np
import ufl
import basix.ufl
from dolfinx import mesh, fem, io
from dolfinx.fem import petsc
from mpi4py import MPI
from scipy.optimize import minimize

# ==========================================
# 1. 物理パラメータと基本メッシュ設定
# ==========================================
f = 500.0  
omega = 2 * np.pi * f
c = 343.0  
k_val = omega / c  

# 最適化が現実的な時間内で完了するよう、メッシュを150x30に設定
x_min, x_max = 0.0, 50.0
y_min, y_max = 0.0, 10.0
nx, ny = 150, 30  
domain = mesh.create_rectangle(MPI.COMM_WORLD, [[x_min, y_min], [x_max, y_max]], [nx, ny])

sub_element = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
element_p = basix.ufl.blocked_element(sub_element, shape=(2,))
V = fem.functionspace(domain, element_p)
V_scalar = fem.functionspace(domain, sub_element)
coords = V_scalar.tabulate_dof_coordinates()

# ==========================================
# 2. 「固定壁」と「2mの内部最適化領域」の定義
# ==========================================
t = 0.4 # 外壁の厚さ (外牆厚度)

# A. 常に固定される外周のコの字型壁（密度は常に 1.0）
fixed_left = (coords[:, 0] >= 10.0 - t) & (coords[:, 0] <= 10.0) & (coords[:, 1] >= 2.0)
fixed_top = (coords[:, 1] >= 10.0 - t) & (coords[:, 1] <= 10.0) & (coords[:, 0] >= 10.0 - t) & (coords[:, 0] <= 40.0 + t)
fixed_right = (coords[:, 0] >= 40.0) & (coords[:, 0] <= 40.0 + t) & (coords[:, 1] >= 2.0)
fixed_mask = fixed_left | fixed_top | fixed_right
fixed_indices = np.where(fixed_mask)[0]

# B. コの字型壁の内縁2mの最適化設計領域（アルゴリズムの動作範囲）
design_left = (coords[:, 0] >= 10.0) & (coords[:, 0] <= 12.0) & (coords[:, 1] >= 2.0) & (coords[:, 1] <= 9.6)
design_right = (coords[:, 0] >= 38.0) & (coords[:, 0] <= 40.0) & (coords[:, 1] >= 2.0) & (coords[:, 1] <= 9.6)
design_top = (coords[:, 0] >= 12.0) & (coords[:, 0] <= 38.0) & (coords[:, 1] >= 7.6) & (coords[:, 1] <= 9.6)
design_mask = design_left | design_right | design_top
design_indices = np.where(design_mask)[0]
num_design_vars = len(design_indices)

# ==========================================
# 3. 境界、音源、および観測点
# ==========================================
x = ufl.SpatialCoordinate(domain)
left_monitor = ((x[0] - 5.0)**2 + (x[1] - 2.0)**2) <= 0.4**2
right_monitor = ((x[0] - 45.0)**2 + (x[1] - 2.0)**2) <= 0.4**2
monitor_mask = ufl.conditional(ufl.Or(left_monitor, right_monitor), 1.0, 0.0)

def open_boundaries(x):
    return (np.isclose(x[0], x_min) | np.isclose(x[0], x_max) | 
            np.isclose(x[1], y_min) | np.isclose(x[1], y_max))
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
open_facets = mesh.locate_entities_boundary(domain, fdim, open_boundaries)
facet_tag = mesh.meshtags(domain, fdim, open_facets, np.full_like(open_facets, 2))
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)

f_expr = fem.Function(V)
source_coord = np.array([25.0, 2.0, 0.0])
sigma = 0.2

def source_gaussian(x):
    r_sq = (x[0] - source_coord[0]) ** 2 + (x[1] - source_coord[1]) ** 2
    val = (1.0 / (2 * np.pi * sigma**2)) * np.exp(-r_sq / (2 * sigma**2))
    return np.vstack((val, np.zeros_like(val)))
f_expr.interpolate(source_gaussian)

# ==========================================
# 4. トポロジー最適化コアエンジン
# ==========================================
rho = fem.Function(V_scalar)
eval_count = 0

# 速度を確保するため、FEniCSxの変分コンパイルをループの外に移動
p, v = ufl.TrialFunction(V), ufl.TestFunction(V)
p_r, p_i, v_r, v_i = p[0], p[1], v[0], v[1]
k_constant = fem.Constant(domain, k_val)
L = (f_expr[0] * v_r + f_expr[1] * v_i) * ufl.dx

latest_p_sol = None  # 💡 関数外で空の変数を宣言し、音圧を一時保存する

def objective_function(x_design):
    global eval_count, latest_p_sol  # 💡 latest_p_sol をグローバル変数として設定
    eval_count += 1
    
    # 密度場を構成：それ以外は空気(0)、外周のコの字型壁は固定(1)、内縁2mは最適化変数を代入
    rho.x.array[:] = 0.0
    rho.x.array[fixed_indices] = 1.0
    rho.x.array[design_indices] = x_design
    
    # SIMP 物理補間
    wall_damping = k_val * 3.0
    kr = k_val * (1.0 - 0.8 * rho**3)
    ki = wall_damping * (rho**3)
    
    a = (ufl.dot(ufl.grad(p_r), ufl.grad(v_r)) - (kr**2 - ki**2) * p_r * v_r - (2 * kr * ki) * p_i * v_r) * ufl.dx
    a += (ufl.dot(ufl.grad(p_i), ufl.grad(v_i)) - (kr**2 - ki**2) * p_i * v_i + (2 * kr * ki) * p_r * v_i) * ufl.dx
    a += -k_constant * p_i * v_r * ds(2) + k_constant * p_r * v_i * ds(2)
    
    # 行列が特異な場合のクラッシュを防ぐため try-except を使用
    try:
        problem = petsc.LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, petsc_options_prefix=f"opt_")
        p_sol = problem.solve()
        latest_p_sol = p_sol
        J_value = fem.assemble_scalar(fem.form((p_sol[0]**2 + p_sol[1]**2) * monitor_mask * ufl.dx))
    except:
        return 1e5
        
    # 差分法はバックグラウンドで数百回実行されるため、クラッシュしていないことを確認できるよう100回ごとに進捗を出力
    if eval_count % 100 == 0:
        print(f"  [バックグラウンドで勾配計算中] {eval_count} 回のサブ状態を評価しました...")
        
    return J_value * 1e6

# ==========================================
# 5. 最適化イテレーション完了時の出力 (Callback)
# ==========================================
iter_count = 0
def on_iteration_end(x_design):
    global iter_count, latest_p_sol  # 💡 先ほど保存した音圧を呼び出す
    iter_count += 1
    
    # 密度場を構成
    rho.x.array[:] = 0.0
    rho.x.array[fixed_indices] = 1.0
    rho.x.array[design_indices] = x_design
    
    # 1. 材料密度マップ (Density) を出力
    with io.XDMFFile(domain.comm, f"opt_inner_density.xdmf", "w") as xdmf_rho:
        xdmf_rho.write_mesh(domain)
        xdmf_rho.write_function(rho)
        
    # 2. 💡 対応する音圧マップ (Pressure) を出力
    if latest_p_sol is not None:
        p_abs = fem.Function(V_scalar, dtype=np.float64)
        # 振幅の絶対値を計算
        p_abs.x.array[:] = np.sqrt(latest_p_sol.x.array[0::2]**2 + latest_p_sol.x.array[1::2]**2)
        with io.XDMFFile(domain.comm, f"opt_inner_pressure.xdmf", "w") as xdmf_p:
            xdmf_p.write_mesh(domain)
            xdmf_p.write_function(p_abs)
            
    print(f"\n✅ 【第 {iter_count} 回メインイテレーション完了】！モデルの density および pressure ファイルを更新しました\n")

# ==========================================
# 6. 起動設定
# ==========================================
# 制約条件：内縁2mの領域は、最大でも50%の材料しか硬い壁にできない。
# これにより、全体を埋め尽くすのではなく、プログラムに「形状」を生成させる
def volume_constraint(x_design):
    return 0.5 - np.mean(x_design)

bounds = [(0.0, 1.0) for _ in range(num_design_vars)]
constraints = {'type': 'ineq', 'fun': volume_constraint}
x0 = np.full(num_design_vars, 0.5)

print("\n" + "="*55)
print("🚀 FEniCSx 音響トポロジー最適化：2m内縁コーティングモード！")
print(f"固定のコの字型壁：ロック済み")
print(f"内部最適化メッシュ数：{num_design_vars} 自由度")
print("⚠️ 注意：各メッシュの成長方向を探すため、プログラムはバックグラウンドでの差分計算に時間を要します。")
print("="*55 + "\n")

res = minimize(
    objective_function, x0, method='SLSQP', 
    bounds=bounds, constraints=constraints, callback=on_iteration_end,
    options={'maxiter': 15, 'disp': True, 'eps': 0.1, 'ftol': 1e-4}
)
