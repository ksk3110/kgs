import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, io
from dolfinx.io import gmsh as fenics_gmsh
from dolfinx.fem import petsc
import gmsh
import ufl
import params
import os

def calculate_a_weighting(freq: float) -> float:
    """周波数 f [Hz] におけるA特性補正量 [dB] を計算する"""
    f2 = freq**2
    num = 12194.0**2 * f2**2
    den = (f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194.0**2)
    ra = num / den
    return 2.0 + 20.0 * np.log10(ra)

def solve_helmholtz_and_evaluate_perceptual_rms(
    domain: mesh.Mesh,
    facet_tags: mesh.MeshTags,
    cell_tags: mesh.MeshTags,
    freq: float,  # 周波数 [Hz]
    source_pos: tuple,  # 音源位置 (x0, y0)
    amplitude_db: float = 96.0,  # 音源の振幅 (音量) [dB]
    c: float = 343.0,
    output_filename: str = None,
    step: int = 0,
) -> float:

    comm = domain.comm
    omega = 2.0 * np.pi * freq
    k_val = omega / c

    p0 = 2.0e-5
    p_amp = p0 * (10.0 ** (amplitude_db / 20.0))

    # --- 1. 実部・虚部を表現するため 2次元ベクトル空間 (実部, 虚部) を作成 ---
    V = fem.functionspace(domain, ("Lagrange", 1, (2,)))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_r, u_i = u[0], u[1]
    v_r, v_i = v[0], v[1]

    # メッシュ情報と境界測度の準備
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    # 放射境界 (ds(2))：下端(y_min)以外の外周
    coords = domain.geometry.x
    y_min = np.min(coords[:, 1])
    eps = 1e-5
    open_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: x[1] > (y_min + eps)
    )
    facet_tag_open = mesh.meshtags(
        domain,
        fdim,
        open_facets,
        np.full_like(open_facets, 2, dtype=np.int32),
    )

    dx = ufl.Measure("dx", domain=domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag_open)

    # 弱形式（領域内）
    a = (
        ufl.dot(ufl.grad(u_r), ufl.grad(v_r))
        - (k_val**2) * u_r * v_r
        + ufl.dot(ufl.grad(u_i), ufl.grad(v_i))
        - (k_val**2) * u_i * v_i
    ) * dx

    # 上・右・左の開放境界における Sommerfeld 放射境界条件 (-i k p v)
    k_const = fem.Constant(domain, k_val)
    a += (-k_const * u_i * v_r + k_const * u_r * v_i) * ds(2)

    # ガウス音源
    x_spatial = ufl.SpatialCoordinate(domain)
    x0, y0 = source_pos
    sigma = 0.15
    r_sq = (x_spatial[0] - x0) ** 2 + (x_spatial[1] - y0) ** 2
    source_expr = p_amp * ufl.exp(-r_sq / (2.0 * sigma**2))

    # 音源（実部のみに入力）
    L = (source_expr * v_r) * dx

    # 求解
    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="helmholtz_solver_",
    )
    p_field = problem.solve()

    # --- 2. 物理的な RMS 音圧 [Pa] の計算 ---
    p_r_sol, p_i_sol = p_field[0], p_field[1]

    # 領域全体 (dx) での |p|^2 の体積積分
    p_sq_form = fem.form((p_r_sol**2 + p_i_sol**2) * dx)
    vol_form = fem.form(1.0 * dx)

    total_p_sq = comm.allreduce(
        np.real(fem.assemble_scalar(p_sq_form)), op=MPI.SUM
    )
    total_vol = comm.allreduce(
        np.real(fem.assemble_scalar(vol_form)), op=MPI.SUM
    )

    p_rms_phys = np.sqrt(total_p_sq / total_vol)  # 領域内の実効音圧 [Pa]

    # --- 3. 音圧レベル (SPL [dB]) および 聴覚補正 (dBA) の計算 ---
    spl_dB = (
        20.0 * np.log10(p_rms_phys / p0) if p_rms_phys > 1e-12 else -120.0
    )
    a_weight_dB = calculate_a_weighting(freq)
    spl_dBA = spl_dB + a_weight_dB  # A特性補正後の音圧レベル [dBA]

    # === Paraview用に可視化 ===
    if output_filename is not None:
        V_scalar = fem.functionspace(domain, ("Lagrange", 1))

        # 音圧絶対値 |p| [Pa]
        p_abs = fem.Function(V_scalar, name="Pressure_Abs_Pa")
        expr_p_abs = fem.Expression(
            ufl.sqrt(p_r_sol**2 + p_i_sol**2),
            V_scalar.element.interpolation_points,
        )
        p_abs.interpolate(expr_p_abs)
        p_abs.x.scatter_forward()

        # 音圧レベル Lp [dB SPL]
        p_db = fem.Function(V_scalar, name="Pressure_Level_dB")
        p_abs_vals = p_abs.x.array
        p_db.x.array[:] = 20.0 * np.log10(np.maximum(p_abs_vals, 1e-12) / p0)
        p_db.x.scatter_forward()

        # メッシュと音圧フィールドの書き出し
        with io.XDMFFile(domain.comm, output_filename, "w") as xdmf:
            xdmf.write_mesh(domain)
            xdmf.write_function(p_db, step)

        # 壁面メッシュへの書き出し処理
        wall_facets = facet_tags.find(10)
        if len(wall_facets) > 0:
            wall_mesh, _, _, _ = mesh.create_submesh(
                domain, domain.topology.dim - 1, wall_facets
            )

            dir_name, base_name = os.path.split(output_filename)
            wall_filename = os.path.join(dir_name, f"wall_{base_name}")

            with io.XDMFFile(domain.comm, wall_filename, "w") as xdmf_wall:
                xdmf_wall.write_mesh(wall_mesh)

                V_wall = fem.functionspace(wall_mesh, ("Lagrange", 1))
                j_func = fem.Function(V_wall, name="Wall_J_dBA")

                # 壁面全節点に計算した評価値 spl_dBA を全代入
                j_func.x.array[:] = spl_dBA
                j_func.x.scatter_forward()

                xdmf_wall.write_function(j_func, step)

    return float(spl_dBA)

def rho_fields_from_controls(
    control_points: list,
    lc_domain: float = 2.0,    # 領域全体のメッシュサイズ (100x30に対して2.0~3.0が適正)
    lc_wall: float = 0.5,      # 壁周辺のメッシュサイズ (制御点間隔より十分小さく設定)
    comm: MPI.Comm = MPI.COMM_WORLD
):

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    # --- エラー回路回避用オプション ---
    # 1. 幾何交差のトレランス（許容誤差）を少し緩める
    gmsh.option.setNumber("Geometry.Tolerance", 1e-4)

    # 2. メッシュ分割アルゴリズム: 1=MeshAdapt, 6=Frontal-Delaunay (エラーに最も強い)
    gmsh.option.setNumber("Mesh.Algorithm", 6)

    # 3. 1Dメッシュ（曲線）の分解能を高めて Edge Recovery エラーを防止
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20) # 曲線に沿って細分化

    gmsh.model.add("sound_domain_robust")

    x_min, y_min = params.p_min
    x_max, y_max = params.p_max
    width = x_max - x_min
    height = y_max - y_min

    # 1. 領域と壁の作成
    rect = gmsh.model.occ.addRectangle(x_min, y_min, 0, width, height)
    wall_pts = [gmsh.model.occ.addPoint(pt[0], pt[1], 0, lc_wall) for pt in control_points]
    wall_curve = gmsh.model.occ.addBSpline(wall_pts)

    gmsh.model.occ.synchronize()

    # 2. 領域内に壁の曲線を埋め込み
    gmsh.model.mesh.embed(1, [wall_curve], 2, rect)

    # 3. Physical Group の割り当て（タグ重複防止のため -1 を使用）
    gmsh.model.addPhysicalGroup(2, [rect], 1, name="domain")
    gmsh.model.addPhysicalGroup(1, [wall_curve], 10, name="wall")

    # 4. メッシュサイズの設定と生成
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc_domain)
    wall_entities = [(0, p) for p in wall_pts]
    gmsh.model.mesh.setSize(wall_entities, lc_wall)
    gmsh.model.mesh.generate(2)

    # 5. FEniCSx メッシュおよび Tags の抽出
    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.none)
    mesh_data = fenics_gmsh.model_to_mesh(
        gmsh.model, comm=comm, rank=0, gdim=2, partitioner=partitioner
    )

    domain = mesh_data.mesh

    target_cells = mesh.locate_entities(
        domain,
        domain.topology.dim,
        params.target_region
    )

    # 3. セル用の Meshtags を作成して タグ「30」を付与
    cell_values = np.full_like(target_cells, 30, dtype=np.int32)
    cell_tags = mesh.meshtags(domain, domain.topology.dim, target_cells, cell_values)

    facet_tags = mesh_data.facet_tags

    gmsh.finalize()
    return domain, cell_tags, facet_tags

def right_side_cp_to_whole(right_side):
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
    """

    left_side = right_side[::-1] * [-1, 1]  # 右半分の制御点のx座標を反転して左半分にする
    if left_side[-1][0] == 0:
        left_side = left_side[:-1]  # 原点座標が0の場合は除去
    coords = np.concatenate((left_side, right_side))

    # 原点座標を変更
    return coords
